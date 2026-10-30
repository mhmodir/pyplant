"""Alarm management, following the ISA-18.2 state model.

A boolean that turns a row red is not alarm management. The standard exists
because operators need to distinguish four situations that a single "is it on"
flag collapses into two:

    Normal                  quiet, and nobody needs to do anything
    Unacknowledged Alarm    active, and nobody has looked at it yet
    Acknowledged Alarm      active, and somebody has taken responsibility
    Return To Normal Unack  it cleared itself before anyone looked

That last state is the whole point. A condition that comes and goes between
two operator glances leaves no trace in a flag-based list, and the transient
that nobody saw is very often the interesting one. Here it stays on the list,
visibly, until it is acknowledged.

    Normal ──in──> UnackAlarm ──ack──> AckAlarm
      ^                │                  │
      │                │out               │out
      │                v                  │
      └────ack──── RtnUnack <─────────────┘ (via Normal)

Shelving is a deliberate, temporary, expiring silence with the reason
recorded. It is what operators actually do to a chattering nuisance alarm, and
providing it is how you avoid them ignoring the whole list instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import IntEnum, StrEnum

from pyplant.tags import TAGS, Kind


def now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class State(StrEnum):
    NORMAL = "NORMAL"
    UNACK_ALARM = "UNACK_ALARM"  # active, not acknowledged
    ACK_ALARM = "ACK_ALARM"  # active, acknowledged
    RTN_UNACK = "RTN_UNACK"  # cleared, never acknowledged
    SHELVED = "SHELVED"


class Priority(IntEnum):
    """Higher sorts first. Priorities should be few and mean something."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


# Priority is a design decision per alarm, not something to derive from a name.
# An interlock that trips the pump outranks a level approaching its limit.
PRIORITIES: dict[str, Priority] = {
    "ILK_PMP101": Priority.CRITICAL,
    "ALM_LT101_LOLO": Priority.HIGH,
    "ALM_PT101_HIHI": Priority.HIGH,
    "ALM_LT101_HI": Priority.MEDIUM,
    "ALM_HEALTH": Priority.LOW,
}

MESSAGES: dict[str, str] = {
    "ILK_PMP101": "PMP101 interlock tripped - pump stopped, valve driven shut",
    "ALM_LT101_LOLO": "Tank 1 level low low",
    "ALM_PT101_HIHI": "PMP101 discharge pressure high high",
    "ALM_LT101_HI": "Tank 1 level high",
    "ALM_HEALTH": "Model health index degraded",
}

DEFAULT_SHELF = timedelta(minutes=30)
MAX_SHELF = timedelta(hours=8)


def alarm_tags() -> tuple[str, ...]:
    """Every boolean tag this project treats as an alarm."""
    return tuple(name for name in PRIORITIES if name in TAGS
                 and TAGS[name].kind is Kind.BOOL)


@dataclass
class Alarm:
    """One alarm point and its state."""

    tag: str
    priority: Priority = Priority.MEDIUM
    message: str = ""
    state: State = State.NORMAL
    active: bool = False  # the raw process condition
    in_time: datetime | None = None  # when it last became active
    ack_time: datetime | None = None
    out_time: datetime | None = None  # when it last cleared
    count: int = 0  # times it has gone active
    shelved_until: datetime | None = None
    shelve_reason: str = ""
    _state_before_shelve: State = State.NORMAL

    # -- process condition -------------------------------------------------

    def update(self, active: bool, at: datetime | None = None) -> None:
        at = at or now()
        self._expire_shelf(at)
        rising, falling = active and not self.active, self.active and not active
        self.active = active

        if self.state is State.SHELVED:
            # Still tracked underneath, just not shown as demanding attention.
            if rising:
                self.in_time, self.count = at, self.count + 1
                self._state_before_shelve = State.UNACK_ALARM
            elif falling:
                self.out_time = at
                if self._state_before_shelve is State.UNACK_ALARM:
                    self._state_before_shelve = State.RTN_UNACK
                else:
                    self._state_before_shelve = State.NORMAL
            return

        if rising:
            self.in_time, self.ack_time, self.count = at, None, self.count + 1
            self.state = State.UNACK_ALARM
        elif falling:
            self.out_time = at
            # Acknowledged before it cleared: nothing left to tell anyone.
            self.state = (State.NORMAL if self.state is State.ACK_ALARM
                          else State.RTN_UNACK)

    # -- operator actions --------------------------------------------------

    def acknowledge(self, at: datetime | None = None) -> bool:
        """Returns whether anything changed."""
        at = at or now()
        if self.state is State.UNACK_ALARM:
            self.state, self.ack_time = State.ACK_ALARM, at
            return True
        if self.state is State.RTN_UNACK:
            self.state, self.ack_time = State.NORMAL, at
            return True
        return False

    def shelve(self, duration: timedelta = DEFAULT_SHELF, reason: str = "",
               at: datetime | None = None) -> None:
        at = at or now()
        if duration > MAX_SHELF:
            raise ValueError(f"shelving is capped at {MAX_SHELF}")
        if duration <= timedelta(0):
            raise ValueError("shelf duration must be positive")
        if self.state is not State.SHELVED:
            self._state_before_shelve = self.state
        self.state = State.SHELVED
        self.shelved_until = at + duration
        self.shelve_reason = reason

    def unshelve(self, at: datetime | None = None) -> None:
        at = at or now()
        if self.state is not State.SHELVED:
            return
        self.shelved_until = None
        self.shelve_reason = ""
        # Come back to whatever the condition says now, not to what it said
        # when it was shelved.
        restored = self._state_before_shelve
        if self.active and restored is State.NORMAL:
            restored = State.UNACK_ALARM
        self.state = restored

    def _expire_shelf(self, at: datetime) -> None:
        if self.state is State.SHELVED and self.shelved_until is not None \
                and at >= self.shelved_until:
            self.unshelve(at)

    # -- presentation ------------------------------------------------------

    @property
    def demands_attention(self) -> bool:
        return self.state in (State.UNACK_ALARM, State.RTN_UNACK)

    @property
    def visible(self) -> bool:
        """On the active alarm list. Normal points are not."""
        return self.state is not State.NORMAL

    def as_dict(self) -> dict:
        return {
            "tag": self.tag,
            "message": self.message or self.tag,
            "priority": int(self.priority),
            "priority_name": self.priority.name,
            "state": str(self.state),
            "active": self.active,
            "count": self.count,
            "in_time": self.in_time.isoformat() if self.in_time else None,
            "ack_time": self.ack_time.isoformat() if self.ack_time else None,
            "out_time": self.out_time.isoformat() if self.out_time else None,
            "shelved_until": (self.shelved_until.isoformat()
                              if self.shelved_until else None),
            "shelve_reason": self.shelve_reason,
            "demands_attention": self.demands_attention,
        }


@dataclass
class AlarmManager:
    """The alarm list, fed from boolean tags once per HMI update."""

    alarms: dict[str, Alarm] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.alarms:
            return
        for tag in alarm_tags():
            self.alarms[tag] = Alarm(tag=tag,
                                     priority=PRIORITIES[tag],
                                     message=MESSAGES.get(tag, tag))

    def update(self, values: dict[str, float | bool],
               at: datetime | None = None) -> None:
        at = at or now()
        for tag, alarm in self.alarms.items():
            if tag in values:
                alarm.update(bool(values[tag]), at)
            else:
                alarm._expire_shelf(at)

    # -- operator actions --------------------------------------------------

    def acknowledge(self, tag: str, at: datetime | None = None) -> bool:
        return self.alarms[tag].acknowledge(at)

    def acknowledge_all(self, at: datetime | None = None) -> int:
        return sum(a.acknowledge(at) for a in self.alarms.values())

    def shelve(self, tag: str, minutes: float = 30.0, reason: str = "",
               at: datetime | None = None) -> None:
        self.alarms[tag].shelve(timedelta(minutes=minutes), reason, at)

    def unshelve(self, tag: str, at: datetime | None = None) -> None:
        self.alarms[tag].unshelve(at)

    # -- presentation ------------------------------------------------------

    def active_list(self) -> list[dict]:
        """What the alarm page shows: worst and newest first."""
        visible = [a for a in self.alarms.values() if a.visible]
        visible.sort(key=lambda a: (-int(a.priority),
                                    -(a.in_time or datetime.min).timestamp()))
        return [a.as_dict() for a in visible]

    def summary(self) -> dict:
        counts = {s.value: 0 for s in State}
        worst = 0
        for a in self.alarms.values():
            counts[str(a.state)] += 1
            if a.demands_attention:
                worst = max(worst, int(a.priority))
        return {
            "counts": counts,
            "unacknowledged": counts[State.UNACK_ALARM] + counts[State.RTN_UNACK],
            "active": sum(1 for a in self.alarms.values() if a.active),
            "shelved": counts[State.SHELVED],
            "worst_priority": worst,
        }
