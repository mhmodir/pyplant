"""The PLC program.

`execute()` is a pure function of the input image table plus the program's own
retained state. It performs no I/O, sleeps nowhere, and knows nothing about
Modbus. That is what makes it testable one scan at a time, and it is also how a
real PLC is organised: read the inputs into an image table, run the logic
against the frozen image, write the outputs.

Precedence, highest first, as it would be on a real unit:

1. Interlock. Latched on a trip condition, forces the pump off and drives the
   valve shut. Clears only when the operator resets *and* the condition has
   gone away.
2. Manual. The operator's output demand passes through, and the PID tracks it
   so that returning to auto is bumpless.
3. Auto. The PID holds LIC101 at setpoint.

Two details that only showed up once the loop was actually closed
-----------------------------------------------------------------
**The output is initialised from valve feedback on the first scan.** A
velocity-form PID increments from wherever its output already is, so a
controller starting at 0% slams shut the valve of a plant that was running at
66%. Real controllers seed the output from position feedback, or start in
manual. This one uses ZT101.

**Pump protection is minimum flow, not high pressure.** A centrifugal pump
against a shut valve rises to its shutoff head, 60 kPa here. A high-pressure
trip set below that would fire during any legitimate low-output excursion. The
hazard of dead-heading is no flow to carry heat away, so the trip is low flow
on a time delay, and high discharge pressure is an alarm only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pyplant.plc.pid import PID, PidTuning
from pyplant.tags import TAGS

# Thresholds come from the tag dictionary rather than being retyped here, so a
# respanned instrument cannot leave a stale limit behind in the logic.
LT101_HI = TAGS["LT101"].limits.hi        # 500 mm, alarm
LT101_LOLO = TAGS["LT101"].limits.lolo    # 50 mm, alarm
LT101_HIHI = TAGS["LT101"].limits.hihi    # 550 mm, trip
PT101_HIHI = TAGS["PT101"].limits.hihi    # 58 kPa, alarm
FT101_LO = TAGS["FT101"].limits.lo        # 1 L/min, trip on delay

MIN_FLOW_DELAY = 15.0  # s the pump may run below minimum flow before tripping
DEFAULT_SP = 300.0  # mm


@dataclass
class Hysteresis:
    """Latching comparator. Trips at `limit`, clears `deadband` back inside.

    Without this a signal resting on a limit chatters, and an alarm list that
    chatters is one operators stop reading.
    """

    limit: float
    deadband: float
    high: bool = True  # True: alarm when the value rises above the limit
    state: bool = False

    def __call__(self, value: float) -> bool:
        if self.high:
            self.state = (value >= self.limit if not self.state
                          else value > self.limit - self.deadband)
        else:
            self.state = (value <= self.limit if not self.state
                          else value < self.limit + self.deadband)
        return self.state


@dataclass
class OnDelay:
    """TON. Output goes true once the input has been true for `preset`.

    Counts scans, not seconds. Accumulating `elapsed += ts` in floating point
    means ten scans of 0.1 s reach 0.9999999999999999, and a timer compared
    against its preset with `>=` then never fires at all. Real PLC timers count
    ticks for exactly this reason.
    """

    preset: float  # s
    ts: float  # s, scan period
    count: int = 0
    state: bool = False

    def __post_init__(self) -> None:
        self.preset_scans = max(1, round(self.preset / self.ts))

    @property
    def elapsed(self) -> float:
        return self.count * self.ts

    def reset(self) -> None:
        self.count = 0
        self.state = False

    def __call__(self, input_: bool) -> bool:
        self.count = min(self.count + 1, self.preset_scans) if input_ else 0
        self.state = self.count >= self.preset_scans
        return self.state


@dataclass(frozen=True)
class PlcInput:
    """Input image table: one frozen snapshot per scan."""

    # from the plant
    LT101: float = 0.0
    LT102: float = 0.0
    FT101: float = 0.0
    FT102: float = 0.0
    PT101: float = 0.0
    ZT101: float = 0.0
    PMP101_RUN: bool = False
    # from the operator
    LIC101_SP: float = DEFAULT_SP
    LIC101_MAN_OP: float = 0.0
    LIC101_AUTO: bool = True
    CMD_PMP101_START: bool = False
    CMD_ALM_RESET: bool = False


@dataclass(frozen=True)
class PlcOutput:
    """Output image table: everything the program decides in one scan."""

    FCV101_OP: float
    PMP101_CMD: bool
    LIC101_SP: float
    LIC101_PV: float
    LIC101_OP: float
    LIC101_ERR: float
    LIC101_MAN_OP: float
    LIC101_AUTO: bool
    CMD_ALM_RESET: bool
    CMD_PMP101_START: bool
    ALM_LT101_HI: bool
    ALM_LT101_LOLO: bool
    ALM_PT101_HIHI: bool
    ILK_PMP101: bool

    def as_tags(self) -> dict[str, float | bool]:
        return dict(vars(self))


@dataclass
class Program:
    """Retained state plus one scan of logic."""

    ts: float = 0.10
    tuning: PidTuning = field(default_factory=PidTuning)
    level_deadband: float = 10.0  # mm
    pressure_deadband: float = 2.0  # kPa
    flow_deadband: float = 0.5  # L/min

    pid: PID = field(init=False)
    _alm_hi: Hysteresis = field(init=False)
    _alm_lolo: Hysteresis = field(init=False)
    _alm_pt: Hysteresis = field(init=False)
    _trip_level: Hysteresis = field(init=False)
    _low_flow: Hysteresis = field(init=False)
    _low_flow_timer: OnDelay = field(init=False)
    interlocked: bool = False
    first_scan: bool = True
    scans: int = 0

    def __post_init__(self) -> None:
        self.pid = PID(tuning=self.tuning, ts=self.ts)
        self._alm_hi = Hysteresis(LT101_HI, self.level_deadband, high=True)
        self._alm_lolo = Hysteresis(LT101_LOLO, self.level_deadband, high=False)
        self._alm_pt = Hysteresis(PT101_HIHI, self.pressure_deadband, high=True)
        self._trip_level = Hysteresis(LT101_HIHI, self.level_deadband, high=True)
        self._low_flow = Hysteresis(FT101_LO, self.flow_deadband, high=False)
        self._low_flow_timer = OnDelay(MIN_FLOW_DELAY, self.ts)

    # -- one scan ---------------------------------------------------------

    def execute(self, i: PlcInput) -> PlcOutput:
        sp = min(max(i.LIC101_SP, TAGS["LIC101_SP"].lo), TAGS["LIC101_SP"].hi)
        pv = i.LT101

        if self.first_scan:
            # Seed the output from valve feedback so the first scan is bumpless
            # against a plant that may already be running.
            self.pid.track(i.ZT101, sp, pv)

        alm_hi = self._alm_hi(pv)
        alm_lolo = self._alm_lolo(pv)
        alm_pt = self._alm_pt(i.PT101)
        trip_level = self._trip_level(pv)
        starved = self._low_flow_timer(
            i.CMD_PMP101_START and not self.interlocked and self._low_flow(i.FT101)
        )

        # Latch on any trip condition. Reset requires the condition to be gone.
        if trip_level or starved:
            self.interlocked = True
        elif self.interlocked and i.CMD_ALM_RESET:
            self.interlocked = False
            self._low_flow_timer.reset()

        if self.interlocked:
            # Fail safe: shut the valve, stop the pump, and keep the PID
            # tracking the shut position so a reset cannot slam it open.
            self.pid.track(0.0, sp, pv)
            op, pump = 0.0, False
        else:
            op = self.pid.step(sp, pv, i.LIC101_AUTO, i.LIC101_MAN_OP)
            pump = i.CMD_PMP101_START

        self.first_scan = False
        self.scans += 1

        return PlcOutput(
            FCV101_OP=op,
            PMP101_CMD=pump,
            LIC101_SP=sp,
            LIC101_PV=pv,
            LIC101_OP=op,
            LIC101_ERR=sp - pv,
            LIC101_MAN_OP=i.LIC101_MAN_OP,
            LIC101_AUTO=i.LIC101_AUTO and not self.interlocked,
            CMD_ALM_RESET=i.CMD_ALM_RESET,
            CMD_PMP101_START=i.CMD_PMP101_START,
            ALM_LT101_HI=alm_hi,
            ALM_LT101_LOLO=alm_lolo,
            ALM_PT101_HIHI=alm_pt,
            ILK_PMP101=self.interlocked,
        )
