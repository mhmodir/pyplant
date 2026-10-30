"""Fault models.

Faults are objects applied to a running plant, never edits to the physics.
There are three places a fault can enter, and which one it uses is the most
important thing about it:

    parameter   changes the plant itself      leak, fouling, impeller wear
    command     sits between PLC and valve    stiction
    signal      corrupts what an instrument reports    bias, drift, stuck, noise

The distinction is what makes diagnosis a real problem rather than a lookup. A
level transmitter reading 30 mm high and a genuine 30 mm rise produce identical
numbers in `LT101`. They differ only in what the *rest* of the plant does, and
a model that has never seen both cannot tell them apart.

Every fault carries a start, an optional end, and a ramp. Ramp length is not
cosmetic: an abrupt leak and an incipient one over an hour are different
detection problems, and a set of scenarios containing only step changes trains
a detector that is useless on the ones that creep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from pyplant.core.hybrid_tank import Inputs, Params
from pyplant.tags import TAGS

Measurements = dict[str, float | bool]


@dataclass
class Fault:
    """Common timing. Subclasses add the effect."""

    id: str
    t_start: float = 0.0
    t_end: float | None = None
    ramp: float = 0.0  # s to reach full magnitude

    kind: str = field(init=False, default="fault")
    target: str = field(init=False, default="")
    benign: bool = False  # always-on realism (noise), never a positive label

    def __post_init__(self) -> None:
        if self.ramp < 0.0:
            raise ValueError("ramp cannot be negative")
        if self.t_end is not None and self.t_end <= self.t_start:
            raise ValueError("t_end must follow t_start")

    def strength(self, t: float) -> float:
        """0 before it starts, 1 at full magnitude, 0 once it is over."""
        if t < self.t_start or (self.t_end is not None and t >= self.t_end):
            return 0.0
        if self.ramp <= 0.0:
            return 1.0
        return min((t - self.t_start) / self.ramp, 1.0)

    @property
    def magnitude(self) -> float:  # pragma: no cover - overridden
        return 0.0

    def label(self) -> dict:
        return {
            "fault_id": self.id,
            "kind": self.kind,
            "target": self.target,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "ramp": self.ramp,
            "magnitude": self.magnitude,
            "benign": self.benign,
        }


# ---------------------------------------------------------------------------
# Parameter faults: the plant itself changes
# ---------------------------------------------------------------------------

@dataclass
class ParameterFault(Fault):
    def apply(self, p: Params, s: float) -> Params:  # pragma: no cover
        raise NotImplementedError


@dataclass
class Leak(ParameterFault):
    """A hole in tank 1. Effective discharge area in m^2."""

    cda: float = 2.0e-5

    def __post_init__(self) -> None:
        super().__post_init__()
        self.kind, self.target = "leak", "TANK1"

    @property
    def magnitude(self) -> float:
        return self.cda

    def apply(self, p: Params, s: float) -> Params:
        return replace(p, cda_leak=p.cda_leak + self.cda * s)


@dataclass
class Fouling(ParameterFault):
    """Restriction of the tank 1 to tank 2 interconnect, as fraction lost."""

    loss: float = 0.30

    def __post_init__(self) -> None:
        super().__post_init__()
        self.kind, self.target = "fouling", "FE-12"
        if not 0.0 <= self.loss < 1.0:
            raise ValueError("loss must be in [0, 1)")

    @property
    def magnitude(self) -> float:
        return self.loss

    def apply(self, p: Params, s: float) -> Params:
        return replace(p, cda_12_gain=p.cda_12_gain * (1.0 - self.loss * s))


@dataclass
class PumpWear(ParameterFault):
    """Impeller wear, as fraction of developed head lost."""

    loss: float = 0.15

    def __post_init__(self) -> None:
        super().__post_init__()
        self.kind, self.target = "pump_wear", "PMP101"
        if not 0.0 <= self.loss < 1.0:
            raise ValueError("loss must be in [0, 1)")

    @property
    def magnitude(self) -> float:
        return self.loss

    def apply(self, p: Params, s: float) -> Params:
        return replace(p, pump_gain=p.pump_gain * (1.0 - self.loss * s))


# ---------------------------------------------------------------------------
# Command faults: the valve does not do what it is told
# ---------------------------------------------------------------------------

@dataclass
class ValveStiction(Fault):
    """Deadband-plus-slip-jump stiction on FCV101.

    The classic two-parameter data-driven model. The stem does not move until
    the command has moved `S` away from where the stem is stuck; then it breaks
    free and jumps by `J`. In a closed loop with integral action the result is
    a limit cycle: the controller keeps winding until the stem slips, overshoots,
    and winds back the other way.

    This is the most control-relevant fault in the set and the one a detector
    fed only `LT101` will find hardest, because the signature lives in the phase
    relationship between `FCV101_OP` and `ZT101`.
    """

    deadband: float = 0.06  # S, as a fraction of full stroke
    slip: float = 0.04  # J

    _y: float = field(default=0.0, init=False)
    _u_prev: float = field(default=0.0, init=False)
    _dir: float = field(default=0.0, init=False)
    _moving: bool = field(default=False, init=False)
    _primed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.kind, self.target = "stiction", "FCV101"
        if self.slip > self.deadband:
            raise ValueError("slip jump cannot exceed the deadband")

    @property
    def magnitude(self) -> float:
        return self.deadband

    def command(self, u: Inputs, s: float) -> Inputs:
        cmd = min(max(u.valve_cmd, 0.0), 1.0)
        if not self._primed:
            self._y = self._u_prev = cmd
            self._primed = True
        if s <= 0.0:  # healthy: the stem follows exactly
            self._y = self._u_prev = cmd
            self._moving = False
            return u

        band, jump = self.deadband * s, self.slip * s
        delta = cmd - self._u_prev

        if self._moving and delta != 0.0 and math.copysign(1.0, delta) != self._dir:
            self._moving = False  # reversal: it sticks again

        if self._moving:
            self._y = cmd  # already broken free: the stem follows
        elif abs(cmd - self._y) > band:
            # Break away, and slip past by the jump rather than landing on the
            # command. Assigning cmd here as well would erase the jump, and
            # with it the entire limit cycle.
            self._dir = math.copysign(1.0, cmd - self._y)
            self._moving = True
            self._y = cmd - self._dir * (band - jump)

        self._u_prev = cmd
        return replace(u, valve_cmd=min(max(self._y, 0.0), 1.0))


# ---------------------------------------------------------------------------
# Signal faults: the instrument lies
# ---------------------------------------------------------------------------

@dataclass
class SignalFault(Fault):
    tag: str = "LT101"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.tag not in TAGS:
            raise ValueError(f"unknown tag {self.tag!r}")
        self.target = self.tag

    def corrupt(self, value: float, s: float, t: float,
                rng: np.random.Generator) -> float:  # pragma: no cover
        raise NotImplementedError


@dataclass
class SensorBias(SignalFault):
    """A fixed offset in the tag's engineering units."""

    bias: float = 25.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.kind = "sensor_bias"

    @property
    def magnitude(self) -> float:
        return self.bias

    def corrupt(self, value, s, t, rng):
        return value + self.bias * s


@dataclass
class SensorDrift(SignalFault):
    """An offset that accumulates, in engineering units per minute."""

    rate_per_min: float = 5.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.kind = "sensor_drift"

    @property
    def magnitude(self) -> float:
        return self.rate_per_min

    def corrupt(self, value, s, t, rng):
        if s <= 0.0:
            return value
        elapsed_min = max(t - self.t_start, 0.0) / 60.0
        return value + self.rate_per_min * elapsed_min


@dataclass
class SensorStuck(SignalFault):
    """The reading freezes at whatever it was when the fault began."""

    _held: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.kind = "sensor_stuck"

    @property
    def magnitude(self) -> float:
        return 1.0

    def corrupt(self, value, s, t, rng):
        if s <= 0.0:
            self._held = None
            return value
        if self._held is None:
            self._held = value
        return self._held


@dataclass
class SensorNoise(SignalFault):
    """Gaussian noise, sigma in engineering units. Seeded, so runs repeat."""

    sigma: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.kind = "sensor_noise"

    @property
    def magnitude(self) -> float:
        return self.sigma

    def corrupt(self, value, s, t, rng):
        return value + float(rng.normal(0.0, self.sigma * s)) if s > 0.0 else value


# ---------------------------------------------------------------------------
# The set applied to a running plant
# ---------------------------------------------------------------------------

@dataclass
class FaultSet:
    """Everything wrong with the plant right now, and when it started."""

    faults: list[Fault] = field(default_factory=list)
    seed: int = 0
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        ids = [f.id for f in self.faults]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate fault ids: {ids}")

    # -- editing, for the live plant's injection API -----------------------

    def add(self, fault: Fault) -> None:
        if any(f.id == fault.id for f in self.faults):
            raise ValueError(f"fault {fault.id!r} already present")
        self.faults.append(fault)

    def remove(self, fault_id: str) -> bool:
        before = len(self.faults)
        self.faults = [f for f in self.faults if f.id != fault_id]
        return len(self.faults) != before

    def clear(self) -> None:
        self.faults = []

    # -- application -------------------------------------------------------

    def active(self, t: float) -> list[Fault]:
        return [f for f in self.faults if f.strength(t) > 0.0]

    def params(self, base: Params, t: float) -> Params:
        p = base
        for f in self.faults:
            if isinstance(f, ParameterFault):
                s = f.strength(t)
                if s > 0.0:
                    p = f.apply(p, s)
        return p

    def command(self, u: Inputs, t: float) -> Inputs:
        for f in self.faults:
            if isinstance(f, ValveStiction):
                u = f.command(u, f.strength(t))
        return u

    def measure(self, m: Measurements, t: float) -> Measurements:
        out = dict(m)
        for f in self.faults:
            if isinstance(f, SignalFault):
                s = f.strength(t)
                value = out.get(f.tag)
                if value is None or isinstance(value, bool):
                    continue
                corrupted = f.corrupt(float(value), s, t, self.rng)
                out[f.tag] = TAGS[f.tag].clamp(corrupted)
        return out

    # -- ground truth ------------------------------------------------------

    def events(self) -> list[dict]:
        """Ground truth. Benign entries are listed but never labelled faulty."""
        return [f.label() for f in self.faults if not f.benign]

    def labels_at(self, t: float) -> str:
        """The label a sample at time `t` carries. `normal` when nothing is on."""
        on = [f for f in self.active(t) if not f.benign]
        return "+".join(sorted(f.id for f in on)) if on else "normal"


# ---------------------------------------------------------------------------
# Construction from a description, for the live injection API
# ---------------------------------------------------------------------------

# kind -> (class, the attribute a single "magnitude" number sets)
FAULT_KINDS: dict[str, tuple[type, str | None]] = {
    "leak": (Leak, "cda"),
    "fouling": (Fouling, "loss"),
    "pump_wear": (PumpWear, "loss"),
    "stiction": (ValveStiction, "deadband"),
    "sensor_bias": (SensorBias, "bias"),
    "sensor_drift": (SensorDrift, "rate_per_min"),
    "sensor_stuck": (SensorStuck, None),
    "sensor_noise": (SensorNoise, "sigma"),
}


def build_fault(kind: str, id: str, magnitude: float | None = None,
                t_start: float = 0.0, t_end: float | None = None,
                ramp: float = 0.0, tag: str | None = None, **extra) -> Fault:
    """Make a fault from a plain description, as an HTTP body supplies it."""
    if kind not in FAULT_KINDS:
        raise ValueError(f"unknown fault kind {kind!r}; "
                         f"expected one of {sorted(FAULT_KINDS)}")
    cls, magnitude_field = FAULT_KINDS[kind]
    kwargs: dict = {"id": id, "t_start": t_start, "t_end": t_end, "ramp": ramp}
    if issubclass(cls, SignalFault):
        kwargs["tag"] = tag or "LT101"
    elif tag is not None:
        raise ValueError(f"{kind} is not a sensor fault; it takes no tag")
    if magnitude is not None:
        if magnitude_field is None:
            raise ValueError(f"{kind} takes no magnitude")
        kwargs[magnitude_field] = magnitude
    kwargs.update(extra)
    return cls(**kwargs)
