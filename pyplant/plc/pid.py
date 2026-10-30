"""Velocity-form PID.

Velocity form because the controller output is the integrator state. That has
two consequences worth stating plainly, because they are the reason this form
is used on real controllers:

* **Anti-windup is structural.** When the output clamps, the increment that
  would have pushed past the limit is simply discarded. There is no separate
  integral accumulator left growing behind the limit, so the output leaves the
  clamp on the first scan the error reverses.
* **Bumpless transfer is structural.** In manual the output is whatever the
  operator wrote, and the error history is kept fresh against it. Switching to
  auto continues incrementing from that value, so nothing jumps.

Derivative acts on the measurement, not on the error, so a setpoint step
produces no derivative kick. The measurement is low-pass filtered first;
unfiltered derivative on a noisy level signal is worse than no derivative.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PidTuning:
    """Gains in engineering units: output %, error in PV units.

    Defaults are IMC tuning for the two-tank rig at its design point:
    process gain about 15.4 mm/%, open-loop tau about 72 s, closed-loop
    lambda about 13 s, so Kc = tau / (K * lambda) and Ti = tau. Measured
    closed-loop response to a 50 mm setpoint step: no overshoot, 2% settling
    in roughly 40 s, zero offset.
    """

    kc: float = 0.35  # % per PV unit
    ti: float = 72.0  # s, integral time (0 disables integral action)
    td: float = 0.0  # s, derivative time (0 disables derivative action)
    tf: float = 5.0  # s, derivative filter time constant
    out_lo: float = 0.0  # %
    out_hi: float = 100.0  # %
    direct_acting: bool = False  # False: output falls as PV rises

    def __post_init__(self) -> None:
        if self.kc < 0.0:
            raise ValueError("kc must be non-negative; use direct_acting instead")
        if self.ti < 0.0 or self.td < 0.0 or self.tf <= 0.0:
            raise ValueError("ti and td must be non-negative, tf positive")
        if self.out_hi <= self.out_lo:
            raise ValueError("out_hi must exceed out_lo")


@dataclass
class PID:
    """One control loop. `step()` is called once per scan."""

    tuning: PidTuning = field(default_factory=PidTuning)
    ts: float = 0.10  # s, scan period

    out: float = 0.0
    auto: bool = False
    _e1: float = 0.0
    _pvf: float = 0.0
    _pvf1: float = 0.0
    _pvf2: float = 0.0
    _primed: bool = False

    def __post_init__(self) -> None:
        if self.ts <= 0.0:
            raise ValueError("ts must be positive")

    # -- state -----------------------------------------------------------

    @property
    def saturated(self) -> bool:
        t = self.tuning
        return self.out <= t.out_lo + 1e-12 or self.out >= t.out_hi - 1e-12

    def _prime(self, sp: float, pv: float) -> None:
        """Seed the history so the first scan produces no spurious increment."""
        self._e1 = sp - pv
        self._pvf = self._pvf1 = self._pvf2 = pv
        self._primed = True

    def track(self, out: float, sp: float, pv: float) -> None:
        """Force the output and refresh history. Used in manual and on trip."""
        t = self.tuning
        self.out = min(max(out, t.out_lo), t.out_hi)
        self._e1 = sp - pv
        self._pvf = self._pvf1 = self._pvf2 = pv

    def _filter(self, pv: float) -> float:
        a = self.ts / (self.tuning.tf + self.ts)
        return self._pvf + a * (pv - self._pvf)

    # -- execution -------------------------------------------------------

    def step(self, sp: float, pv: float, auto: bool, manual_out: float) -> float:
        """Advance one scan and return the output in percent."""
        t = self.tuning
        if not self._primed:
            self._prime(sp, pv)

        if not auto:
            self.auto = False
            self.track(manual_out, sp, pv)
            return self.out

        was_manual, self.auto = not self.auto, True
        if was_manual:
            # Output already tracks the manual value; only the history matters.
            self._e1 = sp - pv

        e = sp - pv
        if t.direct_acting:
            e = -e

        pvf = self._filter(pv)

        d_p = t.kc * (e - self._e1)
        d_i = t.kc * self.ts / t.ti * e if t.ti > 0.0 else 0.0
        if t.td > 0.0:
            second_diff = pvf - 2.0 * self._pvf1 + self._pvf2
            sign = 1.0 if t.direct_acting else -1.0
            d_d = sign * t.kc * t.td / self.ts * second_diff
        else:
            d_d = 0.0

        # Clamping the sum is the whole of the anti-windup scheme.
        self.out = min(max(self.out + d_p + d_i + d_d, t.out_lo), t.out_hi)

        self._e1 = e
        self._pvf2, self._pvf1, self._pvf = self._pvf1, pvf, pvf
        return self.out
