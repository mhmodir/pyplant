"""The stepper that every runner drives.

There is exactly one loop body in this project. The fast runner in
`runners.fast` drives it as hard as the CPU allows to generate datasets; the
real-time runner in step 5 drives the same object against a wall clock and
serves its state over Modbus. If the two ever disagree about the plant, it is
because of the clock, not because of the physics.

Three rates, and they are deliberately different
------------------------------------------------
    dt              integration step, 50 ms   - accuracy
    control_period  scan interval,   100 ms   - what the controller sees
    log_period      sample interval,    1 s   - what the historian stores

The control input is held constant between scans, so a controller written
against this object sees a genuine zero-order hold at its own sample time
rather than a continuously updated signal. Getting that wrong makes a PID look
better in simulation than it will ever behave on a PLC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from typing import TYPE_CHECKING

from pyplant.core.hybrid_tank import (
    Inputs,
    Params,
    initial_state,
    measure,
    step,
    steady_state,
)

if TYPE_CHECKING:  # pragma: no cover
    from pyplant.faults.model import FaultSet

Measurements = dict[str, float | bool]


class Policy(Protocol):
    """Anything that decides plant inputs from what it can observe.

    A schedule, a PID, or the soft PLC in step 4 all satisfy this. The policy
    receives measurements in engineering units, never the raw state, because
    that is all a real controller gets.
    """

    def __call__(self, t: float, m: Measurements) -> Inputs: ...


def _is_multiple(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a / b - round(a / b)) < tol


@dataclass
class Simulation:
    """Plant plus policy, advanced one integration step at a time."""

    params: Params = field(default_factory=Params)
    policy: Policy | None = None
    dt: float = 0.05
    control_period: float = 0.10
    x: np.ndarray = field(default_factory=initial_state)
    u: Inputs = field(default_factory=Inputs)
    t: float = 0.0
    faults: "FaultSet | None" = None

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.control_period < self.dt:
            raise ValueError("control_period cannot be shorter than dt")
        if not _is_multiple(self.control_period, self.dt):
            raise ValueError(
                f"control_period {self.control_period} is not a whole number "
                f"of dt {self.dt}; the zero-order hold would drift"
            )
        self.x = np.asarray(self.x, dtype=float).copy()
        self._steps_per_scan = round(self.control_period / self.dt)
        self._step_count = 0

    # -- construction ----------------------------------------------------

    @classmethod
    def at_steady_state(cls, u: Inputs, params: Params | None = None, **kw) -> "Simulation":
        """Start already lined out, so a run contains no startup transient."""
        p = params or Params()
        return cls(params=p, x=steady_state(u, p), u=u, **kw)

    # -- stepping --------------------------------------------------------

    @property
    def true_measurements(self) -> Measurements:
        """What the instruments would read if none of them were lying."""
        return measure(self.x, self.u, self.effective_params)

    @property
    def measurements(self) -> Measurements:
        """What the control system actually sees, faults included."""
        m = self.true_measurements
        return m if self.faults is None else self.faults.measure(m, self.t)

    @property
    def effective_params(self) -> Params:
        """Plant parameters with any parameter faults folded in."""
        if self.faults is None:
            return self.params
        return self.faults.params(self.params, self.t)

    @property
    def effective_input(self) -> Inputs:
        """The command the valve actually receives, stiction included."""
        if self.faults is None:
            return self.u
        return self.faults.command(self.u, self.t)

    @property
    def on_scan_boundary(self) -> bool:
        return self._step_count % self._steps_per_scan == 0

    def evaluate(self) -> None:
        """Run the policy if this is a scan boundary. Idempotent within a step
        only in the sense that it is called exactly once per step; calling it
        twice would advance the controller's integrator twice."""
        if self.policy is not None and self.on_scan_boundary:
            self.u = self.policy(self.t, self.measurements)

    def integrate(self) -> None:
        """Advance the plant one dt with the currently held input.

        Faults are applied here rather than inside the physics, so the same
        injection works for the fast dataset runner and the live plant node.
        """
        self.x = step(self.x, self.effective_input, self.effective_params, self.dt)
        self._step_count += 1
        self.t = self._step_count * self.dt  # accumulate from an integer, not +=

    def advance(self) -> None:
        """One scan evaluation followed by one integration step."""
        self.evaluate()
        self.integrate()


# ---------------------------------------------------------------------------
# Policies that need no controller
# ---------------------------------------------------------------------------

def constant(u: Inputs) -> Policy:
    def policy(t: float, m: Measurements) -> Inputs:
        return u
    return policy


def piecewise(schedule: list[tuple[float, Inputs]]) -> Policy:
    """Hold each input until the next scheduled time. Times must ascend.

    Open-loop excitation for step tests and for the training data that teaches
    a model what normal setpoint changes look like.
    """
    if not schedule:
        raise ValueError("schedule is empty")
    times = [t for t, _ in schedule]
    if times[0] != 0.0:
        raise ValueError("schedule must start at t=0")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("schedule times must strictly ascend")

    def policy(t: float, m: Measurements) -> Inputs:
        chosen = schedule[0][1]
        for start, u in schedule:
            if t + 1e-9 >= start:
                chosen = u
            else:
                break
        return chosen

    return policy
