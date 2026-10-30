"""In-process wiring between the PLC program and the plant.

This is a development and test shim. It hands the program a `PlcInput` built
straight from the plant's measurements, with no wire in between. Step 5
replaces it with Modbus, at which point the same `Program` object is driven by
the scan engine instead and this module is only used by tests.

Because it is a shim, it is the one place allowed to know both sides.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from pyplant.core.hybrid_tank import Inputs
from pyplant.plc.program import PlcInput, PlcOutput, Program
from pyplant.runners.base import Measurements


@dataclass
class OperatorPanel:
    """What a human would be holding: setpoint, mode, and the two buttons."""

    setpoint: float = 300.0
    auto: bool = True
    manual_op: float = 0.0
    pump_start: bool = True
    alarm_reset: bool = False


_BLANK: dict[str, float | bool] = {
    name: (False if typ == "bool" else 0.0)
    for name, typ in PlcOutput.__annotations__.items()
}


@dataclass
class PlcHarness:
    """Adapts `Program` to the `Policy` interface `Simulation` expects."""

    program: Program = field(default_factory=Program)
    panel: OperatorPanel = field(default_factory=OperatorPanel)
    last_output: PlcOutput | None = None

    def __call__(self, t: float, m: Measurements) -> Inputs:
        i = PlcInput(
            LT101=float(m["LT101"]),
            LT102=float(m["LT102"]),
            FT101=float(m["FT101"]),
            FT102=float(m["FT102"]),
            PT101=float(m["PT101"]),
            ZT101=float(m["ZT101"]),
            PMP101_RUN=bool(m["PMP101_RUN"]),
            LIC101_SP=self.panel.setpoint,
            LIC101_MAN_OP=self.panel.manual_op,
            LIC101_AUTO=self.panel.auto,
            CMD_PMP101_START=self.panel.pump_start,
            CMD_ALM_RESET=self.panel.alarm_reset,
        )
        o = self.program.execute(i)
        self.last_output = o
        return Inputs(valve_cmd=o.FCV101_OP / 100.0, pump_on=o.PMP101_CMD)

    def tags(self) -> dict[str, float | bool]:
        """Last scan's outputs, keyed by tag name.

        Before the first scan this returns the same keys with de-energised
        values rather than an empty dict. A logger fixes its column set from
        the first sample it takes, so a shape that changes on scan one would
        silently drop every PLC tag from a run that starts from cold.
        """
        return self.last_output.as_tags() if self.last_output else dict(_BLANK)


def scheduled_setpoint(harness: PlcHarness, changes: list[tuple[float, float]]):
    """Wrap a harness so the setpoint follows a timetable. Times must ascend."""
    if any(b[0] <= a[0] for a, b in zip(changes, changes[1:])):
        raise ValueError("setpoint change times must strictly ascend")

    def policy(t: float, m: Measurements) -> Inputs:
        for start, sp in changes:
            if t + 1e-9 >= start:
                harness.panel = replace(harness.panel, setpoint=sp)
        return harness(t, m)

    return policy
