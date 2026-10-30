"""Real-time driver, and the in-process link between plant and PLC.

The plant and the PLC are now separate tasks on separate clocks, which is the
arrangement step 5 keeps when the link becomes Modbus. The plant integrates
every `dt`; the PLC scans every `period`; neither waits for the other. The
zero-order hold that `Simulation` faked in step 2 is now physical: the PLC
writes `sim.u` when it feels like it, and the plant keeps integrating with
whatever it last received.

Concurrency
-----------
Both tasks live on one asyncio event loop and neither awaits in the middle of
mutating shared state, so no locking is needed. `LocalIOLink.read_inputs` takes
its snapshot in a single expression and `write_outputs` assigns one object.
That is the whole discipline, and it stops being sufficient the moment anyone
introduces a thread; step 5 keeps everything on the one loop for this reason.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable

from pyplant.clock import Ticker
from pyplant.core.hybrid_tank import Inputs
from pyplant.plc.program import PlcInput, PlcOutput
from pyplant.plc.scan import ScanStats
from pyplant.runners.base import Simulation


@dataclass
class PlantStats:
    steps: int = 0
    overruns: int = 0
    late_ms: float = 0.0


@dataclass
class PlantService:
    """Integrates a `Simulation` against a clock instead of as fast as possible."""

    sim: Simulation
    speed: float = 1.0
    clock: Callable[[], float] = time.perf_counter
    ticker: Ticker | None = None
    stats: PlantStats = field(default_factory=PlantStats)

    def __post_init__(self) -> None:
        if self.sim.policy is not None:
            raise ValueError(
                "a real-time plant must not carry a policy; the PLC writes its "
                "inputs over the link"
            )
        if self.ticker is None:
            self.ticker = Ticker(self.sim.dt, speed=self.speed, clock=self.clock)

    async def run(
        self,
        steps: int | None = None,
        stop: asyncio.Event | None = None,
    ) -> None:
        if steps is None and stop is None:
            raise ValueError("run() needs a step count or a stop event")
        done = 0
        while True:
            tick = await self.ticker.wait()
            self.sim.integrate()
            self.stats.steps += 1
            self.stats.late_ms = max(tick.lateness, 0.0) * 1000.0
            if tick.overrun:
                self.stats.overruns += 1
            done += 1
            if steps is not None and done >= steps:
                return
            if stop is not None and stop.is_set():
                return


@dataclass
class OperatorPanel:
    """Operator-writable values. The HMI replaces this in step 8."""

    setpoint: float = 300.0
    auto: bool = True
    manual_op: float = 0.0
    pump_start: bool = True
    alarm_reset: bool = False


@dataclass
class LocalIOLink:
    """In-process `IOLink`. Step 5 swaps this for Modbus."""

    sim: Simulation
    panel: OperatorPanel = field(default_factory=OperatorPanel)
    reads: int = 0
    writes: int = 0
    published: dict[str, float | bool] = field(default_factory=dict)

    async def read_inputs(self) -> PlcInput:
        m = self.sim.measurements
        self.reads += 1
        return PlcInput(
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

    async def write_outputs(self, out: PlcOutput, stats: ScanStats) -> None:
        # The stats published here are last scan's: this scan's duration is not
        # known until the write completes. Real controllers publish the previous
        # cycle's timing for the same reason.
        self.sim.u = Inputs(valve_cmd=out.FCV101_OP / 100.0, pump_on=out.PMP101_CMD)
        self.writes += 1
        self.published = {**out.as_tags(), **stats.as_tags()}
