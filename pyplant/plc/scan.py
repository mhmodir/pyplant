"""The scan engine.

This is what turns `Program` into a PLC. Every scan does exactly three things,
in order:

    read inputs  ->  execute logic  ->  write outputs

The logic runs against a frozen input image. It never reaches out mid-scan for
a fresh reading, which is why its behaviour is reproducible and why a scan can
be replayed in a test. All I/O lives behind `IOLink`, so the same `Program` and
the same engine run against an in-process plant here and against Modbus in step
5 with nothing changed but the link.

The engine measures itself. `SCAN_MS`, `SCAN_OVERRUNS` and `PLC_HEARTBEAT` are
real tags on the wire, trended on the HMI like any process value. Being able to
point at a scan-time trend is the difference between a script on a timer and
something that behaves like a controller.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from pyplant.clock import Tick, Ticker
from pyplant.plc.program import PlcInput, PlcOutput, Program
from pyplant.tags import TAGS

HEARTBEAT_WRAP = int(TAGS["PLC_HEARTBEAT"].hi) + 1
OVERRUN_MAX = int(TAGS["SCAN_OVERRUNS"].hi)
SCAN_MS_MAX = TAGS["SCAN_MS"].hi


@dataclass
class ScanStats:
    """Self-diagnostics, published as tags."""

    scans: int = 0
    scan_ms: float = 0.0
    scan_ms_peak: float = 0.0
    overruns: int = 0
    late_ms: float = 0.0
    heartbeat: int = 0

    def record(self, duration_s: float, tick: Tick | None) -> None:
        self.scans += 1
        self.scan_ms = min(duration_s * 1000.0, SCAN_MS_MAX)
        self.scan_ms_peak = max(self.scan_ms_peak, self.scan_ms)
        self.heartbeat = (self.heartbeat + 1) % HEARTBEAT_WRAP
        if tick is not None:
            self.late_ms = max(tick.lateness, 0.0) * 1000.0
            if tick.overrun:
                self.overruns = min(self.overruns + 1, OVERRUN_MAX)

    @property
    def falling_behind(self) -> bool:
        """Lateness against a fixed grid grows without bound when the engine
        cannot keep up. The trend of `late_ms`, not the overrun count, is what
        says the scan period is too short for the work in it."""
        return self.late_ms >= self.scan_ms

    def as_tags(self) -> dict[str, float]:
        return {
            "SCAN_MS": self.scan_ms,
            "SCAN_OVERRUNS": float(self.overruns),
            "PLC_HEARTBEAT": float(self.heartbeat),
        }


class IOLink(Protocol):
    """Everything the engine needs from the outside world.

    Step 4 supplies an in-process implementation; step 5 supplies a Modbus one.
    Nothing above this interface changes when the transport does.
    """

    async def read_inputs(self) -> PlcInput: ...

    async def write_outputs(self, out: PlcOutput, stats: ScanStats) -> None: ...


@dataclass
class ScanEngine:
    """Drives one `Program` at a fixed period."""

    program: Program
    io: IOLink
    period: float = 0.10
    speed: float = 1.0
    clock: Callable[[], float] = time.perf_counter
    ticker: Ticker | None = None
    stats: ScanStats = field(default_factory=ScanStats)
    last_input: PlcInput | None = field(default=None, init=False)
    last_output: PlcOutput | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if abs(self.program.ts - self.period) > 1e-9:
            raise ValueError(
                f"program.ts is {self.program.ts} s but the engine scans every "
                f"{self.period} s; controller gains and timer presets would be "
                f"wrong by a factor of {self.period / self.program.ts:.3g}"
            )
        if self.ticker is None:
            self.ticker = Ticker(self.period, speed=self.speed, clock=self.clock)

    async def scan_once(self, tick: Tick | None = None) -> PlcOutput:
        """One read, one execute, one write."""
        started = self.clock()
        inputs = await self.io.read_inputs()
        out = self.program.execute(inputs)
        await self.io.write_outputs(out, self.stats)
        self.stats.record(self.clock() - started, tick)
        self.last_input, self.last_output = inputs, out
        return out

    async def run(
        self,
        scans: int | None = None,
        stop: asyncio.Event | None = None,
    ) -> None:
        """Scan until `scans` have run or `stop` is set. One or the other."""
        if scans is None and stop is None:
            raise ValueError("run() needs a scan count or a stop event")
        done = 0
        while True:
            tick = await self.ticker.wait()
            await self.scan_once(tick)
            done += 1
            if scans is not None and done >= scans:
                return
            if stop is not None and stop.is_set():
                return
