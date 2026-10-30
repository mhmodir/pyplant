"""Tests for the scan engine and real-time driving.

Timing tests are usually flaky, so almost everything here runs on a virtual
clock: `ManualClock` only advances when something sleeps or when a stub
deliberately burns time. That makes drift, lateness and overrun counting exact
rather than approximate. Exactly one test touches the real clock, and its
tolerances are loose on purpose.
"""

from __future__ import annotations

import asyncio

import pytest

from pyplant.clock import ManualClock, Tick, Ticker
from pyplant.core.hybrid_tank import H1, Inputs
from pyplant.plc.program import PlcInput, PlcOutput, Program
from pyplant.plc.scan import (
    HEARTBEAT_WRAP,
    OVERRUN_MAX,
    SCAN_MS_MAX,
    IOLink,
    ScanEngine,
    ScanStats,
)
from pyplant.runners.base import Simulation
from pyplant.runners.realtime import LocalIOLink, OperatorPanel, PlantService
from pyplant.tags import TAGS


def run(coro):
    return asyncio.run(coro)


# ============================== Ticker =====================================

def test_ticker_validates_its_arguments():
    with pytest.raises(ValueError):
        Ticker(period=0.0)
    with pytest.raises(ValueError):
        Ticker(period=0.1, speed=0.0)


def test_first_wait_fixes_the_origin_and_does_not_sleep():
    c = ManualClock(start=5.0)
    t = Ticker(0.1, clock=c.now, sleeper=c.sleep)
    tick = run(t.wait())
    assert tick.index == 0 and c.now() == 5.0
    assert t.origin == 5.0


def test_schedule_does_not_drift_when_the_body_burns_time():
    # The whole point of the class. Each body takes 70 ms of a 100 ms period;
    # after 100 ticks the phase must still be exactly on the grid.
    c = ManualClock()
    t = Ticker(0.1, clock=c.now, sleeper=c.sleep)

    async def go():
        await t.wait()
        for _ in range(100):
            c.advance(0.07)  # work
            tick = await t.wait()
            assert tick.lateness == pytest.approx(0.0, abs=1e-12)
        return tick

    last = run(go())
    assert last.index == 100
    assert last.target == pytest.approx(10.0, abs=1e-12)
    assert c.now() == pytest.approx(10.0, abs=1e-12)  # not 10.0 + 100 * 0.07


def test_late_body_is_reported_but_the_grid_holds():
    c = ManualClock()
    t = Ticker(0.1, clock=c.now, sleeper=c.sleep)

    async def go():
        await t.wait()
        c.advance(0.25)  # a scan that took two and a half periods
        return await t.wait()

    tick = run(go())
    assert tick.overrun is True
    assert tick.lateness == pytest.approx(0.15)
    assert tick.target == pytest.approx(0.10)  # grid unmoved


def test_speed_scales_the_period():
    c = ManualClock()
    t = Ticker(1.0, speed=100.0, clock=c.now, sleeper=c.sleep)

    async def go():
        await t.wait()
        await t.wait()

    run(go())
    assert t.scaled_period == pytest.approx(0.01)
    assert c.now() == pytest.approx(0.01)


# ============================ ScanStats ====================================

def test_heartbeat_wraps_inside_its_tag_range():
    s = ScanStats(heartbeat=HEARTBEAT_WRAP - 1)
    s.record(0.001, None)
    assert s.heartbeat == 0
    assert TAGS["PLC_HEARTBEAT"].lo <= s.heartbeat <= TAGS["PLC_HEARTBEAT"].hi


def test_overrun_counter_saturates_rather_than_wrapping():
    s = ScanStats(overruns=OVERRUN_MAX)
    s.record(0.001, Tick(1, 0.0, 99.0, 0.1))
    assert s.overruns == OVERRUN_MAX


def test_scan_ms_is_capped_at_the_tag_span():
    s = ScanStats()
    s.record(999.0, None)
    assert s.scan_ms == SCAN_MS_MAX


def test_peak_scan_time_is_retained():
    s = ScanStats()
    s.record(0.020, None)
    s.record(0.005, None)
    assert s.scan_ms == pytest.approx(5.0)
    assert s.scan_ms_peak == pytest.approx(20.0)


def test_published_stats_are_real_tags_within_range():
    s = ScanStats(scan_ms=12.3, overruns=4, heartbeat=77)
    for name, value in s.as_tags().items():
        t = TAGS[name]
        assert t.lo <= value <= t.hi
        assert t.decode(t.encode(value)) == pytest.approx(value, abs=t.resolution)


# ============================ ScanEngine ===================================

class StubLink:
    """Records calls and can burn virtual time inside the scan."""

    def __init__(self, clock: ManualClock | None = None, read_cost: float = 0.0,
                 write_cost: float = 0.0) -> None:
        self.clock, self.read_cost, self.write_cost = clock, read_cost, write_cost
        self.reads = self.writes = 0
        self.inputs = PlcInput(LT101=300.0, ZT101=66.2, FT101=10.0,
                               CMD_PMP101_START=True)
        self.last: PlcOutput | None = None
        self.published: dict = {}

    async def read_inputs(self) -> PlcInput:
        self.reads += 1
        if self.clock:
            self.clock.advance(self.read_cost)
        return self.inputs

    async def write_outputs(self, out, stats) -> None:
        self.writes += 1
        if self.clock:
            self.clock.advance(self.write_cost)
        self.last = out
        self.published = {**out.as_tags(), **stats.as_tags()}


def test_engine_rejects_a_period_that_contradicts_the_program():
    with pytest.raises(ValueError, match="factor of"):
        ScanEngine(program=Program(ts=0.10), io=StubLink(), period=0.50)


def test_run_requires_a_termination_condition():
    engine = ScanEngine(program=Program(ts=0.10), io=StubLink())
    with pytest.raises(ValueError, match="scan count or a stop event"):
        run(engine.run())


def test_one_scan_is_exactly_one_read_and_one_write():
    link = StubLink()
    engine = ScanEngine(program=Program(ts=0.10), io=link)
    run(engine.scan_once())
    assert (link.reads, link.writes) == (1, 1)


def test_logic_sees_the_frozen_input_image():
    link = StubLink()
    engine = ScanEngine(program=Program(ts=0.10), io=link)
    run(engine.scan_once())
    assert engine.last_input is link.inputs
    assert engine.last_output is link.last


def test_scan_time_is_measured_across_read_execute_and_write():
    c = ManualClock()
    link = StubLink(clock=c, read_cost=0.004, write_cost=0.003)
    engine = ScanEngine(program=Program(ts=0.10), io=link, clock=c.now,
                        ticker=Ticker(0.10, clock=c.now, sleeper=c.sleep))
    run(engine.scan_once())
    assert engine.stats.scan_ms == pytest.approx(7.0)


def test_scan_count_and_heartbeat_track_each_other():
    c = ManualClock()
    link = StubLink(clock=c)
    engine = ScanEngine(program=Program(ts=0.10), io=link, clock=c.now,
                        ticker=Ticker(0.10, clock=c.now, sleeper=c.sleep))
    run(engine.run(scans=25))
    assert engine.stats.scans == 25 and engine.stats.heartbeat == 25
    assert link.reads == 25


def test_overruns_are_counted_only_when_a_slot_is_missed():
    c = ManualClock()
    quick = StubLink(clock=c, read_cost=0.002)
    engine = ScanEngine(program=Program(ts=0.10), io=quick, clock=c.now,
                        ticker=Ticker(0.10, clock=c.now, sleeper=c.sleep))
    run(engine.run(scans=20))
    assert engine.stats.overruns == 0

    c2 = ManualClock()
    slow = StubLink(clock=c2, read_cost=0.25)  # two and a half periods
    engine2 = ScanEngine(program=Program(ts=0.10), io=slow, clock=c2.now,
                         ticker=Ticker(0.10, clock=c2.now, sleeper=c2.sleep))
    run(engine2.run(scans=20))
    # 19, not 20: the first tick fixes the phase origin, so it cannot be late.
    assert engine2.stats.overruns == 19
    # Lateness against a fixed grid grows without bound when the engine simply
    # cannot keep up, which is the signal to look at rather than the count.
    assert engine2.stats.late_ms > 1000.0


def test_stop_event_ends_the_loop():
    stop = asyncio.Event()
    c = ManualClock()
    link = StubLink(clock=c)
    engine = ScanEngine(program=Program(ts=0.10), io=link, clock=c.now,
                        ticker=Ticker(0.10, clock=c.now, sleeper=c.sleep))

    async def go():
        async def stopper():
            await asyncio.sleep(0)
            stop.set()
        await asyncio.gather(engine.run(stop=stop), stopper())

    run(go())
    assert engine.stats.scans >= 1


# ========================== PlantService ===================================

def test_plant_service_refuses_a_simulation_that_carries_a_policy():
    with pytest.raises(ValueError, match="must not carry a policy"):
        PlantService(sim=Simulation(policy=lambda t, m: Inputs()))


def test_plant_service_integrates_on_its_own_clock():
    c = ManualClock()
    sim = Simulation(dt=0.05, u=Inputs(0.8, True))
    svc = PlantService(sim=sim, clock=c.now,
                       ticker=Ticker(0.05, clock=c.now, sleeper=c.sleep))
    run(svc.run(steps=200))
    assert svc.stats.steps == 200
    assert sim.t == pytest.approx(10.0)
    assert sim.x[H1] > 0.0


# ======================== engine + plant together ==========================

def test_engine_and_plant_reach_setpoint_in_lockstep():
    # Deterministic integration check with no clock involved: two plant steps
    # per scan, which is what the 50 ms / 100 ms rates mean.
    sim = Simulation(dt=0.05)  # empty tanks, no policy
    link = LocalIOLink(sim, OperatorPanel(setpoint=300.0, pump_start=True))
    engine = ScanEngine(program=Program(ts=0.10), io=link, period=0.10)

    async def go():
        for i in range(int(2400.0 / 0.05)):
            if i % 2 == 0:
                await engine.scan_once()
            sim.integrate()

    run(go())
    assert sim.measurements["LT101"] == pytest.approx(300.0, abs=1.0)
    assert link.published["ILK_PMP101"] is False
    assert link.published["PLC_HEARTBEAT"] > 0


def test_two_tasks_hold_their_rate_ratio_on_a_real_clock():
    # The one wall-clock test. Loose tolerances; the point is that the plant
    # really does tick twice per scan when both run concurrently.
    sim = Simulation(dt=0.05)
    link = LocalIOLink(sim, OperatorPanel(setpoint=300.0, pump_start=True))
    engine = ScanEngine(program=Program(ts=0.10), io=link, period=0.10, speed=10.0)
    plant = PlantService(sim=sim, speed=10.0)

    async def go():
        stop = asyncio.Event()

        async def timer():
            await asyncio.sleep(2.0)
            stop.set()

        await asyncio.gather(plant.run(stop=stop), engine.run(stop=stop), timer())

    run(go())
    assert plant.stats.steps > 100
    ratio = plant.stats.steps / engine.stats.scans
    assert 1.6 < ratio < 2.4
    assert sim.measurements["LT101"] > 0.0
