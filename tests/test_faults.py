"""Tests for fault injection and scenarios.

Two results here carry most of the weight. `test_stiction_produces_a_limit_cycle`
checks that a sticking valve makes a lined-out loop oscillate rather than
merely sit somewhere slightly wrong, because a stiction model that produces no
limit cycle is not modelling stiction. And
`test_a_stuck_transmitter_walks_the_loop_into_a_trip` checks the cascade a
single frozen instrument causes: the controller chases a reading that cannot
move, closes the valve, starves the pump, and the interlock fires.
"""

from __future__ import annotations

import asyncio
import contextlib

import httpx
import numpy as np
import pytest

from pyplant.core.hybrid_tank import Inputs, Params
from pyplant.faults.model import (
    FAULT_KINDS,
    FaultSet,
    Fouling,
    Leak,
    PumpWear,
    SensorBias,
    SensorDrift,
    SensorNoise,
    SensorStuck,
    ValveStiction,
    build_fault,
)
from pyplant.faults.scenarios import (
    NORMAL_MOVES,
    Scenario,
    ScenarioResult,
    grid,
    library,
    run_scenario,
)
from pyplant.runners.base import Simulation
from pyplant.services.plant import PlantNode, create_control_app

DESIGN = Inputs(0.662, True)


def run(coro):
    return asyncio.run(coro)


def quick(name: str, faults, duration: float = 2400.0, **kw) -> ScenarioResult:
    return run_scenario(Scenario(name, duration, faults, [(0.0, 300.0)],
                                 seed=1, **kw))


def after(result: ScenarioResult, t: float) -> np.ndarray:
    return result.trajectory.t > t


# ============================== timing =====================================

def test_strength_is_zero_before_and_one_after():
    f = Leak("l", t_start=100.0)
    assert f.strength(99.0) == 0.0
    assert f.strength(100.0) == 1.0
    assert f.strength(9999.0) == 1.0


def test_ramp_interpolates_and_saturates():
    f = Leak("l", t_start=100.0, ramp=100.0)
    assert f.strength(100.0) == 0.0
    assert f.strength(150.0) == pytest.approx(0.5)
    assert f.strength(200.0) == 1.0
    assert f.strength(300.0) == 1.0


def test_a_fault_can_end():
    f = Leak("l", t_start=100.0, t_end=200.0)
    assert f.strength(150.0) == 1.0
    assert f.strength(200.0) == 0.0


def test_timing_is_validated():
    with pytest.raises(ValueError, match="ramp"):
        Leak("l", ramp=-1.0)
    with pytest.raises(ValueError, match="t_end"):
        Leak("l", t_start=100.0, t_end=50.0)


# ========================== parameter faults ===============================

def test_parameter_faults_scale_with_strength():
    p = Params()
    assert Leak("l", cda=2e-5).apply(p, 0.5).cda_leak == pytest.approx(1e-5)
    assert Fouling("f", loss=0.4).apply(p, 0.5).cda_12_gain == pytest.approx(0.8)
    assert PumpWear("w", loss=0.2).apply(p, 1.0).pump_gain == pytest.approx(0.8)


def test_fractional_losses_are_bounded():
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        Fouling("f", loss=1.0)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        PumpWear("w", loss=-0.1)


def test_faultset_composes_parameter_faults():
    fs = FaultSet([Leak("l", cda=1e-5), PumpWear("w", loss=0.1)])
    p = fs.params(Params(), 0.0)
    assert p.cda_leak == pytest.approx(1e-5)
    assert p.pump_gain == pytest.approx(0.9)


# ============================== stiction ===================================

def test_stiction_sticks_then_slips_past_the_command():
    f = ValveStiction("s", deadband=0.10, slip=0.06)
    f.command(Inputs(0.50, True), 0.0)  # prime while healthy
    seen = [f.command(Inputs(c, True), 1.0).valve_cmd
            for c in (0.53, 0.56, 0.59, 0.62, 0.65, 0.62, 0.56, 0.50)]
    assert seen[:3] == [pytest.approx(0.50)] * 3  # stuck inside the deadband
    assert seen[3] == pytest.approx(0.58)  # breaks away, short of 0.62 by S-J
    assert seen[4] == pytest.approx(0.65)  # then follows
    assert seen[5] == pytest.approx(0.65)  # reversal: sticks again
    assert seen[7] == pytest.approx(0.54)  # breaks the other way


def test_stiction_is_inert_at_zero_strength():
    f = ValveStiction("s", deadband=0.10, slip=0.06)
    for c in (0.2, 0.5, 0.9):
        assert f.command(Inputs(c, True), 0.0).valve_cmd == pytest.approx(c)


def test_slip_cannot_exceed_the_deadband():
    with pytest.raises(ValueError, match="slip jump"):
        ValveStiction("s", deadband=0.05, slip=0.10)


def test_stiction_produces_a_limit_cycle():
    # A sticking valve must make a lined-out loop oscillate. If it only shifts
    # the operating point, it is not stiction.
    healthy = quick("healthy", [], 3600.0)
    stuck = quick("stiction", [ValveStiction("s", t_start=1200.0,
                                             deadband=0.10, slip=0.07)], 3600.0)
    seg_h, seg_s = after(healthy, 2000.0), after(stuck, 2000.0)

    healthy_swing = np.ptp(healthy.trajectory["LT101"][seg_h])
    stuck_swing = np.ptp(stuck.trajectory["LT101"][seg_s])
    assert stuck_swing > 5 * healthy_swing
    assert stuck_swing > 15.0  # mm, visible on any trend

    # The signature that identifies it: command and stem disagree.
    gap = np.abs(stuck.trajectory["LIC101_OP"][seg_s]
                 - stuck.trajectory["ZT101"][seg_s]).mean()
    assert gap > 1.0


# ============================ sensor faults ================================

def test_sensor_bias_offsets_only_its_own_tag():
    fs = FaultSet([SensorBias("b", tag="LT101", bias=25.0)])
    out = fs.measure({"LT101": 300.0, "LT102": 176.0}, 0.0)
    assert out["LT101"] == pytest.approx(325.0)
    assert out["LT102"] == pytest.approx(176.0)


def test_sensor_drift_accumulates():
    fs = FaultSet([SensorDrift("d", t_start=0.0, tag="LT101", rate_per_min=6.0)])
    assert fs.measure({"LT101": 300.0}, 60.0)["LT101"] == pytest.approx(306.0)
    assert fs.measure({"LT101": 300.0}, 600.0)["LT101"] == pytest.approx(360.0)


def test_sensor_stuck_freezes_at_onset():
    fs = FaultSet([SensorStuck("s", t_start=10.0, tag="LT101")])
    assert fs.measure({"LT101": 300.0}, 5.0)["LT101"] == pytest.approx(300.0)
    assert fs.measure({"LT101": 305.0}, 10.0)["LT101"] == pytest.approx(305.0)
    assert fs.measure({"LT101": 400.0}, 50.0)["LT101"] == pytest.approx(305.0)


def test_corrupted_readings_are_clamped_to_the_instrument_span():
    fs = FaultSet([SensorBias("b", tag="LT101", bias=5000.0)])
    assert fs.measure({"LT101": 300.0}, 0.0)["LT101"] == 600.0


def test_sensor_noise_is_reproducible():
    def sample(seed):
        fs = FaultSet([SensorNoise("n", tag="LT101", sigma=1.0)], seed=seed)
        return [fs.measure({"LT101": 300.0}, 0.0)["LT101"] for _ in range(5)]

    assert sample(7) == sample(7)
    assert sample(7) != sample(8)


def test_booleans_are_never_corrupted():
    fs = FaultSet([SensorBias("b", tag="PMP101_RUN", bias=1.0)])
    assert fs.measure({"PMP101_RUN": True}, 0.0)["PMP101_RUN"] is True


def test_unknown_tags_are_refused():
    with pytest.raises(ValueError, match="unknown tag"):
        SensorBias("b", tag="NOPE")


# ============================ the fault set ================================

def test_duplicate_ids_are_refused():
    with pytest.raises(ValueError, match="duplicate"):
        FaultSet([Leak("same"), PumpWear("same")])
    fs = FaultSet([Leak("a")])
    with pytest.raises(ValueError, match="already present"):
        fs.add(Leak("a"))


def test_add_remove_and_clear():
    fs = FaultSet()
    fs.add(Leak("a"))
    assert fs.remove("a") is True
    assert fs.remove("a") is False
    fs.add(Leak("b"))
    fs.clear()
    assert fs.faults == []


def test_labels_ignore_benign_faults():
    fs = FaultSet([SensorNoise("_noise", tag="LT101", sigma=1.0, benign=True),
                   Leak("leak", t_start=100.0)])
    assert fs.labels_at(50.0) == "normal"
    assert fs.labels_at(150.0) == "leak"
    assert [e["fault_id"] for e in fs.events()] == ["leak"]


def test_simultaneous_faults_produce_a_compound_label():
    fs = FaultSet([Leak("leak", t_start=0.0), PumpWear("wear", t_start=0.0)])
    assert fs.labels_at(10.0) == "leak+wear"


# ============================== scenarios ==================================

def test_scenario_validation():
    with pytest.raises(ValueError, match="duration"):
        Scenario("x", 0.0)
    with pytest.raises(ValueError, match="start at t=0"):
        Scenario("x", 100.0, setpoints=[(5.0, 300.0)])
    with pytest.raises(ValueError, match="after the"):
        Scenario("x", 100.0, faults=[Leak("l", t_start=200.0)])


def test_a_normal_run_stays_labelled_normal_and_near_setpoint():
    r = quick("normal", [])
    assert (r.labels["label"] == "normal").all()
    assert not r.labels["faulty"].any()
    assert r.trajectory["LT101"].mean() == pytest.approx(300.0, abs=2.0)
    assert (r.trajectory["ILK_PMP101"] == 0.0).all()


def test_labels_switch_at_the_declared_onset():
    r = quick("leak", [Leak("leak", t_start=1200.0, cda=2e-5)])
    t = r.trajectory.t
    assert not r.labels["faulty"][t < 1200.0].any()
    assert r.labels["faulty"][t >= 1200.0].all()
    assert set(r.labels["label"]) == {"normal", "leak"}


def test_events_carry_ground_truth():
    r = quick("leak", [Leak("leak", t_start=1200.0, cda=2e-5, ramp=300.0)])
    row = r.events.iloc[0]
    assert row["fault_id"] == "leak" and row["kind"] == "leak"
    assert row["t_start"] == 1200.0 and row["ramp"] == 300.0
    assert row["magnitude"] == pytest.approx(2e-5)
    assert row["scenario"] == "leak"


def test_normal_scenarios_still_emit_an_events_row():
    r = quick("normal", [])
    assert len(r.events) == 1 and r.events.iloc[0]["kind"] == "normal"


def test_frame_joins_signals_and_labels():
    r = quick("leak", [Leak("leak", t_start=1200.0)])
    df = r.frame()
    assert {"LT101", "LIC101_OP", "label", "faulty", "scenario"} <= set(df.columns)
    assert len(df) == len(r.trajectory)


def test_faults_are_not_mutated_by_running_a_scenario():
    # Stiction carries a stem position; a scenario re-run must start clean.
    fault = ValveStiction("s", t_start=600.0, deadband=0.08, slip=0.05)
    sc = Scenario("stic", 1800.0, [fault], [(0.0, 300.0)], seed=1)
    a = run_scenario(sc).trajectory["LT101"]
    b = run_scenario(sc).trajectory["LT101"]
    assert np.allclose(a, b)
    assert fault._primed is False  # the library object was never used


def test_ground_truth_and_reported_readings_diverge_under_a_sensor_fault():
    sim = Simulation(dt=0.05, u=DESIGN,
                     faults=FaultSet([SensorBias("b", tag="LT101", bias=30.0)]))
    for _ in range(200):
        sim.integrate()
    assert sim.measurements["LT101"] - sim.true_measurements["LT101"] \
        == pytest.approx(30.0, abs=0.1)


def test_a_stuck_transmitter_walks_the_loop_into_a_trip():
    # One frozen instrument, and the loop destroys itself: the controller
    # chases a reading that cannot move, shuts the valve, starves the pump.
    r = run_scenario([s for s in library(2400.0) if s.name == "sensor_stuck"][0])
    tripped = r.trajectory["ILK_PMP101"]
    assert tripped.max() == 1.0
    trip_time = r.trajectory.t[int(tripped.argmax())]
    assert trip_time > 960.0  # after the fault, not a startup artefact


@pytest.mark.parametrize("name", [s.name for s in library()])
def test_every_library_scenario_runs_and_is_labelled(name):
    scenario = [s for s in library(1200.0) if s.name == name][0]
    r = run_scenario(scenario, log_period=2.0)
    assert len(r.trajectory) == len(r.labels)
    assert r.labels["faulty"].any() == (not scenario.is_normal)
    assert not np.isnan(r.trajectory["LT101"]).any()


def test_grid_varies_magnitude_onset_and_seed():
    g = grid(duration=600.0, magnitudes=3, repeats=2)
    assert len({s.name for s in g}) == len(g)
    assert len({s.seed for s in g}) == len(g)
    onsets = {f.t_start for s in g for f in s.faults}
    assert len(onsets) > 3  # onset is not a constant a model could memorise
    leaks = [f for s in g for f in s.faults if isinstance(f, Leak)]
    assert len({round(f.cda, 12) for f in leaks}) == 3


def test_grid_validates_its_arguments():
    with pytest.raises(ValueError, match="at least 1"):
        grid(magnitudes=0)


# =========================== live injection ================================

@contextlib.asynccontextmanager
async def plant_api():
    # Lined out and open loop: no PLC, so the valve holds 66.2% and a fault
    # moves the equilibrium instead of being masked by a controller.
    from pyplant.core.hybrid_tank import steady_state

    sim = Simulation(dt=0.05, x=steady_state(DESIGN, Params()), u=DESIGN)
    node = PlantNode(sim=sim, port=0)
    await node.start()
    transport = httpx.ASGITransport(app=create_control_app(node))
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://plant") as client:
        try:
            yield client, node
        finally:
            await node.stop()


def test_build_fault_covers_every_declared_kind():
    for kind, (_, magnitude_field) in FAULT_KINDS.items():
        fault = build_fault(kind, f"f_{kind}",
                            magnitude=0.1 if magnitude_field else None)
        assert fault.kind == kind or kind == "stiction"


def test_build_fault_rejects_nonsense():
    with pytest.raises(ValueError, match="unknown fault kind"):
        build_fault("wishful", "x")
    with pytest.raises(ValueError, match="takes no tag"):
        build_fault("leak", "x", tag="LT101")
    with pytest.raises(ValueError, match="takes no magnitude"):
        build_fault("sensor_stuck", "x", magnitude=1.0)


def test_injection_endpoint_changes_the_plant():
    async def go():
        async with plant_api() as (client, node):
            node.tick()
            before = node.store.get("LT101")
            posted = await client.post("/api/fault",
                                       json={"kind": "leak", "id": "L1",
                                             "magnitude": 6e-5})
            listed = (await client.get("/api/faults")).json()
            for _ in range(4000):  # 200 s of plant time
                node.tick()
            after_level = node.store.get("LT101")
            truth = (await client.get("/api/truth")).json()
            return posted.json(), listed, before, after_level, truth

    posted, listed, before, after_level, truth = run(go())
    assert posted["fault_id"] == "L1" and posted["kind"] == "leak"
    assert listed["active"] == ["L1"]
    assert after_level < before - 5.0  # the tank is draining
    assert truth["label"] == "L1"


def test_injected_sensor_fault_is_invisible_in_ground_truth():
    async def go():
        async with plant_api() as (client, node):
            node.tick()
            await client.post("/api/fault", json={"kind": "sensor_bias",
                                                  "id": "B1", "magnitude": 30.0,
                                                  "tag": "LT101"})
            node.tick()
            return (await client.get("/api/truth")).json()

    truth = run(go())
    assert truth["reported"]["LT101"] - truth["true"]["LT101"] \
        == pytest.approx(30.0, abs=0.2)


def test_faults_can_be_removed():
    async def go():
        async with plant_api() as (client, node):
            await client.post("/api/fault", json={"kind": "leak", "id": "L1"})
            gone = await client.delete("/api/fault/L1")
            missing = await client.delete("/api/fault/L1")
            await client.post("/api/fault", json={"kind": "leak", "id": "L2"})
            cleared = await client.delete("/api/faults")
            return gone, missing, cleared.json(), node

    gone, missing, cleared, node = run(go())
    assert gone.status_code == 200
    assert missing.status_code == 404
    assert cleared["removed"] == ["L2"]
    assert node.faults.faults == []


def test_bad_injection_requests_are_rejected():
    async def go():
        async with plant_api() as (client, node):
            return (await client.post("/api/fault", json={"kind": "nope"}),
                    await client.post("/api/fault",
                                      json={"kind": "leak", "tag": "LT101"}))

    unknown, wrong_tag = run(go())
    assert unknown.status_code == 422 and wrong_tag.status_code == 422
