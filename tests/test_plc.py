"""Tests for the controller and the PLC program.

The closed-loop section at the bottom is the point of the whole project. A
leak that the controller compensates leaves the process value exactly on
setpoint and shows up only in the controller output. If the diagnosis model is
fed PV alone it cannot see that fault at all, and
`test_controller_hides_a_leak_in_pv_but_not_in_op` is what pins that down.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pyplant.core.hybrid_tank import Inputs, Params
from pyplant.plc.harness import OperatorPanel, PlcHarness, scheduled_setpoint
from pyplant.plc.pid import PID, PidTuning
from pyplant.plc.program import (
    FT101_LO,
    LT101_HI,
    LT101_HIHI,
    MIN_FLOW_DELAY,
    Hysteresis,
    OnDelay,
    PlcInput,
    PlcOutput,
    Program,
)
from pyplant.runners.base import Simulation
from pyplant.runners.fast import run
from pyplant.tags import TAGS

TS = 0.10
DESIGN = Inputs(0.662, True)


def running_harness(**panel) -> PlcHarness:
    kw = dict(setpoint=300.0, auto=True, pump_start=True)
    kw.update(panel)
    return PlcHarness(panel=OperatorPanel(**kw))


def lined_out(harness: PlcHarness, seconds: float = 900.0) -> Simulation:
    sim = Simulation.at_steady_state(DESIGN, policy=harness)
    run(sim, seconds, 1.0, extra=harness.tags)
    return sim


# ============================ PID ==========================================

def test_tuning_validation():
    with pytest.raises(ValueError):
        PidTuning(kc=-1.0)
    with pytest.raises(ValueError):
        PidTuning(tf=0.0)
    with pytest.raises(ValueError):
        PidTuning(out_lo=50.0, out_hi=50.0)
    with pytest.raises(ValueError):
        PID(ts=0.0)


def test_reverse_acting_output_falls_as_pv_rises():
    pid = PID(ts=TS)
    pid.track(50.0, 100.0, 100.0)
    out = pid.step(sp=100.0, pv=110.0, auto=True, manual_out=0.0)
    assert out < 50.0


def test_direct_acting_flips_the_sign():
    pid = PID(tuning=PidTuning(direct_acting=True), ts=TS)
    pid.track(50.0, 100.0, 100.0)
    assert pid.step(100.0, 110.0, True, 0.0) > 50.0


def test_integral_action_ramps_on_a_standing_error():
    pid = PID(tuning=PidTuning(kc=0.35, ti=72.0), ts=TS)
    pid.track(50.0, 100.0, 100.0)
    outs = [pid.step(110.0, 100.0, True, 0.0) for _ in range(200)]
    assert all(b > a for a, b in zip(outs, outs[1:]))  # monotone climb
    # No proportional term: the first scan is a manual-to-auto transfer, which
    # is deliberately bumpless, and the error is constant after that.
    assert outs[-1] == pytest.approx(50.0 + 0.35 * TS / 72.0 * 10.0 * 200, rel=1e-9)


def test_proportional_kick_applies_once_already_in_auto():
    pid = PID(tuning=PidTuning(kc=0.35, ti=72.0), ts=TS)
    pid.track(50.0, 100.0, 100.0)
    pid.step(100.0, 100.0, True, 0.0)  # transfer scan, zero error
    out = pid.step(110.0, 100.0, True, 0.0)
    assert out == pytest.approx(50.0 + 0.35 * 10.0 + 0.35 * TS / 72.0 * 10.0,
                                rel=1e-9)


def test_output_clamps_at_the_limits():
    pid = PID(tuning=PidTuning(kc=5.0, ti=10.0), ts=TS)
    pid.track(50.0, 100.0, 100.0)
    for _ in range(2000):
        pid.step(600.0, 100.0, True, 0.0)
    assert pid.out == pytest.approx(100.0)
    assert pid.saturated


def test_velocity_form_does_not_wind_up():
    # Sit hard against the upper limit, then reverse the error. A positional
    # controller with a naive integrator would stay pinned for many scans.
    pid = PID(tuning=PidTuning(kc=0.35, ti=72.0), ts=TS)
    pid.track(50.0, 100.0, 100.0)
    for _ in range(5000):
        pid.step(600.0, 100.0, True, 0.0)
    assert pid.out == pytest.approx(100.0)
    first = pid.step(0.0, 100.0, True, 0.0)
    assert first < 100.0  # leaves the clamp on the very next scan


def test_manual_passes_through_and_transfer_is_bumpless():
    pid = PID(ts=TS)
    for _ in range(10):
        pid.step(300.0, 250.0, auto=False, manual_out=40.0)
    assert pid.out == pytest.approx(40.0)
    assert pid.auto is False

    out = pid.step(300.0, 250.0, auto=True, manual_out=40.0)
    increment = 0.35 * TS / 72.0 * 50.0  # integral only; error is unchanged
    assert out == pytest.approx(40.0 + increment, rel=1e-9)


def test_manual_output_is_clamped():
    pid = PID(ts=TS)
    pid.step(300.0, 250.0, auto=False, manual_out=500.0)
    assert pid.out == pytest.approx(100.0)


def test_derivative_acts_on_measurement_not_error():
    # A setpoint step must produce the same increment with and without D.
    def after_sp_step(td: float) -> float:
        pid = PID(tuning=PidTuning(td=td), ts=TS)
        pid.track(50.0, 300.0, 300.0)
        return pid.step(350.0, 300.0, True, 0.0)

    assert after_sp_step(20.0) == pytest.approx(after_sp_step(0.0), rel=1e-12)


def test_derivative_responds_to_a_moving_measurement():
    pid = PID(tuning=PidTuning(td=20.0), ts=TS)
    pid.track(50.0, 300.0, 300.0)
    plain = PID(tuning=PidTuning(td=0.0), ts=TS)
    plain.track(50.0, 300.0, 300.0)
    for pv in (300.0, 302.0, 306.0, 312.0):  # accelerating rise
        d_out = pid.step(300.0, pv, True, 0.0)
        p_out = plain.step(300.0, pv, True, 0.0)
    assert d_out < p_out  # derivative pushes back harder on the rise


# ======================= program building blocks ===========================

def test_hysteresis_high_needs_a_real_retreat_to_clear():
    h = Hysteresis(limit=500.0, deadband=10.0, high=True)
    assert h(499.0) is False
    assert h(500.0) is True
    assert h(495.0) is True  # inside the deadband, stays alarmed
    assert h(489.0) is False


def test_hysteresis_low():
    h = Hysteresis(limit=50.0, deadband=10.0, high=False)
    assert h(51.0) is False
    assert h(50.0) is True
    assert h(55.0) is True
    assert h(61.0) is False


def test_on_delay_needs_the_full_preset_and_resets_on_a_gap():
    t = OnDelay(preset=1.0, ts=0.1)
    for _ in range(9):
        assert t(True) is False
    assert t(True) is True
    assert t(False) is False
    assert t(True) is False  # timer restarted from zero


# ============================ program ======================================

def test_image_tables_only_reference_real_tags():
    assert set(vars(PlcInput())) <= set(TAGS)
    out = Program().execute(PlcInput(ZT101=66.2, LT101=300.0))
    assert set(out.as_tags()) <= set(TAGS)


def test_first_scan_seeds_the_output_from_valve_feedback():
    # Without this the velocity-form controller would step from 0% and slam
    # the valve shut on a plant that is already running.
    out = Program(ts=TS).execute(PlcInput(LT101=300.0, ZT101=66.2, LIC101_SP=300.0,
                                          CMD_PMP101_START=True))
    assert out.FCV101_OP == pytest.approx(66.2, abs=0.05)


def test_setpoint_is_clamped_to_the_tag_span():
    out = Program().execute(PlcInput(LT101=300.0, ZT101=66.0, LIC101_SP=9_999.0))
    assert out.LIC101_SP == TAGS["LIC101_SP"].hi


def test_level_alarms_follow_the_tag_limits():
    p = Program(ts=TS)
    assert p.execute(PlcInput(LT101=LT101_HI + 1, ZT101=50.0)).ALM_LT101_HI
    assert p.execute(PlcInput(LT101=30.0, ZT101=50.0)).ALM_LT101_LOLO


def test_level_hihi_latches_the_interlock():
    p = Program(ts=TS)
    p.execute(PlcInput(LT101=300.0, ZT101=66.0, CMD_PMP101_START=True))
    out = p.execute(PlcInput(LT101=LT101_HIHI + 5, ZT101=66.0, CMD_PMP101_START=True))
    assert out.ILK_PMP101 and out.PMP101_CMD is False and out.FCV101_OP == 0.0
    # Stays latched once the level has fallen right back.
    out = p.execute(PlcInput(LT101=300.0, ZT101=0.0, CMD_PMP101_START=True))
    assert out.ILK_PMP101


def test_reset_is_refused_while_the_condition_persists():
    p = Program(ts=TS)
    p.execute(PlcInput(LT101=LT101_HIHI + 5, ZT101=66.0))
    out = p.execute(PlcInput(LT101=LT101_HIHI + 5, ZT101=66.0, CMD_ALM_RESET=True))
    assert out.ILK_PMP101


def test_reset_clears_once_the_condition_is_gone():
    p = Program(ts=TS)
    p.execute(PlcInput(LT101=LT101_HIHI + 5, ZT101=66.0))
    p.execute(PlcInput(LT101=300.0, ZT101=0.0, CMD_ALM_RESET=True))
    out = p.execute(PlcInput(LT101=300.0, ZT101=0.0, CMD_PMP101_START=True))
    assert out.ILK_PMP101 is False and out.PMP101_CMD is True


def test_auto_is_forced_off_while_interlocked():
    p = Program(ts=TS)
    out = p.execute(PlcInput(LT101=LT101_HIHI + 5, ZT101=66.0, LIC101_AUTO=True))
    assert out.LIC101_AUTO is False


def test_minimum_flow_trip_waits_out_its_delay():
    p = Program(ts=TS)
    scans = int(MIN_FLOW_DELAY / TS)
    starved = PlcInput(LT101=300.0, ZT101=0.0, FT101=FT101_LO - 0.5,
                       CMD_PMP101_START=True)
    for _ in range(scans - 1):
        assert p.execute(starved).ILK_PMP101 is False
    assert p.execute(starved).ILK_PMP101 is True


def test_minimum_flow_trip_ignores_a_stopped_pump():
    p = Program(ts=TS)
    idle = PlcInput(LT101=300.0, ZT101=0.0, FT101=0.0, CMD_PMP101_START=False)
    for _ in range(int(MIN_FLOW_DELAY / TS) * 3):
        assert p.execute(idle).ILK_PMP101 is False


def test_manual_mode_passes_the_operator_demand_through():
    p = Program(ts=TS)
    p.execute(PlcInput(LT101=300.0, ZT101=66.0))
    out = p.execute(PlcInput(LT101=300.0, ZT101=66.0, LIC101_AUTO=False,
                             LIC101_MAN_OP=42.0, CMD_PMP101_START=True))
    assert out.FCV101_OP == pytest.approx(42.0)


def test_scan_counter_and_first_scan_flag():
    p = Program(ts=TS)
    assert p.first_scan is True
    p.execute(PlcInput(LT101=300.0, ZT101=66.0))
    assert p.first_scan is False and p.scans == 1


# ========================== closed loop ====================================

def test_setpoint_step_has_no_offset_and_no_overshoot():
    h = running_harness()
    sim = lined_out(h)
    h.panel.setpoint = 350.0
    tr = run(sim, 900.0, 1.0, extra=h.tags)
    assert tr["LT101"][-1] == pytest.approx(350.0, abs=0.5)
    assert tr["LT101"].max() <= 350.5  # no overshoot


def test_setpoint_step_settles_well_inside_the_open_loop_time_constant():
    h = running_harness()
    sim = lined_out(h)
    h.panel.setpoint = 350.0
    tr = run(sim, 900.0, 1.0, extra=h.tags)
    outside = np.where(np.abs(tr["LT101"] - 350.0) > 1.0)[0]
    settling = tr.t[outside[-1]] - tr.t[0]
    assert settling < 72.0  # open-loop tau


def test_controller_hides_a_leak_in_pv_but_not_in_op():
    h = running_harness()
    sim = lined_out(h)
    op_before = h.last_output.LIC101_OP

    sim.params = replace(sim.params, cda_leak=2.0e-5)
    tr = run(sim, 1800.0, 1.0, extra=h.tags)

    assert tr["LT101"][-1] == pytest.approx(300.0, abs=0.1)  # PV: nothing to see
    assert tr["LIC101_OP"][-1] - op_before > 5.0  # OP: the whole signature


def test_fouling_and_pump_wear_move_the_output_in_opposite_directions():
    def op_shift(**fault) -> float:
        h = running_harness()
        sim = lined_out(h)
        before = h.last_output.LIC101_OP
        sim.params = replace(sim.params, **fault)
        tr = run(sim, 1800.0, 1.0, extra=h.tags)
        return tr["LIC101_OP"][-1] - before

    assert op_shift(cda_12_gain=0.75) < -2.0  # fouling backs the valve off
    assert op_shift(pump_gain=0.85) > 2.0  # a tired pump needs more valve


def test_loop_starts_up_from_an_empty_plant_without_tripping():
    h = running_harness()
    sim = Simulation(policy=h)  # empty tanks, valve shut
    tr = run(sim, 2400.0, 1.0, extra=h.tags)
    assert tr["ILK_PMP101"][-1] == 0.0
    assert tr["LT101"][-1] == pytest.approx(300.0, abs=1.0)


def test_scheduled_setpoint_drives_the_panel():
    h = running_harness()
    sim = Simulation.at_steady_state(
        DESIGN, policy=scheduled_setpoint(h, [(0.0, 300.0), (600.0, 250.0)]))
    tr = run(sim, 1800.0, 1.0, extra=h.tags)
    assert tr["LIC101_SP"][0] == pytest.approx(300.0)
    assert tr["LIC101_SP"][-1] == pytest.approx(250.0)
    assert tr["LT101"][-1] == pytest.approx(250.0, abs=0.5)


def test_scheduled_setpoint_rejects_unsorted_times():
    with pytest.raises(ValueError, match="ascend"):
        scheduled_setpoint(running_harness(), [(0.0, 300.0), (0.0, 250.0)])


def test_harness_reports_stable_columns_before_the_first_scan():
    # A logger fixes its column set from the first sample, so the key set must
    # not change on scan one.
    blank = running_harness().tags()
    h = running_harness()
    h(0.0, {k: 0.0 for k in ("LT101", "LT102", "FT101", "FT102", "PT101",
                             "ZT101", "PMP101_RUN")})
    assert set(blank) == set(h.tags())
    assert blank["ILK_PMP101"] is False
