"""Tests for the plant physics.

Two properties matter more than the rest. Mass has to be conserved, because a
slow leak in the integrator is indistinguishable from the leak fault the model
is later supposed to detect. And the answer has to be independent of the
integration step, because the real-time runner and the dataset generator will
use different ones.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyplant.core.hybrid_tank import (
    G,
    H1,
    H2,
    RHO,
    Z,
    Inputs,
    Params,
    clip_to_tag_ranges,
    derivatives,
    flows,
    initial_state,
    measure,
    step,
    steady_state,
    valve_characteristic,
)
from pyplant.tags import TAGS

P = Params()
DT = 0.05
RUN = Inputs(valve_cmd=0.662, pump_on=True)  # design point, ~300 mm in tank 1


def simulate(x0, u, p=P, dt=DT, seconds=600.0):
    x = np.array(x0, dtype=float)
    n = int(seconds / dt)
    for _ in range(n):
        x = step(x, u, p, dt)
    return x


# --- valve ------------------------------------------------------------------

def test_valve_is_monotonic_and_bounded():
    vals = [valve_characteristic(z, P) for z in np.linspace(0.0, 1.0, 101)]
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert valve_characteristic(0.0, P) == 0.0
    assert valve_characteristic(1.0, P) == pytest.approx(1.0)


def test_valve_actuator_is_first_order():
    x = initial_state(z=0.0)
    u = Inputs(valve_cmd=1.0, pump_on=False)
    for _ in range(int(P.tau_valve / DT)):
        x = step(x, u, P, DT)
    assert x[Z] == pytest.approx(1.0 - math.exp(-1.0), abs=2e-3)


# --- pump/valve algebraic loop ---------------------------------------------

def test_pump_deadheads_against_a_shut_valve():
    f = flows(np.array([0.3, 0.0, 0.0]), Inputs(0.0, True), P)
    assert f.q_in == 0.0
    assert f.p_pump == pytest.approx(P.pump_dp0)


def test_pump_off_gives_no_flow_and_no_pressure():
    f = flows(np.array([0.3, 0.1, 1.0]), Inputs(1.0, False), P)
    assert f.q_in == 0.0 and f.p_pump == 0.0


def test_pump_curve_is_consistent_with_valve_equation():
    # The closed-form solution must satisfy both original equations.
    x = np.array([0.25, 0.1, 0.7])
    f = flows(x, Inputs(0.7, True), P)
    k = P.valve_cv * valve_characteristic(0.7, P)
    assert f.p_pump == pytest.approx(P.pump_dp0 * (1 - (f.q_in / P.pump_qmax) ** 2))
    assert f.q_in == pytest.approx(k * math.sqrt(f.p_pump - RHO * G * x[H1]))


def test_higher_level_reduces_inflow():
    lo = flows(np.array([0.05, 0.0, 0.7]), Inputs(0.7, True), P).q_in
    hi = flows(np.array([0.55, 0.0, 0.7]), Inputs(0.7, True), P).q_in
    assert hi < lo


# --- conservation and integration ------------------------------------------

def test_mass_is_conserved_in_tank_1():
    x = initial_state()
    accumulated = 0.0
    for _ in range(int(300.0 / DT)):
        f = flows(x, RUN, P)  # left endpoint of the interval
        nxt = step(x, RUN, P, DT)
        accumulated += (f.q_in - f.q_12 - f.q_leak) * DT
        x = nxt
    volume_change = (x[H1] - 0.0) * P.area_tank1
    assert volume_change == pytest.approx(accumulated, rel=2e-3)


def test_result_is_independent_of_step_size():
    coarse = simulate(initial_state(), RUN, dt=0.05, seconds=300.0)
    fine = simulate(initial_state(), RUN, dt=0.005, seconds=300.0)
    assert coarse[H1] == pytest.approx(fine[H1], abs=1e-5)
    assert coarse[H2] == pytest.approx(fine[H2], abs=1e-5)


def test_zero_or_negative_dt_is_rejected():
    with pytest.raises(ValueError):
        step(initial_state(), RUN, P, 0.0)


# --- steady state -----------------------------------------------------------

def test_simulation_converges_to_analytic_steady_state():
    predicted = steady_state(RUN, P)
    reached = simulate(initial_state(), RUN, seconds=1800.0)
    assert reached[H1] == pytest.approx(predicted[H1], abs=1e-4)
    assert reached[H2] == pytest.approx(predicted[H2], abs=1e-4)


def test_steady_state_obeys_torricelli_between_the_tanks():
    xs = steady_state(RUN, P)
    q12 = P.cda_12 * math.sqrt(2 * G * xs[H1])
    q_out = P.cda_2out * math.sqrt(2 * G * xs[H2])
    assert q12 == pytest.approx(q_out, rel=1e-6)


def test_design_point_sits_where_the_rig_was_sized():
    # Anchors the parameter set. If someone retunes the pump or the orifices,
    # this is the test that says the operating window moved.
    xs = steady_state(RUN, P)
    assert xs[H1] * 1000 == pytest.approx(300.0, abs=5.0)
    assert flows(xs, RUN, P).q_in * 60_000 == pytest.approx(10.0, abs=0.5)


def test_steady_level_rises_with_valve_demand():
    levels = [steady_state(Inputs(c, True), P)[H1] for c in (0.4, 0.5, 0.6, 0.7)]
    assert all(b > a for a, b in zip(levels, levels[1:]))


# --- limits -----------------------------------------------------------------

def test_tanks_never_go_negative():
    x = simulate(initial_state(0.4, 0.4, 0.0), Inputs(0.0, False), seconds=3600.0)
    assert x[H1] >= 0.0 and x[H2] >= 0.0
    assert x[H1] == pytest.approx(0.0, abs=1e-6)


def test_tanks_never_exceed_the_overflow_lip():
    x = simulate(initial_state(0.59, 0.59, 1.0), Inputs(1.0, True), seconds=1200.0)
    assert x[H1] <= P.height_max + 1e-12
    assert x[H2] <= P.height_max + 1e-12


def test_derivatives_are_clamped_at_the_lip():
    d = derivatives(np.array([P.height_max, 0.0, 1.0]), Inputs(1.0, True), P)
    assert d[H1] == 0.0


# --- fault knobs ------------------------------------------------------------

def test_leak_lowers_the_steady_level():
    leaking = Params(cda_leak=2.0e-5)
    assert steady_state(RUN, leaking)[H1] < steady_state(RUN, P)[H1]


def test_pump_wear_lowers_the_steady_level():
    worn = Params(pump_gain=0.85)
    assert steady_state(RUN, worn)[H1] < steady_state(RUN, P)[H1]


def test_fouling_raises_tank_1_and_lowers_tank_2():
    fouled = Params(cda_12_gain=0.7)
    base, bad = steady_state(RUN, P), steady_state(RUN, fouled)
    assert bad[H1] > base[H1]
    assert bad[H2] < base[H2]


def test_healthy_resets_every_fault_knob():
    sick = Params(pump_gain=0.5, cda_12_gain=0.5, cda_leak=1e-5)
    assert sick.healthy() == Params()


# --- boundary to the tag layer ---------------------------------------------

def test_measurements_cover_every_plant_owned_tag():
    from pyplant.tags import Owner, by_owner

    assert set(measure(steady_state(RUN, P), RUN, P)) == {
        t.name for t in by_owner(Owner.PLANT)
    }


def test_measurements_stay_inside_declared_instrument_spans():
    # Sweep the operating envelope, not just the design point.
    for cmd in np.linspace(0.0, 1.0, 21):
        for pump in (True, False):
            u = Inputs(float(cmd), pump)
            for x in (initial_state(), steady_state(u, P),
                      initial_state(P.height_max, P.height_max, 1.0)):
                for name, value in measure(x, u, P).items():
                    t = TAGS[name]
                    if t.kind.name == "BOOL":
                        continue
                    assert t.lo - 1e-9 <= value <= t.hi + 1e-9, (name, value, cmd)


def test_measurements_survive_the_wire_codec():
    for name, value in measure(steady_state(RUN, P), RUN, P).items():
        t = TAGS[name]
        back = t.decode(t.encode(value))
        if t.kind.name == "BOOL":
            assert back == value
        else:
            assert abs(back - value) <= t.resolution / 2 + 1e-9, name


def test_clip_to_tag_ranges_pegs_instead_of_wrapping():
    m = {"LT101": 9_999.0, "PMP101_RUN": True}
    assert clip_to_tag_ranges(m)["LT101"] == TAGS["LT101"].hi
    assert clip_to_tag_ranges(m)["PMP101_RUN"] is True
