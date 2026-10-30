"""Tests for the stepper and the fast runner.

The load-bearing test here is `test_runner_matches_hand_rolled_integration`.
Everything downstream trusts that driving the plant through `Simulation` is
identical to calling `step()` in a loop; if the runner ever quietly changes the
physics, the datasets and the live plant stop being the same system.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyplant.core.hybrid_tank import H1, Inputs, Params, initial_state, measure, step, steady_state
from pyplant.runners.base import Simulation, constant, piecewise
from pyplant.runners.fast import INPUT_COLUMNS, STATE_COLUMNS, run, settle

P = Params()
U = Inputs(valve_cmd=0.662, pump_on=True)


# --- construction guards ----------------------------------------------------

def test_control_period_must_be_a_whole_number_of_dt():
    with pytest.raises(ValueError, match="whole number"):
        Simulation(dt=0.05, control_period=0.075)


def test_control_period_cannot_be_shorter_than_dt():
    with pytest.raises(ValueError, match="shorter than dt"):
        Simulation(dt=0.05, control_period=0.01)


def test_bad_dt_is_rejected():
    with pytest.raises(ValueError):
        Simulation(dt=0.0)


def test_initial_state_is_copied_not_aliased():
    x0 = initial_state(0.1, 0.1, 0.5)
    sim = Simulation(x=x0, policy=constant(U))
    sim.advance()
    assert x0[H1] == 0.1


# --- zero-order hold --------------------------------------------------------

def test_policy_runs_only_on_scan_boundaries():
    calls = []

    def counting(t, m):
        calls.append(t)
        return U

    sim = Simulation(policy=counting, dt=0.05, control_period=0.10)
    for _ in range(20):  # 1.0 s
        sim.advance()
    assert len(calls) == 10
    assert calls == pytest.approx(np.arange(10) * 0.10)


def test_input_is_held_between_scans():
    # A policy that would ramp continuously must appear as a staircase.
    sim = Simulation(policy=lambda t, m: Inputs(min(t / 100.0, 1.0), True),
                     dt=0.05, control_period=0.50)
    seen = []
    for _ in range(40):  # 2.0 s
        sim.advance()
        seen.append(sim.u.valve_cmd)
    assert len(set(seen)) == 4  # one distinct value per 0.5 s scan


def test_time_does_not_drift_over_a_long_run():
    sim = Simulation(policy=constant(U), dt=0.05)
    for _ in range(100_000):
        sim.advance()
    assert sim.t == pytest.approx(5000.0, abs=1e-9)


# --- policies ---------------------------------------------------------------

def test_constant_policy():
    assert constant(U)(123.0, {}) == U


def test_piecewise_selects_the_active_segment():
    a, b, c = Inputs(0.2, True), Inputs(0.6, True), Inputs(0.0, False)
    pol = piecewise([(0.0, a), (10.0, b), (20.0, c)])
    assert pol(0.0, {}) == a
    assert pol(9.99, {}) == a
    assert pol(10.0, {}) == b
    assert pol(19.9, {}) == b
    assert pol(999.0, {}) == c


@pytest.mark.parametrize("schedule, match", [
    ([], "empty"),
    ([(5.0, Inputs())], "start at t=0"),
    ([(0.0, Inputs()), (0.0, Inputs())], "ascend"),
    ([(0.0, Inputs()), (10.0, Inputs()), (5.0, Inputs())], "ascend"),
])
def test_piecewise_rejects_bad_schedules(schedule, match):
    with pytest.raises(ValueError, match=match):
        piecewise(schedule)


# --- equivalence with the physics core --------------------------------------

def test_runner_matches_hand_rolled_integration():
    sim = Simulation(policy=constant(U), dt=0.05, control_period=0.05)
    tr = run(sim, duration=300.0, log_period=1.0, log_states=True)

    x = initial_state()
    for _ in range(int(300.0 / 0.05)):
        x = step(x, U, P, 0.05)

    assert tr["h1_m"][-1] == pytest.approx(float(x[H1]), abs=1e-12)
    assert tr["LT101"][-1] == pytest.approx(measure(x, U, P)["LT101"], abs=1e-12)


# --- recording --------------------------------------------------------------

def test_sample_count_and_times():
    tr = run(Simulation(policy=constant(U)), duration=100.0, log_period=1.0)
    assert len(tr) == 101  # t=0 included
    assert tr.t[0] == 0.0 and tr.t[-1] == pytest.approx(100.0)
    assert np.allclose(np.diff(tr.t), 1.0)


def test_log_period_must_be_a_whole_number_of_dt():
    with pytest.raises(ValueError, match="whole number"):
        run(Simulation(policy=constant(U)), duration=10.0, log_period=0.07)


def test_zero_duration_is_rejected():
    with pytest.raises(ValueError, match="duration"):
        run(Simulation(policy=constant(U)), duration=0.0)


def test_columns_carry_inputs_as_well_as_measurements():
    tr = run(Simulation(policy=constant(U)), duration=10.0)
    assert set(INPUT_COLUMNS) <= set(tr.names)
    assert tr["FCV101_OP"][-1] == pytest.approx(66.2)  # percent, not fraction
    assert tr["PMP101_CMD"][-1] == 1.0


def test_states_are_excluded_unless_asked_for():
    lean = run(Simulation(policy=constant(U)), duration=10.0)
    full = run(Simulation(policy=constant(U)), duration=10.0, log_states=True)
    assert not set(STATE_COLUMNS) & set(lean.names)
    assert set(STATE_COLUMNS) <= set(full.names)


def test_trajectory_accessors():
    tr = run(Simulation(policy=constant(U)), duration=10.0)
    assert tr["LT101"] is tr.columns["LT101"]
    assert set(tr.final()) == set(tr.names)
    assert tr.meta["dt"] == 0.05 and tr.meta["duration"] == 10.0


def test_runs_are_deterministic():
    a = run(Simulation(policy=constant(U)), duration=200.0)
    b = run(Simulation(policy=constant(U)), duration=200.0)
    for name in a.names:
        assert np.array_equal(a[name], b[name]), name


def test_meta_records_the_parameter_set():
    tr = run(Simulation(params=Params(cda_leak=1e-5), policy=constant(U)),
             duration=10.0)
    assert tr.meta["params"]["cda_leak"] == 1e-5


# --- steady start and settling ---------------------------------------------

def test_at_steady_state_produces_no_transient():
    sim = Simulation.at_steady_state(U)
    tr = run(sim, duration=600.0)
    assert np.ptp(tr["LT101"]) < 1e-6


def test_settle_reaches_the_analytic_equilibrium():
    sim = Simulation(policy=constant(U))
    elapsed = settle(sim)
    assert elapsed > 0.0
    assert sim.x[H1] == pytest.approx(steady_state(U, P)[H1], abs=1e-4)


def test_settle_gives_up_loudly():
    with pytest.raises(RuntimeError, match="did not settle"):
        settle(Simulation(policy=constant(U)), max_seconds=120.0)


# --- throughput -------------------------------------------------------------

def test_fast_runner_is_much_faster_than_real_time():
    tr = run(Simulation.at_steady_state(U, policy=constant(U)), duration=600.0)
    # Loose bound: the point is that a scenario grid is practical, and a
    # regression to real time should fail rather than just feel slow.
    assert tr.meta["speedup"] > 50.0


# --- optional pandas bridge -------------------------------------------------

def test_to_dataframe_when_pandas_is_available():
    pd = pytest.importorskip("pandas")
    df = run(Simulation(policy=constant(U)), duration=10.0).to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "t" and "LT101" in df.columns
    assert len(df) == 11


def test_policy_is_evaluated_once_per_scan_by_the_runner():
    # The runner primes the policy before its first sample. That must not turn
    # into a double evaluation, which would advance a controller's integrator
    # twice on scan zero.
    calls = []

    def counting(t, m):
        calls.append(t)
        return U

    sim = Simulation(policy=counting, dt=0.05, control_period=0.10)
    run(sim, duration=1.0, log_period=0.10)
    assert calls == pytest.approx(np.arange(11) * 0.10)
    assert len(calls) == len(set(calls))
