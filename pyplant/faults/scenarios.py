"""Scenarios: a plant run with a script and a ground-truth label.

This is what simulation gives you that a real rig never will. Every sample has
a label, every fault has a known start time, magnitude and ramp, and the whole
run is reproducible from a seed. Detection delay, false alarm rate and
isolation accuracy stop being adjectives and become numbers.

A scenario is warmed up before recording begins, so no run opens with a startup
transient that a model would learn as "faults happen at the start".
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from pyplant.core.hybrid_tank import Inputs, Params
from pyplant.faults.model import (
    Fault,
    FaultSet,
    Fouling,
    Leak,
    PumpWear,
    SensorBias,
    SensorDrift,
    SensorNoise,
    SensorStuck,
    ValveStiction,
)
from pyplant.plc.harness import OperatorPanel, PlcHarness, scheduled_setpoint
from pyplant.plc.program import Program
from pyplant.runners.base import Simulation
from pyplant.runners.fast import Trajectory, run, settle

DESIGN = Inputs(0.662, True)
DEFAULT_SP = 300.0

# Instrument noise, in engineering units, present on every run.
#
# This is not decoration. A perfectly quiet loop never moves its valve, so
# stiction cannot manifest and produces no limit cycle at all; and a detector
# trained on noiseless signals learns a threshold that collapses the moment it
# meets a real transmitter. Noise is applied as `benign` faults: recorded in
# the fault set, never counted as a positive label.
DEFAULT_NOISE: dict[str, float] = {
    "LT101": 0.4, "LT102": 0.4, "FT101": 0.03, "PT101": 0.08, "ZT101": 0.05,
}


@dataclass
class Scenario:
    """One reproducible run: a setpoint programme plus a fault timeline."""

    name: str
    duration: float = 3600.0
    faults: list[Fault] = field(default_factory=list)
    setpoints: list[tuple[float, float]] = field(default_factory=list)
    seed: int = 0
    params: Params = field(default_factory=Params)
    noise: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_NOISE))
    description: str = ""

    def __post_init__(self) -> None:
        if self.duration <= 0.0:
            raise ValueError("duration must be positive")
        if not self.setpoints:
            self.setpoints = [(0.0, DEFAULT_SP)]
        if self.setpoints[0][0] != 0.0:
            raise ValueError("the setpoint programme must start at t=0")
        for fault in self.faults:
            if fault.t_start >= self.duration:
                raise ValueError(
                    f"fault {fault.id!r} starts at {fault.t_start}s, after the "
                    f"{self.duration}s scenario ends")

    @property
    def is_normal(self) -> bool:
        return not self.faults


@dataclass
class ScenarioResult:
    scenario: Scenario
    trajectory: Trajectory
    events: pd.DataFrame
    labels: pd.DataFrame

    def frame(self) -> pd.DataFrame:
        """Everything a training pipeline needs, in one table."""
        df = self.trajectory.to_dataframe()
        df["label"] = self.labels["label"].to_numpy()
        df["faulty"] = self.labels["faulty"].to_numpy()
        df["scenario"] = self.scenario.name
        return df


def run_scenario(
    scenario: Scenario,
    dt: float = 0.05,
    control_period: float = 0.10,
    log_period: float = 1.0,
    warmup: float = 1800.0,
) -> ScenarioResult:
    """Line the plant out, then record `duration` seconds with the script running.

    The fault objects are stateful — stiction carries a stem position — so a
    fresh `FaultSet` is built per run and the scenario's own fault list is
    never mutated.
    """
    harness = PlcHarness(
        program=Program(ts=control_period),
        panel=OperatorPanel(setpoint=scenario.setpoints[0][1], auto=True,
                            pump_start=True),
    )
    sim = Simulation.at_steady_state(DESIGN, params=scenario.params,
                                     dt=dt, control_period=control_period,
                                     policy=harness)
    settle(sim, max_seconds=warmup)

    # Faults are timed from the start of the recording, not from the warm-up.
    noise = [SensorNoise(f"_noise_{tag}", tag=tag, sigma=sigma, benign=True)
             for tag, sigma in scenario.noise.items() if sigma > 0.0]
    faults = FaultSet(faults=[replace(f) for f in scenario.faults] + noise,
                      seed=scenario.seed)
    sim.faults = faults
    sim.t = 0.0
    sim._step_count = 0
    sim.policy = scheduled_setpoint(harness, scenario.setpoints)

    trajectory = run(sim, duration=scenario.duration, log_period=log_period,
                     extra=harness.tags)

    labels = pd.DataFrame({
        "t": trajectory.t,
        "label": [faults.labels_at(float(t)) for t in trajectory.t],
    })
    labels["faulty"] = labels["label"] != "normal"

    events = pd.DataFrame(faults.events() or
                          [{"fault_id": None, "kind": "normal", "target": None,
                            "t_start": None, "t_end": None, "ramp": None,
                            "magnitude": None}])
    events.insert(0, "scenario", scenario.name)
    events["seed"] = scenario.seed
    events["duration"] = scenario.duration

    return ScenarioResult(scenario=scenario, trajectory=trajectory,
                          events=events, labels=labels)


# ---------------------------------------------------------------------------
# The library
# ---------------------------------------------------------------------------

NORMAL_MOVES: list[tuple[float, float]] = [
    (0.0, 300.0), (900.0, 340.0), (1800.0, 280.0), (2700.0, 300.0),
]


def library(duration: float = 3600.0) -> list[Scenario]:
    """One representative scenario per fault mode, plus normal operation.

    Onset is deliberately late enough that every run contains a substantial
    healthy stretch: a detector needs to learn normal from the same
    distribution it will be judged against.
    """
    onset = duration * 0.4
    return [
        Scenario("normal", duration, [], NORMAL_MOVES, seed=1,
                 description="setpoint moves only, no fault"),
        Scenario("leak_abrupt", duration,
                 [Leak("leak", t_start=onset, cda=2.0e-5)],
                 NORMAL_MOVES, seed=2,
                 description="tank 1 leak, step onset"),
        Scenario("leak_incipient", duration,
                 [Leak("leak", t_start=onset, cda=2.0e-5, ramp=900.0)],
                 NORMAL_MOVES, seed=3,
                 description="tank 1 leak opening over fifteen minutes"),
        Scenario("pump_wear", duration,
                 [PumpWear("wear", t_start=onset, loss=0.18, ramp=600.0)],
                 NORMAL_MOVES, seed=4,
                 description="impeller wear, gradual"),
        Scenario("fouling", duration,
                 [Fouling("fouling", t_start=onset, loss=0.35, ramp=1200.0)],
                 NORMAL_MOVES, seed=5,
                 description="interconnect restriction, gradual"),
        Scenario("sensor_bias", duration,
                 [SensorBias("bias", t_start=onset, tag="LT101", bias=25.0)],
                 NORMAL_MOVES, seed=6,
                 description="LT101 reads 25 mm high"),
        Scenario("sensor_drift", duration,
                 [SensorDrift("drift", t_start=onset, tag="LT101",
                              rate_per_min=1.5)],
                 NORMAL_MOVES, seed=7,
                 description="LT101 drifting up"),
        Scenario("sensor_stuck", duration,
                 [SensorStuck("stuck", t_start=onset, tag="LT101")],
                 NORMAL_MOVES, seed=8,
                 description="LT101 frozen"),
        Scenario("stiction", duration,
                 [ValveStiction("stiction", t_start=onset, deadband=0.06,
                                slip=0.04)],
                 NORMAL_MOVES, seed=9,
                 description="FCV101 sticking, deadband 6% slip 4%"),
    ]


def grid(duration: float = 3600.0, magnitudes: int = 3,
         repeats: int = 2) -> list[Scenario]:
    """The scenario grid a dataset is built from.

    Each fault mode is swept across magnitudes and repeated with different
    seeds and onset times, so a model cannot learn "the fault starts at 24
    minutes" instead of learning the fault.
    """
    if magnitudes < 1 or repeats < 1:
        raise ValueError("magnitudes and repeats must be at least 1")
    scale = np.linspace(0.5, 1.5, magnitudes)
    rng = np.random.default_rng(0)
    out: list[Scenario] = []
    seed = 1000

    for _ in range(repeats):
        seed += 1
        out.append(Scenario(f"normal_s{seed}", duration, [], NORMAL_MOVES,
                            seed=seed, description="normal operation"))

    for base in library(duration):
        if base.is_normal:
            continue
        for k in scale:
            for _ in range(repeats):
                seed += 1
                onset = float(rng.uniform(0.25, 0.6) * duration)
                out.append(Scenario(
                    name=f"{base.name}_x{k:.2f}_s{seed}",
                    duration=duration,
                    faults=[_scaled(f, k, onset) for f in base.faults],
                    setpoints=NORMAL_MOVES,
                    seed=seed,
                    description=base.description,
                ))
    return out


def _scaled(fault: Fault, k: float, onset: float) -> Fault:
    """A copy of a fault at `k` times its magnitude, starting at `onset`."""
    clone = replace(fault, t_start=onset)
    for attr in ("cda", "bias", "rate_per_min", "sigma"):
        if hasattr(clone, attr):
            setattr(clone, attr, getattr(clone, attr) * k)
    for attr in ("loss", "deadband"):
        if hasattr(clone, attr):
            setattr(clone, attr, min(getattr(clone, attr) * k, 0.9))
    if hasattr(clone, "slip") and hasattr(clone, "deadband"):
        clone.slip = min(clone.slip * k, clone.deadband)
    return clone
