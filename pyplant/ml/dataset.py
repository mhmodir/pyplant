"""Datasets.

Generation is the easy part. The split is where this goes wrong.

Split by scenario, never by sample
----------------------------------
Windows overlap and neighbouring windows within one run are nearly identical.
A random sample split puts window 41 in training and window 42 in test, and the
model scores brilliantly on data it has effectively already seen. The number
that comes out is not wrong by a few points; it is meaningless.

Splitting whole runs is the only honest option, and it has a second benefit:
because each grid entry has its own seed, magnitude and onset, held-out runs
differ from training runs in the ways that matter operationally rather than
just in their noise realisation.

Train on normal only
--------------------
The detector never sees a fault during training. That is not a purity
exercise: a plant has years of normal operation and a handful of documented
failures, so a supervised classifier is trained on the rarest thing it will
ever meet and cannot recognise a mode nobody has labelled. Fault data is used
only to measure the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pyplant.faults.scenarios import Scenario, grid, run_scenario
from pyplant.ml.features import WindowSpec, add_derived, windows


@dataclass
class Split:
    """Scenario names, partitioned. Runs, not rows."""

    train: list[str] = field(default_factory=list)  # normal only
    calibrate: list[str] = field(default_factory=list)  # normal, sets threshold
    test_normal: list[str] = field(default_factory=list)
    test_faulty: list[str] = field(default_factory=list)

    @property
    def all(self) -> list[str]:
        return self.train + self.calibrate + self.test_normal + self.test_faulty

    def of(self, name: str) -> str:
        for part in ("train", "calibrate", "test_normal", "test_faulty"):
            if name in getattr(self, part):
                return part
        raise KeyError(name)


def build(scenarios: list[Scenario], log_period: float = 1.0,
          progress: bool = False) -> pd.DataFrame:
    """Run every scenario and stack the labelled signal frames."""
    frames = []
    for i, scenario in enumerate(scenarios, 1):
        result = run_scenario(scenario, log_period=log_period)
        frame = result.frame().reset_index()
        frame["fault_kind"] = (scenario.faults[0].kind if scenario.faults
                               else "normal")
        frame["fault_target"] = (scenario.faults[0].target if scenario.faults
                                 else "")
        frame["onset"] = (scenario.faults[0].t_start if scenario.faults
                          else np.nan)
        # Isolation must be judged after the fault has finished ramping, so the
        # ramp has to travel with the data.
        frame["ramp"] = (scenario.faults[0].ramp if scenario.faults else 0.0)
        frames.append(frame)
        if progress:
            print(f"  [{i}/{len(scenarios)}] {scenario.name}", flush=True)
    return pd.concat(frames, ignore_index=True)


def save(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def load(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def split_scenarios(frame: pd.DataFrame, seed: int = 0,
                    calibrate_fraction: float = 0.2,
                    test_fraction: float = 0.2) -> Split:
    """Partition run names. Normal runs are divided three ways.

    Training fits the model, calibration sets the threshold on data the model
    was not fitted to, and the normal test runs are what the false alarm rate
    is measured on. Using one pool for all three flatters the threshold.
    """
    rng = np.random.default_rng(seed)
    normal = sorted(frame.loc[~frame["faulty"].groupby(frame["scenario"])
                              .transform("any"), "scenario"].unique())
    faulty = sorted(set(frame["scenario"].unique()) - set(normal))
    if len(normal) < 3:
        raise ValueError(
            f"need at least three normal runs to split train, calibrate and "
            f"test; the dataset has {len(normal)}")

    normal = list(rng.permutation(normal))
    n_test = max(1, int(round(len(normal) * test_fraction)))
    n_cal = max(1, int(round(len(normal) * calibrate_fraction)))
    return Split(
        train=sorted(normal[n_test + n_cal:]),
        calibrate=sorted(normal[n_test:n_test + n_cal]),
        test_normal=sorted(normal[:n_test]),
        test_faulty=faulty,
    )


@dataclass
class WindowedSet:
    """Feature rows with the provenance needed to score them honestly."""

    X: np.ndarray
    scenario: np.ndarray
    t: np.ndarray
    label: np.ndarray
    faulty: np.ndarray
    kind: np.ndarray
    onset: np.ndarray
    ramp: np.ndarray
    names: list[str]

    def __len__(self) -> int:
        return len(self.X)

    def where(self, mask: np.ndarray) -> "WindowedSet":
        return WindowedSet(
            X=self.X[mask], scenario=self.scenario[mask], t=self.t[mask],
            label=self.label[mask], faulty=self.faulty[mask],
            kind=self.kind[mask], onset=self.onset[mask],
            ramp=self.ramp[mask], names=self.names)

    def of_scenarios(self, names: list[str]) -> "WindowedSet":
        return self.where(np.isin(self.scenario, list(names)))


def windowed(frame: pd.DataFrame, spec: WindowSpec | None = None) -> WindowedSet:
    """Turn a signal dataset into feature rows, one run at a time."""
    spec = spec or WindowSpec()
    parts: list[tuple] = []
    for name, run in frame.groupby("scenario", sort=True):
        run = run.sort_values("t")
        X, ends = windows(add_derived(run), spec)
        if len(X) == 0:
            continue
        tail = run.iloc[ends]
        parts.append((X, np.full(len(X), name), tail["t"].to_numpy(),
                      tail["label"].to_numpy(), tail["faulty"].to_numpy(),
                      tail["fault_kind"].to_numpy(), tail["onset"].to_numpy(),
                      tail["ramp"].to_numpy()))
    if not parts:
        raise ValueError("no run was long enough to fill a window")
    cols = [np.concatenate([p[i] for p in parts]) for i in range(1, 8)]
    return WindowedSet(np.vstack([p[0] for p in parts]), *cols, names=spec.names)


def default_dataset(duration: float = 3600.0, magnitudes: int = 3,
                    repeats: int = 2, normal_runs: int = 10) -> list[Scenario]:
    """The grid a real run of this project uses.

    Extra normal runs on purpose: the split needs enough of them to fit,
    calibrate and measure a false alarm rate without reusing any.
    """
    scenarios = grid(duration=duration, magnitudes=magnitudes, repeats=repeats)
    normals = [s for s in scenarios if s.is_normal]
    seed = 90_000
    while len(normals) < normal_runs:
        seed += 1
        extra = Scenario(f"normal_s{seed}", duration, [],
                         scenarios[0].setpoints, seed=seed,
                         description="normal operation")
        scenarios.append(extra)
        normals.append(extra)
    return scenarios
