"""Non-real-time driver, for generating datasets.

Same `Simulation` object the real-time runner uses, driven with no clock. A
simulated hour costs a couple of seconds, which is what makes a scenario grid
of a few hundred runs practical.

The recorded trajectory is in engineering units and carries the input tags
alongside the measurements, because the diagnosis model needs the controller
output: in a closed loop the controller absorbs part of the fault, so the
signature lives in OP as much as in PV.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from pyplant.core.hybrid_tank import H1, H2, Inputs, Params, Z
from pyplant.runners.base import Measurements, Policy, Simulation, _is_multiple

# Columns written for every sample, on top of the plant measurements.
INPUT_COLUMNS = ("FCV101_OP", "PMP101_CMD")
STATE_COLUMNS = ("h1_m", "h2_m", "valve_pos")


@dataclass
class Trajectory:
    """A logged run. Columns are engineering units; `t` is seconds."""

    t: np.ndarray
    columns: dict[str, np.ndarray]
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.t)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.columns[name]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.columns)

    def final(self) -> dict[str, float]:
        return {k: float(v[-1]) for k, v in self.columns.items()}

    def to_dataframe(self):
        import pandas as pd  # optional; only step 10 onward needs it

        df = pd.DataFrame({"t": self.t, **self.columns})
        return df.set_index("t")


class _Recorder:
    def __init__(self, names: tuple[str, ...], capacity: int) -> None:
        self.t = np.empty(capacity, dtype=float)
        self.data = {n: np.empty(capacity, dtype=float) for n in names}
        self.n = 0

    def append(self, t: float, row: dict[str, float | bool]) -> None:
        self.t[self.n] = t
        for name, buf in self.data.items():
            buf[self.n] = float(row[name])
        self.n += 1

    def trim(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        return self.t[: self.n], {k: v[: self.n] for k, v in self.data.items()}


def _row(sim: Simulation) -> dict[str, float | bool]:
    m: Measurements = dict(sim.measurements)
    m["FCV101_OP"] = sim.u.valve_cmd * 100.0
    m["PMP101_CMD"] = sim.u.pump_on
    m["h1_m"], m["h2_m"], m["valve_pos"] = (float(sim.x[i]) for i in (H1, H2, Z))
    return m


def run(
    sim: Simulation,
    duration: float,
    log_period: float = 1.0,
    log_states: bool = False,
    extra: Callable[[], dict[str, float | bool]] | None = None,
) -> Trajectory:
    """Drive `sim` for `duration` seconds, sampling every `log_period`.

    The first sample is taken at t=0 before any stepping, so a trajectory of
    `duration` seconds has `duration/log_period + 1` rows.

    `extra` supplies additional columns per sample, for signals the plant does
    not produce - controller faceplate values and alarm bits, which the
    diagnosis model needs. It is called once per logged sample and must return
    the same keys every time. The runner stays ignorant of what those keys
    mean, so it never grows a dependency on the PLC.
    """
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    if log_period < sim.dt or not _is_multiple(log_period, sim.dt):
        raise ValueError(
            f"log_period {log_period} must be a whole number of dt {sim.dt}"
        )

    steps = round(duration / sim.dt)
    every = round(log_period / sim.dt)

    def sample() -> dict[str, float | bool]:
        row = _row(sim)
        if extra is not None:
            row.update(extra())
        return row

    names = tuple(sample())
    if not log_states:
        names = tuple(n for n in names if n not in STATE_COLUMNS)

    rec = _Recorder(names, steps // every + 1)
    started = time.perf_counter()
    t0 = sim.t

    # Prime the controller before sampling, so the t=0 row carries real
    # faceplate and alarm values instead of a de-energised placeholder.
    sim.evaluate()
    rec.append(sim.t, sample())
    for i in range(1, steps + 1):
        sim.integrate()
        sim.evaluate()
        if i % every == 0:
            rec.append(sim.t, sample())

    wall = time.perf_counter() - started
    t, columns = rec.trim()
    return Trajectory(
        t=t,
        columns=columns,
        meta={
            "dt": sim.dt,
            "control_period": sim.control_period,
            "log_period": log_period,
            "duration": duration,
            "t_start": t0,
            "params": vars(sim.params).copy(),
            "wall_seconds": wall,
            "speedup": duration / wall if wall > 0 else float("inf"),
        },
    )


def settle(
    sim: Simulation,
    max_seconds: float = 7200.0,
    tol_mm: float = 0.01,
    window: float = 60.0,
) -> float:
    """Advance until tank levels stop moving. Returns the seconds consumed.

    Cheaper than guessing a warm-up length, and it fails loudly rather than
    silently handing a transient to the dataset.
    """
    every = round(window / sim.dt)
    last = np.array([sim.x[H1], sim.x[H2]])
    for i in range(1, round(max_seconds / sim.dt) + 1):
        sim.advance()
        if i % every == 0:
            now = np.array([sim.x[H1], sim.x[H2]])
            if np.max(np.abs(now - last)) * 1000.0 < tol_mm:
                return i * sim.dt
            last = now
    raise RuntimeError(
        f"plant did not settle within {max_seconds} s; check the policy"
    )
