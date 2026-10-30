"""Evaluation.

Window-level accuracy is close to useless here. A fault that lasts half a run
makes "fraction of windows correctly classified" mostly a measure of how long
the fault was left on, and it says nothing about the two things an operator
cares about: how often the thing cries wolf, and how long it takes to notice.

So the report is three numbers per fault mode.

    false alarm rate     alarms per day on held-out *normal* runs. Not a
                         percentage of windows, because operators experience
                         alarms per shift, not per sample.
    detection delay      seconds from the declared onset to the first
                         *persisted* detection. A single flagged window is not
                         a detection; `persistence` consecutive ones are.
    isolation accuracy   how often the top contributing channel is one a
                         first-principles argument says it should be.

Persistence is doing real work. A detector that flickers above threshold for
one window in twenty has a fine-looking detection rate and is unusable, and
requiring three consecutive windows is the cheapest way to make the delay
number honest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pyplant.ml.dataset import WindowedSet
from pyplant.ml.detector import Detector

PERSISTENCE = 3  # consecutive flagged windows before a detection is declared
SECONDS_PER_DAY = 86_400.0

# Only the modes where a conservation argument names one channel unambiguously.
#
# The sensor faults are deliberately absent, and the reason is worth stating.
# In a closed loop a level transmitter reading 25 mm high is driven back to
# setpoint by the controller: the *reading* looks perfect, and the true level
# sits 25 mm low. The signature therefore appears everywhere except the faulty
# instrument, and "the top channel should be LT101" is exactly wrong. Scoring
# sensor faults against a first-principles expectation would be scoring them
# against a mistake, so they are reported by consistency instead.
EXPECTED_CHANNEL: dict[str, tuple[str, ...]] = {
    "leak": ("FLOW_IMBALANCE",),
    "pump_wear": ("HEAD_RESIDUAL",),
    "stiction": ("OP_MINUS_ZT",),
    "fouling": ("LT102", "FT102"),
}

UNSCORED_MODES = ("sensor_bias", "sensor_drift", "sensor_stuck")

DEVELOPED_AFTER = 300.0  # s past the end of the ramp before judging isolation


def _persistent_events(flags: np.ndarray, persistence: int) -> int:
    """Number of separate excursions lasting at least `persistence` windows."""
    events, streak, counted = 0, 0, False
    for flag in flags:
        if flag:
            streak += 1
            if streak == persistence:
                events, counted = events + 1, True
        else:
            streak, counted = 0, False
    return events


def first_persistent(flags: np.ndarray, persistence: int = PERSISTENCE) -> int:
    """Index of the first flag in the first run of `persistence` consecutive."""
    if len(flags) < persistence:
        return -1
    streak = 0
    for i, flag in enumerate(flags):
        streak = streak + 1 if flag else 0
        if streak >= persistence:
            return i - persistence + 1
    return -1


@dataclass
class Report:
    per_run: pd.DataFrame
    per_mode: pd.DataFrame
    false_alarms_per_day: float
    normal_window_flag_rate: float
    detection_rate: float
    median_delay: float
    isolation_accuracy: float
    isolation_consistency: float

    def summary(self) -> str:
        lines = [
            f"false alarms   {self.false_alarms_per_day:6.2f} per day "
            f"({self.normal_window_flag_rate * 100:.2f}% of normal windows)",
            f"detected       {self.detection_rate * 100:6.1f}% of faulty runs",
            f"median delay   {self.median_delay:6.0f} s",
            f"isolation      {self.isolation_accuracy * 100:6.1f}% correct "
            f"(scored modes only)",
            f"consistency    {self.isolation_consistency * 100:6.1f}% same "
            f"channel within a mode",
        ]
        return "\n".join(lines)


def evaluate(detector: Detector, normal: WindowedSet, faulty: WindowedSet,
             window_seconds: float = 60.0, stride_seconds: float = 10.0,
             persistence: int = PERSISTENCE) -> Report:
    rows: list[dict] = []

    for name in np.unique(normal.scenario):
        run = normal.of_scenarios([name])
        flags = detector.flags(run.X)
        # Count alarm *events* the same way a detection is counted: a run of
        # `persistence` consecutive flags. Counting every rising edge would
        # punish single-window flicker that the detection side ignores, and the
        # two numbers would not be measuring the same detector.
        events = _persistent_events(flags, persistence)
        covered = float(run.t[-1] - run.t[0] + window_seconds)
        rows.append({"scenario": name, "kind": "normal", "faulty": False,
                     "windows": len(run), "flagged": int(flags.sum()),
                     "events": events, "seconds": covered,
                     "detected": bool(first_persistent(flags, persistence) >= 0),
                     "delay": np.nan, "top": None, "isolated": np.nan})

    for name in np.unique(faulty.scenario):
        run = faulty.of_scenarios([name])
        order = np.argsort(run.t)
        run = run.where(order)
        flags = detector.flags(run.X)
        kind = str(run.kind[0])
        onset = float(run.onset[0])

        after = run.t >= onset
        idx = first_persistent(flags[after], persistence)
        detected = idx >= 0
        delay = float(run.t[after][idx] - onset) if detected else np.nan

        # Judge isolation only once the fault has finished ramping. A fouling
        # fault ramping over twenty minutes is still a transient five minutes
        # in, and the transient implicates the controlled level rather than the
        # downstream one the settled fault points at.
        ramp = float(run.ramp[0]) if len(run.ramp) else 0.0
        settled = after & (run.t >= onset + ramp + DEVELOPED_AFTER)
        fully_developed = bool(settled.any())
        if not fully_developed:
            # The run ended before the ramp finished. Judge on the last quarter
            # of what there is and say so, rather than silently reporting
            # nothing: a slow fault in a short run is a real situation.
            tail = onset + 0.75 * (run.t[-1] - onset) if len(run.t) else onset
            settled = after & (run.t >= tail)
        developed = run.where(settled)
        if len(developed):
            tops = detector.top_channel(developed.X)
            top = pd.Series(tops).mode().iloc[0]
            isolated = (top in EXPECTED_CHANNEL[kind]
                        if kind in EXPECTED_CHANNEL else np.nan)
        else:
            top, isolated = None, np.nan

        before = flags[~after]
        rows.append({"scenario": name, "kind": kind, "faulty": True,
                     "windows": len(run), "flagged": int(flags.sum()),
                     "events": int(before.sum()), "seconds": np.nan,
                     "detected": detected, "delay": delay,
                     "top": top, "isolated": isolated,
                     "fully_developed": fully_developed})

    per_run = pd.DataFrame(rows)
    normal_runs = per_run[~per_run["faulty"]]
    faulty_runs = per_run[per_run["faulty"]]

    normal_seconds = float(normal_runs["seconds"].sum()) or 1.0
    far = float(normal_runs["events"].sum()) * SECONDS_PER_DAY / normal_seconds
    flag_rate = (float(normal_runs["flagged"].sum())
                 / max(float(normal_runs["windows"].sum()), 1.0))

    # Consistency: does one fault mode always point at the same channel? It
    # needs no ground truth, so it is the only isolation number available for
    # the sensor faults.
    consistency = []
    for kind, group in faulty_runs.groupby("kind"):
        tops = group["top"].dropna()
        consistency.append(float(tops.value_counts().iloc[0] / len(tops))
                           if len(tops) else np.nan)

    per_mode = (faulty_runs.groupby("kind")
                .agg(runs=("scenario", "size"),
                     detected=("detected", "mean"),
                     median_delay=("delay", "median"),
                     isolation=("isolated", "mean"),
                     developed=("fully_developed", "mean"),
                     top=("top", lambda s: s.dropna().mode().iloc[0]
                          if len(s.dropna()) else None))
                .reset_index())
    per_mode["consistency"] = consistency
    per_mode["scored"] = per_mode["kind"].isin(EXPECTED_CHANNEL)

    return Report(
        per_run=per_run,
        per_mode=per_mode,
        false_alarms_per_day=far,
        normal_window_flag_rate=flag_rate,
        detection_rate=float(faulty_runs["detected"].mean()) if len(faulty_runs) else 0.0,
        median_delay=float(faulty_runs["delay"].median()) if len(faulty_runs) else np.nan,
        isolation_accuracy=float(faulty_runs["isolated"].mean(skipna=True))
        if faulty_runs["isolated"].notna().any() else float("nan"),
        isolation_consistency=float(np.nanmean(consistency))
        if consistency else float("nan"),
    )
