"""Windows and features.

A detector does not look at one sample. It looks at a window, because the
things that distinguish faults from each other are shapes over time: a leak
shifts a level, a drift tilts it, stiction makes it oscillate. A single sample
cannot express any of those.

Per-tag summaries, not flattened windows
----------------------------------------
Sixty samples of fifteen tags is nine hundred numbers, most of them nearly
identical to their neighbours. Each tag is instead reduced to four numbers over
the window — last value, mean, standard deviation, and slope — which is small
enough to train on a modest dataset and, more usefully, keeps every feature
attributable to one tag. Isolation is only meaningful if a contribution can be
traced back to an instrument.

    last   where it is now
    mean   where it has been
    std    how much it is moving      <- stiction lives here
    slope  which way it is going      <- drift lives here

Derived channels
----------------
Three channels are computed rather than measured, each because a first-
principles argument says the fault shows up in a difference that no single tag
contains:

    OP_MINUS_ZT     command minus stem position. Zero on a healthy valve;
                    a sticking one opens a gap. This is the stiction signature.
    FLOW_IMBALANCE  what goes into tank 1 minus what leaves tank 2. Zero at
                    steady state by conservation of mass; a leak makes it
                    positive. This is the leak signature, and it is a
                    conservation law rather than a learned correlation.
    HEAD_RESIDUAL   pump pressure minus what the pump curve says it should
                    develop at the flow being delivered. Impeller wear opens
                    this gap.

Nothing here is a trend the model had to discover. Giving a model the residual
of a physical relationship is cheaper than making it learn the relationship.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pyplant.core.hybrid_tank import Params
from pyplant.tags import TAGS

# Signals the detector is allowed to see. Controller output and alarm bits are
# in here for the reason step 3 demonstrated: in a closed loop the controller
# absorbs the fault, so PV alone hides it.
BASE_TAGS: tuple[str, ...] = (
    "LT101", "LT102", "FT101", "FT102", "PT101", "ZT101",
    "FCV101_OP", "LIC101_SP", "LIC101_ERR",
)

DERIVED: tuple[str, ...] = ("OP_MINUS_ZT", "FLOW_IMBALANCE", "HEAD_RESIDUAL")

CHANNELS: tuple[str, ...] = BASE_TAGS + DERIVED

STATS: tuple[str, ...] = ("last", "mean", "std", "slope")

WINDOW = 60  # samples, one minute at 1 Hz
STRIDE = 10


def feature_names(channels: tuple[str, ...] = CHANNELS) -> list[str]:
    return [f"{c}::{s}" for c in channels for s in STATS]


def channel_of(feature: str) -> str:
    return feature.split("::", 1)[0]


def add_derived(frame: pd.DataFrame, params: Params | None = None) -> pd.DataFrame:
    """Append the physics-informed channels to a signal frame."""
    p = params or Params()
    out = frame.copy()
    out["OP_MINUS_ZT"] = out["FCV101_OP"] - out["ZT101"]
    out["FLOW_IMBALANCE"] = out["FT101"] - out["FT102"]

    # Expected pump pressure at the flow actually being delivered, in kPa.
    q = np.clip(out["FT101"].to_numpy() / 60_000.0, 0.0, p.pump_qmax)
    expected = p.pump_dp0 * (1.0 - (q / p.pump_qmax) ** 2) / 1000.0
    out["HEAD_RESIDUAL"] = out["PT101"].to_numpy() - expected
    return out


def _slope(block: np.ndarray) -> np.ndarray:
    """Least-squares slope per column, in units per sample."""
    n = block.shape[0]
    x = np.arange(n, dtype=float)
    x -= x.mean()
    denom = float((x * x).sum()) or 1.0
    return (x[:, None] * (block - block.mean(axis=0))).sum(axis=0) / denom


@dataclass
class WindowSpec:
    window: int = WINDOW
    stride: int = STRIDE
    channels: tuple[str, ...] = CHANNELS

    @property
    def names(self) -> list[str]:
        return feature_names(self.channels)


def windows(frame: pd.DataFrame, spec: WindowSpec | None = None
            ) -> tuple[np.ndarray, np.ndarray]:
    """Summarise a run into feature rows.

    Returns `(features, end_index)`, where `end_index[i]` is the row of `frame`
    the window closes on. A window is labelled by its *last* sample, never its
    first: a window straddling a fault onset is mostly healthy data, and calling
    it faulty from the first sample would make detection delay look negative.
    """
    spec = spec or WindowSpec()
    missing = [c for c in spec.channels if c not in frame.columns]
    if missing:
        raise KeyError(f"frame is missing channels: {missing}")
    data = frame[list(spec.channels)].to_numpy(dtype=float)
    n = len(data)
    if n < spec.window:
        return (np.empty((0, len(spec.names))), np.empty(0, dtype=int))

    rows, ends = [], []
    for start in range(0, n - spec.window + 1, spec.stride):
        block = data[start:start + spec.window]
        rows.append(np.concatenate([
            block[-1], block.mean(axis=0), block.std(axis=0), _slope(block)]))
        ends.append(start + spec.window - 1)

    # Interleave so features stay grouped by channel, matching `feature_names`.
    flat = np.asarray(rows)
    k = len(spec.channels)
    order = np.concatenate([[c + k * s for s in range(len(STATS))]
                            for c in range(k)])
    return flat[:, order], np.asarray(ends, dtype=int)


def check_channels() -> None:
    """Every base channel must be a real tag. Derived ones must not be."""
    unknown = [c for c in BASE_TAGS if c not in TAGS]
    if unknown:
        raise KeyError(f"BASE_TAGS names non-existent tags: {unknown}")
    clashing = [c for c in DERIVED if c in TAGS]
    if clashing:
        raise KeyError(f"derived channels shadow real tags: {clashing}")


check_channels()
