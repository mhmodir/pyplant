"""Detectors.

Four of them, sharing one interface, because a learned detector's numbers mean
nothing without simpler baselines beside them. If the autoencoder cannot beat
a z-score limit check on this plant, the honest report says so.

    ZScoreDetector        the most straightforward method there is: how many
                          standard deviations each feature sits from its
                          normal mean, squared and summed. No model at all.

    IsolationForestDetector
                          a tree ensemble that scores how easily a window is
                          isolated from normal data. No reconstruction, so its
                          contributions fall back to the z-score deviations.

    PCADetector           multivariate SPE, the Q statistic of classical MSPC.
                          Fit a linear subspace on normal data; the anomaly
                          score is how far a window sits off it.

    AutoencoderDetector   a small neural bottleneck trained to reconstruct
                          normal windows. Same score, same contributions, a
                          non-linear manifold instead of a linear one.

Both score by reconstruction residual, which is what makes isolation fall out
for free: the residual is per-feature, every feature belongs to one channel, so
summing residuals by channel says which instrument is behaving oddly. That is
the whole of the contribution analysis, and it is the same argument classical
contribution plots rest on.

Thresholds are quantiles of the score on *held-out normal* data. Setting one on
the training set measures how well the model memorised, not how often it will
cry wolf.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from pyplant.ml.features import CHANNELS, STATS, channel_of

HEALTH_FLOOR = 0.0
HEALTH_CEILING = 100.0


@dataclass
class Detector:
    """Common machinery: scaling, thresholding, contributions, health index."""

    names: list[str] = field(default_factory=list)
    scaler: StandardScaler = field(default_factory=StandardScaler)
    threshold: float = np.inf
    quantile: float = 0.995
    fitted: bool = False

    kind: str = field(init=False, default="detector")

    # -- to implement ------------------------------------------------------

    def _fit_core(self, Z: np.ndarray) -> None:  # pragma: no cover
        raise NotImplementedError

    def _reconstruct(self, Z: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    # -- shared ------------------------------------------------------------

    def fit(self, X: np.ndarray, names: list[str]) -> "Detector":
        if len(X) < 10:
            raise ValueError(f"only {len(X)} training windows; need at least 10")
        self.names = list(names)
        Z = self.scaler.fit_transform(X)
        self._fit_core(Z)
        self.fitted = True
        return self

    def calibrate(self, X_normal: np.ndarray) -> float:
        """Set the threshold from held-out normal windows."""
        if not self.fitted:
            raise RuntimeError("fit before calibrating")
        self.threshold = float(np.quantile(self.score(X_normal), self.quantile))
        return self.threshold

    def residuals(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler.transform(X)
        return Z - self._reconstruct(Z)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Squared prediction error per window."""
        r = self.residuals(X)
        return np.einsum("ij,ij->i", r, r)

    def flags(self, X: np.ndarray) -> np.ndarray:
        return self.score(X) > self.threshold

    def health_from_score(self, score: np.ndarray) -> np.ndarray:
        """Map a raw score onto 0-100, with the threshold pinned at 50.

        Not cosmetic. Raw reconstruction error has no stable scale: retrain the
        model and it changes entirely. Pinning the threshold to 50 is what lets
        the alarm limit on `HEALTH_INDEX` stay where the alarm engineer put it
        across a retrain. A decade per 50 points keeps a badly broken plant
        distinguishable from a marginal one instead of both pegging at 100.
        """
        ratio = np.asarray(score, dtype=float) / (self.threshold or 1.0)
        index = 50.0 * np.log10(np.maximum(ratio, 1e-6)) + 50.0
        return np.clip(index, HEALTH_FLOOR, HEALTH_CEILING)

    def health(self, X: np.ndarray) -> np.ndarray:
        return self.health_from_score(self.score(X))

    def contributions(self, X: np.ndarray) -> np.ndarray:
        """Per-channel share of the residual, rows summing to one."""
        r2 = self.residuals(X) ** 2
        k = len(CHANNELS)
        by_channel = r2.reshape(len(X), k, len(STATS)).sum(axis=2)
        total = by_channel.sum(axis=1, keepdims=True)
        return by_channel / np.where(total > 0, total, 1.0)

    def top_channel(self, X: np.ndarray) -> np.ndarray:
        idx = self.contributions(X).argmax(axis=1)
        return np.asarray(CHANNELS)[idx]

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "Detector":
        import joblib

        return joblib.load(path)

    def manifest(self) -> dict:
        return {"kind": self.kind, "features": self.names,
                "channels": list(CHANNELS), "stats": list(STATS),
                "threshold": self.threshold, "quantile": self.quantile}


@dataclass
class PCADetector(Detector):
    """Classical MSPC. The baseline every learned detector should beat."""

    variance: float = 0.95
    pca: PCA | None = None

    def __post_init__(self) -> None:
        self.kind = "pca"

    def _fit_core(self, Z: np.ndarray) -> None:
        self.pca = PCA(n_components=self.variance, svd_solver="full")
        self.pca.fit(Z)

    def _reconstruct(self, Z: np.ndarray) -> np.ndarray:
        assert self.pca is not None
        return self.pca.inverse_transform(self.pca.transform(Z))

    @property
    def components(self) -> int:
        return 0 if self.pca is None else int(self.pca.n_components_)

    def manifest(self) -> dict:
        return {**super().manifest(), "components": self.components}


@dataclass
class AutoencoderDetector(Detector):
    """A neural bottleneck trained to reproduce normal operation."""

    hidden: tuple[int, ...] = (32, 8, 32)
    max_iter: int = 1500
    seed: int = 0
    net: MLPRegressor | None = None

    def __post_init__(self) -> None:
        self.kind = "autoencoder"

    def _fit_core(self, Z: np.ndarray) -> None:
        self.net = MLPRegressor(
            hidden_layer_sizes=self.hidden, activation="tanh", solver="adam",
            learning_rate_init=1e-3, max_iter=self.max_iter,
            early_stopping=True, n_iter_no_change=25, validation_fraction=0.15,
            random_state=self.seed)
        self.net.fit(Z, Z)

    def _reconstruct(self, Z: np.ndarray) -> np.ndarray:
        assert self.net is not None
        return self.net.predict(Z)

    def manifest(self) -> dict:
        return {**super().manifest(), "hidden": list(self.hidden),
                "iterations": int(self.net.n_iter_) if self.net else 0}


@dataclass
class ZScoreDetector(Detector):
    """Distance from the normal mean, in standard deviations, per feature.

    The scaler already centres and scales on normal data, so the
    reconstruction is simply the normal mean: zero in scaled units.
    """

    def __post_init__(self) -> None:
        self.kind = "zscore"

    def _fit_core(self, Z: np.ndarray) -> None:
        pass

    def _reconstruct(self, Z: np.ndarray) -> np.ndarray:
        return np.zeros_like(Z)


@dataclass
class IsolationForestDetector(Detector):
    """Random partitioning: anomalies are the windows isolated in few splits.

    The score is the negated sklearn score, which lies in (0, 1]: larger still
    means worse and stays positive, so the threshold and the log-scaled health
    index work unchanged.
    """

    trees: int = 200
    seed: int = 0
    forest: IsolationForest | None = None

    def __post_init__(self) -> None:
        self.kind = "isolation_forest"

    def _fit_core(self, Z: np.ndarray) -> None:
        self.forest = IsolationForest(n_estimators=self.trees,
                                      random_state=self.seed)
        self.forest.fit(Z)

    def _reconstruct(self, Z: np.ndarray) -> np.ndarray:
        return np.zeros_like(Z)

    def score(self, X: np.ndarray) -> np.ndarray:
        assert self.forest is not None
        return -self.forest.score_samples(self.scaler.transform(X))

    def manifest(self) -> dict:
        return {**super().manifest(), "trees": self.trees}


# The methods the workbench offers, simplest first.
METHODS: dict[str, str] = {
    "zscore": "Z-score limits - distance from normal, no model",
    "pca": "PCA / MSPC - classical linear baseline",
    "isolation_forest": "Isolation forest - tree ensemble outlier score",
    "autoencoder": "Autoencoder - neural non-linear reconstruction",
}


def make_detector(method: str, quantile: float = 0.995, seed: int = 0
                  ) -> Detector:
    if method == "zscore":
        return ZScoreDetector(quantile=quantile)
    if method == "pca":
        return PCADetector(quantile=quantile)
    if method == "isolation_forest":
        return IsolationForestDetector(quantile=quantile, seed=seed)
    if method == "autoencoder":
        return AutoencoderDetector(quantile=quantile, seed=seed)
    raise ValueError(f"unknown method {method!r}; choose from {list(METHODS)}")


def write_manifest(detector: Detector, path: str | Path, extra: dict | None = None
                   ) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**detector.manifest(), **(extra or {})},
                               indent=2, default=str))
    return path
