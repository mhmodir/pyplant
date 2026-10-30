"""Tests for the learning pipeline.

`test_no_scenario_appears_in_two_splits` is the one that matters most. Windows
overlap, and neighbouring windows from one run are nearly identical, so a
random sample split hands the model its own test set and produces a number that
looks excellent and means nothing. Splitting whole runs is the only honest
option, and it is the kind of mistake that is invisible unless something
asserts against it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyplant.core.hybrid_tank import Params
from pyplant.faults.model import Leak, ValveStiction
from pyplant.faults.scenarios import Scenario
from pyplant.ml.dataset import (
    build,
    load,
    save,
    split_scenarios,
    windowed,
)
from pyplant.ml.detector import (
    METHODS,
    AutoencoderDetector,
    Detector,
    PCADetector,
    make_detector,
    write_manifest,
)
from pyplant.ml.evaluate import (
    EXPECTED_CHANNEL,
    UNSCORED_MODES,
    _persistent_events,
    evaluate,
    first_persistent,
)
from pyplant.ml.features import (
    BASE_TAGS,
    CHANNELS,
    DERIVED,
    STATS,
    WindowSpec,
    add_derived,
    channel_of,
    feature_names,
    windows,
)
from pyplant.ml.train import fit_and_evaluate
from pyplant.tags import TAGS

SPEC = WindowSpec(window=20, stride=5)


# ============================== features ===================================

def test_base_channels_are_real_tags_and_derived_ones_are_not():
    assert all(c in TAGS for c in BASE_TAGS)
    assert not any(c in TAGS for c in DERIVED)


def test_feature_names_are_grouped_by_channel():
    names = feature_names(("A", "B"))
    assert names == ["A::last", "A::mean", "A::std", "A::slope",
                     "B::last", "B::mean", "B::std", "B::slope"]
    assert channel_of("FLOW_IMBALANCE::std") == "FLOW_IMBALANCE"


def test_derived_channels_are_the_residuals_they_claim_to_be():
    frame = pd.DataFrame({"FCV101_OP": [66.0], "ZT101": [60.0],
                          "FT101": [10.0], "FT102": [9.0], "PT101": [50.0]})
    out = add_derived(frame, Params())
    assert out["OP_MINUS_ZT"].iloc[0] == pytest.approx(6.0)
    assert out["FLOW_IMBALANCE"].iloc[0] == pytest.approx(1.0)

    p = Params()
    q = 10.0 / 60_000.0
    expected = p.pump_dp0 * (1 - (q / p.pump_qmax) ** 2) / 1000.0
    assert out["HEAD_RESIDUAL"].iloc[0] == pytest.approx(50.0 - expected)


def test_flow_imbalance_is_a_conservation_law_not_a_correlation():
    # Balanced in, balanced out: zero, whatever the levels happen to be.
    frame = pd.DataFrame({"FCV101_OP": [66.0], "ZT101": [66.0],
                          "FT101": [10.0], "FT102": [10.0], "PT101": [50.0]})
    assert add_derived(frame)["FLOW_IMBALANCE"].iloc[0] == 0.0


def _signal_frame(n=60, slope=0.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    base = 300.0 + slope * t + rng.normal(0, noise, n)
    return pd.DataFrame({c: base if c == "LT101" else np.full(n, 1.0)
                         for c in CHANNELS})


def test_window_summaries_capture_level_movement_and_direction():
    spec = WindowSpec(window=20, stride=20, channels=("LT101",))
    X, ends = windows(_signal_frame(40, slope=2.0), spec)
    assert X.shape == (2, 4)
    last, mean, std, slope = X[0]
    assert last == pytest.approx(300.0 + 2.0 * 19)
    assert slope == pytest.approx(2.0)
    assert std > 0
    assert list(ends) == [19, 39]


def test_std_captures_oscillation_and_slope_does_not():
    spec = WindowSpec(window=20, stride=20, channels=("LT101",))
    frame = _signal_frame(20)
    frame["LT101"] = 300.0 + 10.0 * np.sin(np.arange(20) * np.pi / 5)
    _, _, std, slope = windows(frame, spec)[0][0]
    assert std > 5.0
    assert abs(slope) < 0.5


def test_a_window_is_labelled_by_its_last_sample():
    # Otherwise a window straddling the onset counts as faulty from its first
    # sample, and detection delay comes out negative.
    spec = WindowSpec(window=20, stride=10, channels=("LT101",))
    _, ends = windows(_signal_frame(50), spec)
    assert ends[0] == 19 and ends[-1] == 49


def test_short_runs_produce_no_windows():
    spec = WindowSpec(window=100, channels=("LT101",))
    X, ends = windows(_signal_frame(50), spec)
    assert len(X) == 0 and len(ends) == 0


def test_missing_channels_are_reported():
    with pytest.raises(KeyError, match="missing channels"):
        windows(pd.DataFrame({"LT101": [1.0] * 30}), SPEC)


# ============================== dataset ====================================

@pytest.fixture(scope="module")
def small_dataset() -> pd.DataFrame:
    """Four normal runs and three faulty ones, short enough to be quick."""
    moves = [(0.0, 300.0), (300.0, 320.0)]
    scenarios = [Scenario(f"normal_{i}", 600.0, [], moves, seed=i)
                 for i in range(4)]
    scenarios += [
        Scenario("leak_a", 600.0, [Leak("leak", t_start=300.0, cda=4e-5)],
                 moves, seed=10),
        Scenario("leak_b", 600.0, [Leak("leak", t_start=250.0, cda=6e-5)],
                 moves, seed=11),
        Scenario("stiction_a", 600.0,
                 [ValveStiction("stiction", t_start=300.0, deadband=0.12,
                                slip=0.08)], moves, seed=12),
    ]
    return build(scenarios)


def test_dataset_carries_labels_and_ground_truth(small_dataset):
    assert {"scenario", "label", "faulty", "fault_kind", "onset", "ramp"} \
        <= set(small_dataset.columns)
    leak = small_dataset[small_dataset["scenario"] == "leak_a"]
    assert not leak.loc[leak["t"] < 300.0, "faulty"].any()
    assert leak.loc[leak["t"] >= 300.0, "faulty"].all()
    assert (small_dataset[small_dataset["scenario"] == "normal_0"]["faulty"]
            == False).all()  # noqa: E712


def test_parquet_round_trip(small_dataset, tmp_path):
    path = save(small_dataset, tmp_path / "d.parquet")
    back = load(path)
    assert len(back) == len(small_dataset)
    assert list(back.columns) == list(small_dataset.columns)


def test_no_scenario_appears_in_two_splits(small_dataset):
    split = split_scenarios(small_dataset, seed=0)
    parts = [split.train, split.calibrate, split.test_normal, split.test_faulty]
    flat = [name for part in parts for name in part]
    assert len(flat) == len(set(flat))
    assert set(flat) == set(small_dataset["scenario"].unique())


def test_only_normal_runs_are_used_for_fitting(small_dataset):
    split = split_scenarios(small_dataset, seed=0)
    faulty_runs = set(small_dataset.loc[small_dataset["faulty"], "scenario"])
    assert not set(split.train) & faulty_runs
    assert not set(split.calibrate) & faulty_runs
    assert set(split.test_faulty) == faulty_runs


def test_split_refuses_to_run_on_too_few_normal_runs(small_dataset):
    one_normal = small_dataset[small_dataset["scenario"].isin(
        ["normal_0", "leak_a"])]
    with pytest.raises(ValueError, match="at least three normal runs"):
        split_scenarios(one_normal)


def test_windowing_keeps_provenance_aligned(small_dataset):
    data = windowed(small_dataset, SPEC)
    assert len(data.X) == len(data.t) == len(data.scenario) == len(data.faulty)
    assert data.X.shape[1] == len(CHANNELS) * len(STATS)
    assert data.names == feature_names()
    leak = data.of_scenarios(["leak_a"])
    assert set(leak.scenario) == {"leak_a"}
    assert leak.faulty[leak.t < 300.0].sum() == 0


def test_windowing_a_frame_with_no_usable_run_raises(small_dataset):
    with pytest.raises(ValueError, match="long enough"):
        windowed(small_dataset, WindowSpec(window=100_000))


# ============================== detectors ==================================

@pytest.fixture(scope="module")
def fitted(small_dataset):
    data = windowed(small_dataset, SPEC)
    split = split_scenarios(small_dataset, seed=0)
    det = PCADetector()
    det.fit(data.of_scenarios(split.train).X, data.names)
    det.calibrate(data.of_scenarios(split.calibrate).X)
    return det, data, split


def test_fit_requires_enough_windows():
    with pytest.raises(ValueError, match="at least 10"):
        PCADetector().fit(np.zeros((5, len(CHANNELS) * len(STATS))),
                          feature_names())


def test_calibrating_before_fitting_is_refused():
    with pytest.raises(RuntimeError, match="fit before"):
        PCADetector().calibrate(np.zeros((20, 4)))


def test_threshold_comes_from_held_out_normal_data(fitted):
    det, data, split = fitted
    assert np.isfinite(det.threshold) and det.threshold > 0
    calibrate = data.of_scenarios(split.calibrate)
    flagged = det.flags(calibrate.X).mean()
    assert flagged <= (1.0 - det.quantile) + 0.02


def test_scores_are_non_negative(fitted):
    det, data, _ = fitted
    assert (det.score(data.X) >= 0).all()


def test_health_index_is_bounded_and_monotonic(fitted):
    det, data, _ = fitted
    health = det.health(data.X)
    score = det.score(data.X)
    assert (health >= 0).all() and (health <= 100).all()
    order = np.argsort(score)
    assert np.all(np.diff(health[order]) >= -1e-9)  # never falls as score rises


def test_the_threshold_maps_to_fifty_whatever_the_score_scale(fitted):
    # This is what lets an alarm limit on HEALTH_INDEX survive a retrain that
    # changes the raw score's scale entirely.
    det, _, _ = fitted
    assert det.health_from_score(np.array([det.threshold]))[0] \
        == pytest.approx(50.0)
    assert det.health_from_score(np.array([det.threshold * 10]))[0] \
        == pytest.approx(100.0)
    assert det.health_from_score(np.array([det.threshold / 10]))[0] \
        == pytest.approx(0.0)


def test_contributions_are_a_per_channel_share(fitted):
    det, data, _ = fitted
    contributions = det.contributions(data.X[:50])
    assert contributions.shape == (50, len(CHANNELS))
    assert np.allclose(contributions.sum(axis=1), 1.0)
    assert (contributions >= 0).all()


def test_top_channel_names_a_real_channel(fitted):
    det, data, _ = fitted
    assert set(det.top_channel(data.X[:50])) <= set(CHANNELS)


def test_a_leak_isolates_to_the_mass_balance(fitted):
    det, data, _ = fitted
    leak = data.of_scenarios(["leak_a", "leak_b"])
    developed = leak.where(leak.t >= 450.0)
    top = pd.Series(det.top_channel(developed.X)).mode().iloc[0]
    assert top in EXPECTED_CHANNEL["leak"]


def test_detector_round_trips_through_disk(fitted, tmp_path):
    det, data, _ = fitted
    path = det.save(tmp_path / "d.joblib")
    back = Detector.load(path)
    assert back.threshold == det.threshold
    assert np.allclose(back.score(data.X[:20]), det.score(data.X[:20]))


def test_manifest_records_what_the_service_needs(fitted, tmp_path):
    det, _, _ = fitted
    path = write_manifest(det, tmp_path / "m.json", extra={"selected": "pca"})
    import json

    manifest = json.loads(path.read_text())
    assert manifest["kind"] == "pca"
    assert manifest["channels"] == list(CHANNELS)
    assert manifest["threshold"] == det.threshold
    assert manifest["selected"] == "pca"


def test_autoencoder_satisfies_the_same_interface(small_dataset):
    data = windowed(small_dataset, SPEC)
    split = split_scenarios(small_dataset, seed=0)
    det = AutoencoderDetector(max_iter=120, seed=0)
    det.fit(data.of_scenarios(split.train).X, data.names)
    det.calibrate(data.of_scenarios(split.calibrate).X)
    assert det.kind == "autoencoder"
    assert np.allclose(det.contributions(data.X[:10]).sum(axis=1), 1.0)
    assert det.manifest()["hidden"] == [32, 8, 32]


@pytest.mark.parametrize("method", ["zscore", "isolation_forest"])
def test_the_simple_methods_satisfy_the_same_interface(small_dataset, method):
    data = windowed(small_dataset, SPEC)
    split = split_scenarios(small_dataset, seed=0)
    det, report = fit_and_evaluate(data, split, make_detector(method))
    assert det.kind == method
    assert np.isfinite(det.threshold) and det.threshold > 0
    assert (det.score(data.X) > 0).all()  # the health index takes a log
    assert np.allclose(det.contributions(data.X[:10]).sum(axis=1), 1.0)
    assert set(det.top_channel(data.X[:10])) <= set(CHANNELS)
    assert 0.0 <= report.detection_rate <= 1.0


def test_every_offered_method_can_be_built():
    for method in METHODS:
        assert make_detector(method).kind == method
    with pytest.raises(ValueError, match="unknown method"):
        make_detector("crystal_ball")


# ============================== evaluation =================================

def test_persistence_ignores_a_single_flagged_window():
    flags = np.array([0, 1, 0, 1, 0, 1, 1, 1, 0], dtype=bool)
    assert first_persistent(flags, 3) == 5
    assert first_persistent(np.array([1, 0, 1, 0], dtype=bool), 3) == -1


def test_events_count_excursions_not_windows():
    flags = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1], dtype=bool)
    assert _persistent_events(flags, 3) == 2  # two sustained excursions
    assert _persistent_events(flags, 5) == 0  # neither lasts five


def test_sensor_modes_are_deliberately_unscored():
    # Under closed loop a level transmitter fault is driven back to setpoint,
    # so the reading looks perfect and the signature moves elsewhere. Scoring
    # against "it should be LT101" would be scoring against a mistake.
    assert not set(UNSCORED_MODES) & set(EXPECTED_CHANNEL)
    assert set(EXPECTED_CHANNEL) == {"leak", "pump_wear", "stiction", "fouling"}


def test_evaluate_reports_detection_delay_and_false_alarms(fitted):
    det, data, split = fitted
    report = evaluate(det,
                      normal=data.of_scenarios(split.test_normal),
                      faulty=data.of_scenarios(split.test_faulty),
                      persistence=2)
    assert set(report.per_run["scenario"]) == set(split.test_normal
                                                  + split.test_faulty)
    assert report.false_alarms_per_day >= 0.0
    assert 0.0 <= report.detection_rate <= 1.0
    faulty_rows = report.per_run[report.per_run["faulty"]]
    detected = faulty_rows[faulty_rows["detected"]]
    assert (detected["delay"] >= 0).all()  # never negative
    assert "consistency" in report.per_mode.columns
    assert "false alarms" in report.summary()


def test_the_whole_pipeline_runs_end_to_end(small_dataset):
    data = windowed(small_dataset, SPEC)
    split = split_scenarios(small_dataset, seed=0)
    det, report = fit_and_evaluate(data, split, PCADetector())
    assert det.fitted and np.isfinite(det.threshold)
    assert len(report.per_run) > 0
