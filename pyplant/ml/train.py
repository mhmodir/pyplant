"""Train the detector and write an artefact the inference service can load.

    python -m pyplant.ml.train --out artifacts --duration 3600

Produces `dataset.parquet`, `detector.joblib`, `manifest.json`, a per-mode
report, and the same numbers for the PCA baseline so the comparison is on the
record rather than in somebody's memory.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pyplant.ml.dataset import (
    Split,
    WindowedSet,
    build,
    default_dataset,
    load,
    save,
    split_scenarios,
    windowed,
)
from pyplant.ml.detector import (
    AutoencoderDetector,
    Detector,
    PCADetector,
    write_manifest,
)
from pyplant.ml.evaluate import Report, evaluate
from pyplant.ml.features import STRIDE, WINDOW, WindowSpec


def fit_and_evaluate(data: WindowedSet, split: Split, detector: Detector
                     ) -> tuple[Detector, Report]:
    train = data.of_scenarios(split.train)
    calibrate = data.of_scenarios(split.calibrate)
    detector.fit(train.X, data.names)
    detector.calibrate(calibrate.X)
    report = evaluate(detector,
                      normal=data.of_scenarios(split.test_normal),
                      faulty=data.of_scenarios(split.test_faulty))
    return detector, report


def main() -> None:  # pragma: no cover - entry point
    ap = argparse.ArgumentParser(description="train the pyplant fault detector")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--duration", type=float, default=3600.0)
    ap.add_argument("--magnitudes", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--normal-runs", type=int, default=6)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--quantile", type=float, default=0.995)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse", action="store_true",
                    help="use an existing dataset.parquet instead of regenerating")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    parquet = out / "dataset.parquet"

    if args.reuse and parquet.exists():
        print(f"reusing {parquet}", flush=True)
        frame = load(parquet)
    else:
        scenarios = default_dataset(args.duration, args.magnitudes,
                                    args.repeats, args.normal_runs)
        print(f"generating {len(scenarios)} runs of {args.duration:.0f}s",
              flush=True)
        started = time.perf_counter()
        frame = build(scenarios, progress=True)
        save(frame, parquet)
        print(f"  {len(frame):,} samples in "
              f"{time.perf_counter() - started:.0f}s -> {parquet}", flush=True)

    spec = WindowSpec(window=args.window, stride=args.stride)
    data = windowed(frame, spec)
    split = split_scenarios(frame, seed=args.seed)
    print(f"{len(data):,} windows | train {len(split.train)} "
          f"calibrate {len(split.calibrate)} normal-test "
          f"{len(split.test_normal)} faulty-test {len(split.test_faulty)}",
          flush=True)

    results = {}
    for name, detector in (("pca", PCADetector(quantile=args.quantile)),
                           ("autoencoder",
                            AutoencoderDetector(quantile=args.quantile,
                                                seed=args.seed))):
        fitted, report = fit_and_evaluate(data, split, detector)
        print(f"\n== {name} ==\n{report.summary()}\n"
              f"{report.per_mode.to_string(index=False)}", flush=True)
        results[name] = {
            "manifest": fitted.manifest(),
            "false_alarms_per_day": report.false_alarms_per_day,
            "detection_rate": report.detection_rate,
            "median_delay_s": report.median_delay,
            "isolation_accuracy": report.isolation_accuracy,
            "per_mode": report.per_mode.to_dict(orient="records"),
        }
        fitted.save(out / f"detector_{name}.joblib")
        report.per_run.to_csv(out / f"per_run_{name}.csv", index=False)

    # The service loads whichever detects more, ties going to the simpler model.
    best = max(results, key=lambda k: (results[k]["detection_rate"],
                                       -results[k]["false_alarms_per_day"]))
    Detector.load(out / f"detector_{best}.joblib").save(out / "detector.joblib")
    write_manifest(Detector.load(out / "detector.joblib"), out / "manifest.json",
                   extra={"selected": best, "window": args.window,
                          "stride": args.stride, "seed": args.seed})
    (out / "metrics.json").write_text(
        json.dumps({"selected": best, "results": results}, indent=2,
                   default=lambda o: None if isinstance(o, float)
                   and np.isnan(o) else str(o)))
    print(f"\nselected {best} -> {out / 'detector.joblib'}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
