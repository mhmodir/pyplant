"""Work that takes minutes.

These run in their own process, launched by `JobRunner`. Two constraints shape
them:

* They must be importable at module level, because the spawn start method
  pickles them by name. A closure or a nested function will not survive.
* Heavy imports happen inside the functions, not at the top. The workbench
  imports this module at startup, and pulling in scikit-learn to draw a page
  nobody has clicked on yet is a second of cold start for nothing.

Every job appends a record to the workspace run log, so what produced a result
is recoverable afterwards rather than living only in the terminal that ran it.
"""

from __future__ import annotations

from typing import Callable


def generate_dataset(progress: Callable, workspace: str, duration: float = 3600.0,
                     magnitudes: int = 3, repeats: int = 2,
                     normal_runs: int = 10) -> dict:
    """Run the scenario grid and write the labelled Parquet file."""
    from pyplant.ml.dataset import build, default_dataset, save
    from pyplant.workbench.workspace import Workspace

    ws = Workspace(workspace)
    scenarios = default_dataset(duration, magnitudes, repeats, normal_runs)
    record = ws.start_run("dataset", config={
        "duration": duration, "magnitudes": magnitudes, "repeats": repeats,
        "normal_runs": normal_runs, "scenarios": len(scenarios)})

    progress(0.0, f"0 of {len(scenarios)} runs")
    frames = []
    from pyplant.faults.scenarios import run_scenario
    import pandas as pd

    try:
        for i, scenario in enumerate(scenarios, 1):
            result = run_scenario(scenario)
            frame = result.frame().reset_index()
            frame["fault_kind"] = (scenario.faults[0].kind if scenario.faults
                                   else "normal")
            frame["fault_target"] = (scenario.faults[0].target
                                     if scenario.faults else "")
            frame["onset"] = (scenario.faults[0].t_start if scenario.faults
                              else float("nan"))
            frame["ramp"] = (scenario.faults[0].ramp if scenario.faults else 0.0)
            frames.append(frame)
            progress(i / len(scenarios), f"{i} of {len(scenarios)} runs")

        data = pd.concat(frames, ignore_index=True)
        save(data, ws.dataset_path)
        result = {"runs": len(scenarios), "samples": int(len(data)),
                  "path": str(ws.dataset_path),
                  "bytes": ws.dataset_path.stat().st_size}
        ws.finish_run(record, result)
        progress(1.0, f"{len(data):,} samples")
        return result
    except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
        ws.finish_run(record, {"error": str(exc)}, status="failed")
        raise


def run_default_scenario(name: str = "leak_abrupt", duration: float = 1800.0,
                         setpoint_moves: bool = False) -> dict:
    """One library scenario through the fast runner, in plant time.

    Needs no running stack, so it is the first thing a new user can do. Takes
    a few seconds, which is short enough to run inside a callback.

    The setpoint is held by default: a planned move landing after the onset
    would otherwise be read as part of what the fault did.
    """
    from dataclasses import replace

    from pyplant.faults.scenarios import DEFAULT_SP, library, run_scenario

    scenarios = {s.name: s for s in library(duration)}
    if name not in scenarios:
        raise ValueError(f"unknown scenario {name!r}")
    scenario = scenarios[name]
    if not setpoint_moves:
        scenario = replace(scenario, setpoints=[(0.0, DEFAULT_SP)])
    result = run_scenario(scenario)
    frame = result.frame()
    events = [e for e in result.events.to_dict("records") if e.get("kind")
              != "normal"]
    tags = ["LT101", "LIC101_SP", "LIC101_OP", "ZT101", "LT102", "FT101",
            "FT102", "PT101", "ILK_PMP101"]
    return {
        "name": name,
        "description": scenario.description,
        "duration": duration,
        "t": [float(t) for t in result.trajectory.t],
        "series": {tag: frame[tag].astype(float).round(3).tolist()
                   for tag in tags if tag in frame},
        "onset": float(events[0]["t_start"]) if events else None,
        "events": [{k: e.get(k) for k in ("fault_id", "kind", "target",
                                           "t_start", "ramp", "magnitude")}
                   for e in events],
    }


def train_detector(progress: Callable, workspace: str, window: int = 60,
                   stride: int = 10, quantile: float = 0.995,
                   seed: int = 0, methods: list[str] | None = None) -> dict:
    """Fit the chosen detectors, measure them, and keep the best one."""
    from pyplant.ml.dataset import load, split_scenarios, windowed
    from pyplant.ml.detector import (
        METHODS,
        Detector,
        make_detector,
        write_manifest,
    )
    from pyplant.ml.features import WindowSpec
    from pyplant.ml.train import fit_and_evaluate
    from pyplant.workbench.workspace import Workspace

    ws = Workspace(workspace)
    if not ws.dataset_path.exists():
        raise FileNotFoundError(
            "there is no dataset yet; generate one on the Data tab first")

    methods = [m for m in (methods or list(METHODS)) if m in METHODS]
    if not methods:
        raise ValueError("choose at least one detection method")

    record = ws.start_run("training", config={
        "window": window, "stride": stride, "quantile": quantile, "seed": seed,
        "methods": methods})
    try:
        progress(0.05, "loading the dataset")
        frame = load(ws.dataset_path)
        data = windowed(frame, WindowSpec(window=window, stride=stride))
        split = split_scenarios(frame, seed=seed)
        progress(0.15, f"{len(data):,} windows, {len(split.train)} training runs")

        results: dict[str, dict] = {}
        detectors = {m: make_detector(m, quantile=quantile, seed=seed)
                     for m in methods}
        for i, (name, detector) in enumerate(detectors.items()):
            progress(0.2 + 0.7 * i / len(detectors), f"fitting {name}")
            fitted, report = fit_and_evaluate(data, split, detector)
            fitted.save(ws.artifacts / f"detector_{name}.joblib")
            report.per_run.to_csv(ws.artifacts / f"per_run_{name}.csv",
                                  index=False)
            results[name] = {
                "detection_rate": report.detection_rate,
                "median_delay_s": report.median_delay,
                "false_alarms_per_day": report.false_alarms_per_day,
                "window_flag_rate": report.normal_window_flag_rate,
                "isolation": report.isolation_accuracy,
                "consistency": report.isolation_consistency,
                "per_mode": report.per_mode.to_dict(orient="records"),
                "manifest": fitted.manifest(),
            }

        # More detections wins; a tie goes to fewer false alarms, and then to
        # the simpler model. A learned detector that cannot beat PCA should not
        # be the one deployed just because it is newer.
        progress(0.95, "selecting")
        best = max(results, key=lambda k: (results[k]["detection_rate"],
                                           -results[k]["false_alarms_per_day"]))
        chosen = Detector.load(ws.artifacts / f"detector_{best}.joblib")
        chosen.save(ws.detector_path)
        write_manifest(chosen, ws.artifacts / "manifest.json",
                       extra={"selected": best, "window": window,
                              "stride": stride, "seed": seed})

        import json

        summary = {"selected": best, "results": results}
        (ws.artifacts / "metrics.json").write_text(
            json.dumps(summary, indent=2, default=str))
        ws.finish_run(record, {"selected": best,
                               "detection_rate": results[best]["detection_rate"]})
        progress(1.0, f"selected {best}")
        return summary
    except Exception as exc:  # noqa: BLE001
        ws.finish_run(record, {"error": str(exc)}, status="failed")
        raise
