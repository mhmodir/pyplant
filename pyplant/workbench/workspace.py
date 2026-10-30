"""The workspace.

A workbench session owns a directory. Everything it produces goes there, and
nothing it produces goes anywhere else: configuration, the generated dataset,
trained artefacts, the historian database, and a log of every run.

This exists because "the user does the steps themselves" needs somewhere to put
the results between sessions, and because three audiences want three different
things from the same directory. A student needs to be able to delete it and
start over without breaking anything. A demo needs it to survive a restart so
the stack comes back with the same ports and the same trained model. Research
needs every result to carry the configuration that produced it, or the numbers
in it are anecdotes.

Nothing here imports Dash. The workbench UI is one client of this; the tests
are another.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator

from pyplant.core.hybrid_tank import Params
from pyplant.plc.pid import PidTuning

CONFIG_NAME = "config.json"
RUNS_NAME = "runs.jsonl"
DATASET_NAME = "dataset.parquet"
HISTORY_NAME = "history.duckdb"
ARTIFACTS_DIR = "artifacts"
JOBS_DIR = "jobs"


@dataclass
class Ports:
    """One stack's listening ports.

    Allocated once and stored, so restarting the workbench reattaches to the
    same addresses instead of scattering a new set. Several students on one
    machine get non-overlapping blocks by passing different `base` values.
    """

    plant_modbus: int = 5020
    plant_control: int = 8000
    plc_modbus: int = 5021
    plc_opcua: int = 4840
    historian_api: int = 8010
    hmi: int = 8080

    @staticmethod
    def from_base(base: int) -> "Ports":
        return Ports(plant_modbus=base, plant_control=base + 1,
                     plc_modbus=base + 2, plc_opcua=base + 3,
                     historian_api=base + 4, hmi=base + 5)

    def as_dict(self) -> dict[str, int]:
        return asdict(self)

    def in_use(self) -> list[str]:
        """Which of these already have a listener. Checked before starting."""
        busy = []
        for name, port in self.as_dict().items():
            with socket.socket() as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind(("127.0.0.1", port))
                except OSError:
                    busy.append(name)
        return busy


@dataclass
class PlantConfig:
    """Everything the user can change about the plant and its controller."""

    area_tank1: float = 0.020
    area_tank2: float = 0.020
    height_max: float = 0.600
    pump_dp0: float = 60_000.0
    pump_qmax: float = 4.17e-4
    valve_cv: float = 2.429e-6
    valve_rangeability: float = 30.0
    tau_valve: float = 2.0
    cda_12: float = 6.90e-5
    cda_2out: float = 9.00e-5

    def to_params(self) -> Params:
        return Params(**asdict(self))

    @staticmethod
    def from_params(p: Params) -> "PlantConfig":
        keep = set(PlantConfig().__dataclass_fields__)
        return PlantConfig(**{k: v for k, v in vars(p).items() if k in keep})


@dataclass
class LoopConfig:
    kc: float = 0.35
    ti: float = 72.0
    td: float = 0.0
    tf: float = 5.0
    setpoint: float = 300.0
    scan_period: float = 0.10

    def to_tuning(self) -> PidTuning:
        return PidTuning(kc=self.kc, ti=self.ti, td=self.td, tf=self.tf)


@dataclass
class DataConfig:
    duration: float = 3600.0
    magnitudes: int = 3
    repeats: int = 2
    normal_runs: int = 10
    window: int = 60
    stride: int = 10
    quantile: float = 0.995
    seed: int = 0


@dataclass
class Config:
    """The whole workspace configuration, saved as one JSON file."""

    name: str = "workspace"
    speed: float = 1.0
    dt: float = 0.05
    ports: Ports = field(default_factory=Ports)
    plant: PlantConfig = field(default_factory=PlantConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "Config":
        return Config(
            name=raw.get("name", "workspace"),
            speed=float(raw.get("speed", 1.0)),
            dt=float(raw.get("dt", 0.05)),
            ports=Ports(**raw.get("ports", {})),
            plant=PlantConfig(**raw.get("plant", {})),
            loop=LoopConfig(**raw.get("loop", {})),
            data=DataConfig(**raw.get("data", {})),
        )


@dataclass
class RunRecord:
    """One thing the user did, with what it was given and what came back.

    Appended, never rewritten. A result whose configuration was not recorded
    beside it cannot be reproduced or argued with.
    """

    id: str
    kind: str  # step_test, tuning, dataset, training, fault, stack
    started: float
    finished: float | None = None
    status: str = "running"  # running, ok, failed
    config: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    note: str = ""

    @property
    def seconds(self) -> float:
        return (self.finished or time.time()) - self.started


class Workspace:
    """A directory plus the config and history inside it."""

    def __init__(self, path: str | Path, base_port: int | None = None) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / ARTIFACTS_DIR).mkdir(exist_ok=True)
        (self.path / JOBS_DIR).mkdir(exist_ok=True)
        self.config = self._load_config(base_port)

    # -- paths -------------------------------------------------------------

    @property
    def config_path(self) -> Path:
        return self.path / CONFIG_NAME

    @property
    def runs_path(self) -> Path:
        return self.path / RUNS_NAME

    @property
    def dataset_path(self) -> Path:
        return self.path / DATASET_NAME

    @property
    def history_path(self) -> Path:
        return self.path / HISTORY_NAME

    @property
    def artifacts(self) -> Path:
        return self.path / ARTIFACTS_DIR

    @property
    def jobs(self) -> Path:
        return self.path / JOBS_DIR

    @property
    def detector_path(self) -> Path:
        return self.artifacts / "detector.joblib"

    # -- config ------------------------------------------------------------

    def _load_config(self, base_port: int | None) -> Config:
        if self.config_path.exists():
            config = Config.from_dict(json.loads(self.config_path.read_text()))
            if base_port is not None:
                config = replace(config, ports=Ports.from_base(base_port))
                self._write_config(config)
            return config
        config = Config(name=self.path.name)
        if base_port is not None:
            config = replace(config, ports=Ports.from_base(base_port))
        self._write_config(config)
        return config

    def _write_config(self, config: Config) -> None:
        # Write to a temporary file and move it into place: a workbench killed
        # mid-save must not leave a workspace that cannot be reopened.
        tmp = self.config_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(config.to_dict(), indent=2))
        os.replace(tmp, self.config_path)

    def save(self, config: Config | None = None) -> Config:
        if config is not None:
            self.config = config
        self._write_config(self.config)
        return self.config

    def update(self, **sections: Any) -> Config:
        """Replace whole sections, e.g. `update(loop=LoopConfig(kc=0.5))`."""
        self.config = replace(self.config, **sections)
        self._write_config(self.config)
        return self.config

    # -- run log -----------------------------------------------------------

    def start_run(self, kind: str, config: dict | None = None,
                  note: str = "") -> RunRecord:
        record = RunRecord(id=uuid.uuid4().hex[:12], kind=kind,
                           started=time.time(), config=config or {}, note=note)
        self._append(record)
        return record

    def finish_run(self, record: RunRecord, result: dict | None = None,
                   status: str = "ok") -> RunRecord:
        record.finished = time.time()
        record.status = status
        record.result = result or {}
        self._append(record)
        return record

    def _append(self, record: RunRecord) -> None:
        with self.runs_path.open("a") as handle:
            handle.write(json.dumps(asdict(record), default=str) + "\n")

    def runs(self, kind: str | None = None, limit: int = 50) -> list[RunRecord]:
        """Most recent first, with each run collapsed to its latest state."""
        if not self.runs_path.exists():
            return []
        latest: dict[str, RunRecord] = {}
        for line in self.runs_path.read_text().splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            latest[raw["id"]] = RunRecord(**raw)
        records = sorted(latest.values(), key=lambda r: r.started, reverse=True)
        if kind is not None:
            records = [r for r in records if r.kind == kind]
        return records[:limit]

    # -- state the UI gates on --------------------------------------------

    def status(self) -> dict[str, Any]:
        """What exists so far. The left rail's step gates read this."""
        dataset = self.dataset_path.exists()
        detector = self.detector_path.exists()
        metrics = self.artifacts / "metrics.json"
        return {
            "name": self.config.name,
            "path": str(self.path),
            "has_dataset": dataset,
            "dataset_bytes": self.dataset_path.stat().st_size if dataset else 0,
            "has_detector": detector,
            "has_metrics": metrics.exists(),
            "has_history": self.history_path.exists(),
            "runs": len(self.runs(limit=10_000)),
            "ports": self.config.ports.as_dict(),
        }

    def reset(self, keep_config: bool = True) -> None:
        """Throw away results. A student must be able to start over safely."""
        config = self.config
        for target in (self.dataset_path, self.history_path, self.runs_path):
            target.unlink(missing_ok=True)
        shutil.rmtree(self.artifacts, ignore_errors=True)
        shutil.rmtree(self.jobs, ignore_errors=True)
        self.artifacts.mkdir(exist_ok=True)
        self.jobs.mkdir(exist_ok=True)
        if keep_config:
            self.save(config)
        else:
            self.config_path.unlink(missing_ok=True)
            self.config = self._load_config(None)


def discover(root: str | Path) -> Iterator[Workspace]:
    """Every workspace under a directory, for a workspace picker."""
    for candidate in sorted(Path(root).expanduser().glob("*")):
        if (candidate / CONFIG_NAME).exists():
            yield Workspace(candidate)
