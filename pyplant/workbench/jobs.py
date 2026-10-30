"""Background jobs.

Generating a dataset takes about four minutes. A Dash callback that blocks for
four minutes shows the user a spinner, then a timeout, then nothing. So the
long work runs in a separate process and reports progress through a file that
the UI polls.

A file rather than shared memory or a queue, deliberately. It survives the
workbench being restarted mid-job, it is trivially inspectable when something
goes wrong, and it means the polling callback does one cheap read with no
locking and no connection to the worker at all. The cost is a few milliseconds
of staleness, which nobody notices on a four-minute job.

Processes rather than threads: dataset generation is a Python loop calling a
Python integrator, so the GIL would make a thread pool no faster than doing it
inline. Nothing in this module imports Dash.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

TERMINAL = (STATUS_DONE, STATUS_FAILED, STATUS_CANCELLED)


@dataclass
class JobState:
    """What the UI polls. Everything needed to draw a progress bar."""

    id: str
    kind: str
    status: str = STATUS_PENDING
    progress: float = 0.0  # 0..1
    message: str = ""
    started: float = field(default_factory=time.time)
    finished: float | None = None
    result: dict = field(default_factory=dict)
    error: str = ""
    pid: int | None = None

    @property
    def seconds(self) -> float:
        return (self.finished or time.time()) - self.started

    @property
    def running(self) -> bool:
        return self.status in (STATUS_PENDING, STATUS_RUNNING)

    @property
    def eta(self) -> float | None:
        """Rough seconds remaining, from elapsed time and fraction done."""
        if not self.running or self.progress <= 0.01:
            return None
        return self.seconds * (1.0 - self.progress) / self.progress


class Progress:
    """Handed to the worker. Its only way to say how it is getting on."""

    def __init__(self, path: Path, state: JobState) -> None:
        self._path = path
        self._state = state
        self._last_write = 0.0

    def __call__(self, fraction: float, message: str = "") -> None:
        self._state.progress = min(max(float(fraction), 0.0), 1.0)
        if message:
            self._state.message = message
        # Throttle: a scenario loop calling this hundreds of times should not
        # turn into hundreds of file writes.
        now = time.time()
        if now - self._last_write > 0.25 or self._state.progress >= 1.0:
            self._last_write = now
            self.flush()

    def flush(self) -> None:
        _write_state(self._path, self._state)


def _write_state(path: Path, state: JobState) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state), default=str))
    # Atomic, so a poll never reads half a file. On Windows the replace fails
    # while a reader has the target open, so it is retried briefly.
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.01)


def read_state(path: Path) -> JobState | None:
    for attempt in range(20):
        try:
            return JobState(**json.loads(Path(path).read_text()))
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return None
        except PermissionError:
            # Windows: the writer is mid-replace. It holds the file for
            # microseconds, so try again rather than report no state.
            if attempt == 19:
                return None
            time.sleep(0.01)
    return None


def _entry(path: str, state_dict: dict, target: Callable, kwargs: dict) -> None:
    """Runs inside the worker process."""
    path_obj = Path(path)
    state = JobState(**state_dict)
    state.status = STATUS_RUNNING
    state.pid = os.getpid()
    _write_state(path_obj, state)
    progress = Progress(path_obj, state)
    try:
        result = target(progress=progress, **kwargs)
        state.status, state.result = STATUS_DONE, (result or {})
        state.progress = 1.0
        state.message = state.message or "finished"
    except Exception:  # noqa: BLE001 - the traceback is the useful part
        state.status = STATUS_FAILED
        state.error = traceback.format_exc(limit=8)
        state.message = "failed"
    finally:
        state.finished = time.time()
        _write_state(path_obj, state)


@dataclass
class JobRunner:
    """Starts jobs, tracks them by id, and reads their state back.

    One job of a given kind at a time. Two dataset generations writing the same
    parquet file is not a race worth being clever about; it is a mistake worth
    refusing.
    """

    directory: Path
    _processes: dict[str, mp.Process] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, job_id: str) -> Path:
        return self.directory / f"{job_id}.json"

    def submit(self, kind: str, target: Callable, **kwargs: Any) -> JobState:
        active = self.active(kind)
        if active is not None:
            raise RuntimeError(
                f"a {kind} job is already running (id {active.id}); "
                f"wait for it or cancel it")
        job_id = f"{kind}_{uuid.uuid4().hex[:8]}"
        state = JobState(id=job_id, kind=kind, message="starting")
        path = self.path_for(job_id)
        _write_state(path, state)
        process = mp.get_context("spawn").Process(
            target=_entry, args=(str(path), asdict(state), target, kwargs),
            daemon=True)
        process.start()
        self._processes[job_id] = process
        return state

    def state(self, job_id: str) -> JobState | None:
        state = read_state(self.path_for(job_id))
        if state is None:
            return None
        # A worker killed outright never gets to write a final state, so the
        # file would say "running" forever. Reconcile against the process.
        process = self._processes.get(job_id)
        if state.running and process is not None and not process.is_alive():
            state.status = STATUS_FAILED
            state.error = state.error or "the worker exited without finishing"
            state.finished = state.finished or time.time()
            _write_state(self.path_for(job_id), state)
        return state

    def all(self, kind: str | None = None) -> list[JobState]:
        states = [read_state(p) for p in sorted(self.directory.glob("*.json"))]
        found = [self.state(s.id) or s for s in states if s is not None]
        if kind is not None:
            found = [s for s in found if s.kind == kind]
        return sorted(found, key=lambda s: s.started, reverse=True)

    def active(self, kind: str | None = None) -> JobState | None:
        for state in self.all(kind):
            if state.running:
                return state
        return None

    def cancel(self, job_id: str) -> bool:
        process = self._processes.get(job_id)
        if process is None or not process.is_alive():
            return False
        process.terminate()
        process.join(timeout=5.0)
        state = read_state(self.path_for(job_id)) or JobState(id=job_id,
                                                              kind="unknown")
        state.status = STATUS_CANCELLED
        state.finished = time.time()
        state.message = "cancelled"
        _write_state(self.path_for(job_id), state)
        return True

    def wait(self, job_id: str, timeout: float = 300.0,
             poll: float = 0.1) -> JobState | None:
        """Block until a job finishes. For scripts and tests, never the UI."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = self.state(job_id)
            if state is not None and not state.running:
                return state
            time.sleep(poll)
        return self.state(job_id)

    def clear_finished(self) -> int:
        removed = 0
        for state in self.all():
            if not state.running:
                self.path_for(state.id).unlink(missing_ok=True)
                self._processes.pop(state.id, None)
                removed += 1
        return removed
