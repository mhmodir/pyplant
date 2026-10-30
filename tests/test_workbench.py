"""Tests for the workbench foundation.

Nothing here imports Dash. The workspace, the job runner and the supervisor are
the parts that are hard to get right and miserable to debug through a browser,
so they are built and tested as plain Python first. The UI on top of them is
allowed to be thin precisely because these are not.
"""

from __future__ import annotations

import json
import time

import httpx
import pytest

from pyplant.workbench.jobs import (
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_FAILED,
    JobRunner,
    JobState,
    read_state,
)
from pyplant.workbench.supervisor import SERVICES, Supervisor, port_is_open
from pyplant.workbench.workspace import (
    Config,
    DataConfig,
    LoopConfig,
    PlantConfig,
    Ports,
    Workspace,
    discover,
)

# Module-level so the spawn context can pickle them.


def _counts_to(progress, n: int = 5, sleep: float = 0.02) -> dict:
    for i in range(n):
        progress((i + 1) / n, f"step {i + 1} of {n}")
        time.sleep(sleep)
    return {"counted": n}


def _explodes(progress) -> dict:
    progress(0.5, "about to fail")
    raise ValueError("a deliberate failure")


def _sleeps(progress, seconds: float = 30.0) -> dict:
    progress(0.01, "sleeping")
    time.sleep(seconds)
    return {}


# ============================== workspace ==================================

def test_a_new_workspace_writes_a_config(tmp_path):
    ws = Workspace(tmp_path / "w")
    assert ws.config_path.exists()
    assert ws.artifacts.is_dir() and ws.jobs.is_dir()
    assert ws.config.name == "w"


def test_config_survives_reopening(tmp_path):
    ws = Workspace(tmp_path / "w")
    ws.update(loop=LoopConfig(kc=0.9, ti=40.0, setpoint=250.0))
    again = Workspace(tmp_path / "w")
    assert again.config.loop.kc == 0.9
    assert again.config.loop.setpoint == 250.0


def test_config_round_trips_through_json(tmp_path):
    ws = Workspace(tmp_path / "w")
    ws.update(plant=PlantConfig(area_tank1=0.033),
              data=DataConfig(duration=1200.0, repeats=4))
    raw = json.loads(ws.config_path.read_text())
    restored = Config.from_dict(raw)
    assert restored.plant.area_tank1 == 0.033
    assert restored.data.repeats == 4
    assert restored.plant.to_params().area_tank1 == 0.033


def test_a_half_written_config_cannot_be_left_behind(tmp_path):
    # The save writes a temporary file and renames it, so a workbench killed
    # mid-save leaves either the old config or the new one, never a broken one.
    ws = Workspace(tmp_path / "w")
    ws.update(loop=LoopConfig(kc=0.5))
    assert not list(tmp_path.glob("w/*.tmp"))
    assert json.loads(ws.config_path.read_text())["loop"]["kc"] == 0.5


def test_ports_are_allocated_in_a_block(tmp_path):
    ws = Workspace(tmp_path / "w", base_port=5600)
    ports = ws.config.ports.as_dict()
    assert ports["plant_modbus"] == 5600
    assert ports["hmi"] == 5605
    assert len(set(ports.values())) == len(ports)  # no collisions


def test_a_base_port_survives_a_reopen_and_can_be_changed(tmp_path):
    Workspace(tmp_path / "w", base_port=5600)
    assert Workspace(tmp_path / "w").config.ports.plant_modbus == 5600
    assert Workspace(tmp_path / "w", base_port=5700).config.ports.plant_modbus \
        == 5700


def test_ports_report_what_is_already_taken():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        taken = s.getsockname()[1]
        assert "hmi" in Ports(hmi=taken).in_use()


def test_run_log_collapses_a_run_to_its_latest_state(tmp_path):
    ws = Workspace(tmp_path / "w")
    record = ws.start_run("dataset", config={"duration": 600})
    assert ws.runs()[0].status == "running"
    ws.finish_run(record, result={"rows": 1234})
    runs = ws.runs()
    assert len(runs) == 1  # one run, not two lines
    assert runs[0].status == "ok" and runs[0].result["rows"] == 1234
    assert runs[0].config["duration"] == 600


def test_run_log_filters_by_kind_and_orders_newest_first(tmp_path):
    ws = Workspace(tmp_path / "w")
    ws.finish_run(ws.start_run("tuning"))
    time.sleep(0.01)
    ws.finish_run(ws.start_run("dataset"))
    assert ws.runs()[0].kind == "dataset"
    assert [r.kind for r in ws.runs(kind="tuning")] == ["tuning"]


def test_status_reports_what_the_step_gates_need(tmp_path):
    ws = Workspace(tmp_path / "w")
    assert ws.status()["has_dataset"] is False
    assert ws.status()["has_detector"] is False
    ws.dataset_path.write_bytes(b"not really parquet")
    ws.detector_path.write_bytes(b"not really a model")
    status = ws.status()
    assert status["has_dataset"] and status["has_detector"]
    assert status["dataset_bytes"] > 0


def test_reset_clears_results_but_keeps_the_configuration(tmp_path):
    ws = Workspace(tmp_path / "w")
    ws.update(loop=LoopConfig(kc=0.77))
    ws.dataset_path.write_bytes(b"x")
    ws.detector_path.write_bytes(b"x")
    ws.finish_run(ws.start_run("dataset"))
    ws.reset()
    assert not ws.dataset_path.exists() and not ws.detector_path.exists()
    assert ws.runs() == []
    assert ws.artifacts.is_dir()  # still usable, not just deleted
    assert Workspace(tmp_path / "w").config.loop.kc == 0.77


def test_discover_finds_only_real_workspaces(tmp_path):
    Workspace(tmp_path / "alice")
    Workspace(tmp_path / "bob")
    (tmp_path / "not_a_workspace").mkdir()
    assert sorted(w.path.name for w in discover(tmp_path)) == ["alice", "bob"]


# ============================== job runner =================================

def test_a_job_runs_and_reports_its_result(tmp_path):
    runner = JobRunner(tmp_path / "jobs")
    job = runner.submit("count", _counts_to, n=4)
    final = runner.wait(job.id, timeout=60)
    assert final.status == STATUS_DONE
    assert final.result == {"counted": 4}
    assert final.progress == 1.0
    assert final.seconds > 0


def test_progress_is_visible_while_the_job_is_still_running(tmp_path):
    # The whole reason this module exists: a four-minute job must be able to
    # say how it is getting on without blocking anything.
    runner = JobRunner(tmp_path / "jobs")
    job = runner.submit("count", _counts_to, n=20, sleep=0.05)
    seen = []
    deadline = time.time() + 30
    while time.time() < deadline:
        state = runner.state(job.id)
        if state:
            seen.append(state.progress)
            if not state.running:
                break
        time.sleep(0.05)
    assert max(seen) == 1.0
    assert any(0.0 < p < 1.0 for p in seen)  # observed mid-flight, not just at the end


def test_a_failing_job_keeps_its_traceback(tmp_path):
    runner = JobRunner(tmp_path / "jobs")
    job = runner.submit("boom", _explodes)
    final = runner.wait(job.id, timeout=60)
    assert final.status == STATUS_FAILED
    assert "a deliberate failure" in final.error
    assert "ValueError" in final.error


def test_only_one_job_of_a_kind_at_a_time(tmp_path):
    runner = JobRunner(tmp_path / "jobs")
    runner.submit("sleep", _sleeps, seconds=10.0)
    with pytest.raises(RuntimeError, match="already running"):
        runner.submit("sleep", _sleeps, seconds=1.0)
    # A different kind is fine: generating data and tuning are unrelated.
    runner.submit("count", _counts_to, n=2)
    runner.cancel(runner.active("sleep").id)


def test_cancelling_stops_the_worker(tmp_path):
    runner = JobRunner(tmp_path / "jobs")
    job = runner.submit("sleep", _sleeps, seconds=60.0)
    time.sleep(0.5)
    assert runner.cancel(job.id) is True
    assert runner.state(job.id).status == STATUS_CANCELLED
    assert runner.active("sleep") is None
    assert runner.cancel(job.id) is False  # already gone


def test_a_worker_killed_outright_is_not_left_running_forever(tmp_path):
    # Without reconciliation the status file would say "running" for ever,
    # and the UI would refuse to start another job on a dead one.
    runner = JobRunner(tmp_path / "jobs")
    job = runner.submit("sleep", _sleeps, seconds=60.0)
    time.sleep(0.5)
    runner._processes[job.id].kill()
    runner._processes[job.id].join(timeout=5)
    state = runner.state(job.id)
    assert state.status == STATUS_FAILED
    assert "without finishing" in state.error


def test_states_are_written_atomically(tmp_path):
    runner = JobRunner(tmp_path / "jobs")
    job = runner.submit("count", _counts_to, n=3)
    runner.wait(job.id, timeout=60)
    assert not list((tmp_path / "jobs").glob("*.tmp"))
    assert read_state(runner.path_for(job.id)).status == STATUS_DONE


def test_history_is_listed_and_can_be_cleared(tmp_path):
    runner = JobRunner(tmp_path / "jobs")
    for _ in range(2):
        runner.wait(runner.submit("count", _counts_to, n=2).id, timeout=60)
    assert len(runner.all("count")) == 2
    assert runner.clear_finished() == 2
    assert runner.all() == []


def test_eta_is_only_offered_once_there_is_something_to_estimate_from():
    running = JobState(id="x", kind="k", status="running", progress=0.0)
    assert running.eta is None
    running.progress = 0.5
    running.started = time.time() - 10.0
    assert running.eta == pytest.approx(10.0, abs=1.0)
    assert JobState(id="x", kind="k", status=STATUS_DONE).eta is None


# ============================== supervisor =================================

def test_command_lines_use_the_workspace_ports(tmp_path):
    ws = Workspace(tmp_path / "w", base_port=5800)
    sup = Supervisor(ws)
    plant = " ".join(sup.command("plant"))
    plc = " ".join(sup.command("plc"))
    hmi = " ".join(sup.command("hmi"))
    assert "--port 5800" in plant and "--control-port 5801" in plant
    assert "--plant-port 5800" in plc and "--opcua-port 5803" in plc
    assert "--history http://127.0.0.1:5804" in hmi
    assert "--port 5805" in hmi


def test_the_command_line_is_the_one_a_user_could_type(tmp_path):
    # The Runtime page shows these, so they have to be honest.
    sup = Supervisor(Workspace(tmp_path / "w", base_port=5800))
    for spec in SERVICES:
        command = sup.command(spec.name)
        assert command[1:3] == ["-m", spec.module]
        assert spec.port_attrs


def test_services_are_declared_in_dependency_order():
    assert [s.name for s in SERVICES] == ["plant", "plc", "historian", "hmi"]


def test_nothing_is_running_before_it_is_started(tmp_path):
    sup = Supervisor(Workspace(tmp_path / "w", base_port=5800))
    assert not sup.healthy()
    assert all(not s.running for s in sup.states())


def test_a_port_held_by_something_else_is_refused(tmp_path):
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        base = s.getsockname()[1]
        ws = Workspace(tmp_path / "w")
        ws.update(ports=Ports.from_base(base))
        with pytest.raises(RuntimeError, match="already in use"):
            Supervisor(ws).start()


@pytest.mark.slow
def test_the_whole_stack_starts_serves_and_stops(tmp_path):
    ws = Workspace(tmp_path / "w", base_port=5900)
    with Supervisor(ws) as sup:
        states = sup.start()
        assert len(states) == 4
        assert sup.healthy()

        time.sleep(3.0)
        values = httpx.get(f"{sup.hmi_url}/api/state", timeout=5).json()["values"]
        assert values["LT101"] == pytest.approx(300.0, abs=3.0)
        assert values["SCAN_MS"] > 0

        health = httpx.get(f"{sup.history_url}/api/health", timeout=5).json()
        assert health["rows"] > 0

        # A fault injected on the plant reaches the operator page.
        httpx.post(f"{sup.plant_control_url}/api/fault", timeout=5,
                   json={"kind": "leak", "id": "L1", "magnitude": 5e-5})
        time.sleep(6.0)
        after = httpx.get(f"{sup.hmi_url}/api/state", timeout=5).json()["values"]
        assert after["LIC101_OP"] > values["LIC101_OP"] + 3.0

    assert not any(port_is_open(p) for p in ws.config.ports.as_dict().values())
