"""Tests for the workbench application.

Dash callbacks are tested through a browser or not at all, so the rule here is
that nothing worth testing lives inside one. The rendering helpers are plain
functions over plain data and are tested directly; the callbacks that use them
are thin enough to read.

One slow test boots the app and a real stack together, because the thing most
likely to break is not the rendering but the wiring: a workbench that cannot
reach the services it started is the failure that matters.
"""

from __future__ import annotations

import threading
import time

import httpx
import pytest

from pyplant.faults.model import FAULT_KINDS
from pyplant.faults.scenarios import library
from pyplant.workbench.app import (
    DATA_PRESETS,
    DEFAULT_MAGNITUDE,
    DEFAULT_SCENARIO,
    LIVE_DEMO_FAULT,
    PANELS,
    PRESETS,
    MAGNITUDE_UNITS,
    SCENARIO_CHOICES,
    SENSOR_KINDS,
    TREND_TAGS,
    WORKFLOW,
    build_app,
    scenario_figure,
    scenario_summary,
    workflow_steps,
    fmt,
    metrics_rows,
    per_mode_rows,
    table,
    tank,
    clicked,
    mimic,
    trend_figure,
)
from pyplant.faults.model import build_fault
from pyplant.workbench.clients import (
    StackClients,
    StackUnavailable,
    forget_failures,
)
from pyplant.workbench.tasks import run_default_scenario
from pyplant.workbench.workspace import Ports, Workspace


# ============================== helpers ====================================

def test_fmt_survives_missing_values():
    assert fmt(300.25) == "300.2"
    assert fmt(None) == "--"
    assert fmt("") == "--"
    assert fmt(10.0, 2) == "10.00"


def test_tank_fill_is_clamped_to_the_vessel():
    def height(mm):
        return tank("T", mm).children[0].children.style["height"]

    assert height(0) == "0.0%"
    assert height(300) == "50.0%"
    assert height(9999) == "100.0%"  # a pegged transmitter does not overflow
    assert height(None) == "0.0%"


def test_table_says_so_when_there_is_nothing():
    assert "No alarms" in table(["a"], [], "No alarms").children
    rendered = table(["a", "b"], [[1, 2]])
    assert len(rendered.children[1].children) == 1


def test_controller_output_gets_its_own_axis():
    # Level in millimetres and output in percent on one axis would make the
    # output a flat line at the bottom of the chart.
    figure = trend_figure({"t": [1, 2],
                           "series": {"LT101": [300, 301],
                                      "LIC101_OP": [66, 67]}})
    axes = {trace["name"]: trace["yaxis"] for trace in figure["data"]}
    assert axes["LT101"] == "y" and axes["LIC101_OP"] == "y2"
    assert figure["layout"]["yaxis2"]["range"] == [0, 100]


def test_an_empty_trend_still_produces_a_figure():
    figure = trend_figure({"t": [], "series": {}})
    assert figure["data"] == []


def test_trend_shows_setpoint_beside_measurement():
    assert "LT101" in TREND_TAGS and "LIC101_SP" in TREND_TAGS
    assert "LIC101_OP" in TREND_TAGS  # where a masked fault actually shows


def test_metrics_rows_mark_the_selected_model():
    metrics = {"selected": "autoencoder", "results": {
        "pca": {"detection_rate": 0.83, "median_delay_s": 256.0,
                "false_alarms_per_day": 0.0, "window_flag_rate": 0.002,
                "isolation": 0.6, "per_mode": []},
        "autoencoder": {"detection_rate": 1.0, "median_delay_s": 70.0,
                        "false_alarms_per_day": 0.0, "window_flag_rate": 0.0014,
                        "isolation": 0.73, "per_mode": []}}}
    rows = metrics_rows(metrics)
    assert len(rows) == 2
    assert any("selected" in str(row[0]) for row in rows)
    assert ["autoencoder (selected)", "100%", "70 s"] == rows[1][:3]


def test_per_mode_rows_name_the_channel_physics_expects():
    metrics = {"selected": "pca", "results": {"pca": {"per_mode": [
        {"kind": "leak", "runs": 6, "detected": 1.0, "median_delay": 40.0,
         "top": "FLOW_IMBALANCE"}]}}}
    row = per_mode_rows(metrics)[0]
    assert row[0] == "leak" and row[4] == "FLOW_IMBALANCE"
    assert "FLOW_IMBALANCE" in row[5]


def test_sensor_kinds_are_the_ones_that_need_a_tag():
    assert SENSOR_KINDS < set(FAULT_KINDS)
    for kind in FAULT_KINDS:
        assert kind in DEFAULT_MAGNITUDE, kind
        assert kind in MAGNITUDE_UNITS, kind
    # The only fault with no magnitude to set is the one that freezes a reading.
    assert [k for k, v in DEFAULT_MAGNITUDE.items() if v is None] \
        == ["sensor_stuck"]


def test_clicked_treats_an_untouched_group_as_untouched():
    # Dash fires pattern-matching callbacks once with every n_clicks at None,
    # and hands back a scalar when the group has one member.
    assert clicked(None) is False
    assert clicked([None, None]) is False
    assert clicked([]) is False
    assert clicked([None, 1]) is True
    assert clicked(1) is True  # scalar, not a list


def test_the_mimic_draws_with_or_without_a_running_plant():
    live = mimic({"LT101": 300.0, "LT102": 176.0, "FT101": 10.0,
                  "FT102": 10.0, "PT101": 50.3, "ZT101": 66.2,
                  "PMP101_RUN": True})
    assert len(live.children) > 20
    # A stack that is not running must still render the diagram, empty.
    assert len(mimic({}).children) == len(live.children)


def test_the_mimic_marks_a_trip():
    tripped = str(mimic({"ILK_PMP101": True}).children)
    assert "INTERLOCK TRIPPED" in tripped
    assert "INTERLOCK" not in str(mimic({"ILK_PMP101": False}).children)


def test_every_preset_builds_a_real_fault():
    for name, spec in PRESETS.items():
        fault = build_fault(id="preview", **spec)
        assert fault.kind == spec["kind"] or spec["kind"] == "stiction", name
        if "tag" in spec:
            assert fault.target == spec["tag"]


# ============================== clients ====================================

def test_clients_build_urls_from_the_workspace_ports():
    clients = StackClients(Ports.from_base(5400))
    assert clients.hmi.endswith(":5405")
    assert clients.plant.endswith(":5401")
    assert clients.history.endswith(":5404")


def test_an_absent_service_raises_rather_than_returning_nothing():
    # A page that silently blanks is worse than one that says the stack is down.
    clients = StackClients(Ports.from_base(1))
    assert clients.reachable() is False
    with pytest.raises(StackUnavailable):
        clients.state()


def test_a_service_that_just_refused_is_reported_down_at_once():
    # Windows takes about two seconds to refuse a local connection; a page
    # polling several endpoints a second would fall a minute behind.
    clients = StackClients(Ports.from_base(1))
    forget_failures()
    assert clients.reachable() is False
    started = time.perf_counter()
    for _ in range(20):
        assert clients.reachable() is False
    assert time.perf_counter() - started < 0.5
    forget_failures()


# ============================== the app ====================================

def test_the_app_builds_with_every_panel_present(tmp_path):
    app = build_app(str(tmp_path / "ws"), base_port=5700)
    ids = {c.id for c in app.layout.children if getattr(c, "id", None)}
    assert {f"panel-{name}" for name in PANELS} <= ids
    assert len(app.callback_map) > 8


def test_the_page_opens_on_quick_start_with_the_default_scenario(tmp_path):
    app = build_app(str(tmp_path / "ws"), base_port=5700)
    tabs = next(c for c in app.layout.children if getattr(c, "id", "") == "tabs")
    assert tabs.value == "quickstart"
    assert DEFAULT_SCENARIO in SCENARIO_CHOICES
    assert DEFAULT_SCENARIO in {s.name for s in library(600.0)}
    assert set(SCENARIO_CHOICES) == {s.name for s in library(600.0)}
    assert LIVE_DEMO_FAULT in PRESETS


def test_workflow_steps_mark_what_is_done():
    cards = workflow_steps({"scenario": True, "has_dataset": True})
    assert len(cards) == len(WORKFLOW)
    done = [c.className.endswith("done") for c in cards]
    assert done == [True, False, False, True, False]


def test_data_presets_are_splittable():
    # split_scenarios needs three normal runs to fit, calibrate and test.
    for name, preset in DATA_PRESETS.items():
        assert preset["normal_runs"] >= 3, name
        assert preset["duration"] >= 600.0, name


def test_scenario_figure_marks_the_onset_and_splits_the_axes():
    result = {"t": [0.0, 1.0, 2.0], "onset": 1.0,
              "series": {"LT101": [300, 300, 300], "LIC101_OP": [66, 70, 75],
                         "FT101": [10, 10, 10]}}
    figure = scenario_figure(result)
    axes = {trace["name"]: trace["yaxis"] for trace in figure["data"]}
    assert axes == {"LT101": "y", "LIC101_OP": "y2"}
    assert any(s["type"] == "line" and s["x0"] == 1.0
               for s in figure["layout"]["shapes"])
    assert scenario_figure({})["data"] == []


def test_scenario_summary_reports_the_shift_a_leak_causes():
    t = [float(i) for i in range(1200)]
    op = [66.0 if x < 600 else 76.0 for x in t]
    result = {"t": t, "onset": 600.0, "series": {
        "LT101": [300.0] * 1200, "LIC101_OP": op,
        "ILK_PMP101": [False] * 1200}}
    summary = {k: v for k, v, _ in scenario_summary(result)}
    assert summary["Controller OP change"] == "+10.00"
    assert summary["LT101 change"] == "+0.00"
    assert summary["Pump trip"] == "none"
    assert summary["Fault onset"] == "600"


@pytest.mark.slow
def test_the_default_scenario_runs_offline():
    result = run_default_scenario(DEFAULT_SCENARIO, 900.0)
    assert result["onset"] is not None
    assert len(result["t"]) == len(result["series"]["LT101"])
    summary = {k: v for k, v, _ in scenario_summary(result)}
    # The leak is held at setpoint; only the controller output moves.
    assert abs(float(summary["LT101 change"])) < 3.0
    assert float(summary["Controller OP change"]) > 3.0


def test_building_the_app_creates_the_workspace(tmp_path):
    build_app(str(tmp_path / "fresh"), base_port=5700)
    assert (tmp_path / "fresh" / "config.json").exists()
    assert Workspace(tmp_path / "fresh").config.ports.plant_modbus == 5700


@pytest.mark.slow
def test_the_page_serves_and_tabs_switch(tmp_path):
    app = build_app(str(tmp_path / "ws"), base_port=5710)
    threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=8061),
        daemon=True).start()
    for _ in range(60):
        try:
            index = httpx.get("http://127.0.0.1:8061/", timeout=2)
            break
        except httpx.HTTPError:
            time.sleep(0.2)
    else:  # pragma: no cover
        pytest.fail("the workbench never came up")

    assert index.status_code == 200

    response = httpx.post(
        "http://127.0.0.1:8061/_dash-update-component", timeout=10,
        json={"output": "..panel-quickstart.style...panel-runtime.style..."
                        "panel-faults.style...panel-data.style..."
                        "panel-model.style..",
              "outputs": [{"id": f"panel-{n}", "property": "style"}
                          for n in PANELS],
              "inputs": [{"id": "tabs", "property": "value", "value": "faults"}],
              "changedPropIds": ["tabs.value"]})
    assert response.status_code == 200
    styles = response.json()["response"]
    assert styles["panel-faults"]["style"]["display"] == "block"
    assert styles["panel-runtime"]["style"]["display"] == "none"
    assert styles["panel-quickstart"]["style"]["display"] == "none"


@pytest.mark.slow
def test_the_workbench_reaches_a_stack_it_started(tmp_path):
    # The wiring, not the rendering: the workbench must be able to talk to the
    # services it launched, with the ports it allocated.
    from pyplant.workbench.supervisor import Supervisor

    ws = Workspace(tmp_path / "ws", base_port=5720)
    clients = StackClients(ws.config.ports)
    with Supervisor(ws) as supervisor:
        supervisor.start()
        time.sleep(3.0)
        assert clients.reachable()

        values = clients.state()["values"]
        assert values["LT101"] == pytest.approx(300.0, abs=3.0)

        assert clients.faults()["faults"] == []
        injected = clients.inject("leak", magnitude=5e-5, fault_id="L1")
        assert injected["fault_id"] == "L1"
        assert clients.faults()["active"] == ["L1"]

        truth = clients.truth()
        assert truth["label"] == "L1"
        # A process fault: the instruments are honest, the plant is not.
        assert truth["true"]["LT101"] == pytest.approx(
            truth["reported"]["LT101"], abs=1.0)

        time.sleep(6.0)
        after = clients.state()["values"]
        assert after["LIC101_OP"] > values["LIC101_OP"] + 3.0

        assert clients.clear_faults()["removed"] == ["L1"]
        assert clients.history_health()["rows"] > 0


@pytest.mark.slow
def test_instrument_noise_is_not_reported_as_a_fault(tmp_path):
    # The live plant carries the same noise every scenario does, because a
    # perfectly quiet loop never moves its valve and stiction cannot manifest.
    # It must not appear in the operator's fault list on a healthy plant.
    from pyplant.workbench.supervisor import Supervisor

    ws = Workspace(tmp_path / "ws", base_port=5760)
    clients = StackClients(ws.config.ports)
    with Supervisor(ws) as supervisor:
        supervisor.start()
        time.sleep(3.0)
        listing = clients.faults()
        assert listing["faults"] == [] and listing["active"] == []
        assert len(listing["instruments"]) >= 3
        assert all(f["benign"] for f in listing["instruments"])

        clients.inject("leak", magnitude=5e-5, fault_id="L1")
        assert clients.faults()["active"] == ["L1"]

        # Clearing faults returns the plant to healthy, not to silent.
        assert clients.clear_faults()["removed"] == ["L1"]
        assert len(clients.faults()["instruments"]) >= 3
