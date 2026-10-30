"""The workbench.

One page for the whole project: start the stack, watch the plant, drive the
controller, inject faults, generate data, train a detector and read its
numbers. Four tabs over the same four services.

Two rules hold this together.

**The workbench owns no process state.** Every value on screen came from the
HMI, the plant's fault API or the historian a moment ago. There is no cached
copy of the plant here to disagree with the real one.

**Callbacks stay thin.** Anything worth testing lives in `clients.py`,
`tasks.py`, `supervisor.py` or `workspace.py`, all of which run headless.
Logic inside a callback is logic that can only be exercised through a browser.

All four panels are always present in the layout and hidden with CSS rather
than being created when a tab is selected, so no callback ever fires against a
component that does not exist yet.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import dash
from dash import ALL, Input, Output, State, dcc, html, no_update

from pyplant.faults.model import FAULT_KINDS
from pyplant.ml.detector import METHODS
from pyplant.ml.evaluate import EXPECTED_CHANNEL
from pyplant.workbench.clients import (
    StackClients,
    StackUnavailable,
    forget_failures,
)
from pyplant.workbench.jobs import JobRunner
from pyplant.workbench.supervisor import SERVICES, Supervisor
from pyplant.workbench.tasks import (
    generate_dataset,
    run_default_scenario,
    train_detector,
)
from pyplant.workbench.workspace import DataConfig, LoopConfig, Workspace

TREND_TAGS = ["LT101", "LIC101_SP", "LIC101_OP", "LT102"]
SENSOR_KINDS = {"sensor_bias", "sensor_drift", "sensor_stuck", "sensor_noise"}
SENSOR_TAGS = ["LT101", "LT102", "FT101", "FT102", "PT101", "ZT101"]

# Sensible starting magnitudes, in each fault's own units.
DEFAULT_MAGNITUDE = {
    "leak": 5e-5, "fouling": 0.35, "pump_wear": 0.18, "stiction": 0.10,
    "sensor_bias": 25.0, "sensor_drift": 1.5, "sensor_stuck": None,
    "sensor_noise": 2.0,
}
MAGNITUDE_UNITS = {
    "leak": "m² effective leak area", "fouling": "fraction of flow area lost",
    "pump_wear": "fraction of head lost", "stiction": "deadband, fraction of stroke",
    "sensor_bias": "offset in the tag's units", "sensor_drift": "units per minute",
    "sensor_stuck": "no magnitude", "sensor_noise": "sigma in the tag's units",
}

# One click each, because the point of the Faults tab is to break the plant and
# watch what happens, not to fill in a form correctly first.
PRESETS: dict[str, dict] = {
    "Leak in tank 1": {"kind": "leak", "magnitude": 5e-5},
    "Slow leak (15 min ramp)": {"kind": "leak", "magnitude": 5e-5,
                                "ramp": 900.0},
    "Sticking valve": {"kind": "stiction", "magnitude": 0.10},
    "Worn pump": {"kind": "pump_wear", "magnitude": 0.18, "ramp": 600.0},
    "Fouled interconnect": {"kind": "fouling", "magnitude": 0.35,
                            "ramp": 1200.0},
    "LT101 reads 25 mm high": {"kind": "sensor_bias", "magnitude": 25.0,
                               "tag": "LT101"},
    "LT101 frozen": {"kind": "sensor_stuck", "tag": "LT101"},
}

# A frozen transmitter sitting exactly on setpoint gives the controller nothing
# to act on. Move the setpoint and it has to chase a reading that cannot
# follow, which is what walks the loop into the interlock.
PRESET_NEXT_STEP = {
    "LT101 frozen": "now move the setpoint on the Runtime tab - the "
                    "controller will chase a reading that cannot follow",
    "Sticking valve": "watch the trend: the level should start hunting, and "
                      "FCV101 and ZT101 stop agreeing",
}

PLOT_TEXT = "#5f6b78"
PLOT_GRID = "#e3e7ec"
PLOT_COLORS = ["#2f7bd0", "#7a8694", "#e07b39", "#2e9a58", "#8e5cc8",
               "#cc3a40"]

# The scenario the Quick start tab opens with. A leak is the clearest fault on
# this plant: the level is held at setpoint and only the controller output
# shows it, which is the point the project exists to make.
DEFAULT_SCENARIO = "leak_abrupt"
SCENARIO_CHOICES = {
    "normal": "Normal operation - setpoint moves only",
    "leak_abrupt": "Leak in tank 1 (default)",
    "leak_incipient": "Slow leak, 15 minute ramp",
    "pump_wear": "Pump impeller wear",
    "fouling": "Fouled interconnect",
    "stiction": "Sticking control valve",
    "sensor_bias": "LT101 reads 25 mm high",
    "sensor_drift": "LT101 drifting",
    "sensor_stuck": "LT101 frozen",
}
LIVE_DEMO_FAULT = "Leak in tank 1"

# Dataset sizes. Quick is enough to see the whole pipeline work end to end;
# standard is the configuration the README's numbers come from.
DATA_PRESETS: dict[str, dict] = {
    "Quick (~1.5 min)": {"duration": 1200.0, "magnitudes": 1, "repeats": 1,
                      "normal_runs": 6},
    "Standard (~4 min)": {"duration": 3600.0, "magnitudes": 3, "repeats": 2,
                          "normal_runs": 10},
    "Thorough (~10 min)": {"duration": 3600.0, "magnitudes": 5, "repeats": 3,
                           "normal_runs": 16},
}

CSS = """
:root { --bg:#e3e6ea; --panel:#ffffff; --line:#d3d8de; --text:#1f2933;
        --dim:#5f6b78; --ok:#2e9a58; --warn:#c48612; --bad:#cc3a40;
        --accent:#2f7bd0; --vessel:#f1f3f6; --head:#3b4450; --soft:#f4f5f7; }
body { margin:0; background:var(--bg); color:var(--text);
       font:14px/1.45 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif; }
h1 { font-size:16px; margin:0; font-weight:650; letter-spacing:.2px; color:#fff;
     white-space:nowrap; }
h2 { font-size:12px; letter-spacing:.08em; text-transform:uppercase;
     color:var(--dim); margin:0 0 12px; font-weight:650;
     padding-bottom:8px; border-bottom:1px solid var(--soft); }
.bar { display:flex; align-items:center; gap:12px; padding:11px 20px;
       background:var(--head); box-shadow:0 2px 6px rgba(0,0,0,.15);
       position:sticky; top:0; z-index:10; }
.bar .sub { color:#aeb8c3; font-size:12px; white-space:nowrap; }
.bar button { white-space:nowrap; }
.bar { flex-wrap:wrap; }
@media (max-width:1100px) { .bar .sub { display:none; } }
.bar .pill { border-color:rgba(255,255,255,.25); color:#d5dbe2; }
.bar .pill.on { color:#7fd6a0; border-color:#7fd6a0; }
.bar .pill.bad { color:#ff9a9e; border-color:#ff9a9e; }
.bar .pill.warn { color:#f3c66b; border-color:#f3c66b; }
.bar .hint { color:#c4ccd5; }
.pill { padding:2px 10px; border-radius:11px; font-size:12px;
        border:1px solid var(--line); color:var(--dim); white-space:nowrap;
        background:transparent; }
.pill.on { color:var(--ok); border-color:var(--ok); }
.pill.bad { color:var(--bad); border-color:var(--bad); }
.pill.warn { color:var(--warn); border-color:var(--warn); }
.page { max-width:1320px; margin:0 auto; padding:0 14px; box-sizing:border-box; }
[id^="panel-"] { padding-bottom:30px; }
.tabs { margin-top:14px; }
.panel { background:var(--panel); border:1px solid var(--line);
         border-radius:10px; padding:14px 16px; margin:10px 0;
         box-shadow:0 1px 3px rgba(20,30,40,.07); }
.cols { display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr));
        gap:0 14px; align-items:start; }
.kv { display:grid; grid-template-columns:repeat(auto-fit,minmax(110px,1fr));
      gap:8px; }
.cell { background:var(--soft); border:1px solid var(--line); border-radius:8px;
        padding:7px 10px; }
.cell .k { color:var(--dim); font-size:11px; text-transform:uppercase;
           letter-spacing:.05em; }
.cell .v { font-size:18px; font-weight:650; }
.cell .u { color:var(--dim); font-size:11px; font-weight:400; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th { text-align:left; font-size:11px; text-transform:uppercase; color:var(--dim);
     background:var(--soft); border-bottom:1px solid var(--line);
     padding:6px 8px; font-weight:650; }
td { padding:6px 8px; border-bottom:1px solid var(--soft); }
tr:hover td { background:#fafbfc; }
button { background:#fff; color:var(--text); border:1px solid var(--line);
         border-radius:6px; padding:6px 12px; font:inherit; font-size:13px;
         cursor:pointer; transition:background .12s, border-color .12s; }
button:hover { border-color:var(--accent); background:#f5f9fe; }
button.go { background:var(--ok); border-color:var(--ok); color:#fff; }
button.go:hover { background:#278a4d; }
button.primary { background:var(--accent); border-color:var(--accent);
                 color:#fff; font-weight:600; }
button.primary:hover { background:#2569b5; }
button.stop { border-color:var(--bad); color:var(--bad); }
button.stop:hover { background:#fdf2f2; }
.bar button { border-color:rgba(255,255,255,.3); }
input { background:#fff; color:var(--text); border:1px solid var(--line);
        border-radius:6px; padding:5px 8px; font:inherit; font-size:13px; }
input:focus { outline:2px solid #bcd6f3; border-color:var(--accent); }
.row { display:flex; gap:8px; align-items:center; flex-wrap:wrap;
       margin-bottom:10px; }
.row label { color:var(--dim); font-size:12px; min-width:74px; }
.msg { font-size:12px; color:var(--dim); min-height:17px; margin:4px 0 0; }
.msg.bad { color:var(--bad); }
.msg.ok { color:var(--ok); }
.tank { height:120px; width:52px; border:1px solid var(--line); border-radius:3px;
        position:relative; background:var(--vessel); }
.tank .fill { position:absolute; bottom:0; left:0; right:0;
              background:var(--accent); opacity:.55; }
.hint { color:var(--dim); font-size:12px; line-height:1.55; }
.prog { height:8px; background:var(--soft); border:1px solid var(--line);
        border-radius:4px; overflow:hidden; margin-top:4px; }
.prog > div { height:100%; background:var(--accent); transition:width .4s; }
.tab { padding:8px 18px !important; border:1px solid var(--line) !important;
       border-radius:8px 8px 0 0; background:#d8dce1 !important;
       color:var(--dim); cursor:pointer; font-size:13px; font-weight:600; }
.tab--selected { background:var(--panel) !important; color:var(--text) !important;
                 border-top:3px solid var(--accent) !important;
                 border-bottom-color:var(--panel) !important; }
.steps { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
         gap:10px; }
.step { border:1px solid var(--line); border-radius:10px; padding:12px;
        background:var(--soft); display:flex; flex-direction:column; gap:6px; }
.step.done { border-color:var(--ok); background:#f0f8f3; }
.step .num { width:24px; height:24px; border-radius:50%; background:var(--head);
             color:#fff; font-size:12px; font-weight:700; display:flex;
             align-items:center; justify-content:center; }
.step.done .num { background:var(--ok); }
.step .title { font-weight:650; }
.step button { align-self:flex-start; margin-top:auto; }
.methods label { display:flex !important; align-items:center; gap:8px;
                 margin:3px 0; font-size:13px; color:var(--text) !important;
                 min-width:0 !important; cursor:pointer; }
.methods input { margin:0; accent-color:var(--accent); }
"""


# ---------------------------------------------------------------------------
# Small pure helpers. Kept out of the callbacks so they can be tested.
# ---------------------------------------------------------------------------

def clicked(values: Any) -> bool:
    """Did a pattern-matching group actually get clicked?

    Dash fires these callbacks once at registration with every `n_clicks` at
    None, and hands back a scalar rather than a list when the group has one
    member. Both cases have to mean "nobody pressed anything" rather than a
    crash.
    """
    if values is None:
        return False
    if not isinstance(values, (list, tuple)):
        values = [values]
    return any(bool(v) for v in values)


def fmt(value: Any, digits: int = 1, dash: str = "--") -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return dash


def cell(label: str, value: str, unit: str = "") -> html.Div:
    return html.Div(className="cell", children=[
        html.Div(label, className="k"),
        html.Div([html.Span(value), html.Span(f" {unit}", className="u")],
                 className="v")])


MIMIC_W, MIMIC_H = 700, 250


def _box(**style) -> dict:
    base = {"position": "absolute"}
    base.update(style)
    return base


def _pipe(x: float, y: float, w: float, h: float, live: bool) -> html.Div:
    return html.Div(style=_box(
        left=f"{x}px", top=f"{y}px", width=f"{w}px", height=f"{h}px",
        background="var(--accent)" if live else "var(--line)",
        opacity="0.9" if live else "0.5"))


def _vessel(x: float, y: float, w: float, h: float, mm: Any, label: str,
            tag: str, alarm: bool, span: float = 600.0) -> list:
    try:
        fraction = min(max(float(mm) / span, 0.0), 1.0)
    except (TypeError, ValueError):
        fraction = 0.0
    fill = h * fraction
    return [
        html.Div(style=_box(
            left=f"{x}px", top=f"{y}px", width=f"{w}px", height=f"{h}px",
            border=f"2px solid {'var(--bad)' if alarm else 'var(--line)'}",
            borderRadius="4px", background="var(--vessel)", overflow="hidden")),
        html.Div(style=_box(
            left=f"{x + 2}px", top=f"{y + h - fill}px", width=f"{w - 4}px",
            height=f"{max(fill - 2, 0)}px", background="var(--accent)",
            opacity="0.5")),
        html.Div(label, style=_box(
            left=f"{x}px", top=f"{y - 17}px", width=f"{w}px",
            textAlign="center", fontSize="11px", color="var(--dim)")),
        html.Div(f"{fmt(mm)} mm", style=_box(
            left=f"{x}px", top=f"{y + h / 2 - 16}px", width=f"{w}px",
            textAlign="center", fontSize="15px", fontWeight="600")),
        html.Div(tag, style=_box(
            left=f"{x}px", top=f"{y + h / 2 + 2}px", width=f"{w}px",
            textAlign="center", fontSize="10px", color="var(--dim)")),
    ]


def mimic(values: dict, alarms: set[str] | None = None) -> html.Div:
    """The process, drawn from live tag values.

    Positioned HTML rather than SVG: Dash has no SVG components and escapes raw
    markup, and this keeps the diagram themed with everything else on the page.
    """
    alarms = alarms or set()
    running = bool(values.get("PMP101_RUN"))
    flow_in = float(values.get("FT101") or 0.0)
    flow_out = float(values.get("FT102") or 0.0)
    stem = float(values.get("ZT101") or 0.0)
    tripped = bool(values.get("ILK_PMP101"))

    children: list = []
    # pump
    children.append(html.Div(style=_box(
        left="14px", top="168px", width="42px", height="42px",
        borderRadius="50%",
        border=f"2px solid {'var(--ok)' if running else 'var(--line)'}",
        background="var(--vessel)")))
    children.append(html.Div("P", style=_box(
        left="14px", top="180px", width="42px", textAlign="center",
        fontSize="15px", fontWeight="600",
        color="var(--ok)" if running else "var(--dim)")))
    children.append(html.Div("PMP101", style=_box(
        left="4px", top="214px", width="62px", textAlign="center",
        fontSize="10px", color="var(--dim)")))

    children.append(_pipe(56, 185, 54, 6, running and flow_in > 0.1))
    # control valve, filled to its measured stem position
    children.append(html.Div(style=_box(
        left="110px", top="172px", width="34px", height="32px",
        border="2px solid var(--line)", borderRadius="3px",
        background="var(--vessel)", overflow="hidden")))
    children.append(html.Div(style=_box(
        left="112px", top=f"{174 + 28 * (1 - stem / 100.0)}px", width="30px",
        height=f"{28 * stem / 100.0:.1f}px", background="var(--accent)",
        opacity="0.55")))
    children.append(html.Div(f"FCV101  {fmt(stem)}%", style=_box(
        left="90px", top="214px", width="76px", textAlign="center",
        fontSize="10px", color="var(--dim)")))

    children.append(_pipe(144, 185, 46, 6, running and flow_in > 0.1))
    children.append(_pipe(184, 60, 6, 131, running and flow_in > 0.1))
    children.extend(_vessel(190, 40, 120, 150, values.get("LT101"), "TANK 1",
                            "LT101", "ALM_LT101_HI" in alarms
                            or "ALM_LT101_LOLO" in alarms))
    children.append(_pipe(310, 178, 60, 6, flow_out > 0.1))
    children.append(_pipe(364, 80, 6, 104, flow_out > 0.1))
    children.extend(_vessel(370, 60, 120, 130, values.get("LT102"), "TANK 2",
                            "LT102", False))
    children.append(_pipe(490, 178, 56, 6, flow_out > 0.1))
    children.append(_pipe(540, 178, 6, 54, flow_out > 0.1))
    children.append(html.Div(f"drain  {fmt(flow_out, 2)} L/min", style=_box(
        left="500px", top="234px", fontSize="10px", color="var(--dim)")))

    children.append(html.Div(f"FT101  {fmt(flow_in, 2)} L/min", style=_box(
        left="60px", top="152px", fontSize="10px", color="var(--dim)")))
    children.append(html.Div(
        f"PT101  {fmt(values.get('PT101'))} kPa", style=_box(
            left="60px", top="136px", fontSize="10px", color="var(--dim)")))

    if tripped:
        children.append(html.Div("INTERLOCK TRIPPED", style=_box(
            left="190px", top="8px", padding="2px 10px", fontSize="11px",
            border="1px solid var(--bad)", borderRadius="10px",
            color="var(--bad)")))

    return html.Div(style={"position": "relative", "width": f"{MIMIC_W}px",
                           "height": f"{MIMIC_H}px", "maxWidth": "100%"},
                    children=children)


def tank(label: str, mm: Any, span: float = 600.0) -> html.Div:
    try:
        fraction = min(max(float(mm) / span, 0.0), 1.0)
    except (TypeError, ValueError):
        fraction = 0.0
    return html.Div(style={"textAlign": "center"}, children=[
        html.Div(className="tank", children=html.Div(
            className="fill", style={"height": f"{fraction * 100:.1f}%"})),
        html.Div(label, className="k", style={"marginTop": "4px",
                                              "color": "var(--dim)",
                                              "fontSize": "11px"}),
        html.Div(f"{fmt(mm)} mm", style={"fontSize": "12px"})])


def table(header: list[str], rows: list[list[Any]],
          empty: str = "nothing yet") -> html.Table | html.Div:
    if not rows:
        return html.Div(empty, className="hint")
    return html.Table([
        html.Thead(html.Tr([html.Th(h) for h in header])),
        html.Tbody([html.Tr([html.Td(c) for c in row]) for row in rows])])


def trend_figure(data: dict) -> dict:
    """Level tags on the left axis, controller output on its own on the right."""
    traces = []
    for name in data.get("series", {}):
        series = data["series"][name]
        traces.append({
            "x": data["t"], "y": series, "name": name, "mode": "lines",
            "line": {"width": 1.6},
            "yaxis": "y2" if name == "LIC101_OP" else "y",
        })
    return {
        "data": traces,
        "layout": {
            "margin": {"l": 48, "r": 48, "t": 10, "b": 32},
            "height": 300, "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "font": {"color": PLOT_TEXT, "size": 11},
            "colorway": PLOT_COLORS,
            "xaxis": {"gridcolor": PLOT_GRID},
            "yaxis": {"title": "mm", "gridcolor": PLOT_GRID},
            "yaxis2": {"title": "%", "overlaying": "y", "side": "right",
                       "range": [0, 100], "showgrid": False},
            "legend": {"orientation": "h", "y": 1.15},
            "showlegend": True,
        },
    }


SCENARIO_LEFT = ("LT101", "LIC101_SP", "LT102")


def scenario_figure(result: dict) -> dict:
    """A finished offline scenario, with the fault onset marked.

    Levels on the left axis, controller output and stem position on the
    right: under a process fault the level is held and the story is in OP.
    """
    t = result.get("t", [])
    traces = []
    for name, series in result.get("series", {}).items():
        if name not in SCENARIO_LEFT + ("LIC101_OP", "ZT101"):
            continue
        traces.append({
            "x": t, "y": series, "name": name, "mode": "lines",
            "line": {"width": 1.6,
                     "dash": "dot" if name == "LIC101_SP" else "solid"},
            "yaxis": "y" if name in SCENARIO_LEFT else "y2",
        })
    shapes, notes = [], []
    onset = result.get("onset")
    if onset is not None and t:
        shapes.append({"type": "rect", "xref": "x", "yref": "paper",
                       "x0": onset, "x1": t[-1], "y0": 0, "y1": 1,
                       "fillcolor": "rgba(204,58,64,0.07)", "line": {"width": 0}})
        shapes.append({"type": "line", "xref": "x", "yref": "paper",
                       "x0": onset, "x1": onset, "y0": 0, "y1": 1,
                       "line": {"color": "#cc3a40", "width": 1.5,
                                "dash": "dash"}})
        notes.append({"x": onset, "y": 1.02, "xref": "x", "yref": "paper",
                      "text": "fault onset", "showarrow": False,
                      "font": {"color": "#cc3a40", "size": 11},
                      "xanchor": "left"})
    return {
        "data": traces,
        "layout": {
            "margin": {"l": 50, "r": 50, "t": 26, "b": 36}, "height": 340,
            "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "#fbfbfc",
            "font": {"color": PLOT_TEXT, "size": 11}, "colorway": PLOT_COLORS,
            "xaxis": {"title": "plant time (s)", "gridcolor": PLOT_GRID},
            "yaxis": {"title": "level (mm)", "gridcolor": PLOT_GRID},
            "yaxis2": {"title": "%", "overlaying": "y", "side": "right",
                       "range": [0, 100], "showgrid": False,
                       "tickmode": "array",
                       "tickvals": [0, 20, 40, 60, 80, 100]},
            "legend": {"orientation": "h", "y": 1.16},
            "shapes": shapes, "annotations": notes,
        },
    }


def _mean(values: list, mask: list[bool]) -> float | None:
    picked = [v for v, keep in zip(values, mask) if keep and v is not None]
    return sum(picked) / len(picked) if picked else None


def scenario_summary(result: dict) -> list[tuple[str, str, str]]:
    """Before-and-after numbers that say what the fault did.

    Compares the last five minutes before onset against the last five
    minutes of the run, because both stretches are lined out.
    """
    t = result.get("t", [])
    series = result.get("series", {})
    onset = result.get("onset")
    if not t:
        return []
    end = t[-1]
    cut = onset if onset is not None else end / 2
    before = [cut - 300 <= x < cut for x in t]
    after = [x >= end - 300 for x in t]
    out = []
    for tag, label, unit in (("LT101", "LT101", "mm"),
                             ("LIC101_OP", "Controller OP", "%"),
                             ("LT102", "LT102", "mm"),
                             ("FT101", "FT101", "L/min")):
        if tag not in series:
            continue
        b, a = _mean(series[tag], before), _mean(series[tag], after)
        delta = None if a is None or b is None else a - b
        out.append((f"{label} change", f"{'+' if (delta or 0) >= 0 else ''}"
                    f"{fmt(delta, 2)}", unit))
    trips = [x for x, v in zip(t, series.get("ILK_PMP101", [])) if v]
    out.append(("Pump trip", f"t = {trips[0]:.0f} s" if trips else "none", ""))
    out.append(("Fault onset", "none" if onset is None else f"{onset:.0f}",
                "" if onset is None else "s"))
    return out


def metrics_rows(metrics: dict) -> list[list[Any]]:
    rows = []
    for name, result in metrics.get("results", {}).items():
        chosen = " (selected)" if name == metrics.get("selected") else ""
        rows.append([
            name + chosen,
            f"{result['detection_rate'] * 100:.0f}%",
            fmt(result["median_delay_s"], 0) + " s",
            fmt(result["false_alarms_per_day"], 2),
            f"{result['window_flag_rate'] * 100:.2f}%",
            fmt(float(result["isolation"] or 0) * 100, 0) + "%",
        ])
    return rows


def per_mode_rows(metrics: dict) -> list[list[Any]]:
    selected = metrics.get("selected")
    result = metrics.get("results", {}).get(selected, {})
    rows = []
    for mode in result.get("per_mode", []):
        expected = ", ".join(EXPECTED_CHANNEL.get(mode["kind"], ())) or "not scored"
        rows.append([mode["kind"], int(mode["runs"]),
                     f"{float(mode['detected']) * 100:.0f}%",
                     fmt(mode["median_delay"], 0) + " s",
                     mode.get("top") or "--", expected])
    return rows


# ---------------------------------------------------------------------------
# The application
# ---------------------------------------------------------------------------

def build_app(workspace_path: str = "./work/demo",
              base_port: int | None = 5400) -> dash.Dash:
    ws = Workspace(workspace_path, base_port=base_port)
    supervisor = Supervisor(ws)
    runner = JobRunner(ws.jobs)
    clients = StackClients(ws.config.ports)

    app = dash.Dash(__name__, suppress_callback_exceptions=True,
                    title="pyplant workbench")
    app.index_string = f"""<!DOCTYPE html><html><head>{{%metas%}}<title>{{%title%}}
</title>{{%favicon%}}{{%css%}}<style>{CSS}</style></head><body>{{%app_entry%}}
<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer></body></html>"""

    app.layout = html.Div([
        dcc.Interval(id="tick", interval=1000),
        dcc.Interval(id="slow", interval=3000),
        html.Div(className="bar", children=[
            html.Div([html.H1("pyplant workbench"),
                      html.Div("two-tank plant, soft PLC, fault diagnosis",
                               className="sub")]),
            html.Span(ws.config.name, className="pill"),
            html.Span("services --", id="stack-pill", className="pill"),
            html.Span("alarms --", id="alarm-pill", className="pill"),
            html.Span("faults --", id="fault-pill", className="pill"),
            html.Div(style={"flex": "1"}),
            html.Label("Speed", className="hint"),
            dcc.Dropdown(id="speed", value=ws.config.speed, clearable=False,
                         style={"width": "150px", "color": "#1f2933"},
                         options=[{"label": "1x real time", "value": 1.0},
                                  {"label": "2x", "value": 2.0},
                                  {"label": "4x", "value": 4.0},
                                  {"label": "10x", "value": 10.0},
                                  {"label": "20x", "value": 20.0}]),
            html.Button("Start stack", id="start", className="go"),
            html.Button("Stop", id="stop", className="stop"),
            html.A("Open HMI", id="hmi-link", href=clients.hmi, target="_blank",
                   className="pill", style={"textDecoration": "none"}),
        ]),
        html.Div(id="bar-msg", className="msg page"),

        dcc.Tabs(id="tabs", value="quickstart", parent_className="page tabs",
                 mobile_breakpoint=0, children=[
                     dcc.Tab(label="Quick start", value="quickstart",
                             className="tab",
                             selected_className="tab--selected"),
                     dcc.Tab(label="Runtime", value="runtime", className="tab",
                             selected_className="tab--selected"),
                     dcc.Tab(label="Faults", value="faults", className="tab",
                             selected_className="tab--selected"),
                     dcc.Tab(label="Data", value="data", className="tab",
                             selected_className="tab--selected"),
                     dcc.Tab(label="Model", value="model", className="tab",
                             selected_className="tab--selected"),
                 ]),

        html.Div(id="panel-quickstart", className="page",
                 children=_quickstart_panel()),
        html.Div(id="panel-runtime", className="page",
                 children=_runtime_panel()),
        html.Div(id="panel-faults", className="page",
                 children=_faults_panel()),
        html.Div(id="panel-data", className="page",
                 children=_data_panel(ws.config.data)),
        html.Div(id="panel-model", className="page",
                 children=_model_panel(ws.config.data)),
    ])

    _register(app, ws, supervisor, runner, clients)
    return app


WORKFLOW = [
    ("Run the default scenario", "Below, offline, in a few seconds. No "
     "services needed.", None),
    ("Start the live stack", "Plant, PLC, historian and HMI as four "
     "processes. Watch it on the Runtime tab.", "runtime"),
    ("Break something", "Inject a leak, a sticking valve or a lying sensor "
     "and see who notices.", "faults"),
    ("Generate data", "Sweep every fault mode into a labelled dataset. Start "
     "with the Quick preset.", "data"),
    ("Train and compare", "Fit up to four detectors, from z-score limits to "
     "an autoencoder, and compare them.", "model"),
]


def _quickstart_panel() -> list:
    return [
        html.Div(className="panel", children=[
            html.H2("Workflow"),
            html.Div(id="qs-steps", className="steps",
                     children=workflow_steps({})),
        ]),
        html.Div(className="panel", children=[
            html.H2("Default scenario"),
            html.Div(className="row", children=[
                html.Label("Scenario"),
                dcc.Dropdown(id="qs-scenario", value=DEFAULT_SCENARIO,
                             clearable=False,
                             style={"width": "300px", "color": "#1f2933"},
                             options=[{"label": v, "value": k}
                                      for k, v in SCENARIO_CHOICES.items()]),
                html.Label("Length"),
                dcc.Dropdown(id="qs-duration", value=1800.0, clearable=False,
                             style={"width": "130px", "color": "#1f2933"},
                             options=[{"label": "15 minutes", "value": 900.0},
                                      {"label": "30 minutes", "value": 1800.0},
                                      {"label": "1 hour", "value": 3600.0}]),
                dcc.Checklist(id="qs-moves", value=[], className="methods",
                              options=[{"label": "setpoint moves",
                                        "value": "moves"}]),
                html.Button("Run scenario", id="qs-run", className="primary"),
            ]),
            dcc.Loading(type="circle", color="#2f7bd0", children=[
                html.Div(id="qs-summary", className="kv",
                         style={"margin": "6px 0 4px"}),
                dcc.Graph(id="qs-graph", config={"displayModeBar": False},
                          figure=scenario_figure({})),
                html.Div(id="qs-msg", className="msg"),
            ]),
            html.Div(className="hint", children=[
                "Runs the plant, the PLC program and the fault through the "
                "fast runner in plant time, exactly as dataset generation "
                "does. Under the default leak the level returns to setpoint "
                "and only the controller output moves: a detector watching "
                "the level alone would never see it."]),
        ]),
        html.Div(className="panel", children=[
            html.H2("Live demo in one click"),
            html.Div(className="row", children=[
                html.Button("Start stack and inject a leak", id="qs-live",
                            className="go"),
                html.Button("Clear faults", id="qs-clear"),
            ]),
            dcc.Loading(type="dot", color="#2e9a58",
                        children=html.Div(id="qs-live-msg", className="msg")),
            html.Div(className="hint", children=[
                "Starts the four services if they are not already up, waits "
                "for them, then injects a tank 1 leak on the running plant. "
                "Switch to the Runtime tab and watch the controller output "
                "climb while the level holds."]),
        ]),
    ]


def workflow_steps(status: dict) -> list:
    """The numbered steps, each marked done from what the workspace holds."""
    done = [bool(status.get("scenario")), bool(status.get("stack")),
            bool(status.get("faults")), bool(status.get("has_dataset")),
            bool(status.get("has_metrics"))]
    cards = []
    for i, ((title, text, tab), finished) in enumerate(zip(WORKFLOW, done), 1):
        children = [html.Div(str(i), className="num"),
                    html.Div(title, className="title"),
                    html.Div(text, className="hint")]
        if tab:
            children.append(html.Button(f"Go to {tab}",
                                        id={"type": "goto", "tab": tab}))
        cards.append(html.Div(className="step" + (" done" if finished else ""),
                              children=children))
    return cards


def _runtime_panel() -> list:
    return [
        html.Div(className="panel", children=[
            html.H2("Services"),
            html.Div(id="services"),
            html.Div(id="service-hint", className="hint",
                     style={"marginTop": "8px"}),
        ]),
        html.Div(className="panel", children=[
            html.H2("Unit 1"),
            html.Div(id="mimic", style={"overflowX": "auto"}),
            html.Div(id="plant-values", className="kv",
                     style={"marginTop": "10px"}),
        ]),
        html.Div(className="panel", children=[
            html.H2("LIC101"),
            html.Div(id="faceplate", className="kv",
                     style={"marginBottom": "10px"}),
            html.Div(className="row", children=[
                html.Label("Setpoint"),
                dcc.Input(id="sp-in", type="number", value=300, step=5,
                          style={"width": "92px"}),
                html.Button("Write", id="sp-write"),
                html.Label("Manual OP", style={"marginLeft": "14px"}),
                dcc.Input(id="man-in", type="number", value=66, step=1,
                          style={"width": "92px"}),
                html.Button("Write", id="man-write"),
            ]),
            html.Div(className="row", children=[
                html.Label("Mode"),
                html.Button("Auto", id="auto-on"),
                html.Button("Manual", id="auto-off"),
                html.Label("Pump", style={"marginLeft": "14px"}),
                html.Button("Start", id="pump-on"),
                html.Button("Stop", id="pump-off"),
                html.Button("Reset interlock", id="ilk-reset",
                            style={"marginLeft": "14px"}),
            ]),
            html.Div(id="control-msg", className="msg"),
        ]),
        html.Div(className="panel", children=[
            html.H2("Trend"),
            html.Div(className="row", children=[
                html.Label("Window"),
                dcc.Dropdown(id="trend-span", value=300, clearable=False,
                             style={"width": "150px", "color": "#1f2933"},
                             options=[{"label": "2 minutes", "value": 120},
                                      {"label": "5 minutes", "value": 300},
                                      {"label": "15 minutes", "value": 900},
                                      {"label": "1 hour", "value": 3600}]),
            ]),
            dcc.Graph(id="trend", config={"displayModeBar": False}),
        ]),
        html.Div(className="panel", children=[
            html.H2("Alarms"),
            html.Div(className="row", children=[
                html.Button("Acknowledge all", id="ack-all")]),
            html.Div(id="alarms"),
            html.Div(className="hint", style={"marginTop": "6px"}, children=[
                "An alarm that cleared before anyone looked stays on this list "
                "as RTN UNACK until it is acknowledged. The transient nobody "
                "saw is usually the interesting one."]),
        ]),
    ]


def _faults_panel() -> list:
    return [
        html.Div(className="panel", children=[
            html.H2("Break something"),
            html.Div(className="row", children=[
                html.Button(name, id={"type": "preset", "name": name},
                            className="stop") for name in PRESETS]),
            html.Div(className="row", children=[
                html.Button("Clear all faults", id="fault-clear")]),
            html.Div(id="preset-msg", className="msg"),
        ]),
        html.Div(className="panel", children=[
            html.H2("Or build one"),
            html.Div(className="row", children=[
                html.Label("Kind"),
                dcc.Dropdown(id="fault-kind", value="leak", clearable=False,
                             style={"width": "190px", "color": "#1f2933"},
                             options=[{"label": k, "value": k}
                                      for k in sorted(FAULT_KINDS)]),
                html.Label("Magnitude"),
                dcc.Input(id="fault-mag", type="number", value=5e-5,
                          style={"width": "120px"}),
                html.Span(id="fault-units", className="hint"),
            ]),
            html.Div(className="row", children=[
                html.Label("Sensor"),
                dcc.Dropdown(id="fault-tag", value="LT101", clearable=False,
                             style={"width": "150px", "color": "#1f2933"},
                             options=[{"label": t, "value": t}
                                      for t in SENSOR_TAGS]),
                html.Label("Ramp (s)"),
                dcc.Input(id="fault-ramp", type="number", value=0, step=30,
                          style={"width": "100px"}),
                html.Button("Inject", id="fault-inject", className="stop"),
            ]),
            html.Div(id="fault-msg", className="msg"),
            html.Div(className="hint", style={"marginTop": "6px"}, children=[
                "Injection is deliberately separate from the control system. "
                "Neither the operator page nor a diagnosis model is told a "
                "fault was injected; the only way to know is to watch the "
                "plant."]),
        ]),
        html.Div(className="cols", children=[
            html.Div(className="panel", children=[
                html.H2("Active faults"), html.Div(id="fault-list")]),
            html.Div(className="panel", children=[
                html.H2("Ground truth against what the instruments report"),
                html.Div(id="truth"),
                html.Div(className="hint", style={"marginTop": "6px"},
                         children=[
                    "Under a sensor fault these disagree. Under a process "
                    "fault they agree, and the fault shows up in the "
                    "controller output instead of the measurement."]),
            ]),
        ]),
    ]


def _data_panel(data: DataConfig) -> list:
    return [
        html.Div(className="panel", children=[
            html.H2("Generate a dataset"),
            html.Div(className="row", children=[
                html.Label("Preset"),
                *[html.Button(name, id={"type": "d-preset", "name": name})
                  for name in DATA_PRESETS],
            ]),
            html.Div(className="row", children=[
                html.Label("Run length"),
                dcc.Input(id="d-duration", type="number", value=data.duration,
                          step=600, style={"width": "110px"}),
                html.Label("Magnitudes"),
                dcc.Input(id="d-mag", type="number", value=data.magnitudes,
                          min=1, step=1, style={"width": "80px"}),
                html.Label("Repeats"),
                dcc.Input(id="d-rep", type="number", value=data.repeats,
                          min=1, step=1, style={"width": "80px"}),
                html.Label("Normal runs"),
                dcc.Input(id="d-normal", type="number", value=data.normal_runs,
                          min=3, step=1, style={"width": "80px"}),
                html.Button("Generate", id="d-go", className="go"),
                html.Button("Cancel", id="d-cancel"),
            ]),
            html.Div(className="prog", children=html.Div(id="d-bar",
                                                         style={"width": "0%"})),
            html.Div(id="d-msg", className="msg"),
            html.Div(className="hint", children=[
                "Each fault mode is swept across magnitudes with a randomised "
                "onset and its own seed, so a model cannot learn when the "
                "fault starts instead of what it looks like. Training uses "
                "normal runs only; the faulty ones measure the result."]),
        ]),
        html.Div(className="panel", children=[
            html.H2("Recent runs"), html.Div(id="run-log")]),
    ]


def _model_panel(data: DataConfig) -> list:
    return [
        html.Div(className="panel", children=[
            html.H2("Train"),
            html.Div(className="row", style={"alignItems": "flex-start"},
                     children=[
                html.Label("Methods"),
                dcc.Checklist(id="m-methods", className="methods",
                              value=list(METHODS),
                              options=[{"label": text, "value": key}
                                       for key, text in METHODS.items()]),
            ]),
            html.Div(className="row", children=[
                html.Label("Window"),
                dcc.Input(id="m-window", type="number", value=data.window,
                          step=10, style={"width": "90px"}),
                html.Label("Stride"),
                dcc.Input(id="m-stride", type="number", value=data.stride,
                          step=5, style={"width": "90px"}),
                html.Label("Quantile"),
                dcc.Input(id="m-quantile", type="number", value=data.quantile,
                          step=0.001, style={"width": "100px"}),
                html.Button("Train", id="m-go", className="go"),
            ]),
            html.Div(className="prog", children=html.Div(id="m-bar",
                                                         style={"width": "0%"})),
            html.Div(id="m-msg", className="msg"),
        ]),
        html.Div(className="panel", children=[
            html.H2("Results"), html.Div(id="m-metrics"),
            html.Div(className="hint", style={"marginTop": "8px"}, children=[
                "Z-score limits and PCA are the straightforward baselines. A "
                "learned detector that cannot beat them has not earned its "
                "place, and the comparison stays on the record either way. "
                "The best detection rate wins; ties go to fewer false alarms, "
                "then to the simpler method."]),
        ]),
        html.Div(className="panel", children=[
            html.H2("By fault mode"), html.Div(id="m-modes")]),
    ]


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _register(app: dash.Dash, ws: Workspace, supervisor: Supervisor,
              runner: JobRunner, clients: StackClients) -> None:

    @app.callback([Output(f"panel-{name}", "style") for name in PANELS],
                  Input("tabs", "value"))
    def show_tab(selected):
        # Every panel is in the layout at all times and merely hidden, so no
        # callback ever fires against a component that has not been created.
        return [{"display": "block" if name == selected else "none"}
                for name in PANELS]

    # -- quick start -------------------------------------------------------

    ran_scenario = {"done": False}

    @app.callback(Output("qs-graph", "figure"), Output("qs-summary", "children"),
                  Output("qs-msg", "children"), Output("qs-msg", "className"),
                  Input("qs-run", "n_clicks"), State("qs-scenario", "value"),
                  State("qs-duration", "value"), State("qs-moves", "value"))
    def quick_scenario(_, name, duration, moves):
        # Not prevent_initial_call: the page opens with the default scenario
        # already run, so there is something to look at before any click.
        try:
            result = run_default_scenario(name or DEFAULT_SCENARIO,
                                          float(duration or 1800.0),
                                          setpoint_moves=bool(moves))
        except Exception as exc:  # noqa: BLE001 - shown, never swallowed
            return scenario_figure({}), [], str(exc), "msg bad"
        ran_scenario["done"] = True
        events = result["events"]
        what = (", ".join(f"{e['kind']} on {e['target']}" for e in events)
                or "no fault")
        return (scenario_figure(result),
                [cell(k, v, u) for k, v, u in scenario_summary(result)],
                f"{result['description']} - {what}, "
                f"{result['duration'] / 60:.0f} minutes of plant time",
                "msg ok")

    @app.callback(Output("qs-live-msg", "children"),
                  Output("qs-live-msg", "className"),
                  Input("qs-live", "n_clicks"), Input("qs-clear", "n_clicks"),
                  State("speed", "value"), prevent_initial_call=True)
    def live_demo(live_clicks, clear_clicks, speed):
        which = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
        try:
            if which == "qs-clear":
                removed = clients.clear_faults()["removed"]
                return f"cleared {len(removed)} fault(s)", "msg ok"
            if not clients.reachable():
                ws.update(speed=float(speed or 1.0))
                supervisor.start()
                forget_failures()
            spec = dict(PRESETS[LIVE_DEMO_FAULT])
            injected = clients.inject(**spec)
            ws.finish_run(ws.start_run("fault", config=injected))
            return (f"stack up, injected {injected['fault_id']} on "
                    f"{injected['target']} - open the Runtime tab"), "msg ok"
        except Exception as exc:  # noqa: BLE001
            return str(exc), "msg bad"

    @app.callback(Output("tabs", "value"),
                  Input({"type": "goto", "tab": ALL}, "n_clicks"),
                  prevent_initial_call=True)
    def goto(clicks):
        triggered = dash.callback_context.triggered_id
        if not isinstance(triggered, dict) or not clicked(clicks):
            return no_update
        return triggered["tab"]

    @app.callback(Output("qs-steps", "children"), Input("slow", "n_intervals"))
    def steps(_):
        status = ws.status()
        status["scenario"] = ran_scenario["done"]
        status["stack"] = clients.reachable()
        status["faults"] = any(r.kind == "fault" for r in ws.runs(limit=200))
        return workflow_steps(status)

    @app.callback(Output("d-duration", "value"), Output("d-mag", "value"),
                  Output("d-rep", "value"), Output("d-normal", "value"),
                  Input({"type": "d-preset", "name": ALL}, "n_clicks"),
                  prevent_initial_call=True)
    def data_preset(clicks):
        triggered = dash.callback_context.triggered_id
        if not isinstance(triggered, dict) or not clicked(clicks):
            return no_update, no_update, no_update, no_update
        p = DATA_PRESETS[triggered["name"]]
        return p["duration"], p["magnitudes"], p["repeats"], p["normal_runs"]

    @app.callback(Output("bar-msg", "children"), Output("bar-msg", "className"),
                  Input("start", "n_clicks"), Input("stop", "n_clicks"),
                  State("speed", "value"), prevent_initial_call=True)
    def start_stop(start_clicks, stop_clicks, speed):
        which = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
        try:
            if which == "start":
                # Speed is a real-time factor. Slow faults need it: a stiction
                # limit cycle takes several process time constants to build,
                # and this plant's is 72 seconds. Above about 4x on Windows the
                # scan period drops under the 15.6 ms timer granularity and
                # SCAN_OVERRUNS will climb - which is the engine telling the
                # truth, not a malfunction.
                ws.update(speed=float(speed or 1.0))
                supervisor.start()
                forget_failures()
                return f"stack started at {speed}x", "msg ok"
            supervisor.stop()
            return "stack stopped", "msg"
        except Exception as exc:  # noqa: BLE001 - shown, never swallowed
            return str(exc), "msg bad"

    @app.callback(Output("stack-pill", "children"),
                  Output("stack-pill", "className"),
                  Output("services", "children"),
                  Output("service-hint", "children"),
                  Input("slow", "n_intervals"))
    def services(_):
        states = supervisor.states()
        up = sum(1 for s in states if s.running and s.listening)
        rows = [[s.name,
                 html.Span("up" if s.listening else "down",
                           className="pill " + ("on" if s.listening else "bad")),
                 s.pid or "--", s.port or "--",
                 BY_DESC.get(s.name, "")] for s in states]
        pill = f"{up}/{len(states)} services"
        style = "pill on" if up == len(states) else (
            "pill bad" if up == 0 else "pill warn")
        hint = ("Not running. Start the stack to see live values."
                if up == 0 else supervisor.command("plc"))
        return pill, style, table(["service", "state", "pid", "port", "role"],
                                  rows), (" ".join(hint)
                                          if isinstance(hint, list) else hint)

    @app.callback(Output("mimic", "children"),
                  Output("plant-values", "children"),
                  Output("faceplate", "children"),
                  Output("alarms", "children"),
                  Output("alarm-pill", "children"),
                  Output("alarm-pill", "className"),
                  Input("tick", "n_intervals"))
    def live(_):
        try:
            state = clients.state()
        except StackUnavailable:
            blank = html.Div("Stack not running. Press Start stack above.",
                             className="hint")
            return (mimic({}), blank, blank, blank, "alarms --", "pill")
        v = state["values"]
        plant = html.Div(className="kv", children=[
            cell("FT101", fmt(v.get("FT101"), 2), "L/min"),
            cell("FT102", fmt(v.get("FT102"), 2), "L/min"),
            cell("PT101", fmt(v.get("PT101")), "kPa"),
            cell("ZT101", fmt(v.get("ZT101")), "%"),
            cell("Scan", fmt(v.get("SCAN_MS"), 2), "ms"),
            cell("Overruns", fmt(v.get("SCAN_OVERRUNS"), 0), ""),
        ])
        face = html.Div(className="kv", children=[
            cell("PV", fmt(v.get("LIC101_PV")), "mm"),
            cell("SP", fmt(v.get("LIC101_SP")), "mm"),
            cell("OP", fmt(v.get("LIC101_OP")), "%"),
            cell("Error", fmt(v.get("LIC101_ERR")), "mm"),
            cell("Mode", "AUTO" if v.get("LIC101_AUTO") else "MANUAL", ""),
            cell("Pump", "RUN" if v.get("PMP101_RUN") else "STOP", ""),
        ])
        rows = [[a["priority_name"], a["tag"], a["message"],
                 html.Span(a["state"].replace("_", " "),
                           className="pill " + ("bad" if a["demands_attention"]
                                                else "warn")),
                 (a["in_time"] or "")[11:19], a["count"],
                 html.Div([
                     html.Button("Ack", id={"type": "alarm-ack",
                                            "tag": a["tag"]},
                                 style={"padding": "1px 8px",
                                        "fontSize": "12px"}),
                     html.Button("Shelve", id={"type": "alarm-shelve",
                                               "tag": a["tag"]},
                                 style={"padding": "1px 8px",
                                        "fontSize": "12px",
                                        "marginLeft": "4px"}),
                 ])]
                for a in state["alarms"]]
        active_alarms = {a["tag"] for a in state["alarms"]
                         if a["demands_attention"]}
        unacked = state["summary"]["unacknowledged"]
        return (mimic(v, active_alarms), plant, face,
                table(["priority", "tag", "message", "state", "in", "count",
                       ""], rows, "No active alarms."),
                f"alarms {unacked}",
                "pill bad" if unacked else "pill on")

    @app.callback(Output("trend", "figure"), Input("tick", "n_intervals"),
                  State("trend-span", "value"))
    def trend(_, seconds):
        try:
            return trend_figure(clients.trend(TREND_TAGS, float(seconds or 300)))
        except StackUnavailable:
            return trend_figure({"t": [], "series": {}})

    @app.callback(Output("control-msg", "children"),
                  Output("control-msg", "className"),
                  Input("sp-write", "n_clicks"), Input("man-write", "n_clicks"),
                  Input("auto-on", "n_clicks"), Input("auto-off", "n_clicks"),
                  Input("pump-on", "n_clicks"), Input("pump-off", "n_clicks"),
                  Input("ilk-reset", "n_clicks"), Input("ack-all", "n_clicks"),
                  State("sp-in", "value"), State("man-in", "value"),
                  prevent_initial_call=True)
    def control(*args):
        setpoint, manual = args[-2], args[-1]
        which = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
        actions = {
            "sp-write": lambda: clients.setpoint(float(setpoint)),
            "man-write": lambda: clients.manual(float(manual)),
            "auto-on": lambda: clients.mode(True),
            "auto-off": lambda: clients.mode(False),
            "pump-on": lambda: clients.pump(True),
            "pump-off": lambda: clients.pump(False),
            "ilk-reset": clients.reset,
            "ack-all": lambda: clients.acknowledge(None),
        }
        try:
            actions[which]()
            return f"{which} accepted", "msg ok"
        except StackUnavailable as exc:
            # Out-of-span setpoints land here: the HMI rejects rather than
            # silently clamping, so the operator sees the refusal.
            return str(exc), "msg bad"

    @app.callback(Output("control-msg", "children", allow_duplicate=True),
                  Output("control-msg", "className", allow_duplicate=True),
                  Input({"type": "alarm-ack", "tag": ALL}, "n_clicks"),
                  Input({"type": "alarm-shelve", "tag": ALL}, "n_clicks"),
                  prevent_initial_call=True)
    def alarm_row_action(ack_clicks, shelve_clicks):
        triggered = dash.callback_context.triggered_id
        if not isinstance(triggered, dict) or not (
                clicked(ack_clicks) or clicked(shelve_clicks)):
            return no_update, no_update
        tag = triggered["tag"]
        try:
            if triggered["type"] == "alarm-ack":
                clients.acknowledge(tag)
                return f"acknowledged {tag}", "msg ok"
            clients.shelve(tag, 30.0, "shelved from the workbench")
            return f"shelved {tag} for 30 minutes", "msg"
        except StackUnavailable as exc:
            return str(exc), "msg bad"

    # -- faults ------------------------------------------------------------

    @app.callback(Output("preset-msg", "children"),
                  Output("preset-msg", "className"),
                  Input({"type": "preset", "name": ALL}, "n_clicks"),
                  prevent_initial_call=True)
    def preset(clicks):
        triggered = dash.callback_context.triggered_id
        if not isinstance(triggered, dict) or not clicked(clicks):
            return no_update, no_update
        spec = dict(PRESETS[triggered["name"]])
        try:
            injected = clients.inject(**spec)
            ws.finish_run(ws.start_run("fault", config=injected))
            hint = PRESET_NEXT_STEP.get(triggered["name"],
                                        "watch the Runtime tab")
            return (f"injected {injected['fault_id']} on "
                    f"{injected['target']} - {hint}"), "msg bad"
        except StackUnavailable as exc:
            return str(exc), "msg bad"

    @app.callback(Output("fault-mag", "value"), Output("fault-units", "children"),
                  Output("fault-tag", "disabled"), Input("fault-kind", "value"))
    def fault_defaults(kind):
        return (DEFAULT_MAGNITUDE.get(kind), MAGNITUDE_UNITS.get(kind, ""),
                kind not in SENSOR_KINDS)

    @app.callback(Output("fault-msg", "children"),
                  Output("fault-msg", "className"),
                  Input("fault-inject", "n_clicks"),
                  Input("fault-clear", "n_clicks"),
                  State("fault-kind", "value"), State("fault-mag", "value"),
                  State("fault-tag", "value"), State("fault-ramp", "value"),
                  prevent_initial_call=True)
    def faults(inject_clicks, clear_clicks, kind, magnitude, tag, ramp):
        which = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
        try:
            if which == "fault-clear":
                removed = clients.clear_faults()["removed"]
                return f"cleared {len(removed)} fault(s)", "msg ok"
            injected = clients.inject(
                kind=kind,
                magnitude=None if magnitude in (None, "") else float(magnitude),
                tag=tag if kind in SENSOR_KINDS else None,
                ramp=float(ramp or 0.0))
            ws.finish_run(ws.start_run("fault", config=injected))
            return (f"injected {injected['fault_id']} "
                    f"({injected['kind']} on {injected['target']})"), "msg bad"
        except StackUnavailable as exc:
            return str(exc), "msg bad"

    @app.callback(Output("fault-list", "children"), Output("truth", "children"),
                  Output("fault-pill", "children"),
                  Output("fault-pill", "className"),
                  Input("tick", "n_intervals"))
    def fault_view(_):
        try:
            listing = clients.faults()
            truth = clients.truth()
        except StackUnavailable:
            blank = html.Div("stack not running", className="hint")
            return blank, blank, "faults --", "pill"
        active = set(listing["active"])
        rows = [[f["fault_id"], f["kind"], f["target"],
                 html.Span("active" if f["fault_id"] in active else "pending",
                           className="pill " + ("bad" if f["fault_id"] in active
                                                else "")),
                 fmt(f["magnitude"], 6), fmt(f["t_start"], 0),
                 fmt(f["ramp"], 0)] for f in listing["faults"]]
        compare = []
        for name in ("LT101", "LT102", "FT101", "PT101", "ZT101"):
            true, reported = truth["true"].get(name), truth["reported"].get(name)
            try:
                delta = float(reported) - float(true)
            except (TypeError, ValueError):
                delta = None
            compare.append([name, fmt(true, 2), fmt(reported, 2),
                            html.Span(fmt(delta, 2),
                                      style={"color": "var(--bad)"}
                                      if delta and abs(delta) > 0.5 else {})])
        return (table(["id", "kind", "target", "state", "magnitude",
                       "start (s)", "ramp (s)"], rows, "Plant is healthy."),
                table(["tag", "true", "reported", "difference"], compare),
                f"faults {len(active)}",
                "pill bad" if active else "pill on")

    # -- jobs --------------------------------------------------------------

    @app.callback(Output("d-msg", "children"), Output("d-msg", "className"),
                  Input("d-go", "n_clicks"), Input("d-cancel", "n_clicks"),
                  State("d-duration", "value"), State("d-mag", "value"),
                  State("d-rep", "value"), State("d-normal", "value"),
                  prevent_initial_call=True)
    def dataset(go_clicks, cancel_clicks, duration, magnitudes, repeats, normal):
        which = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
        if which == "d-cancel":
            active = runner.active("dataset")
            if active is None:
                return "nothing to cancel", "msg"
            runner.cancel(active.id)
            return "cancelled", "msg"
        try:
            ws.update(data=replace(ws.config.data, duration=float(duration),
                                   magnitudes=int(magnitudes),
                                   repeats=int(repeats),
                                   normal_runs=int(normal)))
            runner.submit("dataset", generate_dataset, workspace=str(ws.path),
                          duration=float(duration), magnitudes=int(magnitudes),
                          repeats=int(repeats), normal_runs=int(normal))
            return "started", "msg ok"
        except Exception as exc:  # noqa: BLE001
            return str(exc), "msg bad"

    @app.callback(Output("m-msg", "children"), Output("m-msg", "className"),
                  Input("m-go", "n_clicks"), State("m-window", "value"),
                  State("m-stride", "value"), State("m-quantile", "value"),
                  State("m-methods", "value"), prevent_initial_call=True)
    def train(_, window, stride, quantile, methods):
        if not ws.dataset_path.exists():
            return "generate a dataset first, on the Data tab", "msg bad"
        if not methods:
            return "tick at least one method", "msg bad"
        try:
            ws.update(data=replace(ws.config.data, window=int(window),
                                   stride=int(stride),
                                   quantile=float(quantile)))
            runner.submit("training", train_detector, workspace=str(ws.path),
                          window=int(window), stride=int(stride),
                          quantile=float(quantile), methods=list(methods))
            return f"started: {', '.join(methods)}", "msg ok"
        except Exception as exc:  # noqa: BLE001
            return str(exc), "msg bad"

    @app.callback(Output("d-bar", "style"), Output("m-bar", "style"),
                  Output("run-log", "children"), Output("m-metrics", "children"),
                  Output("m-modes", "children"),
                  Input("slow", "n_intervals"))
    def job_progress(_):
        def bar(kind):
            state = runner.all(kind)
            if not state:
                return {"width": "0%"}
            return {"width": f"{state[0].progress * 100:.0f}%"}

        rows = []
        for record in ws.runs(limit=12):
            rows.append([record.kind, record.status, f"{record.seconds:.0f} s",
                         json.dumps(record.result)[:70]])
        metrics_path = ws.artifacts / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            summary = table(["model", "detected", "median delay",
                             "false alarms/day", "flagged windows",
                             "isolation"], metrics_rows(metrics))
            modes = table(["fault", "runs", "detected", "median delay",
                           "top channel", "expected"], per_mode_rows(metrics))
        else:
            summary = html.Div("no model trained yet", className="hint")
            modes = html.Div("", className="hint")
        return (bar("dataset"), bar("training"),
                table(["kind", "status", "took", "result"], rows,
                      "nothing run yet"),
                summary, modes)


BY_DESC = {s.name: s.description for s in SERVICES}
PANELS = ("quickstart", "runtime", "faults", "data", "model")


def main() -> None:  # pragma: no cover - entry point
    import argparse

    ap = argparse.ArgumentParser(description="pyplant workbench")
    ap.add_argument("--workspace", default="./work/demo")
    ap.add_argument("--base-port", type=int, default=5400)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    app = build_app(args.workspace, args.base_port)
    print(f"workbench on http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":  # pragma: no cover
    main()
