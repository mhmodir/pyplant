"""The HMI service.

A FastAPI app in front of three things that already exist: the OPC UA server
for live values and operator writes, the historian for trends, and the alarm
manager for the alarm list. It holds no process state of its own.

Two rules it enforces on the way through:

* Only tags the dictionary marks writable can be written, and the write goes to
  OPC UA rather than to any local copy. The HMI is a client of the control
  system, not a participant in it.
* Setpoints are clamped and rejected outside their span before they reach the
  wire, so a fat-fingered 3000 mm produces a clear error rather than a silently
  clamped value the operator never notices.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

import httpx

from pyplant.hist.db import Historian
from pyplant.hmi.alarms import AlarmManager, alarm_tags
from pyplant.net.opcua import DEFAULT_ENDPOINT, OpcClient
from pyplant.services.historian import default_tags
from pyplant.tags import TAGS

PAGE = Path(__file__).with_name("page.html")

MIMIC_TAGS = ("LT101", "LT102", "FT101", "FT102", "PT101", "ZT101",
              "PMP101_RUN", "LIC101_SP", "LIC101_PV", "LIC101_OP",
              "LIC101_ERR", "LIC101_MAN_OP", "LIC101_AUTO",
              "CMD_PMP101_START", "SCAN_MS", "SCAN_OVERRUNS", "PLC_HEARTBEAT")

TREND_DEFAULT = ("LT101", "LIC101_SP", "LIC101_OP")


class SetpointBody(BaseModel):
    value: float


class ModeBody(BaseModel):
    auto: bool


class PumpBody(BaseModel):
    start: bool


class ShelveBody(BaseModel):
    tag: str
    minutes: float = Field(default=30.0, gt=0.0)
    reason: str = ""


class AckBody(BaseModel):
    tag: str | None = None  # None acknowledges everything


@dataclass
class RemoteHistorian:
    """The historian, over HTTP.

    DuckDB will not let two processes open one database, so the HMI cannot read
    the collector's file directly. It asks the collector instead. The in-process
    `Historian` is still accepted, which is what the tests use.
    """

    url: str
    timeout: float = 3.0

    def window(self, tags: list[str], seconds: float, freq: float) -> dict:
        response = httpx.get(f"{self.url}/api/window", timeout=self.timeout,
                             params={"tags": ",".join(tags), "seconds": seconds,
                                     "freq": freq})
        if response.status_code >= 400:
            raise HTTPException(response.status_code,
                                response.json().get("detail", "history error"))
        return response.json()


def _window_payload(source, tags: list[str], seconds: float,
                    freq: float) -> dict:
    if isinstance(source, RemoteHistorian):
        return source.window(tags, seconds, freq)
    frame = source.latest_window(tags, seconds=seconds, freq=freq)
    if frame.empty:
        return {"t": [], "series": {t: [] for t in tags}}
    return {"t": [ts.isoformat() for ts in frame.index],
            "series": {t: [None if v != v else float(v) for v in frame[t]]
                       for t in tags}}


@dataclass
class HmiState:
    """Live snapshot plus the alarm list, refreshed by a background task."""

    endpoint: str = DEFAULT_ENDPOINT
    db: "Historian | RemoteHistorian | None" = None
    poll_period: float = 0.5
    connect_retries: int = 40
    client: OpcClient | None = field(init=False, default=None)
    alarms: AlarmManager = field(default_factory=AlarmManager)
    values: dict[str, float | bool] = field(default_factory=dict)
    updated: datetime | None = field(init=False, default=None)
    connected: bool = field(init=False, default=False)
    polls: int = 0
    errors: int = 0
    available: list[str] = field(default_factory=list, init=False)
    missing: list[str] = field(default_factory=list, init=False)
    _task: asyncio.Task | None = field(init=False, default=None)

    @property
    def watched(self) -> list[str]:
        """Everything the page would like to show."""
        return sorted((set(MIMIC_TAGS) | set(alarm_tags())) & set(TAGS))

    async def _resolve(self) -> None:
        """Keep only the tags this server actually publishes.

        A display configured for a point its controller does not expose is
        normal - `ALM_HEALTH` has no publisher until the inference service
        exists - and it must degrade to a blank field, not take the page down.
        """
        assert self.client is not None
        self.available, self.missing = [], []
        for tag in self.watched:
            try:
                await self.client.node(tag)
                self.available.append(tag)
            except Exception:  # noqa: BLE001 - asyncua raises BadNoMatch
                self.missing.append(tag)

    async def start(self) -> None:
        self.client = OpcClient(self.endpoint)
        await self.client.connect(retries=self.connect_retries)
        await self._resolve()
        self.connected = True
        await self.poll()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self.client is not None:
            await self.client.close()
            self.client = None
        self.connected = False

    async def poll(self) -> None:
        assert self.client is not None
        self.values = await self.client.read_many(self.available)
        self.alarms.update(self.values)
        self.updated = datetime.utcnow()
        self.polls += 1

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.poll_period)
            try:
                await self.poll()
            except Exception:  # noqa: BLE001 - a dropped link must not kill the page
                self.errors += 1
                self.connected = False

    async def write(self, tag: str, value: float | bool) -> None:
        if tag not in TAGS:
            raise HTTPException(404, f"unknown tag {tag}")
        spec = TAGS[tag]
        if not spec.writable:
            raise HTTPException(403, f"{tag} is not operator-writable")
        if spec.kind.name == "ANALOG" and not (spec.lo <= float(value) <= spec.hi):
            raise HTTPException(
                422, f"{value} is outside {tag} span "
                     f"[{spec.lo}, {spec.hi}] {spec.unit}")
        if self.client is None:
            raise HTTPException(503, "not connected to the control system")
        await self.client.write(tag, value)
        self.values[tag] = value


def create_app(state: HmiState) -> FastAPI:
    app = FastAPI(title="pyplant HMI")
    app.state.hmi = state

    @app.on_event("startup")
    async def _startup() -> None:  # pragma: no cover - exercised by the service
        if state.client is None:
            await state.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:  # pragma: no cover
        await state.stop()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return PAGE.read_text()

    @app.get("/api/state")
    async def get_state() -> JSONResponse:
        return JSONResponse({
            "connected": state.connected,
            "updated": state.updated.isoformat() if state.updated else None,
            "values": {k: (bool(v) if isinstance(v, bool) else float(v))
                       for k, v in state.values.items()},
            "alarms": state.alarms.active_list(),
            "summary": state.alarms.summary(),
            "polls": state.polls,
            "errors": state.errors,
            "missing": state.missing,
        })

    @app.get("/api/tags")
    async def get_tags() -> JSONResponse:
        return JSONResponse({
            name: {"desc": t.desc, "unit": t.unit, "lo": t.lo, "hi": t.hi,
                   "kind": t.kind.name, "writable": t.writable}
            for name, t in TAGS.items()})

    @app.get("/api/trend")
    async def trend(tags: str = ",".join(TREND_DEFAULT),
                    seconds: float = 300.0,
                    freq: float = 1.0) -> JSONResponse:
        if state.db is None:
            raise HTTPException(503, "no historian is attached")
        wanted = [t.strip() for t in tags.split(",") if t.strip()]
        unknown = [t for t in wanted if t not in TAGS]
        if unknown:
            raise HTTPException(404, f"unknown tags: {unknown}")
        try:
            return JSONResponse(_window_payload(state.db, wanted, seconds, freq))
        except httpx.HTTPError as exc:
            raise HTTPException(503, f"history unavailable: {exc}") from exc

    @app.post("/api/setpoint")
    async def set_setpoint(body: SetpointBody) -> dict:
        await state.write("LIC101_SP", body.value)
        return {"LIC101_SP": body.value}

    @app.post("/api/manual")
    async def set_manual(body: SetpointBody) -> dict:
        await state.write("LIC101_MAN_OP", body.value)
        return {"LIC101_MAN_OP": body.value}

    @app.post("/api/mode")
    async def set_mode(body: ModeBody) -> dict:
        await state.write("LIC101_AUTO", body.auto)
        return {"LIC101_AUTO": body.auto}

    @app.post("/api/pump")
    async def set_pump(body: PumpBody) -> dict:
        await state.write("CMD_PMP101_START", body.start)
        return {"CMD_PMP101_START": body.start}

    @app.post("/api/reset")
    async def reset() -> dict:
        """Pulse the reset bit. A latched interlock needs an edge, not a level."""
        await state.write("CMD_ALM_RESET", True)
        await asyncio.sleep(max(state.poll_period, 0.3))
        await state.write("CMD_ALM_RESET", False)
        return {"reset": True}

    @app.post("/api/alarms/ack")
    async def ack(body: AckBody) -> dict:
        if body.tag is None:
            return {"acknowledged": state.alarms.acknowledge_all()}
        if body.tag not in state.alarms.alarms:
            raise HTTPException(404, f"{body.tag} is not an alarm point")
        return {"acknowledged": int(state.alarms.acknowledge(body.tag))}

    @app.post("/api/alarms/shelve")
    async def shelve(body: ShelveBody) -> dict:
        if body.tag not in state.alarms.alarms:
            raise HTTPException(404, f"{body.tag} is not an alarm point")
        try:
            state.alarms.shelve(body.tag, body.minutes, body.reason)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return state.alarms.alarms[body.tag].as_dict()

    @app.post("/api/alarms/unshelve")
    async def unshelve(body: AckBody) -> dict:
        if body.tag is None or body.tag not in state.alarms.alarms:
            raise HTTPException(404, "unknown alarm point")
        state.alarms.unshelve(body.tag)
        return state.alarms.alarms[body.tag].as_dict()

    return app
