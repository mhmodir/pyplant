"""The historian collector.

Subscribes to the PLC's OPC UA server and buffers what arrives into DuckDB.

Exception reporting plus a heartbeat
------------------------------------
A subscription already pushes only on change, which is most of what a
historian's compression does. What it does not give you is an anchor for a tag
that has been quiet for an hour: a window query opening after the last change
would find nothing to carry forward. So each tag is also re-stored if it has
been silent for `heartbeat` seconds. That is the classic exception-reporting
pair — store on change, and never let the gap exceed a maximum.

Writes are batched. Inserting one row at a time is the usual way to make DuckDB
look slow, so notifications land in a buffer and a flush task drains it.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException

from pyplant.hist.db import Historian, Sample
from pyplant.net.opcua import DEFAULT_ENDPOINT, OpcClient
from pyplant.services.plc import PLC_BLOCKS
from pyplant.tags import TAGS, tags_in


def default_tags() -> tuple[str, ...]:
    """Everything the PLC serves: process, faceplate, alarms, diagnostics."""
    return tuple(t.name for b in PLC_BLOCKS for t in tags_in(b))


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass
class CollectorStats:
    received: int = 0
    stored: int = 0
    heartbeats: int = 0
    flushes: int = 0
    bad: int = 0


@dataclass
class HistorianService:
    """OPC UA subscription in, batched DuckDB writes out."""

    db: Historian
    endpoint: str = DEFAULT_ENDPOINT
    tags: tuple[str, ...] = field(default_factory=default_tags)
    publish_ms: int = 200
    flush_period: float = 1.0
    heartbeat: float = 30.0  # s, maximum gap between stored samples per tag
    connect_retries: int = 40  # the PLC may still be opening its endpoint

    client: OpcClient | None = field(init=False, default=None)
    stats: CollectorStats = field(default_factory=CollectorStats)
    _buffer: list[Sample] = field(default_factory=list, init=False)
    _last: dict[str, tuple[datetime, float]] = field(default_factory=dict,
                                                     init=False)
    _sub = None
    _tasks: list[asyncio.Task] = field(default_factory=list, init=False)

    # -- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        self.client = OpcClient(self.endpoint)
        await self.client.connect(retries=self.connect_retries)
        self._sub = await self.client.subscribe(list(self.tags), self._on_change,
                                                self.publish_ms)
        self._tasks = [asyncio.create_task(self._flush_loop()),
                       asyncio.create_task(self._heartbeat_loop())]

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks = []
        if self._sub is not None:
            with contextlib.suppress(Exception):
                await self._sub.delete()
            self._sub = None
        self.flush()  # do not lose what is still buffered
        if self.client is not None:
            await self.client.close()
            self.client = None

    async def __aenter__(self) -> "HistorianService":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    # -- collection -------------------------------------------------------

    def _on_change(self, name: str, value: float | bool, good: bool) -> None:
        ts = _now()
        self.stats.received += 1
        if not good:
            self.stats.bad += 1
        self._buffer.append(Sample.of(ts, name, value, good))
        if good:
            self._last[name] = (ts, float(value))

    def flush(self) -> int:
        if not self._buffer:
            return 0
        batch, self._buffer = self._buffer, []
        written = self.db.write(batch)
        self.stats.stored += written
        self.stats.flushes += 1
        return written

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(self.flush_period)
            with contextlib.suppress(Exception):
                self.flush()

    async def _heartbeat_loop(self) -> None:
        """Re-store any tag that has gone quiet for longer than `heartbeat`."""
        interval = max(self.heartbeat / 4.0, 0.05)
        while True:
            await asyncio.sleep(interval)
            now = _now()
            for name, (ts, value) in list(self._last.items()):
                if (now - ts).total_seconds() >= self.heartbeat:
                    self._buffer.append(Sample.of(now, name, value))
                    self._last[name] = (now, value)
                    self.stats.heartbeats += 1


def create_query_app(db: Historian, service: "HistorianService | None" = None
                     ) -> FastAPI:
    """Read access to the history, over HTTP.

    DuckDB takes an exclusive lock on its file, so a second process cannot open
    the same database even to read. That is not a limitation to work around: a
    historian is a service you query, not a file you share. Everything that
    wants history asks here, and exactly one process writes.
    """
    app = FastAPI(title="pyplant historian")

    @app.get("/api/health")
    async def health() -> dict:
        t0, t1 = db.span()
        return {"rows": db.count(), "tags": len(db.tags()),
                "from": t0.isoformat() if t0 else None,
                "to": t1.isoformat() if t1 else None,
                "stats": vars(service.stats) if service else {}}

    @app.get("/api/window")
    async def window(tags: str, seconds: float = 300.0,
                     freq: float = 1.0) -> dict:
        wanted = [t.strip() for t in tags.split(",") if t.strip()]
        unknown = [t for t in wanted if t not in TAGS]
        if unknown:
            raise HTTPException(404, f"unknown tags: {unknown}")
        frame = db.latest_window(wanted, seconds=seconds, freq=freq)
        if frame.empty:
            return {"t": [], "series": {t: [] for t in wanted}}
        return {"t": [ts.isoformat() for ts in frame.index],
                "series": {t: [None if v != v else float(v)
                               for v in frame[t]] for t in wanted}}

    @app.get("/api/latest")
    async def latest(tags: str = "") -> dict:
        wanted = [t.strip() for t in tags.split(",") if t.strip()] or None
        return db.latest(wanted)

    return app


async def main() -> None:  # pragma: no cover - process entry point
    ap = argparse.ArgumentParser(description="pyplant historian collector")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--db", default="pyplant.duckdb")
    ap.add_argument("--publish-ms", type=int, default=200)
    ap.add_argument("--flush", type=float, default=1.0)
    ap.add_argument("--heartbeat", type=float, default=30.0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--api-port", type=int, default=0,
                    help="serve the query API here; 0 disables it")
    args = ap.parse_args()

    db = Historian(args.db)
    service = HistorianService(db=db, endpoint=args.endpoint,
                               publish_ms=args.publish_ms,
                               flush_period=args.flush, heartbeat=args.heartbeat)
    await service.start()
    print(f"historian collecting {len(service.tags)} tags from {args.endpoint} "
          f"into {args.db}", flush=True)

    server = None
    if args.api_port:
        import uvicorn

        server = uvicorn.Server(uvicorn.Config(
            create_query_app(db, service), host=args.host, port=args.api_port,
            log_level="warning"))
        print(f"query api on http://{args.host}:{args.api_port}", flush=True)

    try:
        if server is not None:
            await server.serve()
        else:
            while True:
                await asyncio.sleep(10.0)
    finally:
        if server is not None:
            server.should_exit = True
        await service.stop()
        db.close()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
