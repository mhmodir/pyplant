"""HMI process entry point.

Serves the operator page and its JSON API. It attaches to the historian
read-only: the collector in `services.historian` owns writing, and two
processes writing one DuckDB file is not something to invite.
"""

from __future__ import annotations

import argparse

import uvicorn

from pyplant.hist.db import Historian
from pyplant.hmi.app import HmiState, RemoteHistorian, create_app
from pyplant.net.opcua import DEFAULT_ENDPOINT


def build(endpoint: str, history: str | None, poll: float = 0.5):
    """`history` is a URL when the collector is a separate process, or a path
    when the HMI owns the database (tests and single-process demos)."""
    if history is None:
        source = None
    elif history.startswith("http"):
        source = RemoteHistorian(history.rstrip("/"))
    else:
        source = Historian(history)
    return create_app(HmiState(endpoint=endpoint, db=source, poll_period=poll))


def main() -> None:  # pragma: no cover - process entry point
    ap = argparse.ArgumentParser(description="pyplant operator HMI")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--history", default=None,
                    help="historian query URL, or a DuckDB path if this "
                         "process is the only one using it")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--poll", type=float, default=0.5)
    args = ap.parse_args()

    app = build(args.endpoint, args.history, args.poll)
    print(f"hmi on http://{args.host}:{args.port} "
          f"reading {args.endpoint}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":  # pragma: no cover
    main()
