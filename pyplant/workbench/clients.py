"""Synchronous clients for a running stack.

Dash callbacks are synchronous; OPC UA is not. Rather than run an event loop
beside the web server, the workbench talks HTTP to the APIs the services
already expose — the HMI's operator API, the plant's fault API, the historian's
query API. Nothing here reimplements control logic, physics or thresholds. If
that ever changes, the workbench has become a second implementation of the
plant and will disagree with the first one.

Every call returns data or raises `StackUnavailable`. A workbench whose page
goes blank because a service is still starting is worse than one that says so.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from pyplant.workbench.workspace import Ports

TIMEOUT = httpx.Timeout(4.0, connect=0.5)

# Windows takes about two seconds to refuse a connection to a closed local
# port. The page polls several endpoints every second, so with the stack down
# those waits pile up until every callback is a minute behind. A service that
# just refused is reported down at once until this many seconds have passed.
DOWN_HOLD = 2.5
_down_since: dict[str, float] = {}


class StackUnavailable(RuntimeError):
    """A service did not answer, or answered with an error."""


def forget_failures() -> None:
    """Call after starting services, so they are asked again at once."""
    _down_since.clear()


def _request(method: str, url: str, **kwargs) -> dict:
    origin = httpx.URL(url).netloc.decode()
    failed = _down_since.get(origin)
    if failed is not None and time.monotonic() - failed < DOWN_HOLD:
        raise StackUnavailable(f"{origin} is not answering")
    try:
        response = httpx.request(method, url, timeout=TIMEOUT, **kwargs)
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        _down_since[origin] = time.monotonic()
        raise StackUnavailable(f"{url} did not answer: {exc}") from exc
    except httpx.HTTPError as exc:
        raise StackUnavailable(f"{url} did not answer: {exc}") from exc
    _down_since.pop(origin, None)
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:  # noqa: BLE001
            detail = response.text
        raise StackUnavailable(str(detail))
    return response.json()


@dataclass
class StackClients:
    """One handle for all three service APIs."""

    ports: Ports
    host: str = "127.0.0.1"

    @property
    def hmi(self) -> str:
        return f"http://{self.host}:{self.ports.hmi}"

    @property
    def plant(self) -> str:
        return f"http://{self.host}:{self.ports.plant_control}"

    @property
    def history(self) -> str:
        return f"http://{self.host}:{self.ports.historian_api}"

    # -- process values and operator actions ------------------------------

    def state(self) -> dict:
        return _request("GET", f"{self.hmi}/api/state")

    def trend(self, tags: list[str], seconds: float = 300.0,
              freq: float = 1.0) -> dict:
        return _request("GET", f"{self.hmi}/api/trend",
                        params={"tags": ",".join(tags), "seconds": seconds,
                                "freq": freq})

    def setpoint(self, value: float) -> dict:
        return _request("POST", f"{self.hmi}/api/setpoint", json={"value": value})

    def manual(self, value: float) -> dict:
        return _request("POST", f"{self.hmi}/api/manual", json={"value": value})

    def mode(self, auto: bool) -> dict:
        return _request("POST", f"{self.hmi}/api/mode", json={"auto": auto})

    def pump(self, start: bool) -> dict:
        return _request("POST", f"{self.hmi}/api/pump", json={"start": start})

    def reset(self) -> dict:
        return _request("POST", f"{self.hmi}/api/reset", json={})

    def acknowledge(self, tag: str | None = None) -> dict:
        return _request("POST", f"{self.hmi}/api/alarms/ack", json={"tag": tag})

    def shelve(self, tag: str, minutes: float = 30.0, reason: str = "") -> dict:
        return _request("POST", f"{self.hmi}/api/alarms/shelve",
                        json={"tag": tag, "minutes": minutes, "reason": reason})

    # -- faults ------------------------------------------------------------

    def faults(self) -> dict:
        return _request("GET", f"{self.plant}/api/faults")

    def inject(self, kind: str, magnitude: float | None = None,
               tag: str | None = None, ramp: float = 0.0,
               duration: float | None = None, delay: float = 0.0,
               fault_id: str | None = None) -> dict:
        body: dict = {"kind": kind, "ramp": ramp, "delay": delay}
        if magnitude is not None:
            body["magnitude"] = magnitude
        if tag:
            body["tag"] = tag
        if duration:
            body["duration"] = duration
        if fault_id:
            body["id"] = fault_id
        return _request("POST", f"{self.plant}/api/fault", json=body)

    def clear_fault(self, fault_id: str) -> dict:
        return _request("DELETE", f"{self.plant}/api/fault/{fault_id}")

    def clear_faults(self) -> dict:
        return _request("DELETE", f"{self.plant}/api/faults")

    def truth(self) -> dict:
        """Ground truth beside what the instruments report.

        The control system cannot see this, and neither can the diagnosis
        model. It is here so a person can check whether a detector was right.
        """
        return _request("GET", f"{self.plant}/api/truth")

    # -- history -----------------------------------------------------------

    def history_health(self) -> dict:
        return _request("GET", f"{self.history}/api/health")

    def reachable(self) -> bool:
        try:
            self.state()
            return True
        except StackUnavailable:
            return False
