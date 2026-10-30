"""The stack supervisor.

Starts the four services as subprocesses, watches them, and stops them again.
Subprocesses rather than tasks inside the workbench, for three reasons:

* It is what step 12's compose file will do. The workbench and the deployment
  then mean the same thing by "a running stack" instead of being two different
  arrangements that drift apart.
* A crash in the plant does not take the UI down with it, which matters when
  the UI is how you find out something crashed.
* The processes are exactly the ones a user can start by hand from the README,
  so the workbench teaches the command line rather than hiding it. `command()`
  returns the line it ran, and the Runtime page shows it.

Startup order is not arbitrary. The PLC polls the plant, the historian
subscribes to the PLC, and the HMI reads both — so each waits for the one below
it to answer before the next is launched. Shutdown reverses it.
"""

from __future__ import annotations

import contextlib
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from pyplant.workbench.workspace import Workspace

START_TIMEOUT = 25.0
STOP_TIMEOUT = 8.0


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    module: str
    port_attrs: tuple[str, ...]  # every port it must be listening on to be up
    description: str


# In dependency order. Each is only started once the one before it answers.
SERVICES: tuple[ServiceSpec, ...] = (
    ServiceSpec("plant", "pyplant.services.plant",
                ("plant_modbus", "plant_control"),
                "physics, Modbus slave, fault injection API"),
    # Both of the PLC's ports, because the historian and the HMI depend on the
    # OPC UA endpoint, which opens after the Modbus one. Waiting on Modbus
    # alone let the next service race it.
    ServiceSpec("plc", "pyplant.services.plc", ("plc_modbus", "plc_opcua"),
                "scan engine, Modbus client and server, OPC UA"),
    ServiceSpec("historian", "pyplant.services.historian", ("historian_api",),
                "OPC UA subscription into DuckDB, and the query API over it"),
    ServiceSpec("hmi", "pyplant.services.hmi", ("hmi",),
                "operator page and JSON API"),
)

BY_NAME = {s.name: s for s in SERVICES}


def port_is_open(port: int, host: str = "127.0.0.1",
                 timeout: float = 0.3) -> bool:
    with socket.socket() as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def wait_for_port(port: int, timeout: float = START_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_is_open(port):
            return True
        time.sleep(0.2)
    return False


@dataclass
class ServiceState:
    name: str
    running: bool
    pid: int | None
    port: int | None
    listening: bool
    returncode: int | None
    command: str
    log_tail: str = ""


@dataclass
class Supervisor:
    """Owns the four processes for one workspace."""

    workspace: Workspace
    python: str = field(default_factory=lambda: sys.executable)
    _procs: dict[str, subprocess.Popen] = field(default_factory=dict, init=False)
    _logs: dict[str, Path] = field(default_factory=dict, init=False)

    # -- command lines -----------------------------------------------------

    def command(self, name: str) -> list[str]:
        """The exact command line, so the UI can show what it ran."""
        config = self.workspace.config
        ports = config.ports
        opc = f"opc.tcp://127.0.0.1:{ports.plc_opcua}/pyplant/"
        base = [self.python, "-m", BY_NAME[name].module]
        if name == "plant":
            return base + ["--host", "127.0.0.1",
                           "--port", str(ports.plant_modbus),
                           "--control-port", str(ports.plant_control),
                           "--dt", str(config.dt),
                           "--speed", str(config.speed),
                           "--start", "steady"]
        if name == "plc":
            return base + ["--plant-port", str(ports.plant_modbus),
                           "--host", "127.0.0.1",
                           "--port", str(ports.plc_modbus),
                           "--opcua-port", str(ports.plc_opcua),
                           "--period", str(config.loop.scan_period),
                           "--speed", str(config.speed),
                           "--setpoint", str(config.loop.setpoint)]
        if name == "historian":
            return base + ["--endpoint", opc,
                           "--db", str(self.workspace.history_path),
                           "--host", "127.0.0.1",
                           "--api-port", str(ports.historian_api)]
        if name == "hmi":
            return base + ["--endpoint", opc,
                           "--history", self.history_url,
                           "--host", "127.0.0.1", "--port", str(ports.hmi)]
        raise KeyError(name)  # pragma: no cover

    def ports_of(self, name: str) -> list[int]:
        ports = self.workspace.config.ports
        return [getattr(ports, attr) for attr in BY_NAME[name].port_attrs]

    def port_of(self, name: str) -> int | None:
        ports = self.ports_of(name)
        return ports[0] if ports else None

    def log_path(self, name: str) -> Path:
        return self.workspace.jobs / f"{name}.log"

    # -- lifecycle ---------------------------------------------------------

    def start_one(self, name: str) -> ServiceState:
        if self.is_running(name):
            return self.state(name)
        log = self.log_path(name)
        handle = log.open("w")
        self._logs[name] = log
        self._procs[name] = subprocess.Popen(
            self.command(name), stdout=handle, stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).resolve().parents[2]))
        for port in self.ports_of(name):
            if not wait_for_port(port):
                state = self.state(name)
                self.stop_one(name)
                raise RuntimeError(
                    f"{name} did not start listening on {port}. Log tail:\n"
                    f"{state.log_tail or '(empty)'}")
        return self.state(name)

    def start(self, names: Iterable[str] | None = None) -> list[ServiceState]:
        """Start in dependency order, refusing if the ports are already taken."""
        wanted = list(names) if names else [s.name for s in SERVICES]
        busy = [p for p in self.workspace.config.ports.in_use()
                if not self._owns(p)]
        if busy:
            raise RuntimeError(
                f"ports already in use by something else: {busy}. Another "
                f"workspace is probably running; give this one a different "
                f"base port.")
        states = []
        for spec in SERVICES:
            if spec.name in wanted:
                states.append(self.start_one(spec.name))
        return states

    def _owns(self, port_name: str) -> bool:
        return any(port_name in spec.port_attrs and self.is_running(name)
                   for name, spec in BY_NAME.items())

    def stop_one(self, name: str) -> None:
        process = self._procs.pop(name, None)
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                process.kill()
                with contextlib.suppress(Exception):
                    process.wait(timeout=2.0)

    def stop(self) -> None:
        """Reverse dependency order, so nothing is left polling a dead peer."""
        for spec in reversed(SERVICES):
            self.stop_one(spec.name)

    def restart(self) -> list[ServiceState]:
        self.stop()
        return self.start()

    # -- inspection --------------------------------------------------------

    def is_running(self, name: str) -> bool:
        process = self._procs.get(name)
        return process is not None and process.poll() is None

    def state(self, name: str) -> ServiceState:
        process = self._procs.get(name)
        port = self.port_of(name)
        return ServiceState(
            name=name,
            running=self.is_running(name),
            pid=process.pid if process else None,
            port=port,
            listening=all(port_is_open(p) for p in self.ports_of(name))
            if self.ports_of(name) else self.is_running(name),
            returncode=process.poll() if process else None,
            command=" ".join(self.command(name)),
            log_tail=self.tail(name),
        )

    def states(self) -> list[ServiceState]:
        return [self.state(s.name) for s in SERVICES]

    def healthy(self) -> bool:
        return all(s.running and s.listening for s in self.states())

    def tail(self, name: str, lines: int = 12) -> str:
        log = self.log_path(name)
        if not log.exists():
            return ""
        return "\n".join(log.read_text().splitlines()[-lines:])

    # -- endpoints the UI talks to ----------------------------------------

    @property
    def history_url(self) -> str:
        return f"http://127.0.0.1:{self.workspace.config.ports.historian_api}"

    @property
    def hmi_url(self) -> str:
        return f"http://127.0.0.1:{self.workspace.config.ports.hmi}"

    @property
    def plant_control_url(self) -> str:
        return f"http://127.0.0.1:{self.workspace.config.ports.plant_control}"

    @property
    def opcua_endpoint(self) -> str:
        return f"opc.tcp://127.0.0.1:{self.workspace.config.ports.plc_opcua}/pyplant/"

    def __enter__(self) -> "Supervisor":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
