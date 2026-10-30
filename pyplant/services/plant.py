"""The plant node.

A process that integrates the physics in real time and presents itself on the
network as a Modbus slave, the way a field device would. It publishes its
measurements and reads back the actuator commands the PLC writes into it.

The plant is deliberately dumb. It has no controller, no alarms and no memory
of what the PLC intends; it applies whatever is currently in its command
registers. If the PLC stops writing, the last command simply persists, which is
exactly what happens to a real valve when its controller dies.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pyplant.clock import Ticker
from pyplant.core.hybrid_tank import Inputs, Params, clip_to_tag_ranges, steady_state
from pyplant.faults.model import (
    FAULT_KINDS,
    Fault,
    FaultSet,
    SensorNoise,
    build_fault,
)
from pyplant.net.link import TagServer
from pyplant.net.store import TagStore
from pyplant.runners.base import Simulation

PLANT_BLOCKS = ("PLANT_MEAS", "PLANT_STATUS", "PLC_CMD", "PLC_CMD_BITS")


@dataclass
class PlantNode:
    """Physics plus a Modbus server, ticking on a clock."""

    sim: Simulation
    host: str = "127.0.0.1"
    port: int = 5020
    speed: float = 1.0
    store: TagStore = field(default_factory=lambda: TagStore(PLANT_BLOCKS))
    server: TagServer = field(init=False)
    ticker: Ticker = field(init=False)
    faults: FaultSet = field(default_factory=FaultSet)
    steps: int = 0

    def __post_init__(self) -> None:
        if self.sim.policy is not None:
            raise ValueError("the plant node takes its commands over Modbus, "
                             "not from a policy")
        # Faults live on the plant, where the physics is. Nothing upstream is
        # told a fault was injected; that is the point of the exercise.
        self.sim.faults = self.faults
        self.server = TagServer(self.store, host=self.host, port=self.port)
        self.ticker = Ticker(self.sim.dt, speed=self.speed)
        self._publish()
        # Seed the command registers from the plant's own initial input so a
        # PLC that has not scanned yet does not slam the valve to zero.
        self.store.publish("PLC_CMD", {"FCV101_OP": self.sim.u.valve_cmd * 100.0})
        self.store.publish("PLC_CMD_BITS", {"PMP101_CMD": self.sim.u.pump_on})

    async def start(self) -> int:
        self.port = await self.server.start()
        return self.port

    async def stop(self) -> None:
        await self.server.stop()

    # -- one tick ----------------------------------------------------------

    def _apply_commands(self) -> None:
        cmd = self.store.read("PLC_CMD")
        bits = self.store.read("PLC_CMD_BITS")
        self.sim.u = Inputs(valve_cmd=float(cmd["FCV101_OP"]) / 100.0,
                            pump_on=bool(bits["PMP101_CMD"]))

    def _publish(self) -> None:
        m = clip_to_tag_ranges(self.sim.measurements)
        self.store.publish("PLANT_MEAS", m)
        self.store.publish("PLANT_STATUS", m)

    def tick(self) -> None:
        self._apply_commands()
        self.sim.integrate()
        self._publish()
        self.steps += 1

    async def run(self, steps: int | None = None,
                  stop: asyncio.Event | None = None) -> None:
        if steps is None and stop is None:
            raise ValueError("run() needs a step count or a stop event")
        done = 0
        while True:
            await self.ticker.wait()
            self.tick()
            done += 1
            if steps is not None and done >= steps:
                return
            if stop is not None and stop.is_set():
                return


class FaultBody(BaseModel):
    kind: str
    id: str | None = None
    magnitude: float | None = None
    tag: str | None = None
    delay: float = Field(default=0.0, ge=0.0)  # s from now
    duration: float | None = Field(default=None, gt=0.0)
    ramp: float = Field(default=0.0, ge=0.0)


def create_control_app(node: "PlantNode") -> FastAPI:
    """Fault injection for the live plant.

    Deliberately a separate endpoint from anything the control system can see.
    An operator, an HMI or a diagnosis model must not be able to find out that
    a fault was injected except by observing the plant.
    """
    app = FastAPI(title="pyplant plant control")

    @app.get("/api/faults")
    async def list_faults() -> dict:
        """Real faults only.

        Instrument noise is modelled the same way a fault is, but it is not
        one: it is always present and carries no label. Reporting it here would
        put "sensor_noise on LT101" in the operator's fault list on a perfectly
        healthy plant. It is listed separately as `instruments`.
        """
        real = [f for f in node.faults.faults if not f.benign]
        benign = [f for f in node.faults.faults if f.benign]
        return {"t": node.sim.t, "kinds": sorted(FAULT_KINDS),
                "faults": [f.label() for f in real],
                "instruments": [f.label() for f in benign],
                "active": [f.id for f in real
                           if f.strength(node.sim.t) > 0.0]}

    @app.post("/api/fault")
    async def inject(body: FaultBody) -> dict:
        now = node.sim.t
        fault_id = body.id or f"{body.kind}_{int(now)}"
        try:
            fault = build_fault(
                kind=body.kind, id=fault_id, magnitude=body.magnitude,
                tag=body.tag, t_start=now + body.delay, ramp=body.ramp,
                t_end=(now + body.delay + body.duration
                       if body.duration else None))
            node.faults.add(fault)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return fault.label()

    @app.delete("/api/fault/{fault_id}")
    async def clear_one(fault_id: str) -> dict:
        if not node.faults.remove(fault_id):
            raise HTTPException(404, f"no fault {fault_id!r}")
        return {"removed": fault_id}

    @app.delete("/api/faults")
    async def clear_all() -> dict:
        """Clear the faults and leave the instruments alone.

        "Clear all faults" must return the plant to healthy, not to an
        unrealistically silent one. Wiping the noise too would make stiction
        undetectable afterwards and nobody would know why.
        """
        removed = [f.id for f in node.faults.faults if not f.benign]
        for fault_id in removed:
            node.faults.remove(fault_id)
        return {"removed": removed}

    @app.get("/api/truth")
    async def truth() -> dict:
        """Ground truth: what the plant really is, versus what it reports."""
        return {"t": node.sim.t,
                "label": node.faults.labels_at(node.sim.t),
                "true": {k: float(v) if not isinstance(v, bool) else v
                         for k, v in node.sim.true_measurements.items()},
                "reported": {k: float(v) if not isinstance(v, bool) else v
                             for k, v in node.sim.measurements.items()}}

    return app


async def main() -> None:  # pragma: no cover - process entry point
    ap = argparse.ArgumentParser(description="pyplant plant simulator node")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5020)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--start", choices=("empty", "steady"), default="steady")
    ap.add_argument("--control-port", type=int, default=0,
                    help="serve the fault injection API here; 0 disables it")
    ap.add_argument("--no-noise", action="store_true",
                    help="run with perfectly clean instruments")
    args = ap.parse_args()

    u = Inputs(0.662, True)
    sim = (Simulation(dt=args.dt, x=steady_state(u, Params()), u=u)
           if args.start == "steady" else Simulation(dt=args.dt))
    node = PlantNode(sim=sim, host=args.host, port=args.port, speed=args.speed)
    if not args.no_noise:
        # The same instrument noise every scenario carries, for the same
        # reason: a perfectly quiet loop never moves its valve, so a sticking
        # valve produces no limit cycle and a frozen transmitter sitting on
        # setpoint produces no error to act on. Marked benign, so it is never
        # a fault label.
        from pyplant.faults.scenarios import DEFAULT_NOISE

        for tag, sigma in DEFAULT_NOISE.items():
            node.faults.add(SensorNoise(f"_noise_{tag}", tag=tag, sigma=sigma,
                                        benign=True))
    port = await node.start()
    print(f"plant listening on {args.host}:{port}, dt={args.dt}s, "
          f"speed={args.speed}x", flush=True)

    tasks = [asyncio.create_task(node.run(stop=asyncio.Event()))]
    control = None
    if args.control_port:
        import uvicorn

        # The fault API runs on the same event loop as the physics, so an
        # injection takes effect on the very next integration step rather than
        # whenever a second process happens to be scheduled.
        control = uvicorn.Server(uvicorn.Config(
            create_control_app(node), host=args.host, port=args.control_port,
            log_level="warning"))
        tasks.append(asyncio.create_task(control.serve()))
        print(f"fault api on http://{args.host}:{args.control_port}", flush=True)

    try:
        await asyncio.gather(*tasks)
    finally:
        if control is not None:
            control.should_exit = True
        for task in tasks:
            task.cancel()
        await node.stop()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
