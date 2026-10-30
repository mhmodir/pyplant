"""The PLC node.

Two roles at once, which is what a real PLC does: a Modbus *client* polling the
field device below it, and a Modbus *server* presenting itself to the SCADA
layer above. The scan engine and the program from steps 3 and 4 are unchanged;
only the `IOLink` is different.

Read path per scan
    plant  : PLANT_MEAS, PLANT_STATUS      two round trips
    local  : OPER_SP, OPER_CMD             what the operator last wrote

Write path per scan
    plant  : PLC_CMD, PLC_CMD_BITS         two round trips
    local  : PLC_FACE, PLC_ALARMS, PLC_DIAG, plus a mirror of the plant
             measurements so the HMI has one address to poll

The PLC never writes the OPER blocks. They are the operator's, and a block
written by two parties loses updates in whatever gap exists between the reader
and the writer. That is why the faceplate was split in two at step 5.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field

from pyplant.net.link import TagClient, TagServer
from pyplant.net.opcua import TagOpcServer
from pyplant.net.store import TagStore
from pyplant.plc.program import PlcInput, PlcOutput, Program
from pyplant.plc.scan import ScanEngine, ScanStats
from pyplant.tags import Owner, block, tags_in

# Blocks the PLC serves upward. Operator blocks are here because the HMI writes
# them into the PLC; the mirrored plant blocks save the HMI a second connection.
PLC_BLOCKS = (
    "PLANT_MEAS", "PLANT_STATUS",  # mirrored for the HMI
    "OPER_SP", "OPER_CMD",  # written by the HMI, read by the PLC
    "PLC_FACE", "PLC_DIAG", "PLC_ALARMS",  # published by the PLC
)
PUBLISHED_BY_PLC = ("PLC_FACE", "PLC_DIAG", "PLC_ALARMS")


@dataclass
class ModbusIOLink:
    """`IOLink` over the wire. Same interface the in-process link satisfies."""

    client: TagClient
    store: TagStore
    failures: int = 0
    last_field: dict[str, float | bool] = field(default_factory=dict)

    async def read_inputs(self) -> PlcInput:
        meas = await self.client.read("PLANT_MEAS")
        status = await self.client.read("PLANT_STATUS")
        oper_sp = self.store.read("OPER_SP")
        oper_cmd = self.store.read("OPER_CMD")
        self.last_field = {**meas, **status}
        return PlcInput(
            LT101=float(meas["LT101"]),
            LT102=float(meas["LT102"]),
            FT101=float(meas["FT101"]),
            FT102=float(meas["FT102"]),
            PT101=float(meas["PT101"]),
            ZT101=float(meas["ZT101"]),
            PMP101_RUN=bool(status["PMP101_RUN"]),
            LIC101_SP=float(oper_sp["LIC101_SP"]),
            LIC101_MAN_OP=float(oper_sp["LIC101_MAN_OP"]),
            LIC101_AUTO=bool(oper_cmd["LIC101_AUTO"]),
            CMD_PMP101_START=bool(oper_cmd["CMD_PMP101_START"]),
            CMD_ALM_RESET=bool(oper_cmd["CMD_ALM_RESET"]),
        )

    async def write_outputs(self, out: PlcOutput, stats: ScanStats) -> None:
        tags = out.as_tags()
        await self.client.write("PLC_CMD", tags)
        await self.client.write("PLC_CMD_BITS", tags)

        published = {**tags, **stats.as_tags()}
        for name in PUBLISHED_BY_PLC:
            self.store.publish(name, published)
        # Mirror what the field said this scan so the HMI polls one node.
        for name in ("PLANT_MEAS", "PLANT_STATUS"):
            if self.last_field:
                self.store.publish(name, self.last_field)


@dataclass
class PlcNode:
    """Scan engine, upward server, and downward client in one process."""

    plant_host: str = "127.0.0.1"
    plant_port: int = 5020
    host: str = "127.0.0.1"
    port: int = 5021
    period: float = 0.10
    speed: float = 1.0
    setpoint: float = 300.0
    opc_endpoint: str | None = None  # None disables the OPC UA layer

    program: Program = field(init=False)
    store: TagStore = field(init=False)
    server: TagServer = field(init=False)
    client: TagClient = field(init=False)
    io: ModbusIOLink = field(init=False)
    engine: ScanEngine = field(init=False)
    opc: TagOpcServer | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.program = Program(ts=self.period)
        self.store = TagStore(PLC_BLOCKS)
        self.server = TagServer(self.store, host=self.host, port=self.port)
        self.client = TagClient(host=self.plant_host, port=self.plant_port)
        self.io = ModbusIOLink(client=self.client, store=self.store)
        self.engine = ScanEngine(program=self.program, io=self.io,
                                 period=self.period, speed=self.speed)
        # Cold-start defaults for the operator blocks, until the HMI writes.
        self.store.publish("OPER_SP", {"LIC101_SP": self.setpoint,
                                       "LIC101_MAN_OP": 0.0})
        self.store.publish("OPER_CMD", {"LIC101_AUTO": True,
                                        "CMD_ALM_RESET": False,
                                        "CMD_PMP101_START": True})

    async def start(self) -> int:
        self.port = await self.server.start()
        await self.client.connect()
        # One scan before anything upstream can look. Otherwise the mirrored
        # blocks are all zeros until the first tick, and a historian
        # subscribing at startup faithfully records a level of 0 mm that the
        # plant never had. Same lesson as seeding the PID from valve feedback.
        await self.engine.scan_once()
        if self.opc_endpoint:
            # The OPC UA server is a view of the same store the Modbus server
            # holds, not a second process image.
            self.opc = TagOpcServer(store=self.store, endpoint=self.opc_endpoint,
                                    refresh_period=max(self.period, 0.05))
            await self.opc.start()
        return self.port

    async def stop(self) -> None:
        if self.opc is not None:
            await self.opc.stop()
            self.opc = None
        await self.client.close()
        await self.server.stop()

    async def run(self, scans: int | None = None,
                  stop: asyncio.Event | None = None) -> None:
        await self.engine.run(scans=scans, stop=stop)


def oper_blocks_are_never_published_by_the_plc() -> bool:
    """Static check of the rule this node's design rests on."""
    owned = {t.name for name in PUBLISHED_BY_PLC for t in tags_in(name)}
    operator = {t.name for name in ("OPER_SP", "OPER_CMD") for t in tags_in(name)}
    return not (owned & operator) and all(
        block(name).owner is Owner.PLC for name in PUBLISHED_BY_PLC
    )


async def main() -> None:  # pragma: no cover - process entry point
    ap = argparse.ArgumentParser(description="pyplant soft PLC node")
    ap.add_argument("--plant-host", default="127.0.0.1")
    ap.add_argument("--plant-port", type=int, default=5020)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5021)
    ap.add_argument("--period", type=float, default=0.10)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--setpoint", type=float, default=300.0)
    ap.add_argument("--opcua-port", type=int, default=4840)
    ap.add_argument("--no-opcua", action="store_true")
    args = ap.parse_args()

    endpoint = (None if args.no_opcua
                else f"opc.tcp://{args.host}:{args.opcua_port}/pyplant/")
    node = PlcNode(plant_host=args.plant_host, plant_port=args.plant_port,
                   host=args.host, port=args.port, period=args.period,
                   speed=args.speed, setpoint=args.setpoint,
                   opc_endpoint=endpoint)
    port = await node.start()
    print(f"plc listening on {args.host}:{port}, polling "
          f"{args.plant_host}:{args.plant_port} every {args.period}s", flush=True)
    if endpoint:
        print(f"opc ua endpoint {endpoint}", flush=True)
    try:
        await node.run(stop=asyncio.Event())
    finally:
        await node.stop()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
