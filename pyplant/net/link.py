"""Modbus TCP endpoints.

`TagServer` runs a pymodbus server over a `TagStore`. `TagClient` reads and
writes the same blocks from another process, converting through the same codec,
so a value that leaves one node in millimetres arrives at the next in
millimetres.

Version note: pymodbus is pinned to 3.8.x. The 3.15 line is midway through
replacing the datastore with SimData/SimDevice and no longer exposes a way to
update served values from outside the server, which is exactly what a live
plant needs to do sixty times a second.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from pymodbus.client import AsyncModbusTcpClient
from pymodbus.server import ModbusTcpServer

from pyplant.net.store import TagStore
from pyplant.tags import Table, block, tags_in


class LinkError(IOError):
    """A Modbus transaction failed or came back as an exception response."""


@dataclass
class TagServer:
    """Serves a `TagStore` on a TCP port."""

    store: TagStore
    host: str = "127.0.0.1"
    port: int = 5020
    _server: ModbusTcpServer | None = field(default=None, init=False)
    _task: asyncio.Task | None = field(default=None, init=False)

    async def start(self) -> int:
        """Begin serving. Returns the port actually bound (useful with port 0)."""
        self._server = ModbusTcpServer(self.store.context,
                                       address=(self.host, self.port))
        self._task = asyncio.create_task(self._server.serve_forever())
        # Poll for the bound socket rather than sleeping a guessed interval,
        # and read the port back off it so tests can bind port 0.
        for _ in range(400):
            await asyncio.sleep(0.005)
            transport = getattr(self._server, "transport", None)
            sockets = getattr(transport, "sockets", None)
            if sockets:
                self.port = sockets[0].getsockname()[1]
                return self.port
        raise LinkError(f"server did not come up on {self.host}:{self.port}")

    async def stop(self) -> None:
        if self._server is not None:
            await self._server.shutdown()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._server = self._task = None


@dataclass
class TagClient:
    """Reads and writes blocks on a remote node."""

    host: str = "127.0.0.1"
    port: int = 5020
    device_id: int = 1
    timeout: float = 1.0
    reads: int = 0
    writes: int = 0
    _client: AsyncModbusTcpClient | None = field(default=None, init=False)

    async def connect(self) -> None:
        self._client = AsyncModbusTcpClient(self.host, port=self.port,
                                            timeout=self.timeout, retries=1)
        await self._client.connect()
        if not self._client.connected:
            raise LinkError(f"cannot reach {self.host}:{self.port}")

    async def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    @property
    def connected(self) -> bool:
        return self._client is not None and self._client.connected

    def _require(self) -> AsyncModbusTcpClient:
        if self._client is None:
            raise LinkError("client is not connected")
        return self._client

    async def read(self, name: str) -> dict[str, float | bool]:
        b = block(name)
        c = self._require()
        if b.table is Table.HR:
            rsp = await c.read_holding_registers(b.start, count=b.count,
                                                 slave=self.device_id)
            raw = None if rsp.isError() else rsp.registers
        else:
            rsp = await c.read_coils(b.start, count=b.count, slave=self.device_id)
            raw = None if rsp.isError() else rsp.bits[: b.count]
        if raw is None:
            raise LinkError(f"read of {name} failed: {rsp}")
        self.reads += 1
        return {t.name: t.decode(int(v)) for t, v in zip(tags_in(name), raw)}

    async def write(self, name: str, values: dict[str, float | bool]) -> None:
        b = block(name)
        members = tags_in(name)
        missing = [t.name for t in members if t.name not in values]
        if missing:
            raise KeyError(f"write of {name} is missing {missing}")
        c = self._require()
        if b.table is Table.HR:
            words = [t.encode(values[t.name]) for t in members]
            rsp = await c.write_registers(b.start, words, slave=self.device_id)
        else:
            bits = [bool(values[t.name]) for t in members]
            rsp = await c.write_coils(b.start, bits, slave=self.device_id)
        if rsp.isError():
            raise LinkError(f"write of {name} failed: {rsp}")
        self.writes += 1
