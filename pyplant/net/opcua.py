"""OPC UA layer.

This is the boundary. Below it everything speaks Modbus registers and scale
factors; above it — historian, HMI, inference service — everything speaks tag
names with units attached, and none of those three imports a Modbus symbol.

The address space is generated from `TAGS`, never hand-listed. Each tag's
`area` becomes its folder path, so `LT101` lands at `Plant/Unit1/LT101` and
`SCAN_MS` at `PLC/Diagnostics/SCAN_MS`. Every analog node carries
`EngineeringUnits` and `EURange` properties built from its declared span, which
is what lets a client draw a correctly scaled trend without being told anything
about the plant.

One source of truth
-------------------
The server is a *view* of the `TagStore`, not a second copy of the process
image. Values are pushed from the store on a refresh tick, and writes from OPC
UA clients are pushed back into the store.

Two write paths onto the same tag is a design smell, and an operator setpoint
can genuinely be lost between a refresh reading the old value and a client
write landing. The fix here is to track the last value each writable tag was
*seen* to hold in the store: a refresh only overwrites a writable node when the
store changed underneath us, which is a real external write, and never merely
because a refresh happened to run between a client write and its handler.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from asyncua import Client, Server, ua

from pyplant.net.store import TagStore
from pyplant.tags import TAGS, Kind, Tag, tags_in

NAMESPACE = "urn:pyplant"
DEFAULT_ENDPOINT = "opc.tcp://127.0.0.1:4840/pyplant/"


def free_port() -> int:
    """An unused TCP port. asyncua endpoints need one named up front."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def browse_path(tag: Tag, idx: int) -> list[str]:
    """Path from the root, as asyncua's `get_child` wants it."""
    parts = ["0:Objects"] + [f"{idx}:{p}" for p in tag.area.split("/")]
    return parts + [f"{idx}:{tag.name}"]


def _variant(tag: Tag) -> ua.VariantType:
    return ua.VariantType.Boolean if tag.kind is Kind.BOOL else ua.VariantType.Double


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class _WriteHandler:
    """Receives data changes on writable nodes and pushes them to the store."""

    def __init__(self, owner: "TagOpcServer") -> None:
        self.owner = owner

    async def datachange_notification(self, node, value, data) -> None:  # noqa: D401
        await self.owner._client_wrote(node, value)


@dataclass
class TagOpcServer:
    """Serves a `TagStore` as an OPC UA address space."""

    store: TagStore
    endpoint: str = DEFAULT_ENDPOINT
    name: str = "pyplant"
    refresh_period: float = 0.2  # s

    server: Server = field(init=False, default=None)
    idx: int = field(init=False, default=0)
    nodes: dict[str, object] = field(default_factory=dict, init=False)
    _by_nodeid: dict[object, str] = field(default_factory=dict, init=False)
    _node_cache: dict[str, float | bool] = field(default_factory=dict, init=False)
    _last_store: dict[str, float | bool] = field(default_factory=dict, init=False)
    _task: asyncio.Task | None = field(init=False, default=None)
    _sub = None
    refreshes: int = 0
    client_writes: int = 0

    @property
    def tags(self) -> tuple[Tag, ...]:
        return tuple(t for b in self.store.blocks for t in tags_in(b))

    # -- lifecycle --------------------------------------------------------

    async def start(self) -> str:
        self.server = Server()
        await self.server.init()
        self.server.set_endpoint(self.endpoint)
        self.server.set_server_name(self.name)
        self.idx = await self.server.register_namespace(NAMESPACE)

        await self._build_address_space()
        await self.server.start()
        await self.refresh(force=True)
        await self._watch_writable()
        self._task = asyncio.create_task(self._loop())
        return self.endpoint

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._sub is not None:
            with contextlib.suppress(Exception):
                await self._sub.delete()
            self._sub = None
        if self.server is not None:
            with contextlib.suppress(Exception):
                await self.server.stop()
            self.server = None

    async def __aenter__(self) -> "TagOpcServer":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    # -- address space ----------------------------------------------------

    async def _folder(self, path: str):
        """Create or reuse the folder chain for an `area` path."""
        node = self.server.nodes.objects
        for part in path.split("/"):
            try:
                node = await node.get_child(f"{self.idx}:{part}")
            except Exception:  # noqa: BLE001 - asyncua raises BadNoMatch here
                node = await node.add_folder(self.idx, part)
        return node

    async def _build_address_space(self) -> None:
        for tag in self.tags:
            parent = await self._folder(tag.area)
            initial = False if tag.kind is Kind.BOOL else 0.0
            node = await parent.add_variable(self.idx, tag.name, initial,
                                             _variant(tag))
            await node.set_writable(tag.writable)
            await node.write_attribute(
                ua.AttributeIds.Description,
                ua.DataValue(ua.Variant(ua.LocalizedText(tag.desc))),
            )
            if tag.kind is Kind.ANALOG:
                # What makes a client able to scale a trend on its own.
                await node.add_property(
                    self.idx, "EngineeringUnits",
                    ua.EUInformation(DisplayName=ua.LocalizedText(tag.unit),
                                     Description=ua.LocalizedText(tag.desc)))
                await node.add_property(self.idx, "EURange",
                                        ua.Range(Low=tag.lo, High=tag.hi))
            self.nodes[tag.name] = node
            self._by_nodeid[node.nodeid] = tag.name

    async def _watch_writable(self) -> None:
        writable = [self.nodes[t.name] for t in self.tags if t.writable]
        if not writable:
            return
        self._sub = await self.server.create_subscription(
            int(self.refresh_period * 1000), _WriteHandler(self))
        await self._sub.subscribe_data_change(writable)

    # -- store <-> address space -----------------------------------------

    async def refresh(self, force: bool = False) -> None:
        """Push store values outward. Cheap: only changed nodes are written."""
        for block in self.store.blocks:
            for name, value in self.store.read(block).items():
                tag = TAGS[name]
                if tag.writable and not force:
                    # Only follow the store when something else changed it.
                    if self._last_store.get(name) == value:
                        continue
                self._last_store[name] = value
                if not force and self._node_cache.get(name) == value:
                    continue
                self._node_cache[name] = value
                await self.nodes[name].write_value(
                    ua.DataValue(ua.Variant(value, _variant(tag))))
        self.refreshes += 1

    async def _client_wrote(self, node, value) -> None:
        name = self._by_nodeid.get(node.nodeid)
        if name is None:
            return
        tag = TAGS[name]
        block = _home_block_of(name, self.store)
        current = self.store.read(block)
        new = bool(value) if tag.kind is Kind.BOOL else float(value)
        if current[name] == new:
            return
        self.store.publish(block, {**current, name: new})
        self._last_store[name] = self.store.read(block)[name]
        self._node_cache[name] = new
        self.client_writes += 1

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.refresh_period)
            with contextlib.suppress(Exception):
                await self.refresh()


def _home_block_of(tag_name: str, store: TagStore) -> str:
    for block in store.blocks:
        if any(t.name == tag_name for t in tags_in(block)):
            return block
    raise KeyError(f"{tag_name} is not served by this node")  # pragma: no cover


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

@dataclass
class OpcClient:
    """Tag-name access to a `TagOpcServer`, for the historian, HMI and ML."""

    endpoint: str = DEFAULT_ENDPOINT
    client: Client = field(init=False, default=None)
    idx: int = field(init=False, default=0)
    _nodes: dict[str, object] = field(default_factory=dict, init=False)

    async def connect(self, retries: int = 1, delay: float = 0.5) -> None:
        """Connect, optionally waiting for the server to finish coming up.

        A collector that gives up because its server was two hundred
        milliseconds late is not a collector, it is a race. Services that start
        alongside the PLC pass a retry count.
        """
        last: Exception | None = None
        for attempt in range(max(retries, 1)):
            try:
                self.client = Client(self.endpoint)
                await self.client.connect()
                self.idx = await self.client.get_namespace_index(NAMESPACE)
                return
            except Exception as exc:  # noqa: BLE001 - asyncua raises broadly
                last = exc
                self.client = None
                if attempt + 1 < retries:
                    await asyncio.sleep(delay)
        raise ConnectionError(
            f"could not reach {self.endpoint} after {retries} attempts: {last}")

    async def close(self) -> None:
        if self.client is not None:
            with contextlib.suppress(Exception):
                await self.client.disconnect()
            self.client = None
            self._nodes.clear()

    async def __aenter__(self) -> "OpcClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def node(self, name: str):
        if name not in self._nodes:
            self._nodes[name] = await self.client.nodes.root.get_child(
                browse_path(TAGS[name], self.idx))
        return self._nodes[name]

    async def read(self, name: str) -> float | bool:
        return await (await self.node(name)).read_value()

    async def read_many(self, names: list[str]) -> dict[str, float | bool]:
        nodes = [await self.node(n) for n in names]
        values = await self.client.read_values(nodes)
        return dict(zip(names, values))

    async def write(self, name: str, value: float | bool) -> None:
        tag = TAGS[name]
        await (await self.node(name)).write_value(
            ua.DataValue(ua.Variant(value, _variant(tag))))

    async def engineering_units(self, name: str) -> tuple[str, float, float]:
        node = await self.node(name)
        eu = await (await node.get_child(f"{self.idx}:EngineeringUnits")).read_value()
        rng = await (await node.get_child(f"{self.idx}:EURange")).read_value()
        return eu.DisplayName.Text, rng.Low, rng.High

    async def subscribe(
        self,
        names: list[str],
        callback: Callable[[str, float | bool, bool], Awaitable[None] | None],
        period_ms: int = 500,
    ):
        """Push notification on change. This is why the historian is not a poller.

        The callback receives `(tag, value, good)`. Quality is passed through
        rather than assumed: a historian that records everything as good has a
        quality column that means nothing.
        """
        by_id: dict[object, str] = {}

        class _Handler:
            async def datachange_notification(self, node, value, data):
                name = by_id.get(node.nodeid)
                if name is None:
                    return
                item = getattr(data, "monitored_item", None)
                data_value = getattr(item, "Value", None)
                status = getattr(data_value, "StatusCode", None)
                good = True if status is None else status.is_good()
                result = callback(name, value, good)
                if asyncio.iscoroutine(result):
                    await result

        nodes = []
        for name in names:
            n = await self.node(name)
            by_id[n.nodeid] = name
            nodes.append(n)
        sub = await self.client.create_subscription(period_ms, _Handler())
        await sub.subscribe_data_change(nodes)
        return sub
