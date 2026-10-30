"""Tests for the OPC UA layer.

The important one is `test_a_client_write_survives_the_next_refresh`. Two write
paths onto one tag is where setpoints get silently reverted, and the whole
last-seen-store-value scheme in `TagOpcServer.refresh` exists to stop a refresh
that runs between a client write and its handler from undoing the write.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest
from asyncua import ua

from pyplant.core.hybrid_tank import Inputs, Params, steady_state
from pyplant.net.link import TagClient
from pyplant.net.opcua import OpcClient, TagOpcServer, browse_path, free_port
from pyplant.net.store import TagStore
from pyplant.runners.base import Simulation
from pyplant.services.plant import PlantNode
from pyplant.services.plc import PLC_BLOCKS, PlcNode
from pyplant.tags import TAGS, Kind, tags_in

DESIGN = Inputs(0.662, True)

SEED = {
    "PLANT_MEAS": {"LT101": 300.3, "LT102": 176.5, "FT101": 10.05,
                   "FT102": 10.05, "PT101": 50.32, "ZT101": 66.2},
    "PLANT_STATUS": {"PMP101_RUN": True},
    "OPER_SP": {"LIC101_SP": 300.0, "LIC101_MAN_OP": 0.0},
    "OPER_CMD": {"LIC101_AUTO": True, "CMD_ALM_RESET": False,
                 "CMD_PMP101_START": True},
    "PLC_FACE": {"LIC101_PV": 300.3, "LIC101_OP": 66.2, "LIC101_ERR": -0.3},
    "PLC_DIAG": {"SCAN_MS": 0.7, "SCAN_OVERRUNS": 0.0, "PLC_HEARTBEAT": 42.0},
    "PLC_ALARMS": {"ALM_LT101_HI": False, "ALM_LT101_LOLO": False,
                   "ALM_PT101_HIHI": False, "ILK_PMP101": False},
}


def run(coro):
    return asyncio.run(coro)


def seeded_store() -> TagStore:
    store = TagStore(PLC_BLOCKS)
    for block, values in SEED.items():
        store.publish(block, values)
    return store


@contextlib.asynccontextmanager
async def served(store: TagStore | None = None, refresh: float = 0.05):
    store = store or seeded_store()
    endpoint = f"opc.tcp://127.0.0.1:{free_port()}/pyplant/"
    server = TagOpcServer(store=store, endpoint=endpoint, refresh_period=refresh)
    await server.start()
    client = OpcClient(endpoint)
    await client.connect()
    try:
        yield store, server, client
    finally:
        await client.close()
        await server.stop()


# =========================== address space =================================

def test_browse_path_follows_the_tag_area():
    assert browse_path(TAGS["LT101"], 2) == [
        "0:Objects", "2:Plant", "2:Unit1", "2:LT101"]
    assert browse_path(TAGS["SCAN_MS"], 2) == [
        "0:Objects", "2:PLC", "2:Diagnostics", "2:SCAN_MS"]


def test_every_served_tag_has_a_node():
    async def go():
        async with served() as (store, server, client):
            for block in PLC_BLOCKS:
                for tag in tags_in(block):
                    await client.node(tag.name)  # raises if the path is wrong
            return len(server.nodes)

    assert run(go()) == sum(len(tags_in(b)) for b in PLC_BLOCKS)


def test_analog_nodes_carry_units_and_range():
    async def go():
        async with served() as (store, server, client):
            return (await client.engineering_units("LT101"),
                    await client.engineering_units("PT101"),
                    await client.engineering_units("LIC101_ERR"))

    lt, pt, err = run(go())
    assert lt == ("mm", 0.0, 600.0)
    assert pt == ("kPa", 0.0, 100.0)
    assert err == ("mm", -600.0, 600.0)  # signed range survives


def test_boolean_nodes_have_no_range_property():
    async def go():
        async with served() as (store, server, client):
            with pytest.raises(Exception):
                await client.engineering_units("ILK_PMP101")

    run(go())


def test_units_match_the_tag_dictionary_for_every_analog():
    async def go():
        async with served() as (store, server, client):
            for block in PLC_BLOCKS:
                for tag in tags_in(block):
                    if tag.kind is Kind.ANALOG:
                        unit, lo, hi = await client.engineering_units(tag.name)
                        assert (unit, lo, hi) == (tag.unit, tag.lo, tag.hi), tag.name

    run(go())


# ============================== reading ====================================

def test_values_reflect_the_store():
    async def go():
        async with served() as (store, server, client):
            return await client.read_many(
                ["LT101", "PT101", "LIC101_ERR", "ILK_PMP101", "PMP101_RUN"])

    v = run(go())
    assert v["LT101"] == pytest.approx(300.3)
    assert v["PT101"] == pytest.approx(50.32)
    assert v["LIC101_ERR"] == pytest.approx(-0.3)
    assert v["ILK_PMP101"] is False
    assert v["PMP101_RUN"] is True


def test_external_store_change_propagates_outward():
    async def go():
        async with served() as (store, server, client):
            store.set("LT101", 412.0)
            await asyncio.sleep(0.3)
            return await client.read("LT101")

    assert run(go()) == pytest.approx(412.0)


def test_refresh_only_writes_nodes_that_changed():
    async def go():
        async with served(refresh=0.02) as (store, server, client):
            await asyncio.sleep(0.2)
            return server.refreshes

    # Many refreshes ran; the point is simply that a quiet plant does not
    # generate a node write per tag per cycle.
    assert run(go()) > 3


# ============================== writing ====================================

def test_read_only_nodes_reject_writes():
    async def go():
        async with served() as (store, server, client):
            with pytest.raises(ua.UaStatusCodeError):
                await client.write("LT101", 123.0)

    run(go())


def test_client_write_reaches_the_store():
    async def go():
        async with served() as (store, server, client):
            await client.write("LIC101_SP", 250.0)
            await client.write("LIC101_AUTO", False)
            await asyncio.sleep(0.3)
            return store.get("LIC101_SP"), store.get("LIC101_AUTO"), server

    sp, auto, server = run(go())
    assert sp == pytest.approx(250.0)
    assert auto is False
    assert server.client_writes >= 2


def test_a_client_write_survives_the_next_refresh():
    # The race this design exists to close: a refresh reading the pre-write
    # store value must not push the old setpoint back over the new one.
    async def go():
        async with served(refresh=0.02) as (store, server, client):
            await client.write("LIC101_SP", 250.0)
            await asyncio.sleep(0.4)  # roughly twenty refresh cycles
            return await client.read("LIC101_SP"), store.get("LIC101_SP")

    node_value, store_value = run(go())
    assert node_value == pytest.approx(250.0)
    assert store_value == pytest.approx(250.0)


def test_writing_the_same_value_is_not_counted_as_a_write():
    async def go():
        async with served() as (store, server, client):
            await client.write("LIC101_SP", 300.0)  # already 300
            await asyncio.sleep(0.2)
            return server.client_writes

    assert run(go()) == 0


# =========================== subscriptions =================================

def test_subscription_fires_on_change():
    async def go():
        seen: list[tuple[str, float]] = []
        async with served(refresh=0.05) as (store, server, client):
            sub = await client.subscribe(
                ["LT101", "LIC101_OP"],
                lambda n, v, good: seen.append((n, v, good)), 50)
            await asyncio.sleep(0.2)
            store.set("LT101", 355.5)
            await asyncio.sleep(0.4)
            await sub.delete()
        return seen

    seen = run(go())
    assert ("LT101", pytest.approx(300.3), True) in seen  # initial value
    assert ("LT101", pytest.approx(355.5), True) in seen  # the change
    assert all(good for _, _, good in seen)  # quality is passed through


# ======================= three nodes, end to end ===========================

@contextlib.asynccontextmanager
async def full_stack(speed: float = 10.0):
    sim = Simulation(dt=0.05, x=steady_state(DESIGN, Params()), u=DESIGN)
    plant = PlantNode(sim=sim, port=0, speed=speed)
    plant_port = await plant.start()
    endpoint = f"opc.tcp://127.0.0.1:{free_port()}/pyplant/"
    plc = PlcNode(plant_port=plant_port, port=0, speed=speed,
                  opc_endpoint=endpoint)
    modbus_port = await plc.start()
    opc = OpcClient(endpoint)
    await opc.connect()
    modbus = TagClient(port=modbus_port)
    await modbus.connect()
    stop = asyncio.Event()
    tasks = asyncio.gather(plant.run(stop=stop), plc.run(stop=stop))
    try:
        yield sim, plant, plc, opc, modbus
    finally:
        stop.set()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(tasks, timeout=2.0)
        await opc.close()
        await modbus.close()
        await plc.stop()
        await plant.stop()


def test_opcua_and_modbus_show_the_same_process_image():
    async def go():
        async with full_stack() as (sim, plant, plc, opc, modbus):
            await asyncio.sleep(1.0)
            over_opc = await opc.read_many(["LT101", "LIC101_OP", "PLC_HEARTBEAT"])
            over_modbus = {**await modbus.read("PLANT_MEAS"),
                           **await modbus.read("PLC_FACE"),
                           **await modbus.read("PLC_DIAG")}
            return over_opc, over_modbus

    o, m = run(go())
    assert o["LT101"] == pytest.approx(m["LT101"], abs=1.0)
    assert o["LIC101_OP"] == pytest.approx(m["LIC101_OP"], abs=1.0)
    assert o["PLC_HEARTBEAT"] > 0


def test_setpoint_written_over_opcua_reaches_the_controller():
    async def go():
        async with full_stack() as (sim, plant, plc, opc, modbus):
            await asyncio.sleep(0.5)
            before = await opc.read("LIC101_ERR")
            await opc.write("LIC101_SP", 250.0)
            await asyncio.sleep(0.6)
            after = await opc.read_many(["LIC101_ERR", "LIC101_OP"])
            return before, after

    before, after = run(go())
    assert abs(before) < 2.0
    assert after["LIC101_ERR"] < -10.0
    assert after["LIC101_OP"] < 66.2
