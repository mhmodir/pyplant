"""Tests for the networked split.

The store tests are cheap and exact. The two integration tests start real
servers on ephemeral ports and run real TCP round trips, so they cost a couple
of seconds each and their tolerances are loose. They earn it: this is the first
point where a scale factor, an address, or a block boundary can be wrong in a
way no single-process test would notice.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from pyplant.core.hybrid_tank import Inputs, Params, steady_state
from pyplant.net.link import LinkError, TagClient, TagServer
from pyplant.net.store import BlockError, TagStore
from pyplant.runners.base import Simulation
from pyplant.services.plant import PLANT_BLOCKS, PlantNode
from pyplant.services.plc import (
    PLC_BLOCKS,
    PUBLISHED_BY_PLC,
    PlcNode,
    oper_blocks_are_never_published_by_the_plc,
)
from pyplant.tags import TAGS, Owner, by_owner, tags_in, writable

DESIGN = Inputs(0.662, True)


def run(coro):
    return asyncio.run(coro)


# ============================== TagStore ===================================

def test_store_rejects_unknown_blocks():
    with pytest.raises(BlockError, match="unknown blocks"):
        TagStore(("NOT_A_BLOCK",))


def test_store_refuses_blocks_it_does_not_carry():
    store = TagStore(("PLANT_MEAS",))
    with pytest.raises(BlockError, match="does not carry"):
        store.read("PLC_DIAG")


def test_block_round_trip_through_the_registers():
    store = TagStore(("PLANT_MEAS",))
    values = {"LT101": 300.3, "LT102": 176.5, "FT101": 10.05,
              "FT102": 10.05, "PT101": 50.32, "ZT101": 66.2}
    store.publish("PLANT_MEAS", values)
    back = store.read("PLANT_MEAS")
    for name, v in values.items():
        assert back[name] == pytest.approx(v, abs=TAGS[name].resolution)


def test_partial_publish_is_refused():
    store = TagStore(("PLANT_MEAS",))
    with pytest.raises(KeyError, match="missing"):
        store.publish("PLANT_MEAS", {"LT101": 300.0})


def test_signed_tag_survives_the_datastore():
    store = TagStore(("PLC_FACE",))
    store.publish("PLC_FACE", {"LIC101_PV": 300.0, "LIC101_OP": 66.0,
                               "LIC101_ERR": -212.5})
    assert store.read("PLC_FACE")["LIC101_ERR"] == pytest.approx(-212.5)


def test_single_tag_access_and_snapshot():
    store = TagStore(("PLANT_MEAS", "PLANT_STATUS"))
    store.set("LT101", 412.3)
    store.set("PMP101_RUN", True)
    assert store.get("LT101") == pytest.approx(412.3)
    assert store.get("PMP101_RUN") is True
    snap = store.snapshot()
    assert set(snap) == {t.name for name in ("PLANT_MEAS", "PLANT_STATUS")
                         for t in tags_in(name)}


# ========================== server and client ==============================

@contextlib.asynccontextmanager
async def served(store: TagStore):
    server = TagServer(store, port=0)
    port = await server.start()
    client = TagClient(port=port)
    await client.connect()
    try:
        yield client
    finally:
        await client.close()
        await server.stop()


def test_client_reads_what_the_server_holds():
    store = TagStore(("PLANT_MEAS",))
    store.publish("PLANT_MEAS", {"LT101": 300.3, "LT102": 176.5, "FT101": 10.05,
                                 "FT102": 10.05, "PT101": 50.32, "ZT101": 66.2})

    async def go():
        async with served(store) as client:
            return await client.read("PLANT_MEAS")

    values = run(go())
    assert values["LT101"] == pytest.approx(300.3, abs=0.05)
    assert values["PT101"] == pytest.approx(50.32, abs=0.01)


def test_client_writes_land_in_the_store():
    store = TagStore(("PLC_CMD", "PLC_CMD_BITS"))

    async def go():
        async with served(store) as client:
            await client.write("PLC_CMD", {"FCV101_OP": 42.5})
            await client.write("PLC_CMD_BITS", {"PMP101_CMD": True})

    run(go())
    assert store.get("FCV101_OP") == pytest.approx(42.5)
    assert store.get("PMP101_CMD") is True


def test_engineering_units_survive_the_wire_to_full_resolution():
    store = TagStore(("PLANT_MEAS",))

    async def go():
        async with served(store) as client:
            for level in (0.0, 0.05, 123.46, 599.99, 600.0):
                store.set("LT101", level)
                got = (await client.read("PLANT_MEAS"))["LT101"]
                assert abs(got - TAGS["LT101"].clamp(level)) <= \
                    TAGS["LT101"].resolution / 2 + 1e-9, level

    run(go())


def test_partial_write_is_refused_before_any_traffic():
    async def go():
        async with served(TagStore(("PLANT_MEAS",))) as client:
            with pytest.raises(KeyError, match="missing"):
                await client.write("PLANT_MEAS", {"LT101": 1.0})

    run(go())


def test_unconnected_client_raises():
    async def go():
        with pytest.raises(LinkError, match="not connected"):
            await TagClient(port=1).read("PLANT_MEAS")

    run(go())


def test_connecting_to_nothing_raises():
    async def go():
        with pytest.raises(LinkError, match="cannot reach"):
            await TagClient(port=1, timeout=0.2).connect()

    run(go())


# ============================ PlantNode ====================================

def test_plant_node_refuses_a_policy():
    with pytest.raises(ValueError, match="over Modbus"):
        PlantNode(sim=Simulation(policy=lambda t, m: Inputs()))


def test_plant_node_seeds_its_command_registers():
    # Otherwise a PLC that has not scanned yet would appear to command 0%.
    sim = Simulation(dt=0.05, x=steady_state(DESIGN, Params()), u=DESIGN)
    node = PlantNode(sim=sim, port=0)
    assert node.store.get("FCV101_OP") == pytest.approx(66.2, abs=0.05)
    assert node.store.get("PMP101_CMD") is True


def test_plant_node_obeys_whatever_is_in_its_registers():
    sim = Simulation(dt=0.05, x=steady_state(DESIGN, Params()), u=DESIGN)
    node = PlantNode(sim=sim, port=0)
    node.store.publish("PLC_CMD", {"FCV101_OP": 20.0})
    node.store.publish("PLC_CMD_BITS", {"PMP101_CMD": True})
    for _ in range(20):
        node.tick()
    assert sim.u.valve_cmd == pytest.approx(0.20)
    assert node.store.get("LT101") < 300.3  # already draining


def test_plant_node_publishes_only_plant_blocks():
    assert set(PLANT_BLOCKS) == {"PLANT_MEAS", "PLANT_STATUS",
                                 "PLC_CMD", "PLC_CMD_BITS"}


# ============================= PlcNode =====================================

def test_plc_never_publishes_an_operator_block():
    # The rule the faceplate split exists to enforce: no block written by two
    # parties, or an operator setpoint can be lost in the gap between the
    # PLC's read and its publish.
    assert oper_blocks_are_never_published_by_the_plc()
    published = {t.name for name in PUBLISHED_BY_PLC for t in tags_in(name)}
    assert not published & {t.name for t in by_owner(Owner.OPER)}


def test_plc_serves_every_block_the_hmi_needs():
    assert {"OPER_SP", "OPER_CMD"} <= set(PLC_BLOCKS)
    assert set(PUBLISHED_BY_PLC) <= set(PLC_BLOCKS)


def test_operator_writable_tags_are_exactly_the_operator_owned_ones():
    assert {t.name for t in writable()} == {t.name for t in by_owner(Owner.OPER)}


# =========================== two nodes, real TCP ===========================

@contextlib.asynccontextmanager
async def plant_and_plc(speed: float = 1.0, setpoint: float = 300.0,
                        start_at_design: bool = True):
    """Real time, deliberately.

    Running this at speed=10 puts the scan period at 10 ms and the plant tick
    at 5 ms. Windows' default timer granularity is about 15.6 ms, so every
    sleep overshoots, every scan is late, and `SCAN_OVERRUNS` climbs - which is
    the engine reporting a genuine problem, not a flaky test. Asserting zero
    overruns is only meaningful when the scaled period comfortably exceeds the
    platform's clock resolution, so this runs at wall speed where 100 ms and
    50 ms are safe everywhere.
    """
    sim = (Simulation(dt=0.05, x=steady_state(DESIGN, Params()), u=DESIGN)
           if start_at_design else Simulation(dt=0.05))
    plant = PlantNode(sim=sim, port=0, speed=speed)
    plant_port = await plant.start()
    plc = PlcNode(plant_port=plant_port, port=0, speed=speed, setpoint=setpoint)
    hmi_port = await plc.start()
    hmi = TagClient(port=hmi_port)
    await hmi.connect()
    stop = asyncio.Event()
    tasks = asyncio.gather(plant.run(stop=stop), plc.run(stop=stop))
    try:
        yield sim, plant, plc, hmi, stop
    finally:
        stop.set()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(tasks, timeout=2.0)
        await hmi.close()
        await plc.stop()
        await plant.stop()


def test_control_holds_setpoint_across_the_wire():
    async def go():
        async with plant_and_plc() as (sim, plant, plc, hmi, stop):
            await asyncio.sleep(1.5)
            meas = await hmi.read("PLANT_MEAS")
            face = await hmi.read("PLC_FACE")
            diag = await hmi.read("PLC_DIAG")
            alarms = await hmi.read("PLC_ALARMS")
            return sim, plant, plc, meas, face, diag, alarms

    sim, plant, plc, meas, face, diag, alarms = run(go())

    assert meas["LT101"] == pytest.approx(300.0, abs=2.0)
    assert face["LIC101_PV"] == pytest.approx(meas["LT101"], abs=0.2)
    assert face["LIC101_OP"] == pytest.approx(66.2, abs=2.0)
    assert alarms["ILK_PMP101"] is False
    assert diag["PLC_HEARTBEAT"] > 0
    assert diag["SCAN_OVERRUNS"] == 0
    assert 0.0 < diag["SCAN_MS"] < 20.0  # four round trips, well inside 100 ms

    ratio = plant.steps / plc.engine.stats.scans
    assert 1.6 < ratio < 2.4  # 50 ms plant, 100 ms scan


def test_operator_setpoint_written_over_modbus_reaches_the_controller():
    async def go():
        async with plant_and_plc() as (sim, plant, plc, hmi, stop):
            await asyncio.sleep(0.5)
            before = await hmi.read("PLC_FACE")
            await hmi.write("OPER_SP", {"LIC101_SP": 250.0, "LIC101_MAN_OP": 0.0})
            await asyncio.sleep(0.5)
            after = await hmi.read("PLC_FACE")
            return before, after

    before, after = run(go())
    assert abs(before["LIC101_ERR"]) < 2.0  # lined out on the old setpoint
    assert after["LIC101_ERR"] < -10.0  # controller now sees the new one
    assert after["LIC101_OP"] < before["LIC101_OP"]  # and is closing the valve
