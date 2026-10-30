"""Tests for alarm management and the HMI API.

The alarm section is the substantial one. The state a flag-based alarm list
cannot express is `RTN_UNACK`: a condition that came and went between two
operator glances. It has to stay on the list until somebody acknowledges it,
because the transient nobody saw is very often the interesting one.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timedelta

import httpx
import pytest

from pyplant.core.hybrid_tank import Inputs, Params, steady_state
from pyplant.hist.db import Historian
from pyplant.hmi.alarms import (
    MAX_SHELF,
    Alarm,
    AlarmManager,
    Priority,
    State,
    alarm_tags,
)
from pyplant.hmi.app import HmiState, create_app
from pyplant.net.opcua import free_port
from pyplant.runners.base import Simulation
from pyplant.services.historian import HistorianService
from pyplant.services.plant import PlantNode
from pyplant.services.plc import PlcNode

T0 = datetime(2026, 1, 1, 12, 0, 0)
DESIGN = Inputs(0.662, True)


def at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def alarm() -> Alarm:
    return Alarm(tag="ALM_LT101_HI", priority=Priority.MEDIUM, message="level high")


# ========================= the state machine ===============================

def test_starts_normal_and_invisible(alarm):
    assert alarm.state is State.NORMAL
    assert not alarm.visible and not alarm.demands_attention


def test_activation_raises_an_unacknowledged_alarm(alarm):
    alarm.update(True, at(10))
    assert alarm.state is State.UNACK_ALARM
    assert alarm.in_time == at(10) and alarm.count == 1
    assert alarm.visible and alarm.demands_attention


def test_acknowledged_while_still_active(alarm):
    alarm.update(True, at(10))
    assert alarm.acknowledge(at(15)) is True
    assert alarm.state is State.ACK_ALARM
    assert alarm.visible  # still on the list: the condition has not gone away
    assert not alarm.demands_attention


def test_clearing_after_acknowledgement_returns_to_normal(alarm):
    alarm.update(True, at(10))
    alarm.acknowledge(at(15))
    alarm.update(False, at(20))
    assert alarm.state is State.NORMAL
    assert alarm.out_time == at(20) and not alarm.visible


def test_a_transient_stays_on_the_list_until_acknowledged(alarm):
    # The state a boolean flag cannot represent.
    alarm.update(True, at(10))
    alarm.update(False, at(11))
    assert alarm.state is State.RTN_UNACK
    assert alarm.visible and alarm.demands_attention
    assert alarm.active is False  # the process condition is gone
    assert alarm.acknowledge(at(30)) is True
    assert alarm.state is State.NORMAL and not alarm.visible


def test_reactivation_from_rtn_unack_alarms_again(alarm):
    alarm.update(True, at(10))
    alarm.update(False, at(11))
    alarm.update(True, at(12))
    assert alarm.state is State.UNACK_ALARM
    assert alarm.count == 2


def test_acknowledging_a_normal_point_changes_nothing(alarm):
    assert alarm.acknowledge(at(5)) is False
    assert alarm.state is State.NORMAL


def test_repeated_activation_without_clearing_is_one_occurrence(alarm):
    alarm.update(True, at(10))
    for i in range(5):
        alarm.update(True, at(11 + i))
    assert alarm.count == 1
    assert alarm.in_time == at(10)


def test_acknowledgement_does_not_survive_a_new_occurrence(alarm):
    alarm.update(True, at(10))
    alarm.acknowledge(at(11))
    alarm.update(False, at(12))
    alarm.update(True, at(13))
    assert alarm.state is State.UNACK_ALARM
    assert alarm.ack_time is None


# ============================== shelving ===================================

def test_shelving_hides_an_alarm_and_expires_on_its_own(alarm):
    alarm.update(True, at(10))
    alarm.shelve(timedelta(minutes=5), reason="valve work", at=at(11))
    assert alarm.state is State.SHELVED
    assert not alarm.demands_attention
    alarm.update(True, at(11 + 4 * 60))  # still shelved
    assert alarm.state is State.SHELVED
    alarm.update(True, at(11 + 5 * 60))  # shelf expires
    assert alarm.state is State.UNACK_ALARM


def test_unshelving_reflects_the_condition_now_not_then(alarm):
    alarm.update(True, at(10))
    alarm.shelve(timedelta(minutes=5), at=at(11))
    alarm.update(False, at(20))  # cleared while shelved
    alarm.unshelve(at(30))
    assert alarm.state is State.RTN_UNACK  # it did occur; say so


def test_shelving_a_quiet_point_then_seeing_it_alarm(alarm):
    alarm.shelve(timedelta(minutes=5), at=at(0))
    alarm.update(True, at(10))
    assert alarm.state is State.SHELVED
    assert alarm.count == 1  # still counted underneath
    alarm.unshelve(at(20))
    assert alarm.state is State.UNACK_ALARM


def test_shelf_duration_is_bounded(alarm):
    with pytest.raises(ValueError, match="capped"):
        alarm.shelve(MAX_SHELF + timedelta(minutes=1))
    with pytest.raises(ValueError, match="positive"):
        alarm.shelve(timedelta(0))


def test_unshelving_something_not_shelved_is_harmless(alarm):
    alarm.update(True, at(10))
    alarm.unshelve(at(11))
    assert alarm.state is State.UNACK_ALARM


# ============================== manager ====================================

def test_manager_covers_the_declared_alarm_points():
    m = AlarmManager()
    assert set(m.alarms) == set(alarm_tags())
    assert m.alarms["ILK_PMP101"].priority is Priority.CRITICAL


def test_manager_updates_only_from_tags_it_was_given():
    m = AlarmManager()
    m.update({"ALM_LT101_HI": True}, at(10))
    assert m.alarms["ALM_LT101_HI"].state is State.UNACK_ALARM
    assert m.alarms["ILK_PMP101"].state is State.NORMAL


def test_active_list_sorts_worst_and_newest_first():
    m = AlarmManager()
    m.update({"ALM_LT101_HI": True, "ILK_PMP101": False}, at(10))
    m.update({"ALM_LT101_HI": True, "ILK_PMP101": True}, at(20))
    rows = m.active_list()
    assert [r["tag"] for r in rows] == ["ILK_PMP101", "ALM_LT101_HI"]
    assert rows[0]["priority_name"] == "CRITICAL"


def test_normal_points_are_not_on_the_list():
    m = AlarmManager()
    m.update({t: False for t in alarm_tags()}, at(0))
    assert m.active_list() == []


def test_acknowledge_all_reports_how_many_changed():
    m = AlarmManager()
    m.update({"ALM_LT101_HI": True, "ILK_PMP101": True}, at(10))
    assert m.acknowledge_all(at(11)) == 2
    assert m.acknowledge_all(at(12)) == 0


def test_summary_counts_and_worst_priority():
    m = AlarmManager()
    m.update({"ALM_LT101_HI": True, "ILK_PMP101": True}, at(10))
    s = m.summary()
    assert s["unacknowledged"] == 2 and s["active"] == 2
    assert s["worst_priority"] == int(Priority.CRITICAL)
    m.acknowledge_all(at(11))
    assert m.summary()["worst_priority"] == 0  # nothing demands attention now


def test_manager_expires_shelves_even_for_tags_not_reporting():
    m = AlarmManager()
    m.update({"ALM_LT101_HI": True}, at(0))
    m.shelve("ALM_LT101_HI", minutes=1, at=at(1))
    m.update({}, at(120))  # nothing reported, but time passed
    assert m.alarms["ALM_LT101_HI"].state is not State.SHELVED


# ============================== the API ====================================

@contextlib.asynccontextmanager
async def hmi_stack():
    sim = Simulation(dt=0.05, x=steady_state(DESIGN, Params()), u=DESIGN)
    plant = PlantNode(sim=sim, port=0, speed=1.0)
    plant_port = await plant.start()
    endpoint = f"opc.tcp://127.0.0.1:{free_port()}/pyplant/"
    plc = PlcNode(plant_port=plant_port, port=0, speed=1.0, opc_endpoint=endpoint)
    await plc.start()
    db = Historian()
    collector = HistorianService(db=db, endpoint=endpoint, publish_ms=100,
                                 flush_period=0.3)
    stop = asyncio.Event()
    tasks = asyncio.gather(plant.run(stop=stop), plc.run(stop=stop))
    await collector.start()
    state = HmiState(endpoint=endpoint, db=db, poll_period=0.2)
    await state.start()
    app = create_app(state)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://hmi") as client:
        try:
            yield client, state, sim, collector
        finally:
            stop.set()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(tasks, timeout=2.0)
            await state.stop()
            await collector.stop()
            await plc.stop()
            await plant.stop()
            db.close()


def test_page_is_served():
    async def go():
        async with hmi_stack() as (client, *_):
            return await client.get("/")

    r = run(go())
    assert r.status_code == 200
    assert "pyplant" in r.text and "<svg" in r.text


def test_state_endpoint_reports_the_live_plant():
    async def go():
        async with hmi_stack() as (client, state, sim, collector):
            await asyncio.sleep(0.5)
            return (await client.get("/api/state")).json()

    s = run(go())
    assert s["connected"] is True
    assert s["values"]["LT101"] == pytest.approx(300.3, abs=2.0)
    assert s["values"]["LIC101_AUTO"] is True
    assert s["summary"]["unacknowledged"] == 0
    assert s["alarms"] == []


def test_setpoint_write_goes_through_to_the_controller():
    async def go():
        async with hmi_stack() as (client, state, sim, collector):
            await asyncio.sleep(0.3)
            posted = await client.post("/api/setpoint", json={"value": 260.0})
            await asyncio.sleep(0.8)
            return posted, (await client.get("/api/state")).json()

    posted, s = run(go())
    assert posted.status_code == 200
    assert s["values"]["LIC101_SP"] == pytest.approx(260.0)
    assert s["values"]["LIC101_ERR"] < -10.0  # the PLC is acting on it


def test_out_of_span_setpoint_is_rejected_rather_than_clamped():
    async def go():
        async with hmi_stack() as (client, *_):
            return await client.post("/api/setpoint", json={"value": 3000.0})

    r = run(go())
    assert r.status_code == 422
    assert "span" in r.json()["detail"]


def test_read_only_tags_cannot_be_written_through_the_hmi():
    async def go():
        async with hmi_stack() as (client, state, *_):
            with pytest.raises(Exception) as exc:
                await state.write("LT101", 123.0)
            return exc.value

    err = run(go())
    assert getattr(err, "status_code", None) == 403


def test_mode_and_pump_commands():
    async def go():
        async with hmi_stack() as (client, state, sim, collector):
            await asyncio.sleep(0.3)
            await client.post("/api/mode", json={"auto": False})
            await client.post("/api/manual", json={"value": 40.0})
            await asyncio.sleep(0.8)
            return (await client.get("/api/state")).json()

    s = run(go())
    assert s["values"]["LIC101_AUTO"] is False
    assert s["values"]["LIC101_OP"] == pytest.approx(40.0, abs=0.5)


def test_trend_endpoint_returns_a_regular_grid():
    async def go():
        async with hmi_stack() as (client, state, sim, collector):
            await asyncio.sleep(1.5)
            collector.flush()
            return (await client.get(
                "/api/trend?tags=LT101,LIC101_SP&seconds=60&freq=0.5")).json()

    d = run(go())
    assert len(d["t"]) > 1
    assert set(d["series"]) == {"LT101", "LIC101_SP"}
    assert len(d["series"]["LT101"]) == len(d["t"])


def test_trend_rejects_unknown_tags():
    async def go():
        async with hmi_stack() as (client, *_):
            return await client.get("/api/trend?tags=NOPE")

    assert run(go()).status_code == 404


def test_alarm_endpoints():
    async def go():
        async with hmi_stack() as (client, state, sim, collector):
            await asyncio.sleep(0.3)
            state.alarms.update({"ALM_LT101_HI": True})
            before = (await client.get("/api/state")).json()
            shelved = await client.post("/api/alarms/shelve",
                                        json={"tag": "ALM_LT101_HI",
                                              "minutes": 5, "reason": "testing"})
            unshelved = await client.post("/api/alarms/unshelve",
                                          json={"tag": "ALM_LT101_HI"})
            acked = await client.post("/api/alarms/ack",
                                      json={"tag": "ALM_LT101_HI"})
            missing = await client.post("/api/alarms/ack", json={"tag": "LT101"})
            return before, shelved.json(), unshelved.json(), acked.json(), missing

    before, shelved, unshelved, acked, missing = run(go())
    assert before["alarms"][0]["tag"] == "ALM_LT101_HI"
    assert before["summary"]["unacknowledged"] == 1
    assert shelved["state"] == "SHELVED" and shelved["shelve_reason"] == "testing"
    assert unshelved["state"] == "UNACK_ALARM"
    assert acked["acknowledged"] == 1
    assert missing.status_code == 404


def test_tags_without_a_publisher_are_dropped_not_fatal():
    # ALM_HEALTH has no publisher until the inference service exists in step 11.
    async def go():
        async with hmi_stack() as (client, state, sim, collector):
            s = (await client.get("/api/state")).json()
            return state, s

    state, s = run(go())
    assert "ALM_HEALTH" in state.missing
    assert "ALM_HEALTH" not in state.available
    assert "LT101" in state.available
    assert s["connected"] is True and s["missing"] == state.missing
