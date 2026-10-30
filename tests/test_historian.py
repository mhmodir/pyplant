"""Tests for the historian.

The resampling tests matter most. Samples are stored by exception, so between
two rows the signal was *held*, not drifting. If `get_window` ever starts
interpolating, a valve position acquires ramps it never had and an alarm bit
acquires values between 0 and 1, and every model trained on that data learns
something about the resampler rather than about the plant.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timedelta

import pytest

from pyplant.core.hybrid_tank import Inputs, Params, steady_state
from pyplant.hist.db import Historian, Quality, Sample, UnknownTag
from pyplant.net.opcua import OpcClient, free_port
from pyplant.runners.base import Simulation
from pyplant.services.historian import HistorianService, default_tags
from pyplant.services.plant import PlantNode
from pyplant.services.plc import PlcNode

T0 = datetime(2026, 1, 1, 12, 0, 0)
DESIGN = Inputs(0.662, True)


def run(coro):
    return asyncio.run(coro)


def at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


@pytest.fixture
def db():
    h = Historian()
    yield h
    h.close()


# ============================== writing ====================================

def test_batch_write_and_count(db):
    assert db.write([Sample.of(at(0), "LT101", 300.0),
                     Sample.of(at(1), "LT101", 301.0)]) == 2
    assert db.count() == 2
    assert db.tags() == ["LT101"]


def test_empty_write_is_a_no_op(db):
    assert db.write([]) == 0
    assert db.count() == 0


def test_unknown_tags_are_refused(db):
    with pytest.raises(UnknownTag, match="NOT_A_TAG"):
        db.write([Sample.of(at(0), "NOT_A_TAG", 1.0)])
    assert db.count() == 0  # the whole batch was rejected


def test_booleans_are_stored_as_numbers(db):
    db.write([Sample.of(at(0), "ILK_PMP101", True),
              Sample.of(at(1), "ILK_PMP101", False)])
    assert db.latest(["ILK_PMP101"]) == {"ILK_PMP101": 0.0}


def test_bad_quality_is_recorded_and_excluded_by_default(db):
    db.write([Sample.of(at(0), "LT101", 300.0),
              Sample.of(at(1), "LT101", -999.0, good=False)])
    assert db.count() == 2
    assert db.latest(["LT101"]) == {"LT101": 300.0}
    window = db.get_window(["LT101"], at(0), at(2), 1.0)
    assert (window["LT101"] == 300.0).all()
    dirty = db.get_window(["LT101"], at(0), at(2), 1.0, include_bad=True)
    assert dirty["LT101"].iloc[-1] == -999.0


def test_span_and_raw(db):
    db.write([Sample.of(at(0), "LT101", 300.0), Sample.of(at(9), "LT101", 310.0)])
    assert db.span() == (at(0), at(9))
    raw = db.raw("LT101", at(0), at(9))
    assert list(raw["value"]) == [300.0, 310.0]
    assert set(raw["quality"]) == {int(Quality.GOOD)}


# ============================ resampling ===================================

def test_window_carries_the_last_value_forward(db):
    db.write([Sample.of(at(0), "LT101", 300.0), Sample.of(at(5), "LT101", 350.0)])
    w = db.get_window(["LT101"], at(0), at(10), 2.0)
    assert list(w["LT101"]) == [300.0, 300.0, 300.0, 350.0, 350.0, 350.0]


def test_window_does_not_interpolate(db):
    # A ramp between the two samples would put values strictly between them.
    db.write([Sample.of(at(0), "ZT101", 20.0), Sample.of(at(10), "ZT101", 80.0)])
    w = db.get_window(["ZT101"], at(0), at(10), 1.0)
    assert set(w["ZT101"]) == {20.0, 80.0}


def test_window_reaches_back_before_its_own_start(db):
    # A tag quiet for an hour must still have a value at the window's opening.
    db.write([Sample.of(at(0), "LT101", 300.0)])
    w = db.get_window(["LT101"], at(3600), at(3610), 5.0)
    assert (w["LT101"] == 300.0).all()


def test_window_is_null_before_a_tag_has_ever_reported(db):
    db.write([Sample.of(at(10), "LT101", 300.0)])
    w = db.get_window(["LT101"], at(0), at(20), 10.0)
    assert w["LT101"].isna().iloc[0]
    assert w["LT101"].iloc[-1] == 300.0


def test_window_columns_follow_the_requested_order(db):
    db.write([Sample.of(at(0), t, 1.0) for t in ("LT101", "FT101", "LIC101_OP")])
    w = db.get_window(["LIC101_OP", "LT101", "FT101"], at(0), at(1), 1.0)
    assert list(w.columns) == ["LIC101_OP", "LT101", "FT101"]


def test_window_grid_is_regular_and_inclusive(db):
    db.write([Sample.of(at(0), "LT101", 1.0)])
    w = db.get_window(["LT101"], at(0), at(10), 2.5)
    assert len(w) == 5
    assert (w.index[-1] - w.index[0]).total_seconds() == 10.0


def test_window_validates_its_arguments(db):
    with pytest.raises(UnknownTag):
        db.get_window(["NOPE"], at(0), at(1))
    with pytest.raises(ValueError, match="freq"):
        db.get_window(["LT101"], at(0), at(1), 0.0)
    with pytest.raises(ValueError, match="precedes"):
        db.get_window(["LT101"], at(10), at(0))


def test_latest_window_takes_the_tail(db):
    db.write([Sample.of(at(i), "LT101", float(i)) for i in range(100)])
    w = db.latest_window(["LT101"], seconds=10.0, freq=1.0)
    assert len(w) == 11
    assert w["LT101"].iloc[-1] == 99.0


def test_latest_window_on_an_empty_database_is_empty(db):
    assert db.latest_window(["LT101"], 10.0).empty


def test_parquet_export(db, tmp_path):
    db.write([Sample.of(at(i), "LT101", float(i)) for i in range(10)])
    out = tmp_path / "samples.parquet"
    db.export_parquet(str(out))
    # Read it back through DuckDB rather than pandas: the export is DuckDB's
    # own writer, so this checks the file without pulling in pyarrow.
    n = db.conn.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchall()
    assert n[0][0] == 10


def test_covered_span_starts_where_every_tag_has_data(db):
    db.write([Sample.of(at(0), "LT101", 300.0),
              Sample.of(at(4), "FT101", 10.0),
              Sample.of(at(9), "LT101", 310.0)])
    assert db.covered_span(["LT101", "FT101"]) == (at(4), at(9))
    assert db.span() == (at(0), at(9))  # the whole database is wider


def test_covered_span_is_empty_when_a_tag_never_reported(db):
    db.write([Sample.of(at(0), "LT101", 300.0)])
    assert db.covered_span(["LT101", "FT101"]) == (None, None)


def test_a_window_over_the_covered_span_has_no_leading_nulls(db):
    db.write([Sample.of(at(0), "LT101", 300.0), Sample.of(at(4), "FT101", 10.0)])
    t0, t1 = db.covered_span(["LT101", "FT101"])
    w = db.get_window(["LT101", "FT101"], t0, t1 + timedelta(seconds=5), 1.0)
    assert not w.isna().any().any()


def test_persists_to_a_file(tmp_path):
    path = str(tmp_path / "h.duckdb")
    with Historian(path) as h:
        h.write([Sample.of(at(0), "LT101", 300.0)])
    with Historian(path) as h:
        assert h.count() == 1


# ============================ collector ====================================

def test_default_tags_cover_the_plc_address_space():
    names = default_tags()
    assert {"LT101", "LIC101_OP", "SCAN_MS", "ILK_PMP101", "LIC101_SP"} <= set(names)
    assert len(names) == len(set(names))


def test_flush_is_a_no_op_when_nothing_arrived(db):
    service = HistorianService(db=db, tags=("LT101",))
    assert service.flush() == 0
    assert service.stats.flushes == 0


def test_on_change_buffers_and_flush_writes(db):
    service = HistorianService(db=db, tags=("LT101",))
    service._on_change("LT101", 300.0, True)
    service._on_change("LT101", 305.0, True)
    assert db.count() == 0  # still buffered
    assert service.flush() == 2
    assert db.count() == 2
    assert service.stats.received == 2 and service.stats.stored == 2


def test_bad_quality_does_not_become_the_heartbeat_value(db):
    service = HistorianService(db=db, tags=("LT101",))
    service._on_change("LT101", 300.0, True)
    service._on_change("LT101", -999.0, False)
    service.flush()
    assert service.stats.bad == 1
    assert service._last["LT101"][1] == 300.0  # last *good* value


# ======================== live, against the stack ==========================

@contextlib.asynccontextmanager
async def collecting_stack(heartbeat: float = 30.0):
    sim = Simulation(dt=0.05, x=steady_state(DESIGN, Params()), u=DESIGN)
    plant = PlantNode(sim=sim, port=0, speed=1.0)
    plant_port = await plant.start()
    endpoint = f"opc.tcp://127.0.0.1:{free_port()}/pyplant/"
    plc = PlcNode(plant_port=plant_port, port=0, speed=1.0, opc_endpoint=endpoint)
    await plc.start()
    db = Historian()
    service = HistorianService(db=db, endpoint=endpoint, publish_ms=100,
                               flush_period=0.3, heartbeat=heartbeat)
    stop = asyncio.Event()
    tasks = asyncio.gather(plant.run(stop=stop), plc.run(stop=stop))
    await service.start()
    try:
        yield db, service, endpoint
    finally:
        stop.set()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(tasks, timeout=2.0)
        await service.stop()
        await plc.stop()
        await plant.stop()
        db.close()


def test_collector_records_a_running_plant():
    async def go():
        async with collecting_stack() as (db, service, endpoint):
            await asyncio.sleep(2.0)
            service.flush()
            wanted = ["LT101", "LIC101_OP", "ILK_PMP101", "PLC_HEARTBEAT"]
            t0, t1 = db.covered_span(wanted)
            return db, service, db.get_window(wanted, t0, t1, 0.5)

    db, service, window = run(go())
    assert service.stats.received > 0
    assert service.stats.bad == 0
    assert not window.isna().any().any()  # covered_span leaves no leading nulls
    # The PLC is primed with one scan before it serves, so the very first
    # recorded level is the real one and not a zero from an unscanned mirror.
    assert window["LT101"].iloc[0] == pytest.approx(300.3, abs=1.0)
    assert (window["ILK_PMP101"] == 0.0).all()
    assert window["PLC_HEARTBEAT"].iloc[-1] > window["PLC_HEARTBEAT"].iloc[0]


def test_a_setpoint_change_appears_in_the_history():
    async def go():
        async with collecting_stack() as (db, service, endpoint):
            await asyncio.sleep(1.0)
            async with OpcClient(endpoint) as c:
                await c.write("LIC101_SP", 320.0)
            await asyncio.sleep(1.5)
            service.flush()
            wanted = ["LIC101_SP", "LIC101_OP"]
            t0, t1 = db.covered_span(wanted)
            return db.get_window(wanted, t0, t1, 0.5)

    w = run(go())
    assert w["LIC101_SP"].iloc[0] == pytest.approx(300.0)
    assert w["LIC101_SP"].iloc[-1] == pytest.approx(320.0)
    assert set(w["LIC101_SP"].dropna()) == {300.0, 320.0}  # a step, not a ramp
    assert w["LIC101_OP"].iloc[-1] > w["LIC101_OP"].iloc[0]


def test_heartbeat_keeps_quiet_tags_anchored():
    async def go():
        async with collecting_stack(heartbeat=0.4) as (db, service, endpoint):
            await asyncio.sleep(2.0)
            service.flush()
            return service.stats

    stats = run(go())
    assert stats.heartbeats > 0
