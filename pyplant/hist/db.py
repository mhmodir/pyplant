"""The historian.

Long format, one row per sample: `(ts, tag, value, quality)`. Wide tables look
tidier until the day someone adds a tag, at which point they need a migration;
long tables do not. Everything that wants a rectangle asks `get_window()` for
one.

Resampling is last-value-carried-forward, not interpolation
-----------------------------------------------------------
Samples arrive by exception: a value is stored when it changes, plus a
heartbeat so a quiet tag still has anchors. Between two stored samples the
signal did not drift — it was *held*. Linear interpolation would invent a ramp
that never existed, and on a valve position or an alarm bit that is simply
wrong. DuckDB's ASOF JOIN gives the correct step-hold semantics directly, which
is the main reason it is here rather than a hand-rolled merge.

Timestamps are wall clock, as a real historian's are. When the stack runs at
`speed > 1` the recorded timeline is compressed by that factor, so a window
that is 60 wall seconds long covers `60 * speed` seconds of plant time. Run the
live demo at `speed=1` if that distinction matters; step 10's training data
comes from the fast runner in plant time and never touches this database.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum
from typing import Iterable, Sequence

import duckdb
import pandas as pd

from pyplant.tags import TAGS

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    ts        TIMESTAMP NOT NULL,
    tag       VARCHAR   NOT NULL,
    value     DOUBLE    NOT NULL,
    quality   UTINYINT  NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS samples_tag_ts ON samples (tag, ts);
"""


class Quality(IntEnum):
    """OPC UA status, flattened to what a query needs to filter on."""

    GOOD = 0
    UNCERTAIN = 1
    BAD = 2


@dataclass(frozen=True)
class Sample:
    ts: datetime
    tag: str
    value: float
    quality: Quality = Quality.GOOD

    @staticmethod
    def of(ts: datetime, tag: str, value: float | bool,
           good: bool = True) -> "Sample":
        return Sample(ts, tag, float(value),
                      Quality.GOOD if good else Quality.BAD)


class UnknownTag(KeyError):
    """A sample or query named a tag that is not in the dictionary."""


@dataclass
class Historian:
    """A DuckDB file (or in-memory database) holding the sample history."""

    path: str = ":memory:"
    conn: duckdb.DuckDBPyConnection = field(init=False)

    def __post_init__(self) -> None:
        self.conn = duckdb.connect(self.path)
        self.conn.execute(SCHEMA)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.conn.close()

    def __enter__(self) -> "Historian":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- writing ----------------------------------------------------------

    def write(self, samples: Sequence[Sample]) -> int:
        """Insert a batch. Row-at-a-time inserts are the usual way to make
        DuckDB look slow, so the collector buffers and calls this."""
        if not samples:
            return 0
        unknown = {s.tag for s in samples} - set(TAGS)
        if unknown:
            raise UnknownTag(f"not in the tag dictionary: {sorted(unknown)}")
        frame = pd.DataFrame(
            {"ts": [s.ts for s in samples],
             "tag": [s.tag for s in samples],
             "value": [float(s.value) for s in samples],
             "quality": [int(s.quality) for s in samples]})
        self.conn.register("_batch", frame)
        try:
            self.conn.execute("INSERT INTO samples SELECT * FROM _batch")
        finally:
            self.conn.unregister("_batch")
        return len(samples)

    # -- reading ----------------------------------------------------------

    # Every query below uses fetchall(), never fetchone(). From DuckDB 2.0 a
    # fetchone() on a one-row result leaves the cursor open until a second
    # fetch returns nothing, and an open cursor blocks the next statement on
    # the connection - so a single count() would break the following write.
    # fetchall() consumes the result fully and behaves the same on 1.x.

    def count(self) -> int:
        return int(self.conn.execute("SELECT count(*) FROM samples").fetchall()[0][0])

    def tags(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT tag FROM samples ORDER BY tag").fetchall()
        return [r[0] for r in rows]

    def span(self) -> tuple[datetime | None, datetime | None]:
        lo, hi = self.conn.execute(
            "SELECT min(ts), max(ts) FROM samples").fetchall()[0]
        return lo, hi

    def covered_span(self, tags: Sequence[str]) -> tuple[datetime | None,
                                                        datetime | None]:
        """The interval over which *every* requested tag has a value.

        `span()` is the extent of the whole database, and a window opened there
        almost always starts with nulls: tags report a few milliseconds apart,
        so the earliest sample belongs to exactly one of them. The start of a
        usable window is the latest of the per-tag first samples. This is the
        function callers should use to ask "give me everything you have".
        """
        rows = self.conn.execute(
            "SELECT tag, min(ts), max(ts) FROM samples "
            "WHERE tag = ANY(?) AND quality = 0 GROUP BY tag",
            [list(tags)]).fetchall()
        if len(rows) < len(set(tags)):
            return None, None  # at least one tag has never reported
        return max(r[1] for r in rows), max(r[2] for r in rows)

    def latest(self, tags: Iterable[str] | None = None) -> dict[str, float]:
        """Most recent stored value per tag."""
        sql = ("SELECT tag, last(value ORDER BY ts) FROM samples "
               "WHERE quality = 0 {} GROUP BY tag")
        if tags is None:
            rows = self.conn.execute(sql.format("")).fetchall()
        else:
            rows = self.conn.execute(
                sql.format("AND tag = ANY(?)"), [list(tags)]).fetchall()
        return {t: float(v) for t, v in rows}

    def raw(self, tag: str, t0: datetime, t1: datetime) -> pd.DataFrame:
        """Every stored sample for one tag, unresampled."""
        return self.conn.execute(
            "SELECT ts, value, quality FROM samples "
            "WHERE tag = ? AND ts BETWEEN ? AND ? ORDER BY ts",
            [tag, t0, t1]).df()

    def get_window(
        self,
        tags: Sequence[str],
        t0: datetime,
        t1: datetime,
        freq: float = 1.0,
        include_bad: bool = False,
    ) -> pd.DataFrame:
        """A regular grid from `t0` to `t1` inclusive, one column per tag.

        Values are carried forward from the last sample at or before each grid
        instant, including samples taken before `t0`, so a window never opens
        with a column of nulls just because that tag happened to be quiet.
        """
        unknown = set(tags) - set(TAGS)
        if unknown:
            raise UnknownTag(f"not in the tag dictionary: {sorted(unknown)}")
        if freq <= 0.0:
            raise ValueError("freq must be positive")
        if t1 < t0:
            raise ValueError("t1 precedes t0")

        quality_clause = "" if include_bad else "AND quality = 0"
        sql = f"""
        WITH grid AS (
            SELECT g.ts, t.tag
            FROM (SELECT unnest(generate_series(?, ?, to_microseconds(?))) AS ts) g
            CROSS JOIN (SELECT unnest(?::VARCHAR[]) AS tag) t
        ),
        history AS (
            SELECT ts, tag, value FROM samples
            WHERE tag = ANY(?) AND ts <= ? {quality_clause}
        )
        SELECT grid.ts, grid.tag, history.value
        FROM grid ASOF LEFT JOIN history
          ON grid.tag = history.tag AND grid.ts >= history.ts
        ORDER BY grid.ts, grid.tag
        """
        micros = int(round(freq * 1_000_000))
        long = self.conn.execute(
            sql, [t0, t1, micros, list(tags), list(tags), t1]).df()
        wide = long.pivot(index="ts", columns="tag", values="value")
        return wide.reindex(columns=list(tags))

    def latest_window(self, tags: Sequence[str], seconds: float,
                      freq: float = 1.0) -> pd.DataFrame:
        """The last `seconds` of history. What the inference service asks for."""
        _, end = self.span()
        if end is None:
            return pd.DataFrame(columns=list(tags))
        return self.get_window(tags, end - timedelta(seconds=seconds), end, freq)

    # -- export -----------------------------------------------------------

    def export_parquet(self, path: str, t0: datetime | None = None,
                       t1: datetime | None = None) -> None:
        where, params = "", []
        if t0 is not None and t1 is not None:
            where, params = "WHERE ts BETWEEN ? AND ?", [t0, t1]
        self.conn.execute(
            f"COPY (SELECT * FROM samples {where} ORDER BY ts) "
            f"TO '{path}' (FORMAT PARQUET)", params)
