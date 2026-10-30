"""Tests for the tag dictionary.

The codec tests matter more than they look. Every service in the stack encodes
and decodes through this module, so a silent factor-of-ten here shows up much
later as a level that reads 4520 in one process and 452.0 in another.
"""

from __future__ import annotations

import random

import pytest

from pyplant.tags import (
    DIAG_ORDER,
    INT16_MAX,
    INT16_MIN,
    TAGS,
    Kind,
    Limits,
    Owner,
    Table,
    Tag,
    BLOCKS,
    block,
    blocks_owned_by,
    by_owner,
    by_table,
    decode,
    encode,
    feature_tags,
    register_map,
    tag,
    tags_in,
    writable,
)

ANALOGS = [t for t in TAGS.values() if t.kind is Kind.ANALOG]
BOOLS = [t for t in TAGS.values() if t.kind is Kind.BOOL]


# --- registry integrity ----------------------------------------------------

def test_names_are_unique_and_match_keys():
    assert all(name == t.name for name, t in TAGS.items())


def test_no_address_collisions():
    seen: set[tuple[Table, int]] = set()
    for t in TAGS.values():
        key = (t.table, t.address)
        assert key not in seen, f"{t.name} collides at {key}"
        seen.add(key)


def test_kind_matches_table():
    assert all(t.table is Table.HR for t in ANALOGS)
    assert all(t.table is Table.COIL for t in BOOLS)


def test_every_analog_range_fits_int16():
    for t in ANALOGS:
        for edge in (t.lo, t.hi):
            assert INT16_MIN <= round(edge * t.scale) <= INT16_MAX, t.name


def test_alarm_limits_lie_inside_range():
    for t in ANALOGS:
        for label, value in t.limits.as_dict().items():
            assert t.lo <= value <= t.hi, f"{t.name}.{label}"


def test_writable_set_is_exactly_what_we_intend():
    # Guards against a measurement accidentally becoming operator-writable.
    assert {t.name for t in writable()} == {
        "LIC101_SP", "LIC101_MAN_OP", "LIC101_AUTO",
        "CMD_ALM_RESET", "CMD_PMP101_START",
    }


def test_diag_order_is_valid():
    assert DIAG_ORDER[0] == "NONE"
    assert all(name in TAGS for name in DIAG_ORDER[1:])
    assert len(set(DIAG_ORDER)) == len(DIAG_ORDER)


def test_opcua_paths_are_unique():
    paths = [t.path for t in TAGS.values()]
    assert len(set(paths)) == len(paths)


def test_feature_tags_all_exist():
    assert all(name in TAGS for name in feature_tags())


# --- codec -----------------------------------------------------------------

def test_analog_roundtrip_within_resolution():
    rng = random.Random(0)
    for t in ANALOGS:
        for _ in range(200):
            v = rng.uniform(t.lo, t.hi)
            back = t.decode(t.encode(v))
            assert abs(back - v) <= t.resolution / 2 + 1e-9, t.name


def test_range_edges_roundtrip_exactly():
    for t in ANALOGS:
        for edge in (t.lo, t.hi):
            assert t.decode(t.encode(edge)) == pytest.approx(edge)


def test_encoded_words_are_unsigned_16_bit():
    for t in ANALOGS:
        for v in (t.lo, (t.lo + t.hi) / 2, t.hi):
            assert 0 <= t.encode(v) <= 0xFFFF


def test_negative_values_use_twos_complement():
    t = tag("LIC101_ERR")  # -600..600 mm, scale 10
    assert t.encode(-250.0) == (-2500) & 0xFFFF
    assert t.decode(t.encode(-250.0)) == pytest.approx(-250.0)
    assert t.encode(-250.0) > INT16_MAX  # really is the high half of the word


def test_out_of_range_is_clamped_not_wrapped():
    t = tag("LT101")  # 0..600 mm
    assert t.decode(t.encode(9_999.0)) == pytest.approx(t.hi)
    assert t.decode(t.encode(-9_999.0)) == pytest.approx(t.lo)


def test_bool_codec():
    t = tag("PMP101_RUN")
    assert t.encode(True) == 1 and t.encode(False) == 0
    assert t.decode(1) is True and t.decode(0) is False


def test_module_level_helpers_match_tag_methods():
    assert encode("LT101", 123.4) == tag("LT101").encode(123.4)
    assert decode("LT101", 1234) == tag("LT101").decode(1234)


def test_unknown_tag_raises():
    with pytest.raises(KeyError):
        tag("NOPE")


# --- lookups ---------------------------------------------------------------

def test_by_owner_partitions_the_registry():
    total = sum(len(by_owner(o)) for o in Owner)
    assert total == len(TAGS)


def test_by_table_is_address_sorted():
    for table in Table:
        addrs = [t.address for t in by_table(table)]
        assert addrs == sorted(addrs)


def test_register_map_round_trips():
    for table in Table:
        m = register_map(table)
        assert all(m[t.address] is t for t in by_table(table))


def test_blocks_have_no_holes():
    # Every address inside a polled block must be a real tag, or a read wastes
    # a word and an address-plan typo goes unnoticed.
    for b in BLOCKS.values():
        assert len(tags_in(b.name)) == b.count, b.name


def test_blocks_do_not_overlap():
    for a in BLOCKS.values():
        for c in BLOCKS.values():
            if a is not c and a.table is c.table:
                assert not (a.start < c.end and c.start < a.end), (a.name, c.name)


def test_every_tag_lives_in_exactly_one_block():
    covered = [t.name for b in BLOCKS for t in tags_in(b)]
    assert sorted(covered) == sorted(TAGS)


def test_block_owner_matches_tag_owner():
    for b in BLOCKS.values():
        assert all(t.owner is b.owner for t in tags_in(b.name)), b.name


def test_blocks_owned_by_partitions_the_block_map():
    assert sum(len(blocks_owned_by(o)) for o in Owner) == len(BLOCKS)


def test_unknown_block_raises():
    with pytest.raises(KeyError):
        block("NOPE")


# --- construction guards ---------------------------------------------------

def test_scale_overflow_is_rejected():
    with pytest.raises(ValueError, match="outside int16"):
        Tag("XX101", "too big", Kind.ANALOG, Owner.PLANT, Table.HR, 900,
            unit="mm", lo=0.0, hi=600.0, scale=1000)


def test_lowercase_name_is_rejected():
    with pytest.raises(ValueError, match="UPPER_CASE"):
        Tag("lt999", "bad name", Kind.ANALOG, Owner.PLANT, Table.HR, 901,
            unit="mm", lo=0.0, hi=10.0, scale=10)


def test_bool_in_holding_register_is_rejected():
    with pytest.raises(ValueError, match="must live in COIL"):
        Tag("XX102", "bad table", Kind.BOOL, Owner.PLANT, Table.HR, 902)


def test_limit_outside_range_is_rejected():
    with pytest.raises(ValueError, match="outside range"):
        Tag("XX103", "bad limit", Kind.ANALOG, Owner.PLANT, Table.HR, 903,
            unit="mm", lo=0.0, hi=100.0, scale=10, limits=Limits(hihi=500.0))


def test_inverted_range_is_rejected():
    with pytest.raises(ValueError, match="hi must exceed lo"):
        Tag("XX104", "inverted", Kind.ANALOG, Owner.PLANT, Table.HR, 904,
            unit="mm", lo=10.0, hi=0.0, scale=10)


def test_no_block_is_written_by_two_parties():
    # Each block has exactly one owner by construction, but the point of the
    # rule is the pairing: everything an operator can write lives in a block
    # that no other node publishes.
    from pyplant.tags import Owner as _Owner

    oper_blocks = {b.name for b in BLOCKS.values() if b.owner is _Owner.OPER}
    oper_tags = {t.name for b in oper_blocks for t in tags_in(b)}
    assert oper_tags == {t.name for t in writable()}
    for b in BLOCKS.values():
        if b.name not in oper_blocks:
            assert not {t.name for t in tags_in(b.name)} & oper_tags
