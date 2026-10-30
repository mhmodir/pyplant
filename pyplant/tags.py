"""Single source of truth for every signal in the stack.

Every process - plant simulator, soft PLC, OPC UA server, historian, HMI and
the inference service - imports its I/O definitions from this module. Nothing
else in the project is permitted to hard-code a register address, a scale
factor or an engineering range.

Wire representation
-------------------
Analog signals live in 16-bit Modbus holding registers. An engineering value
is encoded as a signed 16-bit integer::

    raw = round(value * scale)

and transmitted as its unsigned two's-complement pattern, which is what
pymodbus puts on the wire. `scale` is chosen per tag so that the full
engineering range fits inside [-32768, 32767]; this is enforced at import.

Boolean signals live in coils and encode to 0 or 1.

Ownership
---------
`owner` names the process that is the source of truth for a tag. A process
writes only the tags it owns and treats every other tag as read-only. This is
what keeps the plant from writing controller outputs and the HMI from writing
measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

INT16_MIN = -32768
INT16_MAX = 32767
UINT16_MASK = 0xFFFF


class Table(str, Enum):
    """Modbus data table the tag lives in."""

    HR = "HR"  # 16-bit holding register
    COIL = "COIL"  # single bit


class Kind(str, Enum):
    ANALOG = "ANALOG"
    BOOL = "BOOL"


class Owner(str, Enum):
    """Process that is the source of truth for the tag."""

    PLANT = "PLANT"  # physics simulator
    PLC = "PLC"  # soft PLC scan engine
    OPER = "OPER"  # the operator, through the HMI
    ML = "ML"  # inference service


@dataclass(frozen=True)
class Limits:
    """Alarm limits in engineering units. None means the limit is not used."""

    lolo: float | None = None
    lo: float | None = None
    hi: float | None = None
    hihi: float | None = None

    def as_dict(self) -> dict[str, float]:
        return {k: v for k, v in vars(self).items() if v is not None}


NO_LIMITS = Limits()


@dataclass(frozen=True)
class Block:
    """A contiguous, single-owner span of one Modbus table.

    Clients poll whole blocks in one transaction rather than issuing a request
    per tag. Blocks are declared rather than derived so that a mis-addressed
    tag fails at import instead of quietly widening a read.
    """

    name: str
    table: Table
    start: int
    count: int
    owner: Owner
    desc: str

    @property
    def end(self) -> int:
        """One past the last address in the block."""
        return self.start + self.count


@dataclass(frozen=True)
class Tag:
    name: str
    desc: str
    kind: Kind
    owner: Owner
    table: Table
    address: int
    unit: str = ""
    lo: float = 0.0
    hi: float = 1.0
    scale: int = 1
    writable: bool = False  # may a remote client write it (setpoint, reset)
    area: str = "Plant"  # OPC UA folder path
    limits: Limits = NO_LIMITS

    # ---- validation -----------------------------------------------------

    def __post_init__(self) -> None:
        if not self.name.isidentifier() or self.name != self.name.upper():
            raise ValueError(f"{self.name!r}: name must be an UPPER_CASE identifier")
        if self.address < 0:
            raise ValueError(f"{self.name}: address must be non-negative")
        if self.scale < 1 or int(self.scale) != self.scale:
            raise ValueError(f"{self.name}: scale must be a positive integer")

        if self.kind is Kind.ANALOG:
            if self.table is not Table.HR:
                raise ValueError(f"{self.name}: analog tags must live in HR")
            if self.hi <= self.lo:
                raise ValueError(f"{self.name}: hi must exceed lo")
            for edge in (self.lo, self.hi):
                raw = round(edge * self.scale)
                if not INT16_MIN <= raw <= INT16_MAX:
                    raise ValueError(
                        f"{self.name}: {edge} {self.unit} x scale {self.scale} "
                        f"= {raw}, outside int16. Reduce scale."
                    )
            for label, value in self.limits.as_dict().items():
                if not self.lo <= value <= self.hi:
                    raise ValueError(
                        f"{self.name}: {label} limit {value} outside range "
                        f"[{self.lo}, {self.hi}]"
                    )
        else:
            if self.table is not Table.COIL:
                raise ValueError(f"{self.name}: boolean tags must live in COIL")
            if self.scale != 1:
                raise ValueError(f"{self.name}: boolean tags must have scale 1")
            if self.limits != NO_LIMITS:
                raise ValueError(f"{self.name}: boolean tags cannot carry limits")

    # ---- derived --------------------------------------------------------

    @property
    def resolution(self) -> float:
        """Smallest change representable on the wire, in engineering units."""
        return 1.0 / self.scale

    @property
    def path(self) -> str:
        """OPC UA browse path, e.g. 'Plant/Unit1/LT101'."""
        return f"{self.area}/{self.name}"

    # ---- codec ----------------------------------------------------------

    def clamp(self, value: float) -> float:
        return min(max(float(value), self.lo), self.hi)

    def encode(self, value: float | bool) -> int:
        """Engineering value -> unsigned 16-bit wire word (or coil bit)."""
        if self.kind is Kind.BOOL:
            return 1 if value else 0
        raw = round(self.clamp(value) * self.scale)
        return raw & UINT16_MASK

    def decode(self, raw: int) -> float | bool:
        """Wire word -> engineering value."""
        if self.kind is Kind.BOOL:
            return bool(raw)
        raw &= UINT16_MASK
        if raw > INT16_MAX:  # two's complement
            raw -= 1 << 16
        return raw / self.scale


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# Address plan
#   HR   0- 9  plant measurements       (PLANT -> PLC)
#   HR  10-19  actuator commands        (PLC   -> PLANT)
#   HR  20-21  operator setpoints       (HMI   -> PLC)
#   HR  22-29  controller faceplate     (PLC   -> HMI)
#   HR  30-39  PLC diagnostics          (PLC   -> HMI)
#   HR  40-49  inference results        (ML    -> HMI)
#   COIL 0- 9  plant status feedback    (PLANT -> PLC)
#   COIL 10    actuator command        (PLC   -> PLANT)
#   COIL 11-19 operator commands        (HMI   -> PLC)
#   COIL 20-29 alarms and interlocks    (PLC   -> HMI)
#   COIL 30-39 inference alarms         (ML    -> HMI)

_TAG_LIST: tuple[Tag, ...] = (
    # -- plant measurements ------------------------------------------------
    Tag("LT101", "Tank 1 level", Kind.ANALOG, Owner.PLANT, Table.HR, 0,
        unit="mm", lo=0.0, hi=600.0, scale=10, area="Plant/Unit1",
        limits=Limits(lolo=50.0, lo=100.0, hi=500.0, hihi=550.0)),
    Tag("LT102", "Tank 2 level", Kind.ANALOG, Owner.PLANT, Table.HR, 1,
        unit="mm", lo=0.0, hi=600.0, scale=10, area="Plant/Unit1",
        limits=Limits(hi=500.0, hihi=550.0)),
    Tag("FT101", "Pump 1 discharge flow", Kind.ANALOG, Owner.PLANT, Table.HR, 2,
        unit="L/min", lo=0.0, hi=20.0, scale=100, area="Plant/Unit1",
        limits=Limits(lo=1.0)),
    Tag("FT102", "Tank 2 outlet flow", Kind.ANALOG, Owner.PLANT, Table.HR, 3,
        unit="L/min", lo=0.0, hi=20.0, scale=100, area="Plant/Unit1"),
    Tag("PT101", "Pump 1 discharge pressure", Kind.ANALOG, Owner.PLANT, Table.HR, 4,
        unit="kPa", lo=0.0, hi=100.0, scale=100, area="Plant/Unit1",
        limits=Limits(hihi=58.0)),
    Tag("ZT101", "FCV101 measured stem position", Kind.ANALOG, Owner.PLANT, Table.HR, 5,
        unit="%", lo=0.0, hi=100.0, scale=100, area="Plant/Unit1"),
    Tag("PMP101_RUN", "Pump 1 running feedback", Kind.BOOL, Owner.PLANT, Table.COIL, 0,
        area="Plant/Unit1"),

    # -- actuator commands -------------------------------------------------
    Tag("FCV101_OP", "FCV101 commanded position", Kind.ANALOG, Owner.PLC, Table.HR, 10,
        unit="%", lo=0.0, hi=100.0, scale=100, area="Plant/Unit1"),
    Tag("PMP101_CMD", "Pump 1 start command", Kind.BOOL, Owner.PLC, Table.COIL, 10,
        area="Plant/Unit1"),

    # -- controller faceplate ---------------------------------------------
    Tag("LIC101_SP", "LIC101 setpoint", Kind.ANALOG, Owner.OPER, Table.HR, 20,
        unit="mm", lo=0.0, hi=600.0, scale=10, writable=True, area="PLC/LIC101"),
    Tag("LIC101_MAN_OP", "LIC101 manual output demand", Kind.ANALOG, Owner.OPER,
        Table.HR, 21, unit="%", lo=0.0, hi=100.0, scale=100, writable=True,
        area="PLC/LIC101"),
    Tag("LIC101_PV", "LIC101 process value", Kind.ANALOG, Owner.PLC, Table.HR, 22,
        unit="mm", lo=0.0, hi=600.0, scale=10, area="PLC/LIC101"),
    Tag("LIC101_OP", "LIC101 controller output", Kind.ANALOG, Owner.PLC, Table.HR, 23,
        unit="%", lo=0.0, hi=100.0, scale=100, area="PLC/LIC101"),
    Tag("LIC101_ERR", "LIC101 control error", Kind.ANALOG, Owner.PLC, Table.HR, 24,
        unit="mm", lo=-600.0, hi=600.0, scale=10, area="PLC/LIC101"),
    Tag("LIC101_AUTO", "LIC101 in automatic", Kind.BOOL, Owner.OPER, Table.COIL, 11,
        writable=True, area="PLC/LIC101"),
    Tag("CMD_PMP101_START", "Operator pump start request", Kind.BOOL, Owner.OPER,
        Table.COIL, 13, writable=True, area="PLC/Commands"),

    # -- PLC diagnostics ---------------------------------------------------
    Tag("SCAN_MS", "PLC scan time", Kind.ANALOG, Owner.PLC, Table.HR, 30,
        unit="ms", lo=0.0, hi=1000.0, scale=10, area="PLC/Diagnostics",
        limits=Limits(hi=80.0, hihi=150.0)),
    Tag("SCAN_OVERRUNS", "PLC scan overrun count", Kind.ANALOG, Owner.PLC, Table.HR, 31,
        unit="count", lo=0.0, hi=30000.0, scale=1, area="PLC/Diagnostics"),
    Tag("PLC_HEARTBEAT", "PLC heartbeat counter", Kind.ANALOG, Owner.PLC, Table.HR, 32,
        unit="count", lo=0.0, hi=30000.0, scale=1, area="PLC/Diagnostics"),

    # -- alarms and interlocks --------------------------------------------
    Tag("ALM_LT101_HI", "Tank 1 level high", Kind.BOOL, Owner.PLC, Table.COIL, 20,
        area="PLC/Alarms"),
    Tag("ALM_LT101_LOLO", "Tank 1 level low low", Kind.BOOL, Owner.PLC, Table.COIL, 21,
        area="PLC/Alarms"),
    Tag("ALM_PT101_HIHI", "Pump 1 discharge pressure high high", Kind.BOOL, Owner.PLC,
        Table.COIL, 22, area="PLC/Alarms"),
    Tag("ILK_PMP101", "Pump 1 interlock latched", Kind.BOOL, Owner.PLC, Table.COIL, 23,
        area="PLC/Alarms"),
    Tag("CMD_ALM_RESET", "Alarm and interlock reset", Kind.BOOL, Owner.OPER,
        Table.COIL, 12, writable=True, area="PLC/Alarms"),

    # -- inference results -------------------------------------------------
    Tag("HEALTH_INDEX", "Model health index", Kind.ANALOG, Owner.ML, Table.HR, 40,
        unit="%", lo=0.0, hi=100.0, scale=100, area="ML/Diagnostics",
        limits=Limits(hi=60.0, hihi=85.0)),
    Tag("TOP_CONTRIB", "Top contributing tag, index into DIAG_ORDER",
        Kind.ANALOG, Owner.ML, Table.HR, 41,
        unit="enum", lo=0.0, hi=64.0, scale=1, area="ML/Diagnostics"),
    Tag("ML_HEARTBEAT", "Inference service heartbeat", Kind.ANALOG, Owner.ML, Table.HR, 42,
        unit="count", lo=0.0, hi=30000.0, scale=1, area="ML/Diagnostics"),
    Tag("ALM_HEALTH", "Health index degraded", Kind.BOOL, Owner.ML, Table.COIL, 30,
        area="ML/Diagnostics"),
)

TAGS: dict[str, Tag] = {t.name: t for t in _TAG_LIST}

# Ordinal encoding for TOP_CONTRIB. Index 0 means "no single dominant tag".
# Append only - never reorder, or historised values change meaning.
DIAG_ORDER: tuple[str, ...] = (
    "NONE", "LT101", "LT102", "FT101", "FT102", "PT101", "ZT101", "FCV101_OP",
)


_BLOCK_LIST: tuple[Block, ...] = (
    Block("PLANT_MEAS", Table.HR, 0, 6, Owner.PLANT, "plant measurements"),
    Block("PLANT_STATUS", Table.COIL, 0, 1, Owner.PLANT, "plant status feedback"),
    Block("PLC_CMD", Table.HR, 10, 1, Owner.PLC, "analog actuator commands"),
    Block("PLC_CMD_BITS", Table.COIL, 10, 1, Owner.PLC, "discrete actuator commands"),
    Block("OPER_SP", Table.HR, 20, 2, Owner.OPER, "operator setpoint and manual demand"),
    Block("OPER_CMD", Table.COIL, 11, 3, Owner.OPER, "operator command bits"),
    Block("PLC_FACE", Table.HR, 22, 3, Owner.PLC, "LIC101 faceplate readback"),
    Block("PLC_DIAG", Table.HR, 30, 3, Owner.PLC, "scan engine diagnostics"),
    Block("PLC_ALARMS", Table.COIL, 20, 4, Owner.PLC, "alarms and interlocks"),
    Block("ML_RESULT", Table.HR, 40, 3, Owner.ML, "inference results"),
    Block("ML_ALARMS", Table.COIL, 30, 1, Owner.ML, "inference alarms"),
)

BLOCKS: dict[str, Block] = {b.name: b for b in _BLOCK_LIST}


def _validate_registry() -> None:
    seen: dict[tuple[Table, int], str] = {}
    for tag in _TAG_LIST:
        key = (tag.table, tag.address)
        if key in seen:
            raise ValueError(
                f"address collision: {tag.name} and {seen[key]} both at "
                f"{tag.table.value}[{tag.address}]"
            )
        seen[key] = tag.name
    if len(TAGS) != len(_TAG_LIST):
        raise ValueError("duplicate tag name in registry")
    for name in DIAG_ORDER[1:]:
        if name not in TAGS:
            raise ValueError(f"DIAG_ORDER references unknown tag {name!r}")
    if len(DIAG_ORDER) - 1 > TAGS["TOP_CONTRIB"].hi:
        raise ValueError("DIAG_ORDER is longer than TOP_CONTRIB can encode")

    # Blocks must not overlap.
    for a in _BLOCK_LIST:
        for b in _BLOCK_LIST:
            if a is not b and a.table is b.table and a.start < b.end and b.start < a.end:
                raise ValueError(f"blocks {a.name} and {b.name} overlap")

    # Every tag sits in exactly one block, owned by the same process.
    for t in _TAG_LIST:
        home = [b for b in _BLOCK_LIST
                if b.table is t.table and b.start <= t.address < b.end]
        if len(home) != 1:
            raise ValueError(
                f"{t.name} at {t.table.value}[{t.address}] belongs to "
                f"{len(home)} blocks, expected exactly 1"
            )
        if home[0].owner is not t.owner:
            raise ValueError(
                f"{t.name} is owned by {t.owner.value} but sits in block "
                f"{home[0].name} owned by {home[0].owner.value}"
            )

    # No holes: a polled block that contains an undefined address wastes a
    # word and hides a typo in the address plan.
    for b in _BLOCK_LIST:
        for addr in range(b.start, b.end):
            if (b.table, addr) not in seen:
                raise ValueError(f"block {b.name} has no tag at address {addr}")


_validate_registry()


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def tag(name: str) -> Tag:
    try:
        return TAGS[name]
    except KeyError:
        raise KeyError(f"unknown tag {name!r}") from None


def encode(name: str, value: float | bool) -> int:
    return tag(name).encode(value)


def decode(name: str, raw: int) -> float | bool:
    return tag(name).decode(raw)


def by_owner(owner: Owner) -> tuple[Tag, ...]:
    return tuple(t for t in _TAG_LIST if t.owner is owner)


def by_table(table: Table) -> tuple[Tag, ...]:
    return tuple(sorted((t for t in _TAG_LIST if t.table is table),
                        key=lambda t: t.address))


def writable() -> tuple[Tag, ...]:
    return tuple(t for t in _TAG_LIST if t.writable)


def register_map(table: Table) -> dict[int, Tag]:
    return {t.address: t for t in by_table(table)}


def block(name: str) -> Block:
    try:
        return BLOCKS[name]
    except KeyError:
        raise KeyError(f"unknown block {name!r}") from None


def blocks_owned_by(owner: Owner) -> tuple[Block, ...]:
    return tuple(b for b in BLOCKS.values() if b.owner is owner)


def tags_in(block_name: str) -> tuple[Tag, ...]:
    b = block(block_name)
    return tuple(sorted(
        (t for t in _TAG_LIST
         if t.table is b.table and b.start <= t.address < b.start + b.count),
        key=lambda t: t.address,
    ))


def feature_tags() -> tuple[str, ...]:
    """Tags the diagnosis model consumes.

    Includes controller output and alarm bits, not just measurements: in a
    closed loop the controller absorbs part of the fault, so OP carries
    signature that PV does not.
    """
    return (
        "LT101", "LT102", "FT101", "FT102", "PT101", "ZT101",
        "FCV101_OP", "LIC101_SP", "LIC101_ERR",
        "PMP101_RUN", "LIC101_AUTO", "ALM_LT101_HI", "ALM_LT101_LOLO",
        "ALM_PT101_HIHI", "ILK_PMP101",
    )


def describe(tags: Iterable[Tag] | None = None) -> str:
    rows = list(tags if tags is not None else _TAG_LIST)
    head = f"{'TAG':<16}{'OWNER':<7}{'ADDR':<9}{'RANGE':<20}{'RES':<9}DESCRIPTION"
    lines = [head, "-" * len(head)]
    for t in rows:
        addr = f"{t.table.value}[{t.address}]"
        if t.kind is Kind.BOOL:
            rng, res = "0/1", "-"
        else:
            rng = f"{t.lo:g}..{t.hi:g} {t.unit}"
            res = f"{t.resolution:g}"
        flag = "*" if t.writable else " "
        lines.append(f"{t.name:<16}{t.owner.value:<7}{addr:<9}{rng:<20}{res:<9}{t.desc}{flag}")
    lines.append("")
    lines.append("* writable by a remote client")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
    print()
    print(f"{'BLOCK':<15}{'OWNER':<7}{'READ':<22}{'TAGS':<6}DESCRIPTION")
    print("-" * 72)
    for b in BLOCKS.values():
        span = f"{b.table.value}[{b.start}..{b.end - 1}] x{b.count}"
        print(f"{b.name:<15}{b.owner.value:<7}{span:<22}"
              f"{len(tags_in(b.name)):<6}{b.desc}")
