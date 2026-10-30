"""A Modbus datastore addressed by tag name.

pymodbus stores 16-bit words and bits. Everything above this module thinks in
millimetres, litres per minute and booleans. `TagStore` is the only place the
two meet, and it does the conversion through the step-0 codec so a scale factor
exists exactly once in the project.

Reads and writes are by *block*, not by tag. One `read_holding_registers` for
six measurements costs one round trip; six separate reads cost six, and on a
100 ms scan that difference is the whole timing budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pymodbus.datastore import (
    ModbusSequentialDataBlock,
    ModbusServerContext,
    ModbusSlaveContext,
)

from pyplant.tags import BLOCKS, TAGS, Block, Table, block, tags_in

# Modbus function codes, as pymodbus wants them for datastore access.
FC_COIL = 1
FC_HOLDING = 3

_FC = {Table.COIL: FC_COIL, Table.HR: FC_HOLDING}

# Allocated generously: the whole registry fits well inside this and a sparse
# map is easier to read than a tightly packed one.
_SIZE = 64


class BlockError(LookupError):
    """A block was addressed that this store does not carry."""


@dataclass
class TagStore:
    """Holds the blocks one node serves, and converts on the way in and out."""

    blocks: tuple[str, ...]
    slave: ModbusSlaveContext = field(init=False)
    context: ModbusServerContext = field(init=False)

    def __post_init__(self) -> None:
        unknown = [b for b in self.blocks if b not in BLOCKS]
        if unknown:
            raise BlockError(f"unknown blocks: {unknown}")
        self.slave = ModbusSlaveContext(
            hr=ModbusSequentialDataBlock(0, [0] * _SIZE),
            co=ModbusSequentialDataBlock(0, [False] * _SIZE),
        )
        self.context = ModbusServerContext(slaves=self.slave, single=True)

    # -- guards -----------------------------------------------------------

    def _block(self, name: str) -> Block:
        if name not in self.blocks:
            raise BlockError(
                f"this node does not carry block {name!r}; it has {self.blocks}"
            )
        return block(name)

    # -- whole-block access ------------------------------------------------

    def publish(self, name: str, values: dict[str, float | bool]) -> None:
        """Write every tag of a block in one datastore operation.

        Every tag in the block must be supplied. A partial publish would leave
        a stale word next to a fresh one inside what a client reads as a single
        consistent snapshot.
        """
        b = self._block(name)
        members = tags_in(name)
        missing = [t.name for t in members if t.name not in values]
        if missing:
            raise KeyError(f"publish of {name} is missing {missing}")
        words = [t.encode(values[t.name]) for t in members]
        self.slave.setValues(_FC[b.table], b.start, words)

    def read(self, name: str) -> dict[str, float | bool]:
        """Read a whole block back in engineering units."""
        b = self._block(name)
        raw = self.slave.getValues(_FC[b.table], b.start, b.count)
        return {t.name: t.decode(int(v)) for t, v in zip(tags_in(name), raw)}

    # -- single tag, for tests and for the HMI ----------------------------

    def set(self, tag_name: str, value: float | bool) -> None:
        t = TAGS[tag_name]
        self._block(_home_block(tag_name))
        self.slave.setValues(_FC[t.table], t.address, [t.encode(value)])

    def get(self, tag_name: str) -> float | bool:
        t = TAGS[tag_name]
        self._block(_home_block(tag_name))
        return t.decode(int(self.slave.getValues(_FC[t.table], t.address, 1)[0]))

    def snapshot(self) -> dict[str, float | bool]:
        """Everything this node carries, for logging and for the HMI."""
        out: dict[str, float | bool] = {}
        for name in self.blocks:
            out.update(self.read(name))
        return out


def _home_block(tag_name: str) -> str:
    t = TAGS[tag_name]
    for b in BLOCKS.values():
        if b.table is t.table and b.start <= t.address < b.end:
            return b.name
    raise BlockError(f"{tag_name} belongs to no block")  # pragma: no cover
