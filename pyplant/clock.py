"""Drift-free periodic scheduling.

`await asyncio.sleep(period)` inside a loop schedules the *next* wake relative
to the moment the last one finished, so every microsecond of work in the body
is added permanently to the phase. Over an hour at 100 ms that is minutes of
accumulated slip, and two loops running at different periods drift apart from
each other as well as from the wall clock.

A `Ticker` instead computes absolute targets from a fixed origin::

    target(n) = origin + n * period / speed

and sleeps until that instant. Work in the body eats into the slack rather
than into the phase, so tick n always lands near its nominal time no matter
what happened during tick n-1. When the body takes longer than the period the
tick is reported late and, past a full period, as an overrun; the schedule
itself never shifts.

`speed` is a real-time factor: 1.0 is wall clock, 100.0 runs a hundred times
faster, which is how a scenario that takes an hour on the plant is exercised in
36 seconds. `clock` and `sleeper` are injectable so tests can drive a virtual
clock and stay deterministic instead of racing a real one.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable


@dataclass(frozen=True)
class Tick:
    """What happened on one tick."""

    index: int
    target: float  # when it should have fired
    actual: float  # when it did
    period: float  # scaled period actually in use

    @property
    def lateness(self) -> float:
        return self.actual - self.target

    @property
    def overrun(self) -> bool:
        """Late by a full period or more: a slot was missed."""
        return self.lateness >= self.period


@dataclass
class Ticker:
    """Periodic scheduler with a fixed phase origin."""

    period: float  # s, in plant time
    speed: float = 1.0  # >1 runs faster than real time
    clock: Callable[[], float] = time.perf_counter
    sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep

    index: int = field(default=0, init=False)
    origin: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.period <= 0.0:
            raise ValueError("period must be positive")
        if self.speed <= 0.0:
            raise ValueError("speed must be positive")

    @property
    def scaled_period(self) -> float:
        return self.period / self.speed

    def start(self) -> None:
        """Fix the phase origin. Called automatically on the first wait."""
        self.origin = self.clock()
        self.index = 0

    async def wait(self) -> Tick:
        """Sleep until the next scheduled instant and return what happened."""
        if self.origin is None:
            self.start()
            return Tick(0, self.origin, self.origin, self.scaled_period)

        self.index += 1
        target = self.origin + self.index * self.scaled_period
        delay = target - self.clock()
        if delay > 0.0:
            await self.sleeper(delay)
        else:
            # Late already. Yield so a starved sibling task can still run.
            await self.sleeper(0.0)
        return Tick(self.index, target, self.clock(), self.scaled_period)


class ManualClock:
    """Virtual clock for tests. Time only moves when something sleeps.

    Single-task use only: with two coroutines sharing one of these, each would
    advance time out from under the other.
    """

    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds

    async def sleep(self, delay: float) -> None:
        self.t += max(delay, 0.0)
        await asyncio.sleep(0)  # give the loop a chance to switch tasks
