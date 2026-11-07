"""A market of its own on the dashboard: its clock, its run loop and who to tell (the Pairs and Stat-arb views)."""

from __future__ import annotations

import asyncio
import time
from typing import Optional

FRAME = 0.05  # seconds between the run loop's frames
CHUNK = 2_000  # events between yields while a step request runs


class LiveMarket:
    """What a view's market does on its own: runs on at a speed in units of time a second, and wakes its streams.

    A subclass has ``market``, with its ``time``, ``_event`` for the next event and what it changes, and whatever makes
    the market; it adds to ``version`` when the chart starts over, and to ``changes`` when what the page shows changes
    outside the clock.
    """

    def __init__(self, speed: int) -> None:
        self.speed = speed
        self.is_running = False
        self.version = 0  # incremented when the chart starts over (a reset)
        self.changes = 0  # incremented by whatever changes what the page shows outside the clock
        self._subscribers: set[asyncio.Queue] = set()
        self._task: Optional[asyncio.Task] = None
        self.lock = asyncio.Lock()

    def _event(self) -> None:
        raise NotImplementedError

    def run_for(self, units: float) -> None:
        """Run on by ``units`` of time."""
        until = self.market.time + units
        while self.market.time < until:
            self._event()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def notify(self) -> None:
        """Wake every event stream (a full queue drops its oldest wake-up: one is enough)."""
        for q in list(self._subscribers):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(True)

    async def broadcast(self) -> None:
        self.notify()

    async def start(self) -> None:
        self.is_running = True
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())
        self.changes += 1
        await self.broadcast()

    async def stop(self) -> None:
        self.is_running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        self.changes += 1
        await self.broadcast()

    async def advance(self, units: float) -> None:
        """Run on by ``units`` of time now (paused or not), yielding to the event loop every ``CHUNK`` events."""
        async with self.lock:
            until = self.market.time + units
            while self.market.time < until:
                for _ in range(CHUNK):
                    self._event()
                    if self.market.time >= until:
                        break
                await asyncio.sleep(0)
        await self.broadcast()

    def shutdown(self) -> None:
        """Stop the run loop without waiting for it (when the session is evicted)."""
        self.is_running = False
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None
        self._subscribers.clear()

    async def _run(self) -> None:
        # The clock follows the wall clock: each frame runs on by the speed times the time since the last (at
        # most five frames' worth, so a stall is not made up in a burst), to a running target, so a frame's
        # overshoot (up to one event) comes off the next one's run rather than adding up. The target is taken
        # afresh after a reset, or when the clock has moved far from it (a step request).
        target, version, last = self.market.time, self.version, time.monotonic()
        try:
            while self.is_running:
                await asyncio.sleep(FRAME)
                now = time.monotonic()
                elapsed, last = min(now - last, 5 * FRAME), now
                async with self.lock:
                    if version != self.version or abs(self.market.time - target) > max(1.0, 5 * self.speed * FRAME):
                        target, version = self.market.time, self.version
                    target += self.speed * elapsed
                    while self.market.time < target:
                        self._event()
                self.notify()
        except asyncio.CancelledError:
            pass
        finally:
            if self._task is asyncio.current_task():  # ended by an error, not by stop(): show it paused
                self.is_running = False
                self.changes += 1
                self.notify()
