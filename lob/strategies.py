"""Quantitative strategies that trade in the simulated order book.

A strategy decides at regular intervals of simulation time from what a trader could see: the best
quotes, the lots resting at them, and its own position. It answers with a plan: quotes to rest in
the book (a market maker), or a position to hold (the signal strategies), which its runner reaches
either by crossing the spread (``aggressive``) or by resting an order at the best quote on its side
(``passive``). The runner (the dashboard's ``LOBManager``) carries plans out and keeps each
strategy's account, so a strategy holds only the state of its own signal.

- ``MarketMaking`` quotes a bid and an ask around the mid, leaning both against its inventory.
- ``Momentum`` follows the trend: long while a fast moving average of the mid is above a slow one
  by more than a threshold, short while it is below, flat once they cross back.
- ``MeanReversion`` fades large moves: short when the mid is far above its recent average (in
  standard deviations), long when far below, flat once it is back near the average.
- ``QueueImbalance`` reads the queues at the best quotes, after Gould and Bonart (2016): the
  side with fewer lots is likelier to run out first, so the mid tends to move toward it. Long once
  the bid queue is the larger by a margin, short once the ask queue is, flat when it tips back.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

from lob.order import Side

EXECUTIONS = ("aggressive", "passive")


@dataclass(frozen=True)
class View:
    """What a strategy sees when it decides."""
    time: float
    best_bid: Optional[int]
    best_ask: Optional[int]
    bid_depth: int  # visible lots at the best bid
    ask_depth: int
    position: int  # its own, in lots (positive when long)

    @property
    def mid(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2


@dataclass
class Plan:
    """What a strategy wants now: quotes to rest, or a position to hold (None: leave things as they are)."""
    quotes: Optional[dict[Side, tuple[int, int]]] = None  # side -> (price, lots), for a market maker
    target: Optional[int] = None  # the position it wants, in lots
    signal: Optional[float] = None  # the value it decided on, to show


class Strategy:
    """The interface: ``decide`` every ``interval`` of simulation time."""
    kind = ""
    interval = 1.0
    execution = "aggressive"
    protection = 5  # ticks an aggressive order may walk past the best opposite price

    def decide(self, view: View) -> Plan:
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError

    def warming_up(self) -> Optional[tuple[int, int]]:
        """(decisions seen, decisions needed) while it is still gathering history before it trades, else None."""
        return None

    def _every(self) -> str:
        return f"every {self.interval:g} time unit{'s' if self.interval != 1 else ''}"


class MarketMaking(Strategy):
    """Quote a bid and an ask ``half_spread`` ticks from the mid, ``size`` lots each.

    Both quotes move against its inventory, by ``lean`` half spreads at its limit (long: lower,
    so it buys less readily and sells more readily), never across the spread; at its limit it
    quotes only the side that reduces its inventory. New quotes every ``interval``: each joins
    the back of its price's queue.
    """
    kind = "market_making"

    def __init__(self, half_spread: int = 2, size: int = 5, limit: int = 50, lean: float = 0.5,
                 interval: float = 1.0):
        if limit < size:
            raise ValueError("the inventory limit is below the quote size")
        self.half_spread, self.size, self.limit, self.lean, self.interval = half_spread, size, limit, lean, interval

    def decide(self, view: View) -> Plan:
        mid = view.mid
        if mid is None:  # with a side empty there is no price to quote around
            return Plan(quotes={})
        position = view.position
        lean = self.lean * self.half_spread * position / self.limit
        bid = min(math.floor(mid - self.half_spread - lean), view.best_ask - 1)
        ask = max(math.ceil(mid + self.half_spread - lean), view.best_bid + 1)
        quotes = {}
        if position < self.limit and bid >= 1:
            quotes[Side.BID] = (bid, min(self.size, self.limit - position))
        if position > -self.limit:
            quotes[Side.ASK] = (ask, min(self.size, self.limit + position))
        return Plan(quotes=quotes)

    def describe(self) -> str:
        return (f"quotes {self.half_spread} tick{'s' if self.half_spread != 1 else ''} either side of the mid, "
                f"{self.size} lots a side, at most {self.limit} lots of inventory, leaning {self.lean:g}, "
                f"{self._every()}")


class _Positional(Strategy):
    """A strategy that holds -size, 0 or +size lots."""

    def __init__(self, size: int, interval: float, execution: str, protection: int):
        if execution not in EXECUTIONS:
            raise ValueError(f"execution must be one of {EXECUTIONS}")
        self.size, self.interval, self.execution, self.protection = size, interval, execution, protection
        self.target = 0

    def _how(self) -> str:
        if self.execution == "passive":
            return "resting at the best quote"
        return f"crossing the spread (walking at most {self.protection} ticks)"


class Momentum(_Positional):
    """Long while the fast moving average of the mid is above the slow one by more than ``threshold``.

    Exponential moving averages over ``fast`` and ``slow`` decisions. Short while the fast one is
    below by more than the threshold; flat once they cross back. Flat until it has seen ``slow``
    decisions.
    """
    kind = "momentum"

    def __init__(self, fast: int = 5, slow: int = 20, threshold: float = 0.5, size: int = 5,
                 interval: float = 0.5, execution: str = "aggressive", protection: int = 5):
        super().__init__(size, interval, execution, protection)
        if not 1 <= fast < slow:
            raise ValueError("the fast average must be shorter than the slow one")
        self.fast, self.slow, self.threshold = fast, slow, threshold
        self._fast = self._slow = None
        self._seen = 0

    def decide(self, view: View) -> Plan:
        mid = view.mid
        if mid is None:
            return Plan()
        if self._fast is None:
            self._fast = self._slow = mid
        else:
            self._fast += 2 / (self.fast + 1) * (mid - self._fast)
            self._slow += 2 / (self.slow + 1) * (mid - self._slow)
        self._seen += 1
        gap = self._fast - self._slow
        if self._seen < self.slow:
            return Plan(target=0, signal=gap)
        if gap > self.threshold:
            self.target = self.size
        elif gap < -self.threshold:
            self.target = -self.size
        elif (self.target > 0 and gap <= 0) or (self.target < 0 and gap >= 0):
            self.target = 0
        return Plan(target=self.target, signal=gap)

    def warming_up(self) -> Optional[tuple[int, int]]:
        return (self._seen, self.slow) if self._seen < self.slow else None

    def describe(self) -> str:
        return (f"long or short {self.size} lots while its {self.fast}- and {self.slow}-step averages of the mid "
                f"are more than {self.threshold:g} ticks apart, flat once they cross; {self._how()}, {self._every()}")


class MeanReversion(_Positional):
    """Short when the mid is more than ``entry`` standard deviations above its average over ``window`` decisions.

    Long when as far below; flat once it is back within ``exit`` of the average. Flat until the
    window is full.
    """
    kind = "mean_reversion"

    def __init__(self, window: int = 20, entry: float = 2.0, exit: float = 0.5, size: int = 5,
                 interval: float = 0.5, execution: str = "aggressive", protection: int = 5):
        super().__init__(size, interval, execution, protection)
        if not 0 <= exit < entry:
            raise ValueError("the exit level must be below the entry level")
        self.window, self.entry, self.exit = window, entry, exit
        self._mids: deque[float] = deque()
        self._sum = self._sum_sq = 0.0  # exact: mids are multiples of half a tick

    def decide(self, view: View) -> Plan:
        mid = view.mid
        if mid is None:
            return Plan()
        mids = self._mids
        mids.append(mid)
        self._sum += mid
        self._sum_sq += mid * mid
        if len(mids) > self.window:
            old = mids.popleft()
            self._sum -= old
            self._sum_sq -= old * old
        if len(mids) < self.window:
            return Plan(target=0)
        mean = self._sum / self.window
        variance = max(0.0, self._sum_sq / self.window - mean * mean)
        z = (mid - mean) / math.sqrt(variance) if variance > 0 else 0.0
        if z > self.entry:
            self.target = -self.size
        elif z < -self.entry:
            self.target = self.size
        elif abs(z) < self.exit:
            self.target = 0
        return Plan(target=self.target, signal=z)

    def warming_up(self) -> Optional[tuple[int, int]]:
        return (len(self._mids), self.window) if len(self._mids) < self.window else None

    def describe(self) -> str:
        return (f"short (long) {self.size} lots when the mid is more than {self.entry:g} standard deviations above "
                f"(below) its {self.window}-step average, flat within {self.exit:g}; {self._how()}, {self._every()}")


class QueueImbalance(_Positional):
    """Long once (bid lots - ask lots) / (bid lots + ask lots) at the best quotes is above ``threshold``.

    Short once it is below minus the threshold; flat when it tips to the other side of zero
    (Gould and Bonart 2016: the smaller queue is likelier to run out first, so the mid tends to
    move toward it). Waiting for the tip, rather than leaving as soon as the imbalance is back
    within the threshold, trades less often: each round trip pays the spread.
    """
    kind = "queue_imbalance"

    def __init__(self, threshold: float = 0.6, size: int = 5, interval: float = 0.25,
                 execution: str = "aggressive", protection: int = 5):
        super().__init__(size, interval, execution, protection)
        if not 0 <= threshold < 1:
            raise ValueError("the threshold must be from 0 to below 1")
        self.threshold = threshold

    def decide(self, view: View) -> Plan:
        total = view.bid_depth + view.ask_depth
        if view.mid is None or total == 0:  # nothing to read: hold
            return Plan()
        imbalance = (view.bid_depth - view.ask_depth) / total
        if imbalance > self.threshold:
            self.target = self.size
        elif imbalance < -self.threshold:
            self.target = -self.size
        elif (self.target > 0 and imbalance < 0) or (self.target < 0 and imbalance > 0):
            self.target = 0
        return Plan(target=self.target, signal=imbalance)

    def describe(self) -> str:
        return (f"long (short) {self.size} lots once the best bid's queue outweighs the ask's (or the reverse) by "
                f"an imbalance above {self.threshold:g}, flat when it tips the other way; {self._how()}, "
                f"{self._every()}")
