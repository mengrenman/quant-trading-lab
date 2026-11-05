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
- ``AvellanedaStoikov`` quotes around a reservation price that moves against its inventory, with a
  spread that widens with the price's volatility, after Avellaneda and Stoikov (2008).
- ``PennyJumping`` rests one tick better than the best bid and ask when the spread is wide enough,
  and joins them otherwise; it keeps its orders while they are where it wants them.
- ``Breakout`` follows a move out of the recent range: long above the highest mid of its window,
  short below the lowest, flat back past the middle (a Donchian channel).
- ``OrderFlowImbalance`` adds up the order flow imbalance at the best quotes over its window (Cont,
  Kukanov and Stoikov 2014), in average queue lengths: long when buying pressure is high, short
  when selling pressure is, flat when it tips back.
- ``RandomTrader`` holds a position drawn at random now and then: a benchmark for what trading
  without information costs.
- ``QuoteJoiner`` rests at the best bid and ask and does nothing else: the null for making markets.
  Where providing liquidity just breaks even, it makes nothing before fees.

A strategy that ``keeps_queue`` keeps a resting quote while it is at the price it wants (so it
keeps its place in the queue); the others' quotes are withdrawn and placed afresh each time.
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
    keeps_queue = False  # keep a resting quote while it is at the price wanted, rather than placing it afresh

    def decide(self, view: View) -> Plan:
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError

    def warming_up(self) -> Optional[tuple[int, int]]:
        """(decisions seen, decisions needed) while it is still gathering history before it trades, else None."""
        return None

    def state(self) -> dict:
        """What it has learned so far (not its settings), in JSON-friendly values, for a recording."""
        return {}

    def restore(self, state: dict) -> None:
        """Take up a state from ``state()``; raises ValueError for one this strategy could not have had."""

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


def _quotes(view: View, bid: Optional[int], ask: Optional[int], size: int, limit: int) -> dict[Side, tuple[int, int]]:
    """A quoting strategy's bid and ask: never across the spread; at its limit, only the side that reduces it."""
    position = view.position
    quotes = {}
    if bid is not None:
        bid = min(bid, view.best_ask - 1)
        if position < limit and bid >= 1:
            quotes[Side.BID] = (bid, min(size, limit - position))
    if ask is not None:
        ask = max(ask, view.best_bid + 1)
        if position > -limit:
            quotes[Side.ASK] = (ask, min(size, limit + position))
    return quotes


def _whole(value, low: int, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ValueError(f"{what} must be a whole number from {low}")
    return value


def _finite_list(values, most: int, what: str) -> list[float]:
    if not (isinstance(values, list) and len(values) <= most and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)):
        raise ValueError(f"{what} must be a list of at most {most} numbers")
    return [float(v) for v in values]


class AvellanedaStoikov(Strategy):
    """Quote around a reservation price, after Avellaneda and Stoikov (2008), in its stationary form.

    The reservation price is the mid less ``position * gamma * sigma^2 * horizon``: long, it would
    rather sell, so both quotes move down, more when the price is more volatile. The quotes are a
    total spread of ``gamma * sigma^2 * horizon + (2 / gamma) * ln(1 + gamma / k)`` apart around it:
    wider with volatility and risk aversion (``gamma``), narrower when orders arrive faster near the
    mid (the arrival rate falls as ``exp(-k * distance)``). ``sigma^2`` is the mid's variance per unit
    of time, from its changes over the last ``window`` decisions; until it has them, it does not quote.
    """
    kind = "avellaneda_stoikov"
    keeps_queue = True

    def __init__(self, gamma: float = 0.01, k: float = 1.5, horizon: float = 10.0, window: int = 20,
                 size: int = 5, limit: int = 50, interval: float = 0.5):
        if not (gamma > 0 and k > 0 and horizon > 0):
            raise ValueError("risk aversion, order arrival decay and horizon must be above 0")
        if window < 2:
            raise ValueError("the volatility needs a window of at least 2 decisions")
        if limit < size:
            raise ValueError("the inventory limit is below the quote size")
        self.gamma, self.k, self.horizon, self.window = gamma, k, horizon, window
        self.size, self.limit, self.interval = size, limit, interval
        self._last: Optional[float] = None
        self._changes: deque[float] = deque(maxlen=window)
        self.variance: Optional[float] = None  # per unit of time, as last estimated

    def decide(self, view: View) -> Plan:
        mid = view.mid
        if mid is None:
            return Plan(quotes={})
        if self._last is not None:
            self._changes.append(mid - self._last)
        self._last = mid
        if len(self._changes) < self.window:
            return Plan(quotes={})
        variance = sum(c * c for c in self._changes) / len(self._changes) / self.interval
        self.variance = variance
        skew = -view.position * self.gamma * variance * self.horizon
        spread = self.gamma * variance * self.horizon + 2 / self.gamma * math.log1p(self.gamma / self.k)
        reservation = mid + skew
        quotes = _quotes(view, math.floor(reservation - spread / 2), math.ceil(reservation + spread / 2),
                         self.size, self.limit)
        return Plan(quotes=quotes, signal=skew)

    def warming_up(self) -> Optional[tuple[int, int]]:
        return (len(self._changes), self.window) if len(self._changes) < self.window else None

    def describe(self) -> str:
        return (f"quotes {self.size} lots a side around a reservation price that moves against its inventory "
                f"(risk aversion {self.gamma:g}, order arrival decay {self.k:g} a tick, horizon {self.horizon:g}), "
                f"volatility from its last {self.window} steps, at most {self.limit} lots of inventory, "
                f"{self._every()}")

    def state(self) -> dict:
        return {"last_mid": self._last, "changes": list(self._changes), "variance": self.variance}

    def restore(self, state: dict) -> None:
        last, variance = state.get("last_mid"), state.get("variance")
        changes = _finite_list(state.get("changes"), self.window, "an Avellaneda-Stoikov strategy's mid changes")
        for value, what in ((last, "last mid"), (variance, "variance")):
            if value is not None and not (isinstance(value, (int, float)) and not isinstance(value, bool)
                                          and math.isfinite(value) and value >= 0):
                raise ValueError(f"an Avellaneda-Stoikov strategy's {what} must be a number from 0")
        if last is None and changes:
            raise ValueError("an Avellaneda-Stoikov strategy with mid changes must have a last mid")
        self._last, self.variance = last, variance
        self._changes = deque(changes, maxlen=self.window)


class PennyJumping(Strategy):
    """Rest one tick better than the best bid and the best ask when the spread is ``min_spread`` or more.

    Otherwise it joins them. Holding a position, it stops stepping ahead on the side that would add
    to it (it joins there instead), and at its ``limit`` quotes only the side that reduces it. It
    keeps an order while it is at the price wanted, so it keeps its place in the queue.
    """
    kind = "penny_jumping"
    keeps_queue = True

    def __init__(self, min_spread: int = 3, size: int = 5, limit: int = 50, interval: float = 0.5):
        if min_spread < 2:
            raise ValueError("stepping ahead on both sides needs a spread of at least 2 ticks")
        if limit < size:
            raise ValueError("the inventory limit is below the quote size")
        self.min_spread, self.size, self.limit, self.interval = min_spread, size, limit, interval

    def decide(self, view: View) -> Plan:
        if view.mid is None:
            return Plan(quotes={})
        jump = view.best_ask - view.best_bid >= self.min_spread
        bid = view.best_bid + (1 if jump and view.position <= 0 else 0)
        ask = view.best_ask - (1 if jump and view.position >= 0 else 0)
        return Plan(quotes=_quotes(view, bid, ask, self.size, self.limit))

    def describe(self) -> str:
        return (f"rests {self.size} lots a side one tick ahead of the best quotes when the spread is "
                f"{self.min_spread} ticks or more, at them otherwise, keeping its place in the queue; at most "
                f"{self.limit} lots of inventory, {self._every()}")


class QuoteJoiner(Strategy):
    """Rest ``size`` lots at the best bid and the best ask, and nothing else: the null for making markets.

    It keeps its orders while they are at the best quotes (keeping its place in the queue), and at
    its ``limit`` quotes only the side that reduces its inventory. Where providing liquidity just
    breaks even (Wyart et al. 2008: no strategy more than marginally profitable), it makes
    nothing before fees, so what it does make, the order flow hands it.
    """
    kind = "quote_joiner"
    keeps_queue = True

    def __init__(self, size: int = 5, limit: int = 50, interval: float = 0.5):
        if limit < size:
            raise ValueError("the inventory limit is below the quote size")
        self.size, self.limit, self.interval = size, limit, interval

    def decide(self, view: View) -> Plan:
        if view.mid is None:
            return Plan(quotes={})
        return Plan(quotes=_quotes(view, view.best_bid, view.best_ask, self.size, self.limit))

    def describe(self) -> str:
        return (f"rests {self.size} lots a side at the best bid and ask and nothing else, keeping its place in the "
                f"queue; at most {self.limit} lots of inventory, {self._every()}")


class _Positional(Strategy):
    """A strategy that holds -size, 0 or +size lots."""

    def __init__(self, size: int, interval: float, execution: str, protection: int):
        if execution not in EXECUTIONS:
            raise ValueError(f"execution must be one of {EXECUTIONS}")
        self.size, self.interval, self.execution, self.protection = size, interval, execution, protection
        self.target = 0

    def state(self) -> dict:
        return {"target": self.target}

    def restore(self, state: dict) -> None:
        target = state.get("target")
        if target not in (-self.size, 0, self.size) or isinstance(target, bool):
            raise ValueError(f"a {self.kind} strategy of {self.size} lots cannot want {target!r}")
        self.target = target

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

    def state(self) -> dict:
        return {**super().state(), "fast_average": self._fast, "slow_average": self._slow, "seen": self._seen}

    def restore(self, state: dict) -> None:
        super().restore(state)
        fast, slow, seen = state.get("fast_average"), state.get("slow_average"), state.get("seen")
        if not (isinstance(seen, int) and not isinstance(seen, bool) and seen >= 0):
            raise ValueError("a momentum strategy's count of decisions must be a whole number from 0")
        if (fast is None) != (slow is None) or (fast is None) != (seen == 0):
            raise ValueError("a momentum strategy has both averages once it has decided, and neither before")
        if fast is not None and not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                                        for v in (fast, slow)):
            raise ValueError("a momentum strategy's averages must be numbers")
        self._fast, self._slow, self._seen = fast, slow, seen

    def describe(self) -> str:
        return (f"long or short {self.size} lots while its {self.fast}- and {self.slow}-step averages of the mid "
                f"are more than {self.threshold:g} ticks apart, flat once they cross; {self._how()}, {self._every()}")


class MeanReversion(_Positional):
    """Short when the mid is more than ``entry`` standard deviations above its average over ``window`` decisions.

    Long when as far below; flat once it is back within ``exit`` of the average, or past it (an
    exit of 0: at the average). Flat until the window is full.
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
        elif (self.target < 0 and z < self.exit) or (self.target > 0 and z > -self.exit):
            self.target = 0  # back within exit of the average, or past it (an exit of 0: at the average)
        return Plan(target=self.target, signal=z)

    def warming_up(self) -> Optional[tuple[int, int]]:
        return (len(self._mids), self.window) if len(self._mids) < self.window else None

    def state(self) -> dict:
        return {**super().state(), "mids": list(self._mids)}

    def restore(self, state: dict) -> None:
        super().restore(state)
        mids = state.get("mids")
        if not (isinstance(mids, list) and len(mids) <= self.window and all(
                isinstance(m, (int, float)) and not isinstance(m, bool) and math.isfinite(m) for m in mids)):
            raise ValueError(f"a mean-reversion strategy keeps a list of at most {self.window} mid-prices")
        self._mids = deque(float(m) for m in mids)
        self._sum = sum(self._mids)
        self._sum_sq = sum(m * m for m in self._mids)

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


class Breakout(_Positional):
    """Long when the mid rises above the highest mid of its last ``window`` decisions; short below the lowest.

    It goes flat when the mid comes back past the middle of that range (a Donchian channel), and
    holds in between. Flat until it has seen ``window`` decisions.
    """
    kind = "breakout"

    def __init__(self, window: int = 20, size: int = 5, interval: float = 0.5, execution: str = "aggressive",
                 protection: int = 5):
        super().__init__(size, interval, execution, protection)
        if window < 2:
            raise ValueError("the channel needs a window of at least 2 decisions")
        self.window = window
        self._mids: deque[float] = deque(maxlen=window)

    def decide(self, view: View) -> Plan:
        mid = view.mid
        if mid is None:
            return Plan()
        mids = self._mids
        if len(mids) < self.window:
            mids.append(mid)
            return Plan(target=0)
        high, low = max(mids), min(mids)
        middle, half = (high + low) / 2, (high - low) / 2
        if mid > high:
            self.target = self.size
        elif mid < low:
            self.target = -self.size
        elif (self.target > 0 and mid < middle) or (self.target < 0 and mid > middle):
            self.target = 0
        mids.append(mid)
        return Plan(target=self.target, signal=(mid - middle) / half if half else 0.0)

    def warming_up(self) -> Optional[tuple[int, int]]:
        return (len(self._mids), self.window) if len(self._mids) < self.window else None

    def describe(self) -> str:
        return (f"long (short) {self.size} lots when the mid breaks above (below) its {self.window}-step range, "
                f"flat back past its middle; {self._how()}, {self._every()}")

    def state(self) -> dict:
        return {**super().state(), "mids": list(self._mids)}

    def restore(self, state: dict) -> None:
        super().restore(state)
        self._mids = deque(_finite_list(state.get("mids"), self.window, "a breakout strategy's mids"),
                           maxlen=self.window)


class OrderFlowImbalance(_Positional):
    """Long when the order flow imbalance at the best quotes over its last ``window`` decisions is high.

    The imbalance between two decisions, after Cont, Kukanov and Stoikov (2014): lots joining the
    best bid, or the bid moving up, count as buying pressure; lots leaving it, or it moving down, as
    selling pressure; and the reverse at the ask. Their sum over the window is divided by the
    average lots at the best quotes. Above ``threshold`` it goes long, below minus the threshold
    short, and flat when the sum tips past zero. Flat until it has seen ``window`` changes.
    """
    kind = "order_flow_imbalance"

    def __init__(self, window: int = 20, threshold: float = 2.0, size: int = 5, interval: float = 0.25,
                 execution: str = "aggressive", protection: int = 5):
        super().__init__(size, interval, execution, protection)
        if window < 2:
            raise ValueError("the flow needs a window of at least 2 decisions")
        if not threshold >= 0:
            raise ValueError("the threshold must be 0 or more")
        self.window, self.threshold = window, threshold
        self._previous: Optional[list] = None  # [best bid, its lots, best ask, its lots] at the last decision
        self._flows: deque[float] = deque(maxlen=window)
        self._depths: deque[float] = deque(maxlen=window)

    def decide(self, view: View) -> Plan:
        if view.mid is None or view.bid_depth + view.ask_depth == 0:
            return Plan()
        now = [view.best_bid, view.bid_depth, view.best_ask, view.ask_depth]
        if self._previous is not None:
            bid, bid_lots, ask, ask_lots = self._previous
            flow = (view.bid_depth if view.best_bid >= bid else 0) - (bid_lots if view.best_bid <= bid else 0) \
                - (view.ask_depth if view.best_ask <= ask else 0) + (ask_lots if view.best_ask >= ask else 0)
            self._flows.append(flow)
            self._depths.append((view.bid_depth + view.ask_depth) / 2)
        self._previous = now
        if len(self._flows) < self.window:
            return Plan(target=0)
        pressure = sum(self._flows) / (sum(self._depths) / len(self._depths))
        if pressure > self.threshold:
            self.target = self.size
        elif pressure < -self.threshold:
            self.target = -self.size
        elif (self.target > 0 and pressure < 0) or (self.target < 0 and pressure > 0):
            self.target = 0
        return Plan(target=self.target, signal=pressure)

    def warming_up(self) -> Optional[tuple[int, int]]:
        return (len(self._flows), self.window) if len(self._flows) < self.window else None

    def describe(self) -> str:
        return (f"long (short) {self.size} lots while the order flow imbalance at the best quotes over its last "
                f"{self.window} steps is more than {self.threshold:g} average queues to the buy (sell) side, flat "
                f"when it tips back; {self._how()}, {self._every()}")

    def state(self) -> dict:
        return {**super().state(), "previous": self._previous, "flows": list(self._flows),
                "depths": list(self._depths)}

    def restore(self, state: dict) -> None:
        super().restore(state)
        previous = state.get("previous")
        if previous is not None:
            previous = _finite_list(previous, 4, "an order flow strategy's last quotes")
            if len(previous) != 4 or not all(v.is_integer() and v >= 0 for v in previous):
                raise ValueError("an order flow strategy's last quotes are a bid, its lots, an ask and its lots")
            previous = [int(v) for v in previous]
        flows = _finite_list(state.get("flows"), self.window, "an order flow strategy's flows")
        depths = _finite_list(state.get("depths"), self.window, "an order flow strategy's depths")
        if len(flows) != len(depths) or (flows and previous is None) or any(d <= 0 for d in depths):
            raise ValueError("an order flow strategy's flows and depths do not go together")
        self._previous = previous
        self._flows, self._depths = deque(flows, maxlen=self.window), deque(depths, maxlen=self.window)


_MASK = 2**64 - 1


def _splitmix(seed: int) -> int:
    """A well-mixed, nonzero 64-bit state from any seed."""
    z = (seed + 0x9E3779B97F4A7C15) & _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return (z ^ (z >> 31)) or 1


class RandomTrader(_Positional):
    """At each decision, with probability ``switch``, a new position drawn at random: long, flat or short ``size``.

    It knows nothing, so what it makes or loses is the cost of trading without information: a
    benchmark for the others. Its own random numbers (xorshift64*, from ``seed``) make it repeat
    exactly, in a replay too.
    """
    kind = "random"

    def __init__(self, switch: float = 0.1, seed: int = 1, size: int = 5, interval: float = 0.5,
                 execution: str = "aggressive", protection: int = 5):
        super().__init__(size, interval, execution, protection)
        if not 0 <= switch <= 1:
            raise ValueError("the chance of a new position must be from 0 to 1")
        self.switch, self.seed = switch, seed
        self._rng = _splitmix(seed)

    def _uniform(self) -> float:
        x = self._rng
        x ^= x >> 12
        x ^= (x << 25) & _MASK
        x ^= x >> 27
        self._rng = x
        return (((x * 0x2545F4914F6CDD1D) & _MASK) >> 11) / 2**53

    def decide(self, view: View) -> Plan:
        if view.mid is None:
            return Plan()
        if self._uniform() < self.switch:
            self.target = (-self.size, 0, self.size)[min(2, int(self._uniform() * 3))]
        return Plan(target=self.target)

    def describe(self) -> str:
        return (f"a new random position (long, flat or short {self.size} lots) with a chance of {self.switch:g} at "
                f"each step, from seed {self.seed}; {self._how()}, {self._every()}")

    def state(self) -> dict:
        return {**super().state(), "rng": self._rng}

    def restore(self, state: dict) -> None:
        super().restore(state)
        rng = state.get("rng")
        if isinstance(rng, bool) or not isinstance(rng, int) or not 0 < rng <= _MASK:
            raise ValueError("a random trader's state must be a whole number from 1 to 2**64 - 1")
        self._rng = rng
