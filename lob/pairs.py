"""Two order books whose prices move together, for pairs trading.

Each book's traders place their orders around the book's fair value (``AnchoredZIModel``): limit
orders a few ticks either side of it, market orders and cancellations, as in the zero-intelligence
model but anchored to a value rather than to the opposite best quote. A new limit order that
crosses a stale quote trades at once, so a book reprices as its fair value moves.

The fair values (``FairValues``) share a common factor X, a random walk, and each has a part of
its own, U, which order flow moves (price impact: lots bought aggressively push it up, lots sold
down) and which, given a half-life, decays back to zero::

    F_A = X + U_A
    F_B = level_B + beta * (X - X_0) + U_B
    dU  = -kappa * U dt + sigma_u dW + impact * dQ      (kappa = ln 2 / half-life; dQ: net lots taken)

The gap, ``G = F_B - level_B - beta * (F_A - X_0) = U_B - beta * U_A``, then mean-reverts with that
half-life, with a standard deviation of ``sigma_u * sqrt((1 + beta^2) / (2 kappa))`` from the own
parts' noise alone (order flow adds some). Without a half-life the own parts wander and impact is
permanent, so the gap wanders too; an ``Arbitrageur``, which buys the cheap book and sells the dear
one when the books' mids are far apart, then brings it back through its own impact.

``PairMarket`` runs both books on one clock, with the arbitrageur if there is one, and takes orders
from outside with the same impact as anyone's: a ``PairsTrader``'s, which trades the gap's z-score.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from lob.book import OrderBook
from lob.matching import PriceTimeEngine
from lob.order import Fill, Order, OrderType, Side, cancel_order, limit_order, market_order

A, B = 0, 1
BOOKS = ("A", "B")


class FairValues:
    """The two books' fair values: a common random walk and a part of each book's own (see the module docstring).

    ``half_life`` (time units) makes the own parts, and so the gap, mean-revert; None leaves them
    wandering. ``volatility`` is the common factor's, ``own_volatility`` each own part's (ticks per
    square root of a time unit), ``impact`` the ticks a lot taken moves its book's own part.
    """

    def __init__(self, level_a: float = 100.0, level_b: float = 100.0, beta: float = 1.0, volatility: float = 0.3,
                 own_volatility: float = 0.8, half_life: Optional[float] = 50.0, impact: float = 0.05):
        if half_life is not None and not half_life > 0:
            raise ValueError("the half-life must be above 0, or None for none")
        if not (volatility >= 0 and own_volatility >= 0 and impact >= 0 and beta > 0):
            raise ValueError("volatilities and impact must be 0 or more, and beta above 0")
        self.level_a, self.level_b, self.beta = level_a, level_b, beta
        self.volatility, self.own_volatility = volatility, own_volatility
        self.half_life, self.impact = half_life, impact
        self.kappa = math.log(2) / half_life if half_life is not None else 0.0
        self.x = level_a  # the common factor, starting at A's level
        self.own = [0.0, 0.0]

    @property
    def gap_sd(self) -> Optional[float]:
        """The gap's standard deviation from the own parts' noise alone, when it mean-reverts."""
        if not self.kappa:
            return None
        return self.own_volatility * math.sqrt((1 + self.beta ** 2) / (2 * self.kappa))

    def advance(self, dt: float, rng: np.random.Generator) -> None:
        """Move on by ``dt``, exactly (the own parts as Ornstein-Uhlenbeck processes, or random walks)."""
        if dt <= 0:
            return
        self.x += self.volatility * math.sqrt(dt) * rng.standard_normal()
        if self.kappa:
            decay = math.exp(-self.kappa * dt)
            sd = self.own_volatility * math.sqrt((1 - decay * decay) / (2 * self.kappa))
        else:
            decay, sd = 1.0, self.own_volatility * math.sqrt(dt)
        for k in (A, B):
            self.own[k] = self.own[k] * decay + sd * rng.standard_normal()

    def trade(self, book: int, lots: float) -> None:
        """Lots taken in a book (bought positive, sold negative) push its own part by the impact."""
        self.own[book] += self.impact * lots

    def fair(self, book: int) -> float:
        if book == A:
            return self.x + self.own[A]
        return self.level_b + self.beta * (self.x - self.level_a) + self.own[B]

    def gap(self) -> float:
        """The fair values' gap: B's own part less beta times A's."""
        return self.own[B] - self.beta * self.own[A]

    def state(self) -> dict:
        return {"x": self.x, "own": list(self.own)}


class AnchoredZIModel:
    """Zero-intelligence order flow around a fair value.

    Limit orders arrive at ``limit_rate`` a unit of time, half buys and half sells, at the fair
    value less (buys) or plus (sells) a whole number of ticks from 1 to ``reach``; one priced
    across the opposite best quote trades on arrival, which is how the book follows its fair
    value. Market orders arrive at ``market_rate``, half buys and half sells; each resting order
    is canceled at ``cancel_rate`` a unit of time. Sizes are 1 to ``max_size`` lots.
    """

    def __init__(self, limit_rate: float = 10.0, market_rate: float = 1.5, cancel_rate: float = 0.05,
                 reach: int = 8, max_size: int = 5):
        if not (limit_rate > 0 and market_rate >= 0 and cancel_rate >= 0 and reach >= 1 and max_size >= 1):
            raise ValueError("rates must be 0 or more (limit orders above 0), reach and size at least 1")
        self.limit_rate, self.market_rate, self.cancel_rate = limit_rate, market_rate, cancel_rate
        self.reach, self.max_size = reach, max_size

    def rate(self, book: OrderBook) -> float:
        return self.limit_rate + self.market_rate + self.cancel_rate * book.num_orders()

    def event(self, book: OrderBook, t: float, fair: float, rng: np.random.Generator) -> Optional[Order]:
        u = rng.random() * self.rate(book)
        size = int(rng.integers(1, self.max_size + 1))
        buy = rng.random() < 0.5
        if u < self.limit_rate:
            distance = int(rng.integers(1, self.reach + 1))
            price = max(1, round(fair - distance) if buy else round(fair + distance))
            return limit_order(price, size, t, Side.BID if buy else Side.ASK)
        if u < self.limit_rate + self.market_rate:
            return market_order(size, t, Side.BID if buy else Side.ASK)
        oid = book.random_order_id(rng)
        return cancel_order(oid, t) if oid is not None else None


@dataclass
class Arbitrageur:
    """A trader who keeps the books together: when B's mid is more than ``threshold`` ticks dear against A's
    (after the levels and beta), it sells ``size`` lots of B and buys ``beta * size`` of A with market orders,
    and the reverse when B is cheap. It looks ``rate`` times a unit of time, has no position limit, and its
    trades move the fair values as anyone's do.
    """
    threshold: float = 2.0
    size: int = 5
    rate: float = 1.0
    position: list[int] = field(default_factory=lambda: [0, 0])
    cash: float = 0.0
    trades: int = 0


@dataclass
class PairsTrader:
    """Trades the gap between the books: short it when it is dear, long when cheap, flat once it is back.

    Every ``interval`` of time it takes the gap it sees, ``mid_B - hedge * mid_A``, and its z-score
    against the last ``window`` of them (the gap's level is whatever it is: the rolling mean finds
    it). Above ``entry`` it holds -``size`` lots of B and +``hedge * size`` of A (short the gap);
    below minus ``entry`` the reverse; once back within ``exit`` of zero, or past it, flat (an exit of 0:
    at the mean); in between, what it holds. It
    trades the difference with market orders in both books, which move them as anyone's do.
    Flat until it has ``window`` gaps.
    """
    window: int = 400
    entry: float = 2.0
    exit: float = 0.0
    size: int = 5
    hedge: float = 1.0
    interval: float = 1.0
    position: list[int] = field(default_factory=lambda: [0, 0])  # lots of A and of B
    cash: float = 0.0
    lots: int = 0  # bought and sold, both books
    spread_paid: float = 0.0  # lots times how far from the mid each fill was: what crossing cost
    notional: float = 0.0  # price times lots of every fill
    entries: int = 0  # times it went from flat (or the other way) into a position
    next_time: float = 0.0
    z: Optional[float] = None
    target: int = 0  # the lots of B it wants (A's follow by the hedge)
    # With ``record``, one row per decision: (time, A's mid and B's just before its orders, what it then holds of
    # A and of B, z, target), for a record of what it held between decisions that adds up to its fills.
    record: bool = False
    history: list = field(default_factory=list)

    def __post_init__(self):
        if self.window < 3 or not 0 <= self.exit < self.entry or self.size < 1 or self.hedge <= 0:
            raise ValueError("a window of 3 or more, 0 <= exit < entry, a size of 1 or more and a hedge above 0")
        self._gaps: deque[float] = deque(maxlen=self.window)

    def decide(self, mid_a: Optional[float], mid_b: Optional[float]) -> Optional[int]:
        """The lots of B it wants now (None: no gap to read); A's are -hedge times as many."""
        if mid_a is None or mid_b is None:
            return None
        gaps = self._gaps
        gaps.append(mid_b - self.hedge * mid_a)
        if len(gaps) < self.window:
            self.z = None
            return 0
        mean = sum(gaps) / len(gaps)
        sd = math.sqrt(sum((g - mean) ** 2 for g in gaps) / len(gaps))
        self.z = (gaps[-1] - mean) / sd if sd > 0 else 0.0
        if self.z > self.entry:
            want = -self.size
        elif self.z < -self.entry:
            want = self.size
        elif (self.target < 0 and self.z < self.exit) or (self.target > 0 and self.z > -self.exit):
            want = 0  # back within exit of the mean, or past it
        else:
            want = self.target
        if want and want != self.target:
            self.entries += 1
        self.target = want
        return want

    def value(self, mid_a: Optional[float], mid_b: Optional[float]) -> Optional[float]:
        """Its P&L: cash plus what it holds at the mids."""
        if mid_a is None or mid_b is None:
            return None
        return self.cash + self.position[A] * mid_a + self.position[B] * mid_b


class PairMarket:
    """Two books (A and B) on one clock, each with its traders around its fair value, and an arbitrageur if given.

    ``step`` runs the next event, whichever book or the arbitrageur it falls to; ``submit`` takes an
    order from outside (a pairs strategy's) into a book now. Orders that take liquidity move their
    book's fair value by the impact, except the anchored traders' own limit orders that cross a
    stale quote, which are the book repricing.
    """

    def __init__(self, fair: Optional[FairValues] = None,
                 models: Optional[tuple[AnchoredZIModel, AnchoredZIModel]] = None,
                 arbitrageur: Optional[Arbitrageur] = None, seed: int = 1, depth: int = 8, lots: int = 10,
                 traders: tuple[PairsTrader, ...] = ()):
        self.fair = fair or FairValues()
        self.models = models or (AnchoredZIModel(), AnchoredZIModel())
        self.arbitrageur = arbitrageur
        self.traders = list(traders)
        self.rng = np.random.default_rng(seed)
        self.engine = PriceTimeEngine()
        self.books = (OrderBook(), OrderBook())
        self.time = 0.0
        self.events = 0
        self._last_mid = [self.fair.fair(A), self.fair.fair(B)]  # for valuing positions while a side is empty
        for k, book in enumerate(self.books):  # a starting book around each fair value
            f = self.fair.fair(k)
            for d in range(1, depth + 1):
                book.add_order(limit_order(max(1, round(f - d)), lots, 0.0, Side.BID))
                book.add_order(limit_order(round(f + d), lots, 0.0, Side.ASK))

    def mid(self, book: int) -> Optional[float]:
        return self.books[book].mid_price

    def mark(self, book: int) -> float:
        """What a position in a book is worth a lot: its mid, or the last one it had while a side is empty."""
        mid = self.books[book].mid_price
        if mid is not None:
            self._last_mid[book] = mid
        return self._last_mid[book]

    def gap(self) -> Optional[float]:
        """The books' gap as a trader sees it: B's mid less its level and beta times A's move."""
        a, b = self.mid(A), self.mid(B)
        if a is None or b is None:
            return None
        f = self.fair
        return b - f.level_b - f.beta * (a - f.level_a)

    def submit(self, book: int, order: Order, impact: bool = True) -> list[Fill]:
        """An order into a book now; what it takes moves the book's fair value (unless ``impact`` is off)."""
        fills = self.engine.process(self.books[book], order)
        self.events += 1
        self.mark(book)  # keeps the last mid
        if impact and fills and order.order_type != OrderType.CANCEL:
            taken = sum(f.size for f in fills)
            self.fair.trade(book, taken if order.side == Side.BID else -taken)
        return fills

    def step(self) -> tuple[int, Optional[Order], list[Fill]]:
        """The next event: (which book, the order or None, its fills); the arbitrageur's acts on both books."""
        rates = [self.models[k].rate(self.books[k]) for k in (A, B)]
        arb = self.arbitrageur.rate if self.arbitrageur else 0.0
        total = rates[A] + rates[B] + arb
        dt = self.rng.exponential(1 / total)
        self.fair.advance(dt, self.rng)
        self.time += dt
        u = self.rng.random() * total
        if u >= rates[A] + rates[B]:
            self._arbitrage()
            out: tuple[int, Optional[Order], list[Fill]] = (-1, None, [])
        else:
            k = A if u < rates[A] else B
            order = self.models[k].event(self.books[k], self.time, self.fair.fair(k), self.rng)
            out = (k, order, self.submit(k, order, impact=order.order_type == OrderType.MARKET) if order else [])
        for trader in self.traders:
            if self.time >= trader.next_time:
                self.trade_for(trader)
        return out

    def trade_for(self, trader: PairsTrader) -> None:
        """Ask a pairs trader what it wants, and trade the difference in both books with market orders."""
        trader.next_time = self.time + trader.interval
        before = (self.mark(A), self.mark(B))
        want_b = trader.decide(self.mid(A), self.mid(B))
        if want_b is None:
            return
        want_a = -round(trader.hedge * want_b)
        for book, want in ((B, want_b), (A, want_a)):
            need = want - trader.position[book]
            if not need:
                continue
            mid = self.mid(book)
            side = Side.BID if need > 0 else Side.ASK
            fills = self.submit(book, market_order(abs(need), self.time, side))
            sign = 1 if need > 0 else -1
            for f in fills:
                trader.position[book] += sign * f.size
                trader.cash -= sign * f.price * f.size
                trader.lots += f.size
                trader.notional += f.price * f.size
                if mid is not None:
                    trader.spread_paid += abs(f.price - mid) * f.size
        if trader.record:
            trader.history.append((self.time, *before, trader.position[A], trader.position[B], trader.z,
                                   trader.target))

    def _arbitrage(self) -> None:
        arb, gap = self.arbitrageur, self.gap()
        if gap is None or abs(gap) <= arb.threshold:
            return
        sell_b = gap > 0  # B dear: sell B, buy A
        lots_a = max(1, round(self.fair.beta * arb.size))
        for book, side, lots in ((B, Side.ASK if sell_b else Side.BID, arb.size),
                                 (A, Side.BID if sell_b else Side.ASK, lots_a)):
            fills = self.submit(book, market_order(lots, self.time, side))
            sign = 1 if side == Side.BID else -1
            arb.position[book] += sign * sum(f.size for f in fills)
            arb.cash -= sign * sum(f.price * f.size for f in fills)
        arb.trades += 1

    def run(self, until: float, every: float = 1.0) -> pd.DataFrame:
        """Step to time ``until``, sampling both books, the fair values and the traders every ``every`` units of time.

        Each trader ``i`` adds what it holds of each book (``pos_a_i``, ``pos_b_i``), its P&L at the mids
        (``value_i``), its z-score and the lots of B it wants (``z_i``, ``target_i``).
        """
        rows = []
        next_sample = self.time + every
        while self.time < until:
            self.step()
            while self.time >= next_sample:
                a, b = self.books
                row = {"time": next_sample, "mid_a": a.mid_price, "mid_b": b.mid_price,
                       "fair_a": self.fair.fair(A), "fair_b": self.fair.fair(B), "gap": self.gap(),
                       "true_gap": self.fair.gap(), "spread_a": a.spread, "spread_b": b.spread}
                for i, trader in enumerate(self.traders):
                    row |= {f"pos_a_{i}": trader.position[A], f"pos_b_{i}": trader.position[B],
                            f"value_{i}": trader.value(self.mark(A), self.mark(B)), f"z_{i}": trader.z,
                            f"target_{i}": trader.target}
                rows.append(row)
                next_sample += every
        return pd.DataFrame(rows)
