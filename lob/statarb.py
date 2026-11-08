"""Many order books on one clock, whose fair values share several common factors: stat-arb and many pairs.

``lob/pairs.py`` has two books and one relation between them. Here there are ``n`` books and ``k`` common factors::

    F_i = level_i + sum_j B_ij X_j + U_i
    dX_j = sigma_j dW_j,        dU_i = -kappa U_i dt + sigma_u dW_i + impact dQ_i

The factors are random walks and the own parts, given a half-life, mean-revert, as in ``lob/pairs.py``. Any portfolio
``w`` with ``w'B = 0`` leaves the factors out, so ``w'F`` mean-reverts: the books have ``n - rank(B)`` independent
relations, the space orthogonal to the loadings. So a pair is truly cointegrated only if its two rows of ``B`` are
parallel. With ``B`` of a market factor and sector factors, the pairs within a sector are. With rows in no such order
no pair is, and each relation needs more than two books.

What is outside the traders' reach, the factors and the shocks to the own parts, comes from a random stream of its
own, on a fixed grid of time (``step``). So two markets of one seed have the same factor paths and the same shocks
whatever trades in them, and comparing traders seed by seed takes out what they share. In ``lob/pairs.py`` a trader's
first order changes what every later random number is used for, and the two markets part for good. Order flow still
differs after a trade: it has its own stream, and how many draws it takes depends on the books.

Each book's traders place orders around its fair value (``AnchoredZIModel``), and a fair value stops at ``floor``
ticks, by pushing the first factor up just as far as needed, on which every book must load positively. That moves no
relation, since ``w'B = 0``.

Traders hold lots in every book and trade the difference with market orders. ``RelationTrader`` trades the z-score of
each of a set of portfolios, chosen by a function of past mids: a Johansen vector (``johansen_relations``), a small
basket (``best_baskets``), or a pair (``pairs_by_distance``, ``pairs_by_pvalue``). ``ResidualTrader`` trades each
book's residual against PCA factors (Avellaneda and Lee 2010). ``CopulaPortfolio`` runs ``lob.pairs.CopulaTrader`` on
each of a set of pairs (Rad, Low and Faff 2016). All net their orders across relations, unless told not to.
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from lob.book import OrderBook
from lob.copulas import FAMILIES
from lob.matching import PriceTimeEngine
from lob.order import Fill, Order, OrderType, Side, limit_order, market_order
from lob.pairs import BETA_FLOOR, FLOOR, AnchoredZIModel, CopulaTrader

BLOCK = 256  # grid steps of random numbers drawn at a time
WAIT = 0.03  # time units a decision waits for the next event, once its interval is up: above the mean in two books


class FactorValues:
    """The books' fair values: common factors through ``loadings`` (books by factors) and a part of each book's own.

    ``volatility`` is each factor's and ``own_volatility`` each own part's (ticks per square root of a time unit),
    ``half_life`` makes the own parts mean-revert (None leaves them wandering), ``impact`` is the ticks a lot taken
    moves its book's own part. All move on a grid of ``step`` units of time, from ``reseed``'s stream alone.

    ``loading_sd`` makes each loading an Ornstein-Uhlenbeck process around its start, with that standard deviation
    and ``loading_half_life``, so that no fixed weights stay right. ``pushed`` is what holding the fair values above
    ``floor`` has added to the first factor so far.
    """

    def __init__(self, loadings, level=3000.0, volatility=1.0, own_volatility: float = 0.8,
                 half_life: Optional[float] = 50.0, impact: float = 0.05, floor: float = FLOOR, step: float = 0.2,
                 loading_sd: float = 0.0, loading_half_life: float = 2000.0):
        b = np.array(loadings, dtype=float)
        if b.ndim != 2 or not (b.shape[0] >= 2 and b.shape[1] >= 1):
            raise ValueError("loadings are a table of two or more books by one or more factors")
        if not (b[:, 0] > 0).all():
            raise ValueError("every book must load positively on the first factor: the floor pushes it")
        if half_life is not None and not half_life > 0:
            raise ValueError("the half-life must be above 0, or None for none")
        if not (own_volatility >= 0 and impact >= 0 and step > 0 and loading_sd >= 0 and loading_half_life > 0):
            raise ValueError("volatilities and impact must be 0 or more, the step and the loadings' half-life above 0")
        self.n, self.k = b.shape
        self.mean_loadings, self.loadings = b.copy(), b
        self.level = np.broadcast_to(np.asarray(level, dtype=float), self.n).copy()
        self.volatility = np.broadcast_to(np.asarray(volatility, dtype=float), self.k).copy()
        if (self.volatility < 0).any():
            raise ValueError("volatilities must be 0 or more")
        if not (floor >= 0 and (self.level >= floor).all()):
            raise ValueError("the floor must be 0 or more, and every level at or above it")
        self.own_volatility, self.half_life, self.impact = own_volatility, half_life, impact
        self.floor, self.step = floor, step
        self.loading_sd, self.loading_half_life = loading_sd, loading_half_life
        self._factor_sd = self.volatility * math.sqrt(step)
        kappa = math.log(2) / half_life if half_life is not None else 0.0
        self.kappa = kappa
        self._decay = math.exp(-kappa * step)
        self._own_sd = own_volatility * (math.sqrt((1 - self._decay ** 2) / (2 * kappa)) if kappa else math.sqrt(step))
        self._loading_decay = math.exp(-math.log(2) / loading_half_life * step)
        self._loading_shock = loading_sd * math.sqrt(1 - self._loading_decay ** 2)
        self._draws = self.k + self.n + (self.n * self.k if loading_sd else 0)
        self.factors = np.zeros(self.k)
        self.common = np.zeros(self.n)
        self.own = np.zeros(self.n)
        self.pushed = 0.0
        self.fairs: list[float] = self.level.tolist()  # the fair values now, for the order flow
        self._steps = 0
        self.reseed(1)

    def reseed(self, seed: int) -> None:
        """Set the stream the factors and the own parts' shocks come from: only this and the time decide them."""
        self._rng = np.random.default_rng([seed, 1])
        self._z = self._rng.standard_normal((BLOCK, self._draws))
        self._at = 0

    @property
    def rank(self) -> int:
        """How many independent relations the books have: the books less the factors they tell apart."""
        return self.n - int(np.linalg.matrix_rank(self.mean_loadings))

    def relations(self) -> np.ndarray:
        """A basis of the true relations (one a row): the space orthogonal to the loadings."""
        _, _, vt = np.linalg.svd(self.mean_loadings.T)
        return vt[self.n - self.rank:]

    def advance_to(self, t: float) -> None:
        """Move on to time ``t``: every step of the grid up to it."""
        while (self._steps + 1) * self.step <= t:
            self._steps += 1
            self._move()

    def _move(self) -> None:
        if self._at == BLOCK:
            self._z = self._rng.standard_normal((BLOCK, self._draws))
            self._at = 0
        z = self._z[self._at]
        self._at += 1
        k, n = self.k, self.n
        moves = self._factor_sd * z[:k]
        self.factors += moves
        self.common += self.loadings @ moves
        self.own = self.own * self._decay + self._own_sd * z[k:k + n]
        if self.loading_sd:
            self.loadings = (self.mean_loadings + (self.loadings - self.mean_loadings) * self._loading_decay
                             + self._loading_shock * z[k + n:].reshape(n, k))
            self.loadings[:, 0] = np.maximum(self.loadings[:, 0], BETA_FLOOR)
        values = self.level + self.common + self.own
        short = float(((self.floor - values) / self.loadings[:, 0]).max())
        if short > 0:
            self.factors[0] += short
            self.common += self.loadings[:, 0] * short
            values = values + self.loadings[:, 0] * short
            self.pushed += short
        self.fairs = values.tolist()

    def trade(self, book: int, lots: float) -> None:
        """Lots taken in a book (bought positive, sold negative) push its own part by the impact."""
        move = self.impact * lots
        self.own[book] += move
        self.fairs[book] += move

    def values(self) -> np.ndarray:
        return np.array(self.fairs)


class BookMarket:
    """``n`` books on one clock, each with its traders around its fair value, and traders from outside.

    Orders that take liquidity move their book's fair value by the impact, except the anchored traders' own limit
    orders that cross a stale quote, which are the book repricing.

    ``flow`` and ``volume`` hold, a book each, the lots of the models' own orders that traded on arrival: ``flow``
    signed (plus when the order was a buy) and ``volume`` not. A trader's orders are not counted.
    """

    def __init__(self, fair: FactorValues, models: Optional[list[AnchoredZIModel]] = None, seed: int = 1,
                 depth: int = 8, lots: int = 10, traders=()):
        self.fair = fair
        self.n = fair.n
        self.models = list(models) if models is not None else [AnchoredZIModel() for _ in range(self.n)]
        if len(self.models) != self.n:
            raise ValueError("one model a book")
        fair.reseed(seed)
        self.rng = np.random.default_rng([seed, 2])  # the order flow's, apart from the factors'
        self.engine = PriceTimeEngine()
        self.books = [OrderBook() for _ in range(self.n)]
        self.time = 0.0
        self.events = 0
        self.flow = np.zeros(self.n)
        self.volume = np.zeros(self.n)
        self._last_mid = list(fair.fairs)
        for k, book in enumerate(self.books):  # a starting book around each fair value
            f = fair.fairs[k]
            for d in range(1, depth + 1):
                book.add_order(limit_order(max(1, round(f - d)), lots, 0.0, Side.BID))
                book.add_order(limit_order(round(f + d), lots, 0.0, Side.ASK))
        self._rates = [model.rate(book) for model, book in zip(self.models, self.books, strict=True)]
        self.traders: list[BookTrader] = []
        for trader in traders:
            self.add(trader)

    def add(self, trader: BookTrader) -> None:
        trader.attach(self.n)
        self.traders.append(trader)

    def mid(self, book: int) -> Optional[float]:
        return self.books[book].mid_price

    def mark(self, book: int) -> float:
        """What a position in a book is worth a lot: its mid, or the last one it had while a side is empty."""
        mid = self.books[book].mid_price
        if mid is not None:
            self._last_mid[book] = mid
        return self._last_mid[book]

    def marks(self) -> np.ndarray:
        return np.array([self.mark(book) for book in range(self.n)])

    def submit(self, book: int, order: Order, impact: bool = True) -> list[Fill]:
        """An order into a book now; what it takes moves the book's fair value (unless ``impact`` is off)."""
        target = self.books[book]
        fills = self.engine.process(target, order)
        self.events += 1
        self._rates[book] = self.models[book].rate(target)
        self.mark(book)  # keeps the last mid
        if impact and fills and order.order_type != OrderType.CANCEL:
            taken = sum(f.size for f in fills)
            self.fair.trade(book, taken if order.side == Side.BID else -taken)
        return fills

    def step(self) -> None:
        """The next event, in whichever book it falls to, and any trader's decision that has come due."""
        rates = self._rates
        total = sum(rates)
        self.time += self.rng.exponential(1 / total)
        self.fair.advance_to(self.time)
        u = self.rng.random() * total
        k = 0
        for k, rate in enumerate(rates):  # noqa: B007
            u -= rate
            if u < 0:
                break
        order = self.models[k].event(self.books[k], self.time, self.fair.fairs[k], self.rng)
        if order is not None:
            taken = sum(f.size for f in self.submit(k, order, impact=order.order_type == OrderType.MARKET))
            self.flow[k] += taken if order.side == Side.BID else -taken
            self.volume[k] += taken
        for trader in self.traders:
            if trader.active and self.time >= trader.next_time:
                self.trade_for(trader)

    def trade_for(self, trader: BookTrader) -> None:
        """Ask a trader what it wants, and trade it, book by book, with market orders."""
        trader.next_time = self.time + trader.interval
        mids = [book.mid_price for book in self.books]
        if None in mids:
            return
        before = np.array(mids)
        trader.observe(self)
        moves = trader.decide(before)
        for move in moves:
            for book in np.nonzero(move)[0]:
                self._trade(trader, int(book), int(move[book]))
        if trader.record:
            trader.history.append((self.time, before, trader.position.copy(), trader.signal()))

    def _trade(self, trader: BookTrader, book: int, need: int) -> None:
        mid = self.books[book].mid_price
        fills = self.submit(book, market_order(abs(need), self.time, Side.BID if need > 0 else Side.ASK))
        sign = 1 if need > 0 else -1
        for f in fills:
            trader.position[book] += sign * f.size
            trader.cash -= sign * f.price * f.size
            trader.lots += f.size
            if mid is not None:
                trader.spread_paid += abs(f.price - mid) * f.size

    def run(self, until: float, every: float = 1.0) -> pd.DataFrame:
        """Step to time ``until``, sampling every book's mid and fair value and every trader's P&L."""
        rows, next_sample = [], self.time + every
        while self.time < until:
            self.step()
            while self.time >= next_sample:
                row = {"time": next_sample}
                row |= {f"mid_{i}": book.mid_price for i, book in enumerate(self.books)}
                row |= {f"fair_{i}": value for i, value in enumerate(self.fair.fairs)}
                marks = self.marks()
                row |= {f"value_{j}": trader.value(marks) for j, trader in enumerate(self.traders)}
                rows.append(row)
                next_sample += every
        return pd.DataFrame(rows)


@dataclass
class BookTrader:
    """A trader holding lots in every book: its account, and what it wants at each decision (``decide``)."""
    size: float = 5.0  # lots of the largest leg of each relation
    interval: float = 1.0
    netted: bool = True  # one order a book at each decision, for what all its relations want together
    active: bool = True
    next_time: float = 0.0
    position: Optional[np.ndarray] = None
    cash: float = 0.0
    lots: int = 0
    spread_paid: float = 0.0
    entries: int = 0
    netted_lots: int = 0  # lots its relations wanted that offset each other in one order, and so were never traded
    record: bool = False
    history: list = field(default_factory=list)

    def attach(self, n: int) -> None:
        self.position = np.zeros(n, dtype=int)

    def taking(self, decisions: int) -> float:
        """The time this many decisions take: each waits its interval, and then (``WAIT``) for the next event."""
        return decisions * (self.interval + WAIT)

    def value(self, marks: np.ndarray) -> float:
        """Its P&L: cash plus what it holds at the mids."""
        return self.cash + float(self.position @ marks)

    def observe(self, market: BookMarket) -> None:
        """See the market as it is, just before ``decide``: what a trader wants of it besides the mids."""

    def decide(self, mids: np.ndarray) -> list[np.ndarray]:
        """Take in the mids; the moves to make, each in lots a book (bought positive): sent in that order."""
        raise NotImplementedError

    def signal(self) -> Optional[np.ndarray]:
        """What it compares with its levels now, one a relation."""
        return None

    @property
    def levels(self) -> tuple[float, float]:
        """Where it enters, and where it goes flat: either side of 0."""
        return self.entry, self.exit

    @property
    def warming(self) -> Optional[tuple[int, int]]:
        """What it has seen and needs before it has a signal, else None."""
        return None

    def learn(self, mids: list[np.ndarray]) -> int:
        """Take in past mids, oldest first, trading on none of them: as many as it can use. How many it took."""
        kept = mids[-self._mids.maxlen:]
        self._mids.extend(kept)
        return len(kept)


@dataclass
class StateTrader(BookTrader):
    """Each relation (a row of ``weights``, with its largest leg 1) is short, flat or long: -1, 0 or 1."""
    weights: Optional[np.ndarray] = None
    states: Optional[np.ndarray] = None

    def _become(self, states: np.ndarray) -> list[np.ndarray]:
        """The moves that take it to these states: netted, or each relation's own, in whole lots either way."""
        if self.weights is None or not len(self.weights):
            return [-self.position] if self.position.any() else []
        old = self.states if self.states is not None else np.zeros(len(states), dtype=int)
        self.entries += int(np.count_nonzero((states != 0) & (states != old)))
        self.states = states
        if self.netted:
            move = np.rint(self.size * (states @ self.weights)).astype(int) - self.position
            wanted = np.abs(self.size * ((states - old)[:, None] * self.weights)).sum(axis=0)
            self.netted_lots += int(np.maximum(np.rint(wanted) - np.abs(move), 0).sum())
            return [move] if move.any() else []
        lots = np.rint(self.size * self.weights)
        moves = [(int(new) - int(was)) * lots[i] for i, (new, was) in enumerate(zip(states, old, strict=True))
                 if new != was]
        moves = [m.astype(int) for m in moves]
        left = (states @ lots).astype(int) - self.position - sum(moves, np.zeros_like(self.position))
        return moves + ([left] if left.any() else [])  # what an order could not fill last time


def _largest_one(rows: np.ndarray) -> np.ndarray:
    """Each row scaled so that its largest leg is 1: how many lots of each, for one of the largest."""
    return rows / np.abs(rows).max(axis=1, keepdims=True)


def trace_rank(johansen, level: int = 1) -> int:
    """How many relations the trace test accepts, going up from none (``level``: 0 for 90%, 1 for 95%, 2 for 99%)."""
    rank = 0
    for r in range(len(johansen.lr1)):
        if johansen.lr1[r] <= johansen.cvt[r, level]:
            break
        rank = r + 1
    return rank


def johansen_relations(history: np.ndarray, count: int = 1, lags: int = 2, every: int = 5) -> np.ndarray:
    """The cointegrating vectors the trace test accepts at 95% in these mids (one row in ``every``), strongest first,
    at most ``count``."""
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    j = coint_johansen(history[::every], det_order=0, k_ar_diff=lags)
    take = min(trace_rank(j), count)
    return _largest_one(j.evec[:, :take].T) if take else np.zeros((0, history.shape[1]))


def best_baskets(history: np.ndarray, count: int = 1, legs: int = 3, lags: int = 2, every: int = 5) -> np.ndarray:
    """The ``count`` baskets of ``legs`` books with the largest trace statistic above its 95% value, each from Johansen
    on those books alone, and sharing no book with a stronger one while any are left that do not."""
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    n = history.shape[1]
    history = history[::every]
    found = []
    for books in itertools.combinations(range(n), legs):
        j = coint_johansen(history[:, books], det_order=0, k_ar_diff=lags)
        if j.lr1[0] > j.cvt[0, 1]:
            row = np.zeros(n)
            row[list(books)] = j.evec[:, 0]
            found.append((float(j.lr1[0] - j.cvt[0, 1]), books, row))
    found.sort(key=lambda f: -f[0])
    return _pick(found, count, n)


def _ols_hedge(y: np.ndarray, x: np.ndarray) -> float:
    """The lots of y that go with one of x, from least squares of x on y."""
    return float(np.cov(x, y, ddof=0)[0, 1] / max(np.var(y), 1e-9))


def _pick(found: list, count: int, n: int) -> np.ndarray:
    """The first ``count`` of these (score, books, row), best first: those sharing no book first, then any."""
    chosen, used, rest = [], set(), []
    for item in found:
        if len(chosen) < count and not used & set(item[1]):
            chosen.append(item)
            used |= set(item[1])
        else:
            rest.append(item)
    chosen += rest[:count - len(chosen)]
    return _largest_one(np.array([c[2] for c in chosen])) if chosen else np.zeros((0, n))


def _pair_row(n: int, x: int, y: int, hedge: float) -> np.ndarray:
    row = np.zeros(n)
    row[x], row[y] = 1.0, -hedge
    return row


def pairs_by_distance(history: np.ndarray, count: int = 5, overlap: bool = True) -> np.ndarray:
    """The ``count`` pairs with the least sum of squared differences of prices rebased to 1 (Gatev et al. 2006; Rad et
    al. 2016), each hedged by least squares. A book may be in more than one, as there; with ``overlap`` off, those
    sharing no book come first."""
    n = history.shape[1]
    rebased = history / history[0]
    found = []
    for x, y in itertools.combinations(range(n), 2):
        ssd = float(((rebased[:, x] - rebased[:, y]) ** 2).sum())
        found.append((-ssd, (x, y), _pair_row(n, x, y, _ols_hedge(history[:, y], history[:, x]))))
    found.sort(key=lambda f: -f[0])
    return _largest_one(np.array([f[2] for f in found[:count]])) if overlap else _pick(found, count, n)


def pairs_by_pvalue(history: np.ndarray, count: int = 5, overlap: bool = True, maxlag: int = 20,
                    every: int = 5) -> np.ndarray:
    """The ``count`` pairs with the lowest Engle-Granger p-values below 5% in these mids (one row in ``every``), each
    hedged by least squares."""
    from statsmodels.tsa.stattools import coint

    n = history.shape[1]
    history = history[::every]
    found = []
    for x, y in itertools.combinations(range(n), 2):
        p = float(coint(history[:, x], history[:, y], maxlag=maxlag, autolag="AIC")[1])
        if p < 0.05:
            found.append((-p, (x, y), _pair_row(n, x, y, _ols_hedge(history[:, y], history[:, x]))))
    found.sort(key=lambda f: -f[0])
    if overlap:
        return _largest_one(np.array([f[2] for f in found[:count]])) if found else np.zeros((0, n))
    return _pick(found, count, n)


@dataclass
class RelationTrader(StateTrader):
    """Trades the z-score of each relation: short it above ``entry``, long below minus ``entry``, flat back at ``exit``.

    ``choose`` takes the last ``formation`` decisions' mids and gives the relations (a row each); fixed ``weights``
    need none. It chooses again every ``refit`` decisions, if given: a relation on the same books keeps its state,
    and the rest are closed. A relation's mean and standard deviation are over its last ``window`` decisions.
    """
    choose: Optional[Callable[[np.ndarray], np.ndarray]] = None
    formation: int = 10_000
    window: int = 400
    entry: float = 2.0
    exit: float = 0.0
    refit: Optional[int] = None
    fits: int = 0
    z: Optional[np.ndarray] = None
    # Given ``choose`` and the mids, runs it elsewhere and returns a future: the trader goes on meanwhile, and takes up
    # the answer at its first decision after it is done. Without it, choosing happens at once.
    defer: Optional[Callable] = None

    def __post_init__(self):
        if self.window < 3 or not 0 <= self.exit < self.entry or self.size <= 0:
            raise ValueError("a window of 3 or more, 0 <= exit < entry and a size above 0")
        if self.choose is None and self.weights is None:
            raise ValueError("either relations to trade, or a way to choose them")
        if self.choose is not None and self.formation < self.window:
            raise ValueError("a formation period at least as long as the window")
        self._mids: deque[np.ndarray] = deque(maxlen=self.formation)
        self._seen = 0
        self._next_fit = self.formation if self.choose is not None else None
        self._spreads: Optional[np.ndarray] = None  # the last window's, decisions by relations (a ring)
        self._filled = 0
        self._pending = None  # a choice being made elsewhere
        if self.weights is not None:
            self.weights = _largest_one(np.atleast_2d(np.asarray(self.weights, dtype=float)))
            self.states = np.zeros(len(self.weights), dtype=int)
            self._spreads = np.zeros((self.window, len(self.weights)))

    def _fit(self) -> None:
        old_weights, old_states = self.weights, self.states
        history = np.array(self._mids)
        if self.defer is None:
            weights = self.choose(history)
        elif self._pending is None:
            self._pending = self.defer(self.choose, history)
            return
        elif not self._pending.done():
            return
        else:
            weights, self._pending = self._pending.result(), None
        self.weights = weights if len(weights) else None
        self.fits += 1
        self._next_fit = self._seen + self.refit if self.refit else None
        if self.weights is None:
            self.states, self.z = None, None
            return
        known = history[:-1][-self.window:] @ weights.T  # all but the mids it is deciding on, which come next
        self._spreads = np.zeros((self.window, len(weights)))
        self._spreads[:len(known)] = known
        self._filled = len(known)
        states = np.zeros(len(weights), dtype=int)
        if old_weights is not None:  # the same books: the same relation, re-estimated
            held = {tuple(np.nonzero(w)[0]): s for w, s in zip(old_weights, old_states, strict=True)}
            for i, w in enumerate(weights):
                books = tuple(np.nonzero(w)[0])
                if len(books) < len(w):
                    states[i] = held.get(books, 0)
        self.states = states

    def decide(self, mids: np.ndarray) -> list[np.ndarray]:
        if self.choose is not None:
            self._mids.append(mids)
        self._seen += 1
        if self._next_fit is not None and len(self._mids) >= self.formation and self._seen >= self._next_fit:
            self._fit()
        if self.weights is None:
            return self._become(np.zeros(0, dtype=int))
        spreads = self.weights @ mids
        self._spreads[self._filled % self.window] = spreads
        self._filled += 1
        if self._filled < self.window:
            return []
        sd = self._spreads.std(axis=0)
        z = np.divide(spreads - self._spreads.mean(axis=0), sd, out=np.zeros_like(sd), where=sd > 0)
        self.z = z
        old = self.states
        new = old.copy()
        new[((old < 0) & (z < self.exit)) | ((old > 0) & (z > -self.exit))] = 0
        new[z > self.entry] = -1
        new[z < -self.entry] = 1
        return self._become(new)

    def signal(self):
        return self.z

    @property
    def warming(self):
        if self.choose is not None and self.fits == 0:
            return len(self._mids), self.formation
        return (self._filled, self.window) if self._filled < self.window else None

    @property
    def warm_up(self) -> float:
        """The time it needs before its first order: the decisions up to the first it can trade at (the formation
        period if it chooses its relations, else the window), and the one that follows."""
        return self.taking((self.formation if self.choose is not None else self.window) + 1)

    def learn(self, mids: list[np.ndarray]) -> int:
        taken = super().learn(mids) if self.choose is not None else 0
        self._seen += taken
        return taken


@dataclass
class ResidualTrader(StateTrader):
    """Stat-arb on PCA residuals, after Avellaneda and Lee (2010), in ticks.

    Every ``pca_every`` decisions, the correlation matrix of the books' moves over the last ``pca_window`` decisions
    gives ``factors`` eigenportfolios, each eigenvector over the books' standard deviations. At each decision every
    book's moves over the last ``window`` decisions are regressed on the eigenportfolios'; the residuals, added up, are
    fitted as an Ornstein-Uhlenbeck process by a lag-1 regression, and the s-score says how far the end of that sum is
    from the process's mean, in its standard deviations, the mean being centered across the books. A book whose fit
    reverts too slowly (a mean-reversion time over half the window) is rejected: no new trade, and an open one closes.

    Below minus ``open_at`` it buys one of the book and sells its factor hedge, above ``open_at`` the reverse; it closes
    a long above minus ``close_long`` and a short below ``close_short``. The hedge stays as it was on entry, until
    closed. Lots are ``size`` times the sum over all books, rounded, so hedges net across them.
    """
    factors: int = 2
    window: int = 120
    pca_window: int = 500
    pca_every: int = 20
    open_at: float = 1.25
    close_long: float = 0.5
    close_short: float = 0.75
    center: bool = True
    s: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.factors < 1 or self.window < 10 or self.pca_window < self.window or self.size <= 0:
            raise ValueError("a factor or more, a window of 10 or more, a PCA window no shorter, and a size above 0")
        if not (0 <= self.close_long < self.open_at and 0 <= self.close_short < self.open_at):
            raise ValueError("levels of 0 <= closing < opening")
        self._mids: deque[np.ndarray] = deque(maxlen=self.pca_window + 1)
        self._recent: deque[np.ndarray] = deque(maxlen=self.window + 1)
        self._eigen: Optional[np.ndarray] = None  # books by factors: lots of each book in each eigenportfolio
        self._since = 0
        self.max_b = math.exp(-2 / self.window)  # a mean-reversion time of half the window

    def attach(self, n: int) -> None:
        super().attach(n)
        if self.factors >= n:
            raise ValueError("fewer factors than books")
        self.states = np.zeros(n, dtype=int)
        self.weights = np.zeros((n, n))  # each book's relation, as held

    def _pca(self, moves: np.ndarray) -> None:
        sd = moves.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)  # a book that has not moved: it adds nothing
        standard = (moves - moves.mean(axis=0)) / sd
        values, vectors = np.linalg.eigh(standard.T @ standard / len(moves))
        self._eigen = vectors[:, np.argsort(values)[::-1][:self.factors]] / sd[:, None]

    def s_scores(self, moves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """From the last ``window`` moves: each book's s-score, whether its fit is accepted, and its factor loadings."""
        factors = moves @ self._eigen
        moves_c = moves - moves.mean(axis=0)
        factors_c = factors - factors.mean(axis=0)
        betas = np.linalg.lstsq(factors_c, moves_c, rcond=None)[0]  # factors by books
        x = np.cumsum(moves_c - factors_c @ betas, axis=0)
        before, after = x[:-1], x[1:]
        before_c = before - before.mean(axis=0)
        slope = (before_c * (after - after.mean(axis=0))).sum(axis=0) / np.maximum((before_c ** 2).sum(axis=0), 1e-12)
        b = np.clip(slope, 1e-6, 1 - 1e-6)
        a = after.mean(axis=0) - slope * before.mean(axis=0)
        shock = ((after - a - slope * before) ** 2).mean(axis=0)
        m = a / (1 - b)
        if self.center:
            m = m - m.mean()
        sd = np.sqrt(shock / (1 - b ** 2))
        s = np.divide(x[-1] - m, sd, out=np.zeros_like(sd), where=sd > 0)
        return s, (slope > 0) & (slope < self.max_b), betas

    def decide(self, mids: np.ndarray) -> list[np.ndarray]:
        self._mids.append(mids)
        self._recent.append(mids)
        if len(self._mids) <= self.pca_window:
            return []
        if self._since % self.pca_every == 0:
            self._pca(np.diff(np.array(self._mids), axis=0))
        self._since += 1
        s, good, betas = self.s_scores(np.diff(np.array(self._recent), axis=0))
        self.s = s
        old, new = self.states, self.states.copy()
        new[(old > 0) & (s > -self.close_long)] = 0
        new[(old < 0) & (s < self.close_short)] = 0
        new[~good] = 0
        new[(s < -self.open_at) & good & (old <= 0)] = 1
        new[(s > self.open_at) & good & (old >= 0)] = -1
        hedges = self._eigen @ betas  # books by books: what the factors take of each book's moves
        for i in np.nonzero((new != 0) & (new != old))[0]:
            self.weights[i] = -hedges[:, i]
            self.weights[i, i] += 1.0
        return self._become(new)

    def signal(self):
        return self.s

    @property
    def levels(self) -> tuple[float, float]:
        return self.open_at, min(self.close_long, self.close_short)

    @property
    def warming(self):
        return None if self.s is not None else (len(self._mids), self.pca_window + 1)

    @property
    def warm_up(self) -> float:
        """The time it needs before its first order: ``pca_window + 1`` decisions, and the one that follows."""
        return self.taking(self.pca_window + 2)

    def learn(self, mids: list[np.ndarray]) -> int:
        taken = super().learn(mids)
        self._recent.extend(mids[-self._recent.maxlen:])
        return taken


@dataclass
class CopulaPortfolio(StateTrader):
    """``lob.pairs.CopulaTrader`` on each of ``pairs`` pairs (Rad, Low and Faff 2016), chosen once by ``choose`` from
    the last ``window`` decisions' mids, each hedged as chosen; orders netted across them unless told otherwise.

    Given ``defer`` and the function that chooses the pairs and fits their copulas, with the mids, it runs that
    elsewhere and returns a future (as ``RelationTrader.defer`` does): the trader goes on meanwhile, and takes up the
    answer at its first decision after it is done, learning the mids that came meanwhile.
    """
    pairs: int = 5
    choose: Callable[..., np.ndarray] = pairs_by_distance
    window: int = 400
    trading: int = 200
    entry: float = 1.0
    exit: float = 0.0
    families: tuple[str, ...] = FAMILIES
    interval: float = 5.0
    traders: list = field(default_factory=list)
    defer: Optional[Callable] = None

    def __post_init__(self):
        if self.pairs < 1 or self.size <= 0 or self.trading < 1 or not 0 <= self.exit < self.entry:
            raise ValueError("a pair or more, a size above 0, a trading period of 1 or more, and 0 <= exit < entry")
        self._mids: deque[np.ndarray] = deque(maxlen=self.window + 1)
        self._seen = 0
        self._pending = None  # the pairs being chosen and fitted elsewhere
        self._sent = 0  # the decision they were asked for at

    def _build(self, history: np.ndarray) -> tuple[np.ndarray, list, list]:
        """The pairs chosen from these mids, a copula trader fitted to each, and the books of each, in its lots."""
        weights = self.choose(history, count=self.pairs)
        traders, books = [], []
        for row in weights:
            first, second = np.nonzero(row)[0]
            b, a = (first, second) if abs(row[first]) >= abs(row[second]) else (second, first)
            hedge = -row[a] / row[b]
            trader = CopulaTrader(window=self.window, trading=self.trading, entry=self.entry, exit=self.exit, size=1,
                                  hedge=max(hedge, 1e-6), interval=self.interval, families=self.families)
            trader.learn(list(zip(history[:, a].tolist(), history[:, b].tolist(), strict=True)))
            traders.append(trader)
            books.append((int(a), int(b)))
        # In b's lots, whichever sign the row has: the state's sign is that of b.
        return np.array([row / row[b] for row, (_, b) in zip(weights, books, strict=True)]), traders, books

    def _start(self) -> None:
        history = np.array(self._mids)
        if self.defer is None:
            self._take(self._build(history))
        elif self._pending is None:
            self._pending, self._sent = self.defer(self._build, history), self._seen
        elif self._pending.done():
            built, self._pending = self._pending.result(), None
            self._take(built, self._seen - self._sent)

    def _take(self, built: tuple[np.ndarray, list, list], missed: int = 0) -> None:
        """Trade these pairs from the next decision, after they learn the ``missed`` latest mids they lacked."""
        self.weights, self.traders, self._books = built
        self.states = np.zeros(len(self.weights), dtype=int)
        if missed:
            recent = np.array(list(self._mids)[-missed:])
            for trader, (a, b) in zip(self.traders, self._books, strict=True):
                trader.learn(list(zip(recent[:, a].tolist(), recent[:, b].tolist(), strict=True)))

    def decide(self, mids: np.ndarray) -> list[np.ndarray]:
        self._mids.append(mids)
        self._seen += 1
        if not self.traders:
            if len(self._mids) > self.window:
                self._start()  # these mids are among those it learned from: it decides from the next
            return []
        states = np.array([t.decide(float(mids[a]), float(mids[b]))
                           for t, (a, b) in zip(self.traders, self._books, strict=True)], dtype=int)
        return self._become(states)

    def signal(self):
        return np.array([t.z if t.z is not None else np.nan for t in self.traders]) if self.traders else None

    @property
    def warming(self):
        return None if self.traders else (len(self._mids), self.window + 1)

    @property
    def warm_up(self) -> float:
        """The time it needs before its first order: ``window + 1`` decisions start it, the next can trade, and one
        more."""
        return self.taking(self.window + 3)


def score_portfolio(market: BookMarket, trader: BookTrader, burn: float, span: float) -> dict[str, float]:
    """Run a market on for ``burn`` and then ``span`` of time, and how a trader in it did over ``span`` alone.

    As ``lob.pairs.score_trader``, with what it holds: ``gross``, its mean lots across the books, and ``exposure``, the
    root mean square of what it holds of the factors, through the market's own loadings. ``deviation`` is the standard
    deviation of its P&L's changes per unit of time, and ``netted`` the lots netting saved it.
    """
    while market.time < burn:
        market.step()
    start = trader.value(market.marks())
    paid, lots, entries, netted = trader.spread_paid, trader.lots, trader.entries, trader.netted_lots
    values, gross, exposure = [start], [], []
    next_sample = market.time + 1.0
    until = market.time + span
    while market.time < until:
        market.step()
        while market.time >= next_sample:
            values.append(trader.value(market.marks()))
            gross.append(np.abs(trader.position).sum())
            exposure.append(float(((trader.position @ market.fair.loadings) ** 2).sum()))
            next_sample += 1.0
    pnl = trader.value(market.marks()) - start
    changes = np.diff(values)
    deviation = float(changes.std()) if len(changes) > 1 else 0.0
    costs = trader.spread_paid - paid
    return {"pnl": pnl, "before_costs": pnl + costs, "costs": costs, "lots": trader.lots - lots,
            "entries": trader.entries - entries, "gross": float(np.mean(gross)) if gross else 0.0,
            "exposure": math.sqrt(np.mean(exposure)) if exposure else 0.0, "deviation": deviation,
            "netted": trader.netted_lots - netted,
            "risk_adjusted": float(changes.mean()) / deviation if deviation > 0 else 0.0}


def sector_loadings(sectors: int, books_each: int) -> np.ndarray:
    """A market factor, and each book's sector's: the pairs within a sector are cointegrated, and no others."""
    n = sectors * books_each
    b = np.zeros((n, 1 + sectors))
    b[:, 0] = 1.0
    for s in range(sectors):
        b[s * books_each:(s + 1) * books_each, 1 + s] = 1.0
    return b


def mixed_loadings(books: int, factors: int = 2, seed: int = 1, spread: float = 0.5) -> np.ndarray:
    """A market factor around 1, and other factors with loadings of mean 0 and standard deviation ``spread``: no two
    books' rows are parallel, so no pair is cointegrated, and every ``factors + 1`` books have one relation."""
    rng = np.random.default_rng(seed)
    b = rng.normal(0.0, spread, (books, factors))
    b[:, 0] = 1.0 + 0.2 * rng.standard_normal(books)
    b[:, 0] = np.maximum(b[:, 0], 0.3)
    return b
