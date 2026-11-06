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

Two options make the relation between the books harder to trade, and are off by default. ``beta_sd``
lets beta drift (mean-reverting around its start), so B moves ``beta_t`` times the common factor's moves,
``dF_B = beta_t dX + dU_B``, and no fixed hedge stays right. ``stress_share`` puts the common factor
into a stressed regime for that share of the time, more volatile and perhaps falling, so the books move
together mostly in their large moves: tail dependence, which a Gaussian copula cannot describe.

A random walk knows no floor, but a price does: orders stop at one tick, so a fair value far below that leaves a
book with one price and one side. So the fair values stay at or above ``floor`` ticks. A move that would take one
lower is cut short by pushing the common factor back up, just as far as needed, which moves both books together
and leaves the gap alone. Near the floor the common factor is therefore no longer a martingale; away from it
nothing changes, and no random number is drawn for it.

``PairMarket`` runs both books on one clock, with the arbitrageur if there is one, and takes orders
from outside with the same impact as anyone's. Three kinds of trader: ``PairsTrader`` trades the
z-score of a gap with a fixed hedge; ``KalmanTrader`` estimates the hedge as it goes and trades what
the estimate cannot explain; ``CopulaTrader`` trades the books' returns against each other through a copula.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from lob.book import OrderBook
from lob.copulas import FAMILIES, Copula, CopulaFit, Margin, fit_copula, pseudo_observations
from lob.matching import PriceTimeEngine
from lob.order import Fill, Order, OrderType, Side, cancel_order, limit_order, market_order

A, B = 0, 1
BOOKS = ("A", "B")
BETA_FLOOR = 0.05  # a drifting beta stays above this
FLOOR = 10.0  # the lowest a fair value goes: with orders up to 8 ticks from it (AnchoredZIModel), all above one tick


class FairValues:
    """The two books' fair values: a common random walk and a part of each book's own (see the module docstring).

    ``half_life`` (time units) makes the own parts, and so the gap, mean-revert; None leaves them
    wandering. ``volatility`` is the common factor's, ``own_volatility`` each own part's (ticks per
    square root of a time unit), ``impact`` the ticks a lot taken moves its book's own part.

    ``beta_sd`` makes beta an Ornstein-Uhlenbeck process around its start, with that standard deviation
    and ``beta_half_life`` (kept above ``BETA_FLOOR``). ``stress_share`` is the share of the time the common
    factor spends in a stressed regime, with ``stress_volatility`` and a drift of ``stress_drift`` ticks a unit
    of time; stress lasts ``stress_length`` on average, and the calm regime drifts the other way by
    enough that the factor stays a martingale.

    ``floor`` is the lowest either fair value goes (see the module docstring); ``pushed`` is what holding them
    there has added to the common factor so far.
    """

    def __init__(self, level_a: float = 100.0, level_b: float = 100.0, beta: float = 1.0, volatility: float = 0.3,
                 own_volatility: float = 0.8, half_life: Optional[float] = 50.0, impact: float = 0.05,
                 beta_sd: float = 0.0, beta_half_life: float = 2000.0, stress_share: float = 0.0,
                 stress_volatility: float = 2.0, stress_length: float = 50.0, stress_drift: float = 0.0,
                 floor: float = FLOOR):
        if half_life is not None and not half_life > 0:
            raise ValueError("the half-life must be above 0, or None for none")
        if not (volatility >= 0 and own_volatility >= 0 and impact >= 0 and beta > 0):
            raise ValueError("volatilities and impact must be 0 or more, and beta above 0")
        if not (beta_sd >= 0 and beta_half_life > 0):
            raise ValueError("beta's standard deviation must be 0 or more, and its half-life above 0")
        if not (0 <= stress_share < 1 and stress_volatility >= 0 and stress_length > 0):
            raise ValueError("the share of stress must be from 0 up to 1, its volatility 0 or more, its length above 0")
        if not (floor >= 0 and level_a >= floor and level_b >= floor):
            raise ValueError("the floor must be 0 or more, and both levels at or above it")
        self.level_a, self.level_b, self.beta = level_a, level_b, beta
        self.beta_mean, self.beta_sd, self.beta_half_life = beta, beta_sd, beta_half_life
        self.beta_kappa = math.log(2) / beta_half_life
        self.b_common = level_b  # B's common part, added up as beta drifts (with a fixed beta: from the formula)
        self.stress_share, self.stress_volatility = stress_share, stress_volatility
        self.stress_length, self.stress_drift = stress_length, stress_drift
        self.stressed = False
        self.volatility, self.own_volatility = volatility, own_volatility
        self.half_life, self.impact = half_life, impact
        self.kappa = math.log(2) / half_life if half_life is not None else 0.0
        self.x = level_a  # the common factor, starting at A's level
        self.own = [0.0, 0.0]
        self.floor = floor
        self.pushed = 0.0

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
        volatility, drift = self.volatility, 0.0
        if self.stress_share:  # at most one change of regime in dt (events come far faster than the regimes change)
            s = self.stress_share
            leave = (1 if self.stressed else s / (1 - s)) / self.stress_length
            if rng.random() < 1 - math.exp(-leave * dt):
                self.stressed = not self.stressed
            volatility = self.stress_volatility if self.stressed else self.volatility
            drift = self.stress_drift if self.stressed else -self.stress_drift * s / (1 - s)
        move = drift * dt + volatility * math.sqrt(dt) * rng.standard_normal()
        self.x += move
        if self.beta_sd:
            self.b_common += self.beta * move
            decay = math.exp(-self.beta_kappa * dt)
            self.beta = max(BETA_FLOOR, self.beta_mean + (self.beta - self.beta_mean) * decay
                            + self.beta_sd * math.sqrt(1 - decay * decay) * rng.standard_normal())
        if self.kappa:
            decay = math.exp(-self.kappa * dt)
            sd = self.own_volatility * math.sqrt((1 - decay * decay) / (2 * self.kappa))
        else:
            decay, sd = 1.0, self.own_volatility * math.sqrt(dt)
        for k in (A, B):
            self.own[k] = self.own[k] * decay + sd * rng.standard_normal()
        # What it takes to hold both above the floor, in moves of the common factor (B moves beta times as far).
        short = max(self.floor - self.x - self.own[A], (self.floor - self.common_b() - self.own[B]) / self.beta)
        if short > 0:
            self.x += short
            if self.beta_sd:
                self.b_common += self.beta * short
            self.pushed += short

    def trade(self, book: int, lots: float) -> None:
        """Lots taken in a book (bought positive, sold negative) push its own part by the impact."""
        self.own[book] += self.impact * lots

    def fair(self, book: int) -> float:
        if book == A:
            return self.x + self.own[A]
        return self.common_b() + self.own[B]

    def common_b(self) -> float:
        """B's common part: beta times the common factor's moves, added up."""
        return self.b_common if self.beta_sd else self.level_b + self.beta * (self.x - self.level_a)

    def gap(self) -> float:
        """The fair values' gap: B's own part less beta times A's."""
        return self.own[B] - self.beta * self.own[A]

    def gap_of(self, price_a: float, price_b: float) -> float:
        """The same for two prices: B's above its common part, less beta times A's above the common factor."""
        if self.beta_sd:
            return price_b - self.b_common - self.beta * (price_a - self.x)
        return price_b - self.level_b - self.beta * (price_a - self.level_a)

    def state(self) -> dict:
        return {"x": self.x, "own": list(self.own), "beta": self.beta, "stressed": self.stressed, "pushed": self.pushed}


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
    mean: Optional[float] = None  # the gap's mean and standard deviation over its window, as last measured
    sd: Optional[float] = None
    target: int = 0  # the lots of B it wants (A's follow by the hedge)
    active: bool = True  # stopped, it keeps what it holds but decides no more
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
        if not self.observe(mid_a, mid_b):
            return 0
        want = self.want()
        if want and want != self.target:
            self.entries += 1
        self.target = want
        return want

    def observe(self, mid_a: float, mid_b: float) -> bool:
        """Take in the mids (at a decision, or from the past, to learn from): whether it now has a signal."""
        gaps = self._gaps
        gaps.append(mid_b - self.hedge * mid_a)
        if len(gaps) < self.window:
            self.z = None
            return False
        mean = sum(gaps) / len(gaps)
        sd = math.sqrt(sum((g - mean) ** 2 for g in gaps) / len(gaps))
        self.mean, self.sd = mean, sd
        self.z = (gaps[-1] - mean) / sd if sd > 0 else 0.0
        return True

    def want(self) -> int:
        """The lots of B the signal calls for: beyond the entry level a position, back within the exit level flat."""
        if self.z > self.entry:
            return -self.size
        if self.z < -self.entry:
            return self.size
        if (self.target < 0 and self.z < self.exit) or (self.target > 0 and self.z > -self.exit):
            return 0  # back within exit of the mean, or past it
        return self.target

    def learn(self, mids: Sequence[tuple[float, float]]) -> int:
        """Take in past mids, oldest first, trading none of them: as much as it can use. How many it took."""
        kept = mids[-self.learns_from:] if self.learns_from else mids
        for mid_a, mid_b in kept:
            self.observe(mid_a, mid_b)
        return len(kept)

    @property
    def learns_from(self) -> Optional[int]:
        """How many past mids it can use (None: all of them)."""
        return self.window

    @property
    def warm_up(self) -> float:
        """The time it needs before it has a signal: its first window of decisions, and the one that follows."""
        return (self.window + 1) * self.interval

    @property
    def warming(self) -> Optional[tuple[int, int]]:
        """What it has seen and needs while it has no signal yet, else None."""
        return None if len(self._gaps) >= self.window else (len(self._gaps), self.window)

    def deviation(self, mid_a: float, mid_b: float) -> Optional[float]:
        """How far the gap it trades is from where it expects it, at these mids (None: it has no such gap yet)."""
        return None if self.mean is None else mid_b - self.hedge * mid_a - self.mean

    @property
    def width(self) -> Optional[float]:
        """That distance at its entry level, in ticks."""
        return None if self.sd is None else self.entry * self.sd

    @property
    def signals(self) -> tuple[Optional[float], ...]:
        """What it compares with its levels: here the z-score."""
        return (self.z,)

    @property
    def levels(self) -> tuple[float, float]:
        return self.entry, self.exit

    def value(self, mid_a: Optional[float], mid_b: Optional[float]) -> Optional[float]:
        """Its P&L: cash plus what it holds at the mids."""
        if mid_a is None or mid_b is None:
            return None
        return self.cash + self.position[A] * mid_a + self.position[B] * mid_b


class KalmanHedge:
    """A regression of B's price on A's whose slope and level may drift: the Kalman filter for

        y_t = level_t + beta_t (x_t - pivot_t) + e_t,    e ~ N(0, r)
        beta_t = beta_{t-1} + w_b,  level_t = level_{t-1} + w_l,    w ~ N(0, q)

    as Chan (2013, chapter 3) uses it to hedge a pair, except that the regressor is measured from a
    pivot near the prices: with x near 1,000 and a pivot of 0, as is usual, a small change of slope moves
    the fit by far more than the gap's whole range unless the level moves the other way at once. The
    pivot follows x (``move_pivot``), which changes the coordinates of the line and not the line.
    """

    def __init__(self, beta: float, level: float, pivot: float, p_beta: float, p_level: float,
                 q_beta: float, q_level: float, r: float):
        if not (r > 0 and q_beta >= 0 and q_level >= 0 and p_beta >= 0 and p_level >= 0):
            raise ValueError("the observation variance must be above 0, the others 0 or more")
        self.beta, self.level, self.pivot = beta, level, pivot
        self.p_bb, self.p_bl, self.p_ll = p_beta, 0.0, p_level
        self.q_beta, self.q_level, self.r = q_beta, q_level, r
        self.s = r  # the last innovation's variance

    def move_pivot(self, pivot: float) -> None:
        """The same line and uncertainty, measured from another point of x."""
        d = pivot - self.pivot
        self.level += self.beta * d
        self.p_ll += d * (2 * self.p_bl + d * self.p_bb)
        self.p_bl += d * self.p_bb
        self.pivot = pivot

    def fit(self, x: float) -> float:
        return self.level + self.beta * (x - self.pivot)

    def update(self, x: float, y: float) -> tuple[float, float]:
        """One observation: its innovation (from the estimate before it) and that innovation's variance."""
        p_bb, p_bl, p_ll = self.p_bb + self.q_beta, self.p_bl, self.p_ll + self.q_level
        d = x - self.pivot
        h_b, h_l = d * p_bb + p_bl, d * p_bl + p_ll  # P H'
        s = d * h_b + h_l + self.r
        k_b, k_l = h_b / s, h_l / s
        innovation = y - (self.level + self.beta * d)
        self.beta += k_b * innovation
        self.level += k_l * innovation
        self.p_bb, self.p_bl, self.p_ll = p_bb - k_b * h_b, p_bl - k_b * h_l, p_ll - k_l * h_l
        self.s = s
        return innovation, s


@dataclass
class KalmanTrader(PairsTrader):
    """Trades what a dynamic hedge cannot explain: Chan's (2013) rule on the Kalman filter's innovation.

    Its first ``window`` decisions fit a line by least squares, which starts the filter: the slope is
    the first hedge, and the residual variance the observation variance ``r``. After that, at each
    decision the filter predicts B's mid from A's, and z is the error over its usual size; beyond
    ``entry`` it shorts the gap (sells B, buys ``hedge`` times as much A), below minus ``entry`` the reverse,
    and within ``exit`` of 0, or past it, it goes flat. On entry ``hedge`` is the filter's slope; with
    ``rebalance`` it follows it while holding (in whole lots), and otherwise stays as it was.

    ``memory`` sets the state noise: the number of decisions the estimates in effect average over,
    the time constant of a filter with ``q = r / memory^2`` (for the slope, over the variance of A's
    price in the first window, so that a change of either moves the fit alike). It must be long against the
    gap's own half-life, or the filter follows the gap and leaves nothing to trade.

    Chan divides by the filter's own standard deviation. That needs ``r`` right and the observation noise white,
    and a gap is neither: with a half-life of 50 the first window's residuals understate it, and z's
    standard deviation came out near 2, so an entry level of 2 was really 1. Here the usual size is a running
    average of the squared errors, over the same memory, as the error's size is what the level is compared
    with; the ratio q / r, and so the filter's gains, do not depend on ``r``'s level at all.

    With ``learn_hedge`` off the slope stays at ``hedge``: the same filter on the level alone, for comparison.
    """
    memory: float = 1000.0
    rebalance: bool = True
    learn_hedge: bool = True
    filter: Optional[KalmanHedge] = None
    beta_hat: Optional[float] = None  # the filter's slope, whatever is held
    error_sd: Optional[float] = None  # the usual size of its error, as last measured

    def __post_init__(self):
        super().__post_init__()
        if not self.memory >= 1:
            raise ValueError("a memory of 1 or more")
        self._first: list[tuple[float, float]] = []

    def observe(self, mid_a: float, mid_b: float) -> bool:
        if self.filter is None:
            self._first.append((mid_a, mid_b))
            if len(self._first) < self.window:
                self.z = None
                return False
            self._start()
            return False
        f = self.filter
        f.move_pivot(f.pivot + (mid_a - f.pivot) / self.memory)  # an average of A's price over the memory
        innovation, _ = f.update(mid_a, mid_b)
        self.z = innovation / self.error_sd
        self.error_sd = math.sqrt(self.error_sd ** 2 + (innovation ** 2 - self.error_sd ** 2) / self.memory)
        self.beta_hat = f.beta
        if self.rebalance or not self.target:
            self.hedge = max(f.beta, 1e-6)
        return True

    def _start(self) -> None:
        x, y = np.array(self._first).T
        x_mean, y_mean = x.mean(), y.mean()
        x_var = max(float(np.mean((x - x_mean) ** 2)), 1.0)
        beta = float(np.mean((x - x_mean) * (y - y_mean))) / x_var if self.learn_hedge else self.hedge
        r = max(float(np.mean((y - y_mean - beta * (x - x_mean)) ** 2)), 1e-6)
        m = self.memory
        # The state noise for this memory, and its long-run uncertainty (P = sqrt(q r) for a level alone).
        slope = 1.0 if self.learn_hedge else 0.0
        self.filter = KalmanHedge(beta, float(y_mean), float(x_mean), slope * r / m / x_var, r / m,
                                  slope * r / m ** 2 / x_var, r / m ** 2, r)
        self.error_sd = math.sqrt(r)
        self.beta_hat = beta
        self.hedge = max(beta, 1e-6)
        self._first = []

    @property
    def learns_from(self) -> Optional[int]:
        return None

    @property
    def warming(self) -> Optional[tuple[int, int]]:
        return None if self.filter is not None else (len(self._first), self.window)

    def deviation(self, mid_a: float, mid_b: float) -> Optional[float]:
        return None if self.filter is None else mid_b - self.filter.fit(mid_a)

    @property
    def width(self) -> Optional[float]:
        return None if self.error_sd is None else self.entry * self.error_sd


@dataclass
class CopulaTrader(PairsTrader):
    """Trades the books' returns through a copula: the rule of Rad, Low and Faff (2016), after Xie et al. (2014).

    Each decision takes the returns of both books since the last. The last ``window`` of them are ranked
    (the margins) and the copula with the lowest AIC fitted to the ranks (``families``). For ``trading``
    decisions after that, each return is ranked against that sample, and the mispricing indices

        m_A = P(U_A <= u_A | U_B = u_B) - 0.5,    m_B = P(U_B <= u_B | U_A = u_A) - 0.5

    say how high each book's return was for what the other's did. They add up from 0 into ``flags``: a high
    flag means that book has become dear. When B's flag is above ``entry`` and A's below minus ``entry``, it
    shorts the gap (sells B, buys ``hedge`` times as much A), and the reverse the other way; it goes flat once
    both have come back to ``exit`` (they need not at once). When the trading period ends it closes what is
    still open, fits again on the last ``window`` returns, and the flags start from 0 again.

    Under the right copula with independent returns each m is uniform and independent of the last, so the
    flags are random walks: what makes them mean-revert, if anything does, is the returns' own dependence.
    """
    entry: float = 0.5
    window: int = 400
    interval: float = 5.0
    trading: int = 200
    families: tuple[str, ...] = FAMILIES
    fit: Optional[CopulaFit] = None
    flag_a: float = 0.0
    flag_b: float = 0.0
    fits: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.trading < 1:
            raise ValueError("a trading period of 1 or more")
        self._returns: deque[tuple[float, float]] = deque(maxlen=self.window)
        self._last: Optional[tuple[float, float]] = None
        self._since_fit = 0
        self._back_a = self._back_b = False
        self._margins: Optional[tuple[Margin, Margin]] = None

    @property
    def copula(self) -> Optional[Copula]:
        return self.fit.copula if self.fit is not None else None

    def observe(self, mid_a: float, mid_b: float) -> bool:
        last, self._last = self._last, (mid_a, mid_b)
        if last is None:
            return self.fit is not None
        ret_a, ret_b = mid_a - last[0], mid_b - last[1]
        self._returns.append((ret_a, ret_b))
        if self.fit is None:
            if len(self._returns) < self.window:
                self.z = None
                return False
            self._fit()
            return True
        self._since_fit += 1
        if self._since_fit >= self.trading:
            self._fit()  # its flags start from 0 again, which closes what is open: both are back
            return True
        h_a, h_b = self.fit.copula.conditionals(self._margins[0](ret_a), self._margins[1](ret_b))
        self.flag_a += h_a - 0.5
        self.flag_b += h_b - 0.5
        self.z = (self.flag_b - self.flag_a) / 2
        return True

    def _fit(self) -> None:
        a, b = np.array(self._returns).T
        self.fit = fit_copula(pseudo_observations(a), pseudo_observations(b), self.families)
        self._margins = (Margin(a), Margin(b))
        self.flag_a = self.flag_b = 0.0
        self.z = 0.0
        self._since_fit = 0
        self.fits += 1

    def want(self) -> int:
        a, b, entry, exit_ = self.flag_a, self.flag_b, self.entry, self.exit
        if self.target < 0:  # opened with B's flag high and A's low
            self._back_a = self._back_a or a >= -exit_
            self._back_b = self._back_b or b <= exit_
        elif self.target > 0:
            self._back_a = self._back_a or a <= exit_
            self._back_b = self._back_b or b >= -exit_
        want = self.target
        if want and self._back_a and self._back_b:
            want = 0
        if not want:
            if b > entry and a < -entry:
                want = -self.size
            elif b < -entry and a > entry:
                want = self.size
        if want and want != self.target:
            self._back_a = self._back_b = False
        return want

    @property
    def learns_from(self) -> Optional[int]:
        return self.window + 1  # a window of returns: its fit is then fresh, and its flags at 0

    @property
    def warming(self) -> Optional[tuple[int, int]]:
        return None if self.fit is not None else (len(self._returns), self.window)

    def deviation(self, mid_a: float, mid_b: float) -> Optional[float]:
        return None  # its signal lies in its flags, not in a gap

    @property
    def width(self) -> Optional[float]:
        return None

    @property
    def signals(self) -> tuple[Optional[float], ...]:
        return (self.flag_a, self.flag_b) if self.fit is not None else (None, None)


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
        """The books' gap from their mids: B's mid less its level and beta times A's move (with a drifting beta,
        its common part and beta times A's distance from the common factor)."""
        a, b = self.mid(A), self.mid(B)
        if a is None or b is None:
            return None
        return self.fair.gap_of(a, b)

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
            if trader.active and self.time >= trader.next_time:
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
        (``value_i``), its z-score, the lots of B it wants and its hedge (``z_i``, ``target_i``, ``hedge_i``).
        """
        rows = []
        next_sample = self.time + every
        while self.time < until:
            self.step()
            while self.time >= next_sample:
                a, b = self.books
                row = {"time": next_sample, "mid_a": a.mid_price, "mid_b": b.mid_price,
                       "fair_a": self.fair.fair(A), "fair_b": self.fair.fair(B), "gap": self.gap(),
                       "true_gap": self.fair.gap(), "spread_a": a.spread, "spread_b": b.spread,
                       "beta": self.fair.beta, "common": self.fair.x}
                for i, trader in enumerate(self.traders):
                    row |= {f"pos_a_{i}": trader.position[A], f"pos_b_{i}": trader.position[B],
                            f"value_{i}": trader.value(self.mark(A), self.mark(B)), f"z_{i}": trader.z,
                            f"target_{i}": trader.target, f"hedge_{i}": trader.hedge}
                rows.append(row)
                next_sample += every
        return pd.DataFrame(rows)


def score_trader(market: PairMarket, trader: PairsTrader, burn: float, span: float) -> dict[str, float]:
    """Run a market on for ``burn`` and then ``span`` of time, and how a trader in it did over ``span`` alone.

    Different settings need different warm-ups, so to compare them fairly each is given the same ``burn``, at least
    the longest of their warm-ups, and is measured over the same ``span``. ``risk_adjusted`` is the mean change of
    its P&L per unit of time over the standard deviation of those changes (0 if it never moves).
    """
    while market.time < burn:
        market.step()
    start = trader.value(market.mark(A), market.mark(B))
    paid, lots, entries = trader.spread_paid, trader.lots, trader.entries
    values = [start]
    next_sample = market.time + 1.0
    until = market.time + span
    while market.time < until:
        market.step()
        while market.time >= next_sample:
            values.append(trader.value(market.mark(A), market.mark(B)))
            next_sample += 1.0
    pnl = trader.value(market.mark(A), market.mark(B)) - start
    changes = np.diff(values)
    deviation = float(changes.std()) if len(changes) > 1 else 0.0
    return {"pnl": pnl, "before_costs": pnl + trader.spread_paid - paid, "costs": trader.spread_paid - paid,
            "lots": trader.lots - lots, "entries": trader.entries - entries,
            "risk_adjusted": float(changes.mean()) / deviation if deviation > 0 else 0.0}
