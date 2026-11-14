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

Every trader can trade with its costs in mind, if it nets its orders (the default). With a ``rate`` below 1 its aim
goes only that share of the way to its target at a decision (partial adjustment, Garleanu and Pedersen 2013), and with a
``band`` above 0 it does not trade within that many lots of its aim, and trades to the edge of the band from outside it
(Davis and Norman 1990). The defaults are the whole way to the target, every time. In Garleanu and Pedersen's words the
target here is their aim, and the aim here their position before it is rounded to lots. ``BookTrader`` says how their
aim differs, and what else to know.
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
from lob.deals import Deal, Deals
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

    ``trend`` (one number, or one a factor, each 0 or more) gives each factor a drift in ticks a unit of time: an
    Ornstein-Uhlenbeck process around 0 with that standard deviation and ``trend_half_life``, started from its
    stationary law and drawn from a stream of its own, so that the shocks of the factors and of the own parts, and the
    order flow's stream of random numbers, are those of the same seed without it. In fair values alone the own parts
    are too; in a market they are not, nor is the flow, which follows the fair values the drift moves. With no trend
    nothing is made or drawn.

    ``index`` adds one more book, the last, so that ``m`` books are the others and ``n = m + 1``. Its fair value is the
    weighted sum of the others' and a ``premium``, ``F_m = c . F + P``, for weights ``c`` of 0 or more that add up to 1
    or more (``True``: ones, the sum of the books). That holds above ``floor``: the index is held at the floor from
    below (the floor does not push for it, as it does for a book), and a lot taken can leave it, as it can a book, under
    the floor until the next step. The index is a book like the others, ``scale = sum(c)`` times over (``scales``): its
    order flow reaches ``8 * scale`` ticks, its starting book is ``scale`` ticks apart, its premium is ``scale`` times
    an own part of ``index_volatility`` (and the books' half-life), and a lot taken in it moves that premium by
    ``scale * impact``. That is not all the premium there is: with ``impact`` above 0 the order flow's own market orders
    in the index book make one too, with a standard deviation of about 1.2 ticks a unit of ``scale`` at any
    ``index_volatility``, so that a premium of exactly 0 takes ``index_volatility=0`` and ``impact=0`` both. A lot
    taken in a book moves the index by its weight times what it moved the book. The premium has a stream of its own, so
    that the other books' factors and shocks are those of the same seed without the index, and in fair values alone
    their fair values are too. In a market they are not: the index book's order flow draws from the stream the books'
    do, so their orders are not those of the same seed without it, nor, through the impact, their fair values. The
    index's loadings are ``c . B`` and its own part is ``c . U + P``, so that every relation of the books is one still.

    ``deals`` (a ``lob.deals.Deals``, or None) announces stock deals, which this class runs: see ``lob/deals.py`` for
    what they are. At each grid step, in this order, the deals that are on break, complete or go on (``deals`` is them,
    in the order announced, and ``ended`` those over, with their outcome); one uniform is drawn, and if it is below
    ``rate * step`` a deal is announced; the target's fair value of every deal on is the mix of the offer's and its own;
    and the index is taken from the books' fair values, whatever the deals did to them. The draws are ``[seed, 5]``'s,
    made in ``reseed``, and depend on time alone: one uniform a step, and three more at each announcement whatever
    follows. A book is free for a new deal from the deal's planned end plus ``cool``, whether the deal completed there
    or broke before it, so which books are free depends on the time alone too. So two markets of one seed with the same
    settings have the same deals whatever trades in them: the same two books, announced, ended and broken at the same
    times (but not the same ratios, jumps or shifts, which follow the fair values). Two that differ in ``breaks`` alone,
    with ``priced`` fixed, have the same deals but for how each ends. A break time is ``e / p`` after its announcement,
    for the one exponential ``e`` drawn then, so that with the same seed the deals that break at a lower ``breaks`` are
    some of those that break at a higher one, deal for deal. A break is applied at the first grid point at or after its
    time, if that is at or before the end, and the end is on the grid. ``gapped`` takes (book, jump in ticks) for a
    target at an announcement and at a break, and the index with its weight times that, when that is half a tick or
    more: ``BookMarket`` moves the books' orders by them and empties it. A lot taken in a book moves its own part, and
    the fair values of that book, of a target whose acquirer it is and of the index are taken again from the books' own
    values. ``exposures()`` has a target's row the same mix of its loadings and its acquirer's.
    """

    def __init__(self, loadings, level=3000.0, volatility=1.0, own_volatility: float = 0.8,
                 half_life: Optional[float] = 50.0, impact: float = 0.05, floor: float = FLOOR, step: float = 0.2,
                 loading_sd: float = 0.0, loading_half_life: float = 2000.0, trend=0.0, trend_half_life: float = 500.0,
                 index=None, index_volatility: float = 0.0, deals=None):
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
        # What the books and the traders read of an index and of deals, as it is with none: ``m`` the books that are not
        # an index, ``scales`` a size to each book (None: all alike), ``gapped`` the jumps of a fair value, as (book,
        # ticks), that ``BookMarket.step`` moves the book's orders by and empties.
        self.m = self.n
        self.index_weights: Optional[np.ndarray] = None
        self.scales: Optional[np.ndarray] = None
        self.premium = 0.0
        self.deal_settings = deals
        self.deals: list = []
        self.ended: list = []
        self.gapped: list[tuple[int, float]] = []
        # A trend: each factor's drift, from a stream of its own that ``reseed`` makes if any factor has one.
        self.trend = self._trend_of(trend, trend_half_life)
        self.trend_half_life = trend_half_life
        self.drift = np.zeros(self.k)
        self._trending = bool((self.trend > 0).any())
        self._trend_decay = math.exp(-math.log(2) / trend_half_life * step)
        self._trend_shock = self.trend * math.sqrt(1 - self._trend_decay ** 2)
        self._trng: Optional[np.random.Generator] = None
        self._tz: Optional[np.ndarray] = None
        self._tat = 0
        # An index: the last book, made from the arrays above, which stay those of the ``m`` books in ``_draws`` and in
        # every slice of the shocks. Its premium has a stream of its own, made by ``reseed``.
        if not (index_volatility >= 0 and math.isfinite(index_volatility)):
            raise ValueError("the index's volatility must be 0 or more")
        self.index_volatility = index_volatility
        self._premium_sd = 0.0
        self._prng: Optional[np.random.Generator] = None
        self._pz: Optional[np.ndarray] = None
        self._pat = 0
        if index is not None and index is not False:
            self._add_index(index)
        # Deals: the settings are checked and turned into steps here, the state (``deals``, ``ended``, the stream, the
        # books' cooling times) is made by ``reseed``.
        self._drng: Optional[np.random.Generator] = None
        if deals is not None:
            self._setup_deals(deals)
        self.reseed(1)

    def _trend_of(self, trend, half_life: float) -> np.ndarray:
        """The trend of each factor, from one number or one a factor: each 0 or more, and the half-life above 0."""
        trend = np.asarray(trend, dtype=float)
        if trend.ndim > 1 or (trend.ndim == 1 and len(trend) != self.k):
            raise ValueError(f"a trend is one number, or one for each of the {self.k} factors")
        if not (np.isfinite(trend).all() and (trend >= 0).all()):
            raise ValueError("a trend must be 0 or more")
        if not half_life > 0:
            raise ValueError("the trend's half-life must be above 0")
        return np.broadcast_to(trend, self.k).copy()

    def _add_index(self, index) -> None:
        """Make the index the last book of the ``m = n`` there are, from its weights (``True``: ones)."""
        m = self.n
        c = np.ones(m) if index is True else np.array(index, dtype=float)
        if c.shape != (m,):
            raise ValueError(f"an index has one weight for each of the {m} books, or is True for the sum of them")
        if not (np.isfinite(c).all() and (c >= 0).all() and c.sum() > 0):
            raise ValueError("index weights must be 0 or more and add up to more than 0")
        scale = float(c.sum())
        if math.isclose(scale, 1.0, rel_tol=1e-9):
            scale = 1.0  # weights of 1/m, which may add up to a hair under 1: a book like the others
        if scale < 1:
            raise ValueError("index weights must add up to 1 or more: less is a book quoted in parts of a tick")
        self.m, self.n = m, m + 1
        self.index_weights = c
        self.scales = np.append(np.ones(m), scale)
        self.mean_loadings = np.vstack([self.mean_loadings, c @ self.mean_loadings])
        self.loadings = np.vstack([self.loadings, c @ self.loadings])
        self.level = np.append(self.level, c @ self.level)
        self.common = np.zeros(self.n)
        self.own = np.zeros(self.n)
        self.fairs = self.level.tolist()
        shape = math.sqrt((1 - self._decay ** 2) / (2 * self.kappa)) if self.kappa else math.sqrt(self.step)
        self._premium_sd = self.index_volatility * scale * shape  # an own part's, times the scale

    def _setup_deals(self, deals: Deals) -> None:
        """Turn the deals' settings into grid steps and hazards, and say which books' rows of loadings are parallel."""
        if not isinstance(deals, Deals):
            raise ValueError("deals are a lob.deals.Deals, or None")
        if deals.rate * self.step >= 1:
            raise ValueError("deals are announced at most once a step: the rate times the step must be below 1")
        self._announce_p = deals.rate * self.step
        self._deal_steps = max(1, round(deals.length / self.step))  # the length, to a whole number of steps
        self._deal_p, self._deal_q = deals.hazards(self._deal_steps * self.step)
        self._cool_steps = math.ceil(deals.cool / self.step - 1e-9)
        rows = self.mean_loadings[:self.m]
        unit = rows / np.linalg.norm(rows, axis=1, keepdims=True)
        self._parallel = unit @ unit.T > 1 - 1e-9  # the same direction: every book loads positively on the first factor

    def reseed(self, seed: int) -> None:
        """Set the stream the factors and the own parts' shocks come from: only this and the time decide them.

        A trend's drift has a stream of its own, ``[seed, 4]``, and starts from its stationary law, an index's
        premium ``[seed, 3]``, and the deals ``[seed, 5]``, which start with none on, none ended and every book free,
        each made only if there is one.

        Nothing else starts again: the grid's clock, the factors, the own parts, the premium and the levels that
        completed deals moved are kept. A ``FactorValues`` is for one market, then: given to a second one, its grid is
        where the first left it.
        """
        self._rng = np.random.default_rng([seed, 1])
        self._z = self._rng.standard_normal((BLOCK, self._draws))
        self._at = 0
        if self._trending:
            self._trng = np.random.default_rng([seed, 4])
            self.drift = self.trend * self._trng.standard_normal(self.k)
            self._tz = self._trng.standard_normal((BLOCK, self.k))
            self._tat = 0
        if self.index_weights is not None:
            self._prng = np.random.default_rng([seed, 3])
            self._pz = self._prng.standard_normal(BLOCK)
            self._pat = 0
        if self.deal_settings is not None:
            self._drng = np.random.default_rng([seed, 5])
            self.deals, self.ended, self.gapped = [], [], []
            self._free_at = np.zeros(self.m, dtype=int)  # the step from which each book may be in a deal

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
        if self._trending:
            moves = moves + self._next_drift() * self.step
        self.factors += moves
        self.common += self.loadings @ moves
        if self.index_weights is not None:
            self._move_with_index(z)
            return
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
        if self.deal_settings is not None:
            values = self._deal_step(values)
        self.fairs = values.tolist()

    def _next_drift(self) -> np.ndarray:
        """The factors' drifts for the next step: each goes on from the last, and a shock of the trend's own stream."""
        if self._tat == BLOCK:
            self._tz = self._trng.standard_normal((BLOCK, self.k))
            self._tat = 0
        z = self._tz[self._tat]
        self._tat += 1
        self.drift = self.drift * self._trend_decay + self._trend_shock * z
        return self.drift

    def _move_with_index(self, z: np.ndarray) -> None:
        """The rest of a step in a market with an index: the books' own parts and loadings by the shocks of the ``m``
        books alone, the floor push by those books alone, and then the index from them and its premium.

        ``self.common`` has had the step's moves added, the index's row through its loadings (``c . B``), which are
        kept so; the floor holds the index from below but does not push for it.
        """
        k, m, c = self.k, self.m, self.index_weights
        if self._pat == BLOCK:
            self._pz = self._prng.standard_normal(BLOCK)
            self._pat = 0
        shock = self._pz[self._pat]
        self._pat += 1
        own = self.own[:m] * self._decay + self._own_sd * z[k:k + m]
        self.premium = float(self.premium * self._decay + self._premium_sd * shock)
        self.own = np.append(own, c @ own + self.premium)
        if self.loading_sd:
            rows = (self.mean_loadings[:m] + (self.loadings[:m] - self.mean_loadings[:m]) * self._loading_decay
                    + self._loading_shock * z[k + m:].reshape(m, k))
            rows[:, 0] = np.maximum(rows[:, 0], BETA_FLOOR)
            self.loadings = np.vstack([rows, c @ rows])
        values = self.level[:m] + self.common[:m] + own
        short = float(((self.floor - values) / self.loadings[:m, 0]).max())
        if short > 0:
            self.factors[0] += short
            self.common += self.loadings[:, 0] * short
            values = values + self.loadings[:m, 0] * short
            self.pushed += short
        if self.deal_settings is not None:
            seen = len(self.gapped)
            values = self._deal_step(values)
            jump = sum(float(c[book]) * ticks for book, ticks in self.gapped[seen:])  # the index follows what jumped
            if abs(jump) >= 0.5:
                self.gapped.append((m, jump))
        self.fairs = np.append(values, max(self.floor, float(c @ values + self.premium))).tolist()

    def _deal_step(self, values: np.ndarray) -> np.ndarray:
        """The deals' part of a grid step, from the books' own values (``values``, the floor's push done, one for each
        of the ``m`` books that are not the index) to the fair values they make: the deals on break, complete or go on;
        one uniform is drawn, and a deal may be announced; and each target's fair value is the mix of its offer's and
        its own. A completion changes ``values`` and the target's level by what the offer was worth over it."""
        t = self._steps * self.step
        self._standalone = values
        for deal in list(self.deals):
            if deal._breaks_at is not None and t >= deal._breaks_at:
                deal.outcome = "broken"
                gap = deal.ratio * values[deal.acquirer] - values[deal.target]
                self.gapped.append((deal.target, float(-self._pi(deal) * gap)))
            elif t >= deal.ends:
                deal.outcome = "completed"
                shift = float(deal.ratio * values[deal.acquirer] - values[deal.target])
                self.level[deal.target] += shift  # its value on its own is now the offer's, so that it does not jump
                if self.index_weights is not None:
                    self.level[self.m] += self.index_weights[deal.target] * shift
                values[deal.target] += shift
            else:
                continue
            deal.ended_at = t
            self.deals.remove(deal)
            self.ended.append(deal)
            # free from its planned end plus the cooling time, whether it completed there or broke before it: so who
            # is free for a new deal does not depend on which deals break, and ``breaks`` changes outcomes alone
            self._free_at[[deal.target, deal.acquirer]] = round(deal.ends / self.step) + self._cool_steps
        rng = self._drng
        if rng.random() < self._announce_p:  # three more draws, whatever follows: the count depends on time alone
            pick_target, pick_acquirer, wait = rng.random(), rng.random(), rng.standard_exponential()
            self._announce(values, t, pick_target, pick_acquirer, wait)
        if not self.deals:
            return values
        mixed = values.copy()
        for deal in self.deals:
            pi = self._pi(deal)
            mixed[deal.target] = pi * deal.ratio * values[deal.acquirer] + (1 - pi) * values[deal.target]
        return mixed

    def _announce(self, values: np.ndarray, t: float, pick_target: float, pick_acquirer: float, wait: float) -> None:
        """A deal between two books that are in none and past their cooling time, if there are two: the target by the
        first uniform, the acquirer by the second among the others (those whose loadings are parallel to the target's
        if ``within`` and any are, else all), and the time it breaks by the exponential ``wait``."""
        settings = self.deal_settings
        busy = {book for deal in self.deals for book in (deal.target, deal.acquirer)}
        free = [book for book in range(self.m) if book not in busy and self._steps >= self._free_at[book]]
        if len(free) < 2:
            return
        target = free[int(pick_target * len(free))]
        others = [book for book in free if book != target]
        if settings.within:
            kin = [book for book in others if self._parallel[target, book]]
            others = kin or others
        acquirer = others[int(pick_acquirer * len(others))]
        ends = (self._steps + self._deal_steps) * self.step  # on the grid
        deal = Deal(target, acquirer, float((values[target] + settings.premium) / values[acquirer]), t, ends,
                    settings.premium)
        if self._deal_p > 0 and t + wait / self._deal_p <= ends:
            deal._breaks_at = t + wait / self._deal_p
        self.deals.append(deal)
        self.gapped.append((target, self._pi(deal) * settings.premium))

    def _pi(self, deal: Deal) -> float:
        """The probability the market gives now to a deal going through: ``exp(-q * time left)``, at the last grid
        point (a fair value is the same from one step to the next, but for the lots taken)."""
        return math.exp(-self._deal_q * (deal.ends - self._steps * self.step))

    def trade(self, book: int, lots: float) -> None:
        """Lots taken in a book (bought positive, sold negative) push its own part by the impact.

        With an index, the index's fair value follows a book by the book's weight times what it moved, and a lot taken
        in the index moves its premium by its scale times the impact. With a deal on, the fair values of that book, of
        a target whose acquirer it is and of the index are taken again from the books' own values: a target moves by
        ``1 - pi`` times the impact for a lot in it, and by ``pi * r`` times for a lot in its acquirer."""
        if self.deals:
            self._trade_with_deals(book, lots)
            return
        if self.index_weights is not None:
            self._trade_with_index(book, lots)
            return
        move = self.impact * lots
        self.own[book] += move
        self.fairs[book] += move

    def _trade_with_index(self, book: int, lots: float) -> None:
        m, move = self.m, self.impact * lots
        if book == m:  # the premium moves by the index's scale times the impact, and nothing else does
            follow = self.scales[m] * move
            self.premium += follow
        else:  # the index follows the book by its weight times what the book moved
            self.own[book] += move
            self.fairs[book] += move
            follow = self.index_weights[book] * move
        self.own[m] += follow
        self.fairs[m] += follow

    def _trade_with_deals(self, book: int, lots: float) -> None:
        m, c, move = self.m, self.index_weights, self.impact * lots
        if book == m:  # a lot in the index: its premium, and nothing of the books or the deals
            follow = self.scales[m] * move
            self.premium += follow
            self.own[m] += follow
            self.fairs[m] += follow
            return
        self.own[book] += move
        self._standalone[book] += move
        if c is not None:
            self.own[m] += c[book] * move
        values, fairs = self._standalone, self.fairs
        fairs[book] += move  # as it is for a book in no deal, and a target's is taken again just below
        for deal in self.deals:
            if book in (deal.target, deal.acquirer):
                pi = self._pi(deal)
                fairs[deal.target] = float(pi * deal.ratio * values[deal.acquirer] + (1 - pi) * values[deal.target])
        if c is not None:
            fairs[m] = max(self.floor, float(c @ fairs[:m] + self.premium))

    def values(self) -> np.ndarray:
        return np.array(self.fairs)

    def exposures(self) -> np.ndarray:
        """What a lot of each book holds of each factor, books by factors: the loadings, as they are now (the array
        itself, not a copy) while no deal is on. With deals on, a copy in which a target's row is the same mix of its
        row and its acquirer's as its fair value is of theirs, ``pi * r * row[acquirer] + (1 - pi) * row[target]``, and
        the index's row (``c`` times the books' rows) is taken from those."""
        if not self.deals:
            return self.loadings
        rows = self.loadings.copy()
        for deal in self.deals:
            pi = self._pi(deal)
            rows[deal.target] = pi * deal.ratio * self.loadings[deal.acquirer] + (1 - pi) * self.loadings[deal.target]
        if self.index_weights is not None:
            rows[self.m] = self.index_weights @ rows[:self.m]
        return rows


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
        if models is None and fair.scales is not None:  # a book of scale s is a book like the others, s times over
            self.models = [AnchoredZIModel() if s == 1 else AnchoredZIModel(reach=max(1, round(8 * s)))
                           for s in fair.scales]
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
                gap = d if fair.scales is None else d * fair.scales[k]  # a book of scale s: s ticks apart
                book.add_order(limit_order(max(1, round(f - gap)), lots, 0.0, Side.BID))
                book.add_order(limit_order(round(f + gap), lots, 0.0, Side.ASK))
        self._rates = [model.rate(book) for model, book in zip(self.models, self.books, strict=True)]
        self.traders: list[BookTrader] = []
        for trader in traders:
            self.add(trader)

    def add(self, trader: BookTrader) -> None:
        trader.check(self)  # a trader that needs something this market lacks says so here, and is not added
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

    def _shift(self, book: int, ticks: int) -> None:
        """Move every resting order of a book by ``ticks``, a whole number: its quotes follow a jump of its fair value.

        Time priority is kept, within a price and across prices. A bid does not go below 1 tick, nor an ask below 2,
        so that no bid is at or above an ask: orders a fall takes below that meet at the lowest price there is, in the
        order of their old priority. The book is changed directly, not through ``submit``: it counts no event, no flow,
        no volume and no impact, and moves no fair value. Then the book's rate is taken again, from the orders it now
        has, and its mark from its mid.
        """
        if ticks != int(ticks):
            raise ValueError("a whole number of ticks")
        ticks = int(ticks)
        if not ticks:
            return
        target = self.books[book]
        bids, asks = target.resting_orders(Side.BID), target.resting_orders(Side.ASK)  # best first, in priority order
        for order in bids + asks:
            target.cancel_order(order.order_id)
        for order in bids:
            order.price = max(1, order.price + ticks)
            target.add_order(order)
        for order in asks:
            order.price = max(2, order.price + ticks)
            target.add_order(order)
        self._rates[book] = self.models[book].rate(target)
        self.mark(book)

    def step(self) -> None:
        """The next event, in whichever book it falls to, and any trader's decision that has come due.

        A fair value that jumped (``fair.gapped``) moves its book's orders with it, before the event is drawn and before
        any trader decides.
        """
        rates = self._rates
        total = sum(rates)
        self.time += self.rng.exponential(1 / total)
        self.fair.advance_to(self.time)
        if self.fair.gapped:
            for book, jump in self.fair.gapped:
                self._shift(book, round(jump))
            self.fair.gapped.clear()
            total = sum(rates)  # the shifted books have the rates of the orders they now hold
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
            trader.book_cash[book] -= sign * f.price * f.size
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
    """A trader holding lots in every book: its account, and what it wants at each decision (``decide``).

    It trades toward an aim (``_toward``). At each decision the aim goes ``rate`` of the way from where it was to the
    target, in lots that need not be whole, as in partial adjustment (Garleanu and Pedersen 2013): the share of the gap
    to the target that it closes is ``rate`` on average, however few lots there are, to the rounding. It trades the gap
    from its position to the aim, in whole lots, except within ``band`` lots of the aim, where it does not trade, and
    from outside the band it trades to its edge (Davis and Norman 1990). In a book whose target is 0 it trades all the
    way to the aim, whatever the band. ``rate`` is above 0 and at most 1, and ``band`` is 0 or more. With a rate of 1
    and no band the aim is the target and it trades the gap to it.

    What this leaves out. Garleanu and Pedersen's trader goes part of the way to an aim that, for one signal decaying
    at a steady rate, is the mean-variance portfolio of the signal now, scaled down the more, the faster the signal
    decays and the more trading costs. Here the target is the trader's own rule's, and is scaled to ``size`` lots
    whatever the signal's strength or decay: for a signal of one kind it points as their aim would, but how much it
    holds is the user's choice, not the costs'. Their costs grow with the square of the lots traded, for which going
    part of the way is best; for costs in proportion to them, as a spread's are, a band is (Davis and Norman 1990).
    Crossing the spreads here costs some of each: half a spread for every lot, and more the further an order walks
    into a book.

    Three things to know. A band applies to each book alone, so one leg of a relation can sit inside it while another
    trades. ``entries`` counts the states entered, as they ask, and not the trades made: with a band or a low rate it
    may never get to the lots a state asks for. And a trader of relations that does not net its orders
    (``netted=False``) cannot have a rate below 1 or a band above 0, and says so when made.
    """
    size: float = 5.0  # lots of the largest leg of each relation
    interval: float = 1.0
    netted: bool = True  # one order a book at each decision, for what all its relations want together
    active: bool = True
    next_time: float = 0.0
    position: Optional[np.ndarray] = None
    cash: float = 0.0
    lots: int = 0
    spread_paid: float = 0.0
    entries: int = 0  # times a state went from flat (or the other way) into a position, as asked: not as traded
    netted_lots: int = 0  # lots its relations wanted that offset each other in one order, and so were never traded
    record: bool = False
    history: list = field(default_factory=list)
    rate: float = 1.0
    band: float = 0.0

    def check(self, market: BookMarket) -> None:
        """Raise ValueError, with a sentence a user can read, if the market lacks what this trader needs of it. Called
        by ``BookMarket.add`` before ``attach``. A trader needs nothing but its books by default."""

    def attach(self, n: int) -> None:
        self.position = np.zeros(n, dtype=int)
        self._aim = np.zeros(n)  # the lots it aims at, not whole
        self.book_cash = np.zeros(n)  # what each book's trades have paid into ``cash``: negative for a buy; the sum

    @property
    def _cost_aware(self) -> bool:
        """Whether it trades less than the whole way to its target: a rate below 1, or a band above 0."""
        return self.rate < 1.0 or self.band > 0.0

    def _check_costs(self) -> None:
        if not (0 < self.rate <= 1 and self.band >= 0):
            raise ValueError("a rate above 0 and at most 1, and a band of 0 or more")

    def _toward(self, target: np.ndarray) -> np.ndarray:
        """The lots to trade a book now to move toward ``target``, whole, bought positive.

        The aim goes ``rate`` of the way to the target. Of the gap from the position to the aim it takes what lies
        beyond the band, rounded, but all of it in a book whose target is 0. With a rate of 1 and no band that is the
        gap to the target.
        """
        target = np.asarray(target)
        self._aim = self._aim + self.rate * (target - self._aim)
        gap = self._aim - self.position
        beyond = np.sign(gap) * np.maximum(np.abs(gap) - self.band, 0)
        return np.rint(np.where(target == 0, gap, beyond)).astype(int)

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

    def _check_costs(self) -> None:
        super()._check_costs()
        if self._cost_aware and not self.netted:
            raise ValueError("cost-aware trading needs netting")

    def attach(self, n: int) -> None:
        super().attach(n)
        self._asked = np.zeros(n, dtype=int)  # the lots the states last asked for, which is not what it holds

    def _become(self, states: np.ndarray) -> list[np.ndarray]:
        """The moves toward these states: netted, or each relation's own, in whole lots either way.

        Netted, the aim goes ``rate`` of the way to the lots the states ask for, and it trades toward the aim, not
        at all within ``band`` of it (``_toward``), except in a book whose target is 0, which it closes all the way to
        the aim. Asking again with the same states goes on from where the last moves left it. With no relations it
        closes everything at once, whatever the settings, and forgets the aim. Not netted, all the way.
        """
        if self.weights is None or not len(self.weights):
            self._asked = np.zeros_like(self.position)
            self._aim = np.zeros(len(self.position))
            return [-self.position] if self.position.any() else []
        old = self.states if self.states is not None else np.zeros(len(states), dtype=int)
        self.entries += int(np.count_nonzero((states != 0) & (states != old)))
        self.states = states
        if self.netted:
            target = np.rint(self.size * (states @ self.weights)).astype(int)
            # Netting saved what the relations wanted of the books less what the states now ask of them. Going all the
            # way, that is the gap. Short of it the gap holds lots not yet traded, which netting did not save, and
            # the ask is the change in the target.
            ask = target - (self._asked if self._cost_aware else self.position)
            wanted = np.abs(self.size * ((states - old)[:, None] * self.weights)).sum(axis=0)
            self.netted_lots += int(np.maximum(np.rint(wanted) - np.abs(ask), 0).sum())
            self._asked = target
            move = self._toward(target)
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


JOHANSEN_BOOKS = 12  # the most books Johansen's test has critical values for (statsmodels has none beyond)


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
    at most ``count``. It takes 12 books at most (``JOHANSEN_BOOKS``): for more there is no table of critical values to
    read a rank from, and it raises ValueError rather than give one."""
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    if history.shape[1] > JOHANSEN_BOOKS:
        raise ValueError("Johansen's test has critical values for 12 books at most")
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
        self._check_costs()
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
        self._check_costs()
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
        self._check_costs()
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


def _score(market: BookMarket, trader: BookTrader, burn: float, span: float,
           factors: bool = False) -> tuple[dict[str, float], list[float], float]:
    """What ``score_portfolio`` gives, the trader's P&L a unit of time apart (the first is where the span starts), and,
    with ``factors``, the sum over those units of what it held of each factor at the unit before times that factor's
    move since."""
    while market.time < burn:
        market.step()
    start = trader.value(market.marks())
    paid, lots, entries, netted = trader.spread_paid, trader.lots, trader.entries, trader.netted_lots
    values, gross, exposure = [start], [], []
    part, held, seen = 0.0, None, None
    if factors:
        held, seen = trader.position @ market.fair.exposures(), market.fair.factors.copy()
    next_sample = market.time + 1.0
    until = market.time + span
    while market.time < until:
        market.step()
        while market.time >= next_sample:
            values.append(trader.value(market.marks()))
            gross.append(np.abs(trader.position).sum())
            now = trader.position @ market.fair.exposures()
            exposure.append(float((now ** 2).sum()))
            if factors:
                part += float(held @ (market.fair.factors - seen))
                held, seen = now, market.fair.factors.copy()
            next_sample += 1.0
    pnl = trader.value(market.marks()) - start
    changes = np.diff(values)
    deviation = float(changes.std()) if len(changes) > 1 else 0.0
    costs = trader.spread_paid - paid
    score = {"pnl": pnl, "before_costs": pnl + costs, "costs": costs, "lots": trader.lots - lots,
             "entries": trader.entries - entries, "gross": float(np.mean(gross)) if gross else 0.0,
             "exposure": math.sqrt(np.mean(exposure)) if exposure else 0.0, "deviation": deviation,
             "netted": trader.netted_lots - netted,
             "risk_adjusted": float(changes.mean()) / deviation if deviation > 0 else 0.0}
    return score, values, part


def score_portfolio(market: BookMarket, trader: BookTrader, burn: float, span: float) -> dict[str, float]:
    """Run a market on for ``burn`` and then ``span`` of time, and how a trader in it did over ``span`` alone.

    As ``lob.pairs.score_trader``, with what it holds: ``gross``, its mean lots across the books, and ``exposure``, the
    root mean square of what it holds of the factors, through the market's own exposures (``fair.exposures()``).
    ``deviation`` is the standard deviation of its P&L's changes per unit of time,
    ``netted`` the lots netting saved it, and ``entries`` the times its states entered a position, as they asked (with
    a rate or a band it may never get to their lots).
    """
    return _score(market, trader, burn, span)[0]


def score_holdings(market: BookMarket, trader: BookTrader, burn: float, span: float,
                   horizon: float = 100.0) -> dict[str, float]:
    """What ``score_portfolio`` gives, and the risk of what the trader holds over the long run, and what part of its
    P&L the factors made.

    ``deviation_long`` is the standard deviation of its P&L's changes over successive stretches of ``horizon`` units,
    over the square root of ``horizon``: against ``deviation`` (units a unit), it says how much a position that
    continues, as a factor's drift does, adds to the risk of holding it for long. ``risk_adjusted_long`` is the mean
    change of its P&L a unit over that, and both are 0 with fewer than two stretches. ``factor_part`` is the sum over
    the span's units of what it held of each factor at the unit before (``position @ market.fair.exposures()``) times
    that factor's move since (``fair.factors``). ``pnl`` less it is everything else, which is more than the books' own
    parts and the spread paid: the books' own parts (an index's premium among them); what deals did to what it held
    (the jumps, the drift of ``pi``, the shifts of the levels); what its orders made or lost by trading at mids that
    were off their fair values, with its own impact on the lots it had just taken; the spread paid; and the change
    between the span's two ends in what its position is marked off the fair values. So call it the rest, and do not
    read it as the own parts and the costs alone.
    """
    if not (horizon >= 1 and horizon == round(horizon)):
        raise ValueError("a horizon of a whole number of units, 1 or more")
    score, values, part = _score(market, trader, burn, span, factors=True)
    stretch = round(horizon)
    long_changes = np.diff(values[::stretch])
    deviation_long = float(long_changes.std()) / math.sqrt(stretch) if len(long_changes) > 1 else 0.0
    mean = float(np.diff(values).mean()) if len(values) > 1 else 0.0
    score["deviation_long"] = deviation_long
    score["risk_adjusted_long"] = mean / deviation_long if deviation_long > 0 else 0.0
    score["factor_part"] = part
    return score


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
