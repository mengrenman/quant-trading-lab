"""Cross-sectional long/short traders over the books of ``lob.statarb.BookMarket``.

Each decision a trader scores every book and holds lots in proportion to the scores, made dollar-neutral and neutral to
the books' estimated common factors (the PCA of ``ResidualTrader``, in ``estimate_factors``). What differs is the score.
``ReversalTrader`` buys the books that fell and sells those that rose (Lehmann 1990; Lo and MacKinlay 1990),
``MomentumTrader`` the other way round (Jegadeesh and Titman 1993), ``LeadLagTrader`` buys the books that the others'
last moves say will rise (Lo and MacKinlay 1990; Hou 2007), and ``FlowTrader`` those with the buying, in their trades
or in their queues. ``contrarian_decomposition`` splits the profit of a contrarian portfolio, as Lo and MacKinlay
(1990) do.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from lob.order import Side
from lob.statarb import BookMarket, BookTrader


def neutralize(w: np.ndarray, betas: Optional[np.ndarray] = None) -> np.ndarray:
    """``w`` less its projection on a row of ones and the rows of ``betas`` (factors by books), by least squares.

    What is left sums to 0 and has no exposure to any factor. Without ``betas`` it is ``w`` less its mean.
    """
    w = np.asarray(w, dtype=float)
    if betas is None or np.size(betas) == 0:
        return w - w.mean()
    rows = np.vstack([np.ones(len(w)), betas])
    return w - rows.T @ np.linalg.lstsq(rows.T, w, rcond=None)[0]


def estimate_factors(moves: np.ndarray, factors: int) -> tuple[np.ndarray, np.ndarray]:
    """The eigenportfolios and exposures of these moves (decisions by books), as ``ResidualTrader`` finds them.

    The moves are standardized (a book that has not moved has a standard deviation of 1). The eigenportfolios are the
    ``factors`` largest eigenvectors of their correlation matrix, each over the books' standard deviations: lots of
    each book, books by factors. The exposures, factors by books, regress each book's demeaned moves on the demeaned
    moves of the eigenportfolios.
    """
    sd = moves.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    standard = (moves - moves.mean(axis=0)) / sd
    values, vectors = np.linalg.eigh(standard.T @ standard / len(moves))
    eigen = vectors[:, np.argsort(values)[::-1][:factors]] / sd[:, None]
    portfolios = moves @ eigen
    moves_c = moves - moves.mean(axis=0)
    portfolios_c = portfolios - portfolios.mean(axis=0)
    return eigen, np.linalg.lstsq(portfolios_c, moves_c, rcond=None)[0]


@dataclass
class CrossSectionTrader(BookTrader):
    """Holds lots in proportion to a score of each book, neutral to the dollar and to the books' common factors.

    Every ``hold`` decisions it scores the books (``scores``, which each kind defines), takes the scores less their
    projection on the ones and on the books' exposures to ``factors`` estimated factors (``neutralize``), and holds
    ``size`` lots of the book with the largest weight and the rest in proportion, rounded. The factors are estimated
    again every ``pca_every`` decisions, from the last ``pca_window`` moves. All-equal scores leave nothing to hold.
    It holds nothing until it has the mids it needs (``needed``).

    ``lookback`` is the number of decisions over which a score is measured.
    """
    lookback: int = field(kw_only=True)
    hold: int = 1  # decisions between rebalances
    factors: int = 1  # estimated factors it is neutral to: 0 for the dollar alone
    pca_window: int = 500
    pca_every: int = 20

    def __post_init__(self):
        if self.lookback < 1 or self.hold < 1 or self.factors < 0 or self.pca_every < 1 or self.size <= 0:
            raise ValueError("a lookback, a hold and a PCA interval of 1 or more, 0 or more factors and a size above 0")
        if self.factors and self.pca_window < max(self.lookback + 1, 10):
            raise ValueError("a PCA window of 10 or more, and more than the lookback")
        self._mids: deque[np.ndarray] = deque(maxlen=self.needed)
        self._since = 0  # decisions since it could first rebalance
        self._eigen: Optional[np.ndarray] = None  # books by factors, as estimate_factors gives them
        self._betas: Optional[np.ndarray] = None  # factors by books
        self._mean: Optional[np.ndarray] = None  # each book's mean move over the window they were estimated on
        self._scores: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.states: Optional[np.ndarray] = None

    def attach(self, n: int) -> None:
        if self.factors >= n - 1:
            raise ValueError("fewer factors than the books less one")
        super().attach(n)
        self.weights = np.eye(n)  # each book is its own relation
        self.states = np.zeros(n, dtype=int)  # the sign of what it wants of each book

    def _reach(self) -> int:
        """The moves of the mids that its scores use."""
        return self.lookback

    @property
    def needed(self) -> int:
        """The mids it needs before it can rebalance: those of its scores or of its factors, whichever is more."""
        return 1 + max(self._reach(), self.pca_window if self.factors else 0)

    def scores(self) -> np.ndarray:
        """A score of each book, from the mids it has (and whatever ``observe`` kept): the more, the more it buys."""
        raise NotImplementedError

    def _moves(self) -> np.ndarray:
        return np.diff(np.array(self._mids), axis=0)

    def _recent(self, moves: int) -> np.ndarray:
        """The last ``moves`` moves of the mids."""
        return np.diff(np.array(list(self._mids)[-(moves + 1):]), axis=0)

    def _residual(self, moves: np.ndarray) -> np.ndarray:
        """These moves less what the factors take of them, each demeaned as in ``estimate_factors`` (raw with none)."""
        if not self.factors:
            return moves
        moves_c = moves - self._mean
        return moves_c - (moves_c @ self._eigen) @ self._betas

    def _fit(self) -> None:
        """Estimate what it learns from the mids, again."""
        if self.factors:
            moves = self._moves()
            self._eigen, self._betas = estimate_factors(moves, self.factors)
            self._mean = moves.mean(axis=0)

    def decide(self, mids: np.ndarray) -> list[np.ndarray]:
        self._mids.append(mids)
        if self.warming is not None:
            return []
        if self._since % self.pca_every == 0:
            self._fit()
        due = self._since % self.hold == 0
        self._since += 1
        return self._rebalance() if due else []

    def _rebalance(self) -> list[np.ndarray]:
        scores = self.scores()
        self._scores = scores
        w = neutralize(scores, self._betas if self.factors else None)
        top = float(np.abs(w).max())
        if top > 1e-9 * float(np.abs(scores).max()):  # not zero, but for rounding
            target = np.rint(self.size * w / top).astype(int)
        else:
            target = np.zeros(len(w), dtype=int)
        states = np.sign(target)
        self.entries += int(np.count_nonzero((states != 0) & (states != self.states)))
        self.states = states
        move = target - self.position
        return [move] if move.any() else []

    def signal(self) -> Optional[np.ndarray]:
        return self._scores

    @property
    def levels(self) -> tuple[float, float]:
        return 0.0, 0.0

    @property
    def warming(self) -> Optional[tuple[int, int]]:
        return (len(self._mids), self.needed) if len(self._mids) < self.needed else None

    @property
    def warm_up(self) -> float:
        """The time it needs before its first order: ``needed`` decisions, and the one that follows."""
        return self.taking(self.needed + 1)


@dataclass
class ReversalTrader(CrossSectionTrader):
    """Buys the books that fell over the last ``lookback`` decisions and sells those that rose, after Lehmann (1990)
    and Lo and MacKinlay (1990).

    A book's score is minus the sum of its residual moves over those decisions: its moves less what the factors take
    of them, or its moves with ``factors`` of 0.
    """

    def scores(self) -> np.ndarray:
        return -self._residual(self._recent(self.lookback)).sum(axis=0)


@dataclass
class MomentumTrader(CrossSectionTrader):
    """Buys the books that rose over ``lookback`` decisions that ended ``skip`` decisions ago and sells those that fell,
    after Jegadeesh and Titman (1993).

    A book's score is the sum of its residual moves over those decisions, as ``ReversalTrader`` measures them, but
    plus.
    """
    skip: int = 0

    def __post_init__(self):
        if self.skip < 0:
            raise ValueError("a skip of 0 or more")
        super().__post_init__()
        if self.factors and self.pca_window < self.lookback + self.skip + 1:
            raise ValueError("a PCA window above the lookback and the skip together")

    def _reach(self) -> int:
        return self.lookback + self.skip

    def scores(self) -> np.ndarray:
        return self._residual(self._recent(self.lookback + self.skip)[:self.lookback]).sum(axis=0)


@dataclass
class LeadLagTrader(CrossSectionTrader):
    """Buys the books that the others' last moves say will rise, after Lo and MacKinlay (1990) and Hou (2007).

    Every ``pca_every`` decisions, with the factors, it regresses each book's raw move on all the books' raw moves one
    decision before, over the last ``pca_window`` moves, each demeaned. The penalty of the ridge regression is
    ``ridge`` times the average variance of the regressors. A book's own lag is left out, since that is reversal: the
    diagonal of the result, ``gamma``, is 0. A book's score is its row of ``gamma`` times the last decision's moves.
    ``lookback`` does not enter the score.
    """
    ridge: float = 1.0

    def __post_init__(self):
        if self.ridge < 0 or self.pca_window < 10:
            raise ValueError("a ridge of 0 or more and a PCA window of 10 or more")
        super().__post_init__()
        self.gamma: Optional[np.ndarray] = None

    def _reach(self) -> int:
        return self.pca_window

    def _fit(self) -> None:
        super()._fit()
        moves = self._moves()
        before, after = moves[:-1], moves[1:]
        before, after = before - before.mean(axis=0), after - after.mean(axis=0)
        n, size = before.shape[1], len(before)
        covariance = before.T @ before / size
        penalty = self.ridge * np.trace(covariance) / n
        beta = np.linalg.lstsq(covariance + penalty * np.eye(n), before.T @ after / size, rcond=None)[0]
        self.gamma = beta.T  # row j: what the books' moves say of book j's next
        np.fill_diagonal(self.gamma, 0.0)

    def scores(self) -> np.ndarray:
        return self.gamma @ self._recent(1)[0]


def _best_lots(book, side: Side) -> int:
    """The lots at the best price on a side of a book: 0 if it has none."""
    price = book.best_bid if side == Side.BID else book.best_ask
    return book.depth_at_price(price, side) if price is not None else 0


@dataclass
class FlowTrader(CrossSectionTrader):
    """Buys the books with the buying in the order flow, in trades or in queues; ``measure`` says which.

    With ``"trades"`` a book's score is the change in its flow over the last ``lookback`` decisions over the change in
    its volume (0 with no volume): the share of what traded that was net bought. With ``"queues"`` it is the lots at
    the best bid less those at the best ask, over their sum, now (0 with a side empty). It sees both in ``observe``.
    """
    measure: str = "trades"

    def __post_init__(self):
        if self.measure not in ("trades", "queues"):
            raise ValueError('a measure of "trades" or "queues"')
        super().__post_init__()
        self._seen: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=self.lookback + 1)  # flow and volume
        self._lots: Optional[tuple[np.ndarray, np.ndarray]] = None  # at the best bid and the best ask

    def _reach(self) -> int:
        return self.lookback if self.measure == "trades" else 0

    def observe(self, market: BookMarket) -> None:
        self._seen.append((market.flow.copy(), market.volume.copy()))
        if self.measure == "queues":
            self._lots = (np.array([_best_lots(book, Side.BID) for book in market.books], dtype=float),
                          np.array([_best_lots(book, Side.ASK) for book in market.books], dtype=float))

    def scores(self) -> np.ndarray:
        if self.measure == "queues":
            bid, ask = self._lots
            return np.divide(bid - ask, bid + ask, out=np.zeros(len(bid)), where=(bid > 0) & (ask > 0))
        (flow_then, volume_then), (flow_now, volume_now) = self._seen[0], self._seen[-1]
        traded = volume_now - volume_then
        return np.divide(flow_now - flow_then, traded, out=np.zeros(len(traded)), where=traded > 0)

    @property
    def warming(self) -> Optional[tuple[int, int]]:
        """Its decisions done and ``needed``. It waits for the flow it reads and for the mids its factors need; the
        longer wait counts."""
        left = max(self._reach() + 1 - len(self._seen), self.needed - len(self._mids))
        return (self.needed - left, self.needed) if left > 0 else None

    def learn(self, mids: list[np.ndarray]) -> int:
        """Take in past mids, for the factors alone (the flow and the queues of the past were not seen): how many."""
        return super().learn(mids) if self.factors else 0


def contrarian_decomposition(moves: np.ndarray, lag: int = 1) -> dict[str, float]:
    """Where the profit of the linear contrarian portfolio comes from, after Lo and MacKinlay (1990).

    The moves, ``T`` decisions by ``n`` books, give the portfolio ``w[i, t] = -(r[i, t-lag] - mean_j r[j, t-lag]) / n``
    held over ``t``, whose profit is ``sum_i w[i, t] r[i, t]``. ``profit`` is the mean of that over the ``T - lag``
    pairs of moves ``(r[t-lag], r[t])``. With ``Omega[i, j]`` the covariance of ``r[i, t]`` with ``r[j, t-lag]``, ``mu``
    and ``nu`` the books' means of the later and of the earlier moves, and ``1`` a vector of ones, all over those
    pairs, it is the sum of::

        own   = -(n - 1) / n**2 * trace(Omega)                   the books' own autocovariances
        cross = (1' Omega 1 - trace(Omega)) / n**2               their cross-autocovariances
        means = -sum_i (mu_i - mean(mu)) (nu_i - mean(nu)) / n   the dispersion of their means

    A negative autocovariance adds to ``own``; a book that follows the others' last moves adds to ``cross``; the books
    whose means differ take from ``means`` (it is minus the variance of the means, when ``mu`` and ``nu`` are alike).
    The parts add to ``profit`` up to rounding.
    """
    moves = np.asarray(moves, dtype=float)
    if lag < 1 or len(moves) - lag < 2:
        raise ValueError("a lag of 1 or more, and two or more pairs of moves")
    n = moves.shape[1]
    before, after = moves[:-lag], moves[lag:]
    profit = -float(((before - before.mean(axis=1, keepdims=True)) * after).sum(axis=1).mean()) / n
    earlier, later = before.mean(axis=0), after.mean(axis=0)
    omega = (after - later).T @ (before - earlier) / len(after)
    trace = float(np.trace(omega))
    return {"profit": profit,
            "own": -(n - 1) / n ** 2 * trace,
            "cross": (float(omega.sum()) - trace) / n ** 2,
            "means": -float(((later - later.mean()) * (earlier - earlier.mean())).sum()) / n}
