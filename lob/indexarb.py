"""Index arbitrage: the index book against the books it is made of (``FactorValues(index=...)`` in ``lob/statarb.py``).

An index's fair value is the weighted sum of the books' and a premium of its own, ``F_idx = c . F + P``, so
``(-c, 1)`` is a relation: its spread is the books' own parts, weighted, and the premium, and the factors are not in
it. Trading it is long the index and short each book, in lots, when the spread is low, and the other way when it is
high. ``IndexTrader`` is a ``RelationTrader`` that does this with the one row ``replicate_index`` gives, every book at
its weight, or with the row ``index_basket`` gives from the past mids: a sample of a few books that stand in for the
rest. In a market where the index is the sum of the books (weights of 1) the replication is one lot of the index
against one lot of each book, whatever the size and the rate. In one where the weights add up to 1 the index is the
largest leg and each book's is its weight, so a size that is not a multiple of the number of books does not hold
the hedge it asks for, since each leg is rounded to a whole lot on its own.

What it leaves out: fixed weights with no chooser (a chooser is what ``learn`` and ``warming`` work through),
weights a refit would change (the published weights are the weights), and any model of what the index's premium
is: it trades the spread. The market's ``index_volatility`` is part of how much there is to find, not all: with
``impact`` above 0 the order flow's own orders in the index book make a premium too, with a standard deviation of about
1.2 ticks a unit of the index's scale at any ``index_volatility``, so that the strict control, no premium at all,
takes ``index_volatility=0`` and ``impact=0`` both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from lob.statarb import BookMarket, RelationTrader, _largest_one


def replicate_index(history: np.ndarray, weights) -> np.ndarray:
    """The one relation that replicates an index of these weights, ``(-c, 1)`` with the index the last book: one row,
    its largest leg 1, which is the index's when the weights add up to 1 and a book's when one of them is above it.

    ``history`` (rows are times, columns books) is not read but for its width, which is one more than the weights'.
    That makes this a ``choose`` for a ``RelationTrader``. Raises ValueError for weights that are not finite and 0
    or more, or a history of another width.
    """
    c = np.asarray(weights, dtype=float)
    if c.ndim != 1 or not len(c) or not (np.isfinite(c).all() and (c >= 0).all() and c.sum() > 0):
        raise ValueError("index weights are one for each book, each 0 or more, and add up to more than 0")
    books = np.asarray(history).shape[1] if np.ndim(history) == 2 else 0
    if books != len(c) + 1:
        raise ValueError(f"the history has {books} books, and an index of {len(c)} books has {len(c) + 1}")
    return _largest_one(np.append(-c, 1.0)[None, :])


def index_basket(history: np.ndarray, legs: int, index: int = -1) -> np.ndarray:
    """The ``legs`` books whose levels best explain the index's level, as one relation: one row, the index 1 and each
    book of the sample minus its coefficient, scaled so that its largest leg is 1.

    The coefficients are least squares of the index's level on the sample's with a constant. The sample is chosen by
    forward selection: at each step it adds the book that lowers the residual sum of squares most (the lowest column
    on a tie). ``index`` is the column of the index, the last by default. Raises ValueError for ``legs`` below 1 or
    above the number of other books, or an ``index`` that is no column.

    On levels, a sample explains the index well only with as many legs as the books have factors that tell them apart:
    with fewer, a factor is left in the spread and it wanders. The fit within the history does not say so: levels
    fitted to levels look tighter in the history they were fitted to than they are after it.
    """
    x = np.asarray(history, dtype=float)
    if x.ndim != 2:
        raise ValueError("a history is rows of times by columns of books")
    n = x.shape[1]
    if not -n <= index < n:
        raise ValueError(f"the index is one of the {n} columns")
    index %= n
    if not 1 <= legs <= n - 1:
        raise ValueError(f"legs from 1 to the {n - 1} books besides the index")
    x = x - x.mean(axis=0)  # a constant in the regression, taken out
    y, others = x[:, index], [b for b in range(n) if b != index]
    chosen: list[int] = []
    for _ in range(legs):
        best = None
        for b in others:
            if b in chosen:
                continue
            columns = x[:, chosen + [b]]
            beta = np.linalg.lstsq(columns, y, rcond=None)[0]
            rss = float(((y - columns @ beta) ** 2).sum())
            if best is None or rss < best[0]:
                best = (rss, b)
        chosen.append(best[1])
    beta = np.linalg.lstsq(x[:, chosen], y, rcond=None)[0]
    row = np.zeros(n)
    row[index] = 1.0
    row[chosen] = -beta
    return _largest_one(row[None, :])


@dataclass
class _IndexChoice:
    """What an ``IndexTrader`` chooses with: ``replicate_index`` at the market's weights, or ``index_basket``. A class
    of its own, not a function, so that it can hold the weights ``check`` gives it and still be sent to a worker."""
    legs: int = 0
    weights: Optional[np.ndarray] = None

    def __call__(self, history: np.ndarray) -> np.ndarray:
        if self.legs:
            return index_basket(history, self.legs)
        if self.weights is None:
            raise ValueError("this market has no index book")  # it was never checked against one
        return replicate_index(history, self.weights)


@dataclass
class IndexTrader(RelationTrader):
    """Trades the index against its books, as a ``RelationTrader`` trades any relation (the z-score of its spread over
    its last ``window`` decisions, short above ``entry``, long below minus it, flat at ``exit``).

    ``legs`` 0 trades the one relation of every book at the index's published weights (``replicate_index``). ``legs``
    above 0 trades the one relation ``index_basket`` finds in the last ``formation`` decisions' mids, from that many
    books. It needs no ``choose`` or ``weights`` of its own, and refuses them: ``check`` (which ``BookMarket.add``
    calls) takes the index's weights from the market and raises ValueError if it has no index book. ``formation`` is
    the window unless given. The mids it chooses from are those it has seen, so ``learn``, ``warming`` and ``warm_up``
    work as for any trader that chooses.

    With ``hedged`` off it reads the same spread and z-score, so the same states and entries, and holds the index
    alone: the lots it asks for are the hedged trader's lots of the index, and none of any book. It then holds what
    the index holds of the factors, and carries that risk too.

    ``size`` is lots of the relation's largest leg. With the sum of the books for the index every leg of the full
    replication is 1: ``size`` lots of the index and of each book, exact at every size and every rate. With weights
    that add up to 1 the index is the largest leg, and each book has its weight of it: only a size that is a multiple
    of the number of books (for equal weights) holds the hedge it asks for, since each leg is rounded to whole lots.
    With ``legs`` above 0 the lots are ``size`` times the row, rounded the same way.

    A refit (``refit``, off unless given) does what it does for any relation: it chooses again, and keeps the state
    of a relation on the same books only if it has fewer legs than there are books. The full replication has a leg in
    every book, and so goes flat at each refit, and takes a position again when its z-score says so; the same row is
    all a refit finds. A sample (``legs`` above 0) may change its books at a refit, and keeps its state if it does not.
    """
    legs: int = 0
    hedged: bool = True
    formation: Optional[int] = None  # the window, unless given

    def __post_init__(self):
        if self.choose is not None or self.weights is not None:
            raise ValueError("an index trader makes its own relation: give it legs and hedged, not weights or choose")
        if isinstance(self.legs, bool) or not isinstance(self.legs, (int, np.integer)) or self.legs < 0:
            raise ValueError("legs is 0 for every book at the index's weights, or the number of books to sample")
        self.hedged = bool(self.hedged)
        if self.formation is None:
            self.formation = self.window
        self.choose = _IndexChoice(int(self.legs))
        super().__post_init__()

    def check(self, market: BookMarket) -> None:
        weights = market.fair.index_weights
        if weights is None:
            raise ValueError("this market has no index book")
        if self.legs > len(weights):
            raise ValueError(f"{self.legs} legs from an index of {len(weights)} books")
        self.choose.weights = np.array(weights, dtype=float)

    def _become(self, states: np.ndarray) -> list[np.ndarray]:
        """As ``RelationTrader``'s, except that unhedged only the index's leg of the relation is held."""
        if self.hedged or self.weights is None or not len(self.weights):
            return super()._become(states)
        whole = self.weights
        index_only = np.zeros_like(whole)
        index_only[:, -1] = whole[:, -1]
        self.weights = index_only
        try:
            return super()._become(states)
        finally:
            self.weights = whole
