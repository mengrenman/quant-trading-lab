"""Sparse mean-reverting baskets: the portfolios of a few books that the books' past levels predict least.

``RelationTrader`` (``lob/statarb.py``) trades the z-score of portfolios chosen from past mids. Johansen's test finds
the portfolios that are stationary. Here the criterion is Box and Tiao's (1977) *predictability*: fit a VAR(1) to the
levels, ``x_t = A x_{t-1} + e_t``, and the predictability of a portfolio ``w`` is the share of the variance of ``w'x``
that the books' lagged levels explain, ``w'A gamma A'w / w'gamma w``. It is near 1 for a random walk, phi^2 for an AR(1)
of coefficient phi, and near 0 for noise, so the least predictable portfolio is the most mean-reverting one. Over a
given set of books it is the smallest root of a generalized eigenproblem (``box_tiao``). Over all the books it is
dense, which costs a leg in every book; asking for a few legs only (d'Aspremont 2011, "Identifying small
mean-reverting portfolios", Quantitative Finance 11(3)) is NP-hard. ``greedy_basket`` does what he does for a first
answer, growing a set of books one at a time, each time the book that lowers the least predictability most, and
``sparse_baskets`` does that from every pair and gives what ``best_baskets`` gives, one row of weights a basket with
the largest leg 1, so a ``RelationTrader`` takes it as its ``choose``.

What is left out: any test of whether a basket is a relation (Johansen's choosers return only what the trace test
accepts, and these return the least predictable whether or not it is one); a floor on the variance of a basket's
swings, which Cuturi and d'Aspremont (2013, "Mean reversion with a variance threshold") add so that a basket is not
chosen for swinging too little to pay for its legs; his semidefinite relaxation and exact search, of which the greedy
one is the cheapest and, as he notes, can be far from optimal; and lags beyond the first.
"""

from __future__ import annotations

import itertools

import numpy as np
import scipy.linalg

from lob.statarb import _largest_one, _pick


def fit_var1(levels: np.ndarray, ridge: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """``A`` and ``gamma`` of the VAR(1) ``x_t = A x_{t-1} + e_t`` fitted by least squares to these levels (rows are
    times, columns books), each column less its mean over all the rows.

    With ``X0`` the rows 0 to T-2 and ``X1`` the rows 1 to T-1,
    ``A' = (X0'X0 + ridge * trace(X0'X0)/n * I)^{-1} X0'X1``: ordinary least squares at ``ridge`` 0, and shrunk toward 0
    by a penalty that is a share ``ridge`` of the average variance on the diagonal. (A singular ``X0'X0``, as with a
    book that never moved, gets the minimum-norm solution.)

    ``gamma`` is ``X0'X0 / (T - 1)``, the covariance of the lagged levels: with it, ``A gamma A'`` is exactly the
    covariance of the fitted values ``X0 A'``, and the predictability of ``w`` is the variance of ``w'x_t`` that the
    books' lagged levels explain over the variance of ``w'x_{t-1}``. Box and Tiao's ``gamma`` is the covariance of
    ``x_t``, the same under stationarity; d'Aspremont (2011, eq. 7) divides by the current levels' covariance instead.
    In eight runs of the sectors and mixed markets the least eigenvalue agrees to within 0.002 either way on 2,000 rows
    (0.01 on 400); the largest can exceed 1 by several times 1/T (0.005 at most on 2,000 rows, 0.03 on 400, up to ten
    times 1/T).

    Raises ValueError for a negative ``ridge``, or for fewer than ``n + 2`` rows (at ``n + 1`` the fit is exact, and
    below that ``gamma`` is singular).
    """
    x = np.asarray(levels, dtype=float)
    if x.ndim != 2:
        raise ValueError("levels are rows of times by columns of books")
    if ridge < 0:
        raise ValueError("a ridge of 0 or more")
    t, n = x.shape
    if t < n + 2:
        raise ValueError(f"at least {n + 2} rows for a VAR(1) on {n} books, and there are {t}")
    x = x - x.mean(axis=0)
    x0, x1 = x[:-1], x[1:]
    g = x0.T @ x0
    a_t = np.linalg.lstsq(g + ridge * np.trace(g) / n * np.eye(n), x0.T @ x1, rcond=None)[0]
    return a_t.T, g / (t - 1)


def predictability(w: np.ndarray, a: np.ndarray, gamma: np.ndarray) -> float:
    """``w'A gamma A'w / w'gamma w``: the share of the portfolio's variance that the books' past predicts, one step back
    through the VAR(1) (all the books' lagged levels, not the portfolio's own lag alone).

    Near 1 for a random walk, phi^2 for an AR(1) with coefficient phi whose past alone predicts it, near 0 for noise.
    It does not depend on the scale of ``w``. Raises ValueError for a portfolio with no variance.
    """
    w = np.asarray(w, dtype=float)
    variance = float(w @ gamma @ w)
    if variance <= 0:
        raise ValueError("a portfolio with some variance")
    return float(w @ a @ gamma @ a.T @ w) / variance


def _check_books(books, n: int) -> list[int]:
    chosen = sorted(int(b) for b in books)
    if not chosen or chosen[0] < 0 or chosen[-1] >= n or len(set(chosen)) != len(chosen):
        raise ValueError(f"distinct books among 0 to {n - 1}")
    return chosen


def _least_predictable(m: np.ndarray, gamma: np.ndarray, books: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """``M w = lambda gamma w`` on these books alone: the eigenvalues ascending and the vectors (columns)."""
    m_s, g_s = m[np.ix_(books, books)], gamma[np.ix_(books, books)]
    try:
        return scipy.linalg.eigh(m_s, g_s)
    except np.linalg.LinAlgError:
        # gamma_SS is not positive definite (a book that never moved, or books that are copies of each other): add the
        # least jitter that makes it so, enough to lift its smallest eigenvalue to a trillionth of the average
        # variance. A book that never moved comes out alone with lambda 0; the trader never trades a spread whose sd
        # is 0, so this only matters in degenerate histories.
        scale = max(float(np.trace(g_s)) / len(books), 1e-300)
        jitter = 1e-12 * scale - min(float(np.linalg.eigvalsh(g_s)[0]), 0.0)
        return scipy.linalg.eigh(m_s, g_s + jitter * np.eye(len(books)))


def _moment(a: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    m = a @ gamma @ a.T
    return (m + m.T) / 2  # symmetric but for rounding


def box_tiao(a: np.ndarray, gamma: np.ndarray, books=None) -> tuple[np.ndarray, np.ndarray]:
    """The portfolios of these books (all if None) by predictability, least first: ``(eigenvalues ascending, rows)``.

    They solve the generalized symmetric eigenproblem ``M w = lambda gamma w`` with ``M = A gamma A'``, on the rows and
    columns of ``M`` and ``gamma`` for the books given. The predictability of each row is its eigenvalue, the rows are
    ``gamma``-orthonormal, and each row is over all ``n`` books, zero outside the books. The smallest eigenvalue is the
    least predictability any portfolio on those books has, and its row that portfolio.

    Eigenvalues are not bounded by 1 in sample (the largest can exceed it by several times 1/T), but are not below 0.
    If the ``gamma`` of the books is not positive definite a little is added to its diagonal (see the comment in the
    code).
    """
    n = len(gamma)
    chosen = list(range(n)) if books is None else _check_books(books, n)
    values, vectors = _least_predictable(_moment(a, gamma), gamma, chosen)
    rows = np.zeros((len(chosen), n))
    rows[:, chosen] = vectors.T
    return values, rows


def greedy_basket(a: np.ndarray, gamma: np.ndarray, start, legs: int) -> tuple[float, tuple[int, ...], np.ndarray]:
    """``(lambda, books, row)`` of the basket of ``legs`` books grown from the books in ``start``: one at a time, the
    book whose addition gives the least smallest eigenvalue (ties to the lowest index). ``books`` are sorted, and the
    predictability of ``row`` is ``lambda``.

    Adding a book never raises the least predictability, since the quotient is minimized over a larger set (it stays
    when the new book adds nothing in sample). d'Aspremont's (2011, Section 3.1) search grows one set from the single
    least predictable book and he notes it can be far from the optimum, since the best sets of different sizes need not
    nest; ``sparse_baskets`` grows one from every pair instead, because in these books no single book mean-reverts and a
    first step from one has nothing to go on. From a start that holds all but one of the books of an optimum, the greedy
    step is the exhaustive search; from further away it need not reach it.

    Raises ValueError for a ``start`` that is not distinct books of the ``n``, or has more than ``legs``, or ``legs``
    above ``n``.
    """
    n = len(gamma)
    books = _check_books(start, n)
    if not len(books) <= legs <= n:
        raise ValueError(f"legs between the {len(books)} books to start from and the {n} there are")
    m = _moment(a, gamma)
    values, vectors = _least_predictable(m, gamma, books)
    best, vector = float(values[0]), vectors[:, 0]
    while len(books) < legs:
        trials = []
        for b in range(n):
            if b not in books:
                grown = sorted([*books, b])
                values, vectors = _least_predictable(m, gamma, grown)
                trials.append((float(values[0]), b, grown, vectors[:, 0]))
        best, _, books, vector = min(trials, key=lambda trial: (trial[0], trial[1]))
    row = np.zeros(n)
    row[books] = vector
    return best, tuple(books), row


def sparse_baskets(history: np.ndarray, count: int = 1, legs: int = 3, every: int = 5,
                   ridge: float = 0.0) -> np.ndarray:
    """The ``count`` least predictable baskets of ``legs`` books in these mids (one row in ``every``), a row of weights
    each, the largest leg 1: the shape of ``best_baskets`` and ``johansen_relations``, so a ``RelationTrader`` takes it.

    A VAR(1) is fitted to the sampled mids (``fit_var1``, with ``ridge``). With ``legs`` of 2 or more every pair of
    books is grown greedily to ``legs`` books (``greedy_basket``; a pair is its own start at 2), each distinct set of
    books is kept once, and they are ranked by least predictability, least first. As in ``best_baskets``, the baskets
    sharing no book with a better one come first, then any. With ``legs`` of 0 it is instead the ``count`` least
    predictable portfolios of all the books (``box_tiao``), dense, like ``johansen_relations``, and
    ``min(count, n)`` of them.

    It tests nothing: with ``legs`` of 2 or more it returns ``count`` baskets (fewer only when the distinct sets run
    out) whether or not any is a relation, where Johansen's choosers return only what the trace test accepts, and a book
    added for a small in-sample gain brings a factor in, as a third book from another sector does in a market of
    sectors. With ``legs`` of 0, rows past the number of true relations are not relations.

    It ranks by predictability alone, which is scale-free: it does not see how large a basket's swings are against what
    its legs cost to trade (Cuturi and d'Aspremont 2013 add a variance threshold for that), and it rewards unpredictable
    noise in the mids as much as mean reversion of the fair values, so among true relations it would prefer the noisiest
    books'. That is dormant here, where every book's order flow is alike and the mids' deviation from their fair values
    is a lag that cancels within a relation. Among exact relations with independent own parts of equal variance, as in
    these markets, every relation has the same population predictability, so which it ranks first is sampling noise.

    Raises ValueError for ``count`` below 1, ``legs`` of 1, below 0 or above the number of books, ``every`` below 1,
    a negative ``ridge``, or fewer than ``n + 2`` rows once sampled (at ``n + 1`` the fit is exact, and below that the
    lagged levels' covariance is singular).
    """
    history = np.asarray(history, dtype=float)
    if history.ndim != 2:
        raise ValueError("a history is rows of times by columns of books")
    n = history.shape[1]
    if count < 1:
        raise ValueError("a count of 1 or more: a chooser that gives no baskets chooses nothing")
    if legs == 1:
        raise ValueError("legs of 0 (all the books) or 2 or more: a portfolio of one book is that book, not a relation")
    if legs < 0:
        raise ValueError("legs of 0 (all the books) or 2 or more")
    if legs > n:
        raise ValueError(f"legs of at most the {n} books there are")
    if every < 1:
        raise ValueError("every 1 row or more")
    a, gamma = fit_var1(history[::every], ridge=ridge)
    if legs == 0:
        return _largest_one(box_tiao(a, gamma)[1][:count])
    found: dict[tuple[int, ...], tuple[float, np.ndarray]] = {}
    for pair in itertools.combinations(range(n), 2):
        least, books, row = greedy_basket(a, gamma, pair, legs)
        found.setdefault(books, (least, row))  # another start to the same books gives the same set
    ranked = sorted(found.items(), key=lambda item: (item[1][0], item[0]))
    return _pick([(-least, books, row) for books, (least, row) in ranked], count, n)
