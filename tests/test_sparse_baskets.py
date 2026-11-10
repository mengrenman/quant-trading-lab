"""Sparse mean-reverting baskets: Box-Tiao predictability, a greedy search, and a chooser for RelationTrader
(lob/sparse_baskets.py).

Each rule is held to a computation made another way here: least squares by ``np.linalg.lstsq`` (ridge as extra rows),
generalized eigenvalues as the eigenvalues of ``inv(gamma) M`` from ``np.linalg.eigvals``, the greedy step and the
ranking by loops over every candidate, and predictability by arithmetic worked out by hand.
"""

import itertools
from functools import cache, partial

import numpy as np
import pytest
import scipy.linalg
from scipy.signal import lfilter

from lob.sparse_baskets import box_tiao, fit_var1, greedy_basket, predictability, sparse_baskets
from lob.statarb import BookMarket, FactorValues, RelationTrader, mixed_loadings, sector_loadings

VAR_A = np.array([[0.5, 0.2, 0.0], [-0.1, 0.7, 0.1], [0.0, 0.3, 0.4]])  # stable, and not symmetric


def var1_path(a, t, seed, burn=200):
    """x_t = a x_{t-1} + e_t with unit normal e_t, after a burn-in from 0."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((t + burn, len(a)))
    x = np.zeros_like(e)
    for i in range(1, t + burn):
        x[i] = a @ x[i - 1] + e[i]
    return x[burn:]


def mixed_walks(seed, t=800, n=6):
    """Levels of a stable VAR(1) with random coefficients (its largest root at most 0.8), plus a random walk each."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-0.4, 0.4, (n, n)) + 0.5 * np.eye(n)
    a *= min(1.0, 0.8 / np.abs(np.linalg.eigvals(a)).max())
    return var1_path(a, t, seed) + np.cumsum(rng.standard_normal((t, n)), axis=0) * rng.uniform(0.2, 1.0, n)


def shared_factor(seed, phis, t=3000):
    """Books that each are one random walk plus an AR(1) of their own, of coefficient phi: all pairs are relations, and
    the less phi, the better (a book of phi 0 and one of 0.3 are the best pair)."""
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.standard_normal(t))
    return np.array([f + lfilter([1.0], [1.0, -phi], rng.standard_normal(t)) for phi in phis]).T


def planted(seed, t=20_000):
    """Six books: 1, 3 and 4 hold one relation (x1 - x3 + x4 is an AR(1) of coefficient 0.3, so its predictability is
    0.09), and 0, 2 and 5 are random walks."""
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.standard_normal((t, 2)), axis=0)
    walks = np.cumsum(rng.standard_normal((t, 3)), axis=0)
    u = lfilter([1.0], [1.0, -0.3], rng.standard_normal(t))
    x = np.zeros((t, 6))
    x[:, [0, 2, 5]] = walks
    x[:, 1], x[:, 3], x[:, 4] = f[:, 0], f[:, 0] + f[:, 1], f[:, 1] + u
    return x


def moments(a, gamma):
    return a @ gamma @ a.T


def eigenvalues_by_inverse(a, gamma, books):
    """The eigenvalues of inv(gamma_SS) M_SS, ascending: the generalized problem done another way."""
    s = list(books)
    return np.sort(np.linalg.eigvals(np.linalg.inv(gamma[np.ix_(s, s)]) @ moments(a, gamma)[np.ix_(s, s)]).real)


def lagged(levels):
    """The levels less their means over all the rows, and the rows 0 to T-2 and 1 to T-1 of them."""
    x = levels - levels.mean(axis=0)
    return x[:-1], x[1:]


def system(seed, n=6):
    """A and gamma from a fit of mixed_walks."""
    return fit_var1(mixed_walks(seed, n=n))


# ---- fit_var1 ----

@pytest.mark.parametrize("seed", [1, 2])
def test_a_var_fitted_to_a_long_path_gives_back_its_coefficients_and_its_covariance(seed):
    a, gamma = fit_var1(var1_path(VAR_A, 20_000, seed))
    assert np.abs(a - VAR_A).max() < 0.03  # 0.007 to 0.015 over eight seeds: a standard error of about 0.006
    assert np.abs(a - VAR_A.T).max() > 0.2  # and not its transpose, which is where a wrong convention would land
    stationary = scipy.linalg.solve_discrete_lyapunov(VAR_A, np.eye(3))  # gamma = A gamma A' + I
    assert np.abs(gamma - stationary).max() < 0.15  # up to 0.07 over eight seeds, against entries of 0.3 to 2.1


def test_var1_is_least_squares_of_the_next_levels_on_these_less_their_means_over_all_the_rows():
    rng = np.random.default_rng(5)
    levels = np.cumsum(rng.standard_normal((500, 4)), axis=0) * [1, 3, 10, 0.5] + rng.standard_normal((500, 4))
    a, gamma = fit_var1(levels)
    x0, x1 = lagged(levels)
    assert a == pytest.approx(np.linalg.lstsq(x0, x1, rcond=None)[0].T, abs=1e-10)
    assert gamma == pytest.approx(x0.T @ x0 / 499, abs=1e-10)  # the covariance of the lagged levels, over T - 1
    shifted = fit_var1(levels + [1000.0, -3.0, 50.0, 7.0])  # a level of its own is nothing to a column
    assert shifted[0] == pytest.approx(a, abs=1e-8) and shifted[1] == pytest.approx(gamma, abs=1e-8)
    separately = np.vstack([levels[:-1] - levels[:-1].mean(axis=0), levels[1:] - levels[1:].mean(axis=0)])
    other = np.linalg.lstsq(separately[:499], separately[499:], rcond=None)[0].T
    assert np.abs(other - a).max() > 1e-5  # means over rows 0 to T-2 and 1 to T-1 each differ at order 1/T


@pytest.mark.parametrize("ridge", [0.01, 0.5, 3.0])
def test_a_ridge_is_extra_rows_of_least_squares_that_pull_the_coefficients_toward_zero(ridge):
    levels = mixed_walks(7, t=300, n=4)
    x0, x1 = lagged(levels)
    penalty = ridge * np.trace(x0.T @ x0) / 4  # a share of the average variance on the diagonal
    stacked = np.linalg.lstsq(np.vstack([x0, np.sqrt(penalty) * np.eye(4)]), np.vstack([x1, np.zeros((4, 4))]),
                              rcond=None)[0].T
    a, gamma = fit_var1(levels, ridge=ridge)
    assert a == pytest.approx(stacked, abs=1e-9)
    assert gamma == pytest.approx(fit_var1(levels)[1], abs=0)  # the covariance is not shrunk


def test_more_ridge_shrinks_the_coefficients_and_a_great_deal_leaves_none():
    levels = mixed_walks(7, t=300, n=4)
    norms = [np.linalg.norm(fit_var1(levels, ridge=r)[0]) for r in (0.0, 0.01, 0.1, 1.0, 10.0, 1e8)]
    assert all(later < earlier for earlier, later in itertools.pairwise(norms)) and norms[-1] < 1e-6


def test_a_book_that_never_moved_has_no_coefficients_and_no_variance():
    levels = mixed_walks(3, t=200, n=4)
    levels[:, 2] = 1000.0
    a, gamma = fit_var1(levels)
    assert np.isfinite(a).all() and not a[2].any() and not a[:, 2].any() and not gamma[2].any()


@pytest.mark.parametrize("make, words", [
    (lambda: fit_var1(np.zeros(10)), "rows of times"),
    (lambda: fit_var1(mixed_walks(1, t=100, n=4), ridge=-0.1), "ridge of 0 or more"),
    (lambda: fit_var1(mixed_walks(1, t=5, n=4)), "at least 6 rows for a VAR\\(1\\) on 4 books, and there are 5"),
])
def test_a_var_needs_rows_enough_for_its_covariance_and_a_ridge_that_shrinks(make, words):
    with pytest.raises(ValueError, match=words):
        make()
    assert fit_var1(mixed_walks(1, t=6, n=4))[1].shape == (4, 4)  # n + 2 rows will do


# ---- predictability ----

@pytest.mark.parametrize("w, a, gamma, expected", [
    ([1.0, -1.0], [[0.5, 0.2], [0.1, 0.4]], [[2.0, 0.5], [0.5, 1.0]], 0.14),  # w'A gamma A'w = 0.28, w'gamma w = 2
    ([1.0, 0.0], np.diag([0.9, 0.3]), np.eye(2), 0.81),  # an AR(1) of 0.9
    ([0.0, 1.0], np.diag([0.9, 0.3]), np.diag([5.0, 2.0]), 0.09),  # the variance of a book does not enter
    ([1.0, 1.0], np.diag([0.9, 0.3]), np.eye(2), (0.81 + 0.09) / 2),  # two independent books: the average
    ([3.0, 4.0], np.eye(2), [[2.0, 0.7], [0.7, 1.0]], 1.0),  # a random walk
    ([3.0, 4.0], np.zeros((2, 2)), [[2.0, 0.7], [0.7, 1.0]], 0.0),  # noise
    ([1.0, 0.0], [[0.0, 1.0], [0.0, 0.0]], np.eye(2), 1.0),  # book 0 is the last of book 1: its own past says nothing
    ([0.0, 1.0], [[0.0, 1.0], [0.0, 0.0]], np.eye(2), 0.0),  # but the past of the books says all
])
def test_predictability_by_hand(w, a, gamma, expected):
    assert predictability(np.array(w), np.array(a), np.array(gamma)) == pytest.approx(expected, abs=1e-12)


def test_predictability_does_not_depend_on_the_scale_or_sign_of_the_portfolio():
    a, gamma = system(2)
    w = np.array([1.0, -0.5, 0.0, 2.0, 0.0, 0.3])
    assert predictability(w, a, gamma) > 0
    for scale in (1e-6, 0.37, -1.0, 50.0):
        assert predictability(scale * w, a, gamma) == pytest.approx(predictability(w, a, gamma), rel=1e-9)


def test_predictability_of_simulated_series_is_phi_squared_for_an_ar1_and_near_zero_for_noise():
    ar = var1_path(np.array([[0.9]]), 20_000, 1)
    a, gamma = fit_var1(ar)
    assert predictability(np.array([1.0]), a, gamma) == pytest.approx(0.81, abs=0.03)  # 0.804 to 0.814 over four seeds
    noise = np.random.default_rng(1).standard_normal((20_000, 3))
    a, gamma = fit_var1(noise)
    assert max(predictability(row, a, gamma) for row in np.eye(3)) < 0.01
    walk = np.cumsum(np.random.default_rng(1).standard_normal((20_000, 1)), axis=0)
    a, gamma = fit_var1(walk)
    assert predictability(np.array([1.0]), a, gamma) > 0.99


def test_a_portfolio_with_no_variance_has_no_predictability():
    with pytest.raises(ValueError, match="some variance"):
        predictability(np.array([1.0, 1.0]), np.eye(2), np.array([[1.0, -1.0], [-1.0, 1.0]]))


# ---- box_tiao ----

def test_box_tiao_by_hand_on_two_books():
    c = 0.25
    values, rows = box_tiao(c * np.ones((2, 2)), np.eye(2))  # M = 2 c^2 [[1, 1], [1, 1]]: roots 0 and 4 c^2
    assert values == pytest.approx([0.0, 4 * c * c], abs=1e-12)
    assert np.abs(rows[0] @ [1, 1]) < 1e-12 and abs(abs(rows[0] @ [1, -1]) - np.sqrt(2)) < 1e-12  # (1, -1) / sqrt 2
    assert abs(rows[1] @ [1, -1]) < 1e-12 and abs(abs(rows[1] @ [1, 1]) - np.sqrt(2)) < 1e-12  # (1, 1) / sqrt 2
    values, rows = box_tiao(np.diag([0.8, 0.3]), np.diag([4.0, 9.0]))  # roots phi^2; a book scaled by its sd
    assert values == pytest.approx([0.09, 0.64], abs=1e-12)
    assert np.abs(rows) == pytest.approx(np.array([[0, 1 / 3], [1 / 2, 0]]), abs=1e-12)  # 1 / sqrt(gamma)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_box_tiao_rows_are_the_portfolios_ordered_by_predictability(seed):
    a, gamma = system(seed)
    values, rows = box_tiao(a, gamma)
    assert rows.shape == (6, 6) and (np.diff(values) >= 0).all() and values[0] >= -1e-12  # M is positive semidefinite
    assert values == pytest.approx(eigenvalues_by_inverse(a, gamma, range(6)), abs=1e-9)
    for value, row in zip(values, rows, strict=True):
        assert predictability(row, a, gamma) == pytest.approx(value, abs=1e-9)
        assert moments(a, gamma) @ row == pytest.approx(value * gamma @ row, abs=1e-9)  # M w = lambda gamma w
    assert rows @ gamma @ rows.T == pytest.approx(np.eye(6), abs=1e-9)  # gamma-orthonormal
    rng = np.random.default_rng(seed)
    assert all(predictability(w, a, gamma) >= values[0] - 1e-12 for w in rng.standard_normal((500, 6)))


@pytest.mark.parametrize("books", [[1, 3, 4], [4, 1, 3], (0, 5), [2]])
def test_box_tiao_on_some_books_is_the_problem_of_their_rows_and_columns_with_zero_elsewhere(books):
    a, gamma = system(4)
    values, rows = box_tiao(a, gamma, books)
    s = sorted(books)
    assert rows.shape == (len(s), 6) and values == pytest.approx(eigenvalues_by_inverse(a, gamma, s), abs=1e-9)
    assert not np.delete(rows, s, axis=1).any()  # zero outside the books
    for value, row in zip(values, rows, strict=True):
        assert predictability(row, a, gamma) == pytest.approx(value, abs=1e-9)  # over all the books, as over these
    assert rows @ gamma @ rows.T == pytest.approx(np.eye(len(s)), abs=1e-9)
    assert values[0] >= box_tiao(a, gamma)[0][0] - 1e-12  # a subset has no less than all the books
    assert box_tiao(a, gamma, s)[1] == pytest.approx(rows, abs=0)  # in whatever order the books are given


@pytest.mark.parametrize("books", [[], [1, 1], [0, 6], [-1, 2]])
def test_box_tiao_wants_distinct_books_of_the_ones_there_are(books):
    a, gamma = system(4)
    with pytest.raises(ValueError, match="distinct books among 0 to 5"):
        box_tiao(a, gamma, books)


@pytest.mark.parametrize("scale", [1.0, 1e-5, 1e4])  # the jitter is relative to the variances, whatever their size
def test_a_gamma_that_is_not_positive_definite_gets_a_little_jitter_and_the_still_book_comes_out_alone(scale):
    levels = mixed_walks(3, t=200, n=4) * scale
    levels[:, 2] = 1000.0  # book 2 never moved
    a, gamma = fit_var1(levels)
    values, rows = box_tiao(a, gamma)
    assert np.isfinite(values).all() and np.isfinite(rows).all()
    assert abs(values[0]) < 1e-9 and np.abs(rows[0, [0, 1, 3]]).max() < 1e-6 * abs(rows[0, 2])  # book 2, lambda 0
    assert values[1:] == pytest.approx(eigenvalues_by_inverse(a, gamma, [0, 1, 3]), abs=1e-9)  # the rest as they were
    # the jitter is the least that lifts the smallest eigenvalue (0) to a trillionth of the average variance: gamma
    # plus it is what the rows are orthonormal in, and book 2's row is 1 / sqrt(jitter)
    assert 1 / rows[0, 2] ** 2 == pytest.approx(1e-12 * np.trace(gamma) / 4, rel=1e-6)


def test_a_history_in_which_no_book_moved_has_rows_all_the_same_and_nothing_breaks():
    history = np.full((50, 3), 5.0)
    rows = sparse_baskets(history, count=2, legs=2, every=1)
    assert rows.shape == (2, 3) and np.isfinite(rows).all() and (np.count_nonzero(rows, axis=1) == 1).all()
    assert sparse_baskets(history, count=2, legs=0, every=1).tolist() == np.eye(3)[:2].tolist()


def test_a_gamma_with_a_negative_eigenvalue_is_lifted_to_a_trillionth_of_its_average_variance():
    gamma = np.array([[1.0, 2.0], [2.0, 1.0]])  # roots 3 and -1: not a covariance, but it must not stop a search
    a = np.array([[0.5, 0.1], [0.2, 0.3]])
    values, rows = box_tiao(a, gamma)
    jitter = 1.0 + 1e-12  # -(-1) and a trillionth of the average variance, 1
    assert np.isfinite(values).all() and (np.diff(values) >= 0).all()
    lifted = gamma + jitter * np.eye(2)  # its roots are 4 and a trillionth: the rows are good to 1e-4 or so
    assert rows @ lifted @ rows.T == pytest.approx(np.eye(2), abs=1e-3)
    assert values[0] == pytest.approx(rows[0] @ moments(a, gamma) @ rows[0], rel=1e-2)  # the root, as the quotient


# ---- greedy_basket ----

def exhaustive(a, gamma, legs, start=()):
    """(least predictability, books) of every set of legs books containing start, least first."""
    n = len(gamma)
    rest = [b for b in range(n) if b not in start]
    sets = (tuple(sorted({*start, *extra})) for extra in itertools.combinations(rest, legs - len(start)))
    return sorted((eigenvalues_by_inverse(a, gamma, s)[0], s) for s in sets)


@pytest.mark.parametrize("seed, start", [(1, (0, 1)), (2, (2, 4)), (3, (3,)), (4, (1, 5))])
def test_the_least_predictability_never_rises_as_the_greedy_search_adds_books(seed, start):
    a, gamma = system(seed)
    trail, previous = [], None
    for legs in range(len(start), 7):
        least, books, row = greedy_basket(a, gamma, start, legs)
        assert len(books) == legs and list(books) == sorted(books) and set(start) <= set(books)
        assert least == pytest.approx(eigenvalues_by_inverse(a, gamma, books)[0], abs=1e-9)
        assert predictability(row, a, gamma) == pytest.approx(least, abs=1e-9)  # the row is the lambda's portfolio
        assert np.flatnonzero(row).tolist() == list(books) and not np.delete(row, list(books)).any()
        if previous is not None:  # each step takes the best one book more: tried here one by one
            step = min((eigenvalues_by_inverse(a, gamma, sorted([*previous[1], b]))[0], b)
                       for b in range(6) if b not in previous[1])
            assert books == tuple(sorted([*previous[1], step[1]])) and least == pytest.approx(step[0], abs=1e-9)
            assert set(previous[1]) < set(books)
        trail.append(least)
        previous = (least, books)
    assert (np.diff(trail) <= 1e-9).all()  # never rises
    assert box_tiao(a, gamma)[0][0] == pytest.approx(trail[-1], abs=1e-9)  # and with every book, the least there is


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_in_a_planted_case_greedy_from_a_pair_in_the_relation_finds_it_and_matches_the_exhaustive_search(seed):
    a, gamma = fit_var1(planted(seed))
    others = exhaustive(a, gamma, 3)
    assert others[0][1] == (1, 3, 4) and 0.05 < others[0][0] < 0.13  # 0.09, the AR(1)'s phi squared
    assert others[1][0] > 0.99  # and every other triple is as near a random walk as can be
    for pair in itertools.combinations((1, 3, 4), 2):
        least, books, row = greedy_basket(a, gamma, pair, 3)
        assert books == (1, 3, 4) and least == pytest.approx(others[0][0], abs=1e-9)
        assert row[[1, 3, 4]] / np.abs(row).max() * np.sign(row[1]) == pytest.approx([1.0, -1.0, 1.0], abs=0.02)
        least4, books4, _ = greedy_basket(a, gamma, pair, 4)  # one book more: the exhaustive search of the four there
        best = exhaustive(a, gamma, 4, pair)[0]  # are, which has the relation and one more
        assert books4 == best[1] and least4 == pytest.approx(best[0], abs=1e-9) and set(books) < set(books4)


def test_greedy_from_a_pair_with_one_book_outside_the_relation_does_not_reach_it():
    # The known limit of a greedy search (d'Aspremont 2011, Section 3.1). From books 0 and 1, which are not related, any
    # book added gives a least predictability of 0.99 or more, so the one that wins by a little (4, then 5) takes the
    # search away from the optimum, books 0, 1, 3 and 4 at 0.09. Of the seeds 1 to 8, seed 2 alone has no such pair.
    a, gamma = fit_var1(planted(1))
    assert exhaustive(a, gamma, 3, (0, 1))[0][0] > 0.99  # the first step has nothing to go on
    least, books, _ = greedy_basket(a, gamma, (0, 1), 4)
    best = exhaustive(a, gamma, 4, (0, 1))[0]
    assert best[1] == (0, 1, 3, 4) and best[0] < 0.13
    assert books != best[1] and least > 0.99


def test_a_tie_goes_to_the_lowest_index():
    n = 5
    zero = greedy_basket(np.zeros((n, n)), np.eye(n), [4], 3)  # every book predicts nothing of any
    assert zero[:2] == (0.0, (0, 1, 4))
    same = greedy_basket(0.5 * np.eye(n), np.eye(n), [0, 1], 4)  # every book alike (0.25), exactly
    assert same[:2] == (0.25, (0, 1, 2, 3))
    assert greedy_basket(0.5 * np.eye(n), np.eye(n), [3, 1], 3)[1] == (0, 1, 3)  # and in whatever order it starts


def test_a_start_that_is_all_the_legs_is_the_least_predictable_portfolio_of_those_books():
    a, gamma = system(5)
    least, books, row = greedy_basket(a, gamma, [4, 0, 2], 3)
    values, rows = box_tiao(a, gamma, [0, 2, 4])
    assert (least, books) == (values[0], (0, 2, 4)) and row == pytest.approx(rows[0], abs=0)


@pytest.mark.parametrize("start, legs", [([], 3), ([1, 1], 3), ([0, 9], 3), ([0, 1, 2], 2), ([0, 1], 7)])
def test_greedy_wants_distinct_books_to_start_from_and_legs_between_them_and_all_the_books(start, legs):
    a, gamma = system(5)
    with pytest.raises(ValueError):
        greedy_basket(a, gamma, start, legs)


# ---- sparse_baskets ----

def independent_fit(levels, ridge=0.0):
    x0, x1 = lagged(levels)
    penalty = ridge * np.trace(x0.T @ x0) / x0.shape[1]
    n = x0.shape[1]  # a ridge is extra rows: the penalty's root on the diagonal, and zeros to go with them
    rows, targets = np.vstack([x0, np.sqrt(penalty) * np.eye(n)]), np.vstack([x1, np.zeros((n, n))])
    return np.linalg.lstsq(rows, targets, rcond=None)[0].T, x0.T @ x0 / len(x0)


def smallest(a, gamma, books):
    """(least predictability, its portfolio with the largest leg +1) of these books, from the eigenvalues of
    inv(gamma_SS) M_SS."""
    s = list(books)
    values, vectors = np.linalg.eig(np.linalg.inv(gamma[np.ix_(s, s)]) @ moments(a, gamma)[np.ix_(s, s)])
    k = int(np.argmin(values.real))
    row = np.zeros(len(gamma))
    row[s] = vectors[:, k].real
    return float(values[k].real), row / row[np.argmax(np.abs(row))]


def reference(history, count, legs, every=1, ridge=0.0):
    """Every pair grown by one book at a time, the distinct sets ranked by least predictability, the first count with
    those that share no book first: the books of each and its portfolio, done with plain loops."""
    a, gamma = independent_fit(history[::every], ridge)
    n = history.shape[1]
    found = {}
    for pair in itertools.combinations(range(n), 2):
        books = list(pair)
        while len(books) < legs:
            books = min((sorted([*books, b]) for b in range(n) if b not in books),
                        key=lambda grown: (smallest(a, gamma, grown)[0], sorted(set(grown) - set(books))))
        found[tuple(books)] = smallest(a, gamma, books)
    ranked = sorted(found.items(), key=lambda item: (item[1][0], item[0]))
    chosen, used = [], set()
    for books, _ in ranked:
        if len(chosen) < count and not used & set(books):
            chosen.append(books)
            used |= set(books)
    chosen += [books for books, _ in ranked if books not in chosen][:count - len(chosen)]
    return [(books, found[books][1], found[books][0]) for books in chosen]


def books_of(rows):
    return [tuple(int(i) for i in np.flatnonzero(row)) for row in rows]


def scaled(row):
    """With the largest leg +1: a portfolio and its negative are the same."""
    return row / row[np.argmax(np.abs(row))]


@pytest.mark.parametrize("seed, count, legs, every", [
    (1, 1, 2, 1), (2, 3, 2, 1), (3, 4, 3, 1), (4, 2, 3, 3), (5, 3, 4, 1), (6, 5, 4, 2), (7, 20, 3, 1), (8, 6, 5, 1),
])
def test_sparse_baskets_is_the_greedy_search_from_every_pair_with_the_distinct_sets_ranked_and_picked(
        seed, count, legs, every):
    history = mixed_walks(seed, t=900)
    rows = sparse_baskets(history, count=count, legs=legs, every=every)
    expected = reference(history, count, legs, every)
    assert rows.shape == (len(expected), 6)
    assert books_of(rows) == [books for books, _, _ in expected]  # the same sets, in the same order
    for row, (_, portfolio, _) in zip(rows, expected, strict=True):
        assert scaled(row) == pytest.approx(portfolio, abs=1e-6)
    assert np.abs(rows).max(axis=1) == pytest.approx(1.0, abs=1e-12)  # the largest leg is 1 (or -1)
    assert (np.count_nonzero(rows, axis=1) == legs).all()  # a moving history: every leg is there
    a, gamma = independent_fit(history[::every])
    assert sorted(predictability(row, a, gamma) for row in rows) == pytest.approx(
        sorted(value for _, _, value in expected), abs=1e-6)


def test_the_baskets_sharing_no_book_with_a_better_one_come_first_and_then_any_by_rank():
    history = shared_factor(3, [0.0, 0.3, 0.6, 0.8, 0.9])  # five books of one factor: every pair is a relation
    a, gamma = independent_fit(history)
    ranked = sorted((smallest(a, gamma, pair)[0], pair) for pair in itertools.combinations(range(5), 2))
    assert [pair for _, pair in ranked[:3]] == [(0, 1), (0, 2), (1, 2)]  # the best three all share a book
    assert books_of(sparse_baskets(history, count=1, legs=2, every=1)) == [(0, 1)]
    rows = sparse_baskets(history, count=3, legs=2, every=1)
    assert books_of(rows) == [(0, 1), (2, 3), (0, 2)]  # (2, 3) is sixth by rank, the best sharing no book with (0, 1)
    assert books_of(sparse_baskets(history, count=2, legs=2, every=1)) == [(0, 1), (2, 3)]
    assert books_of(sparse_baskets(history, count=10, legs=2, every=1))[:3] == [(0, 1), (2, 3), (0, 2)]
    assert [pair for pair, _, _ in reference(history, 3, 2)] == [(0, 1), (2, 3), (0, 2)]  # as the loops have it


def test_sets_with_the_same_least_predictability_rank_by_their_books_and_then_are_picked(monkeypatch):
    def tied(a, gamma, start, legs):
        row = np.zeros(len(gamma))
        row[list(start)] = 1.0
        return 0.5, tuple(start), row

    monkeypatch.setattr("lob.sparse_baskets.greedy_basket", tied)  # every pair has the same lambda
    rows = sparse_baskets(mixed_walks(1, t=100, n=4), count=6, legs=2, every=1)
    # ranked (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3); the first not sharing a book with (0, 1) is (2, 3), and
    # then the rest in rank
    assert books_of(rows) == [(0, 1), (2, 3), (0, 2), (0, 3), (1, 2), (1, 3)]

    def leaving_out_the_lower(a, gamma, start, legs):  # the three books but the lower of the pair: found in the
        books = tuple(b for b in range(4) if b != min(start))  # order (1, 2, 3), (0, 2, 3), (0, 1, 3)
        row = np.zeros(4)
        row[list(books)] = 1.0
        return 0.5, books, row

    monkeypatch.setattr("lob.sparse_baskets.greedy_basket", leaving_out_the_lower)
    rows = sparse_baskets(mixed_walks(1, t=100, n=4), count=3, legs=3, every=1)
    assert books_of(rows) == [(0, 1, 3), (0, 2, 3), (1, 2, 3)]  # by their books, not by when they were found


def test_a_pair_of_the_same_factor_is_hedged_one_for_one_and_the_best_pair_is_the_one_with_least_of_its_own():
    row = sparse_baskets(shared_factor(3, [0.0, 0.3, 0.6, 0.8, 0.9]), count=1, legs=2, every=1)[0]
    assert np.abs(row[:2]) == pytest.approx([1.0, 1.0], abs=0.05) and row[0] * row[1] < 0 and not row[2:].any()


def test_with_no_legs_given_it_is_the_least_predictable_portfolios_of_all_the_books():
    history = mixed_walks(2, t=900)
    a, gamma = independent_fit(history)
    values = eigenvalues_by_inverse(a, gamma, range(6))
    rows = sparse_baskets(history, count=3, legs=0, every=1)
    assert rows.shape == (3, 6) and np.count_nonzero(rows, axis=1).tolist() == [6, 6, 6]  # dense
    assert np.abs(rows).max(axis=1) == pytest.approx(1.0, abs=1e-12)
    assert [predictability(row, a, gamma) for row in rows] == pytest.approx(values[:3], abs=1e-6)  # in order
    for k, row in enumerate(rows):  # the k-th smallest root's vector, from the eigenvectors of inv(gamma) M
        roots, vectors = np.linalg.eig(np.linalg.inv(gamma) @ moments(a, gamma))
        assert scaled(row) == pytest.approx(scaled(vectors[:, np.argsort(roots.real)[k]].real), abs=1e-6)
    assert sparse_baskets(history, count=99, legs=0, every=1).shape == (6, 6)  # no more than there are books
    thin = sparse_baskets(history, count=2, legs=0, every=4)
    a, gamma = independent_fit(history[::4])
    roots = eigenvalues_by_inverse(a, gamma, range(6))
    assert [predictability(row, a, gamma) for row in thin] == pytest.approx(roots[:2])


def test_called_with_a_history_alone_it_gives_one_basket_of_three_books_from_every_fifth_row():
    history = mixed_walks(2, t=900)
    rows = sparse_baskets(history)
    assert rows.shape == (1, 6) and np.count_nonzero(rows) == 3
    assert rows == pytest.approx(sparse_baskets(history, count=1, legs=3, every=5, ridge=0.0), abs=0)
    for every in (1, 10):
        assert not np.array_equal(rows, sparse_baskets(history, count=1, legs=3, every=every))


def test_every_samples_the_history_and_a_ridge_shrinks_the_fit():
    history = mixed_walks(4, t=900)
    assert sparse_baskets(history, count=3, legs=3, every=3) == pytest.approx(
        sparse_baskets(history[::3], count=3, legs=3, every=1), abs=0)  # one row in every
    assert not np.array_equal(sparse_baskets(history, count=3, legs=3, every=3),
                              sparse_baskets(history, count=3, legs=3, every=1))
    shrunk = sparse_baskets(history, count=3, legs=3, every=1, ridge=0.5)
    assert books_of(shrunk) == [books for books, _, _ in reference(history, 3, 3, every=1, ridge=0.5)]
    assert books_of(shrunk) != books_of(sparse_baskets(history, count=3, legs=3, every=1))


def test_it_gives_what_it_can_when_the_sets_run_out_and_never_none_for_two_legs_or_more():
    history = mixed_walks(5, t=300, n=4)
    rows = sparse_baskets(history, count=10, legs=3, every=1)  # four sets of three books at most
    assert 1 <= len(rows) <= 4 and len(set(books_of(rows))) == len(rows)
    assert books_of(rows) == [books for books, _, _ in reference(history, 10, 3)]
    assert sparse_baskets(mixed_walks(5, t=300, n=2), count=5, legs=2, every=1).shape == (1, 2)  # one pair
    assert sparse_baskets(history, count=1, legs=4, every=1).shape == (1, 4)  # all four books: one set
    assert sparse_baskets(history, count=1, legs=0, every=1).shape == (1, 4)


def test_it_is_deterministic_and_leaves_the_history_alone():
    history = mixed_walks(6, t=500)
    kept = history.copy()
    first = sparse_baskets(history, count=4, legs=3, every=2)
    assert np.array_equal(first, sparse_baskets(history, count=4, legs=3, every=2)) and np.array_equal(history, kept)
    assert np.array_equal(sparse_baskets(history.tolist(), count=4, legs=3, every=2), first)  # lists will do


def test_a_book_that_never_moved_comes_out_alone_and_nothing_breaks():
    history = mixed_walks(3, t=300, n=5)
    history[:, 2] = 1000.0
    rows = sparse_baskets(history, count=2, legs=3, every=1)
    assert np.isfinite(rows).all() and abs(rows[0, 2]) == 1  # lambda 0: that book alone, whatever legs says
    assert np.abs(np.delete(rows[0], 2)).max() < 1e-12  # (and the same again: every set with it has lambda 0)


@pytest.mark.parametrize("change, words", [
    ({"count": 0}, "count of 1 or more"), ({"count": -2}, "count of 1 or more"),
    ({"legs": 1}, "a portfolio of one book is that book"), ({"legs": -1}, "legs of 0 \\(all the books\\) or 2"),
    ({"legs": 7}, "at most the 6 books there are"), ({"every": 0}, "every 1 row or more"),
    ({"every": 200}, "at least 8 rows for a VAR\\(1\\) on 6 books, and there are 5"),
    ({"ridge": -1.0}, "ridge of 0 or more"),
])
def test_sparse_baskets_refuses_what_cannot_be_asked(change, words):
    with pytest.raises(ValueError, match=words):
        sparse_baskets(mixed_walks(1, t=1000), **{"count": 2, "legs": 3, "every": 1, **change})
    with pytest.raises(ValueError, match="rows of times"):
        sparse_baskets(np.zeros(100))
    n = 6  # n + 2 rows will do, n + 1 will not
    assert sparse_baskets(mixed_walks(1, t=n + 2), count=1, legs=2, every=1).shape == (1, n)


# ---- in a market ----

@cache
def market_mids(shape, seed):
    """2,000 rows of a BookMarket's mids, one a unit of time from 200 on (every book quoted), and its loadings."""
    loadings = sector_loadings(3, 4) if shape == "sectors" else mixed_loadings(12, 3, seed=3)
    market = BookMarket(FactorValues(loadings), seed=seed)
    rows, next_sample = [], 1.0
    while market.time < 2200.0:
        market.step()
        while market.time >= next_sample:
            if next_sample >= 200.0:
                rows.append(market.marks())
            next_sample += 1.0
    return np.array(rows[:2000]), loadings


def leftover(row, loadings):
    """How much of a portfolio is exposed to the factors, for each unit of it: ||w'B|| / ||w||."""
    return float(np.linalg.norm(row @ loadings) / np.linalg.norm(row))


@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("every", [1, 5])
def test_in_a_market_of_sectors_every_pair_it_chooses_is_within_a_sector_and_hedged_about_one_for_one(seed, every):
    mids, loadings = market_mids("sectors", seed)
    rows = sparse_baskets(mids, count=6, legs=2, every=every)  # 18 pairs are within a sector, and 6 can share no book
    assert rows.shape == (6, 12) and len({b for books in books_of(rows) for b in books}) == 12
    for row, (x, y) in zip(rows, books_of(rows), strict=True):
        assert x // 4 == y // 4  # the same sector
        assert row[x] * row[y] < 0 and 0.8 <= min(abs(row[x]), abs(row[y])) <= 1.0  # 0.84 to 1.0 over 8 cases
        assert leftover(row, loadings) < 0.5  # the pair's true hedge is 1:1; the factors are about gone


@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("every", [1, 5])
def test_in_a_market_with_no_pair_related_four_books_each_leave_the_factors_mostly_out(seed, every):
    # Random weights on four books leave a median of 1.04 of each unit in the factors, 0.47 at the 10th percentile,
    # 0.36 at the 5th, 0.21 at the 1st (20,000 draws). Every four books have a relation, so the Box-Tiao vectors of
    # other sets of four would be as good: the comparison is with random weights. The rows found on real mids over
    # seeds 1, 2, 5 and 6, both samplings, ranged from 0.03 to 0.32 at legs 3 and 4, and from 0.04 to 0.29 at legs 4.
    mids, loadings = market_mids("mixed", seed)
    rows = sparse_baskets(mids, count=3, legs=4, every=every)
    exposures = [leftover(row, loadings) for row in rows]
    assert rows.shape == (3, 12) and (np.count_nonzero(rows, axis=1) == 4).all()
    assert max(exposures) < 0.5 and np.mean(exposures) < 0.3
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(2000):
        w = np.zeros(12)
        w[rng.choice(12, 4, replace=False)] = rng.standard_normal(4)
        draws.append(leftover(w, loadings))
    assert 0.9 < np.median(draws) < 1.2 and np.percentile(draws, 10) > 0.4  # so that 0.5 is no great bar


def test_a_relation_trader_takes_the_chooser_trades_its_baskets_and_its_account_adds_up():
    trader = RelationTrader(choose=partial(sparse_baskets, count=2, legs=3, every=1), formation=300, window=50)
    market = BookMarket(FactorValues(sector_loadings(3, 4)), seed=2, traders=(trader,))
    while market.time < 2500.0 and not trader.lots:
        market.step()
    assert trader.fits == 1 and trader.weights.shape == (2, 12) and trader.lots > 0
    assert (np.count_nonzero(trader.weights, axis=1) == 3).all()
    assert np.abs(trader.weights).max(axis=1).tolist() == [1, 1]
    assert trader.value(market.marks()) == pytest.approx(trader.cash + float(trader.position @ market.marks()))
    assert trader.spread_paid >= 0 and trader.lots >= np.abs(trader.position).sum()
