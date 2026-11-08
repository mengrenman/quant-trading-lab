"""Cross-sectional long/short traders over many books (lob/crosssection.py), and the flow BookMarket keeps for them.

Each rule is held to a computation made here from the spec and not from the code: a projection by QR, the PCA written
out, a ridge solved with np.linalg.solve, the contrarian profit as a loop over the pairs, and hand-made histories whose
numbers can be read off. ``worked_run`` is the whole of a reversal, momentum or lead-lag trader written out again.
"""

import dataclasses
from functools import cache
from types import SimpleNamespace

import numpy as np
import pytest

from lob.crosssection import (CrossSectionTrader, FlowTrader, LeadLagTrader, MomentumTrader, ReversalTrader,
                              contrarian_decomposition, estimate_factors, neutralize)
from lob.order import Side, cancel_order, limit_order, market_order
from lob.statarb import WAIT, BookMarket, BookTrader, FactorValues, ResidualTrader, sector_loadings
from tests.test_statarb import CURRENT, SETTING_IDS, SETTINGS, aim_by_hand, assert_as_before, played, toward_by_hand

SECTORS = sector_loadings(2, 3)  # six books


def mids_of(steps, books=6, factors=2, seed=1):
    """A seeded history of mids, steps by books: factors that every book loads on, and a walk of each book's own."""
    rng = np.random.default_rng(seed)
    loadings = rng.uniform(0.4, 1.6, (books, factors))
    common = rng.standard_normal((steps, factors)).cumsum(axis=0) @ loadings.T
    own = rng.standard_normal((steps, books)).cumsum(axis=0) * 0.5
    return 3000.0 + common + own


def feed(trader, history):
    """Hand a trader the mids one decision at a time, and carry out each move it makes in full. What it returned."""
    trader.attach(history.shape[1])
    out = []
    for mids in history:
        moves = trader.decide(mids.copy())
        out.append(moves)
        for move in moves:
            trader.position = trader.position + move
    return out


# ---- neutralize ----

def project_out(w, betas=None):
    """``w`` less its projection on the ones and the rows of ``betas``, by QR and not by least squares."""
    rows = np.ones((1, len(w))) if betas is None else np.vstack([np.ones(len(w)), betas])
    q, _ = np.linalg.qr(rows.T)
    return w - q @ (q.T @ w)


@pytest.mark.parametrize("books, factors", [(9, 1), (9, 3), (5, 2), (4, 1)])
def test_a_neutralized_weight_is_orthogonal_to_the_ones_and_to_every_beta_and_is_the_qr_projection(books, factors):
    rng = np.random.default_rng(books * 10 + factors)
    w, betas = rng.standard_normal(books), rng.standard_normal((factors, books))
    out = neutralize(w, betas)
    assert out.shape == (books,)
    assert abs(out.sum()) < 1e-12
    assert np.abs(betas @ out).max() < 1e-12
    assert np.allclose(out, project_out(w, betas), atol=1e-12)
    assert np.allclose(neutralize(out, betas), out, atol=1e-12)  # idempotent: nothing is left to take


def test_without_betas_a_weight_is_neutralized_to_its_mean():
    w = np.array([3.0, 1.0, 2.0, 0.0])
    assert np.array_equal(neutralize(w), [1.5, -0.5, 0.5, -1.5])
    assert np.array_equal(neutralize(w, None), [1.5, -0.5, 0.5, -1.5])
    assert np.array_equal(neutralize(w, np.zeros((0, 4))), [1.5, -0.5, 0.5, -1.5])
    assert np.allclose(neutralize(w), project_out(w))


def test_neutralizing_by_hand_in_two_blocks():
    # The ones and a beta of [1, 1, 0, 0] span "each block of two has its own mean": each block is taken less its mean.
    out = neutralize(np.array([3.0, 1.0, 2.0, 0.0]), np.array([[1.0, 1.0, 0.0, 0.0]]))
    assert np.allclose(out, [1.0, -1.0, 1.0, -1.0], atol=1e-12)


def test_a_weight_the_betas_cannot_see_stays_and_one_they_span_goes():
    betas = np.array([[0.0, 0.0, 1.0, -1.0], [1.0, 2.0, 3.0, 4.0]])
    unseen = project_out(np.array([1.0, -1.0, 0.0, 0.0]), betas)  # orthogonal to the ones and to both rows
    assert abs(unseen.sum()) < 1e-12 and np.abs(betas @ unseen).max() < 1e-12 and np.abs(unseen).max() > 0.1
    assert np.allclose(neutralize(unseen, betas), unseen, atol=1e-12)
    spanned = 3.0 + 2.0 * betas[0] - 0.5 * betas[1]
    assert np.allclose(neutralize(spanned, betas), 0.0, atol=1e-12)


def test_betas_that_repeat_the_ones_or_each_other_leave_the_same_answer_as_one_of_them():
    rng = np.random.default_rng(3)
    w, beta = rng.standard_normal(7), rng.standard_normal((1, 7))
    base = neutralize(w, beta)
    assert np.allclose(neutralize(w, np.vstack([beta, 2.0 * beta])), base, atol=1e-12)
    assert np.allclose(neutralize(w, np.vstack([beta, np.ones((1, 7))])), base, atol=1e-12)
    assert np.allclose(neutralize(w, np.ones((1, 7))), w - w.mean(), atol=1e-12)


def test_neutralize_leaves_what_it_is_given_and_gives_floats():
    w, betas = np.array([3, 1, 2, 0]), np.array([[1, 1, 0, 0]])
    before = w.copy(), betas.copy()
    out = neutralize(w, betas)
    assert np.array_equal(w, before[0]) and np.array_equal(betas, before[1])
    assert out.dtype == float and np.array_equal(neutralize(w).round(12), [1.5, -0.5, 0.5, -1.5])
    assert np.array_equal(neutralize([3, 1, 2, 0]), [1.5, -0.5, 0.5, -1.5])  # a list is as good as an array


# ---- estimate_factors ----

def pca_by_hand(moves, factors):
    """The eigenportfolios (books by factors) and exposures (factors by books), from the correlation matrix."""
    sd = moves.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    z = (moves - moves.mean(axis=0)) / sd
    values, vectors = np.linalg.eigh(z.T @ z / len(moves))
    eigen = vectors[:, ::-1][:, :factors] / sd[:, None]
    held = moves @ eigen
    held_c, moves_c = held - held.mean(axis=0), moves - moves.mean(axis=0)
    return eigen, np.linalg.solve(held_c.T @ held_c, held_c.T @ moves_c)


@pytest.mark.parametrize("factors", [1, 2, 3])
@pytest.mark.parametrize("constant_book", [False, True])
def test_the_factors_are_found_as_residual_trader_finds_them(factors, constant_book):
    moves = np.diff(mids_of(401, books=7, seed=5), axis=0)
    if constant_book:
        moves[:, 2] = 0.5  # never moves but for its drift: a standard deviation of 0
    eigen, betas = estimate_factors(moves, factors)
    assert eigen.shape == (7, factors) and betas.shape == (factors, 7)
    trader = ResidualTrader(factors=factors, window=len(moves), pca_window=len(moves))
    trader._pca(moves)
    _, _, residual_betas = trader.s_scores(moves)
    assert np.abs(eigen - trader._eigen).max() < 1e-12
    assert np.abs(betas - residual_betas).max() < 1e-12
    by_hand = pca_by_hand(moves, factors)
    assert np.abs(eigen - by_hand[0]).max() < 1e-10 and np.abs(betas - by_hand[1]).max() < 1e-10


def test_the_eigenportfolios_are_the_largest_eigenvectors_of_the_correlation_over_the_standard_deviations():
    moves = np.diff(mids_of(301, books=6, seed=2), axis=0)
    eigen, betas = estimate_factors(moves, 2)
    sd = moves.std(axis=0)
    corr = np.corrcoef(moves.T)
    values = np.linalg.eigvalsh(corr)[::-1]
    for j in range(2):
        vector = eigen[:, j] * sd  # back from lots to the eigenvector of the correlation matrix
        assert np.linalg.norm(vector) == pytest.approx(1.0)
        assert np.allclose(corr @ vector, values[j] * vector, atol=1e-10)  # the j-th largest: they come in order
    # the exposures are a regression: what is left of each book is uncorrelated with the factors, in the sample
    held = moves @ eigen
    left = (moves - moves.mean(axis=0)) - (held - held.mean(axis=0)) @ betas
    assert np.abs((held - held.mean(axis=0)).T @ left).max() < 1e-8
    assert np.abs(left.mean(axis=0)).max() < 1e-10


def test_one_factor_of_books_that_move_together_is_the_market():
    rng = np.random.default_rng(8)
    market = rng.standard_normal((600, 1))
    moves = market @ np.array([[1.0, 2.0, 0.5, 1.5]]) + 0.05 * rng.standard_normal((600, 4))
    eigen, betas = estimate_factors(moves, 1)
    held = moves @ eigen
    assert abs(np.corrcoef(held[:, 0], market[:, 0])[0, 1]) > 0.99
    assert np.allclose(betas[0] / betas[0][0], [1.0, 2.0, 0.5, 1.5], atol=0.05)  # the books' loadings, up to scale


# ---- A trader written out again ----

CLASSES = {"reversal": ReversalTrader, "momentum": MomentumTrader, "leadlag": LeadLagTrader}


def worked_fit(moves, factors, ridge, kind):
    """What a trader re-estimates from the last ``pca_window`` moves: the factors, the mean moves and gamma."""
    fit = {}
    if factors:
        eigen, betas = pca_by_hand(moves, factors)
        fit |= {"eigen": eigen, "betas": betas, "mean": moves.mean(axis=0)}
    if kind == "leadlag":
        before, after = moves[:-1], moves[1:]
        before, after = before - before.mean(axis=0), after - after.mean(axis=0)
        n = moves.shape[1]
        gram = before.T @ before
        beta = np.linalg.solve(gram + ridge * np.trace(gram) / n * np.eye(n), before.T @ after)
        gamma = beta.T  # row j: book j's next move on all the books' last
        fit["gamma"] = gamma - np.diag(np.diag(gamma))
    return fit


def worked_scores(kind, history, t, fit, lookback, skip):
    """Each book's score at decision ``t`` (an index into ``history``), from the mids up to it."""
    if kind == "leadlag":
        return fit["gamma"] @ (history[t] - history[t - 1])
    last = t if kind == "reversal" else t - skip
    block = np.diff(history[last - lookback:last + 1], axis=0)  # the moves from the mid ``lookback`` before to that
    if "eigen" in fit:
        centered = block - fit["mean"]
        block = centered - (block @ fit["eigen"] - fit["mean"] @ fit["eigen"]) @ fit["betas"]
    return -block.sum(axis=0) if kind == "reversal" else block.sum(axis=0)


def worked_run(kind, history, lookback, hold=1, factors=1, pca_window=500, pca_every=20, size=5.0, skip=0, ridge=1.0):
    """Each decision's (scores, betas, target), or None where it does not rebalance, from the spec alone.

    The first decision it can make is the one at which it has the mids that the scores and the factors need. From
    there it re-estimates every ``pca_every`` decisions and rebalances every ``hold``.
    """
    windowed = factors > 0 or kind == "leadlag"
    needed = 1 + (pca_window if windowed else lookback + skip)
    out, fit = [], {}
    for t in range(len(history)):
        k = t - (needed - 1)
        if k < 0:
            out.append(None)
            continue
        if k % pca_every == 0 and windowed:
            fit = worked_fit(np.diff(history[t - pca_window:t + 1], axis=0), factors, ridge, kind)
        if k % hold:
            out.append(None)
            continue
        scores = worked_scores(kind, history, t, fit, lookback, skip)
        betas = fit["betas"] if factors else None
        w = project_out(scores, betas)
        target = np.rint(size * w / np.abs(w).max()).astype(int)
        out.append((scores, betas, target))
    return out


CASES = [
    ("reversal", dict(lookback=4, factors=0)),
    ("reversal", dict(lookback=4, factors=1, pca_window=30, pca_every=1)),
    ("reversal", dict(lookback=6, hold=3, factors=2, pca_window=40, pca_every=4, size=7.0)),
    ("momentum", dict(lookback=5, skip=2, factors=0, hold=2)),
    ("momentum", dict(lookback=5, skip=3, factors=2, pca_window=35, pca_every=1)),
    ("momentum", dict(lookback=4, skip=0, factors=1, pca_window=25, pca_every=5, hold=2, size=9.0)),
    ("leadlag", dict(lookback=3, factors=0, pca_window=30, pca_every=1, ridge=0.0)),
    ("leadlag", dict(lookback=3, factors=1, pca_window=30, pca_every=1, ridge=1.0)),
    ("leadlag", dict(lookback=3, factors=2, pca_window=40, pca_every=7, hold=3, ridge=2.5)),
]
IDS = ["reversal raw", "reversal 1 factor", "reversal hold 3 refit 4", "momentum raw skip 2 hold 2",
       "momentum skip 3", "momentum refit 5 hold 2", "lead-lag ridge 0 raw", "lead-lag 1 factor",
       "lead-lag ridge 2.5 refit 7 hold 3"]


@pytest.mark.parametrize("kind, params", CASES, ids=IDS)
def test_a_trader_scores_and_holds_what_the_spec_says_at_every_decision(kind, params):
    history = mids_of(100, seed=11)
    expected = worked_run(kind, history, **params)
    trader = CLASSES[kind](**params)
    trader.attach(6)
    position = np.zeros(6, dtype=int)
    rebalanced = 0
    for t, mids in enumerate(history):
        moves = trader.decide(mids.copy())
        if expected[t] is None:
            assert moves == [], f"decision {t}"
            continue
        scores, _, target = expected[t]
        rebalanced += 1
        assert np.allclose(trader.signal(), scores, rtol=1e-8, atol=1e-10), f"decision {t}"
        assert target.any()
        if (target - position).any():
            assert len(moves) == 1 and moves[0].dtype.kind == "i"
            assert np.array_equal(moves[0], target - position), f"decision {t}"
        else:
            assert moves == []
        position = target
        trader.position = target.copy()
        assert np.array_equal(trader.states, np.sign(target))
        assert trader.position.max() == params.get("size", 5.0) or -trader.position.min() == params.get("size", 5.0)
    assert rebalanced >= 5


@pytest.mark.parametrize("kind, params", [CASES[2], CASES[5], CASES[8]],
                         ids=["reversal", "momentum", "lead-lag"])
def test_the_book_is_dollar_and_factor_neutral_to_the_rounding(kind, params):
    params = {**params, "size": 1e6, "pca_every": 1, "hold": 1}
    history = mids_of(100, seed=12)
    expected = worked_run(kind, history, **params)
    trader = CLASSES[kind](**params)
    feed(trader, history)
    _, betas, target = expected[-1]
    assert np.array_equal(trader.position, target)
    assert abs(trader.position.sum()) <= 3  # six books, each rounded by at most a half
    assert np.abs(betas @ trader.position).max() <= 0.5 * np.abs(betas).sum(axis=1).max()  # of a million lots
    assert np.abs(trader.position).max() == 1e6


# ---- Reversal and momentum by hand ----

MOVES = np.array([[1.0, 0.0, 2.0, 0.0], [2.0, 1.0, 0.0, -1.0], [0.0, 4.0, 0.0, 0.0]])
HAND = 100.0 + np.vstack([np.zeros(4), np.cumsum(MOVES, axis=0)])  # four mids of four books, after three moves


def test_reversal_sells_what_rose_over_the_lookback_by_hand():
    trader = ReversalTrader(lookback=2, factors=0, size=9.0)
    out = feed(trader, HAND)
    assert out[0] == [] and out[1] == [] and trader.warming is None
    # at the third mid: the last two moves add to [3, 1, 2, -1]; minus that, less its mean -1.25, is [-1.75, .25, -.75,
    # 2.25]; its largest is 2.25, so nine lots of that one and, to a rounded 4 times each, [-7, 1, -3, 9]
    assert np.array_equal(out[2][0], [-7, 1, -3, 9])
    # at the fourth: [2, 5, 0, -1] added up, so [-.5, -3.5, 1.5, 2.5] and, nine over 3.5 times that, [-1, -9, 4, 6]
    assert np.array_equal(trader.signal(), [-2.0, -5.0, 0.0, 1.0])
    assert np.array_equal(out[3][0], [6, -10, 7, -3])  # from [-7, 1, -3, 9]
    assert np.array_equal(trader.position, [-1, -9, 4, 6])


def test_momentum_buys_what_rose_over_the_lookback_that_ended_skip_decisions_ago_by_hand():
    trader = MomentumTrader(lookback=2, skip=1, factors=0, size=9.0)
    out = feed(trader, HAND)
    assert out[:3] == [[], [], []] and trader.warming is None
    # the first two moves, [3, 1, 2, -1], and not the last (a skip of 1); less its mean 1.25: [1.75, -.25, .75, -2.25]
    assert np.array_equal(trader.signal(), [3.0, 1.0, 2.0, -1.0])
    assert np.array_equal(out[3][0], [7, -1, 3, -9])


def test_a_skip_of_none_makes_momentum_the_opposite_of_reversal():
    kwargs = dict(lookback=3, factors=1, pca_window=20, pca_every=1)
    history = mids_of(40, seed=2)
    reverse, momentum = ReversalTrader(**kwargs), MomentumTrader(skip=0, **kwargs)
    feed(reverse, history), feed(momentum, history)
    assert np.allclose(momentum.signal(), -reverse.signal(), atol=1e-12)
    assert np.array_equal(momentum.position, -reverse.position)


def test_with_the_factors_taken_out_the_residual_is_what_is_scored_and_the_raw_moves_are_not():
    kwargs = dict(lookback=4, pca_window=30, pca_every=1)
    history = mids_of(60, seed=3)
    raw, residual = ReversalTrader(factors=0, **kwargs), ReversalTrader(factors=2, **kwargs)
    feed(raw, history), feed(residual, history)
    assert np.allclose(raw.signal(), -(history[-1] - history[-5]), atol=1e-12)
    assert not np.allclose(raw.signal(), residual.signal(), atol=1e-3)
    moves = np.diff(history[-31:], axis=0)
    eigen, betas = pca_by_hand(moves, 2)
    last = moves[-4:] - moves.mean(axis=0)
    assert np.allclose(residual.signal(), -(last - (last @ eigen) @ betas).sum(axis=0), atol=1e-10)


def test_equal_scores_leave_nothing_to_hold():
    # Seven books that all rose by 0.7: no book to prefer. The mean of seven equal scores is not quite the score, and
    # what it leaves is noise, which must not be turned into lots.
    history = np.array([np.zeros(7), np.full(7, 0.7)])
    scores = -(history[1] - history[0])
    assert (scores - scores.mean()).any() and np.abs(scores - scores.mean()).max() < 1e-15
    trader = ReversalTrader(lookback=1, factors=0)
    out = feed(trader, history)
    assert out == [[], []] and not trader.position.any() and trader.entries == 0
    assert not trader.states.any() and np.array_equal(trader.signal(), scores)


def test_a_trader_that_holds_something_closes_it_when_the_scores_come_out_equal():
    first = np.array([1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 0.0])
    history = np.array([np.zeros(7), first, first + 0.7])  # then every book rises by 0.7, near enough
    trader = ReversalTrader(lookback=1, factors=0)
    out = feed(trader, history)
    assert np.array_equal(out[1][0], [-2, 2, -5, 5, -1, 1, 0])  # five lots over 2 a point, the halves to the even
    assert np.array_equal(out[2][0], [2, -2, 5, -5, 1, -1, 0])
    assert not trader.position.any() and not trader.states.any() and trader.entries == 6


def test_books_that_did_not_move_hold_nothing_and_close_what_was_held():
    moves = np.array([[2.0, -2.0, 1.0, -1.0], [0.0, 0.0, 0.0, 0.0]])
    history = 100.0 + np.vstack([np.zeros(4), np.cumsum(moves, axis=0)])
    trader = ReversalTrader(lookback=1, factors=0, size=4.0)
    out = feed(trader, history)
    assert np.array_equal(out[1][0], [-4, 4, -2, 2]) and np.array_equal(out[2][0], [4, -4, 2, -2]) and out[2]
    assert not trader.position.any() and not trader.states.any() and not trader.signal().any()


# ---- The target, states and entries ----

def test_entries_count_the_books_that_go_from_flat_or_the_other_side_to_a_side_at_a_rebalance():
    moves = np.array([[2.0, -2.0, 1.0, -1.0],   # reversal: [-4, 4, -2, 2] from flat: four entries
                      [2.0, -2.0, 1.0, -1.0],   # the same: no move, none
                      [-2.0, 2.0, 1.0, -1.0],   # [4, -4, -2, 2]: two books turn round, two stay as they were
                      [1.0, -2.0, 1.0, -1.0],   # [-3, 4, -3, 2]: two turn round again; a bigger leg is no entry
                      [0.0, 0.0, 0.0, 0.0],     # nothing to prefer: flat, and flat is not an entry
                      [2.0, -2.0, 1.0, -1.0]])  # four entries from flat again
    history = 100.0 + np.vstack([np.zeros(4), np.cumsum(moves, axis=0)])
    trader = ReversalTrader(lookback=1, factors=0, size=4.0)
    trader.attach(4)
    assert trader.entries == 0 and not trader.states.any() and trader.states.dtype.kind == "i"
    assert trader.signal() is None and np.array_equal(trader.weights, np.eye(4)) and trader.levels == (0.0, 0.0)
    assert trader.decide(history[0]) == [] and trader.signal() is None  # one mid is no move
    held = [[-4, 4, -2, 2], [-4, 4, -2, 2], [4, -4, -2, 2], [-3, 4, -3, 2], [0, 0, 0, 0], [-4, 4, -2, 2]]
    entries = [4, 4, 6, 8, 8, 12]
    states = [[-1, 1, -1, 1], [-1, 1, -1, 1], [1, -1, -1, 1], [-1, 1, -1, 1], [0, 0, 0, 0], [-1, 1, -1, 1]]
    for k in range(6):
        before = trader.position.copy()
        out = trader.decide(history[k + 1])
        assert np.array_equal(trader.signal(), -moves[k])
        if (np.array(held[k]) - before).any():
            assert len(out) == 1 and np.array_equal(out[0], np.array(held[k]) - before)
        else:
            assert out == []
        trader.position = np.array(held[k])
        assert trader.entries == entries[k] and np.array_equal(trader.states, states[k]), f"move {k}"


def test_a_rebalance_comes_only_every_hold_decisions_and_the_signal_waits_for_the_next():
    history = mids_of(30, seed=4)
    trader = ReversalTrader(lookback=2, hold=4, factors=0)
    trader.attach(6)
    moved, signals = [], []
    for mids in history:
        out = trader.decide(mids.copy())
        moved.append(bool(out))
        signals.append(None if trader.signal() is None else trader.signal().copy())
        for move in out:
            trader.position = trader.position + move
    # three mids make the first score; then one in four
    assert [t for t, did in enumerate(moved) if did] == [2, 6, 10, 14, 18, 22, 26]
    assert signals[0] is None and signals[1] is None
    for t in range(2, 30):
        base = t - (t - 2) % 4
        assert np.array_equal(signals[t], -(history[base] - history[base - 2])), f"decision {t}"


def test_a_move_is_the_target_less_what_is_held_whatever_that_is():
    history = mids_of(5, seed=6)
    trader = ReversalTrader(lookback=1, factors=0, size=5.0)
    trader.attach(6)
    trader.decide(history[0])
    trader.position = np.array([1, -2, 3, 0, 4, -5])  # held from somewhere
    out = trader.decide(history[1])
    scores = -(history[1] - history[0])
    w = scores - scores.mean()
    target = np.rint(5.0 * w / np.abs(w).max()).astype(int)
    assert len(out) == 1 and np.array_equal(out[0], target - [1, -2, 3, 0, 4, -5])
    trader.position = target  # holding the target already, a trader that is asked again has nothing to send
    again = ReversalTrader(lookback=1, factors=0, size=5.0)
    again.attach(6)
    again.position = target.copy()
    again.decide(history[0])
    assert again.decide(history[1]) == []


def test_the_target_is_rounded_to_the_nearest_lot_with_halves_to_the_even():
    # scores [-5, 5, -2.5, 2.5] have no mean to take; five lots over 5 is one a point: np.rint(-2.5) is -2, (2.5) is 2
    history = 100.0 + np.array([[0.0, 0.0, 0.0, 0.0], [5.0, -5.0, 2.5, -2.5]])
    trader = ReversalTrader(lookback=1, factors=0, size=5.0)
    out = feed(trader, history)
    assert np.array_equal(out[1][0], [-5, 5, -2, 2])


@pytest.mark.parametrize("size", [3.0, 10.0, 0.5])
def test_the_largest_leg_is_the_size(size):
    history = mids_of(8, seed=7)
    trader = ReversalTrader(lookback=3, factors=0, size=size)
    feed(trader, history)
    assert np.abs(trader.position).max() == np.rint(size)


# ---- Warming, warm_up and learn ----

@pytest.mark.parametrize("make, needed", [
    (lambda: ReversalTrader(lookback=4, factors=0), 5),
    (lambda: ReversalTrader(lookback=4, factors=1, pca_window=30), 31),
    (lambda: MomentumTrader(lookback=4, skip=3, factors=0), 8),
    (lambda: MomentumTrader(lookback=4, skip=3, factors=2, pca_window=20), 21),
    (lambda: LeadLagTrader(lookback=4, factors=0, pca_window=25), 26),
    (lambda: LeadLagTrader(lookback=4, factors=1, pca_window=25), 26),
], ids=["reversal raw", "reversal factors", "momentum raw", "momentum factors", "lead-lag raw", "lead-lag factors"])
def test_a_trader_waits_for_the_mids_it_needs_and_says_how_many_it_has(make, needed):
    trader = make()
    trader.attach(6)
    assert trader.signal() is None
    history = mids_of(needed + 3, seed=9)
    for t in range(needed - 1):
        assert trader.decide(history[t].copy()) == []
        assert trader.warming == (t + 1, needed) and trader.signal() is None
        assert not trader.states.any() and trader.entries == 0
    out = trader.decide(history[needed - 1].copy())
    assert trader.warming is None and trader.signal() is not None and len(out) == 1
    assert trader.warm_up == pytest.approx((needed + 1) * (trader.interval + WAIT))


@pytest.mark.parametrize("interval", [1.0, 2.5])
def test_the_warm_up_is_the_time_of_one_decision_more_than_the_mids_it_needs(interval):
    trader = ReversalTrader(lookback=4, factors=0, interval=interval)
    assert trader.warm_up == pytest.approx(6 * (interval + WAIT))
    assert trader.warm_up == trader.taking(6)
    assert ReversalTrader(lookback=4, factors=1, pca_window=30, interval=interval).warm_up == \
        pytest.approx(32 * (interval + WAIT))


def first_order(trader, seed=4):
    """The times of a trader's decisions in a market of its own, up to the one that made its first order."""
    times = []
    decide = trader.decide

    def counted(mids):
        times.append(market.time)
        return decide(mids)

    trader.decide = counted
    market = BookMarket(FactorValues(SECTORS), seed=seed, traders=(trader,))
    while not trader.lots and market.time < 2_000:
        market.step()
    return times


@pytest.mark.parametrize("make, decisions", [
    (lambda: ReversalTrader(lookback=3, factors=0, interval=5.0), 4),
    (lambda: ReversalTrader(lookback=3, factors=1, pca_window=30, interval=5.0), 31),
    (lambda: MomentumTrader(lookback=3, skip=4, factors=0, interval=5.0), 8),
    (lambda: MomentumTrader(lookback=3, skip=4, factors=2, pca_window=25, interval=5.0), 26),
    (lambda: LeadLagTrader(lookback=3, factors=1, pca_window=30, interval=5.0), 31),
    (lambda: LeadLagTrader(lookback=3, factors=0, pca_window=30, interval=5.0), 31),
    (lambda: FlowTrader(lookback=6, factors=0, interval=5.0), 7),
    (lambda: FlowTrader(lookback=6, factors=1, pca_window=25, interval=5.0), 26),
    (lambda: FlowTrader(lookback=6, factors=0, measure="queues", interval=5.0), 1),
    (lambda: FlowTrader(lookback=6, factors=1, pca_window=25, measure="queues", interval=5.0), 26),
], ids=["reversal raw", "reversal", "momentum raw", "momentum", "lead-lag", "lead-lag raw", "flow trades raw",
        "flow trades", "flow queues raw", "flow queues"])
def test_a_trader_makes_its_first_order_at_the_decision_it_has_the_mids_for_and_before_its_warm_up_ends(make,
                                                                                                         decisions):
    trader = make()
    times = first_order(trader)
    # one decision a time it has to wait for, the first being at the start. (The books start with ten lots at every
    # price: a trader of the queues that needs no mids can find them even, and go flat, at the very first decision.)
    assert len(times) == decisions or (decisions == 1 and len(times) == 2)
    decisions = len(times)
    assert trader.position.any() and trader.lots > 0
    gaps = np.diff(times)
    assert (gaps >= 5.0).all() and (gaps < 5.0 + WAIT * 4).all()
    # the decision's number less one intervals in, at the least; the warm-up two intervals and a wait more, and so
    # past the order and past the decision that follows
    assert times[-1] >= (decisions - 1) * 5.0
    assert times[-1] < trader.warm_up - 5.0
    if decisions > 20:  # and not long past it either
        assert trader.warm_up - 2 * 5.0 - decisions * WAIT <= times[-1]


@pytest.mark.parametrize("kind, params", [CASES[1], CASES[4], CASES[7]], ids=["reversal", "momentum", "lead-lag"])
def test_a_trader_that_learned_the_past_mids_rebalances_at_its_first_decision(kind, params):
    history = mids_of(100, seed=14)
    needed = params["pca_window"] + 1
    # exactly what it needs but the latest: it takes them all and trades on none
    learner = CLASSES[kind](**params)
    learner.attach(6)
    assert learner.learn(list(history[:needed - 1])) == needed - 1
    assert learner.warming == (needed - 1, needed) and not learner.position.any() and learner.entries == 0
    assert learner.lots == 0 and learner.signal() is None
    out = learner.decide(history[needed - 1])
    assert learner.warming is None and len(out) == 1
    expected = worked_run(kind, history[:needed], **params)[needed - 1]
    assert np.array_equal(out[0], expected[2]) and np.allclose(learner.signal(), expected[0], atol=1e-10)
    # more than it can use: only the last ones count, as many as it keeps
    more = CLASSES[kind](**params)
    more.attach(6)
    assert more.learn(list(history[:60])) == needed
    assert more.warming is None
    out = more.decide(history[60])
    shifted = worked_run(kind, history[61 - needed:61], **params)[needed - 1]
    assert np.array_equal(out[0], shifted[2]) and np.allclose(more.signal(), shifted[0], atol=1e-10)
    # and too few: it goes on warming from where it was
    few = CLASSES[kind](**params)
    few.attach(6)
    assert few.learn(list(history[:needed - 3])) == needed - 3
    assert few.decide(history[needed - 3]) == [] and few.decide(history[needed - 2]) == []
    assert few.warming == (needed - 1, needed) and few.decide(history[needed - 1])


def test_a_trader_that_needs_no_factors_keeps_only_the_mids_it_scores_with():
    trader = ReversalTrader(lookback=3, factors=0)
    trader.attach(6)
    history = mids_of(10, seed=1)
    assert trader.learn(list(history)) == 4  # a score over three decisions needs four mids
    assert trader.warming is None
    latest = history[9] + 1.0
    trader.decide(latest)
    assert np.allclose(trader.signal(), -(latest - history[7]), atol=1e-12)  # three moves: from the third last mid


# ---- No lookahead, and no memory beyond what is needed ----

def signals_of(kind, params, history):
    """The signal after each decision (None while there is none), as copies."""
    trader = CLASSES[kind](**params)
    trader.attach(history.shape[1])
    out = []
    for mids in history:
        moves = trader.decide(mids.copy())
        for move in moves:
            trader.position = trader.position + move
        out.append(None if trader.signal() is None else trader.signal().copy())
    return out


@pytest.mark.parametrize("kind, params", CASES, ids=IDS)
def test_a_decision_depends_on_the_mids_up_to_it_and_none_after(kind, params):
    history = mids_of(100, seed=21)
    later = 70
    changed = history.copy()
    changed[later] += np.linspace(-9.0, 9.0, 6)  # a later mid, other than it was
    before, after = signals_of(kind, params, history), signals_of(kind, params, changed)
    for t in range(later):
        if before[t] is None:
            assert after[t] is None
        else:
            assert np.array_equal(before[t], after[t]), f"decision {t}"
    assert any(not np.array_equal(a, b) for a, b in zip(before[later:], after[later:], strict=True) if a is not None)


@pytest.mark.parametrize("kind, params", [CASES[0], CASES[1], CASES[3], CASES[4], CASES[6], CASES[7]],
                         ids=["reversal raw", "reversal", "momentum raw", "momentum", "lead-lag raw", "lead-lag"])
def test_a_decision_does_not_depend_on_mids_older_than_the_ones_it_needs(kind, params):
    params = {**params, "hold": 1, "pca_every": 1}
    needed = 1 + (params["pca_window"] if params["factors"] or kind == "leadlag" else params["lookback"]
                  + params.get("skip", 0))
    history = mids_of(80, seed=22)
    changed = history.copy()
    changed[:5] += np.random.default_rng(0).uniform(-20.0, 20.0, (5, 6))  # the oldest, other than they were
    before, after = signals_of(kind, params, history), signals_of(kind, params, changed)
    for t in range(80):
        if t < needed - 1:
            assert before[t] is None and after[t] is None
        elif t - 5 >= needed - 1:  # none of the five is among the last ``needed`` mids
            assert np.array_equal(before[t], after[t]), f"decision {t}"
    assert not np.array_equal(before[needed - 1], after[needed - 1])  # while the first decision does use them


# ---- Lead-lag ----

def test_gamma_is_a_ridge_regression_of_each_books_next_move_on_all_the_books_last_with_a_zero_diagonal():
    history = mids_of(60, seed=31)
    for ridge in (0.0, 1.0, 5.0):
        trader = LeadLagTrader(lookback=1, factors=0, pca_window=40, pca_every=1, ridge=ridge)
        feed(trader, history)
        moves = np.diff(history[-41:], axis=0)  # the last 40 moves, 39 pairs
        before, after = moves[:-1], moves[1:]
        x, y = before - before.mean(axis=0), after - after.mean(axis=0)
        covariance = x.T @ x / 39
        penalty = ridge * np.trace(covariance) / 6
        beta = np.linalg.solve(covariance + penalty * np.eye(6), x.T @ y / 39)
        expected = beta.T.copy()  # [j, i]: book i's move now, in book j's next
        np.fill_diagonal(expected, 0.0)
        assert trader.gamma.shape == (6, 6)
        assert np.diag(trader.gamma).tolist() == [0.0] * 6
        assert np.abs(trader.gamma - expected).max() < 1e-9, ridge
        assert np.allclose(trader.signal(), expected @ (history[-1] - history[-2]), atol=1e-9)


def test_a_book_that_follows_another_is_bought_when_the_other_rose():
    # Book 1 does what book 0 did the decision before (and some noise of its own); book 2 is noise.
    rng = np.random.default_rng(5)
    moves = rng.standard_normal((3_100, 3))
    moves[1:, 1] = 0.6 * moves[:-1, 0] + 0.3 * rng.standard_normal(3_099)
    history = 3000.0 + np.vstack([np.zeros(3), np.cumsum(moves, axis=0)])
    trader = LeadLagTrader(lookback=1, factors=0, pca_window=3_000, pca_every=1_000, ridge=0.0)
    feed(trader, history)
    gamma = trader.gamma
    assert gamma[1, 0] == pytest.approx(0.6, abs=0.06)  # row: the book that follows, column: the book it follows
    assert abs(gamma[0, 1]) < 0.06 and np.abs(gamma[2]).max() < 0.06 and abs(gamma[0, 2]) < 0.06
    assert trader.signal()[1] == pytest.approx(gamma[1, 0] * moves[-1, 0] + gamma[1, 2] * moves[-1, 2], abs=1e-9)
    assert trader.signal()[1] == pytest.approx(0.6 * moves[-1, 0], abs=0.4)


def test_a_larger_ridge_shrinks_gamma():
    history = mids_of(120, seed=33)
    sizes = []
    for ridge in (0.0, 1.0, 10.0, 1e6):
        trader = LeadLagTrader(lookback=1, factors=0, pca_window=100, pca_every=1, ridge=ridge)
        feed(trader, history)
        sizes.append(np.abs(trader.gamma).sum())
    assert sizes[0] > sizes[1] > sizes[2] > sizes[3] and sizes[3] < 1e-4 * sizes[0]


def test_gamma_is_estimated_every_pca_every_decisions_and_not_between():
    history = mids_of(80, seed=34)
    trader = LeadLagTrader(lookback=1, factors=0, pca_window=30, pca_every=5)
    trader.attach(6)
    seen = []
    for mids in history:
        trader.decide(mids.copy())
        seen.append(None if trader.gamma is None else trader.gamma.copy())
    assert seen[29] is None and seen[30] is not None  # 31 mids make 30 moves
    for t in range(30, 80):
        fitted = 30 + (t - 30) // 5 * 5  # at the first decision it can, and each fifth after
        assert np.array_equal(seen[t], seen[fitted]), t
        if t == fitted and t > 30:
            assert not np.array_equal(seen[t], seen[t - 1])
    expected = worked_fit(np.diff(history[75 - 30:76], axis=0), 0, 1.0, "leadlag")["gamma"]
    assert np.abs(seen[79] - expected).max() < 1e-9


# ---- Validation ----

@pytest.mark.parametrize("make, message", [
    (lambda: ReversalTrader(lookback=0), "lookback"),
    (lambda: ReversalTrader(lookback=-3), "lookback"),
    (lambda: ReversalTrader(lookback=3, hold=0), "hold"),
    (lambda: ReversalTrader(lookback=3, factors=-1), "factors"),
    (lambda: ReversalTrader(lookback=3, size=0), "size"),
    (lambda: ReversalTrader(lookback=3, size=-1.0), "size"),
    (lambda: ReversalTrader(lookback=3, pca_every=0), "PCA"),
    (lambda: ReversalTrader(lookback=3, pca_window=9), "PCA window"),
    (lambda: ReversalTrader(lookback=10, pca_window=10), "PCA window"),
    (lambda: ReversalTrader(lookback=30, pca_window=30, factors=2), "PCA window"),
    (lambda: MomentumTrader(lookback=3, skip=-1), "skip"),
    (lambda: MomentumTrader(lookback=3, skip=8, pca_window=11), "PCA window"),
    (lambda: MomentumTrader(lookback=0, skip=2), "lookback"),
    (lambda: LeadLagTrader(lookback=3, ridge=-0.1), "ridge"),
    (lambda: LeadLagTrader(lookback=3, factors=0, pca_window=9), "PCA window"),
    (lambda: LeadLagTrader(lookback=3, pca_window=9), "PCA window"),
    (lambda: LeadLagTrader(lookback=0), "lookback"),
    (lambda: FlowTrader(lookback=3, measure="volume"), "measure"),
    (lambda: FlowTrader(lookback=0), "lookback"),
    (lambda: FlowTrader(lookback=3, hold=0), "hold"),
], ids=lambda x: "" if callable(x) else x)
def test_a_setting_the_trader_cannot_work_with_is_refused_and_says_what_it_needs(make, message):
    with pytest.raises(ValueError, match=message):
        make()


@pytest.mark.parametrize("make", [
    lambda: ReversalTrader(lookback=1, factors=0),
    lambda: ReversalTrader(lookback=1, hold=1, factors=0, size=0.01, pca_every=1),
    lambda: ReversalTrader(lookback=3, factors=1, pca_window=10),
    lambda: ReversalTrader(lookback=9, factors=1, pca_window=10),
    lambda: ReversalTrader(lookback=30, factors=2, pca_window=31),
    lambda: ReversalTrader(lookback=30, factors=0, pca_window=2),  # no factors: the window is not asked for
    lambda: MomentumTrader(lookback=3, skip=0, factors=1, pca_window=10),
    lambda: MomentumTrader(lookback=3, skip=7, factors=1, pca_window=11),
    lambda: MomentumTrader(lookback=3, skip=40, factors=0, pca_window=4),
    lambda: LeadLagTrader(lookback=3, factors=1, pca_window=10, ridge=0.0),
    lambda: LeadLagTrader(lookback=3, factors=0, pca_window=10),
    lambda: FlowTrader(lookback=3, measure="queues"),
    lambda: FlowTrader(lookback=3, measure="trades", factors=0, pca_window=2),
])
def test_the_settings_at_the_edge_are_accepted(make):
    assert isinstance(make(), CrossSectionTrader)


@pytest.mark.parametrize("make, books, ok", [
    (lambda: ReversalTrader(lookback=3, factors=0), 2, True),
    (lambda: ReversalTrader(lookback=3, factors=0), 1, False),
    (lambda: ReversalTrader(lookback=3, factors=1), 3, True),
    (lambda: ReversalTrader(lookback=3, factors=1), 2, False),
    (lambda: MomentumTrader(lookback=3, factors=2), 4, True),
    (lambda: MomentumTrader(lookback=3, factors=2), 3, False),
    (lambda: LeadLagTrader(lookback=3, factors=3), 5, True),
    (lambda: LeadLagTrader(lookback=3, factors=3), 4, False),
    (lambda: FlowTrader(lookback=3, factors=1), 3, True),
    (lambda: FlowTrader(lookback=3, factors=1), 2, False),
])
def test_it_needs_more_books_than_the_factors_and_one(make, books, ok):
    trader = make()
    if ok:
        trader.attach(books)
        assert trader.position.tolist() == [0] * books and trader.states.tolist() == [0] * books
        assert np.array_equal(trader.weights, np.eye(books))
    else:
        with pytest.raises(ValueError, match="factors"):
            trader.attach(books)


def test_a_market_refuses_a_trader_with_too_many_factors_for_its_books():
    with pytest.raises(ValueError, match="factors"):
        BookMarket(FactorValues(SECTORS[:3]), traders=(ReversalTrader(lookback=3, factors=2),))


def test_the_base_class_has_no_score_of_its_own():
    trader = CrossSectionTrader(lookback=1, factors=0)
    trader.attach(3)
    with pytest.raises(NotImplementedError):
        feed(trader, mids_of(3, books=3))


def test_the_lookback_has_no_default_and_the_other_settings_have_the_ones_the_spec_gives():
    with pytest.raises(TypeError):
        ReversalTrader()
    fields = {f.name: f.default for f in dataclasses.fields(ReversalTrader) if f.default is not dataclasses.MISSING}
    assert fields == {"size": 5.0, "interval": 1.0, "netted": True, "active": True, "next_time": 0.0, "position": None,
                      "cash": 0.0, "lots": 0, "spread_paid": 0.0, "entries": 0, "netted_lots": 0, "record": False,
                      "rate": 1.0, "band": 0.0, "hold": 1, "factors": 1, "pca_window": 500, "pca_every": 20}
    assert [f.name for f in dataclasses.fields(ReversalTrader)][:2] == ["size", "interval"]
    assert MomentumTrader(lookback=3).skip == 0 and LeadLagTrader(lookback=3).ridge == 1.0
    assert FlowTrader(lookback=3).measure == "trades"
    for cls in (ReversalTrader, MomentumTrader, LeadLagTrader, FlowTrader):
        assert issubclass(cls, CrossSectionTrader) and issubclass(cls, BookTrader)
        assert dataclasses.is_dataclass(cls)


# ---- The flow a market keeps, from a market whose orders are scripted ----

class Scripted:
    """An order-flow model that sends the orders it is given, one an event, in its book, and nothing else."""

    def __init__(self):
        self.orders = []

    def rate(self, book):
        return 1.0

    def event(self, book, t, fair, rng):
        return self.orders.pop(0) if self.orders else None


def scripted_market(books, **kwargs):
    models = [Scripted() for _ in range(books)]
    return BookMarket(FactorValues(np.ones((books, 1))), models=models, seed=1, **kwargs), models


def play(market, models, script):
    """Send each book its orders, in turn, until every one has been through. Every book starts at 2999 / 3001."""
    for book, orders in script.items():
        models[book].orders.extend(orders)
    while any(model.orders for model in models):
        market.step()


def buy(lots):
    return market_order(lots, 0.0, Side.BID)


def sell(lots):
    return market_order(lots, 0.0, Side.ASK)


def test_the_flow_and_volume_start_at_nothing_a_book_each_and_are_floats():
    market, _ = scripted_market(5)
    assert market.flow.shape == (5,) and market.volume.shape == (5,)
    assert market.flow.dtype == float and market.volume.dtype == float
    assert not market.flow.any() and not market.volume.any()


def test_the_flow_and_volume_count_what_the_models_orders_traded_on_arrival_signed_by_the_side():
    market, models = scripted_market(3)
    resting = limit_order(2990, 5, 0.0, Side.BID)
    play(market, models, {
        0: [buy(3),                                   # +3 of 3
            sell(4),                                  # -1 of 7
            limit_order(3001, 15, 0.0, Side.BID),     # 7 left at 3001 trade, 8 rest: +7 of 7, so 6 of 14
            limit_order(3005, 5, 0.0, Side.ASK)],     # not across the best bid: nothing
        1: [resting,                                  # a bid below the best: nothing
            limit_order(2999, 12, 0.0, Side.ASK),     # across the bids: the 10 at 2999 trade, 2 rest: -10 of 10
            cancel_order(resting.order_id, 0.0)],     # no fills: nothing
        2: [sell(80),                                 # every bid there is: -80 of 80
            sell(5),                                  # nothing left to trade with: nothing
            buy(1)],                                  # +1 of 1: -79 of 81
    })
    assert market.flow.tolist() == [6.0, -10.0, -79.0]
    assert market.volume.tolist() == [14.0, 10.0, 81.0]
    assert [(book.best_bid, book.best_ask) for book in market.books] == [(3001, 3002), (2998, 2999), (None, 3001)]


def test_a_traders_orders_are_not_counted_and_a_models_are_even_with_a_trader_about():
    trader = ReversalTrader(lookback=1, factors=0)
    market, models = scripted_market(3, traders=(trader,))
    market._trade(trader, 0, 7)  # bought: it moved the book, and its position
    market._trade(trader, 1, -9)
    assert trader.position.tolist() == [7, -9, 0] and trader.lots == 16
    assert not market.flow.any() and not market.volume.any()
    asked = []
    trader.decide = lambda mids: asked.append(mids) or [np.array([2, 2, -2])]
    market.trade_for(trader)
    assert len(asked) == 1 and trader.position.tolist() == [9, -7, -2] and trader.lots == 22
    assert not market.flow.any() and not market.volume.any()
    play(market, models, {0: [buy(2)], 1: [sell(1)]})  # and the models' own are
    assert market.flow.tolist() == [2.0, -1.0, 0.0] and market.volume.tolist() == [2.0, 1.0, 0.0]


def test_flow_and_volume_are_the_models_fills_in_a_market_with_a_trader_that_trades():
    # The same count, made another way: from the fills the engine gives for each order that does not come from _trade.
    trader = ReversalTrader(lookback=3, factors=0, interval=2.0)
    market = BookMarket(FactorValues(SECTORS), seed=5, traders=(trader,))
    flow, volume, inside = np.zeros(6), np.zeros(6), [False]
    process, trade = market.engine.process, market._trade

    def traced_process(book, order):
        fills = process(book, order)
        if not inside[0]:
            k, taken = market.books.index(book), sum(f.size for f in fills)
            flow[k] += taken if order.side == Side.BID else -taken
            volume[k] += taken
        return fills

    def traced_trade(*args):
        inside[0] = True
        try:
            trade(*args)
        finally:
            inside[0] = False

    market.engine.process, market._trade = traced_process, traced_trade
    while market.time < 150.0:
        market.step()
    assert trader.lots > 50 and volume.min() > 100
    assert np.array_equal(market.flow, flow) and np.array_equal(market.volume, volume)
    assert (np.abs(market.flow) <= market.volume).all() and ((market.volume - market.flow) % 2 == 0).all()
    assert np.abs(market.flow).max() > 0 and (market.flow != 0).any()


# ---- observe ----

class Recorder(BookTrader):
    """Does nothing but see: what it is shown, and when."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    def observe(self, market):
        self.calls.append(("observe", market, market.flow.copy(), market.volume.copy()))

    def decide(self, mids):
        self.calls.append(("decide", mids.copy(), self.calls[-1][2] if self.calls else None))
        return []


def test_a_trader_observes_the_market_just_before_each_decision_and_is_not_shown_it_without_a_mid():
    recorder = Recorder(interval=2.0)
    market = BookMarket(FactorValues(SECTORS), seed=3, traders=(recorder,))
    market.run(60.0)
    kinds = [call[0] for call in recorder.calls]
    assert len(kinds) > 40 and kinds == ["observe", "decide"] * (len(kinds) // 2)
    for (_, shown, flow, _), (_, mids, flow_then) in zip(recorder.calls[0::2], recorder.calls[1::2], strict=True):
        assert shown is market and flow_then is flow and mids.shape == (6,)
    assert recorder.calls[-2][2].any() and recorder.calls[-2][3].any()  # what it saw was the market as it was then
    volumes = [call[3] for call in recorder.calls[0::2]]
    assert all((later >= earlier).all() for earlier, later in zip(volumes, volumes[1:], strict=False))
    assert volumes[-1].sum() > volumes[0].sum()
    # a book without a mid: neither is called, and it is next asked after its interval
    seen = len(recorder.calls)
    market.submit(0, buy(100_000))
    assert market.mid(0) is None
    market.trade_for(recorder)
    assert len(recorder.calls) == seen and recorder.next_time == market.time + 2.0


def test_observing_is_nothing_unless_a_trader_does_it():
    market, _ = scripted_market(2)
    assert BookTrader().observe(market) is None
    assert ReversalTrader(lookback=2, factors=0).observe(market) is None


class Quiet(BookTrader):
    def decide(self, mids):
        return []


class Watching(ReversalTrader):
    """A reversal trader that looks at everything the market keeps, and uses none of it."""

    def observe(self, market):
        self.looked = (market.flow.copy(), market.volume.copy(),
                       [(book.best_bid, book.best_ask, book.depth_at_price(book.best_bid, Side.BID))
                        for book in market.books])


def path(traders, seed):
    market = BookMarket(FactorValues(SECTORS), seed=seed, traders=traders)
    return market, market.run(300.0)


@pytest.mark.parametrize("seed", [1, 2])
def test_a_market_with_a_trader_that_observes_has_just_the_path_of_one_without(seed):
    alone, alone_frame = path((), seed)
    quiet, quiet_frame = path((Quiet(),), seed)
    watching, watching_frame = path((Watching(lookback=3, factors=0),), seed)
    plain, plain_frame = path((ReversalTrader(lookback=3, factors=0),), seed)
    columns = [c for c in alone_frame if not c.startswith("value_")]
    assert quiet_frame[columns].equals(alone_frame[columns])
    assert alone.events == quiet.events and alone.time == quiet.time
    assert alone.rng.bit_generator.state == quiet.rng.bit_generator.state
    # and for one that trades, whether it looks or not
    assert watching.traders[0].lots > 100 and hasattr(watching.traders[0], "looked")
    assert watching_frame.equals(plain_frame)
    assert watching.events == plain.events and watching.time == plain.time
    assert watching.rng.bit_generator.state == plain.rng.bit_generator.state
    for k, trader in enumerate(("cash", "lots", "spread_paid", "entries")):
        assert getattr(watching.traders[0], trader) == getattr(plain.traders[0], trader), k
    assert np.array_equal(watching.traders[0].position, plain.traders[0].position)
    assert (watching.flow == plain.flow).all() and (watching.volume == plain.volume).all()


# ---- FlowTrader ----

def look(trader, market):
    """What a market does at a decision: the trader sees it, and then decides on the mids."""
    trader.observe(market)
    return trader.decide(np.array([market.mid(k) for k in range(market.n)]))


def test_trades_score_the_change_in_flow_over_the_change_in_volume_over_the_last_lookback_decisions():
    market, models = scripted_market(4)
    trader = FlowTrader(lookback=2, factors=0, size=6.0)
    trader.attach(4)
    assert trader.warming == (0, 3)
    assert look(trader, market) == [] and trader.warming == (1, 3)     # flow [0, 0, 0, 0], volume the same
    play(market, models, {0: [buy(3)], 1: [sell(5)]})                  # flow [3, -5, 0, 0], volume [3, 5, 0, 0]
    assert look(trader, market) == [] and trader.warming == (2, 3) and trader.signal() is None
    play(market, models, {0: [sell(1)], 1: [sell(3)], 2: [buy(2)]})    # [2, -8, 2, 0], volume [4, 8, 2, 0]
    market._trade(trader, 0, 7)  # a trader's own order is no part of it
    assert trader.position[0] == 7 and market.flow.tolist() == [2.0, -8.0, 2.0, 0.0]
    assert market.volume.tolist() == [4.0, 8.0, 2.0, 0.0]
    trader.position = np.zeros(4, dtype=int)
    out = look(trader, market)
    # over the two decisions since the first look: flow [2, -8, 2, 0] over volume [4, 8, 2, 0]; no volume, none: 0
    assert trader.warming is None
    assert np.array_equal(trader.signal(), [0.5, -1.0, 1.0, 0.0])
    # less the mean .125: [.375, -1.125, .875, -.125]; six lots over 1.125 times that is [2, -6, 4.67, -.67]
    assert len(out) == 1 and np.array_equal(out[0], [2, -6, 5, -1])
    trader.position = trader.position + out[0]
    # and the window moves on: the first look drops out
    play(market, models, {0: [buy(4)], 3: [limit_order(2998, 6, 0.0, Side.ASK)]})  # [6, -8, 2, -6], [8, 8, 2, 6]
    out = look(trader, market)
    # from the second look to this: flow [3, -3, 2, -6] over volume [5, 3, 2, 6]
    assert np.array_equal(trader.signal(), [0.6, -1.0, 1.0, -1.0])
    assert np.array_equal(out[0], [2, 1, 1, -4])  # [4, -5, 6, -5], less [2, -6, 5, -1]
    assert trader.entries == 4  # four from flat, and then the same sides again: none


def test_trades_with_a_lookback_of_one_look_at_the_last_decision_alone():
    market, models = scripted_market(3)
    trader = FlowTrader(lookback=1, factors=0)
    trader.attach(3)
    look(trader, market)
    play(market, models, {0: [buy(2), sell(1)], 1: [sell(4)], 2: [buy(1)]})
    look(trader, market)
    assert np.array_equal(trader.signal(), [1 / 3, -1.0, 1.0])
    play(market, models, {0: [sell(3)], 1: [buy(1)]})
    look(trader, market)
    assert np.array_equal(trader.signal(), [-1.0, 1.0, 0.0])


def test_queues_score_the_bid_lots_less_the_ask_lots_over_their_sum_at_the_best_quotes_when_seen():
    market, models = scripted_market(5)
    trader = FlowTrader(lookback=3, factors=0, size=6.0, measure="queues")
    trader.attach(5)
    assert trader.warming == (0, 1)
    play(market, models, {
        0: [limit_order(2999, 5, 0.0, Side.BID)],     # 15 at the best bid, 10 at the best ask: 5 / 25
        1: [limit_order(3001, 30, 0.0, Side.ASK)],    # 10 and 40: -30 / 50
        2: [buy(14)],                                 # the ask at 3001 gone and 4 of 3002's: 10 and 6: 4 / 16
        3: [buy(80)],                                 # no ask at all: 0
        4: [sell(80)],                                # no bid at all: 0
    })
    out = look(trader, market)
    assert trader.warming is None  # one look is all it needs
    assert np.allclose(trader.signal(), [0.2, -0.6, 0.25, 0.0, 0.0], atol=1e-15)
    # less the mean -.03: [.23, -.57, .28, .03, .03]; six lots over .57 times that is [2.4, -6, 2.9, .3, .3]
    assert len(out) == 1 and np.array_equal(out[0], [2, -6, 3, 0, 0])
    # it is the queues as they were when it looked
    market.submit(0, limit_order(2999, 100, 0.0, Side.BID))
    trader.position = trader.position + out[0]
    trader.decide(np.full(5, 3000.0))
    assert np.allclose(trader.signal(), [0.2, -0.6, 0.25, 0.0, 0.0], atol=1e-15)
    look(trader, market)  # and the next time it looks, as they are then
    assert trader.signal()[0] == pytest.approx((115 - 10) / 125)


def test_the_queues_are_the_visible_lots_at_the_best_price_only():
    market, models = scripted_market(3)
    trader = FlowTrader(lookback=1, factors=0, measure="queues")
    trader.attach(3)
    play(market, models, {0: [limit_order(2998, 40, 0.0, Side.BID)],                    # below the best bid: unseen
                          1: [limit_order(2999, 20, 0.0, Side.BID, display_size=5)],    # 5 of 20 show
                          2: [limit_order(3002, 40, 0.0, Side.ASK)]})
    look(trader, market)
    assert trader.signal().tolist() == [0.0, (15 - 10) / 25, 0.0]


def test_trades_read_the_flow_and_volume_and_nothing_else_of_the_market():
    trader = FlowTrader(lookback=1, factors=0)
    trader.attach(3)
    market = SimpleNamespace(flow=np.zeros(3), volume=np.zeros(3))  # no books to look at
    trader.observe(market)
    trader.decide(np.zeros(3))
    market.flow, market.volume = np.array([1.0, -2.0, 0.0]), np.array([2.0, 2.0, 0.0])
    trader.observe(market)
    trader.decide(np.zeros(3))
    assert np.array_equal(trader.signal(), [0.5, -1.0, 0.0])
    market.flow += 5.0  # and what it kept is a copy of what it saw
    assert np.array_equal(trader._seen[-1][0], [1.0, -2.0, 0.0])


def test_a_flow_trader_warms_on_whichever_takes_longer_the_flow_it_must_see_or_the_mids_its_factors_need():
    market, models = scripted_market(5)
    trader = FlowTrader(lookback=3, factors=1, pca_window=12)
    trader.attach(5)
    mids = mids_of(30, books=5, seed=3)
    for k in range(12):  # twelve moves make 13 mids; the four looks of its flow come first, and count for no more
        assert trader.warming == (k, 13)
        trader.observe(market)
        assert trader.decide(mids[k]) == []
    assert trader.warming == (12, 13)
    trader.observe(market)
    trader.decide(mids[12])
    assert trader.warming is None
    assert FlowTrader(lookback=3, factors=0, measure="queues").warming == (0, 1)
    assert FlowTrader(lookback=3, factors=0).warming == (0, 4)
    assert FlowTrader(lookback=3, factors=1, pca_window=12, measure="queues").warming == (0, 13)  # not (0, 1)
    assert FlowTrader(lookback=3, factors=1, pca_window=12).warming == (0, 13)


def test_a_flow_trader_takes_in_past_mids_for_its_factors_alone():
    past = list(mids_of(40, books=5, seed=8))
    with_factors = FlowTrader(lookback=3, factors=1, pca_window=12)
    with_factors.attach(5)
    assert with_factors.learn(past) == 13  # the pca_window moves of 13 mids
    assert with_factors.warming == (9, 13)  # but it has seen no flow: four decisions to go, not 13
    assert "factors" in FlowTrader.learn.__doc__ and "flow" in FlowTrader.learn.__doc__
    market, _ = scripted_market(5)
    for k in range(4):
        with_factors.observe(market)
        out = with_factors.decide(past[k] + 1.0)
        assert with_factors.warming == ((10 + k, 13) if k < 3 else None)
    assert with_factors.signal() is not None  # the mids were there already
    assert out == []  # and no flow, so no score and nothing to hold
    without = FlowTrader(lookback=3, factors=0)
    without.attach(5)
    assert without.learn(past) == 0 and without.warming == (0, 4)
    some = FlowTrader(lookback=3, factors=1, pca_window=12)
    some.attach(5)
    assert some.learn(past[:5]) == 5 and some.warming == (5, 13)  # the mids take longer than the flow
    for k in range(4):
        some.observe(market)
        some.decide(past[5 + k])
    assert some.warming == (9, 13)  # and now the flow, which has been seen, is no longer the wait


def test_a_flow_trader_in_a_market_scores_what_the_market_showed_it_each_decision():
    seen = []

    class Logged(FlowTrader):
        def observe(self, market):
            super().observe(market)
            best = [(book.best_bid, book.best_ask) for book in market.books]
            seen.append((market.flow.copy(), market.volume.copy(),
                         np.array([book.depth_at_price(bid, Side.BID) for book, (bid, _) in zip(market.books, best,
                                                                                               strict=True)]),
                         np.array([book.depth_at_price(ask, Side.ASK) for book, (_, ask) in zip(market.books, best,
                                                                                               strict=True)])))

    for measure in ("trades", "queues"):
        seen.clear()
        trader = Logged(lookback=4, factors=0, measure=measure, interval=3.0, size=5.0)
        market = BookMarket(FactorValues(SECTORS), seed=6, traders=(trader,))
        scores = []
        decide = trader.decide

        def spied(mids, decide=decide, scores=scores, trader=trader):
            out = decide(mids)
            scores.append(None if trader.signal() is None else trader.signal().copy())
            return out

        trader.decide = spied
        while market.time < 400.0:
            market.step()
        assert len(seen) == len(scores) > 100 and trader.lots > 0
        first = 4 if measure == "trades" else 0
        assert all(score is None for score in scores[:first])
        for k in range(first, len(seen)):
            if measure == "trades":
                traded = seen[k][1] - seen[k - 4][1]
                expected = np.where(traded > 0, (seen[k][0] - seen[k - 4][0]) / np.where(traded > 0, traded, 1.0), 0.0)
            else:
                bid, ask = seen[k][2], seen[k][3]
                expected = (bid - ask) / (bid + ask)
            assert np.allclose(scores[k], expected, atol=1e-12), (measure, k)
        assert np.abs(trader.position).max() == 5 and abs(trader.position.sum()) <= 3
        assert (np.sign(trader.position) == trader.states).all()


# ---- Traders in a market ----

@pytest.mark.parametrize("make", [
    lambda: ReversalTrader(lookback=5, factors=1, pca_window=40, pca_every=10, hold=3, interval=2.0),
    lambda: MomentumTrader(lookback=5, skip=2, factors=2, pca_window=40, pca_every=10, hold=3, interval=2.0),
    lambda: LeadLagTrader(lookback=5, factors=1, pca_window=40, pca_every=10, hold=3, interval=2.0),
    lambda: FlowTrader(lookback=5, factors=1, pca_window=40, pca_every=10, hold=3, interval=2.0),
    lambda: FlowTrader(lookback=5, factors=0, measure="queues", hold=3, interval=2.0),
], ids=["reversal", "momentum", "lead-lag", "flow trades", "flow queues"])
def test_every_kind_of_trader_trades_in_a_market_and_holds_what_it_decided(make):
    trader = make()
    market = BookMarket(FactorValues(SECTORS), seed=7, traders=(trader,))
    market.run(400.0)
    assert trader.lots > 50 and trader.entries > 5
    assert np.abs(trader.position).max() == 5  # the largest leg, a whole market order for each
    assert abs(trader.position.sum()) <= 3
    assert (np.sign(trader.position) == trader.states).all()
    assert trader.value(market.marks()) == pytest.approx(trader.cash + float(trader.position @ market.marks()))
    assert trader.spread_paid >= 0 and trader.lots >= np.abs(trader.position).sum()
    assert trader.levels == (0.0, 0.0) and trader.signal().shape == (6,) and np.array_equal(trader.weights, np.eye(6))


def test_the_dashboards_history_of_a_trader_has_its_scores():
    trader = ReversalTrader(lookback=3, factors=0, interval=2.0, record=True)
    market = BookMarket(FactorValues(SECTORS), seed=7, traders=(trader,))
    market.run(40.0)
    assert len(trader.history) > 10
    assert trader.history[0][3] is None and trader.history[-1][3].shape == (6,)
    assert np.array_equal(trader.history[-1][3], trader.signal())


# ---- Lo and MacKinlay's decomposition ----

def contrarian_by_loop(moves, lag):
    """The mean over the pairs (t - lag, t) of the profit sum_i w_i r_i, with w_i = -(r_i - mean r) / n of t - lag."""
    n = moves.shape[1]
    profits = []
    for t in range(lag, len(moves)):
        w = np.array([-(moves[t - lag, i] - moves[t - lag].mean()) / n for i in range(n)])
        profits.append(sum(w[i] * moves[t, i] for i in range(n)))
    return float(np.mean(profits)), len(profits)


@pytest.mark.parametrize("lag", [1, 2, 3])
@pytest.mark.parametrize("books", [2, 3, 5])
def test_the_profit_is_the_mean_of_the_realized_profits_and_the_three_parts_add_to_it(lag, books):
    rng = np.random.default_rng(lag * 10 + books)
    drift = 0.3 * rng.standard_normal(books)
    moves = drift + rng.standard_normal((60, books))
    moves[1:] += 0.4 * moves[:-1]  # autocorrelated, so that the parts are not zero
    moves[1:, 0] += 0.3 * moves[:-1, 1]
    result = contrarian_decomposition(moves, lag)
    expected, pairs = contrarian_by_loop(moves, lag)
    assert pairs == 60 - lag
    assert set(result) == {"profit", "own", "cross", "means"}
    assert result["profit"] == pytest.approx(expected, rel=1e-12, abs=1e-14)
    assert result["own"] + result["cross"] + result["means"] == pytest.approx(result["profit"], rel=1e-9, abs=1e-14)
    assert all(abs(part) > 1e-4 for part in result.values())
    assert all(isinstance(value, float) for value in result.values())
    # the parts, from the sample covariances of the pairs
    later, earlier = moves[lag:], moves[:-lag]
    omega = np.array([[np.mean((later[:, i] - later[:, i].mean()) * (earlier[:, j] - earlier[:, j].mean()))
                       for j in range(books)] for i in range(books)])
    mu, nu = later.mean(axis=0), earlier.mean(axis=0)
    assert result["own"] == pytest.approx(-(books - 1) / books ** 2 * np.trace(omega), rel=1e-9)
    assert result["cross"] == pytest.approx((omega.sum() - np.trace(omega)) / books ** 2, rel=1e-9)
    assert result["means"] == pytest.approx(-np.sum((mu - mu.mean()) * (nu - nu.mean())) / books, rel=1e-9)


def test_books_whose_moves_do_not_depend_on_their_own_past_have_nothing_from_own():
    # Book 0 moves at even decisions and book 1 at odd ones, so that neither follows itself, and 1 repeats what 0 did
    # the decision before. The sums are 0 over both ends, so that no mean is left.
    a = np.array([2.0, -3.0, 4.0, -3.0, 2.0])
    x, y = np.zeros(9), np.zeros(9)
    x[0::2] = a
    y[1::2] = a[:-1]
    moves = np.column_stack([x, y])
    result = contrarian_decomposition(moves)
    # the covariance of book 1 now with book 0 before: (4 + 9 + 16 + 9) / 8; of book 0 now with book 1 before:
    # (2 * -3 + -3 * 4 + 4 * -3 + -3 * 2) / 8; their sum is 2 / 8, over n ** 2 = 4
    assert result["own"] == pytest.approx(0.0, abs=1e-15)
    assert result["means"] == pytest.approx(0.0, abs=1e-15)
    assert result["cross"] == pytest.approx(2 / 8 / 4, abs=1e-15)
    assert result["profit"] == pytest.approx(0.0625, abs=1e-15)
    assert result["profit"] == pytest.approx(contrarian_by_loop(moves, 1)[0], abs=1e-15)


def test_books_that_move_in_blocks_apart_have_nothing_from_cross_and_only_their_own_autocovariance_counts():
    # Three books that each move in a block of decisions the others sit out, with a quiet decision between, in steps
    # that sum to 0 and are away from both ends: no cross-covariance and no mean, so all of the profit is own.
    moves = np.zeros((17, 3))
    moves[1:5, 0] = [1.0, 2.0, -1.0, -2.0]      # sum of lagged products 1 * 2 + 2 * -1 + -1 * -2 = 2
    moves[6:10, 1] = [3.0, -1.0, -3.0, 1.0]     # -3 + 3 - 3 = -3
    moves[11:15, 2] = [2.0, 0.5, -0.5, -2.0]    # 1 - 0.25 + 1 = 1.75
    trace = (2.0 - 3.0 + 1.75) / 16  # over the 16 pairs
    result = contrarian_decomposition(moves)
    assert result["cross"] == pytest.approx(0.0, abs=1e-15) and result["means"] == pytest.approx(0.0, abs=1e-15)
    assert result["own"] == pytest.approx(-(3 - 1) / 9 * trace, abs=1e-15)
    assert result["profit"] == pytest.approx(result["own"], abs=1e-15)
    assert result["profit"] == pytest.approx(contrarian_by_loop(moves, 1)[0], abs=1e-15)


def test_books_that_only_drift_have_only_the_dispersion_of_their_means():
    moves = np.tile([1.0, 2.0, 6.0], (30, 1))  # no autocovariance at all: the means 1, 2, 6 about their mean 3
    result = contrarian_decomposition(moves, lag=2)
    assert result["own"] == pytest.approx(0.0, abs=1e-12) and result["cross"] == pytest.approx(0.0, abs=1e-12)
    # w = -(m - 3) / 3 = [2, 1, -3] / 3, so that the profit is w . m = (2 + 2 - 18) / 3
    assert result["means"] == pytest.approx(-14 / 3, rel=1e-12)
    assert result["profit"] == pytest.approx(-14 / 3, rel=1e-12)


def test_a_contrarian_loses_to_books_that_follow_themselves_and_gains_from_those_that_reverse():
    rng = np.random.default_rng(2)
    shocks = rng.standard_normal((20_000, 4))
    for phi, sign in ((0.5, -1), (-0.5, 1)):
        moves = np.zeros_like(shocks)
        for t in range(1, len(shocks)):
            moves[t] = phi * moves[t - 1] + shocks[t]
        result = contrarian_decomposition(moves)
        # an AR(1) of 1 / (1 - phi ** 2) a variance and phi that times as the lag-1 autocovariance, in each of four
        # books: own is -(n - 1) / n ** 2 * n * (phi / (1 - phi ** 2))
        assert result["own"] == pytest.approx(-0.75 * phi / (1 - phi ** 2), rel=0.1)
        assert np.sign(result["own"]) == sign and np.sign(result["profit"]) == sign
        assert abs(result["cross"]) < 0.05 and abs(result["means"]) < 0.01


def test_books_that_follow_the_others_make_the_cross_part_positive():
    rng = np.random.default_rng(3)
    moves = rng.standard_normal((20_000, 3))
    moves[1:, 1] += 0.7 * moves[:-1, 0]
    moves[1:, 2] += 0.7 * moves[:-1, 0]
    result = contrarian_decomposition(moves)
    # two books follow book 0 with 0.7: the sum of the cross-covariances is 1.4, over n ** 2 = 9
    assert result["cross"] == pytest.approx(1.4 / 9, abs=0.01) and abs(result["own"]) < 0.02


def test_the_docstring_gives_the_formula_with_its_signs():
    doc = contrarian_decomposition.__doc__
    for part in ("w[i, t] = -(r[i, t-lag] - mean_j r[j, t-lag]) / n", "own   = -(n - 1) / n**2 * trace(Omega)",
                 "cross = (1' Omega 1 - trace(Omega)) / n**2",
                 "means = -sum_i (mu_i - mean(mu)) (nu_i - mean(nu)) / n"):
        assert part in doc


def test_the_decomposition_refuses_a_lag_it_has_too_few_moves_for():
    moves = np.random.default_rng(1).standard_normal((10, 3))
    assert contrarian_decomposition(moves, lag=8)["profit"] == pytest.approx(contrarian_by_loop(moves, 8)[0])
    for lag, length in ((0, 10), (-1, 10), (9, 10), (8, 9)):
        with pytest.raises(ValueError, match="lag"):
            contrarian_decomposition(moves[:length], lag)
    assert contrarian_decomposition(moves[:3], lag=1)["profit"] == pytest.approx(contrarian_by_loop(moves[:3], 1)[0])
    assert contrarian_decomposition(list(moves.tolist()), 1)["profit"] == pytest.approx(
        contrarian_by_loop(moves, 1)[0])


# ---- Cost-aware trading ----
# With a rate below 1 or a band above 0 a trader keeps the target of its last rebalance and goes toward it at every
# decision once it can trade, its aim going a share of the way to the target at each; the signal, the states and the
# entries change at a rebalance alone. With the defaults it orders at a rebalance alone. The numbers below are worked
# out by hand, and the longer runs against ``worked_run`` and the plain-Python ``aim_by_hand`` and ``toward_by_hand``
# (in test_statarb).

# Three books and a reversal trader of one decision's move, rebalancing every third decision (the rows 1, 4 and 7, from
# 0). Between rebalances the mids move back, which would turn the target round if it were scored again.
STEPS = np.array([
    [100, 100, 100],
    [97, 100, 103],  # moves -3, 0, 3: scores 3, 0, -3: buy the first, sell the third: [6, 0, -6]
    [100, 100, 100],
    [100, 100, 100],
    [103, 100, 97],  # moves 3, 0, -3: [-6, 0, 6]
    [100, 100, 100],
    [100, 100, 100],
    [97, 103, 97],  # moves -3, 3, -3: scores 3, -3, 3, less their mean of 1: 2, -4, 2: [3, -6, 3]
], dtype=float)


def reversal_by_three(**settings):
    return ReversalTrader(lookback=1, factors=0, hold=3, size=6, **settings)


def run_by_three(trader, fill=True):
    """Per decision: its moves, what it holds, its states, entries and signal. With ``fill`` off, none of it fills."""
    trader.attach(3)
    out = []
    for mids in STEPS:
        moves = trader.decide(mids.copy())
        if fill:
            for move in moves:
                trader.position = trader.position + move
        out.append(([m.tolist() for m in moves], trader.position.tolist(), trader.states.tolist(), trader.entries,
                    None if trader.signal() is None else trader.signal().tolist()))
    return out


def one(a, b, c):
    return [[a, b, c]]


# The orders when none is filled, so that the books are always at 0 and what is ordered is the aim, less the band,
# rounded. At a rate of a half the first book's aim goes 3, 4.5 and 5.25 toward 6, then -0.375, -3.19 and -4.59 toward
# -6, then -0.80 toward 3. The third book's is its opposite until the last, which is 3.80 toward 3, and the second's is
# 0 until the last, which is -3 toward -6.
UNFILLED = [
    ({}, [[], one(6, 0, -6), [], [], one(-6, 0, 6), [], [], one(3, -6, 3)]),  # the defaults: at a rebalance alone
    ({"rate": 0.5}, [[], one(3, 0, -3), one(4, 0, -4), one(5, 0, -5), [], one(-3, 0, 3), one(-5, 0, 5),
                     one(-1, -3, 4)]),  # 4.5 goes to 4, and -0.375 to none
    ({"rate": 0.99}, [[], *[one(6, 0, -6)] * 3, *[one(-6, 0, 6)] * 3, one(3, -6, 3)]),  # all, but below 1: it goes on
    ({"band": 0.25}, [[], *[one(6, 0, -6)] * 3, *[one(-6, 0, 6)] * 3, one(3, -6, 3)]),  # 5.75 and 2.75 are 6 and 3
    ({"band": 1}, [[], *[one(5, 0, -5)] * 3, *[one(-5, 0, 5)] * 3, one(2, -5, 2)]),
    ({"band": 2}, [[], *[one(4, 0, -4)] * 3, *[one(-4, 0, 4)] * 3, one(1, -4, 1)]),
    # 1, 2.5 and 3.25 beyond the band of 2; then the aim is inside it; then 1.19 and 2.59; then -0.80, -3 and 3.80
    ({"rate": 0.5, "band": 2}, [[], one(1, 0, -1), one(2, 0, -2), one(3, 0, -3), [], one(-1, 0, 1), one(-3, 0, 3),
                                one(0, -1, 2)]),
]


@pytest.mark.parametrize("settings, expected", UNFILLED, ids=[str(s) for s, _ in UNFILLED])
def test_with_a_rate_or_a_band_it_orders_toward_its_last_target_every_decision_and_without_only_at_a_rebalance(
        settings, expected):
    run = run_by_three(reversal_by_three(**settings), fill=False)
    assert [moves for moves, *_ in run] == expected
    assert all(position == [0, 0, 0] for _, position, *_ in run)


def test_a_trader_with_a_rate_goes_the_rest_of_the_way_between_rebalances_and_changes_nothing_else_there():
    run = run_by_three(reversal_by_three(rate=0.5))
    keep = ([3.0, 0.0, -3.0], [1, 0, -1], 2)
    assert run[0] == ([], [0, 0, 0], [0, 0, 0], 0, None)
    assert run[1] == (one(3, 0, -3), [3, 0, -3], [1, 0, -1], 2, keep[0])  # the aim is half of 6
    assert run[2] == (one(2, 0, -2), [5, 0, -5], keep[1], 2, keep[0])  # the aim is 4.5: 1.5 off, which rounds to 2
    assert run[3] == ([], [5, 0, -5], keep[1], 2, keep[0])  # the aim is 5.25: a quarter off, which is none
    # turned around: the aim goes half of the way from 5.25 to -6, to -0.375, which is 5.375 from the 5 held, or 5
    assert run[4] == (one(-5, 0, 5), [0, 0, 0], [-1, 0, 1], 4, [-3.0, 0.0, 3.0])
    assert run[5] == (one(-3, 0, 3), [-3, 0, 3], [-1, 0, 1], 4, [-3.0, 0.0, 3.0])  # the aim is -3.19
    assert run[6] == (one(-2, 0, 2), [-5, 0, 5], [-1, 0, 1], 4, [-3.0, 0.0, 3.0])  # -4.59 is 1.59 off
    # the aim goes to [-0.80, -3, 3.80], and it is at [-5, 0, 5]: 4.2 off, 3 and 1.2 the other way
    assert run[7] == (one(4, -3, -1), [-1, -3, 4], [1, -1, 1], 6, [3.0, -3.0, 3.0])


def test_a_trader_with_a_band_goes_to_its_edge_at_a_rebalance_and_orders_nothing_inside_it():
    run = run_by_three(reversal_by_three(band=2))
    assert run[1] == (one(4, 0, -4), [4, 0, -4], [1, 0, -1], 2, [3.0, 0.0, -3.0])  # 6 away, the edge is 2 away
    assert run[2] == ([], [4, 0, -4], [1, 0, -1], 2, [3.0, 0.0, -3.0])  # 2 away: inside
    assert run[3] == ([], [4, 0, -4], [1, 0, -1], 2, [3.0, 0.0, -3.0])
    assert run[4] == (one(-8, 0, 8), [-4, 0, 4], [-1, 0, 1], 4, [-3.0, 0.0, 3.0])  # from 10 away to 2 away
    assert run[5][0] == [] and run[6][0] == []
    # the target [3, -6, 3] from [-4, 0, 4]: 7, -6 and -1 away; the last is inside the band
    assert run[7] == (one(5, -4, 0), [1, -4, 4], [1, -1, 1], 6, [3.0, -3.0, 3.0])


def test_a_book_a_rebalance_leaves_with_no_target_is_closed_all_the_way_whatever_the_band():
    trader = reversal_by_three(band=2)
    trader.attach(3)
    moves = []
    # The second rebalance (the fifth row) has scores of 0, -3 and 3: the first book's target is 0, and it holds 4 lots.
    for mids in ([100, 100, 100], [97, 100, 103], [100, 100, 100], [100, 100, 100], [100, 103, 97]):
        out = trader.decide(np.array(mids, dtype=float))
        for move in out:
            trader.position = trader.position + move
        moves.append([m.tolist() for m in out])
    # The first goes to the band's edge, 4 of 6 and 4 of -6. Then the first is closed, not to 2 of it, the second goes
    # to its edge of a target of -6, and the third turns from -4 toward 6, 8 lots to its edge of 2.
    assert moves == [[], one(4, 0, -4), [], [], one(-4, -4, 8)]
    assert trader.position.tolist() == [0, -4, 4]


def test_a_trader_with_the_defaults_and_its_orders_filled_orders_at_a_rebalance_alone():
    run = run_by_three(reversal_by_three())
    # the last is the target [3, -6, 3] from [-6, 0, 6]
    assert [moves for moves, *_ in run] == [[], one(6, 0, -6), [], [], one(-12, 0, 12), [], [], one(9, -6, -3)]
    assert [position for _, position, *_ in run][-1] == [3, -6, 3]


def test_a_trader_that_can_not_yet_trade_orders_nothing_with_a_rate_or_a_band_and_then_trades_at_once():
    for settings in ({"rate": 0.5}, {"band": 1}, {"rate": 0.5, "band": 1}):
        trader = ReversalTrader(lookback=3, factors=0, size=4, **settings)
        assert trader.needed == 4
        out = feed(trader, mids_of(8, seed=3))
        assert out[:3] == [[], [], []] and trader.warming is None
        assert len(out[3]) == 1 and out[3][0].any()  # at the first decision it can


@pytest.mark.parametrize("kind, params", CASES, ids=IDS)
@pytest.mark.parametrize("rate, band", [(0.5, 0.0), (1.0, 2.0), (0.4, 1.0)])
def test_a_cost_aware_trader_goes_toward_the_specs_target_by_the_specs_rule_at_every_decision(kind, params, rate, band):
    history = mids_of(100, seed=11)
    expected = worked_run(kind, history, **params)
    trader = CLASSES[kind](**params, rate=rate, band=band)
    trader.attach(6)
    position, target, scores, entries, states, between, orders = np.zeros(6, dtype=int), None, None, 0, None, 0, 0
    aim = np.zeros(6)
    for t, mids in enumerate(history):
        moves = trader.decide(mids.copy())
        if expected[t] is not None:  # a rebalance: new scores, a new target, new states
            scores, _, target = expected[t]
            sign = np.sign(target)
            entries += int(((sign != 0) & (sign != (0 if states is None else states))).sum())
            states = sign
        if target is None:
            assert moves == [], f"decision {t}"
            continue
        aim = aim_by_hand(aim, target, rate)
        want = toward_by_hand(aim, position, target, band)
        if any(want):
            assert len(moves) == 1 and moves[0].dtype.kind == "i" and moves[0].tolist() == want, f"decision {t}"
            orders += 1
            between += expected[t] is None
        else:
            assert moves == [], f"decision {t}"
        position = position + want
        trader.position = position.copy()
        assert np.allclose(trader.signal(), scores, rtol=1e-8, atol=1e-10), f"decision {t}"
        assert np.array_equal(trader.states, states) and trader.entries == entries, f"decision {t}"
    assert orders >= 5
    if params.get("hold", 1) > 1 and rate < 1:
        assert between >= 3  # it went on toward the target when it did not rebalance


@pytest.mark.parametrize("rate, band", [(0.3, 1.0), (0.5, 0.0), (0.25, 2.0), (1.0, 3.0)])
def test_with_decisions_enough_between_rebalances_it_is_within_the_band_of_every_target_by_the_next(rate, band):
    history = mids_of(80, seed=15)
    params = {"lookback": 4, "factors": 0, "hold": 20, "size": 9.0}  # .75^20 of 18 lots is under a tenth of a lot
    expected = worked_run("reversal", history, **params)
    trader = ReversalTrader(**params, rate=rate, band=band)
    trader.attach(6)
    target, reached = None, 0
    for t, mids in enumerate(history):
        if expected[t] is not None and target is not None:  # about to rebalance: it has got to the last target
            assert np.abs(target - trader.position).max() <= band
            reached += 1
        for move in trader.decide(mids.copy()):
            trader.position = trader.position + move
        if expected[t] is not None:
            target = expected[t][2]
    assert reached >= 3


MAKERS = {
    "reversal": lambda **kw: ReversalTrader(lookback=3, **kw),
    "momentum": lambda **kw: MomentumTrader(lookback=3, **kw),
    "lead-lag": lambda **kw: LeadLagTrader(lookback=3, **kw),
    "flow of trades": lambda **kw: FlowTrader(lookback=3, **kw),
    "flow of queues": lambda **kw: FlowTrader(lookback=3, measure="queues", **kw),
    "base class": lambda **kw: CrossSectionTrader(lookback=3, **kw),
}


@pytest.mark.parametrize("make", MAKERS.values(), ids=MAKERS.keys())
@pytest.mark.parametrize("settings", [
    {"rate": 0}, {"rate": 0.0}, {"rate": -0.1}, {"rate": 1.01}, {"rate": 2}, {"rate": float("nan")},
    {"band": -1}, {"band": -0.001}, {"band": float("nan")},
    {"rate": 0, "band": 1}, {"rate": 1.5, "band": 2}, {"rate": 0.5, "band": -1},
])
def test_a_rate_outside_0_to_1_or_a_band_below_0_is_refused_and_says_what_it_needs(make, settings):
    with pytest.raises(ValueError, match=r"rate.*band"):
        make(**settings)


@pytest.mark.parametrize("make", MAKERS.values(), ids=MAKERS.keys())
@pytest.mark.parametrize("settings", [
    {}, {"rate": 1}, {"rate": 1.0, "band": 0}, {"rate": 0.5}, {"rate": 0.999}, {"rate": 1e-9}, {"band": 0.5},
    {"band": 1000}, {"rate": 0.25, "band": 3},
])
def test_a_rate_up_to_1_and_a_band_from_0_are_accepted(make, settings):
    trader = make(**settings)
    assert trader.rate == settings.get("rate", 1.0) and trader.band == settings.get("band", 0.0)
    assert trader._cost_aware == (trader.rate < 1 or trader.band > 0)


@pytest.mark.parametrize("make", MAKERS.values(), ids=MAKERS.keys())
def test_it_has_no_netted_path_to_refuse_a_rate_or_a_band_on(make):
    assert make(netted=False, rate=0.5, band=1).band == 1


CROSS_KINDS = {
    "reversal": lambda s, x, **kw: x.ReversalTrader(lookback=3, factors=1, pca_window=40, pca_every=10, hold=3,
                                                    interval=2.0, **kw),
    "momentum": lambda s, x, **kw: x.MomentumTrader(lookback=5, skip=2, factors=1, pca_window=40, pca_every=10, hold=3,
                                                    interval=2.0, **kw),
    "lead-lag": lambda s, x, **kw: x.LeadLagTrader(lookback=5, factors=1, pca_window=40, pca_every=10, hold=3,
                                                   interval=2.0, **kw),
    "flow of trades": lambda s, x, **kw: x.FlowTrader(lookback=5, factors=1, pca_window=40, pca_every=10, hold=3,
                                                      interval=2.0, **kw),
    "flow of queues": lambda s, x, **kw: x.FlowTrader(lookback=1, factors=0, measure="queues", hold=1, interval=2.0,
                                                      **kw),
}


@pytest.mark.parametrize("name", CROSS_KINDS)
def test_with_the_defaults_each_cross_sectional_trader_does_number_for_number_what_it_did_before(name):
    assert_as_before(CROSS_KINDS[name])


@cache
def scored(name, rate=1.0, band=0.0):
    return played(CURRENT, CROSS_KINDS[name], rate=rate, band=band)[2]


@pytest.mark.parametrize("settings", SETTINGS, ids=SETTING_IDS)
@pytest.mark.parametrize("name", CROSS_KINDS)
def test_with_a_rate_or_a_band_each_cross_sectional_trader_trades_fewer_lots_than_with_the_defaults_on_one_seed(
        name, settings):
    plain, careful = scored(name), scored(name, **settings)
    assert plain["lots"] > 300
    assert 0 < careful["lots"] < plain["lots"]
    assert careful["entries"] > 0 and careful["costs"] < plain["costs"]
