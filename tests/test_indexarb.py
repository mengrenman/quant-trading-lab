"""Trading an index against its books: the replication, a sample of books, and the trader (lob/indexarb.py)."""

import pickle
from concurrent.futures import ThreadPoolExecutor
from functools import cache

import numpy as np
import pytest

from lob.indexarb import IndexTrader, _IndexChoice, index_basket, replicate_index
from lob.statarb import BookMarket, FactorValues, RelationTrader, score_portfolio, sector_loadings

SMALL = sector_loadings(2, 2)  # four books, a market factor and two sectors
SECTORS = sector_loadings(3, 4)  # twelve books


def market_with(trader, loadings=SMALL, index=True, seed=1, **fair):
    """A market with an index (the sum of the books unless told otherwise) and the trader in it."""
    return BookMarket(FactorValues(loadings, index=index, **fair), seed=seed, traders=[trader])


def first_entry(size, index, seed=1, rate=1.0, window=20):
    """The lots held at the first decision it holds any, in the twelve sectors books and an index of these weights."""
    trader = IndexTrader(window=window, entry=0.5, size=size, interval=1.0, rate=rate)
    market = market_with(trader, SECTORS, index, seed, index_volatility=0.8)
    while not trader.position.any():
        market.step()
        assert market.time < 400
    return trader.position.copy()


def hand_fed(trader, mids):
    """The moves the trader asks for at each of these rows of mids, and what it holds after each, the lots it asks for
    taken at once."""
    moves, held = [], []
    for row in mids:
        asked = trader.decide(np.asarray(row, dtype=float))
        moves.append([move.copy() for move in asked])
        for move in asked:
            trader.position += move
        held.append(trader.position.copy())
    return moves, np.array(held)


def synthetic_mids(spread, seed=5, books=4):
    """The books as independent random walks, and the sum index at those books plus ``spread``."""
    walk = 1000 + np.cumsum(np.random.default_rng(seed).normal(0, 4, (len(spread), books)), axis=0)
    return np.column_stack([walk, walk.sum(axis=1) + np.asarray(spread)])


@cache
def bare_mids(seed=1, decisions=2000, every=2.5):
    """The mids of every book, every ``every`` units, in a bare market of six books in three sectors and their sum."""
    fair = FactorValues(sector_loadings(3, 2), index=True, index_volatility=0.4)
    market = BookMarket(fair, seed=seed)
    mids, until = [], 0.0
    while len(mids) < decisions:
        until += every
        while market.time < until:
            market.step()
        mids.append([book.mid_price for book in market.books])
    return np.array(mids), fair


# ---- replicate_index ----

@pytest.mark.parametrize("weights, row", [
    ([1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, 1.0]),  # the sum of the books: every leg 1
    ([1 / 3, 1 / 3, 1 / 3], [-1 / 3, -1 / 3, -1 / 3, 1.0]),  # weights adding to 1: the index is the largest leg
    ([2.0, 1.0, 1.0], [-1.0, -0.5, -0.5, 0.5]),  # a book above the index: it is the largest leg
    ([0.0, 1.0, 3.0], [0.0, -1 / 3, -1.0, 1 / 3]),
])
def test_replicate_index_gives_the_one_row_of_minus_the_weights_and_the_index_with_its_largest_leg_1(weights, row):
    got = replicate_index(np.zeros((10, 4)), weights)
    assert got.shape == (1, 4) and np.allclose(got[0], row)
    assert np.abs(got).max() == 1.0


@pytest.mark.parametrize("weights", [[1.0] * 6, [0.5, 1.0, 1.5, 0.2, 2.0, 0.8]])
def test_replicate_index_is_a_relation_of_a_market_with_that_index_and_what_it_leaves_is_the_premium(weights):
    f = FactorValues(sector_loadings(2, 3), index=weights, index_volatility=0.5, impact=0.0, loading_sd=0.2)
    row = replicate_index(np.zeros((3, 7)), f.index_weights)
    f.advance_to(50.0)
    assert np.allclose(row @ f.mean_loadings, 0.0, atol=1e-12)
    assert row[0, -1] * f.premium == pytest.approx(float(row[0] @ f.values()), abs=1e-9)
    assert f.premium != 0.0


def test_replicate_index_reads_the_history_for_its_width_alone():
    rng = np.random.default_rng(1)
    first = replicate_index(np.zeros((1, 4)), [1.0, 1.0, 1.0])
    assert np.array_equal(first, replicate_index(rng.normal(3000, 5, (500, 4)), [1.0, 1.0, 1.0]))
    for wrong in (np.zeros((10, 3)), np.zeros((10, 5)), np.zeros(4), np.zeros((2, 3, 4))):
        with pytest.raises(ValueError, match="the history has"):
            replicate_index(wrong, [1.0, 1.0, 1.0])
    for wrong, books in ((np.zeros((10, 3)), 3), (np.zeros((10, 5)), 5), (np.zeros(4), 0), (np.zeros((2, 3, 4)), 0)):
        with pytest.raises(ValueError, match=f"the history has {books} books, and an index of 3 books has 4"):
            replicate_index(wrong, [1.0, 1.0, 1.0])  # what is no table of rows by books has none


@pytest.mark.parametrize("weights", [[1.0, -0.5, 1.0], [0.0, 0.0, 0.0], [1.0, np.nan, 1.0], [1.0, np.inf, 1.0],
                                     [[1.0, 1.0, 1.0]], []])
def test_replicate_index_refuses_weights_that_cannot_make_an_index(weights):
    with pytest.raises(ValueError, match="index weights are one for each book"):
        replicate_index(np.zeros((10, 4)), weights)


# ---- index_basket ----

def forward_selection(levels, legs, index=-1):
    """What forward selection by least squares with a constant gives, on the raw levels, one column at a time."""
    n = levels.shape[1]
    index %= n
    y, chosen = levels[:, index], []
    for _ in range(legs):
        best = None
        for b in (b for b in range(n) if b != index and b not in chosen):
            x = np.column_stack([np.ones(len(y)), levels[:, chosen + [b]]])
            rss = float(((y - x @ np.linalg.lstsq(x, y, rcond=None)[0]) ** 2).sum())
            if best is None or rss < best[0]:
                best = (rss, b)
        chosen.append(best[1])
    x = np.column_stack([np.ones(len(y)), levels[:, chosen]])
    beta = np.linalg.lstsq(x, y, rcond=None)[0][1:]
    row = np.zeros(n)
    row[index], row[chosen] = 1.0, -beta
    return row / np.abs(row).max()


def levels_of(seed=1, books=5, size=600):
    """Random walks for the first ``books`` columns, and as the last the level 3 x the second and 1 x the fourth,
    plus a constant and a little noise."""
    rng = np.random.default_rng(seed)
    walks = 3000 + np.cumsum(rng.normal(0, 5, (size, books)), axis=0)
    index = 3.0 * walks[:, 1] + walks[:, 3] + 50.0 + rng.normal(0, 0.5, size)
    return np.column_stack([walks, index])


@pytest.mark.parametrize("legs", [1, 2, 3, 5])
def test_index_basket_picks_the_books_that_explain_the_index_level_best_by_least_squares_with_a_constant(legs):
    levels = levels_of()
    row = index_basket(levels, legs)
    assert row.shape == (1, 6) and np.allclose(row[0], forward_selection(levels, legs), atol=1e-9)
    assert (row[0, :-1] != 0).sum() == legs  # as many books in the row as it was asked for


def test_index_basket_finds_the_two_books_an_index_is_made_of_and_their_coefficients():
    row = index_basket(levels_of(), 2)[0]
    assert np.nonzero(row[:-1])[0].tolist() == [1, 3]
    assert row[1] == -1.0 and row[3] == pytest.approx(-1 / 3, abs=0.01) and row[5] == pytest.approx(1 / 3, abs=0.01)


def test_index_basket_takes_a_constant_in_its_levels_and_is_not_moved_by_one_added_to_a_book():
    levels = levels_of()
    shifted = levels + np.array([100.0, -2000.0, 5.0, 0.0, 77.0, 1e4])
    assert np.allclose(index_basket(levels, 3), index_basket(shifted, 3), atol=1e-9)


def test_index_basket_adds_the_book_that_lowers_the_residual_sum_of_squares_most_and_keeps_it():
    rng = np.random.default_rng(7)
    a, b = np.cumsum(rng.normal(0, 1, 800)), np.cumsum(rng.normal(0, 1, 800))
    near = a + b + 0.3 * rng.normal(0, 1, 800)  # alone it explains the index best, and it is no part of it
    levels = np.column_stack([a, b, near, a + b])
    one, two = index_basket(levels, 1)[0], index_basket(levels, 2)[0]
    assert np.nonzero(one[:-1])[0].tolist() == [2]
    assert 2 in np.nonzero(two[:-1])[0] and len(np.nonzero(two[:-1])[0]) == 2  # the first pick stays: not {a, b}
    assert np.allclose(two, forward_selection(levels, 2), atol=1e-9)


def test_index_basket_takes_the_last_column_for_the_index_unless_told_otherwise():
    levels = levels_of()
    last = index_basket(levels, 2)
    assert np.array_equal(last, index_basket(levels, 2, index=-1)) and np.array_equal(last, index_basket(levels, 2, 5))
    first = np.column_stack([levels[:, -1], levels[:, :-1]])  # the index moved to the first column
    row = index_basket(first, 2, index=0)
    assert np.allclose(row[0], np.append(last[0, -1], last[0, :-1]), atol=1e-12)
    assert np.array_equal(index_basket(first, 2, index=-6), row)  # a column counted from the end is the same one


@pytest.mark.parametrize("legs, index", [(0, -1), (-1, -1), (6, -1), (7, -1), (2, 6), (2, -7)])
def test_index_basket_refuses_legs_below_1_or_above_the_other_books_and_an_index_that_is_no_column(legs, index):
    with pytest.raises(ValueError):
        index_basket(levels_of(), legs, index)


def test_index_basket_takes_all_the_other_books_when_asked_for_them_all():
    levels = levels_of()
    row = index_basket(levels, 5)
    assert (row[0, :-1] != 0).all() and np.allclose(row[0], forward_selection(levels, 5), atol=1e-9)


@pytest.mark.parametrize("first", [0, 1])
def test_index_basket_takes_the_lowest_column_of_books_that_explain_the_index_equally_well(first):
    rng = np.random.default_rng(4)
    a = 3000.0 + np.cumsum(rng.normal(0, 5, 400))
    walk = 3000.0 + np.cumsum(rng.normal(0, 5, 400))
    twins = [a, a.copy()]
    # the index is 2 a and 7: books that are the same series as a explain it exactly as well as one another
    levels = np.column_stack(([walk] if first else []) + twins + [2.0 * a + 7.0])
    row = index_basket(levels, 1)[0]
    expected = np.zeros(levels.shape[1])
    expected[[-1, first]] = [0.5, -1.0]  # the lowest of the twins, at a coefficient of 2, scaled to its largest leg 1
    assert np.allclose(row, expected, atol=1e-9)


def test_index_basket_asked_for_two_books_that_are_one_series_splits_the_coefficient_between_them():
    for seed in range(1, 7):
        rng = np.random.default_rng(seed)
        a = 3000.0 + np.cumsum(rng.normal(0, 5, 600))
        y = 3.0 * a + 40.0 + rng.normal(0, 1, 600)
        row = index_basket(np.column_stack([a, a.copy(), y]), 2)[0]
        da, dy = a - a.mean(), y - y.mean()
        beta = (da @ dy) / (da @ da)  # the slope of y on a by hand: the twins have it between them
        expected = np.array([-beta / 2, -beta / 2, 1.0])
        assert np.allclose(row, expected / np.abs(expected).max(), atol=1e-6), seed


def test_index_basket_takes_a_book_that_explains_a_little_more_over_a_copy_of_a_book_it_has():
    rng = np.random.default_rng(1)
    a = 3000.0 + np.cumsum(rng.normal(0, 5, 400))
    y = 3.0 * a + 40.0 + rng.normal(0, 1, 400)
    da, dy = a - a.mean(), y - y.mean()
    left = dy - da * (da @ dy) / (da @ da)  # what is left of the index's level when a is taken
    left /= np.linalg.norm(left)
    w = rng.normal(0, 1, 400)
    w -= w.mean()
    u = w - (w @ left) * left - da * (da @ w) / (da @ da)
    c = (u - u.mean()) + 0.3 * left  # a book that has next to nothing to add to a: a sliver of what a leaves
    row = index_basket(np.column_stack([a, a.copy(), c, y]), 2)[0]
    assert np.nonzero(row[:-1])[0].tolist() == [0, 2]  # the copy of a adds exactly nothing, and c a little
    assert row[0] == -1.0 and row[3] == pytest.approx(1 / 3, abs=1e-3)


def test_index_basket_picks_one_book_a_sector_with_a_third_of_the_scale_each_in_the_mids_of_a_sectors_market():
    mids, f = bare_mids()
    scale = f.scales[-1]
    row = index_basket(mids, 3)[0]
    chosen = np.nonzero(row[:-1])[0]
    assert sorted(book // 2 for book in chosen) == [0, 1, 2]  # books 0-1, 2-3 and 4-5 are the three sectors
    assert -row[chosen] / row[-1] / scale == pytest.approx(1 / 3, abs=0.08)  # each is a third of the scale
    assert np.abs(row).max() == 1.0 and row[-1] > 0
    assert np.linalg.norm(row @ f.mean_loadings) < 0.5  # what is left of the factors in the spread is little


@pytest.mark.parametrize("legs", [1, 2])
def test_index_basket_with_fewer_legs_than_the_tables_rank_leaves_a_factor_in_the_row(legs):
    mids, f = bare_mids()
    assert np.linalg.matrix_rank(f.mean_loadings) == 3 and f.rank == 4  # the sector factors add up to the market's
    row = index_basket(mids, legs)
    assert np.linalg.norm(row @ f.mean_loadings) > 1.5  # the row is far from a relation: a factor is left in it
    assert np.linalg.norm(index_basket(mids, 3) @ f.mean_loadings) < 0.5  # as many legs as the rank leave little


# ---- IndexTrader: what it is made of ----

def test_an_index_trader_needs_no_choose_or_weights_and_has_the_window_for_its_formation_unless_told():
    trader = IndexTrader()
    assert isinstance(trader, RelationTrader) and (trader.legs, trader.hedged) == (0, True)
    assert isinstance(trader.choose, _IndexChoice) and trader.weights is None and trader.refit is None
    assert trader.formation == trader.window == 400
    assert IndexTrader(window=50).formation == 50 and IndexTrader(window=50, formation=300).formation == 300
    assert trader.warm_up == trader.taking(401) and IndexTrader(window=50, formation=300).warm_up == trader.taking(301)
    with pytest.raises(ValueError, match="at least as long as the window"):
        IndexTrader(window=50, formation=40)


@pytest.mark.parametrize("given", [{"choose": replicate_index}, {"weights": [[-1.0, -1.0, 1.0]]},
                                   {"legs": -1}, {"legs": 1.5}, {"legs": True}, {"legs": "2"}])
def test_an_index_trader_refuses_a_choose_or_weights_of_its_own_and_legs_that_cannot_be(given):
    with pytest.raises(ValueError):
        IndexTrader(**given)


def test_an_index_choice_given_no_legs_replicates_the_index_at_its_weights_and_with_legs_takes_a_basket():
    history = levels_of()[:, [0, 1, 2, 5]]  # three books and an index that is no sum of them
    assert np.array_equal(_IndexChoice(weights=np.array([1.0, 2.0, 1.0]))(history),
                          replicate_index(history, [1.0, 2.0, 1.0]))
    assert np.array_equal(_IndexChoice(2, weights=np.array([1.0, 2.0, 1.0]))(history), index_basket(history, 2))


def test_a_market_with_no_index_book_refuses_an_index_trader_when_it_is_added_and_does_not_add_it():
    fair = FactorValues(SMALL)
    trader = IndexTrader(window=10)
    with pytest.raises(ValueError, match="this market has no index book"):
        BookMarket(fair, traders=[trader])
    market = BookMarket(FactorValues(SMALL))
    with pytest.raises(ValueError, match="this market has no index book"):
        market.add(IndexTrader(window=10, legs=2))  # a sample would take the last book for an index, and does not
    assert market.traders == [] and trader.position is None
    market.add(RelationTrader(weights=[[1.0, -1.0, 0.0, 0.0]]))  # the market is as it was: it takes other traders
    assert len(market.traders) == 1


def test_an_index_trader_takes_the_weights_of_the_index_from_the_market_and_keeps_its_own_copy():
    weights = np.array([0.5, 1.0, 1.5, 2.0])
    trader = IndexTrader(window=10)
    market = market_with(trader, index=weights)
    assert np.array_equal(trader.choose.weights, weights) and trader.choose.weights is not market.fair.index_weights
    assert market.traders == [trader] and trader.position.shape == (5,)
    market.fair.index_weights[0] = 9.0  # the weights it has are its own
    assert trader.choose.weights[0] == 0.5


def test_an_index_trader_asked_for_more_legs_than_the_index_has_books_is_refused_when_added():
    with pytest.raises(ValueError, match="5 legs from an index of 4 books"):
        market_with(IndexTrader(window=10, legs=5))
    market_with(IndexTrader(window=10, legs=4))  # all the books is a sample of them


def test_an_index_trader_given_no_check_and_no_weights_says_the_market_has_no_index_book_where_it_chooses():
    trader = IndexTrader(window=5)
    trader.attach(5)  # added by a path that does not check
    with pytest.raises(ValueError, match="this market has no index book"):
        for _ in range(6):
            trader.decide(np.ones(5))


# ---- IndexTrader: what it trades ----

@pytest.mark.parametrize("weights", [True, [0.5, 1.0, 1.5, 2.0]])
def test_with_legs_0_an_index_trader_trades_the_one_row_replicate_index_gives_at_the_markets_weights(weights):
    trader = IndexTrader(window=15, entry=0.5, size=4, interval=1.0, record=True)
    market = market_with(trader, index=weights)
    while not trader.position.any():
        market.step()
        assert market.time < 200
    expected = replicate_index(np.zeros((1, 5)), market.fair.index_weights)
    assert trader.fits == 1 and np.array_equal(trader.weights, expected)
    assert np.array_equal(trader.position, trader.states[0] * np.rint(4 * expected[0]).astype(int))


def test_with_legs_above_0_an_index_trader_trades_the_row_index_basket_gives_from_the_mids_it_has_seen():
    trader = IndexTrader(window=60, formation=120, entry=0.5, size=5, interval=1.0, legs=2, record=True)
    market = market_with(trader, sector_loadings(2, 3))
    while trader.fits == 0:
        market.step()
        assert market.time < 400
    mids = np.array([record[1] for record in trader.history[:120]])
    expected = index_basket(mids, 2)
    assert len(trader.history) == 120 and np.array_equal(trader.weights, expected)
    assert (expected[0, :-1] != 0).sum() == 2 and expected[0, -1] != 0
    while not trader.position.any():
        market.step()
        assert market.time < 800
    assert np.array_equal(trader.position, trader.states[0] * np.rint(5 * expected[0]).astype(int))


@pytest.mark.parametrize("size", [1, 3, 5, 6, 7, 9, 12, 13, 18, 24, 30])
def test_with_the_sum_index_a_replication_holds_a_lot_of_each_book_for_each_lot_of_the_index_at_any_size(size):
    position = first_entry(size, True)
    index = position[-1]
    assert abs(index) == size and (position[:-1] == -index).all()


@pytest.mark.parametrize("size, each", [(1, 0), (3, 0), (5, 0), (6, 0), (7, 1), (9, 1), (12, 1), (13, 1), (18, 2),
                                        (24, 2), (30, 2)])
def test_with_weights_of_1_over_m_each_leg_rounds_alone_so_only_multiples_of_12_hold_the_hedge_asked_for(size, each):
    position = first_entry(size, np.full(12, 1 / 12))
    index = position[-1]
    assert abs(index) == size
    assert (position[:-1] == -np.sign(index) * each).all()  # a twelfth of the size, rounded (a half to the even)


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("band", [0.0, 2.0])
def test_with_the_sum_index_the_hedge_is_exact_at_every_decision_at_a_rate_of_a_quarter_while_it_enters(seed, band):
    trader = IndexTrader(window=20, entry=1.0, size=12, interval=1.0, rate=0.25, band=band, record=True)
    market = market_with(trader, SECTORS, True, seed, index_volatility=0.8)
    market.run(250.0)
    held = np.array([record[2] for record in trader.history])
    index, books = held[:, -1], held[:, :-1]
    assert (index != 0).sum() > 100 and trader.entries > 10
    assert (books == -index[:, None]).all()  # at every decision, each book against the index, a lot for a lot
    assert {int(v) for v in np.abs(index)} >= set(range(0, 10))  # and it went through the sizes in between


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_with_weights_of_1_over_m_a_rate_of_a_quarter_leaves_the_index_with_no_hedge_at_many_decisions(seed):
    trader = IndexTrader(window=20, entry=1.0, size=12, interval=1.0, rate=0.25, record=True)
    market = market_with(trader, SECTORS, np.full(12, 1 / 12), seed, index_volatility=0.8)
    market.run(250.0)
    held = np.array([record[2] for record in trader.history])
    index, books = held[:, -1], held[:, :-1]
    naked = (index != 0) & (books == 0).all(axis=1)
    exact = (index != 0) & (books == -index[:, None] / 12).all(axis=1)
    assert naked.sum() > 0.1 * (index != 0).sum()  # the index leg leads: a lot of the decisions hold it with no book
    assert exact.sum() < 0.5 * (index != 0).sum()


@pytest.mark.parametrize("rate, band, legs", [(1.0, 0.0, 0), (0.25, 0.0, 0), (0.5, 1.0, 0), (1.0, 0.0, 3),
                                              (0.4, 0.0, 3)])
def test_an_index_trader_that_is_not_hedged_reads_the_hedged_traders_spread_and_asks_for_its_lots_of_the_index_only(
        rate, band, legs):
    spread = np.random.default_rng(2).normal(0, 1, 160)
    spread[60:80] += 6.0  # a swing high, and then a swing low
    spread[110:130] -= 6.0
    mids = synthetic_mids(spread)
    runs = {}
    for hedged in (True, False):
        trader = IndexTrader(window=30, entry=2.0, size=6, legs=legs, rate=rate, band=band, hedged=hedged)
        market_with(trader, SMALL)
        runs[hedged] = (trader, *hand_fed(trader, mids))
    (hedged, hedged_moves, hedged_held), (alone, alone_moves, alone_held) = runs[True], runs[False]
    assert hedged.entries > 1 and alone.entries == hedged.entries and hedged.fits == alone.fits == 1
    assert np.array_equal(hedged.weights, alone.weights)  # the relation it reads is whole, and is kept so
    assert np.array_equal(hedged.states, alone.states) and np.array_equal(hedged.signal(), alone.signal())
    assert hedged_held[:, :-1].any() and not alone_held[:, :-1].any()  # one holds books, the other never does
    assert np.array_equal(hedged_held[:, -1], alone_held[:, -1]) and alone_held[:, -1].any()
    for with_hedge, without in zip(hedged_moves, alone_moves, strict=True):  # the same lots of the index at every step
        assert sum(int(m[-1]) for m in without) == sum(int(m[-1]) for m in with_hedge)
        assert all(not m[:-1].any() for m in without)


def test_an_index_trader_that_is_not_hedged_reads_the_spread_with_every_book_at_every_decision():
    spread = np.random.default_rng(2).normal(0, 1, 100)
    mids = synthetic_mids(spread)
    trader = IndexTrader(window=30, entry=2.0, size=6, hedged=False)
    market_with(trader, SMALL)
    hand_fed(trader, mids)
    window = mids[-30:] @ np.array([-1.0, -1.0, -1.0, -1.0, 1.0])  # the relation of every book, the index not alone
    expected = (window[-1] - window.mean()) / window.std()
    assert trader.z[0] == pytest.approx(expected, rel=1e-9)
    assert np.array_equal(trader.weights, replicate_index(mids, np.ones(4)))  # whole after every decision, as it was


def test_an_index_trader_that_is_not_hedged_holds_no_book_but_the_index_in_a_market_where_the_hedged_one_holds_both():
    held = {}
    for hedged in (True, False):
        trader = IndexTrader(window=60, entry=2.0, interval=5.0, hedged=hedged, record=True)
        market = market_with(trader, SMALL, True, 1, index_volatility=3.0)
        score = score_portfolio(market, trader, trader.warm_up, 600)
        held[hedged] = (np.array([record[2] for record in trader.history]), score)
    alone, hedged = held[False][0], held[True][0]
    assert (alone[:, :-1] == 0).all() and (alone[:, -1] != 0).any() and held[False][1]["lots"] > 0
    assert (hedged[:, :-1] != 0).any() and (hedged[:, :-1] == -hedged[:, -1:]).all()
    assert held[True][1]["exposure"] == 0.0 and held[False][1]["exposure"] > 5.0  # it holds what the index holds


# ---- IndexTrader: learning, warming and a refit ----

def test_an_index_trader_learns_the_mids_it_is_given_and_trades_at_its_first_decision_after():
    trader = IndexTrader(window=40, entry=2.0, size=3, interval=1.0)
    market_with(trader, SMALL)
    mids = synthetic_mids(np.random.default_rng(4).normal(0, 1, 100))
    assert trader.warming == (0, 40)
    assert trader.learn(list(mids[:25])) == 25 and trader.warming == (25, 40)
    assert trader.learn(list(mids[25:99])) == 40 and len(trader._mids) == 40  # the last 40: it keeps no more
    assert trader.warming == (40, 40)
    assert trader.fits == 0 and trader.weights is None
    moves = trader.decide(mids[99])  # its first decision: it has what it needs, and trades if the spread says so
    assert trader.fits == 1 and trader.weights is not None and trader.z is not None and trader.warming is None
    assert (trader.states != 0).any() == bool(moves)


def test_an_index_trader_with_a_formation_longer_than_its_window_is_warming_until_it_has_chosen():
    trader = IndexTrader(window=20, formation=60, interval=1.0, legs=2)
    market_with(trader, SMALL)
    mids = synthetic_mids(np.random.default_rng(4).normal(0, 1, 80))
    hand_fed(trader, mids[:59])
    assert trader.warming == (59, 60) and trader.weights is None
    hand_fed(trader, mids[59:60])
    assert trader.fits == 1 and trader.warming is None and (trader.weights[0, :-1] != 0).sum() == 2


def swing_mids(legs):
    """Mids of four books and an index whose spread is small and alternating, then swings high at decision 39 and
    stays above its mean: a short position is held from the swing on, with a z-score between 0 and 2. The
    index is the sum of the books, or with ``legs`` 3 of the first three, so that the sample has books to find."""
    spread = np.where(np.arange(120) % 2 == 0, 1.0, -1.0)
    spread[39], spread[40:] = 8.0, 4.0
    walk = 1000 + np.cumsum(np.random.default_rng(5).normal(0, 4, (120, 4)), axis=0)
    return np.column_stack([walk, walk[:, :legs or 4].sum(axis=1) + spread])


@pytest.mark.parametrize("legs, kept", [(0, False), (3, True)])
def test_a_refit_of_the_full_replication_goes_flat_where_a_sample_on_the_same_books_keeps_its_state(legs, kept):
    trader = IndexTrader(window=30, entry=2.0, size=6, legs=legs, refit=12)
    market_with(trader, SMALL)
    seen = []
    for row in swing_mids(legs):
        trader.decide(row)
        seen.append((trader.fits, trader.states, trader.z, trader.weights))
    assert [i for i in range(1, 120) if seen[i][0] > seen[i - 1][0]] == [29, 41, 53, 65, 77, 89, 101, 113]
    (_, states, _, weights), (_, after, z, refit) = seen[40], seen[41]  # the first refit, just after the swing
    assert states[0] == -1 and 0 < z[0] < 2.0  # a short position held, with a z-score between the exit and the entry
    legs_in = [np.nonzero(w[0])[0].tolist() for w in (weights, refit)]
    assert legs_in[0] == legs_in[1] and len(legs_in[0]) == (5 if legs == 0 else 4)  # on the same books, all or fewer
    assert after[0] == (-1 if kept else 0)  # held by the sample, closed for the relation that has every book


def test_an_index_trader_chooses_elsewhere_when_given_a_way_to_and_its_choice_can_be_sent_to_a_worker():
    spread = np.random.default_rng(2).normal(0, 1, 60)
    mids = synthetic_mids(spread)
    with ThreadPoolExecutor(1) as pool:
        trader = IndexTrader(window=30, entry=2.0, size=3, defer=lambda choose, history: pool.submit(choose, history))
        market_with(trader, SMALL)
        for row in mids[:30]:
            trader.decide(row)
            if trader._pending is not None:
                trader._pending.result()  # wait for the choice, which the trader takes up at its next decision
        trader.decide(mids[30])
    assert trader.fits == 1 and np.array_equal(trader.weights, replicate_index(mids, np.ones(4)))
    sent = pickle.loads(pickle.dumps(trader.choose))
    assert np.array_equal(sent(mids), trader.weights) and sent.weights.tolist() == [1.0] * 4


# ---- What it earns ----

def test_an_index_trader_earns_before_costs_at_a_large_index_volatility_on_seeds_1_to_8_and_loses_without_one():
    far, control = [], []
    for seed in range(1, 9):
        for dial, kept in ((dict(index_volatility=3.0), far), (dict(index_volatility=0.0, impact=0.0), control)):
            trader = IndexTrader(window=60, entry=2.0, interval=5.0)
            market = market_with(trader, SMALL, True, seed, **dial)
            score = score_portfolio(market, trader, trader.warm_up, 800)
            assert score["exposure"] == 0.0 and score["lots"] > 0
            kept.append(score)
    assert all(s["before_costs"] > 0 and s["pnl"] > 0 for s in far)  # a premium to find: it earns, costs and all
    assert all(s["pnl"] < 0 for s in control)  # none to find: the mids' own bounce, before costs, is not enough
    assert np.mean([s["before_costs"] for s in far]) > 1.5 * np.mean([s["before_costs"] for s in control])
