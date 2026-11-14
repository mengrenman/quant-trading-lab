"""An index book: one more book whose fair value is the weighted sum of the others' and a premium (lob/statarb.py)."""

import math

import numpy as np
import pytest

from lob.deals import Deals
from lob.order import Side, market_order
from lob.pairs import BETA_FLOOR, FLOOR, AnchoredZIModel
from lob.statarb import BLOCK, BookMarket, FactorValues, best_baskets, johansen_relations, sector_loadings

SMALL = sector_loadings(2, 3)  # six books, a market factor and two sectors
SECTORS = sector_loadings(3, 4)  # twelve books
MIXED = np.array([[1.0, 0.2], [0.8, 0.9], [1.2, -0.5], [0.6, 0.4], [1.1, 1.0], [0.9, -0.2]])


def path(fair, steps):
    """The fair values after each of ``steps`` grid steps."""
    out = []
    for i in range(1, steps + 1):
        fair.advance_to(i * fair.step)
        out.append(list(fair.fairs))
    return np.array(out)


def premium_by_hand(seed, volatility, scale, half_life, step, steps):
    """The premium after each of ``steps`` grid steps from the stream ``[seed, 3]``, one shock a step in blocks of
    ``BLOCK``: an own part's decay and shock, with the volatility times the scale for the volatility."""
    rng = np.random.default_rng([seed, 3])
    block = rng.standard_normal(BLOCK)
    kappa = math.log(2) / half_life if half_life else 0.0
    decay = math.exp(-kappa * step)
    sd = volatility * scale * (math.sqrt((1 - decay ** 2) / (2 * kappa)) if kappa else math.sqrt(step))
    premium, out = 0.0, []
    for i in range(steps):
        if i and i % BLOCK == 0:
            block = rng.standard_normal(BLOCK)
        premium = premium * decay + sd * block[i % BLOCK]
        out.append(premium)
    return np.array(out)


# ---- The book ----

def test_a_market_with_an_index_has_one_more_book_the_last_and_a_model_for_it():
    f = FactorValues(SMALL, index=True)
    assert (f.n, f.m, f.k) == (7, 6, 3)
    assert f.loadings.shape == f.mean_loadings.shape == (7, 3)
    assert f.level.shape == f.common.shape == f.own.shape == (7,) and len(f.fairs) == 7
    assert f.index_weights.tolist() == [1.0] * 6
    assert f.scales.tolist() == [1.0] * 6 + [6.0]
    market = BookMarket(f, seed=1)
    assert market.n == 7 and len(market.books) == len(market.models) == 7
    assert market.flow.shape == market.volume.shape == (7,)
    assert all(book.depth_profile(Side.BID, 20) and book.depth_profile(Side.ASK, 20) for book in market.books)
    market.run(50.0)  # and it trades, in every book
    assert len(market.marks()) == 7 and market.events > 0


def test_a_market_without_an_index_has_none_of_it():
    f = FactorValues(SMALL)
    assert (f.n, f.m) == (6, 6) and f.index_weights is None and f.scales is None and f.premium == 0.0
    assert f.loadings.shape == (6, 3) and f.own.shape == (6,) and len(f.fairs) == 6
    assert f.index_volatility == 0.0 and f._prng is None and f._pz is None
    f.advance_to(10.0)
    assert f.premium == 0.0 and f.own.shape == (6,) and f.exposures() is f.loadings


def test_a_market_makes_the_premiums_generator_only_with_an_index(monkeypatch):
    made = []
    real = np.random.default_rng
    monkeypatch.setattr(np.random, "default_rng", lambda seed=None: made.append(seed) or real(seed))
    BookMarket(FactorValues(SMALL), seed=5)
    assert made == [[1, 1], [5, 1], [5, 2]]
    made.clear()
    BookMarket(FactorValues(SMALL, index=True, index_volatility=0.0), seed=5)
    assert made == [[1, 1], [1, 3], [5, 1], [5, 3], [5, 2]]  # one for the premium, whatever its volatility


def test_the_sum_index_has_a_model_that_reaches_8_ticks_a_book_and_a_starting_book_spaced_by_the_sum():
    f = FactorValues(SECTORS, index=True)
    market = BookMarket(f, seed=1)
    assert [model.reach for model in market.models] == [8] * 12 + [96]
    book = market.books[12]
    for d, (price, lots) in enumerate(book.depth_profile(Side.BID, 8), start=1):
        assert (price, lots) == (36000 - 12 * d, 10)
    for d, (price, lots) in enumerate(book.depth_profile(Side.ASK, 8), start=1):
        assert (price, lots) == (36000 + 12 * d, 10)
    other = market.books[0].depth_profile(Side.BID, 8)
    assert [price for price, _ in other] == [3000 - d for d in range(1, 9)]  # the others are as they were


def test_the_mid_of_the_sum_index_follows_the_sum_of_the_books_mids():
    f = FactorValues(SECTORS, index=True, index_volatility=0.1)
    frame = BookMarket(f, seed=1).run(1_000.0)
    mids = frame[[f"mid_{i}" for i in range(13)]].to_numpy()
    assert not np.isnan(mids).any()  # no side of any book was ever empty
    total, index = mids[100:, :12].sum(axis=1), mids[100:, 12]
    assert abs((index - total).mean()) < 6 and (index - total).std() < 30  # against a sum that moves by hundreds
    assert np.polyfit(total, index, 1)[0] == pytest.approx(1.0, abs=0.1)  # one lot of the index for a lot of each
    assert np.corrcoef(np.diff(total[::20]), np.diff(index[::20]))[0, 1] > 0.93  # over 20 units, as a book follows


def test_index_weights_of_1_over_m_give_scale_1_and_an_ordinary_book():
    for books in (3, 6, 7, 12):  # for 7 the weights add up to a hair under 1
        f = FactorValues(np.ones((books, 1)), index=[1 / books] * books)
        assert f.scales[-1] == 1.0 and f.index_weights.tolist() == [1 / books] * books
        market = BookMarket(f, seed=2)
        assert market.models[-1].reach == AnchoredZIModel().reach == 8
        for side in (Side.BID, Side.ASK):  # the starting book a tick apart, as every book's is
            assert market.books[-1].depth_profile(side, 20) == market.books[0].depth_profile(side, 20)
        assert len(market.books[-1].depth_profile(Side.BID, 20)) == 8
    f = FactorValues(SMALL, index=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # one book alone, and a premium
    assert f.scales[-1] == 1.0 and f.level[-1] == 3000.0


def test_weights_that_add_up_to_1_within_a_billionth_are_a_book_like_the_others_and_a_hair_further_below_are_not():
    for off in (-9e-10, -5e-10, 0.0, 5e-10, 9e-10):  # the sum is 1 plus this, to the round-off of a sum
        f = FactorValues(SMALL, index=[1 / 6] * 5 + [1 / 6 + off])
        assert f.scales[-1] == 1.0 and BookMarket(f).models[-1].reach == 8
    for off in (1.5e-9, 1e-6):
        scale = FactorValues(SMALL, index=[1 / 6] * 5 + [1 / 6 + off]).scales[-1]
        assert scale == pytest.approx(1.0 + off, rel=0, abs=1e-12)  # not snapped to 1
    for off in (-1.5e-9, -1e-6):
        with pytest.raises(ValueError, match="add up to 1 or more"):
            FactorValues(SMALL, index=[1 / 6] * 5 + [1 / 6 + off])


def test_other_weights_are_kept_and_the_index_level_is_their_sum_of_the_levels():
    c = [1.0, 2.0, 0.5, 0.0, 3.0, 0.5]
    f = FactorValues(SMALL, level=[1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0], index=c)
    assert f.index_weights.tolist() == c and f.scales[-1] == 7.0
    assert f.level[-1] == np.dot(c, [1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0])
    assert f.fairs[-1] == f.level[-1] and f.fairs[:6] == f.level[:6].tolist()
    assert np.allclose(f.loadings[-1], np.dot(c, SMALL)) and np.allclose(f.mean_loadings[-1], np.dot(c, SMALL))
    assert [model.reach for model in BookMarket(f).models] == [8] * 6 + [56]


@pytest.mark.parametrize("weights, message", [
    ([1.0, 1.0, 1.0, 1.0, 1.0, -0.5], "0 or more"),
    ([0.0] * 6, "more than 0"),
    ([1.0] * 5, "one weight for each of the 6 books"),
    ([1.0] * 7, "one weight for each of the 6 books"),
    ([[1.0] * 6], "one weight for each of the 6 books"),
    ([1.0, 1.0, 1.0, 1.0, 1.0, math.nan], "0 or more"),
    ([1.0, 1.0, 1.0, 1.0, 1.0, math.inf], "0 or more"),
    ([0.1] * 6, "add up to 1 or more"),
    ([0.5, 0.4, 0.0, 0.0, 0.0, 0.0], "add up to 1 or more"),
])
def test_weights_that_cannot_make_an_index_are_refused(weights, message):
    with pytest.raises(ValueError, match=message):
        FactorValues(SMALL, index=weights)


@pytest.mark.parametrize("volatility", [-0.1, math.nan, math.inf])
def test_a_volatility_of_the_premium_below_0_is_refused(volatility):
    with pytest.raises(ValueError, match="volatility must be 0 or more"):
        FactorValues(SMALL, index=True, index_volatility=volatility)


def test_a_level_of_the_books_is_given_as_one_number_or_one_a_book_and_not_one_for_the_index():
    assert FactorValues(SMALL, level=1000.0, index=True).level.tolist() == [1000.0] * 6 + [6000.0]
    with pytest.raises(ValueError):
        FactorValues(SMALL, level=[1000.0] * 7, index=True)


def test_deals_are_kept_as_given_and_at_a_rate_of_0_make_none():
    settings = Deals(rate=0.0)
    f = FactorValues(SMALL, deals=settings)
    assert f.deal_settings is settings and f.deals == [] and f.ended == [] and f.gapped == []
    assert FactorValues(SMALL).deal_settings is None
    g = FactorValues(SMALL, index=True, deals=settings)
    g.advance_to(50.0)
    assert g.deal_settings is settings and g.gapped == []


# ---- The fair value ----

@pytest.mark.parametrize("make", [
    lambda: FactorValues(SMALL, index=True, index_volatility=0.5),
    lambda: FactorValues(SMALL, index=[0.5, 0.5, 1.0, 0.0, 1.0, 0.0], index_volatility=0.3, loading_sd=0.2,
                         loading_half_life=30.0),
    lambda: FactorValues(MIXED, index=[0.2, 0.2, 0.2, 0.2, 0.2, 0.4], index_volatility=1.0, half_life=None),
], ids=["sum", "weights with drifting loadings", "no half-life"])
def test_the_index_fair_value_is_the_weighted_sum_of_the_books_and_its_premium(make):
    f = make()
    c, m = f.index_weights, f.m
    f.reseed(3)
    for i in range(1, 801):
        f.advance_to(i * f.step)
        assert f.fairs[m] == pytest.approx(float(c @ f.fairs[:m]) + f.premium, abs=1e-9)
        assert f.fairs[m] == pytest.approx(f.level[m] + f.common[m] + f.own[m], abs=1e-8)  # its parts say the same
        assert f.common[m] == pytest.approx(float(c @ f.common[:m]), abs=1e-8)
    assert f.premium != 0.0 and abs(f.fairs[m] - float(c @ f.fairs[:m])) > 0.1


def test_the_premium_has_the_sd_and_half_life_it_is_given_and_times_the_scale():
    step, half_life = 1.0, 5.0
    for books, scale in ((SMALL, 6.0), (np.ones((2, 1)), 2.0)):
        f = FactorValues(books, step=step, half_life=half_life, index=True, index_volatility=0.5, impact=0.0)
        f.reseed(1)
        seen = []
        for i in range(1, 40_001):
            f.advance_to(float(i))
            seen.append(f.premium)
        seen = np.array(seen)
        stationary = 0.5 * scale / math.sqrt(2 * math.log(2) / half_life)  # an own part's, times the scale
        assert seen.std() == pytest.approx(stationary, rel=0.08)
        for lag, left in ((5, 0.5), (10, 0.25)):
            assert (seen[:-lag] * seen[lag:]).mean() / (seen ** 2).mean() == pytest.approx(left, abs=0.04)


@pytest.mark.parametrize("seed, volatility, scale, half_life", [(1, 0.5, 6.0, 50.0), (2, 0.2, 6.0, None),
                                                                (3, 1.0, 6.0, 7.0)])
def test_the_premium_is_a_stream_of_its_own_by_hand(seed, volatility, scale, half_life):
    f = FactorValues(SMALL, step=0.5, half_life=half_life, index=True, index_volatility=volatility, impact=0.0)
    f.reseed(seed)
    by_hand = premium_by_hand(seed, volatility, scale, half_life, 0.5, 700)  # across two refills of the shocks
    for i in range(1, 701):
        f.advance_to(i * f.step)
        assert f.premium == pytest.approx(by_hand[i - 1], rel=1e-12, abs=1e-15)
    assert f._prng.bit_generator.state != f._rng.bit_generator.state


def test_the_premium_has_no_shocks_at_a_volatility_of_0_and_moves_only_with_a_lot_taken():
    f = FactorValues(SMALL, index=True, index_volatility=0.0, impact=0.05)
    for i in range(1, 201):
        f.advance_to(i * f.step)
        assert f.premium == 0.0 and f.own[6] == pytest.approx(float(f.own[:6].sum()), abs=1e-12)
    f.trade(6, 4)  # a lot taken in the index moves the premium by its scale times the impact
    assert f.premium == pytest.approx(6.0 * 0.05 * 4)
    f.trade(2, 3)  # one in a book does not move it
    assert f.premium == pytest.approx(6.0 * 0.05 * 4)
    first = f.premium
    f.advance_to(f.step * 201)
    assert f.premium == pytest.approx(first * f._decay) and f.premium != 0.0  # and it reverts
    assert f.own[6] == pytest.approx(float(f.own[:6].sum()) + f.premium, abs=1e-9)


def test_the_constituents_move_as_they_do_without_the_index_and_the_streams_are_their_own():
    kinds = [dict(), dict(loading_sd=0.3, loading_half_life=40.0), dict(trend=[0.0, 0.1, 0.1], trend_half_life=100.0),
             dict(half_life=None, own_volatility=1.5)]
    for seed in (1, 2):
        for extra in kinds:
            plain = FactorValues(SMALL, **extra)
            with_index = FactorValues(SMALL, index=[0.5, 1.0, 0.5, 1.0, 0.5, 0.5], index_volatility=0.4, **extra)
            plain.reseed(seed)
            with_index.reseed(seed)
            a, b = path(plain, 700), path(with_index, 700)
            assert np.array_equal(plain.factors, with_index.factors)
            assert np.array_equal(plain.own, with_index.own[:6])
            assert np.array_equal(plain.loadings, with_index.loadings[:6])
            assert np.allclose(plain.common, with_index.common[:6], rtol=0, atol=1e-9)
            assert np.allclose(a, b[:, :6], rtol=0, atol=1e-9)
            assert np.array_equal(plain._z, with_index._z)  # the shocks are those of the six books
            assert plain._rng.bit_generator.state == with_index._rng.bit_generator.state and plain.pushed == 0.0


def test_the_constituents_are_those_of_the_same_seed_at_the_floor_too():
    plain = FactorValues(SMALL, level=15.0, volatility=3.0, step=1.0)
    with_index = FactorValues(SMALL, level=15.0, volatility=3.0, step=1.0, index=True, index_volatility=0.5)
    plain.reseed(2)
    with_index.reseed(2)
    a, b = path(plain, 600), path(with_index, 600)
    assert plain.pushed > 20  # the floor binds, and the index does not push for itself
    assert with_index.pushed == pytest.approx(plain.pushed, abs=1e-9)
    assert np.allclose(a, b[:, :6], rtol=0, atol=1e-8)
    assert np.allclose(plain.factors, with_index.factors, rtol=0, atol=1e-8)
    assert with_index.common[6] == pytest.approx(float(with_index.common[:6].sum()), abs=1e-8)  # the push reached it


# ---- A lot taken ----

def test_a_lot_in_a_book_moves_the_index_by_its_weight_times_the_impact_and_a_lot_in_the_index_moves_its_premium():
    c = [0.5, 1.0, 1.0, 0.0, 2.0, 0.5]
    f = FactorValues(SMALL, index=c, impact=0.05)
    f.advance_to(5.0)
    before_own, before_fairs, before_premium = f.own.copy(), list(f.fairs), f.premium
    f.trade(4, 3)  # three lots bought in a book of weight 2
    assert f.own[4] == pytest.approx(before_own[4] + 0.15) and f.fairs[4] == pytest.approx(before_fairs[4] + 0.15)
    assert f.own[6] == pytest.approx(before_own[6] + 2 * 0.15) and f.fairs[6] == pytest.approx(before_fairs[6] + 0.3)
    assert f.premium == before_premium  # which is the book's doing, not the premium's
    f.trade(3, 5)  # a book of weight 0: the index does not feel it
    assert f.fairs[6] == pytest.approx(before_fairs[6] + 0.3) and f.fairs[3] == pytest.approx(before_fairs[3] + 0.25)
    f.trade(0, -4)
    assert f.fairs[6] == pytest.approx(before_fairs[6] + 0.3 - 0.5 * 0.2)
    scale, mark = f.scales[6], list(f.fairs)
    f.trade(6, 2)  # two lots in the index: its premium by scale * impact * lots, and no book moves
    assert f.premium == pytest.approx(before_premium + scale * 0.05 * 2)
    assert f.fairs[6] == pytest.approx(mark[6] + scale * 0.05 * 2) and f.fairs[:6] == mark[:6]
    assert f.own[6] == pytest.approx(float(np.dot(c, f.own[:6])) + f.premium, abs=1e-12)
    f.advance_to(f.step * 26)  # and the next step starts from where the lots left it
    assert f.fairs[6] == pytest.approx(float(np.dot(c, f.fairs[:6])) + f.premium, abs=1e-9)


def test_an_order_in_the_market_moves_the_fair_values_as_a_lot_taken_does():
    c = np.array([1.0, 0.5, 0.5, 1.0, 0.5, 0.5])
    f = FactorValues(SMALL, index=c, impact=0.1)
    market = BookMarket(f, seed=1)
    before = list(f.fairs)
    taken = sum(fill.size for fill in market.submit(1, market_order(3, 0.0, Side.BID)))
    assert taken == 3 and f.fairs[1] == pytest.approx(before[1] + 0.3) and f.fairs[6] == pytest.approx(before[6] + 0.15)
    taken = sum(fill.size for fill in market.submit(6, market_order(2, 0.0, Side.ASK)))
    assert taken == 2 and f.premium == pytest.approx(-f.scales[6] * 0.1 * 2)
    assert f.fairs[1] == pytest.approx(before[1] + 0.3)  # a book is not moved by a lot in the index


# ---- Loadings and relations ----

def test_the_index_row_of_the_loadings_is_the_weighted_sum_also_when_they_drift():
    c = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1]) * 2.0
    plain = FactorValues(SMALL, loading_sd=0.4, loading_half_life=20.0, step=0.5)
    f = FactorValues(SMALL, loading_sd=0.4, loading_half_life=20.0, step=0.5, index=c)
    plain.reseed(6)
    f.reseed(6)
    assert np.allclose(f.mean_loadings[6], c @ SMALL, rtol=0, atol=1e-12)
    for i in range(1, 801):
        f.advance_to(i * f.step)
        plain.advance_to(i * f.step)
        assert np.allclose(f.loadings[6], c @ f.loadings[:6], rtol=0, atol=1e-12)
        assert f.loadings[:6, 0].min() >= BETA_FLOOR  # the floor on the first loading is the books'
        assert np.array_equal(f.loadings[:6], plain.loadings)  # and the books' rows are those of the same seed
    assert abs(f.loadings[:6] - SMALL).max() > 0.2 and f.exposures() is f.loadings and f.exposures().shape == (7, 3)
    assert np.allclose(f.mean_loadings[6], c @ SMALL, rtol=0, atol=1e-12)  # the mean does not drift


def test_with_an_index_a_relation_is_only_the_own_parts_and_the_premium():
    for index in (True, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]):
        f = FactorValues(SMALL, level=3000.0, index=index, index_volatility=0.7)
        f.reseed(2)
        w = f.relations()
        assert w.shape == (5, 7)
        for i in range(1, 501):
            f.advance_to(i * f.step)
        values = f.values()
        assert w @ (values - f.level) == pytest.approx(w @ f.own, abs=1e-9)
        assert f.own[6] == pytest.approx(float(f.index_weights @ f.own[:6]) + f.premium, abs=1e-9)
        assert abs(f.premium) > 0.1


def test_the_books_have_one_more_relation_with_an_index_and_the_replication_is_one_that_leaves_only_the_premium():
    plain = FactorValues(SMALL)
    for index in (True, [1 / 6] * 6, [3.0, 0.0, 1.0, 0.0, 0.0, 2.0]):
        f = FactorValues(SMALL, index=index, index_volatility=0.6)
        assert f.rank == plain.rank + 1 and f.relations().shape == (plain.rank + 1, 7)
        replication = np.append(-f.index_weights, 1.0)
        assert np.allclose(replication @ f.mean_loadings, 0, atol=1e-12)  # it leaves every factor out
        relations = f.relations()
        assert np.allclose(replication, relations.T @ (relations @ replication), atol=1e-12)  # and is one of them
        f.reseed(4)
        for i in range(1, 401):
            f.advance_to(i * f.step)
        assert replication @ (f.values() - f.level) == pytest.approx(f.premium, abs=1e-8)  # all that is left


# ---- The floor ----

def test_the_index_is_held_at_the_floor_and_does_not_push_it():
    # Two books near the floor and an index of their average (scale 1) whose premium is big and often negative.
    book = np.ones((2, 1))
    plain = FactorValues(book, level=[12.0, 12.0], volatility=0.0, own_volatility=1.0, step=1.0)
    f = FactorValues(book, level=[12.0, 12.0], volatility=0.0, own_volatility=1.0, step=1.0, index=[0.5, 0.5],
                     index_volatility=3.0)
    plain.reseed(1)
    f.reseed(1)
    held, below = 0, 0
    for i in range(1, 1_001):
        plain.advance_to(float(i))
        f.advance_to(float(i))
        assert min(f.fairs) >= FLOOR  # the index never goes below it
        unheld = 0.5 * (f.fairs[0] + f.fairs[1]) + f.premium
        below += unheld < FLOOR
        held += f.fairs[2] == FLOOR
    assert below > 50 and held == below  # it was held there each time its sum would have been under
    assert f.pushed == pytest.approx(plain.pushed, abs=1e-9) and f.pushed > 0  # the books alone decide the push
    assert f.factors[0] == pytest.approx(plain.factors[0], abs=1e-9)


def test_the_push_at_the_floor_is_the_least_that_keeps_every_book_up_whatever_its_loading_on_the_first_factor():
    book = np.array([[0.5], [2.0]])  # a push of the factor of 1 moves them by 0.5 and by 2
    plain = FactorValues(book, level=[12.0, 12.0], volatility=0.0, own_volatility=1.0, step=1.0)
    f = FactorValues(book, level=[12.0, 12.0], volatility=0.0, own_volatility=1.0, step=1.0, index=[0.5, 0.5],
                     index_volatility=1.0)
    plain.reseed(1)
    f.reseed(1)
    pushes, before = 0, 0.0
    for i in range(1, 501):
        plain.advance_to(float(i))
        f.advance_to(float(i))
        assert f.pushed == pytest.approx(plain.pushed, abs=1e-9) and f.fairs[:2] == pytest.approx(plain.fairs)
        if f.pushed > before:  # pushed this step: the book that needed it most is at the floor, and none is below
            pushes += 1
            assert min(f.fairs[:2]) == pytest.approx(FLOOR, abs=1e-9)
        assert min(f.fairs) >= FLOOR - 1e-9
        before = f.pushed
    assert pushes > 20


# ---- Choosers ----

def test_with_an_index_the_sectors_market_has_13_books_and_johansens_choosers_take_12_at_most():
    f = FactorValues(SECTORS, index=True, index_volatility=0.5)
    assert f.n == 13 and f.rank == 10
    history = path(f, 3_000)[::5]
    assert history.shape == (600, 13)
    with pytest.raises(ValueError, match="12 books at most"):
        johansen_relations(history)
    assert best_baskets(history, count=1, legs=2).shape[1] == 13  # a basket of two books from the 13 is as it was
