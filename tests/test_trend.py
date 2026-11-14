"""A factor that trends: a drift of each factor that continues, from a stream of its own (lob/statarb.py)."""

import math

import numpy as np
import pytest

from lob.crosssection import MomentumTrader
from lob.pairs import FLOOR
from lob.statarb import BLOCK, BookMarket, FactorValues, sector_loadings

MIXED = np.array([[1.0, 0.2, -0.3], [0.8, 0.9, 0.1], [1.2, -0.5, 0.4], [0.6, 0.4, -0.2], [1.1, 1.0, 0.5]])
SECTORS = sector_loadings(3, 3)


def path(fair, steps, every=1):
    """The fair values after each ``every`` steps of the grid, and the factors, the common and own parts there."""
    values, factors, common, own = [], [], [], []
    for i in range(every, steps + 1, every):
        fair.advance_to(i * fair.step)
        values.append(list(fair.fairs))
        factors.append(fair.factors.copy())
        common.append(fair.common.copy())
        own.append(fair.own.copy())
    return np.array(values), np.array(factors), np.array(common), np.array(own)


def drift_by_hand(seed, trend, half_life, step, steps, k):
    """The drift after each of ``steps`` grid steps from the stream ``[seed, 4]``: the stationary law first, and then
    blocks of ``(BLOCK, k)`` shocks, one to a step."""
    rng = np.random.default_rng([seed, 4])
    drift = trend * rng.standard_normal(k)
    block = rng.standard_normal((BLOCK, k))
    decay = math.exp(-math.log(2) / half_life * step)
    out = []
    for i in range(steps):
        if i and i % BLOCK == 0:
            block = rng.standard_normal((BLOCK, k))
        drift = drift * decay + trend * math.sqrt(1 - decay ** 2) * block[i % BLOCK]
        out.append(drift)
    return np.array(out)


# ---- With no trend ----

@pytest.mark.parametrize("trend, half_life", [(0.0, 500.0), (0, 500.0), ([0.0, 0.0, 0.0], 500.0), (np.zeros(3), 500.0),
                                              (0.0, 3.0)])
def test_a_trend_of_0_draws_nothing_and_changes_nothing(trend, half_life):
    plain = FactorValues(MIXED, level=1000.0, loading_sd=0.2, loading_half_life=50.0)
    off = FactorValues(MIXED, level=1000.0, loading_sd=0.2, loading_half_life=50.0, trend=trend,
                       trend_half_life=half_life)
    for seed in (1, 2):
        plain.reseed(seed)
        off.reseed(seed)
        for a, b in zip(path(plain, 600), path(off, 600), strict=True):
            assert np.array_equal(a, b)
        assert off.pushed == plain.pushed and np.array_equal(off.loadings, plain.loadings)
        assert off._rng.bit_generator.state == plain._rng.bit_generator.state
        assert off._trng is None and off._tz is None and not off.drift.any() and not off.trend.any()


def test_a_market_makes_the_generators_it_uses_and_none_for_a_trend_that_is_off(monkeypatch):
    made = []
    real = np.random.default_rng
    monkeypatch.setattr(np.random, "default_rng", lambda seed=None: made.append(seed) or real(seed))
    BookMarket(FactorValues(SECTORS, trend=0.0), seed=5)
    assert made == [[1, 1], [5, 1], [5, 2]]  # the factors' at the start and at the market's seed, the order flow's
    made.clear()
    BookMarket(FactorValues(SECTORS, trend=[0.0, 0.0, 0.05, 0.0]), seed=5)
    assert made == [[1, 1], [1, 4], [5, 1], [5, 4], [5, 2]]


def test_the_defaults_have_a_trend_and_a_drift_of_zeros_one_a_factor():
    f = FactorValues(MIXED)
    assert f.trend.tolist() == [0.0, 0.0, 0.0] and f.drift.tolist() == [0.0, 0.0, 0.0]
    assert f.trend_half_life == 500.0
    assert FactorValues(MIXED, trend=0.1).trend.tolist() == [0.1, 0.1, 0.1]


# ---- Settings ----

def test_a_trend_is_given_as_one_number_or_one_a_factor_and_a_wrong_length_is_refused():
    assert FactorValues(MIXED, trend=0.05).trend.tolist() == [0.05, 0.05, 0.05]
    assert FactorValues(MIXED, trend=[0.0, 0.02, 0.05]).trend.tolist() == [0.0, 0.02, 0.05]
    assert FactorValues(MIXED, trend=np.array([0.0, 0.02, 0.05]), trend_half_life=10).trend_half_life == 10
    for wrong in ([0.1, 0.1], [0.1], [0.1] * 4, [], [[0.1, 0.1, 0.1]]):
        with pytest.raises(ValueError, match="one for each of the 3 factors"):
            FactorValues(MIXED, trend=wrong)


@pytest.mark.parametrize("trend, half_life, message", [
    (-0.01, 500.0, "0 or more"),
    ([0.0, 0.1, -0.01], 500.0, "0 or more"),
    (math.nan, 500.0, "0 or more"),
    (math.inf, 500.0, "0 or more"),
    (0.05, 0.0, "half-life"),
    (0.05, -5.0, "half-life"),
    (0.05, math.nan, "half-life"),
    (0.0, 0.0, "half-life"),
])
def test_a_trend_that_cannot_work_is_refused(trend, half_life, message):
    with pytest.raises(ValueError, match=message):
        FactorValues(MIXED, trend=trend, trend_half_life=half_life)


@pytest.mark.parametrize("half_life", [1e-3, 0.5, 1.0])
def test_a_trend_half_life_of_a_unit_or_less_is_taken_and_the_drift_decays_by_it(half_life):
    step = 0.2
    f = FactorValues(MIXED, step=step, trend=0.1, trend_half_life=half_life)
    assert f.trend_half_life == half_life
    f.reseed(1)
    drift = drift_by_hand(1, 0.1, half_life, step, 5, 3)
    f.advance_to(5 * step)
    assert np.allclose(f.drift, drift[-1], rtol=0, atol=1e-12)


# ---- What a drift is ----

def test_a_drift_starts_from_its_stationary_law_and_has_the_sd_and_half_life_it_is_given():
    k = 3_000
    f = FactorValues(np.ones((2, k)), level=1e7, step=1.0, trend=0.05, trend_half_life=20.0)
    f.reseed(1)
    assert f.drift.std() == pytest.approx(0.05, rel=0.06) and abs(f.drift.mean()) < 0.003  # not 0 at the start
    seen = [f.drift.copy()]
    for i in range(1, 201):
        f.advance_to(float(i))
        seen.append(f.drift.copy())
    seen = np.array(seen)
    for t in (1, 100, 200):
        assert seen[t].std() == pytest.approx(0.05, rel=0.06)  # and stays so
    var = (seen ** 2).mean()
    for lag, left in ((1, 0.5 ** (1 / 20)), (20, 0.5), (40, 0.25)):  # a half-life of 20 steps of 1 unit
        assert (seen[:-lag] * seen[lag:]).mean() / var == pytest.approx(left, abs=0.02)


def test_each_factor_has_the_drift_it_is_given_and_a_factor_of_0_has_none():
    trend = np.tile([0.0, 0.03, 0.06], 1_000)
    f = FactorValues(np.ones((2, 3_000)), level=1e7, step=1.0, trend=trend, trend_half_life=50.0)
    for t in range(1, 101):
        f.advance_to(float(t))
    drift = f.drift.reshape(1_000, 3)
    assert (drift[:, 0] == 0).all()
    assert drift[:, 1].std() == pytest.approx(0.03, rel=0.1) and drift[:, 2].std() == pytest.approx(0.06, rel=0.1)
    assert (f.factors.reshape(1_000, 3)[:, 0] != 0).all()  # the shocks still move it


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_drift_adds_its_share_times_the_step_to_each_factors_move_and_the_books_follow_through_their_loadings(seed):
    step = 0.2
    trending = FactorValues(MIXED, step=step, trend=[0.3, 0.2, 0.1], trend_half_life=30.0)
    plain = FactorValues(MIXED, step=step)
    trending.reseed(seed)
    plain.reseed(seed)
    last = [(trending.factors.copy(), plain.factors.copy()), (trending.common.copy(), plain.common.copy())]
    for i in range(1, 701):  # across two refills of the shocks' blocks
        trending.advance_to(i * step)
        plain.advance_to(i * step)
        added = trending.drift * step  # the drift as it is after this step
        assert np.allclose((trending.factors - last[0][0]) - (plain.factors - last[0][1]), added, rtol=0, atol=1e-11)
        assert np.allclose((trending.common - last[1][0]) - (plain.common - last[1][1]), MIXED @ added, rtol=0,
                           atol=1e-11)
        last = [(trending.factors.copy(), plain.factors.copy()), (trending.common.copy(), plain.common.copy())]
        assert np.array_equal(trending.own, plain.own)  # the trend does not reach the own parts
    assert abs(trending.drift).max() > 0.01


def test_without_shocks_the_factors_are_the_running_sum_of_the_drift_times_the_step():
    f = FactorValues(MIXED, level=3000.0, volatility=0.0, own_volatility=0.0, step=0.5, trend=0.2, trend_half_life=40.0)
    f.reseed(4)
    total = np.zeros(3)
    for i in range(1, 601):
        f.advance_to(i * f.step)
        total += f.drift * f.step
        assert np.allclose(f.factors, total, rtol=0, atol=1e-10)
        assert np.allclose(f.fairs, 3000.0 + MIXED @ total, rtol=0, atol=1e-9)  # and each book follows through its row
    assert abs(total).max() > 1.0


def test_a_trend_leaves_every_relation_a_relation():
    f = FactorValues(MIXED, level=3000.0, trend=[0.0, 0.2, 0.2], trend_half_life=100.0)
    w = f.relations()
    assert len(w) == 2
    values, factors, _, own = path(f, 500)
    assert w @ (values[-1] - 3000.0) == pytest.approx(w @ own[-1], abs=1e-9)
    assert abs(factors[-1][1:]).max() > 5  # and the factors did move


# ---- Its stream ----

@pytest.mark.parametrize("seed", [1, 2, 5])
def test_a_trend_comes_from_a_stream_of_its_own_and_the_drift_is_that_stream_by_hand(seed):
    step, k = 0.2, 3
    f = FactorValues(MIXED, step=step, trend=[0.1, 0.05, 0.2], trend_half_life=25.0)
    f.reseed(seed)
    by_hand = drift_by_hand(seed, np.array([0.1, 0.05, 0.2]), 25.0, step, 700, k)
    assert np.allclose(f.drift, np.array([0.1, 0.05, 0.2]) * np.random.default_rng([seed, 4]).standard_normal(k))
    for i in range(1, 701):  # across two refills
        f.advance_to(i * step)
        assert np.allclose(f.drift, by_hand[i - 1], rtol=1e-12, atol=0)


def test_a_trend_leaves_the_factor_shocks_the_own_parts_and_the_order_flow_as_they_were():
    trend = [0.0, 0.1, 0.1, 0.1]
    plain, trending = FactorValues(SECTORS), FactorValues(SECTORS, trend=trend)
    market, trended = BookMarket(plain, seed=3), BookMarket(trending, seed=3)
    assert trended.rng.bit_generator.state == market.rng.bit_generator.state
    assert trended.rng.bit_generator.state == np.random.default_rng([3, 2]).bit_generator.state
    market.step()
    trended.step()
    assert trended.time == market.time  # the first event's time is drawn before any fair value counts
    a, b = FactorValues(SECTORS), FactorValues(SECTORS, trend=trend)
    a.reseed(3)
    b.reseed(3)
    for left, right in zip(path(a, 700), path(b, 700), strict=True):
        assert left.shape == right.shape
    assert np.array_equal(a.own, b.own) and np.array_equal(a._z, b._z)  # the shocks and the own parts are the same
    assert a._rng.bit_generator.state == b._rng.bit_generator.state
    assert b._trng.bit_generator.state != b._rng.bit_generator.state  # and the trend has a stream of its own


def test_the_drift_at_a_step_is_the_same_whatever_trades_and_another_seed_has_another():
    trend = np.array([0.0, 0.1, 0.1, 0.1])

    def drift_after(seed, traders):
        fair = FactorValues(SECTORS, trend=trend)
        market = BookMarket(fair, seed=seed, traders=traders)
        while market.time < 150:
            market.step()
        return fair.drift.copy(), fair._steps

    momentum = MomentumTrader(lookback=20, hold=20, interval=2.0, factors=0, size=5, pca_window=50)
    quiet, busy = drift_after(2, []), drift_after(2, [momentum])
    assert momentum.lots > 0  # it traded, and so the order flow of the two markets is not the same
    for drift, steps in (quiet, busy):  # but at any grid step the drift is the seed's and the time's alone
        assert np.allclose(drift, drift_by_hand(2, trend, 500.0, 0.2, steps, 4)[-1], rtol=1e-12, atol=0)
    other = drift_after(3, [])
    assert np.allclose(other[0], drift_by_hand(3, trend, 500.0, 0.2, other[1], 4)[-1], rtol=1e-12, atol=0)
    assert not np.allclose(other[0], quiet[0])


# ---- What it makes of a long move ----

def test_with_a_trend_over_100_units_about_a_fifth_of_a_move_is_drift_and_one_move_correlates_0_17_with_the_next():
    # The design's numbers: a volatility of 1, a drift of sd 0.05 and a half-life of 500. In closed form the drift's
    # share of a 100-unit move's variance is 0.1928 and successive moves correlate 0.1760. Many factors at once are
    # many independent paths; the drift's part is the move less the same seed's without a trend.
    step, per, k = 0.5, 200, 400
    shares, corrs = [], []
    for seed in (1, 2, 3, 4):
        with_trend = FactorValues(np.ones((2, k)), level=1e7, step=step, trend=0.05, trend_half_life=500.0)
        without = FactorValues(np.ones((2, k)), level=1e7, step=step)
        with_trend.reseed(seed)
        without.reseed(seed)
        moves, drifts = [with_trend.factors.copy()], [without.factors.copy()]
        for i in range(1, 20_001):  # 10,000 units
            with_trend.advance_to(i * step)
            without.advance_to(i * step)
            if i % per == 0:
                moves.append(with_trend.factors.copy())
                drifts.append(without.factors.copy())
        moves = np.diff(moves, axis=0)
        drift_part = moves - np.diff(drifts, axis=0)
        shares.append(drift_part.var() / moves.var())
        corrs.append((moves[:-1] * moves[1:]).mean() / moves.var())
    assert np.mean(shares) == pytest.approx(0.193, abs=0.012)  # a drift applied without its step gives 0.49
    assert np.mean(corrs) == pytest.approx(0.176, abs=0.02)


# ---- The floor ----

def test_a_trend_on_the_first_factor_goes_with_the_floor_push():
    loadings = np.array([[1.0, 0.5], [0.8, -0.3], [1.3, 0.2]])
    pushed_seeds = 0
    for seed in range(1, 9):
        low = FactorValues(loadings, level=40.0, step=1.0, trend=[3.0, 0.0], trend_half_life=500.0)
        high = FactorValues(loadings, level=1e6, step=1.0, trend=[3.0, 0.0], trend_half_life=500.0)
        low.reseed(seed)
        high.reseed(seed)
        dipped = False  # whether the books, as the drift and the shocks alone take them, go below the floor
        for i in range(1, 301):
            low.advance_to(float(i))
            high.advance_to(float(i))
            assert min(low.fairs) >= FLOOR - 1e-9  # never below the floor
            assert np.array_equal(low.drift, high.drift)  # the push does not touch the drift
            dipped = dipped or min(high.fairs) - 1e6 + 40.0 < FLOOR
        # the push is what the floor added to the factor the drift moved, and nothing else of it
        assert low.factors[0] - low.pushed == pytest.approx(high.factors[0], abs=1e-9)
        assert low.factors[1] == pytest.approx(high.factors[1], abs=1e-9)
        assert (low.pushed > 0) == dipped
        pushed_seeds += low.pushed > 0
    assert 0 < pushed_seeds < 8  # a drift of 3 a unit takes 40 down in a few units, or takes it up


# ---- Momentum ----

def test_momentum_with_no_factors_holds_the_trending_factors_and_with_all_of_them_holds_none():
    # Three sectors and the market's factor, which does not trend. The drifts are set by hand (sector 1 up, sector 3
    # down, sector 2 flat) and a half-life too long for them to change, so that the question has the same answer on
    # any seed: does the dollar-neutral momentum of notebook 13 hold what rises?
    def held(factors, burn, span, seed):
        fair = FactorValues(sector_loadings(3, 3), trend=[0.0, 0.3, 0.3, 0.3], trend_half_life=1e6)
        trader = MomentumTrader(lookback=100, hold=100, interval=5.0, factors=factors, size=5, pca_window=120)
        market = BookMarket(fair, seed=seed, traders=[trader])
        fair.drift[:] = [0.0, 0.4, 0.0, -0.4]  # after the market has reseeded it
        while market.time < burn:
            market.step()
        seen, sample = [], market.time + 5.0
        while market.time < burn + span:
            market.step()
            if market.time >= sample:
                seen.append(trader.position @ fair.exposures())  # what it holds of each factor
                sample += 5.0
        return np.array(seen)

    for seed in (1, 2):
        none, three = held(0, 560.0, 400.0, seed), held(3, 680.0, 300.0, seed)
        assert none[:, 1].mean() > 10 and none[:, 3].mean() < -10 and abs(none[:, 2].mean()) < 5
        rms_none, rms_three = (math.sqrt((x[:, 1:] ** 2).sum(axis=1).mean()) for x in (none, three))
        assert rms_none > 18 and rms_three < rms_none / 3  # long the sector that rises; or no sector at all
        assert abs(three[:, 1:].mean(axis=0)).max() < 5
