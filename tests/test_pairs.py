"""Two order books whose prices move together (lob/pairs.py)."""

import math

import numpy as np
import pytest

from lob.order import Side, market_order
from lob.pairs import A, B, AnchoredZIModel, Arbitrageur, FairValues, PairMarket


def half_life(x, lag=1):
    x = np.asarray(x) - np.mean(x)
    phi = np.corrcoef(x[:-lag], x[lag:])[0, 1]
    return -lag * math.log(2) / math.log(phi)


def test_the_fair_values_gap_mean_reverts_with_its_half_life_and_spread():
    rng = np.random.default_rng(3)
    f = FairValues(beta=1.5, half_life=20.0, own_volatility=0.4)
    gaps = []
    for _ in range(40_000):
        f.advance(1.0, rng)
        gaps.append(f.gap())
    assert half_life(gaps) == pytest.approx(20.0, rel=0.15)
    assert np.std(gaps) == pytest.approx(f.gap_sd, rel=0.1)
    assert f.gap_sd == pytest.approx(0.4 * math.sqrt((1 + 1.5**2) / (2 * math.log(2) / 20)))


def test_without_a_half_life_the_gap_is_a_random_walk():
    rng = np.random.default_rng(4)
    f = FairValues(beta=1.0, half_life=None, own_volatility=0.5)
    assert f.gap_sd is None
    gaps = [0.0]
    for _ in range(20_000):
        f.advance(1.0, rng)
        gaps.append(f.gap())
    steps = np.diff(gaps)
    assert np.var(steps) == pytest.approx(0.5**2 * 2, rel=0.05)  # each own part adds its variance
    assert abs(np.corrcoef(steps[:-1], steps[1:])[0, 1]) < 0.03  # no reversion: increments independent


def test_trading_moves_a_books_fair_value_and_the_move_fades():
    f = FairValues(volatility=0.0, own_volatility=0.0, half_life=10.0, impact=0.05)
    f.trade(B, 20)
    assert f.fair(B) == pytest.approx(100 + 1.0) and f.fair(A) == 100 and f.gap() == pytest.approx(1.0)
    f.advance(10.0, np.random.default_rng(0))  # one half-life
    assert f.gap() == pytest.approx(0.5)
    permanent = FairValues(volatility=0.0, own_volatility=0.0, half_life=None, impact=0.05)
    permanent.trade(A, -40)
    permanent.advance(1000.0, np.random.default_rng(0))
    assert permanent.fair(A) == pytest.approx(98.0) and permanent.gap() == pytest.approx(2.0)  # beta 1: -(-2)


def test_b_follows_the_common_factor_by_beta():
    f = FairValues(level_a=100, level_b=250, beta=2.5, volatility=0.0, own_volatility=0.0)
    f.x = 104  # the common factor up 4
    assert f.fair(A) == 104 and f.fair(B) == 250 + 2.5 * 4 and f.gap() == 0


def test_each_book_follows_its_fair_value():
    m = PairMarket(FairValues(half_life=50.0), seed=2)
    d = m.run(1500)
    for k in "ab":  # within a couple of ticks, though the fair values move fast (own volatility 0.8)
        assert np.sqrt(np.mean((d[f"mid_{k}"] - d[f"fair_{k}"]) ** 2)) < 2.5
    assert 1.5 < d.spread_a.mean() < 2.5
    assert np.corrcoef(d.gap, d.true_gap)[0, 1] > 0.85  # what a trader sees follows the true gap


def test_an_arbitrageur_keeps_wandering_books_together():
    alone = PairMarket(FairValues(half_life=None), seed=1).run(3000)
    arb = Arbitrageur(threshold=2.0, size=5)
    together = PairMarket(FairValues(half_life=None), arbitrageur=arb, seed=1)
    kept = together.run(3000)
    assert alone.gap.std() > 5 and kept.gap.std() < 2.5
    assert arb.trades > 100 and arb.position[A] == -arb.position[B]  # beta 1: one lot of A for each of B
    assert half_life(kept.gap, 10) < 40


def test_a_pairs_trader_shorts_a_dear_gap_and_goes_flat_at_the_mean():
    from lob.pairs import PairsTrader

    t = PairsTrader(window=4, entry=1.0, exit=0.0, size=5, hedge=2.0)
    for b in (100, 101, 100, 101):  # B against A at 50: gaps 0, 1, 0, 1
        assert t.decide(50.0, b) == 0
    assert t.decide(50.0, 106) == -5 and t.z > 1.0  # dear: short B (and long 10 of A)
    assert t.decide(50.0, 104) == -5  # still above its mean
    assert t.decide(50.0, 101) == 0 and t.z < 0  # past the mean: flat
    assert t.decide(50.0, 90) == 5 and t.entries == 2  # cheap: long B
    assert t.decide(None, 100) is None
    with pytest.raises(ValueError):
        PairsTrader(entry=1.0, exit=1.0)


def test_a_pairs_trader_in_the_market_trades_both_books():
    from lob.pairs import PairsTrader

    trader = PairsTrader(window=200)
    m = PairMarket(FairValues(half_life=50.0), seed=3, traders=(trader,))
    m.run(3000)
    assert trader.entries > 0 and trader.lots > 0 and trader.spread_paid > 0
    assert trader.position[A] == -round(trader.hedge * trader.position[B])  # hedged
    assert trader.value(m.mid(A), m.mid(B)) is not None


def test_a_pair_market_repeats_for_its_seed():
    one = PairMarket(FairValues(), seed=7).run(300)
    two = PairMarket(FairValues(), seed=7).run(300)
    other = PairMarket(FairValues(), seed=8).run(300)
    assert one.equals(two) and not one.equals(other)


def test_an_order_from_outside_takes_liquidity_and_moves_the_fair_value():
    m = PairMarket(FairValues(volatility=0.0, own_volatility=0.0, impact=0.1), seed=1)
    before = m.fair.fair(A)
    fills = m.submit(A, market_order(15, 0.0, Side.BID))
    assert sum(f.size for f in fills) == 15
    assert m.fair.fair(A) == pytest.approx(before + 1.5) and m.fair.fair(B) == pytest.approx(100.0)
    m.submit(B, market_order(5, 0.0, Side.ASK), impact=False)
    assert m.fair.fair(B) == pytest.approx(100.0)


@pytest.mark.parametrize("make, message", [
    (lambda: FairValues(half_life=0), "half-life"),
    (lambda: FairValues(beta=0), "beta"),
    (lambda: FairValues(impact=-1), "impact"),
    (lambda: AnchoredZIModel(limit_rate=0), "limit orders above 0"),
    (lambda: AnchoredZIModel(reach=0), "reach"),
])
def test_settings_that_cannot_work_are_refused(make, message):
    with pytest.raises(ValueError, match=message):
        make()


def test_a_pairs_traders_decisions_add_up_to_its_fills_before_costs():
    """At the mids just before its orders, its holdings between decisions made what its fills did before costs."""
    from lob.pairs import PairsTrader

    trader = PairsTrader(window=200, record=True)
    m = PairMarket(FairValues(), seed=5, traders=(trader,))
    m.run(4000, every=1000.0)
    h = np.array([row[1:5] for row in trader.history])  # mid A, mid B, then what it held of A and of B
    held = np.vstack([[0, 0], h[:-1, 2:]])  # held over each step: what the decision before left it
    mids = np.vstack([h[:1, :2], h[:, :2]])
    gross = (held * np.diff(mids, axis=0)).sum() + (h[-1, 2:] * (np.array([m.mark(A), m.mark(B)]) - h[-1, :2])).sum()
    assert trader.entries > 2
    assert gross == pytest.approx(trader.value(m.mark(A), m.mark(B)) + trader.spread_paid, abs=1e-6)


def test_positions_are_valued_at_the_last_mid_while_a_side_is_empty():
    m = PairMarket(FairValues(volatility=0.0, own_volatility=0.0), seed=1)
    before = m.mark(A)
    m.submit(A, market_order(10_000, 0.0, Side.ASK), impact=False)  # sells out the whole bid side
    assert m.mid(A) is None and m.mark(A) == before
