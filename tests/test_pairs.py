"""Two order books whose prices move together (lob/pairs.py)."""

import math

import numpy as np
import pytest

from lob.order import Side, market_order
from lob.copulas import Margin, fit_copula, pseudo_observations
from lob.pairs import (A, B, AnchoredZIModel, Arbitrageur, CopulaTrader, FairValues, KalmanHedge, KalmanTrader,
                       PairMarket, PairsTrader, score_trader)


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


# ---- The Kalman hedge ----

def test_with_no_state_noise_the_kalman_filter_is_least_squares():
    """With a flat prior and q = 0, the filter is recursive least squares: after each point, the regression so far."""
    rng = np.random.default_rng(1)
    x = 1000 + np.cumsum(rng.normal(size=300))
    y = 7 + 1.4 * x + rng.normal(scale=3.0, size=300)
    f = KalmanHedge(0.0, 0.0, 1000.0, 1e8, 1e8, 0.0, 0.0, 9.0)
    for i in range(300):
        f.update(x[i], y[i])
        if i >= 5:
            slope, intercept = np.polyfit(x[: i + 1], y[: i + 1], 1)
            assert f.beta == pytest.approx(slope, abs=1e-4)
            assert f.fit(x[i]) == pytest.approx(slope * x[i] + intercept, abs=1e-3)


def test_a_filter_with_a_flat_regressor_reaches_the_local_levels_steady_gain():
    """x constant: a local-level model, whose steady gain solves K = (P + q) / (P + q + r), P = (1 - K)(P + q)."""
    q, r = 0.01, 4.0
    f = KalmanHedge(0.0, 0.0, 100.0, 0.0, 10.0, 0.0, q, r)
    for _ in range(5000):
        _, s = f.update(100.0, 1.0)
    predicted = (q + math.sqrt(q * q + 4 * q * r)) / 2  # P + q in the steady state
    assert f.p_ll + q == pytest.approx(predicted, rel=1e-6) and s == pytest.approx(predicted + r, rel=1e-6)
    f = KalmanHedge(0.0, 0.0, 100.0, 0.0, predicted - q, 0.0, q, r)
    before = f.level
    innovation, _ = f.update(100.0, 2.0)
    assert (f.level - before) / innovation == pytest.approx(predicted / (predicted + r), rel=1e-9)
    assert predicted / (predicted + r) == pytest.approx(1 / 20, rel=0.03)  # about 1 / memory, with q = r / memory^2


def test_the_kalman_filter_is_the_gaussian_posterior_of_its_model():
    """Its final state against the joint Gaussian of the whole path, solved in one go."""
    rng = np.random.default_rng(2)
    n, pivot, r = 60, 50.0, 2.0
    q_b, q_l, p_b, p_l = 1e-3, 0.05, 0.5, 4.0
    x = pivot + np.cumsum(rng.normal(size=n))
    y = 3 + 1.2 * (x - pivot) + rng.normal(scale=math.sqrt(r), size=n)
    # Betas and levels: prior covariance from the random walks (each state is the start plus the noises so far).
    t = np.arange(1, n + 1)
    cov_b = p_b + q_b * np.minimum.outer(t, t)
    cov_l = p_l + q_l * np.minimum.outer(t, t)
    d = x - pivot
    cov_y = np.outer(d, d) * cov_b + cov_l + r * np.eye(n)
    weights = np.linalg.solve(cov_y, y)
    beta_n = (cov_b[-1] * d) @ weights
    level_n = cov_l[-1] @ weights
    f = KalmanHedge(0.0, 0.0, pivot, p_b, p_l, q_b, q_l, r)
    for a, b in zip(x, y, strict=True):
        f.update(a, b)
    assert f.beta == pytest.approx(beta_n, abs=1e-9) and f.level == pytest.approx(level_n, abs=1e-9)
    # Its final uncertainty, too.
    gain_b = (cov_b[-1] * d)
    assert f.p_bb == pytest.approx(cov_b[-1, -1] - gain_b @ np.linalg.solve(cov_y, gain_b), abs=1e-9)


def test_moving_the_pivot_changes_the_coordinates_and_not_the_line():
    f = KalmanHedge(1.3, 500.0, 1000.0, 0.01, 2.0, 0.0, 0.0, 1.0)
    f.p_bl = 0.05
    def spread(x, g):  # the variance of the fit at x
        d = x - g.pivot
        return d * d * g.p_bb + 2 * d * g.p_bl + g.p_ll
    before = [(f.fit(x), spread(x, f)) for x in (900.0, 1000.0, 1100.0)]
    f.move_pivot(1040.0)
    for x, (fit, var) in zip((900.0, 1000.0, 1100.0), before, strict=True):
        assert f.fit(x) == pytest.approx(fit) and spread(x, f) == pytest.approx(var)
    assert f.p_bb == 0.01 and f.pivot == 1040.0


def test_a_kalman_trader_starts_from_least_squares_and_follows_its_filter():
    t = KalmanTrader(window=50, memory=200.0, entry=2.0)
    rng = np.random.default_rng(3)
    a = 1000 + np.cumsum(rng.normal(scale=2.0, size=400))
    b = 1000 + 1.5 * (a - a[0]) + rng.normal(scale=1.0, size=400)
    for i in range(49):
        assert t.decide(a[i], b[i]) == 0 and t.warming == (i + 1, 50) and t.z is None
    assert t.decide(a[49], b[49]) == 0 and t.warming is None and t.filter is not None
    slope = np.polyfit(a[:50], b[:50], 1)[0]
    assert t.hedge == pytest.approx(slope) and t.beta_hat == pytest.approx(slope)
    residuals = b[:50] - np.polyval(np.polyfit(a[:50], b[:50], 1), a[:50])
    assert t.filter.r == pytest.approx(np.mean(residuals ** 2))
    zs = []
    for i in range(50, 400):
        t.decide(a[i], b[i])
        zs.append(t.z)
        assert t.hedge == pytest.approx(t.filter.beta)
    assert t.beta_hat == pytest.approx(1.5, abs=0.05)
    assert np.std(zs) == pytest.approx(1.0, abs=0.25)  # the model is right here, so z is standardized
    assert abs(np.corrcoef(zs[:-1], zs[1:])[0, 1]) < 0.2  # and its innovations about white


def test_a_kalman_trader_opens_on_its_own_z_and_rebalances_or_holds_its_hedge():
    def fed(rebalance):
        t = KalmanTrader(window=10, memory=20.0, entry=2.0, rebalance=rebalance, size=10)
        for i in range(10):
            t.decide(1000.0 + i, 1000.0 + i + (-1) ** i)
        for i in range(10, 40):
            t.decide(1000.0 + i, 1000.0 + i + (-1) ** i)
        return t
    held, moving = fed(False), fed(True)
    for t in (held, moving):
        assert t.decide(1040.0, 1060.0) == -10 and t.z > 2 and t.entries == 1
    at_entry = held.hedge
    for i in range(3):  # B stays dear while A rises, so the slope moves and the position stays
        for t in (held, moving):
            assert t.decide(1041.0 + i, 1060.0 + 3 * i) == -10
    assert held.hedge == at_entry and moving.hedge != at_entry and moving.hedge == pytest.approx(moving.filter.beta)
    assert held.beta_hat == pytest.approx(held.filter.beta)  # it still estimates


def test_a_kalman_trader_in_the_market_stays_hedged_in_whole_lots():
    trader = KalmanTrader(window=100, memory=300.0, size=10)
    fair = FairValues(level_a=1000.0, level_b=1000.0, volatility=1.0, beta_sd=0.3, beta_half_life=500.0)
    m = PairMarket(fair, seed=4, traders=(trader,))
    m.run(3000)
    assert trader.entries > 0 and trader.lots > 0
    assert trader.position[A] == -round(trader.hedge * trader.position[B]) or abs(trader.position[B]) < trader.size
    assert trader.value(m.mark(A), m.mark(B)) is not None


def test_settings_a_kalman_trader_cannot_work_with_are_refused():
    with pytest.raises(ValueError, match="memory"):
        KalmanTrader(memory=0.5)
    with pytest.raises(ValueError, match="observation variance"):
        KalmanHedge(1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)


# ---- The drifting hedge ratio and the stressed regime ----

def test_a_drifting_beta_has_its_spread_and_half_life_and_b_moves_by_it():
    rng = np.random.default_rng(5)
    f = FairValues(level_a=1000.0, level_b=1000.0, volatility=1.0, own_volatility=0.0, beta_sd=0.2,
                   beta_half_life=100.0)
    betas, moved = [], []
    for _ in range(100_000):
        x, b, beta = f.x, f.fair(B), f.beta
        f.advance(1.0, rng)
        betas.append(f.beta)
        moved.append(f.fair(B) - b - beta * (f.x - x))
    betas = np.array(betas)
    assert betas.mean() == pytest.approx(1.0, abs=0.02) and betas.std() == pytest.approx(0.2, rel=0.05)
    assert half_life(betas, 100) == pytest.approx(100.0, rel=0.15)
    assert np.max(np.abs(moved)) < 1e-9  # B's common part moves by beta times the factor's move, and no more


def test_with_a_fixed_beta_nothing_changes():
    a = PairMarket(FairValues(), seed=9).run(300)
    b = PairMarket(FairValues(beta_sd=0.0, stress_share=0.0), seed=9).run(300)
    assert a.equals(b)
    assert (a.beta == 1.0).all()


def test_the_gap_of_two_prices_is_the_fair_gap_at_the_fair_values():
    rng = np.random.default_rng(6)
    for f in (FairValues(beta=1.5, level_b=250.0),
              FairValues(beta=1.5, level_b=250.0, beta_sd=0.1, beta_half_life=50.0)):
        for _ in range(200):
            f.advance(1.0, rng)
        assert f.gap_of(f.fair(A), f.fair(B)) == pytest.approx(f.gap())


def test_the_stressed_regime_has_its_share_of_the_time_and_keeps_the_factor_a_martingale():
    rng = np.random.default_rng(7)
    f = FairValues(level_a=1000.0, own_volatility=0.0, volatility=0.3, stress_share=0.2, stress_volatility=2.0,
                   stress_length=10.0, stress_drift=-0.5)
    stressed, moves = [], []
    for _ in range(1_000_000):
        x = f.x
        f.advance(0.1, rng)
        stressed.append(f.stressed)
        moves.append(f.x - x)
    stressed, moves = np.array(stressed), np.array(moves)
    assert stressed.mean() == pytest.approx(0.2, abs=0.01)
    assert moves[stressed].std() / math.sqrt(0.1) == pytest.approx(2.0, rel=0.01)
    assert moves[~stressed].std() / math.sqrt(0.1) == pytest.approx(0.3, rel=0.01)
    assert moves[stressed].mean() / 0.1 == pytest.approx(-0.5, abs=0.05)
    assert moves.mean() / 0.1 == pytest.approx(0.0, abs=4 * moves.std() / 0.1 / math.sqrt(len(moves)) + 0.005)


@pytest.mark.parametrize("make, message", [
    (lambda: FairValues(beta_sd=-0.1), "beta's standard deviation"),
    (lambda: FairValues(beta_half_life=0.0), "beta's standard deviation"),
    (lambda: FairValues(stress_share=1.0), "share of stress"),
    (lambda: FairValues(stress_length=0.0), "share of stress"),
])
def test_impossible_market_settings_are_refused(make, message):
    with pytest.raises(ValueError, match=message):
        make()


# ---- The copula trader ----

def feed(trader, returns_a, returns_b, a0=1000.0, b0=1000.0):
    """Mids whose changes are these returns, decision by decision; what it wanted at each."""
    a, b = a0, b0
    wants = [trader.decide(a, b)]
    for ra, rb in zip(returns_a, returns_b, strict=True):
        a, b = a + ra, b + rb
        wants.append(trader.decide(a, b))
    return wants


def formation(n=60, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(-6, 7, n) / 2.0, rng.integers(-6, 7, n) / 2.0  # half ticks, as a book's mids move


def test_a_copula_traders_flags_add_up_its_conditional_probabilities():
    t = CopulaTrader(window=60, trading=1000, entry=100.0, exit=0.0, families=("gaussian", "student", "frank"))
    ra, rb = formation()
    assert feed(t, ra, rb)[-1] == 0 and t.fit is not None and t.warming is None and t.fits == 1
    expected = fit_copula(pseudo_observations(ra), pseudo_observations(rb), ("gaussian", "student", "frank"))
    assert t.copula.describe() == expected.copula.describe() and t.fit.tau == pytest.approx(expected.tau)
    margin_a, margin_b = Margin(ra), Margin(rb)
    flag_a = flag_b = 0.0
    new_a, new_b = formation(40, seed=2)
    a, b = 1000.0 + ra.sum(), 1000.0 + rb.sum()
    for x, y in zip(new_a, new_b, strict=True):
        a, b = a + x, b + y
        t.decide(a, b)
        u, v = margin_a(x), margin_b(y)
        flag_a += float(expected.copula.h(u, v)) - 0.5
        flag_b += float(expected.copula.h(v, u)) - 0.5
    assert t.flag_a == pytest.approx(flag_a) and t.flag_b == pytest.approx(flag_b)
    assert t.signals == (t.flag_a, t.flag_b) and t.z == pytest.approx((flag_b - flag_a) / 2)


def rule_trader(**kw):
    """With independence as its only copula, each step of a flag is just the return's rank less 0.5."""
    t = CopulaTrader(window=60, trading=1000, entry=0.6, exit=0.0, families=(), size=5, **kw)
    ra, rb = formation()
    feed(t, ra, rb)
    assert t.copula.family == "independence"
    t._test_mids = (1000.0 + ra.sum(), 1000.0 + rb.sum())
    return t


def step(t, ra, rb):
    a, b = t._test_mids
    t._test_mids = (a + ra, b + rb)
    return t.decide(*t._test_mids)


def test_a_copula_trader_opens_only_when_the_flags_part_and_closes_when_both_are_back():
    t = rule_trader()
    high, low = 10.0, -10.0  # beyond every return in its sample: a rank near 1, or near 0
    assert step(t, high, high) == 0 and step(t, high, high) == 0  # both dear: nothing between them
    assert t.flag_a > 0.6 and t.flag_b > 0.6 and t.target == 0
    t = rule_trader()
    assert step(t, low, high) == 0  # the flags part, but not beyond the entry level yet
    assert step(t, low, high) == -5 and t.flag_b > 0.6 > -0.6 > t.flag_a and t.entries == 1  # B dear, A cheap: short
    assert t.want() == -5
    assert step(t, 0.0, 0.0) == -5  # returns near their medians: the flags hardly move, and it holds
    assert step(t, high, low) == -5  # A's flag is back above 0 now, and B's below it, in one or two steps
    while t.target:
        assert t.entries == 1
        step(t, high, low)
    assert t.flag_a > -0.0 and t.flag_b < 0.0 and t.target == 0  # flat once both have come back
    t = rule_trader()
    step(t, high, low)
    assert step(t, high, low) == 5  # the other way round: long the gap


def test_a_copula_trader_waits_for_both_flags_but_not_for_them_at_once():
    t = rule_trader()
    high, low = 10.0, -10.0
    step(t, low, high)
    assert step(t, low, high) == -5
    # A's flag comes back alone (B's stays high): still short. Then B's comes back, while A's goes away again.
    for _ in range(3):
        assert step(t, high, 0.0) == -5
    assert t.flag_a > 0 and t.flag_b > 0
    assert step(t, low, low) == -5 or t.target == -5
    while t.flag_b > 0:
        want = step(t, low, low)
    assert t.flag_a < 0 and t.flag_b <= 0 and want == 0


def test_a_copula_traders_period_ends_by_closing_fitting_again_and_starting_its_flags_from_zero():
    t = CopulaTrader(window=60, trading=5, entry=0.6, exit=0.0, families=(), size=5)
    ra, rb = formation()
    feed(t, ra, rb)
    t._test_mids = (1000.0 + ra.sum(), 1000.0 + rb.sum())
    assert t.fits == 1
    assert [step(t, -10.0, 10.0) for _ in range(4)][-1] == -5
    assert t.flag_b > 0.6
    assert step(t, -10.0, 10.0) == 0  # the fifth: unconverged, it closes
    assert t.fits == 2 and t.flag_a == 0 == t.flag_b and t.target == 0 and t.entries == 1
    # The new sample has those large returns in it, so they no longer rank at the extreme.
    assert len(t._margins[0].values) == 60 and -10.0 in t._margins[0].values
    assert step(t, -10.0, 10.0) == 0 and t.flag_a < 0 < t.flag_b


def test_a_copula_trader_learns_from_past_mids_without_trading():
    t = CopulaTrader(window=50, trading=20, interval=5.0)
    assert t.warming == (0, 50) and t.learns_from == 51 and t.signals == (None, None)
    rng = np.random.default_rng(9)
    moves = np.cumsum(rng.normal(size=(300, 2)), axis=0)
    mids = [(1000 + float(a), 1000 + float(b)) for a, b in moves]
    assert t.learn(mids[:30]) == 30 and t.warming == (29, 50) and t.fit is None
    t2 = CopulaTrader(window=50, trading=20, interval=5.0)
    assert t2.learn(mids) == 51  # only what it can use, the latest
    assert t2.warming is None and t2.fits == 1 and t2.flag_a == 0.0 and t2.target == 0 and t2.lots == 0
    assert t2._returns[-1] == (mids[-1][0] - mids[-2][0], mids[-1][1] - mids[-2][1])


def test_a_copula_trader_in_the_market_trades_both_books_and_repeats_for_its_seed():
    def run(seed):
        t = CopulaTrader(window=100, trading=50, entry=1.0, interval=5.0, size=5)
        m = PairMarket(FairValues(level_a=1000.0, level_b=1000.0, volatility=1.0), seed=seed, traders=(t,))
        m.run(4000, every=5.0)
        return t, m
    t, m = run(3)
    assert t.fits >= 10 and t.entries > 0 and t.lots > 0
    assert t.position[A] == -t.position[B] or t.position[B] == 0  # a hedge of 1
    t2, _ = run(3)
    assert (t2.cash, t2.lots, t2.flag_a) == (t.cash, t.lots, t.flag_a)
    assert run(4)[0].cash != t.cash


def test_with_a_stressed_market_the_books_returns_get_a_tail_dependent_copula():
    """The chain from the market to the choice of family: mids, returns, ranks, AIC."""
    def family(**kw):
        m = PairMarket(FairValues(level_a=20000.0, level_b=20000.0, **kw), seed=5)
        d = m.run(12000, every=5.0).dropna(subset=["mid_a", "mid_b"])
        ra, rb = np.diff(d.mid_a.to_numpy(float)), np.diff(d.mid_b.to_numpy(float))
        return fit_copula(pseudo_observations(ra), pseudo_observations(rb)).copula
    calm = family(volatility=1.0)
    assert calm.tail_dependence() == pytest.approx((0, 0), abs=0.05)
    both = family(volatility=0.3, stress_share=0.1, stress_volatility=3.0)
    assert both.family == "student" and both.tail_dependence()[0] == both.tail_dependence()[1] > 0.05
    falling = family(volatility=0.3, stress_share=0.1, stress_volatility=3.0, stress_drift=-1.0)
    lower, upper = falling.tail_dependence()
    assert lower > 0.2 and upper == 0


def test_settings_a_copula_trader_cannot_work_with_are_refused():
    with pytest.raises(ValueError, match="trading period"):
        CopulaTrader(trading=0)
    with pytest.raises(ValueError, match="exit < entry"):
        CopulaTrader(entry=1.0, exit=1.0)


def trained_market(trader, seed=3):
    return PairMarket(FairValues(level_a=1000.0, level_b=1000.0), seed=seed, traders=(trader,))


def test_a_traders_score_counts_only_what_happens_in_the_span():
    trader = PairsTrader(window=20, entry=1.0)
    score = score_trader(trained_market(trader), trader, 100.0, 300.0)
    twin = PairsTrader(window=20, entry=1.0)
    m = trained_market(twin)
    while m.time < 100.0:
        m.step()
    start = twin.value(m.mark(A), m.mark(B))
    paid, lots, entries = twin.spread_paid, twin.lots, twin.entries
    assert lots > 0  # it had already traded: that must not count
    until = m.time + 300.0
    while m.time < until:
        m.step()
    assert score["pnl"] == pytest.approx(twin.value(m.mark(A), m.mark(B)) - start)
    assert score["costs"] == pytest.approx(twin.spread_paid - paid) and score["costs"] > 0
    assert score["lots"] == twin.lots - lots and score["entries"] == twin.entries - entries
    assert score["before_costs"] == pytest.approx(score["pnl"] + score["costs"])


def test_the_risk_adjusted_score_is_the_mean_change_over_its_standard_deviation():
    trader = PairsTrader(window=20, entry=1.0)
    score = score_trader(trained_market(trader), trader, 100.0, 300.0)
    twin = PairsTrader(window=20, entry=1.0)
    frame = trained_market(twin).run(400.0)
    # The market's own sample one a unit of time from the burn on, as the score's: the same up to where each starts.
    changes = np.diff(frame["value_0"].to_numpy()[100:])
    assert score["risk_adjusted"] == pytest.approx(changes.mean() / changes.std(), abs=0.03)


def test_a_trader_that_never_moves_scores_zero():
    trader = PairsTrader(window=5_000)
    assert score_trader(trained_market(trader), trader, 10.0, 50.0) == {
        "pnl": 0.0, "before_costs": 0.0, "costs": 0.0, "lots": 0, "entries": 0, "risk_adjusted": 0.0}


def test_two_traders_are_measured_over_the_same_span_whatever_their_windows():
    scores = []
    for window in (10, 80):
        trader = PairsTrader(window=window, entry=1.0)
        market = trained_market(trader)
        scores.append((score_trader(market, trader, 81.0, 200.0), market.time))
    assert abs(scores[0][1] - scores[1][1]) < 1.0  # each ran on to within an event of 281
    assert scores[0][0]["entries"] > 0 and scores[1][0]["entries"] > 0


@pytest.mark.parametrize("make", [
    lambda: PairsTrader(window=30, interval=2.0),
    lambda: KalmanTrader(window=30, interval=2.0),
    lambda: CopulaTrader(window=30, interval=2.0),
], ids=["zscore", "kalman", "copula"])
def test_a_trader_has_its_signal_by_the_end_of_its_warm_up(make):
    trader = make()
    assert trader.warm_up == 62.0
    market = trained_market(trader)
    while market.time < trader.warm_up * 0.9:
        market.step()
    assert trader.warming is not None
    while market.time < trader.warm_up:
        market.step()
    assert trader.warming is None
