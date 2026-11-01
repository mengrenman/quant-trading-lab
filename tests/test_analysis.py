"""Tests for metrics, stylized-fact estimators and plotting helpers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from lob.analysis.plotting import plot_acf, plot_book_snapshot
from lob.analysis.stylized_facts import (
    classify_depth_profile, compute_acf, compute_hurst_rs,
    compute_rank_volatility_acf, fit_loglog_slope, fit_power_law,
    hill_estimator, mean_depth_profile, white_noise_band,
)
from lob.book import OrderBook
from lob.metrics import event_time_returns, mean_virtual_impact, virtual_impact
from lob.order import Side, limit_order


def _ladder():
    """Bids 100 x3, 99 x5; asks 102 x4, 103 x2, 105 x6 (mid 101)."""
    book = OrderBook()
    for price, size, side in [(100, 3, Side.BID), (99, 5, Side.BID),
                              (102, 4, Side.ASK), (103, 2, Side.ASK), (105, 6, Side.ASK)]:
        book.add_order(limit_order(price, size, 0.0, side))
    return book


def test_compute_acf_of_alternating_series():
    acf = compute_acf(np.tile([1.0, -1.0], 500), max_lag=2)
    assert acf[0] == pytest.approx(1.0)
    assert acf[1] == pytest.approx(-1.0, abs=0.01)
    assert acf[2] == pytest.approx(1.0, abs=0.01)


def test_white_noise_band():
    assert white_noise_band(10_000) == pytest.approx(0.0196)


def test_hill_recovers_pareto_tail_index():
    rng = np.random.default_rng(0)
    x = (1 - rng.random(200_000)) ** (-1 / 3.0)  # P(X > x) = x^-3
    assert hill_estimator(x, k=2_000) == pytest.approx(3.0, rel=0.1)


def test_fit_power_law_returns_density_exponent():
    rng = np.random.default_rng(1)
    x = (1 - rng.random(50_000)) ** (-1 / 1.5)  # CCDF exponent 1.5
    assert fit_power_law(x, xmin=1.0)["alpha"] == pytest.approx(2.5, abs=0.05)


def test_hurst_of_white_noise_is_near_half():
    rng = np.random.default_rng(2)
    assert compute_hurst_rs(rng.standard_normal(20_000)) == pytest.approx(0.5, abs=0.1)


def test_fit_loglog_slope():
    x = np.arange(1, 50, dtype=float)
    assert fit_loglog_slope(x, 3 * x ** 0.5) == pytest.approx(0.5)
    assert np.isnan(fit_loglog_slope([1.0], [1.0]))


def test_virtual_impact_walks_the_opposite_side():
    snap = _ladder().snapshot()
    buys = virtual_impact(snap, [1, 4, 5, 6, 12], Side.BID)
    np.testing.assert_allclose(buys[:4], [0.0, 0.5, 0.5, 1.5])
    assert np.isnan(buys[4])  # consumes every recorded level
    sells = virtual_impact(snap, [2, 3, 8], Side.ASK)
    np.testing.assert_allclose(sells[:2], [0.0, 0.5])
    assert np.isnan(sells[2])


def test_mean_depth_profile_counts_empty_levels_as_zero():
    snaps = [_ladder().snapshot(), _ladder().snapshot()]
    deltas, bid, ask = mean_depth_profile(snaps, n_ticks=4)
    np.testing.assert_array_equal(deltas, [0, 1, 2, 3])
    np.testing.assert_allclose(bid, [3, 5, 0, 0])
    np.testing.assert_allclose(ask, [4, 2, 0, 6])


def test_plot_book_snapshot_has_monotonic_price_axis():
    fig, ax = plt.subplots()
    plot_book_snapshot(_ladder(), ax=ax)
    bars = sorted((p.get_y() + p.get_height() / 2, p.get_width()) for p in ax.patches)
    assert [round(y) for y, _ in bars] == [99, 100, 102, 103, 105]
    widths = [w for _, w in bars]
    assert all(w < 0 for w in widths[:2])  # bids extend left
    assert all(w > 0 for w in widths[2:])  # asks extend right
    plt.close(fig)


def test_plot_acf_band_uses_series_length():
    fig, ax = plt.subplots()
    plot_acf(np.zeros(11), ax=ax, n=10_000)
    dashed = sorted(line.get_ydata()[0] for line in ax.get_lines()
                    if line.get_linestyle() == "--")
    assert dashed == pytest.approx([-0.0196, 0.0196])
    plt.close(fig)


def test_event_time_returns_samples_every_k_events():
    r = event_time_returns([100.0, 100.0, 101.0, np.nan, 101.0, 102.0], k=2)
    np.testing.assert_allclose(r, [np.log(101 / 100), np.log(102 / 101)])


def test_mean_virtual_impact_keeps_sizes_most_snapshots_absorb():
    snaps = [_ladder().snapshot()] * 3
    sizes, impact = mean_virtual_impact(snaps, np.arange(1, 20))
    np.testing.assert_array_equal(sizes, np.arange(1, 12))  # 12 lots of asks recorded
    np.testing.assert_allclose(impact[[0, 3, 5]], [0.0, 0.5, 1.5])


def test_rank_volatility_acf_is_robust_to_one_extreme_return():
    rng = np.random.default_rng(3)
    calm, wild = rng.normal(0, 1, 500), rng.normal(0, 5, 500)
    returns = np.concatenate([calm, wild])  # volatility regime switch
    returns[10] = 1e6  # a single huge outlier swamps the raw ACF
    assert compute_acf(np.abs(returns), 1)[1] < 0.05
    assert compute_rank_volatility_acf(returns, 1)[1] > 0.2


@pytest.mark.parametrize("profile, expected", [
    ([2, 5, 5, 5], "plateau"),
    ([2, 5, 3, 1], "hump"),
    ([6, 4, 2, 1], "best quote"),
])
def test_classify_depth_profile(profile, expected):
    assert classify_depth_profile(profile)[0] == expected


def test_orders_at_returns_queue_in_priority_order():
    book = OrderBook()
    first = limit_order(100, 2, 1.0, Side.BID)
    second = limit_order(100, 3, 2.0, Side.BID)
    book.add_order(first)
    book.add_order(second)
    assert [o.order_id for o in book.orders_at(100, Side.BID)] == [first.order_id, second.order_id]
    assert book.orders_at(101, Side.ASK) == []


def test_plot_depth_profile_bars_do_not_overlap():
    from lob.analysis.plotting import plot_depth_profile

    fig, ax = plt.subplots()
    plot_depth_profile(_ladder(), ax=ax)
    centers = [p.get_x() + p.get_width() / 2 for p in ax.patches]
    assert len(centers) == len(set(np.round(centers, 6)))
    plt.close(fig)
