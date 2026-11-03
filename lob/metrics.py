"""Metrics and measurements for LOB analysis.

Implements the quantities defined in Gould et al. (2013) Section 2:
spread, depth profiles, returns, and price impact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lob.book import OrderBook
from lob.order import Side


def compute_spread(book: OrderBook) -> int | None:
    """s(t) = a(t) - b(t)."""
    return book.spread


def compute_mid_price(book: OrderBook) -> float | None:
    """m(t) = (a(t) + b(t)) / 2."""
    return book.mid_price


def compute_depth_profile(book: OrderBook, side: Side,
                          n_levels: int = 10) -> list[tuple[int, int]]:
    """Depth profile: list of (price, total_size) from best to worst."""
    return book.depth_profile(side, n_levels)


def compute_relative_depth_profile(book: OrderBook, side: Side,
                                   n_levels: int = 10) -> list[tuple[int, int]]:
    """Relative depth profile: (delta, total_size) from 0 outward."""
    return book.relative_depth_profile(side, n_levels)


def compute_returns(prices: np.ndarray, lag: int = 1) -> np.ndarray:
    """Simple returns: R(t1,t2) = (p(t2) - p(t1)) / p(t1).

    Args:
        prices: Array of prices (mid-price, bid, or ask).
        lag: Number of periods between t1 and t2.
    """
    prices = np.asarray(prices, dtype=float)
    return (prices[lag:] - prices[:-lag]) / prices[:-lag]


def compute_log_returns(prices: np.ndarray, lag: int = 1) -> np.ndarray:
    """Logarithmic returns: r(t1,t2) = log(p(t2) / p(t1)).

    As defined in Gould et al. (2013) Section 2.3.
    """
    prices = np.asarray(prices, dtype=float)
    return np.log(prices[lag:] / prices[:-lag])


def event_time_returns(mid_prices, k: int) -> np.ndarray:
    """Log returns of a mid-price series sampled every k events.

    Most events leave the mid-price unchanged, so returns over a handful of
    events are dominated by exact zeros, which inflates measures such as the
    kurtosis. Choose k large enough that a typical interval contains several
    price changes.
    """
    mid = np.asarray(mid_prices, dtype=float)
    mid = mid[np.isfinite(mid)]
    return np.diff(np.log(mid[::k]))


def compute_price_impact(trade_sizes: np.ndarray,
                         mid_before: np.ndarray,
                         mid_after: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute price impact as a function of trade size.

    Returns:
        (sizes, impacts) arrays suitable for plotting.
        Impact is |mid_after - mid_before|.
    """
    sizes = np.asarray(trade_sizes, dtype=float)
    impacts = np.abs(np.asarray(mid_after, dtype=float) -
                     np.asarray(mid_before, dtype=float))
    return sizes, impacts


def virtual_impact(snapshot: dict, sizes, side: Side = Side.BID) -> np.ndarray:
    """Mid-price move a market order of each size would cause, without executing it.

    Walks the opposite side of a snapshot as returned by
    ``OrderBook.snapshot()``. Positive values are moves in the order's
    direction (up for buys). Entries are NaN when the recorded levels hold
    no more volume than the order would consume, since the new quote is then
    unknown.
    """
    sizes = np.asarray(sizes, dtype=float)
    out = np.full(len(sizes), np.nan)
    best_bid, best_ask = snapshot["best_bid"], snapshot["best_ask"]
    levels = snapshot["asks"] if side == Side.BID else snapshot["bids"]
    if best_bid is None or best_ask is None or not levels:
        return out

    prices = np.array([p for p, _ in levels], dtype=float)
    cum_depth = np.cumsum([d for _, d in levels])
    # Number of levels an order of each size consumes completely.
    n_cleared = np.searchsorted(cum_depth, sizes, side="right")
    known = n_cleared < len(levels)
    new_quote = prices[n_cleared[known]]
    if side == Side.BID:
        out[known] = (new_quote - best_ask) / 2.0
    else:
        out[known] = (best_bid - new_quote) / 2.0
    return out


def mean_virtual_impact(snapshots: list[dict], sizes, side: Side = Side.BID,
                        min_coverage: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Average virtual impact over snapshots, for the sizes most snapshots can absorb.

    Only sizes that at least ``min_coverage`` of the snapshots hold enough
    recorded depth for are kept; otherwise the largest sizes would be
    averaged over only the deepest books. Zero impacts are included.

    Returns:
        (sizes, mean_impact) for the kept sizes.
    """
    sizes = np.asarray(sizes, dtype=float)
    if not snapshots:
        return sizes[:0], np.array([])
    impacts = np.array([virtual_impact(s, sizes, side) for s in snapshots])
    keep = np.mean(np.isfinite(impacts), axis=0) >= min_coverage
    if not keep.any():
        return sizes[:0], np.array([])
    return sizes[keep], np.nanmean(impacts[:, keep], axis=0)


def compute_order_imbalance(book: OrderBook, n_levels: int = 5) -> float:
    """Order imbalance at the top of the book.

    Defined as (bid_volume - ask_volume) / (bid_volume + ask_volume)
    for the top n_levels. Ranges from -1 (all ask) to +1 (all bid).
    """
    bid_depth = book.depth_profile(Side.BID, n_levels)
    ask_depth = book.depth_profile(Side.ASK, n_levels)
    bid_vol = sum(d for _, d in bid_depth)
    ask_vol = sum(d for _, d in ask_depth)
    total = bid_vol + ask_vol
    if total == 0:
        return 0.0
    return (bid_vol - ask_vol) / total


def trades_by_order(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fills into one row per aggressive (taker) order.

    Takes the per-fill table from ``Simulation.trades`` and returns, in the
    order the takers arrived: taker_order_id, time, aggressor, size (total
    lots filled), vwap (volume-weighted average fill price) and n_fills.
    """
    if trades.empty:
        return pd.DataFrame(columns=["taker_order_id", "time", "aggressor", "size", "vwap", "n_fills"])
    t = trades.assign(notional=trades["price"] * trades["size"])
    grouped = t.groupby("taker_order_id", sort=False)
    out = grouped.agg(time=("time", "first"), aggressor=("aggressor", "first"),
                      size=("size", "sum"), notional=("notional", "sum"), n_fills=("price", "size"))
    out["vwap"] = out.pop("notional") / out["size"]
    return out.reset_index()[["taker_order_id", "time", "aggressor", "size", "vwap", "n_fills"]]


def compute_vwap(fill_prices: np.ndarray, fill_sizes: np.ndarray) -> float:
    """Volume-weighted average price."""
    prices = np.asarray(fill_prices, dtype=float)
    sizes = np.asarray(fill_sizes, dtype=float)
    total = sizes.sum()
    if total == 0:
        return 0.0
    return float(np.dot(prices, sizes) / total)


def order_flow_imbalance(best_bid, bid_size, best_ask, ask_size) -> np.ndarray:
    """Order flow imbalance e_n between consecutive book states (Cont, Kukanov and Stoikov 2014).

    Inputs are the best quotes and the depth at them after each event (an empty
    side as a NaN price). For each change from state n-1 to n:

        e_n = 1{b_n >= b_(n-1)} q^b_n - 1{b_n <= b_(n-1)} q^b_(n-1)
            - 1{a_n <= a_(n-1)} q^a_n + 1{a_n >= a_(n-1)} q^a_(n-1)

    so lots added at the bid, or taken from the ask, count as buying pressure.
    The OFI of an interval is the sum of its e_n. Returns n - 1 values; a change
    with an empty side at either end contributes 0.
    """
    pb, qb, pa, qa = (np.asarray(x, dtype=float) for x in (best_bid, bid_size, best_ask, ask_size))
    pb0, pb1, qb0, qb1 = pb[:-1], pb[1:], qb[:-1], qb[1:]
    pa0, pa1, qa0, qa1 = pa[:-1], pa[1:], qa[:-1], qa[1:]
    with np.errstate(invalid="ignore"):
        e = ((pb1 >= pb0) * qb1 - (pb1 <= pb0) * qb0
             - (pa1 <= pa0) * qa1 + (pa1 >= pa0) * qa0)
    valid = np.isfinite(pb0) & np.isfinite(pb1) & np.isfinite(pa0) & np.isfinite(pa1)
    return np.where(valid, e, 0.0)


def ofi_regression(ofi, price_change) -> dict[str, float]:
    """Least-squares fit of price changes on OFI: price_change = alpha + beta * ofi.

    Returns beta (ticks per lot), alpha, r2 and beta_se (its usual standard
    error, which treats intervals as independent).
    """
    x = np.asarray(ofi, dtype=float)
    y = np.asarray(price_change, dtype=float)
    n = len(x)
    if n < 3 or x.var() == 0:
        return {"beta": np.nan, "alpha": np.nan, "r2": np.nan, "beta_se": np.nan}
    beta, alpha = np.polyfit(x, y, 1)
    resid = y - (alpha + beta * x)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
    beta_se = np.sqrt((resid ** 2).sum() / (n - 2) / ((x - x.mean()) ** 2).sum())
    return {"beta": float(beta), "alpha": float(alpha), "r2": float(r2), "beta_se": float(beta_se)}

