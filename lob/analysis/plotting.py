from __future__ import annotations

"""Reusable plotting helpers for LOB visualization.

Provides consistent, publication-quality plots used across all notebooks.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from lob.book import OrderBook
from lob.order import Side


def set_style():
    """Set a clean plotting style for all notebooks."""
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
    })


def plot_book_snapshot(book: OrderBook, n_levels: int = 10,
                       ax: plt.Axes | None = None, title: str = "Order Book") -> plt.Axes:
    """Visualize the LOB as a horizontal bar chart (price ladder).

    Bids on the left (green), asks on the right (red).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    bids = book.depth_profile(Side.BID, n_levels)
    asks = book.depth_profile(Side.ASK, n_levels)

    bid_prices = [p for p, _ in bids]
    bid_depths = [d for _, d in bids]
    ask_prices = [p for p, _ in asks]
    ask_depths = [d for _, d in asks]

    all_prices = bid_prices + ask_prices
    if not all_prices:
        ax.set_title(title + " (empty)")
        return ax

    y_pos_bids = range(len(bid_prices))
    y_pos_asks = range(len(ask_prices))

    ax.barh([str(p) for p in bid_prices], bid_depths,
            color="#2ecc71", alpha=0.8, label="Bids")
    ax.barh([str(p) for p in ask_prices], [-d for d in ask_depths],
            color="#e74c3c", alpha=0.8, label="Asks")

    ax.axhline(y=-0.5, color="gray", linewidth=0.5)
    ax.set_xlabel("Depth (lots)")
    ax.set_ylabel("Price (ticks)")
    ax.set_title(title)
    ax.legend(loc="lower right")

    return ax


def plot_depth_profile(book: OrderBook, n_levels: int = 20,
                       ax: plt.Axes | None = None) -> plt.Axes:
    """Plot the relative depth profile (volume vs distance from best).

    The hump shape is a key empirical finding (Gould et al. Section 4.3).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    bid_profile = book.relative_depth_profile(Side.BID, n_levels)
    ask_profile = book.relative_depth_profile(Side.ASK, n_levels)

    if bid_profile:
        deltas, depths = zip(*bid_profile)
        ax.bar([-d for d in deltas], depths, color="#2ecc71",
               alpha=0.7, label="Bids", width=0.8)
    if ask_profile:
        deltas, depths = zip(*ask_profile)
        ax.bar(list(deltas), depths, color="#e74c3c",
               alpha=0.7, label="Asks", width=0.8)

    ax.set_xlabel("Relative price (ticks from best)")
    ax.set_ylabel("Depth (lots)")
    ax.set_title("Depth Profile")
    ax.legend()
    return ax


def plot_price_series(times: np.ndarray, mid_prices: np.ndarray,
                      bid_prices: np.ndarray | None = None,
                      ask_prices: np.ndarray | None = None,
                      ax: plt.Axes | None = None) -> plt.Axes:
    """Plot mid-price (and optionally bid/ask) over time."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(times, mid_prices, color="#3498db", linewidth=0.8, label="Mid price")
    if bid_prices is not None:
        ax.plot(times, bid_prices, color="#2ecc71", linewidth=0.5, alpha=0.5, label="Bid")
    if ask_prices is not None:
        ax.plot(times, ask_prices, color="#e74c3c", linewidth=0.5, alpha=0.5, label="Ask")

    ax.set_xlabel("Time")
    ax.set_ylabel("Price (ticks)")
    ax.set_title("Price Series")
    ax.legend()
    return ax


def plot_return_distribution(returns: np.ndarray, n_bins: int = 100,
                             ax: plt.Axes | None = None) -> plt.Axes:
    """Plot return distribution with Gaussian overlay.

    Heavy tails relative to the normal are a key stylized fact
    (Gould et al. Section 4.7).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    returns = np.asarray(returns)
    returns = returns[np.isfinite(returns)]

    ax.hist(returns, bins=n_bins, density=True, alpha=0.7,
            color="#3498db", label="Empirical")

    # Gaussian overlay.
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 300)
    from scipy.stats import norm
    ax.plot(x, norm.pdf(x, mu, sigma), "r-", linewidth=2,
            label=f"Normal($\\mu$={mu:.2e}, $\\sigma$={sigma:.2e})")

    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.set_title("Return Distribution")
    ax.legend()
    ax.set_yscale("log")
    return ax


def plot_acf(acf_values: np.ndarray, title: str = "Autocorrelation Function",
             ax: plt.Axes | None = None) -> plt.Axes:
    """Plot the autocorrelation function."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    lags = np.arange(len(acf_values))
    ax.bar(lags, acf_values, color="#3498db", alpha=0.7, width=1.0)
    ax.axhline(y=0, color="black", linewidth=0.5)

    # Approximate 95% confidence interval for white noise.
    n = len(acf_values) * 10  # rough estimate
    ci = 1.96 / np.sqrt(n)
    ax.axhline(y=ci, color="red", linestyle="--", alpha=0.5, label="95% CI")
    ax.axhline(y=-ci, color="red", linestyle="--", alpha=0.5)

    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_price_impact(sizes: np.ndarray, impacts: np.ndarray,
                      ax: plt.Axes | None = None) -> plt.Axes:
    """Log-log plot of price impact vs trade size with sqrt fit.

    The concave (square-root) relationship is discussed in
    Gould et al. Section 4.6.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    sizes = np.asarray(sizes, dtype=float)
    impacts = np.asarray(impacts, dtype=float)

    # Bin the data for cleaner visualization.
    mask = (sizes > 0) & (impacts > 0)
    sizes, impacts = sizes[mask], impacts[mask]

    if len(sizes) == 0:
        ax.set_title("Price Impact (no data)")
        return ax

    ax.scatter(sizes, impacts, alpha=0.3, s=10, color="#3498db", label="Data")

    # Square-root reference line.
    s_range = np.linspace(sizes.min(), sizes.max(), 100)
    # Fit: impact = c * size^0.5
    c = np.median(impacts / np.sqrt(sizes))
    ax.plot(s_range, c * np.sqrt(s_range), "r-", linewidth=2,
            label=r"$\Delta p \propto V^{0.5}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Trade size")
    ax.set_ylabel("Price impact")
    ax.set_title("Price Impact vs Trade Size")
    ax.legend()
    return ax


def plot_spread_timeseries(times: np.ndarray, spreads: np.ndarray,
                           ax: plt.Axes | None = None) -> plt.Axes:
    """Plot bid-ask spread over time."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(times, spreads, color="#9b59b6", linewidth=0.5, alpha=0.8)
    mean_spread = np.nanmean(spreads)
    ax.axhline(y=mean_spread, color="red", linestyle="--",
               label=f"Mean = {mean_spread:.2f}")

    ax.set_xlabel("Time")
    ax.set_ylabel("Spread (ticks)")
    ax.set_title("Bid-Ask Spread")
    ax.legend()
    return ax
