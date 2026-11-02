"""Reusable plotting helpers for LOB visualization.

Provides consistent, publication-quality plots used across all notebooks.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from lob.book import OrderBook
from lob.order import Side
from lob.analysis.stylized_facts import white_noise_band


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

    Prices run up the vertical axis. Bids extend to the left (green), asks to
    the right (red), and the dashed line marks the mid-price.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    bids = book.depth_profile(Side.BID, n_levels)
    asks = book.depth_profile(Side.ASK, n_levels)
    if not bids and not asks:
        ax.set_title(title + " (empty)")
        return ax

    if bids:
        prices, depths = zip(*bids, strict=True)
        ax.barh(prices, [-d for d in depths], height=0.8,
                color="#2ecc71", alpha=0.8, label="Bids")
    if asks:
        prices, depths = zip(*asks, strict=True)
        ax.barh(prices, depths, height=0.8,
                color="#e74c3c", alpha=0.8, label="Asks")

    if book.mid_price is not None:
        ax.axhline(y=book.mid_price, color="gray", linestyle="--",
                   linewidth=0.8, label="Mid-price")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_yticks([p for p, _ in bids + asks])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{abs(x):g}"))
    ax.set_xlabel("Depth (lots)")
    ax.set_ylabel("Price (ticks)")
    ax.set_title(title)
    ax.legend(loc="lower right")

    return ax


def plot_depth_profile(book: OrderBook, n_levels: int = 20,
                       ax: plt.Axes | None = None) -> plt.Axes:
    """Plot the relative depth profile (volume vs distance from best).

    This is a single snapshot; the hump shape discussed in Gould et al.
    Section 4.3 refers to the time-averaged profile (see
    ``lob.analysis.stylized_facts.mean_depth_profile``).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    bid_profile = book.relative_depth_profile(Side.BID, n_levels)
    ask_profile = book.relative_depth_profile(Side.ASK, n_levels)

    # Side-by-side bars at each distance, so the two best quotes do not overlap.
    if bid_profile:
        deltas, depths = zip(*bid_profile, strict=True)
        ax.bar([d - 0.2 for d in deltas], depths, color="#2ecc71",
               alpha=0.7, label="Bids", width=0.4)
    if ask_profile:
        deltas, depths = zip(*ask_profile, strict=True)
        ax.bar([d + 0.2 for d in deltas], depths, color="#e74c3c",
               alpha=0.7, label="Asks", width=0.4)

    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlabel("Distance from the same-side best quote (ticks)")
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
             ax: plt.Axes | None = None, n: int | None = None) -> plt.Axes:
    """Plot the autocorrelation function.

    Args:
        acf_values: ACF from lag 0 upward.
        n: Length of the series the ACF was computed from. When given, the
            approximate 95% band for white noise (+/- 1.96 / sqrt(n)) is drawn.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    lags = np.arange(len(acf_values))
    ax.bar(lags, acf_values, color="#3498db", alpha=0.7, width=1.0)
    ax.axhline(y=0, color="black", linewidth=0.5)

    if n is not None:
        band = white_noise_band(n)
        ax.axhline(y=band, color="red", linestyle="--", alpha=0.5,
                   label=f"95% band, white noise (n={n:,})")
        ax.axhline(y=-band, color="red", linestyle="--", alpha=0.5)
        ax.legend()

    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.set_title(title)
    return ax


def plot_price_impact(sizes: np.ndarray, impacts: np.ndarray,
                      ax: plt.Axes | None = None) -> plt.Axes:
    """Log-log plot of mean price impact against order size.

    Impacts are averaged per distinct size with zeros included: dropping
    orders that did not move the price would overstate the impact of small
    orders. The legend reports the fitted log-log slope next to a
    square-root reference (Gould et al. Section 4.6).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    sizes = np.asarray(sizes, dtype=float)
    impacts = np.asarray(impacts, dtype=float)
    keep = np.isfinite(sizes) & np.isfinite(impacts) & (sizes > 0)
    sizes, impacts = sizes[keep], impacts[keep]

    uniq = np.unique(sizes)
    mean_impact = np.array([impacts[sizes == s].mean() for s in uniq])
    pos = mean_impact > 0
    if pos.sum() < 2:
        ax.set_title("Price Impact (not enough data)")
        return ax
    x, y = uniq[pos], mean_impact[pos]

    ax.plot(x, y, "o", color="#3498db", alpha=0.8, label="Mean impact")
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    xs = np.geomspace(x.min(), x.max(), 100)
    ax.plot(xs, np.exp(intercept) * xs ** slope, "r-", linewidth=2,
            label=f"Fit: slope {slope:.2f}")
    ax.plot(xs, y[0] * (xs / x[0]) ** 0.5, "k--", alpha=0.5,
            label="Slope 0.5 (square root)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Order size")
    ax.set_ylabel("Mean price impact")
    ax.set_title("Price Impact vs Order Size")
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
