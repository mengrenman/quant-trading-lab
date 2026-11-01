"""Statistical analysis routines for LOB stylized facts.

Implements the key statistical tools referenced in Gould et al. (2013)
Section 4 for detecting and quantifying empirical regularities:
- Power-law fitting (Clauset et al. MLE method)
- Autocorrelation function
- Hurst exponent (R/S analysis)
- Hill tail estimator
- Time-averaged depth profiles
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def fit_power_law(data: np.ndarray, xmin: float | None = None
                  ) -> dict:
    """Fit a continuous power-law distribution using MLE (Clauset et al. 2009).

    The density is p(x) ~ x^{-alpha} for x >= xmin, so the complementary CDF
    decays as P(X > x) ~ x^{-(alpha - 1)}: a sample whose CCDF falls off as
    x^{-1.5} has alpha = 2.5.

    If xmin is not provided, it is estimated by minimizing the KS distance.

    Returns:
        dict with keys: alpha, xmin, ks_stat, n_tail
    """
    data = np.asarray(data, dtype=float)
    data = data[data > 0]

    if xmin is None:
        xmin = _estimate_xmin(data)

    tail = data[data >= xmin]
    n = len(tail)
    if n < 2:
        return {"alpha": np.nan, "xmin": xmin, "ks_stat": np.nan, "n_tail": n}

    # Continuous MLE (Clauset et al. 2009, eq. 3.1). For integer data with a
    # small xmin it is biased; their discrete approximation uses xmin - 0.5.
    alpha = 1.0 + n / np.sum(np.log(tail / xmin))

    # KS statistic.
    sorted_tail = np.sort(tail)
    empirical_cdf = np.arange(1, n + 1) / n
    theoretical_cdf = 1.0 - (xmin / sorted_tail) ** (alpha - 1)
    ks_stat = np.max(np.abs(empirical_cdf - theoretical_cdf))

    return {"alpha": alpha, "xmin": xmin, "ks_stat": ks_stat, "n_tail": n}


def _estimate_xmin(data: np.ndarray) -> float:
    """Estimate xmin by minimizing KS distance over candidate values."""
    candidates = np.unique(data)
    if len(candidates) > 100:
        candidates = np.percentile(data, np.linspace(1, 99, 100))
        candidates = np.unique(candidates)

    best_ks = np.inf
    best_xmin = candidates[0]

    for xm in candidates:
        tail = data[data >= xm]
        n = len(tail)
        if n < 10:
            continue
        alpha = 1.0 + n / np.sum(np.log(tail / xm))
        sorted_tail = np.sort(tail)
        ecdf = np.arange(1, n + 1) / n
        tcdf = 1.0 - (xm / sorted_tail) ** (alpha - 1)
        ks = np.max(np.abs(ecdf - tcdf))
        if ks < best_ks:
            best_ks = ks
            best_xmin = xm

    return best_xmin


def compute_acf(series: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """Compute the autocorrelation function up to max_lag.

    Returns array of shape (max_lag+1,) where index 0 is lag 0 (= 1.0).
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    mean = series.mean()
    var = series.var()
    if var == 0:
        return np.ones(max_lag + 1)

    acf = np.zeros(max_lag + 1)
    centered = series - mean
    for lag in range(max_lag + 1):
        if lag >= n:
            break
        acf[lag] = np.mean(centered[:n - lag] * centered[lag:]) / var
    return acf


def white_noise_band(n: int, z: float = 1.96) -> float:
    """Half-width of the approximate 95% band for the ACF of n white-noise points."""
    return z / np.sqrt(n)


def compute_hurst_rs(series: np.ndarray, min_window: int = 10) -> float:
    """Estimate the Hurst exponent using rescaled range (R/S) analysis.

    H > 0.5: long memory (persistent)
    H = 0.5: no memory (random walk)
    H < 0.5: anti-persistent

    R/S is biased upward in small windows, and in a finite sample it cannot
    tell long memory apart from strong short memory (e.g. a persistent GARCH
    process), so compare against a white-noise reference of the same length.

    See Gould et al. (2013) Section 4.7 and references therein.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)

    # Compute R/S for different window sizes.
    window_sizes = []
    rs_values = []

    size = min_window
    while size <= n // 2:
        window_sizes.append(size)
        n_windows = n // size
        rs_list = []
        for i in range(n_windows):
            window = series[i * size:(i + 1) * size]
            mean = window.mean()
            cumdev = np.cumsum(window - mean)
            R = cumdev.max() - cumdev.min()
            S = window.std(ddof=1)
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            rs_values.append(np.mean(rs_list))
        size = int(size * 1.5)

    if len(window_sizes) < 3:
        return 0.5

    log_n = np.log(window_sizes)
    log_rs = np.log(rs_values)
    slope, _, _, _, _ = stats.linregress(log_n, log_rs)
    return slope


def hill_estimator(data: np.ndarray, k: int | None = None) -> float:
    """Hill estimator for the tail index of a heavy-tailed distribution.

    Returns the estimated tail exponent alpha of P(|X| > x) ~ x^{-alpha}. For
    the inverse cubic law of returns (Gould et al. Section 4.7), alpha should
    be near 3.

    Assumes continuous data: ties among the largest values (e.g. returns that
    only take multiples of half a tick) bias the estimate badly.

    Args:
        data: Positive values (e.g., absolute returns).
        k: Number of order statistics to use. Defaults to sqrt(n).
    """
    data = np.sort(np.abs(data))[::-1]
    data = data[data > 0]
    n = len(data)
    if k is None:
        k = max(10, int(np.sqrt(n)))
    k = min(k, n - 1)

    if k < 2:
        return np.nan

    log_ratios = np.log(data[:k]) - np.log(data[k])
    gamma = np.mean(log_ratios)
    if gamma <= 0:
        return np.nan
    return 1.0 / gamma


def compute_volatility_clustering(returns: np.ndarray,
                                  max_lag: int = 100) -> np.ndarray:
    """ACF of absolute returns — measures volatility clustering.

    Slow decay of this ACF indicates volatility clustering
    (Gould et al. Section 4.7).
    """
    return compute_acf(np.abs(returns), max_lag)


def compute_rank_volatility_acf(returns: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """ACF of the ranks of absolute returns: volatility clustering robust to outliers.

    A handful of extreme returns can dominate the ACF of |r| itself and hide
    clustering; ranking first bounds the influence of any single return.
    """
    return compute_acf(stats.rankdata(np.abs(returns)), max_lag)


def compute_order_sign_acf(signs: np.ndarray, max_lag: int = 500) -> np.ndarray:
    """ACF of order signs (+1 for buy, -1 for sell).

    Long memory in this series is a key stylized fact
    (Lillo et al. 2005, discussed in Gould et al. Section 4.5).
    """
    return compute_acf(signs, max_lag)


def fit_loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of log(y) on log(x), over points where both are positive."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if mask.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0])


def mean_depth_profile(snapshots: list[dict], n_ticks: int = 30
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time-averaged depth at each distance from the same-side best quote.

    Empirical depth profiles (Gould et al. Section 4.3) are averages over
    time, so a single snapshot is not enough to judge their shape.

    Args:
        snapshots: Book snapshots as recorded by ``Simulation`` (each holds
            ``best_bid``, ``best_ask``, ``bids`` and ``asks``). Snapshots with
            an empty side are skipped.
        n_ticks: Number of relative price levels (0 = at the best quote).
            Must not exceed the number of levels stored per snapshot (50).

    Returns:
        (deltas, mean_bid_depth, mean_ask_depth), with empty levels counted
        as zero depth.
    """
    bid_sum = np.zeros(n_ticks)
    ask_sum = np.zeros(n_ticks)
    used = 0
    for snap in snapshots:
        best_bid, best_ask = snap["best_bid"], snap["best_ask"]
        if best_bid is None or best_ask is None:
            continue
        for price, depth in snap["bids"]:
            k = best_bid - price
            if k < n_ticks:
                bid_sum[k] += depth
        for price, depth in snap["asks"]:
            k = price - best_ask
            if k < n_ticks:
                ask_sum[k] += depth
        used += 1
    deltas = np.arange(n_ticks)
    if used == 0:
        return deltas, bid_sum, ask_sum
    return deltas, bid_sum / used, ask_sum / used


def classify_depth_profile(profile: np.ndarray, min_rise: float = 0.05,
                           min_fall: float = 0.2) -> tuple[str, float, float, int]:
    """Describe a mean depth profile as a hump, a plateau, or peaking at the best quote.

    Args:
        profile: Mean depth by distance from the best quote (index 0 = at it).
        min_rise: Fraction by which the deepest level beyond the best quote
            must exceed the depth at the quote.
        min_fall: Fraction by which the far end must fall below that peak
            for the profile to count as a hump rather than a plateau.

    Returns:
        (shape, rise, fall, peak): shape is "hump", "plateau" or "best quote";
        peak is the distance of the deepest level beyond the best quote.
    """
    profile = np.asarray(profile, dtype=float)
    peak = 1 + int(np.argmax(profile[1:]))
    rise = profile[peak] / profile[0] - 1 if profile[0] > 0 else np.inf
    fall = 1 - profile[-1] / profile[peak] if profile[peak] > 0 else 0.0
    if rise < min_rise:
        shape = "best quote"
    elif fall < min_fall:
        shape = "plateau"
    else:
        shape = "hump"
    return shape, float(rise), float(fall), peak
