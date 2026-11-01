from __future__ import annotations

"""Statistical analysis routines for LOB stylized facts.

Implements the key statistical tools referenced in Gould et al. (2013)
Section 4 for detecting and quantifying empirical regularities:
- Power-law fitting (Clauset et al. MLE method)
- Autocorrelation function
- Hurst exponent (R/S analysis and DFA)
- Hill tail estimator
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar


def fit_power_law(data: np.ndarray, xmin: float | None = None
                  ) -> dict:
    """Fit a power-law distribution using MLE (Clauset et al. 2009).

    P(x) ~ x^{-alpha} for x >= xmin.

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

    # MLE for discrete power law: alpha = 1 + n / sum(ln(x / xmin))
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


def compute_hurst_rs(series: np.ndarray, min_window: int = 10) -> float:
    """Estimate the Hurst exponent using rescaled range (R/S) analysis.

    H > 0.5: long memory (persistent)
    H = 0.5: no memory (random walk)
    H < 0.5: anti-persistent

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

    Returns the estimated tail exponent alpha. For the inverse cubic law
    of returns (Gould et al. Section 4.7), alpha should be near 3.

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


def compute_order_sign_acf(signs: np.ndarray, max_lag: int = 500) -> np.ndarray:
    """ACF of order signs (+1 for buy, -1 for sell).

    Long memory in this series is a key stylized fact
    (Lillo et al. 2005, discussed in Gould et al. Section 4.5).
    """
    return compute_acf(signs, max_lag)
