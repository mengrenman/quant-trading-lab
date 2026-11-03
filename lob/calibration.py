"""Estimate the parameters of the Mike and Farmer (2008) model from order flow.

Mike and Farmer built their model from regularities measured directly in order flow,
not by matching the model's output to statistics. This module does the same for
``MikeFarmer2008Model``, from an event log in the simulator's format (the DataFrame
``Simulation.run()`` returns: time, event_type, side, price, size, order_id, where a
cancellation carries the ID of the order it removes). Each part of the model has its own
maximum-likelihood estimator:

- **Order arrivals** form a Poisson process: ``lam`` is the number of orders over the
  elapsed time.
- **Prices** are set ``round(loc + scale * t_nu)`` ticks less aggressive than the
  same-side best quote: ``nu``, ``loc`` and ``scale`` maximize the likelihood of the
  observed integer offsets under that rounded Student's t.
- **Order signs** are the signs of Gaussian FARIMA(0, H - 1/2, 0) noise, so their
  autocorrelation is (2/pi) arcsin of the noise's; ``hurst`` is fitted to the sample
  autocorrelation over the first lags.
- **Cancellations**: each resting order is canceled at rate
  ``A (1 - exp(-y)) (s + B) / (1 + B)``, with y its distance to the opposite best quote
  over that distance when placed and s the share of resting orders on its side. Replaying
  the log through a book gives the likelihood; the best A has a closed form for each B,
  which leaves a one-dimensional maximization over B (``imbalance_offset``).

The log must start from an empty book (``SimulationConfig(warmup_events=0)``) so that the
replay knows every order's placement. Real data would have to be converted to this form
first; order-level data such as LOBSTER's message files carry the needed information.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import optimize, stats

from lob.book import OrderBook
from lob.matching import PriceTimeEngine
from lob.order import Side, limit_order, market_order

SIGN_LAGS = 100  # lags of the sign autocorrelation fitted for the Hurst exponent


def estimate_arrival_rate(log: pd.DataFrame) -> float:
    """Orders (limit and market) per unit time."""
    orders = log["event_type"] != "cancel"
    span = log["time"].iloc[-1] - log["time"].iloc[0]
    return float(orders.sum() / span) if span > 0 else math.nan


# ---- Order signs ----

def farima_acf(d: float, max_lag: int) -> np.ndarray:
    """Autocorrelation of FARIMA(0, d, 0) noise at lags 1..max_lag."""
    k = np.arange(1, max_lag + 1)
    return np.cumprod((k - 1 + d) / (k - d))


def sample_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation at lags 1..max_lag (as ``compute_acf`` defines it), via FFT."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    spectrum = np.fft.rfft(x, 2 ** int(np.ceil(np.log2(2 * n))))
    acov = np.fft.irfft(spectrum * np.conj(spectrum))[: max_lag + 1]
    acov /= np.arange(n, n - max_lag - 1, -1)  # mean over the n - k products, as compute_acf does
    return acov[1:] / acov[0]  # normalized by the variance (lag 0)


def estimate_sign_hurst(signs: np.ndarray, max_lag: int = SIGN_LAGS) -> float:
    """Hurst exponent of order signs that are signs of Gaussian FARIMA(0, H - 1/2, 0) noise."""
    acf = sample_acf(signs, max_lag)

    def loss(d: float) -> float:
        return float(((acf - 2 / math.pi * np.arcsin(farima_acf(d, max_lag))) ** 2).sum())

    d = optimize.minimize_scalar(loss, bounds=(0.0, 0.499), method="bounded").x
    return float(d + 0.5)


# ---- Prices relative to the same-side best quote ----

def relative_offsets(log: pd.DataFrame) -> np.ndarray:
    """Each order's distance from the same-side best quote before it arrived (positive: less aggressive).

    Orders that arrived while their own side of the book was empty are left out.
    """
    before_bid = log["best_bid"].shift(1)
    before_ask = log["best_ask"].shift(1)
    orders = (log["event_type"] == "limit").to_numpy()
    bid = (log["side"] == "bid").to_numpy()
    offset = np.where(bid, before_bid - log["price"], log["price"] - before_ask)
    keep = orders & np.isfinite(offset)
    return offset[keep].astype(int)


def fit_rounded_t(offsets: np.ndarray) -> dict[str, float]:
    """Maximum likelihood (nu, loc, scale) for integers drawn as round(loc + scale * t_nu)."""
    values, counts = np.unique(offsets, return_counts=True)

    def neg_log_likelihood(theta: np.ndarray) -> float:
        nu, loc, scale = math.exp(theta[0]), theta[1], math.exp(theta[2])
        mass = (stats.t.cdf((values + 0.5 - loc) / scale, nu) - stats.t.cdf((values - 0.5 - loc) / scale, nu))
        return float(-(counts * np.log(np.maximum(mass, 1e-300))).sum())

    start = np.array([math.log(2.0), float(np.median(offsets)),
                      math.log(max(1.0, stats.iqr(offsets) / 2))])
    theta = optimize.minimize(neg_log_likelihood, start, method="Nelder-Mead",
                              options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 4000}).x
    return {"nu": math.exp(theta[0]), "loc": float(theta[1]), "scale": math.exp(theta[2])}


# ---- Cancellations ----

class _RestingSide:
    """Prices and placement distances of one side's resting orders, in arrays (swap-remove)."""

    def __init__(self) -> None:
        self.price = np.empty(256)
        self.placed = np.empty(256)
        self.ids: list[int] = []
        self.pos: dict[int, int] = {}

    def add(self, oid: int, price: float, placed: float) -> None:
        n = len(self.ids)
        if n == len(self.price):
            self.price = np.resize(self.price, 2 * n)
            self.placed = np.resize(self.placed, 2 * n)
        self.price[n], self.placed[n] = price, placed
        self.ids.append(oid)
        self.pos[oid] = n

    def remove(self, oid: int) -> Optional[int]:
        i = self.pos.pop(oid, None)
        if i is None:
            return None
        last = len(self.ids) - 1
        if i != last:
            moved = self.ids[last]
            self.ids[i], self.price[i], self.placed[i] = moved, self.price[last], self.placed[last]
            self.pos[moved] = i
        self.ids.pop()
        return i

    def weight(self, opposite: Optional[float]) -> float:
        """Sum over the side's orders of 1 - exp(-y), y = |opposite - price| / placement distance."""
        n = len(self.ids)
        if n == 0:
            return 0.0
        if opposite is None:
            return n * (1 - math.exp(-1.0))
        y = np.abs(opposite - self.price[:n]) / self.placed[:n]
        return float((1 - np.exp(-y)).sum())

    def weight_of(self, oid: int, opposite: Optional[float]) -> float:
        i = self.pos[oid]
        y = 1.0 if opposite is None else abs(opposite - self.price[i]) / self.placed[i]
        return 1 - math.exp(-y)


def cancellation_statistics(log: pd.DataFrame, initial_mid: Optional[float] = None) -> dict[str, Any]:
    """Replay the log and collect what the cancellation likelihood needs.

    ``initial_mid`` is the model's reference price while the book is empty (its
    ``initial_mid``); by default the first order's price stands in, which affects only the
    placement distances of the first few orders.

    Returns the time-weighted sums W_U = sum dt (s_b Y_b + s_a Y_a) and W_V = sum dt (Y_b + Y_a)
    over the intervals between events (Y_side is the side's sum of 1 - exp(-y) and s_side
    its share of resting orders), and for each cancellation the canceled order's
    1 - exp(-y) and side share just before it.
    """
    book, engine = OrderBook(), PriceTimeEngine()
    sides = {Side.BID: _RestingSide(), Side.ASK: _RestingSide()}
    last_mid = initial_mid if initial_mid is not None else float(log["price"].iloc[0])
    w_u = w_v = 0.0
    cancel_weight, cancel_share = [], []
    times = log["time"].to_numpy()
    kinds = log["event_type"].to_numpy()
    side_values = log["side"].to_numpy()
    prices = log["price"].to_numpy()
    sizes = log["size"].to_numpy()
    ids = log["order_id"].to_numpy()

    def quote(side: Side) -> float:
        """The opposite-side reference the model uses, with its stand-ins for an empty side."""
        if side == Side.ASK:
            if book.best_ask is not None:
                return book.best_ask
            if book.best_bid is not None:
                return book.best_bid + 1
            return math.floor(last_mid) + 1
        if book.best_bid is not None:
            return book.best_bid
        if book.best_ask is not None:
            return book.best_ask - 1
        return math.floor(last_mid) - 1

    for i in range(len(log)):
        oid = int(ids[i])
        side = Side.BID if side_values[i] == "bid" else Side.ASK
        if kinds[i] == "cancel":
            resting = book.get_order(oid)
            if resting is not None:
                n_side, n_all = book.num_orders(resting.side), book.num_orders()
                opposite = book.best_ask if resting.side == Side.BID else book.best_bid
                cancel_weight.append(sides[resting.side].weight_of(oid, opposite))
                cancel_share.append(n_side / n_all)
                book.cancel_order(oid)
                sides[resting.side].remove(oid)
        else:
            if book.mid_price is not None:
                last_mid = book.mid_price
            if kinds[i] == "limit":
                price = int(prices[i])
                opposite = quote(Side.ASK if side == Side.BID else Side.BID)
                placed = max(1, (opposite - price) if side == Side.BID else (price - opposite))
                order = limit_order(price, int(sizes[i]), float(times[i]), side, order_id=oid)
            else:
                order, placed = market_order(int(sizes[i]), float(times[i]), side, order_id=oid), None
            fills = engine.process(book, order)
            for f in fills:
                if book.get_order(f.maker_order_id) is None:
                    sides[Side.ASK if side == Side.BID else Side.BID].remove(f.maker_order_id)
            if placed is not None and book.get_order(oid) is not None:
                sides[side].add(oid, price, placed)
        # The state now holds until the next event.
        if i + 1 < len(log):
            dt = times[i + 1] - times[i]
            n_b, n_a = book.num_orders(Side.BID), book.num_orders(Side.ASK)
            if dt > 0 and n_b + n_a > 0:
                y_b = sides[Side.BID].weight(book.best_ask)
                y_a = sides[Side.ASK].weight(book.best_bid)
                total = n_b + n_a
                w_u += dt * (n_b / total * y_b + n_a / total * y_a)
                w_v += dt * (y_b + y_a)
    return {"w_u": w_u, "w_v": w_v, "weight": np.array(cancel_weight), "share": np.array(cancel_share)}


def fit_cancellation(cancel_stats: dict[str, Any], b_max: float = 50.0) -> dict[str, float]:
    """Maximum likelihood (cancel_rate A, imbalance_offset B) from ``cancellation_statistics``.

    log L(A, B) = sum_j [log A + log g_j + log(s_j + B) - log(1 + B)] - A (W_U + B W_V) / (1 + B).
    For each B the best A is n (1 + B) / (W_U + B W_V); what remains to maximize over B is
    sum_j log(s_j + B) - n log(W_U + B W_V).
    """
    n = len(cancel_stats["share"])
    shares, w_u, w_v = cancel_stats["share"], cancel_stats["w_u"], cancel_stats["w_v"]

    def neg_profile(b: float) -> float:
        return float(-(np.log(shares + b).sum() - n * math.log(w_u + b * w_v)))

    b = optimize.minimize_scalar(neg_profile, bounds=(0.0, b_max), method="bounded",
                                 options={"xatol": 1e-6}).x
    return {"cancel_rate": n * (1 + b) / (w_u + b * w_v), "imbalance_offset": float(b)}


def calibrate_mf2008(log: pd.DataFrame, initial_mid: Optional[float] = None) -> dict[str, Any]:
    """Every parameter of ``MikeFarmer2008Model`` from an event log that starts from an empty book."""
    signs = np.where(log.loc[log["event_type"] != "cancel", "side"] == "bid", 1.0, -1.0)
    sizes = log.loc[log["event_type"] != "cancel", "size"]
    prices = fit_rounded_t(relative_offsets(log))
    cancels = fit_cancellation(cancellation_statistics(log, initial_mid))
    return {
        "lam": estimate_arrival_rate(log),
        **cancels,
        "hurst": estimate_sign_hurst(signs),
        **prices,
        "lot_size": int(sizes.mode().iloc[0]),
        "n_orders": int(len(signs)),
        "n_cancels": int((log["event_type"] == "cancel").sum()),
    }
