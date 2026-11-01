"""Simplified Mike and Farmer (2008) agent-based model.

This model incorporates three empirical features missing from the basic ZI
model (Gould et al. 2013, Section 5.2.5):

1. Power-law distributed relative limit order prices.
2. Long-memory (fractionally integrated) order signs.
3. Heavy-tailed order sizes.

It simplifies the original in two ways: relative prices are measured from
the mid-price and are always at least one tick (Mike and Farmer measure them
from the same-side best quote and allow orders inside the spread), and
cancellations are uniform at a constant rate per order (the original makes
them depend on the order's position and on the state of the book).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from lob.book import OrderBook
from lob.order import Order, Side, limit_order, market_order, cancel_order


def _generate_fractional_noise(n: int, H: float,
                               rng: np.random.Generator) -> np.ndarray:
    """Generate fractional Gaussian noise via the Davies-Harte (FFT) method.

    The exact fGn autocovariance gamma(k) is embedded in a circulant matrix
    whose first row is [gamma(0), ..., gamma(n), gamma(n-1), ..., gamma(1)].
    The row must be symmetric: without the mirrored half, the generated
    series is fGn mixed with an equal amount of white noise and every
    autocorrelation comes out halved.

    Args:
        n: Length of the series.
        H: Hurst exponent (0.5 = white noise, >0.5 = long memory).
        rng: Numpy random generator.

    Returns:
        Array of length n with unit variance and the fGn autocorrelation.
    """
    k = np.arange(n + 1)
    gamma = 0.5 * (np.abs(k - 1) ** (2 * H) - 2 * k ** (2 * H) +
                   (k + 1) ** (2 * H))
    row = np.concatenate([gamma, gamma[-2:0:-1]])
    m = len(row)

    eigenvalues = np.fft.fft(row).real
    # Clamp small negatives from numerical noise.
    eigenvalues = np.maximum(eigenvalues, 0)

    z = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    w = np.fft.fft(np.sqrt(eigenvalues / m) * z)
    return w.real[:n]


class MikeFarmerModel:
    """Simplified Mike and Farmer (2008) agent-based LOB model.

    Parameters:
        mu: Market order arrival rate.
        lam: Limit order arrival rate.
        delta: Cancellation rate per active order.
        alpha_price: Tail exponent of the relative-price distribution,
            P(delta >= x) ~ x^{-alpha_price}.
        hurst: Hurst exponent for order sign long memory (H > 0.5).
        size_mean: Mean of log order size (log-normal).
        size_sigma: Std of log order size.
        initial_mid: Starting mid-price.
        lot_size: Minimum order size.
    """

    def __init__(
        self,
        mu: float = 1.0,
        lam: float = 2.0,
        delta: float = 0.05,
        alpha_price: float = 1.5,
        hurst: float = 0.7,
        size_mean: float = 1.0,
        size_sigma: float = 0.5,
        initial_mid: int = 10000,
        lot_size: int = 1,
    ):
        self.mu = mu
        self.lam = lam
        self.delta = delta
        self.alpha_price = alpha_price
        self.hurst = hurst
        self.size_mean = size_mean
        self.size_sigma = size_sigma
        self.initial_mid = initial_mid
        self.lot_size = lot_size
        self._t = 0.0
        self._last_mid = float(initial_mid)
        self._signs: np.ndarray | None = None
        self._sign_idx = 0

    def _ensure_signs(self, rng: np.random.Generator, n: int = 50000) -> None:
        """Pre-generate a block of long-memory signs.

        Successive blocks are independent, so memory does not carry across
        block boundaries.
        """
        if self._signs is None or self._sign_idx >= len(self._signs):
            fgn = _generate_fractional_noise(n, self.hurst, rng)
            self._signs = np.sign(fgn)
            self._signs[self._signs == 0] = 1.0
            self._sign_idx = 0

    def _next_sign(self, rng: np.random.Generator) -> int:
        """Get the next order sign (+1 buy, -1 sell)."""
        self._ensure_signs(rng)
        sign = int(self._signs[self._sign_idx])
        self._sign_idx += 1
        return sign

    def _sample_relative_price(self, rng: np.random.Generator) -> int:
        """Sample relative price from a power-law distribution.

        P(delta >= x) ~ x^{-alpha}, so delta = xmin * U^{-1/alpha}
        where U ~ Uniform(0,1).
        """
        xmin = 1.0
        u = rng.random()
        if u == 0:
            u = 1e-10
        delta = xmin * u ** (-1.0 / self.alpha_price)
        return max(1, int(delta))

    def _sample_size(self, rng: np.random.Generator) -> int:
        """Sample order size from a log-normal distribution."""
        raw = rng.lognormal(self.size_mean, self.size_sigma)
        size = max(self.lot_size, int(round(raw)))
        # Round to lot size.
        size = (size // self.lot_size) * self.lot_size
        return max(self.lot_size, size)

    def generate_event(self, book: OrderBook, t: float,
                       rng: np.random.Generator) -> Optional[Order]:
        # Honor the caller's clock when it is ahead of ours.
        self._t = max(self._t, t)
        n_active = book.num_orders()

        limit_rate = self.lam
        market_rate = self.mu
        cancel_rate = self.delta * n_active
        total_rate = limit_rate + market_rate + cancel_rate

        if total_rate <= 0:
            return self._bootstrap_order(book, rng)

        dt = rng.exponential(1.0 / total_rate)
        self._t += dt

        u = rng.random() * total_rate

        if u < limit_rate:
            return self._generate_limit_order(book, rng)
        elif u < limit_rate + market_rate:
            return self._generate_market_order(book, rng)
        else:
            return self._generate_cancellation(book, rng)

    def _generate_limit_order(self, book: OrderBook,
                              rng: np.random.Generator) -> Order:
        sign = self._next_sign(rng)
        side = Side.BID if sign > 0 else Side.ASK
        delta = self._sample_relative_price(rng)
        size = self._sample_size(rng)

        mid = book.mid_price
        if mid is None:
            mid = self._last_mid
        else:
            self._last_mid = mid

        # Round toward the mid on both sides so a bid and an ask with the same
        # delta sit at the same distance from it, even when the mid falls
        # between ticks.
        if side == Side.BID:
            price = max(1, math.ceil(mid - delta))
        else:
            price = math.floor(mid + delta)

        return limit_order(price=price, size=size,
                           timestamp=self._t, side=side)

    def _generate_market_order(self, book: OrderBook,
                               rng: np.random.Generator) -> Optional[Order]:
        sign = self._next_sign(rng)
        side = Side.BID if sign > 0 else Side.ASK
        size = self._sample_size(rng)

        if side == Side.BID and book.best_ask is None:
            return None
        if side == Side.ASK and book.best_bid is None:
            return None

        return market_order(size=size, timestamp=self._t, side=side)

    def _generate_cancellation(self, book: OrderBook,
                               rng: np.random.Generator) -> Optional[Order]:
        chosen_id = book.random_order_id(rng)
        if chosen_id is None:
            return None
        return cancel_order(order_id=chosen_id, timestamp=self._t)

    def _bootstrap_order(self, book: OrderBook,
                         rng: np.random.Generator) -> Order:
        self._t += rng.exponential(0.01)
        side = Side.BID if rng.random() < 0.5 else Side.ASK
        offset = rng.integers(1, 5)
        if side == Side.BID:
            price = self.initial_mid - offset
        else:
            price = self.initial_mid + offset
        return limit_order(price=price, size=self.lot_size,
                           timestamp=self._t, side=side)
