from __future__ import annotations

"""Mike and Farmer (2008) agent-based model.

This model incorporates three key empirical features missing from the
basic ZI model (Gould et al. 2013, Section 5.2.5):

1. Power-law distributed relative limit order prices.
2. Long-memory (fractionally integrated) order signs.
3. Heavy-tailed order sizes.

Together these produce several stylized facts simultaneously: heavy-tailed
returns, volatility clustering, and realistic depth profiles.
"""

from typing import Optional
import numpy as np
from lob.book import OrderBook
from lob.order import Order, Side, limit_order, market_order, cancel_order


def _generate_fractional_noise(n: int, H: float,
                               rng: np.random.Generator) -> np.ndarray:
    """Generate fractional Gaussian noise via the Davies-Harte (FFT) method.

    Args:
        n: Length of the series.
        H: Hurst exponent (0.5 = white noise, >0.5 = long memory).
        rng: Numpy random generator.

    Returns:
        Array of length n with the desired correlation structure.
    """
    # Autocovariance of fGn.
    m = 2 * n
    k = np.arange(0, m)
    gamma = 0.5 * (np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) +
                    np.abs(k + 1) ** (2 * H))

    # Circulant embedding.
    eigenvalues = np.fft.fft(gamma).real
    # Clamp small negatives from numerical noise.
    eigenvalues = np.maximum(eigenvalues, 0)

    sqrt_eig = np.sqrt(eigenvalues / m)
    z = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    w = np.fft.ifft(sqrt_eig * z).real
    return w[:n]


class MikeFarmerModel:
    """Mike and Farmer (2008) agent-based LOB model.

    Parameters:
        mu: Market order arrival rate.
        lam: Limit order arrival rate.
        delta: Cancellation rate per active order.
        alpha_price: Power-law exponent for relative limit order prices.
        hurst: Hurst exponent for order sign long memory (H > 0.5).
        size_mean: Mean order size (log-normal).
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
        self._signs: np.ndarray | None = None
        self._sign_idx = 0

    def _ensure_signs(self, rng: np.random.Generator, n: int = 50000) -> None:
        """Pre-generate a long-memory sign series."""
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
            mid = self.initial_mid

        if side == Side.BID:
            price = max(1, int(mid) - delta)
        else:
            price = int(mid) + delta

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
