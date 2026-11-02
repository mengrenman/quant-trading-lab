"""Simplified Mike and Farmer (2008) agent-based model.

This model incorporates three empirical features missing from the basic ZI
model (Gould et al. 2013, Section 5.2.5):

1. Power-law distributed relative limit order prices.
2. Long-memory (fractionally integrated) order signs, drawn as a continuous
   stream (see ``lob.models.long_memory.LongMemorySigns``).
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
from lob.models.long_memory import LongMemorySigns
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
        self._signs = LongMemorySigns(hurst)

    def _next_sign(self, rng: np.random.Generator) -> int:
        """Get the next order sign (+1 buy, -1 sell)."""
        return self._signs.next(rng)

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


class MikeFarmer2008Model:
    """Order flow with the structure of Mike and Farmer (2008).

    Closer to the original than ``MikeFarmerModel``:

    1. Order signs have long memory (``LongMemorySigns``).
    2. Each new order's price is set relative to the *same-side* best quote,
       x ticks less aggressive than it, with x drawn from a Student's t
       distribution (Gould et al. 2013, Section 5.2.5). Negative x places the
       order inside the spread; x at or below minus the spread makes it cross,
       so it executes as an effective market order. There is no separate
       market order process.
    3. Cancellation depends on the state of the book. A resting order is
       canceled at rate ``cancel_rate * (1 - exp(-y)) * (n_imb + B) / (1 + B)``,
       where y is its current distance to the opposite best quote divided by
       that distance when it was placed (orders the market has moved away
       from are canceled faster) and n_imb is the share of resting orders on
       its own side. Orders this model did not place count as y = 1.

    The structure follows Mike and Farmer's description; the parameter values
    are illustrative, not their calibration to LSE data.

    Parameters:
        lam: Order arrival rate (per unit time).
        cancel_rate: Maximum cancellation rate per resting order.
        imbalance_offset: B in the cancellation rate.
        hurst: Hurst exponent of the order signs.
        nu: Degrees of freedom of the Student's t relative price.
        loc, scale: Location and scale of the relative price, in ticks.
        initial_mid: Reference price while the book is empty.
        lot_size: Size of every order.
    """

    def __init__(
        self,
        lam: float = 1.0,
        cancel_rate: float = 0.01,
        imbalance_offset: float = 0.2,
        hurst: float = 0.75,
        nu: float = 1.3,
        loc: float = 1.0,
        scale: float = 3.0,
        initial_mid: int = 10000,
        lot_size: int = 1,
    ):
        self.lam = lam
        self.cancel_rate = cancel_rate
        self.imbalance_offset = imbalance_offset
        self.hurst = hurst
        self.nu = nu
        self.loc = loc
        self.scale = scale
        self.initial_mid = initial_mid
        self.lot_size = lot_size
        self._t = 0.0
        self._last_mid = float(initial_mid)
        self._signs = LongMemorySigns(hurst)
        # Distance to the opposite best quote when each order was placed.
        self._placement_distance: dict[int, int] = {}

    def generate_event(self, book: OrderBook, t: float,
                       rng: np.random.Generator) -> Optional[Order]:
        # Honor the caller's clock when it is ahead of ours.
        self._t = max(self._t, t)
        if book.mid_price is not None:
            self._last_mid = book.mid_price

        n_active = book.num_orders()
        candidate_cancel_rate = self.cancel_rate * n_active
        total_rate = self.lam + candidate_cancel_rate
        if total_rate <= 0:
            return None

        self._t += rng.exponential(1.0 / total_rate)
        if rng.random() * total_rate < self.lam:
            return self._place_order(book, rng)
        return self._maybe_cancel(book, rng)

    def _quote(self, book: OrderBook, side: Side) -> int:
        """Best quote on one side, or a stand-in next to the other side or the last mid."""
        if side == Side.BID:
            if book.best_bid is not None:
                return book.best_bid
            if book.best_ask is not None:
                return book.best_ask - 1
            return math.floor(self._last_mid) - 1
        if book.best_ask is not None:
            return book.best_ask
        if book.best_bid is not None:
            return book.best_bid + 1
        return math.floor(self._last_mid) + 1

    def _place_order(self, book: OrderBook, rng: np.random.Generator) -> Order:
        side = Side.BID if self._signs.next(rng) > 0 else Side.ASK
        offset = int(round(self.loc + self.scale * rng.standard_t(self.nu)))
        if side == Side.BID:
            price = max(1, self._quote(book, Side.BID) - offset)
            distance = self._quote(book, Side.ASK) - price
        else:
            price = self._quote(book, Side.ASK) + offset
            distance = price - self._quote(book, Side.BID)
        order = limit_order(price=price, size=self.lot_size, timestamp=self._t, side=side)
        # Orders leave the book without telling the model, so prune now and then.
        if len(self._placement_distance) > 2 * max(1_000, book.num_orders()):
            live = set(book.all_order_ids())
            self._placement_distance = {k: v for k, v in self._placement_distance.items() if k in live}
        self._placement_distance[order.order_id] = max(1, distance)
        return order

    def cancel_probability(self, book: OrderBook, order: Order) -> float:
        """Cancellation rate of a resting order as a fraction of ``cancel_rate``."""
        opposite = book.best_ask if order.side == Side.BID else book.best_bid
        placed = self._placement_distance.get(order.order_id)
        if opposite is None or placed is None:
            y = 1.0
        else:
            y = abs(opposite - order.price) / placed
        own_share = book.num_orders(order.side) / book.num_orders()
        b = self.imbalance_offset
        return (1.0 - math.exp(-y)) * (own_share + b) / (1.0 + b)

    def _maybe_cancel(self, book: OrderBook, rng: np.random.Generator) -> Optional[Order]:
        """Pick a resting order uniformly and cancel it with its state-dependent probability.

        Candidates arrive at the maximum rate ``cancel_rate`` per order and are
        kept with probability ``cancel_probability``: exact thinning of the
        state-dependent cancellation process.
        """
        order_id = book.random_order_id(rng)
        if order_id is None:
            return None
        if rng.random() >= self.cancel_probability(book, book.get_order(order_id)):
            return None
        return cancel_order(order_id=order_id, timestamp=self._t)
