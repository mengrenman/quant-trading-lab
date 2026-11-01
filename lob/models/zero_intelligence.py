"""Zero-intelligence model of Daniels et al. (2003) and Smith et al. (2003).

This is the foundational continuous-time ZI model discussed in Gould et al.
(2013) Section 5.2.4. All order flow is governed by independent Poisson
processes:

1. Limit orders arrive at rate lambda per tick per unit time on each side. In
   the paper's model an incoming order is equally likely to land at any
   relative price in (-s(t), inf), i.e. a buy can land at any price below the
   best ask. We truncate this to the L ticks nearest the opposite quote: buys
   are uniform on {a(t) - L, ..., a(t) - 1} and sells on
   {b(t) + 1, ..., b(t) + L}. Orders therefore land inside the spread with
   probability proportional to its width, which is how the spread emerges
   from the order flow.
2. Market orders arrive at total rate mu per unit time, half buys and half
   sells.
3. Each resting limit order is independently canceled at rate delta per unit
   time (memoryless).

The model is deliberately simple — it captures the mechanical aspects of
LOB dynamics without any strategic behavior, serving as a null model
against which richer models can be compared.
"""

from typing import Optional

import numpy as np

from lob.book import OrderBook
from lob.order import Order, Side, limit_order, market_order, cancel_order


class SmithFarmerModel:
    """Smith et al. (2003) zero-intelligence model.

    Parameters:
        mu: Total market order arrival rate (orders per unit time), split
            evenly between buys and sells.
        lam: Limit order arrival rate per tick per unit time, on each side.
        delta: Cancellation rate per resting order per unit time.
        price_range: Width L of the placement window in ticks, measured from
            the opposite best quote.
        initial_mid: Reference quote used while a side of the book has never
            been populated.
        lot_size: Size of each order in lots.

    Subclasses can override ``_sample_side`` and ``_sample_size`` to change
    how order signs and sizes are drawn; the price is always chosen after
    the side, so it stays consistent with it.
    """

    def __init__(
        self,
        mu: float = 1.0,
        lam: float = 0.5,
        delta: float = 0.1,
        price_range: int = 50,
        initial_mid: int = 10000,
        lot_size: int = 1,
    ):
        self.mu = mu
        self.lam = lam
        self.delta = delta
        self.price_range = price_range
        self.initial_mid = initial_mid
        self.lot_size = lot_size
        self._t = 0.0
        # Last observed best quotes: the placement reference when a side is empty.
        self._last_bid = initial_mid
        self._last_ask = initial_mid

    def _total_rate(self, n_active: int) -> float:
        """Total event rate given current number of active orders."""
        return 2 * self.lam * self.price_range + self.mu + self.delta * n_active

    def generate_event(self, book: OrderBook, t: float,
                       rng: np.random.Generator) -> Optional[Order]:
        """Generate the next event using Gillespie-style sampling.

        The caller's clock ``t`` is honored when it is ahead of the model's
        own clock, e.g. when this model takes over an existing book.
        """
        self._t = max(self._t, t)
        if book.best_bid is not None:
            self._last_bid = book.best_bid
        if book.best_ask is not None:
            self._last_ask = book.best_ask

        limit_rate = 2 * self.lam * self.price_range
        market_rate = self.mu
        total_rate = self._total_rate(book.num_orders())
        if total_rate <= 0:
            return None

        # Sample inter-arrival time (exponential), then the event type.
        self._t += rng.exponential(1.0 / total_rate)
        u = rng.random() * total_rate

        if u < limit_rate:
            return self._generate_limit_order(book, rng)
        elif u < limit_rate + market_rate:
            return self._generate_market_order(book, rng)
        else:
            return self._generate_cancellation(book, rng)

    def _sample_side(self, rng: np.random.Generator) -> Side:
        return Side.BID if rng.random() < 0.5 else Side.ASK

    def _sample_size(self, rng: np.random.Generator) -> int:
        return self.lot_size

    def _generate_limit_order(self, book: OrderBook,
                              rng: np.random.Generator) -> Order:
        """Place a limit order uniformly on the L ticks next to the opposite quote."""
        side = self._sample_side(rng)
        offset = int(rng.integers(1, self.price_range + 1))
        if side == Side.BID:
            price = max(1, self._last_ask - offset)
        else:
            price = self._last_bid + offset
        return limit_order(price=price, size=self._sample_size(rng),
                           timestamp=self._t, side=side)

    def _generate_market_order(self, book: OrderBook,
                               rng: np.random.Generator) -> Optional[Order]:
        """Submit a market order to one side."""
        side = self._sample_side(rng)
        size = self._sample_size(rng)

        # Only submit if there's liquidity on the opposite side.
        if side == Side.BID and book.best_ask is None:
            return None
        if side == Side.ASK and book.best_bid is None:
            return None

        return market_order(size=size, timestamp=self._t, side=side)

    def _generate_cancellation(self, book: OrderBook,
                               rng: np.random.Generator) -> Optional[Order]:
        """Cancel a uniformly random resting order."""
        chosen_id = book.random_order_id(rng)
        if chosen_id is None:
            return None
        return cancel_order(order_id=chosen_id, timestamp=self._t)
