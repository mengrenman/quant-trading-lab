"""Zero-intelligence model based on Smith et al. (2003).

This is the foundational continuous-time ZI model discussed in Gould et al.
(2013) Section 5.2.4. The assumptions are:

1. Limit orders arrive as a Poisson process at rate lambda per unit time
   per tick, uniformly distributed across a price range on each side.
2. Market orders arrive as a Poisson process at rate mu per unit time.
3. Each active limit order is independently cancelled at rate delta per
   unit time (memoryless).

The model is deliberately simple — it captures the mechanical aspects of
LOB dynamics without any strategic behaviour, serving as a null model
against which richer models can be compared.
"""

from typing import Optional

import numpy as np

from lob.book import OrderBook
from lob.order import Order, Side, OrderType, limit_order, market_order, cancel_order


class SmithFarmerModel:
    """Smith et al. (2003) zero-intelligence model.

    Parameters:
        mu: Market order arrival rate (orders per unit time).
        lam: Limit order arrival rate per tick per unit time.
        delta: Cancellation rate per order per unit time.
        price_range: Number of ticks on each side of the mid-price where
            limit orders can be placed.
        initial_mid: Initial mid-price (in ticks) used to seed the book.
        lot_size: Size of each order in lots.
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

    def _total_rate(self, n_active: int) -> float:
        """Total event rate given current number of active orders."""
        # Limit orders: lam * price_range ticks * 2 sides
        limit_rate = self.lam * self.price_range * 2
        market_rate = self.mu
        cancel_rate = self.delta * n_active
        return limit_rate + market_rate + cancel_rate

    def generate_event(self, book: OrderBook, t: float,
                       rng: np.random.Generator) -> Optional[Order]:
        """Generate the next event using Gillespie-style sampling."""
        n_active = book.num_orders()

        # Compute rates.
        limit_rate = self.lam * self.price_range * 2
        market_rate = self.mu
        cancel_rate = self.delta * n_active
        total_rate = limit_rate + market_rate + cancel_rate

        if total_rate <= 0:
            # Bootstrap: place initial limit orders.
            return self._bootstrap_order(book, rng)

        # Sample inter-arrival time (exponential).
        dt = rng.exponential(1.0 / total_rate)
        self._t += dt

        # Sample event type.
        u = rng.random() * total_rate

        if u < limit_rate:
            return self._generate_limit_order(book, rng)
        elif u < limit_rate + market_rate:
            return self._generate_market_order(book, rng)
        else:
            return self._generate_cancellation(book, rng)

    def _generate_limit_order(self, book: OrderBook,
                              rng: np.random.Generator) -> Order:
        """Place a limit order uniformly within price_range of the mid or inside the spread."""
        side = Side.BID if rng.random() < 0.5 else Side.ASK
        mid = book.mid_price
        if mid is not None:
            self._last_mid = mid
        else:
            mid = getattr(self, "_last_mid", float(self.initial_mid))

        best_b = book.best_bid
        best_a = book.best_ask

        # Offset from best price (1 to price_range ticks into the book).
        offset = int(rng.integers(1, self.price_range + 1))

        if side == Side.BID:
            # If spread is wide (> 1 tick), traders can improve the bid by placing inside the spread
            if best_b is not None and best_a is not None and (best_a - best_b > 1) and rng.random() < 0.4:
                price = int(rng.integers(best_b + 1, best_a))
            else:
                best = best_b if best_b is not None else (best_a - 1 if best_a is not None else int(mid))
                price = best - offset + 1
        else:
            # If spread is wide (> 1 tick), traders can improve the ask by placing inside the spread
            if best_b is not None and best_a is not None and (best_a - best_b > 1) and rng.random() < 0.4:
                price = int(rng.integers(best_b + 1, best_a))
            else:
                best = best_a if best_a is not None else (best_b + 1 if best_b is not None else int(mid))
                price = best + offset - 1

        price = max(1, price)  # keep positive
        return limit_order(price=price, size=self.lot_size,
                           timestamp=self._t, side=side)

    def _generate_market_order(self, book: OrderBook,
                               rng: np.random.Generator) -> Optional[Order]:
        """Submit a market order to one side."""
        side = Side.BID if rng.random() < 0.5 else Side.ASK

        # Only submit if there's liquidity on the opposite side.
        if side == Side.BID and book.best_ask is None:
            return None
        if side == Side.ASK and book.best_bid is None:
            return None

        return market_order(size=self.lot_size, timestamp=self._t, side=side)

    def _generate_cancellation(self, book: OrderBook,
                               rng: np.random.Generator) -> Optional[Order]:
        """Cancel a random active order."""
        chosen_id = book.random_order_id(rng)
        if chosen_id is None:
            return None
        return cancel_order(order_id=chosen_id, timestamp=self._t)

    def _bootstrap_order(self, book: OrderBook,
                         rng: np.random.Generator) -> Order:
        """Seed the book with an initial order when empty."""
        self._t += rng.exponential(0.01)
        side = Side.BID if rng.random() < 0.5 else Side.ASK
        offset = int(rng.integers(1, 5))
        base_mid = int(getattr(self, "_last_mid", float(self.initial_mid)))
        if side == Side.BID:
            price = max(1, base_mid - offset)
        else:
            price = base_mid + offset
        return limit_order(price=price, size=self.lot_size,
                           timestamp=self._t, side=side)
