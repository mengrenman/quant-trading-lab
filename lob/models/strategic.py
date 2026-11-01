"""Simplified strategic placement model inspired by Parlour (1998).

In Parlour's model (Gould et al. 2013, Section 5.1), traders choose between
limit orders and market orders based on the current state of the LOB.
The key insight is the feedback loop: the book state determines order choice,
and order choice determines the book state.

This implementation captures the essential mechanism:
- When the spread is wide, limit orders are more attractive (better prices).
- When queue depth is large, market orders are more attractive (limit orders
  may not fill).
- Traders condition on both spread and queue depth.
"""

from typing import Optional

import numpy as np

from lob.book import OrderBook
from lob.order import Order, Side, limit_order, market_order, cancel_order


class ParlourModel:
    """Strategic order placement model inspired by Parlour (1998).

    Traders arrive at a constant rate and choose between:
    1. Submitting a market order (immediate execution)
    2. Submitting a limit order at or near the best price
    3. Cancelling an existing order

    The probability of choosing a market order increases with:
    - Small spread (limit orders don't offer much price improvement)
    - Large queue depth at best price (limit orders unlikely to fill)

    Parameters:
        arrival_rate: Total trader arrival rate.
        delta: Cancellation rate per active order.
        spread_threshold: Spread (in ticks) above which limit orders dominate.
        depth_threshold: Queue depth above which market orders become attractive.
        initial_mid: Starting mid-price.
        price_range: Max distance from mid for limit order placement.
        lot_size: Order size.
    """

    def __init__(
        self,
        arrival_rate: float = 5.0,
        delta: float = 0.05,
        spread_threshold: int = 5,
        depth_threshold: int = 10,
        initial_mid: int = 10000,
        price_range: int = 20,
        lot_size: int = 1,
    ):
        self.arrival_rate = arrival_rate
        self.delta = delta
        self.spread_threshold = spread_threshold
        self.depth_threshold = depth_threshold
        self.initial_mid = initial_mid
        self.price_range = price_range
        self.lot_size = lot_size
        self._t = 0.0

    def _prob_market_order(self, book: OrderBook) -> float:
        """Compute probability of a market order given book state."""
        spread = book.spread
        if spread is None:
            return 0.0  # No opposite side to trade against.

        # Spread effect: narrow spread -> more market orders.
        spread_factor = max(0, 1.0 - spread / self.spread_threshold)

        # Depth effect: deep queue -> more market orders (limit orders
        # queue behind too many orders).
        best_bid_depth = book.depth_at_price(book.best_bid, Side.BID) if book.best_bid else 0
        best_ask_depth = book.depth_at_price(book.best_ask, Side.ASK) if book.best_ask else 0
        avg_depth = (best_bid_depth + best_ask_depth) / 2
        depth_factor = min(1.0, avg_depth / self.depth_threshold)

        # Combined probability (logistic-like blend).
        p_market = 0.3 * spread_factor + 0.3 * depth_factor + 0.1
        return min(0.9, p_market)

    def generate_event(self, book: OrderBook, t: float,
                       rng: np.random.Generator) -> Optional[Order]:
        n_active = book.num_orders()
        cancel_rate = self.delta * n_active
        total_rate = self.arrival_rate + cancel_rate

        if total_rate <= 0:
            return self._bootstrap_order(book, rng)

        dt = rng.exponential(1.0 / total_rate)
        self._t += dt

        u = rng.random() * total_rate

        if u < cancel_rate:
            return self._generate_cancellation(book, rng)

        # Trader arrives: decide between market and limit order.
        side = Side.BID if rng.random() < 0.5 else Side.ASK
        p_market = self._prob_market_order(book)

        if rng.random() < p_market:
            return self._try_market_order(book, rng, side)
        else:
            return self._generate_limit_order(book, rng, side)

    def _try_market_order(self, book: OrderBook,
                          rng: np.random.Generator,
                          side: Side) -> Optional[Order]:
        if side == Side.BID and book.best_ask is None:
            return self._generate_limit_order(book, rng, side)
        if side == Side.ASK and book.best_bid is None:
            return self._generate_limit_order(book, rng, side)
        return market_order(size=self.lot_size, timestamp=self._t, side=side)

    def _generate_limit_order(self, book: OrderBook,
                              rng: np.random.Generator,
                              side: Side) -> Order:
        mid = book.mid_price
        if mid is not None:
            self._last_mid = mid
        else:
            mid = getattr(self, "_last_mid", float(self.initial_mid))

        best_b = book.best_bid
        best_a = book.best_ask
        spread = book.spread if book.spread is not None else self.spread_threshold

        # Geometric distribution for distance: most orders near best price.
        offset = int(min(rng.geometric(p=0.3), self.price_range))

        if side == Side.BID:
            # When spread is wider than threshold, higher probability of placing inside the spread
            p_inside = 0.6 if spread > self.spread_threshold else 0.3
            if best_b is not None and best_a is not None and (best_a - best_b > 1) and rng.random() < p_inside:
                price = int(rng.integers(best_b + 1, best_a))
            else:
                best = best_b if best_b is not None else (best_a - 1 if best_a is not None else int(mid) - 1)
                price = best - offset + 1
        else:
            p_inside = 0.6 if spread > self.spread_threshold else 0.3
            if best_b is not None and best_a is not None and (best_a - best_b > 1) and rng.random() < p_inside:
                price = int(rng.integers(best_b + 1, best_a))
            else:
                best = best_a if best_a is not None else (best_b + 1 if best_b is not None else int(mid) + 1)
                price = best + offset - 1

        price = max(1, price)
        return limit_order(price=price, size=self.lot_size,
                           timestamp=self._t, side=side)

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
        offset = int(rng.integers(1, 5))
        base_mid = int(getattr(self, "_last_mid", float(self.initial_mid)))
        if side == Side.BID:
            price = max(1, base_mid - offset)
        else:
            price = base_mid + offset
        return limit_order(price=price, size=self.lot_size,
                           timestamp=self._t, side=side)
