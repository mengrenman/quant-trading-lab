"""Order book implementation.

The OrderBook stores active orders partitioned into bid-side B(t) and
ask-side A(t) as defined in Gould et al. (2013) Section 2.

Uses SortedDict for each side, keyed by price. Within each price level,
orders are stored in a deque (FIFO) to support price-time priority.
"""

from collections import deque
from typing import Optional

from sortedcontainers import SortedDict

from lob.order import Order, Side, OrderType


class OrderBook:
    """A limit order book with configurable tick and lot sizes.

    Parameters:
        tick_size: Smallest permissible price interval (pi in the paper).
        lot_size: Smallest tradeable quantity (sigma in the paper).
    """

    def __init__(self, tick_size: int = 1, lot_size: int = 1):
        self.tick_size = tick_size
        self.lot_size = lot_size

        # Bids: SortedDict with negated keys so that iterating gives
        # descending price order (highest bid first).
        self._bids: SortedDict = SortedDict()
        # Asks: SortedDict with natural ordering (lowest ask first).
        self._asks: SortedDict = SortedDict()
        # Fast lookup by order_id.
        self._orders: dict[int, Order] = {}

        # O(1) depth and order count tracking.
        self._bid_depth_at: dict[int, int] = {}
        self._ask_depth_at: dict[int, int] = {}
        self._total_bid_depth: int = 0
        self._total_ask_depth: int = 0
        self._num_bid_orders: int = 0
        self._num_ask_orders: int = 0

        # Compact array + index map for O(1) random order sampling & removal.
        self._id_list: list[int] = []
        self._id_pos: dict[int, int] = {}

    # ---- Properties matching paper notation ----

    @property
    def best_bid(self) -> Optional[int]:
        """b(t): highest price among active buy orders."""
        if not self._bids:
            return None
        return -self._bids._list[0]  # negate back

    @property
    def best_ask(self) -> Optional[int]:
        """a(t): lowest price among active sell orders."""
        if not self._asks:
            return None
        return self._asks._list[0]

    @property
    def spread(self) -> Optional[int]:
        """s(t) = a(t) - b(t)."""
        if not self._bids or not self._asks:
            return None
        return self._asks._list[0] - (-self._bids._list[0])

    @property
    def mid_price(self) -> Optional[float]:
        """m(t) = (a(t) + b(t)) / 2."""
        if not self._bids or not self._asks:
            return None
        return (self._asks._list[0] + (-self._bids._list[0])) / 2.0

    # ---- Internal ID pool management ----

    def _register_order(self, order: Order) -> None:
        oid = order.order_id
        self._orders[oid] = order
        self._id_pos[oid] = len(self._id_list)
        self._id_list.append(oid)

    def _unregister_order_id(self, oid: int) -> Optional[Order]:
        order = self._orders.pop(oid, None)
        if order is None:
            return None
        pos = self._id_pos.pop(oid)
        last_oid = self._id_list.pop()
        if pos < len(self._id_list):
            self._id_list[pos] = last_oid
            self._id_pos[last_oid] = pos
        return order

    # ---- Core operations ----

    def add_order(self, order: Order) -> None:
        """Add an active limit order to the book.

        This method only inserts resting orders. Matching (for market orders
        or crossing limit orders) is handled by the matching engine.
        """
        if order.order_type == OrderType.CANCEL:
            self.cancel_order(order.order_id)
            return

        if self.tick_size != 1 and order.price % self.tick_size != 0:
            raise ValueError(
                f"Price {order.price} not a multiple of tick size {self.tick_size}"
            )
        if self.lot_size != 1 and order.size % self.lot_size != 0:
            raise ValueError(
                f"Size {order.size} not a multiple of lot size {self.lot_size}"
            )

        if order.side == Side.BID:
            key = -order.price
            queue = self._bids.get(key)
            if queue is None:
                queue = deque()
                self._bids[key] = queue
            queue.append(order)
            self._bid_depth_at[order.price] = self._bid_depth_at.get(order.price, 0) + order.size
            self._total_bid_depth += order.size
            self._num_bid_orders += 1
        else:
            key = order.price
            queue = self._asks.get(key)
            if queue is None:
                queue = deque()
                self._asks[key] = queue
            queue.append(order)
            self._ask_depth_at[order.price] = self._ask_depth_at.get(order.price, 0) + order.size
            self._total_ask_depth += order.size
            self._num_ask_orders += 1

        self._register_order(order)

    def cancel_order(self, order_id: int) -> Optional[Order]:
        """Remove an active order by its ID. Returns the removed order or None."""
        order = self._unregister_order_id(order_id)
        if order is None:
            return None

        if order.side == Side.BID:
            key = -order.price
            queue = self._bids.get(key)
            if queue is not None:
                try:
                    queue.remove(order)
                except ValueError:
                    pass
                if not queue:
                    del self._bids[key]
                    self._bid_depth_at.pop(order.price, None)
                else:
                    self._bid_depth_at[order.price] -= order.size
            self._total_bid_depth -= order.size
            self._num_bid_orders -= 1
        else:
            key = order.price
            queue = self._asks.get(key)
            if queue is not None:
                try:
                    queue.remove(order)
                except ValueError:
                    pass
                if not queue:
                    del self._asks[key]
                    self._ask_depth_at.pop(order.price, None)
                else:
                    self._ask_depth_at[order.price] -= order.size
            self._total_ask_depth -= order.size
            self._num_ask_orders -= 1

        return order

    def get_order(self, order_id: int) -> Optional[Order]:
        """Look up an active order by ID."""
        return self._orders.get(order_id)

    def _on_fill(self, maker: Order, fill_size: int) -> None:
        """Update cached depth and order counts when a resting order is filled."""
        maker.size -= fill_size
        if maker.side == Side.BID:
            self._total_bid_depth -= fill_size
            new_depth = self._bid_depth_at[maker.price] - fill_size
            if new_depth <= 0:
                self._bid_depth_at.pop(maker.price, None)
            else:
                self._bid_depth_at[maker.price] = new_depth
            if maker.size == 0:
                self._num_bid_orders -= 1
                self._unregister_order_id(maker.order_id)
        else:
            self._total_ask_depth -= fill_size
            new_depth = self._ask_depth_at[maker.price] - fill_size
            if new_depth <= 0:
                self._ask_depth_at.pop(maker.price, None)
            else:
                self._ask_depth_at[maker.price] = new_depth
            if maker.size == 0:
                self._num_ask_orders -= 1
                self._unregister_order_id(maker.order_id)

    def _remove_filled(self, order: Order) -> None:
        """Remove a fully filled order from the book (called by matching engine)."""
        self.cancel_order(order.order_id)

    # ---- Depth profiles (Section 2 of the paper) ----

    def depth_at_price(self, price: int, side: Side) -> int:
        """n^b(p,t) or n^a(p,t): total depth at a given price in O(1)."""
        if side == Side.BID:
            return self._bid_depth_at.get(price, 0)
        return self._ask_depth_at.get(price, 0)

    def depth_profile(self, side: Side, n_levels: int = 10) -> list[tuple[int, int]]:
        """Return (price, total_depth) pairs for the top n_levels.

        For bids: descending from best_bid.
        For asks: ascending from best_ask.
        """
        if side == Side.BID:
            keys = self._bids._list[:n_levels]
            depth_map = self._bid_depth_at
            return [(-k, depth_map[-k]) for k in keys]
        else:
            keys = self._asks._list[:n_levels]
            depth_map = self._ask_depth_at
            return [(k, depth_map[k]) for k in keys]

    def relative_depth_profile(self, side: Side, n_levels: int = 10) -> list[tuple[int, int]]:
        """Return (relative_price, total_depth) pairs.

        Relative price delta^b = b(t) - p for bids, delta^a = p - a(t) for asks.
        """
        ref = self.best_bid if side == Side.BID else self.best_ask
        if ref is None:
            return []
        profile = self.depth_profile(side, n_levels)
        if side == Side.BID:
            return [(ref - price, depth) for price, depth in profile]
        return [(price - ref, depth) for price, depth in profile]

    def total_depth(self, side: Side) -> int:
        """Total volume on one side of the book in O(1)."""
        return self._total_bid_depth if side == Side.BID else self._total_ask_depth

    def num_orders(self, side: Optional[Side] = None) -> int:
        """Number of active orders in O(1), optionally filtered by side."""
        if side is None:
            return len(self._orders)
        return self._num_bid_orders if side == Side.BID else self._num_ask_orders

    def all_order_ids(self) -> list[int]:
        """Return list of all active order IDs."""
        return list(self._id_list)

    def random_order_id(self, rng) -> Optional[int]:
        """Select a uniformly random active order ID in O(1)."""
        n = len(self._id_list)
        if n == 0:
            return None
        return self._id_list[int(rng.integers(0, n))]

    # ---- Display ----

    def snapshot(self) -> dict:
        """Return a serializable snapshot of the current book state."""
        return {
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mid_price": self.mid_price,
            "spread": self.spread,
            "bids": self.depth_profile(Side.BID, n_levels=50),
            "asks": self.depth_profile(Side.ASK, n_levels=50),
        }

    def __repr__(self) -> str:
        b = self.best_bid
        a = self.best_ask
        return (
            f"OrderBook(best_bid={b}, best_ask={a}, "
            f"spread={self.spread}, "
            f"bid_orders={self.num_orders(Side.BID)}, "
            f"ask_orders={self.num_orders(Side.ASK)})"
        )
