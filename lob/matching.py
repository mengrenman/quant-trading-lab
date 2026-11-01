"""Matching engines for the limit order book.

Implements price-time priority (Section 3.4 of Gould et al. 2013) and
pro-rata matching. The matching engine is decoupled from the book so that
notebooks can swap engines and compare behaviour.
"""

from abc import ABC, abstractmethod

from lob.order import Order, Fill, Side, OrderType
from lob.book import OrderBook


class MatchingEngine(ABC):
    """Abstract base for matching engines."""

    @abstractmethod
    def process(self, book: OrderBook, incoming: Order) -> list[Fill]:
        """Process an incoming order against the book.

        Returns a list of fills. Any unfilled residual of a limit order is
        added to the book as a resting order. Market order residuals that
        cannot be filled are discarded.
        """
        ...


class PriceTimeEngine(MatchingEngine):
    """Standard price-time priority matching.

    Incoming orders match against the best price on the opposite side first.
    Within a price level, earlier orders have priority (FIFO).
    Fills always occur at the resting (maker) order's price.
    """

    def process(self, book: OrderBook, incoming: Order) -> list[Fill]:
        if incoming.order_type == OrderType.CANCEL:
            book.cancel_order(incoming.order_id)
            return []

        fills: list[Fill] = []
        remaining = incoming.size
        is_market = incoming.order_type == OrderType.MARKET

        if incoming.side == Side.BID:
            opposite = book._asks
            key_list = opposite._list
            while key_list and remaining > 0:
                key = key_list[0]
                maker_price = key
                if not is_market and incoming.price < maker_price:
                    break

                queue = opposite[key]
                while queue and remaining > 0:
                    maker = queue[0]
                    fill_size = remaining if remaining < maker.size else maker.size
                    fills.append(Fill(
                        maker_order_id=maker.order_id,
                        taker_order_id=incoming.order_id,
                        price=maker_price,
                        size=fill_size,
                        timestamp=incoming.timestamp,
                    ))
                    remaining -= fill_size
                    book._on_fill(maker, fill_size)
                    if maker.size == 0:
                        queue.popleft()

                if not queue:
                    del opposite[key]
        else:
            opposite = book._bids
            key_list = opposite._list
            while key_list and remaining > 0:
                key = key_list[0]
                maker_price = -key
                if not is_market and incoming.price > maker_price:
                    break

                queue = opposite[key]
                while queue and remaining > 0:
                    maker = queue[0]
                    fill_size = remaining if remaining < maker.size else maker.size
                    fills.append(Fill(
                        maker_order_id=maker.order_id,
                        taker_order_id=incoming.order_id,
                        price=maker_price,
                        size=fill_size,
                        timestamp=incoming.timestamp,
                    ))
                    remaining -= fill_size
                    book._on_fill(maker, fill_size)
                    if maker.size == 0:
                        queue.popleft()

                if not queue:
                    del opposite[key]

        # Any unfilled residual of a limit order rests in the book.
        if remaining > 0 and incoming.order_type == OrderType.LIMIT:
            if remaining != incoming.size:
                resting = Order(
                    price=incoming.price,
                    size=remaining,
                    timestamp=incoming.timestamp,
                    side=incoming.side,
                    order_type=OrderType.LIMIT,
                    order_id=incoming.order_id,
                )
            else:
                resting = incoming
            book.add_order(resting)

        return fills


class ProRataEngine(MatchingEngine):
    """Pro-rata matching at the best price level.

    At each price level, the incoming order's size is allocated proportionally
    to resting orders based on their size. Integer rounding uses floor with
    remainder allocated FIFO (hybrid pro-rata as used in practice).

    See Gould et al. (2013) Section 3.4 and Field and Large (2008).
    """

    def process(self, book: OrderBook, incoming: Order) -> list[Fill]:
        if incoming.order_type == OrderType.CANCEL:
            book.cancel_order(incoming.order_id)
            return []

        fills: list[Fill] = []
        remaining = incoming.size
        is_market = incoming.order_type == OrderType.MARKET
        is_bid = incoming.side == Side.BID
        opposite = book._asks if is_bid else book._bids
        key_list = opposite._list

        while key_list and remaining > 0:
            key = key_list[0]
            maker_price = key if is_bid else -key
            if not is_market:
                if is_bid and incoming.price < maker_price:
                    break
                if not is_bid and incoming.price > maker_price:
                    break

            queue = opposite[key]
            orders_at_level = list(queue)
            total_at_level = (
                book._ask_depth_at[maker_price]
                if is_bid
                else book._bid_depth_at[maker_price]
            )
            to_fill_at_level = min(remaining, total_at_level)

            # Allocate proportionally.
            allocations = []
            allocated_sum = 0
            for order in orders_at_level:
                share = (order.size * to_fill_at_level) // total_at_level
                allocations.append(share)
                allocated_sum += share

            # Distribute remainder FIFO.
            leftover = to_fill_at_level - allocated_sum
            for i, order in enumerate(orders_at_level):
                if leftover <= 0:
                    break
                can_add = order.size - allocations[i]
                add = min(leftover, can_add)
                allocations[i] += add
                leftover -= add

            # Execute fills and rebuild queue without O(N^2) queue.remove().
            queue.clear()
            for order, alloc in zip(orders_at_level, allocations):
                if alloc > 0:
                    fills.append(Fill(
                        maker_order_id=order.order_id,
                        taker_order_id=incoming.order_id,
                        price=maker_price,
                        size=alloc,
                        timestamp=incoming.timestamp,
                    ))
                    book._on_fill(order, alloc)
                    remaining -= alloc
                if order.size > 0:
                    queue.append(order)

            if not queue:
                del opposite[key]

        if remaining > 0 and incoming.order_type == OrderType.LIMIT:
            if remaining != incoming.size:
                resting = Order(
                    price=incoming.price,
                    size=remaining,
                    timestamp=incoming.timestamp,
                    side=incoming.side,
                    order_type=OrderType.LIMIT,
                    order_id=incoming.order_id,
                )
            else:
                resting = incoming
            book.add_order(resting)

        return fills
