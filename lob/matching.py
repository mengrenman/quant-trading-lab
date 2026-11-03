"""Matching engines for the limit order book.

Implements price-time priority (Section 3.4 of Gould et al. 2013) and
pro-rata matching. The matching engine is decoupled from the book so that
notebooks can swap engines and compare behavior.
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
        cannot be filled are discarded. Resting iceberg orders show only their
        peak: when it is used up, the next peak comes from the hidden reserve
        and joins the back of the queue at its price, so an incoming order can
        fill more than the visible depth.
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

        # Reject invalid orders before any fill changes the book.
        book.validate(incoming)
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
                    refilled = book._on_fill(maker, fill_size)
                    if maker.size == 0:
                        queue.popleft()
                    elif refilled:  # a new iceberg peak joins the back of the queue
                        queue.popleft()
                        queue.append(maker)

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
                    refilled = book._on_fill(maker, fill_size)
                    if maker.size == 0:
                        queue.popleft()
                    elif refilled:  # a new iceberg peak joins the back of the queue
                        queue.popleft()
                        queue.append(maker)

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
                    display_size=incoming.display_size,
                )
            else:
                resting = incoming
            book.add_order(resting)

        return fills


class ProRataEngine(MatchingEngine):
    """Pro-rata matching at the best price level.

    At each price level, the incoming order's size is allocated proportionally
    to resting orders based on their size. Shares are rounded down to whole
    lots and the leftover lots go to orders in time priority (hybrid pro-rata
    as used in practice), so fills and remainders stay multiples of the lot
    size.

    See Gould et al. (2013) Section 3.4 and Field and Large (2008).
    """

    def process(self, book: OrderBook, incoming: Order) -> list[Fill]:
        if incoming.order_type == OrderType.CANCEL:
            book.cancel_order(incoming.order_id)
            return []

        # Reject invalid orders before any fill changes the book.
        book.validate(incoming)
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

            # Allocate proportionally, in lots: validation guarantees every size
            # here is a multiple of the lot size.
            lot = book.lot_size
            total_lots = total_at_level // lot
            fill_lots = to_fill_at_level // lot
            allocations = []
            allocated_sum = 0
            for order in orders_at_level:
                share = (order.size // lot) * fill_lots // total_lots
                allocations.append(share)
                allocated_sum += share

            # Distribute leftover lots FIFO.
            leftover = fill_lots - allocated_sum
            for i, order in enumerate(orders_at_level):
                if leftover <= 0:
                    break
                add = min(leftover, order.size // lot - allocations[i])
                allocations[i] += add
                leftover -= add
            allocations = [a * lot for a in allocations]

            # Execute fills and rebuild queue without O(N^2) queue.remove(). Icebergs
            # whose peak was used up show a new one at the back of the queue; if the
            # incoming order is not yet filled, the level is allocated again.
            queue.clear()
            refilled_orders = []
            for order, alloc in zip(orders_at_level, allocations, strict=True):
                if alloc > 0:
                    fills.append(Fill(
                        maker_order_id=order.order_id,
                        taker_order_id=incoming.order_id,
                        price=maker_price,
                        size=alloc,
                        timestamp=incoming.timestamp,
                    ))
                    remaining -= alloc
                    if book._on_fill(order, alloc):
                        refilled_orders.append(order)
                        continue
                if order.size > 0:
                    queue.append(order)
            queue.extend(refilled_orders)

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
                    display_size=incoming.display_size,
                )
            else:
                resting = incoming
            book.add_order(resting)

        return fills
