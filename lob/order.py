"""Order representation for the limit order book.

Following Gould et al. (2013), an order x = (p_x, omega_x, t_x) is a commitment
to trade up to |omega_x| units at price p_x, submitted at time t_x.

Prices and sizes are integers (in ticks and lots respectively) to mirror
the paper's discrete framework and avoid floating-point issues.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

_next_order_id = 0


def _gen_order_id() -> int:
    global _next_order_id
    _next_order_id += 1
    return _next_order_id


def reset_order_id_counter() -> None:
    """Reset the global order ID counter (useful for tests)."""
    global _next_order_id
    _next_order_id = 0


class Side(Enum):
    BID = "bid"
    ASK = "ask"


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"
    CANCEL = "cancel"


@dataclass(slots=True)
class Order:
    """A single order in the LOB.

    Attributes:
        price: Order price in ticks. For market orders, this is ignored
               (matched at best available).
        size: Order size in lots (always positive).
        timestamp: Submission time.
        side: BID (buy) or ASK (sell).
        order_type: LIMIT, MARKET, or CANCEL.
        order_id: Unique identifier, auto-generated if not provided.
    """
    price: int
    size: int
    timestamp: float
    side: Side
    order_type: OrderType = OrderType.LIMIT
    order_id: int = field(default_factory=_gen_order_id)

    def __post_init__(self):
        if self.order_type != OrderType.CANCEL and self.size <= 0:
            raise ValueError(f"Order size must be positive, got {self.size}")


@dataclass(slots=True)
class Fill:
    """A single fill resulting from a matching.

    Attributes:
        maker_order_id: ID of the resting (passive) order.
        taker_order_id: ID of the incoming (aggressive) order.
        price: Fill price (always the maker's price).
        size: Number of lots filled.
        timestamp: Time of the fill.
    """
    maker_order_id: int
    taker_order_id: int
    price: int
    size: int
    timestamp: float


def limit_order(price: int, size: int, timestamp: float, side: Side,
                order_id: Optional[int] = None) -> Order:
    """Create a limit order."""
    kw = {"order_id": order_id} if order_id is not None else {}
    return Order(
        price=price, size=size, timestamp=timestamp,
        side=side, order_type=OrderType.LIMIT,
        **kw,
    )


def market_order(size: int, timestamp: float, side: Side,
                 order_id: Optional[int] = None) -> Order:
    """Create a market order. Price is set to 0 (ignored during matching)."""
    kw = {"order_id": order_id} if order_id is not None else {}
    return Order(
        price=0, size=size, timestamp=timestamp,
        side=side, order_type=OrderType.MARKET,
        **kw,
    )


def cancel_order(order_id: int, timestamp: float) -> Order:
    """Create a cancellation event referencing an existing order."""
    return Order(
        price=0, size=0, timestamp=timestamp,
        side=Side.BID,  # side is irrelevant for cancellations
        order_type=OrderType.CANCEL,
        order_id=order_id,
    )
