"""Tests for order.py."""

import pytest
from lob.order import (
    Side, OrderType, Fill,
    limit_order, market_order, cancel_order,
    reset_order_id_counter,
)


@pytest.fixture(autouse=True)
def reset_ids():
    reset_order_id_counter()
    yield
    reset_order_id_counter()


def test_limit_order_creation():
    o = limit_order(price=100, size=5, timestamp=1.0, side=Side.BID)
    assert o.price == 100
    assert o.size == 5
    assert o.side == Side.BID
    assert o.order_type == OrderType.LIMIT
    assert o.timestamp == 1.0


def test_market_order_creation():
    o = market_order(size=3, timestamp=2.0, side=Side.ASK)
    assert o.price == 0
    assert o.size == 3
    assert o.side == Side.ASK
    assert o.order_type == OrderType.MARKET


def test_cancel_order_creation():
    o = cancel_order(order_id=42, timestamp=3.0)
    assert o.order_id == 42
    assert o.order_type == OrderType.CANCEL


def test_invalid_size_raises():
    with pytest.raises(ValueError):
        limit_order(price=100, size=0, timestamp=1.0, side=Side.BID)
    with pytest.raises(ValueError):
        limit_order(price=100, size=-1, timestamp=1.0, side=Side.BID)


def test_order_ids_are_unique():
    o1 = limit_order(price=100, size=1, timestamp=1.0, side=Side.BID)
    o2 = limit_order(price=100, size=1, timestamp=2.0, side=Side.BID)
    assert o1.order_id != o2.order_id


def test_fill_dataclass():
    f = Fill(maker_order_id=1, taker_order_id=2, price=100, size=5, timestamp=1.0)
    assert f.price == 100
    assert f.size == 5
