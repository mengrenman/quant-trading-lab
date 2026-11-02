"""Tests for book.py."""

import pytest
from lob.order import limit_order, Side, reset_order_id_counter
from lob.book import OrderBook


@pytest.fixture(autouse=True)
def reset_ids():
    reset_order_id_counter()
    yield
    reset_order_id_counter()


@pytest.fixture
def book():
    return OrderBook(tick_size=1, lot_size=1)


def test_empty_book(book):
    assert book.best_bid is None
    assert book.best_ask is None
    assert book.spread is None
    assert book.mid_price is None


def test_add_bid(book):
    book.add_order(limit_order(100, 5, 1.0, Side.BID))
    assert book.best_bid == 100
    assert book.best_ask is None
    assert book.num_orders(Side.BID) == 1


def test_add_ask(book):
    book.add_order(limit_order(105, 3, 1.0, Side.ASK))
    assert book.best_ask == 105
    assert book.best_bid is None


def test_spread_and_mid(book):
    book.add_order(limit_order(100, 5, 1.0, Side.BID))
    book.add_order(limit_order(104, 3, 2.0, Side.ASK))
    assert book.spread == 4
    assert book.mid_price == 102.0


def test_multiple_bids_best_is_highest(book):
    book.add_order(limit_order(98, 1, 1.0, Side.BID))
    book.add_order(limit_order(100, 1, 2.0, Side.BID))
    book.add_order(limit_order(99, 1, 3.0, Side.BID))
    assert book.best_bid == 100


def test_multiple_asks_best_is_lowest(book):
    book.add_order(limit_order(107, 1, 1.0, Side.ASK))
    book.add_order(limit_order(105, 1, 2.0, Side.ASK))
    book.add_order(limit_order(106, 1, 3.0, Side.ASK))
    assert book.best_ask == 105


def test_cancel_order(book):
    o = limit_order(100, 5, 1.0, Side.BID)
    book.add_order(o)
    assert book.num_orders() == 1
    removed = book.cancel_order(o.order_id)
    assert removed is not None
    assert book.num_orders() == 0
    assert book.best_bid is None


def test_cancel_nonexistent(book):
    assert book.cancel_order(9999) is None


def test_depth_at_price(book):
    book.add_order(limit_order(100, 5, 1.0, Side.BID))
    book.add_order(limit_order(100, 3, 2.0, Side.BID))
    assert book.depth_at_price(100, Side.BID) == 8
    assert book.depth_at_price(99, Side.BID) == 0


def test_depth_profile(book):
    book.add_order(limit_order(100, 5, 1.0, Side.BID))
    book.add_order(limit_order(99, 3, 2.0, Side.BID))
    book.add_order(limit_order(98, 1, 3.0, Side.BID))
    profile = book.depth_profile(Side.BID, n_levels=3)
    assert profile == [(100, 5), (99, 3), (98, 1)]


def test_relative_depth_profile(book):
    book.add_order(limit_order(100, 5, 1.0, Side.BID))
    book.add_order(limit_order(98, 3, 2.0, Side.BID))
    profile = book.relative_depth_profile(Side.BID, n_levels=2)
    # relative price = best_bid - price: 0 for 100, 2 for 98
    assert profile == [(0, 5), (2, 3)]


def test_tick_size_validation(book):
    book_big_tick = OrderBook(tick_size=5, lot_size=1)
    with pytest.raises(ValueError):
        book_big_tick.add_order(limit_order(101, 1, 1.0, Side.BID))
    # This should work fine:
    book_big_tick.add_order(limit_order(100, 1, 1.0, Side.BID))


def test_lot_size_validation():
    book = OrderBook(tick_size=1, lot_size=10)
    with pytest.raises(ValueError):
        book.add_order(limit_order(100, 3, 1.0, Side.BID))
    book.add_order(limit_order(100, 10, 1.0, Side.BID))
    assert book.depth_at_price(100, Side.BID) == 10


def test_snapshot(book):
    book.add_order(limit_order(100, 5, 1.0, Side.BID))
    book.add_order(limit_order(105, 3, 2.0, Side.ASK))
    snap = book.snapshot()
    assert snap["best_bid"] == 100
    assert snap["best_ask"] == 105
    assert snap["spread"] == 5


def test_total_depth_and_num_orders_consistency(book):
    o1 = limit_order(100, 5, 1.0, Side.BID)
    o2 = limit_order(99, 3, 2.0, Side.BID)
    o3 = limit_order(105, 4, 3.0, Side.ASK)
    book.add_order(o1)
    book.add_order(o2)
    book.add_order(o3)

    assert book.total_depth(Side.BID) == 8
    assert book.total_depth(Side.ASK) == 4
    assert book.num_orders(Side.BID) == 2
    assert book.num_orders(Side.ASK) == 1
    assert book.num_orders() == 3

    book.cancel_order(o1.order_id)
    assert book.total_depth(Side.BID) == 3
    assert book.depth_at_price(100, Side.BID) == 0
    assert book.num_orders(Side.BID) == 1
    assert book.num_orders() == 2


def test_random_order_id(book):
    import numpy as np
    rng = np.random.default_rng(42)
    assert book.random_order_id(rng) is None

    o1 = limit_order(100, 5, 1.0, Side.BID)
    o2 = limit_order(105, 3, 2.0, Side.ASK)
    book.add_order(o1)
    book.add_order(o2)

    sampled = {book.random_order_id(rng) for _ in range(20)}
    assert sampled == {o1.order_id, o2.order_id}

    book.cancel_order(o1.order_id)
    assert book.random_order_id(rng) == o2.order_id



# ---- Validation happens before the book changes ----

def test_crossing_off_tick_order_is_rejected_before_matching():
    from lob.matching import PriceTimeEngine, ProRataEngine

    for engine in (PriceTimeEngine(), ProRataEngine()):
        book = OrderBook(tick_size=5)
        book.add_order(limit_order(100, 5, 0.0, Side.ASK))
        with pytest.raises(ValueError, match="tick size"):
            engine.process(book, limit_order(103, 10, 1.0, Side.BID))
        assert book.depth_at_price(100, Side.ASK) == 5  # untouched


def test_market_order_off_lot_is_rejected_before_matching():
    from lob.matching import PriceTimeEngine
    from lob.order import market_order

    book = OrderBook(lot_size=10)
    book.add_order(limit_order(100, 20, 0.0, Side.ASK))
    with pytest.raises(ValueError, match="lot size"):
        PriceTimeEngine().process(book, market_order(15, 1.0, Side.BID))
    assert book.depth_at_price(100, Side.ASK) == 20


def test_duplicate_order_id_is_rejected():
    from lob.matching import PriceTimeEngine

    book = OrderBook()
    book.add_order(limit_order(100, 1, 0.0, Side.BID, order_id=7))
    with pytest.raises(ValueError, match="already in the book"):
        book.add_order(limit_order(99, 1, 0.0, Side.BID, order_id=7))
    with pytest.raises(ValueError, match="already in the book"):
        PriceTimeEngine().process(book, limit_order(98, 1, 0.0, Side.BID, order_id=7))
    assert book.num_orders() == 1 and book.all_order_ids() == [7]
    assert book.cancel_order(7) is not None
    assert book.num_orders() == 0 and book.best_bid is None
