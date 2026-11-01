"""Tests for matching.py — price-time and pro-rata engines."""

import pytest
from lob.order import (
    limit_order, market_order, cancel_order,
    Side, reset_order_id_counter,
)
from lob.book import OrderBook
from lob.matching import PriceTimeEngine, ProRataEngine


@pytest.fixture(autouse=True)
def reset_ids():
    reset_order_id_counter()
    yield
    reset_order_id_counter()


# ---- Price-Time Engine ----

class TestPriceTimeEngine:
    @pytest.fixture
    def engine(self):
        return PriceTimeEngine()

    @pytest.fixture
    def book(self):
        return OrderBook()

    def test_market_buy_matches_best_ask(self, book, engine):
        ask1 = limit_order(105, 3, 1.0, Side.ASK)
        ask2 = limit_order(106, 5, 2.0, Side.ASK)
        engine.process(book, ask1)
        engine.process(book, ask2)

        mo = market_order(2, 3.0, Side.BID)
        fills = engine.process(book, mo)
        assert len(fills) == 1
        assert fills[0].price == 105
        assert fills[0].size == 2
        # Remaining at 105: 1 lot
        assert book.depth_at_price(105, Side.ASK) == 1

    def test_market_sell_matches_best_bid(self, book, engine):
        bid1 = limit_order(100, 5, 1.0, Side.BID)
        bid2 = limit_order(99, 3, 2.0, Side.BID)
        engine.process(book, bid1)
        engine.process(book, bid2)

        mo = market_order(4, 3.0, Side.ASK)
        fills = engine.process(book, mo)
        assert len(fills) == 1
        assert fills[0].price == 100
        assert fills[0].size == 4
        assert book.depth_at_price(100, Side.BID) == 1

    def test_market_order_sweeps_multiple_levels(self, book, engine):
        engine.process(book, limit_order(105, 2, 1.0, Side.ASK))
        engine.process(book, limit_order(106, 3, 2.0, Side.ASK))
        engine.process(book, limit_order(107, 5, 3.0, Side.ASK))

        mo = market_order(4, 4.0, Side.BID)
        fills = engine.process(book, mo)
        assert len(fills) == 2
        assert fills[0].price == 105
        assert fills[0].size == 2
        assert fills[1].price == 106
        assert fills[1].size == 2
        assert book.best_ask == 106
        assert book.depth_at_price(106, Side.ASK) == 1

    def test_crossing_limit_order_matches(self, book, engine):
        engine.process(book, limit_order(100, 5, 1.0, Side.ASK))
        # Buy limit at 100 crosses the ask.
        lo = limit_order(100, 3, 2.0, Side.BID)
        fills = engine.process(book, lo)
        assert len(fills) == 1
        assert fills[0].price == 100
        assert fills[0].size == 3

    def test_limit_order_rests_if_no_cross(self, book, engine):
        engine.process(book, limit_order(105, 5, 1.0, Side.ASK))
        lo = limit_order(100, 3, 2.0, Side.BID)
        fills = engine.process(book, lo)
        assert fills == []
        assert book.best_bid == 100
        assert book.depth_at_price(100, Side.BID) == 3

    def test_partial_fill_limit_order_rests_residual(self, book, engine):
        engine.process(book, limit_order(100, 2, 1.0, Side.ASK))
        # Buy limit order for 5 at 100: fills 2, rests 3.
        lo = limit_order(100, 5, 2.0, Side.BID)
        fills = engine.process(book, lo)
        assert len(fills) == 1
        assert fills[0].size == 2
        assert lo.size == 5  # Incoming order object is not mutated in-place
        assert book.best_bid == 100
        assert book.depth_at_price(100, Side.BID) == 3
        assert book.total_depth(Side.BID) == 3
        assert book.total_depth(Side.ASK) == 0

    def test_fifo_within_price_level(self, book, engine):
        o1 = limit_order(100, 3, 1.0, Side.ASK)
        o2 = limit_order(100, 5, 2.0, Side.ASK)
        engine.process(book, o1)
        engine.process(book, o2)

        mo = market_order(4, 3.0, Side.BID)
        fills = engine.process(book, mo)
        # Should fill o1 (3) first, then 1 from o2.
        assert len(fills) == 2
        assert fills[0].maker_order_id == o1.order_id
        assert fills[0].size == 3
        assert fills[1].maker_order_id == o2.order_id
        assert fills[1].size == 1

    def test_cancel_via_engine(self, book, engine):
        o = limit_order(100, 5, 1.0, Side.BID)
        engine.process(book, o)
        assert book.num_orders() == 1
        engine.process(book, cancel_order(o.order_id, 2.0))
        assert book.num_orders() == 0

    def test_market_order_no_liquidity(self, book, engine):
        """Market order with no opposite-side liquidity: no fills, no resting."""
        mo = market_order(5, 1.0, Side.BID)
        fills = engine.process(book, mo)
        assert fills == []
        assert book.num_orders() == 0


# ---- Pro-Rata Engine ----

class TestProRataEngine:
    @pytest.fixture
    def engine(self):
        return ProRataEngine()

    @pytest.fixture
    def book(self):
        return OrderBook()

    def test_proportional_allocation(self, book, engine):
        # Two asks at same price: 60 lots and 40 lots.
        o1 = limit_order(100, 60, 1.0, Side.ASK)
        o2 = limit_order(100, 40, 2.0, Side.ASK)
        engine.process(book, o1)
        engine.process(book, o2)

        mo = market_order(50, 3.0, Side.BID)
        fills = engine.process(book, mo)

        total_filled = sum(f.size for f in fills)
        assert total_filled == 50

        # o1 should get ~60% = 30, o2 should get ~40% = 20
        fill_map = {f.maker_order_id: f.size for f in fills}
        assert fill_map[o1.order_id] == 30
        assert fill_map[o2.order_id] == 20

    def test_single_order_gets_all(self, book, engine):
        o = limit_order(100, 10, 1.0, Side.ASK)
        engine.process(book, o)
        mo = market_order(5, 2.0, Side.BID)
        fills = engine.process(book, mo)
        assert len(fills) == 1
        assert fills[0].size == 5
