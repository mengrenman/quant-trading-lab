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


# ---- Randomized invariants (both engines) ----

def _assert_book_consistent(book):
    """Cached depths, counts and the order-ID index must match the queues."""
    for levels, depth_cache, side in ((book._bids, book._bid_depth_at, Side.BID),
                                      (book._asks, book._ask_depth_at, Side.ASK)):
        recomputed, n_orders, total = {}, 0, 0
        for key, queue in levels.items():
            price = -key if side == Side.BID else key
            assert queue, "empty price level left in book"
            for o in queue:
                assert o.size > 0 and o.price == price and o.side == side
                assert book._orders.get(o.order_id) is o
            recomputed[price] = sum(o.size for o in queue)
            n_orders += len(queue)
            total += recomputed[price]
        assert recomputed == depth_cache
        assert n_orders == book.num_orders(side)
        assert total == book.total_depth(side)
    assert len(book._orders) == len(book._id_list) == len(book._id_pos)
    assert all(book._id_pos[oid] == i for i, oid in enumerate(book._id_list))
    if book.best_bid is not None and book.best_ask is not None:
        assert book.best_bid < book.best_ask, "crossed book"


@pytest.mark.parametrize("lot", [1, 5])
@pytest.mark.parametrize("engine_cls", [PriceTimeEngine, ProRataEngine])
def test_random_event_stream_preserves_book_invariants(engine_cls, lot):
    import numpy as np

    rng = np.random.default_rng(11)
    book, engine = OrderBook(lot_size=lot), engine_cls()
    for step in range(5_000):
        r = rng.random()
        if r < 0.55:
            # Limit orders within 12 ticks of the mid, so many of them cross.
            mid = book.mid_price or 1000
            price = max(1, int(mid + rng.integers(-12, 13)))
            size = int(rng.integers(1, 15)) * lot
            side = Side.BID if rng.random() < 0.5 else Side.ASK
            order = limit_order(price, size, float(step), side)
            fills = engine.process(book, order)
            filled = sum(f.size for f in fills)
            assert filled <= size
            resting = book.get_order(order.order_id)
            assert (resting.size if resting else 0) == size - filled
        elif r < 0.75:
            size = int(rng.integers(1, 30)) * lot
            side = Side.BID if rng.random() < 0.5 else Side.ASK
            fills = engine.process(book, market_order(size, float(step), side))
            assert sum(f.size for f in fills) <= size
        else:
            oid = book.random_order_id(rng) if rng.random() < 0.9 else 10**9
            fills = []
            if oid is not None:
                engine.process(book, cancel_order(oid, float(step)))
        assert all(f.size % lot == 0 for f in fills)
        if step % 50 == 0:
            _assert_book_consistent(book)
            assert all(book.get_order(oid).size % lot == 0 for oid in book.all_order_ids())
    _assert_book_consistent(book)


def test_pro_rata_allocates_whole_lots():
    book, engine = OrderBook(lot_size=10), ProRataEngine()
    first = limit_order(100, 30, 1.0, Side.ASK)
    second = limit_order(100, 70, 2.0, Side.ASK)
    engine.process(book, first)
    engine.process(book, second)
    fills = engine.process(book, market_order(50, 3.0, Side.BID))
    # 3 and 7 lots share 5: floor shares 1 and 3, the leftover lot goes to the earlier order.
    assert {f.maker_order_id: f.size for f in fills} == {first.order_id: 20, second.order_id: 30}
    assert [o.size for o in book.orders_at(100, Side.ASK)] == [10, 40]


# ---- Iceberg orders ----

def test_market_order_fills_beyond_an_icebergs_peak():
    book, engine = OrderBook(), PriceTimeEngine()
    iceberg = limit_order(101, 100, 0.0, Side.ASK, display_size=10)
    engine.process(book, iceberg)
    fills = engine.process(book, market_order(35, 1.0, Side.BID))
    assert [f.size for f in fills] == [10, 10, 10, 5]  # each peak, refilled from the reserve
    assert {f.price for f in fills} == {101}
    assert (iceberg.size, iceberg.hidden) == (5, 60)  # 65 left: 5 showing, 60 hidden
    assert book.depth_at_price(101, Side.ASK) == 5


def test_a_new_iceberg_peak_loses_time_priority():
    book, engine = OrderBook(), PriceTimeEngine()
    iceberg = limit_order(101, 30, 0.0, Side.ASK, display_size=10)
    plain = limit_order(101, 20, 1.0, Side.ASK)
    engine.process(book, iceberg)
    engine.process(book, plain)
    fills = engine.process(book, market_order(15, 2.0, Side.BID))
    assert [(f.maker_order_id, f.size) for f in fills] == [(iceberg.order_id, 10), (plain.order_id, 5)]
    assert [o.order_id for o in book.orders_at(101, Side.ASK)] == [plain.order_id, iceberg.order_id]
    fills = engine.process(book, market_order(20, 3.0, Side.BID))
    assert [(f.maker_order_id, f.size) for f in fills] == [(plain.order_id, 15), (iceberg.order_id, 5)]


@pytest.mark.parametrize("engine_cls", [PriceTimeEngine, ProRataEngine])
def test_an_incoming_iceberg_trades_its_full_size_and_rests_as_an_iceberg(engine_cls):
    book, engine = OrderBook(), engine_cls()
    engine.process(book, limit_order(101, 20, 0.0, Side.ASK))
    buy = limit_order(101, 100, 1.0, Side.BID, display_size=10)
    fills = engine.process(book, buy)
    assert sum(f.size for f in fills) == 20
    resting = book.get_order(buy.order_id)
    assert (resting.size, resting.hidden) == (10, 70)
    assert book.depth_at_price(101, Side.BID) == 10


def test_pro_rata_allocates_again_after_icebergs_refill():
    book, engine = OrderBook(), ProRataEngine()
    iceberg = limit_order(101, 50, 0.0, Side.ASK, display_size=10)
    plain = limit_order(101, 10, 1.0, Side.ASK)
    engine.process(book, iceberg)
    engine.process(book, plain)
    fills = engine.process(book, market_order(40, 2.0, Side.BID))
    by_maker = {}
    for f in fills:
        by_maker[f.maker_order_id] = by_maker.get(f.maker_order_id, 0) + f.size
    assert by_maker == {iceberg.order_id: 30, plain.order_id: 10}
    assert (iceberg.size, iceberg.hidden) == (10, 10)
    _assert_book_consistent(book)


@pytest.mark.parametrize("lot", [1, 5])
@pytest.mark.parametrize("engine_cls", [PriceTimeEngine, ProRataEngine])
def test_random_event_stream_with_icebergs_conserves_quantity(engine_cls, lot):
    """Book invariants hold, and every order's resting total equals its size less its fills."""
    import numpy as np

    rng = np.random.default_rng(23)
    book, engine = OrderBook(lot_size=lot), engine_cls()
    submitted, filled, canceled = {}, {}, set()
    for step in range(5_000):
        r = rng.random()
        if r < 0.55:
            mid = book.mid_price or 1000
            price = max(1, int(mid + rng.integers(-12, 13)))
            size = int(rng.integers(1, 40)) * lot
            side = Side.BID if rng.random() < 0.5 else Side.ASK
            display = int(rng.integers(1, 6)) * lot if rng.random() < 0.4 else None
            order = limit_order(price, size, float(step), side, display_size=display)
            submitted[order.order_id] = size
            fills = engine.process(book, order)
        elif r < 0.75:
            size = int(rng.integers(1, 60)) * lot
            order = market_order(size, float(step), Side.BID if rng.random() < 0.5 else Side.ASK)
            fills = engine.process(book, order)
            assert sum(f.size for f in fills) <= size
        else:
            oid = book.random_order_id(rng)
            fills = []
            if oid is not None:
                engine.process(book, cancel_order(oid, float(step)))
                canceled.add(oid)
        for f in fills:
            assert f.size % lot == 0
            for oid in (f.maker_order_id, f.taker_order_id):
                filled[oid] = filled.get(oid, 0) + f.size
        if step % 50 == 0:
            _assert_book_consistent(book)
    _assert_book_consistent(book)
    for oid in book.all_order_ids():
        o = book.get_order(oid)
        assert o.hidden >= 0 and (o.display_size is None or o.size <= o.display_size)
        assert o.size + o.hidden == submitted[oid] - filled.get(oid, 0)
    for oid, size in submitted.items():  # gone from the book: filled in full, or canceled
        if book.get_order(oid) is None and oid not in canceled:
            assert filled.get(oid, 0) == size
