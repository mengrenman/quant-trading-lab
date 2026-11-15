"""Tests for lob.feeds: frames in LOBSTER's format as the stream of events the study classes take.

Three kinds of test. The identity tests run each of notebook 14's five models once through ``Simulation``'s own
observer and once through ``simulated_lobster`` and ``lobster_events``, and hold the two to the same events and the
same unit samples of every study (conditions are in their docstring). The hand-made frames test each rule that a
real file adds to what the simulator writes: ties in time, a cancel of part, hidden executions inside a run and alone,
a replace, a remainder priced behind the fills, a crossed row, a break. The last test reads the real day's frames
for NVDA (skipped without the downloaded data) and checks that the stream tiles them.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.model_specs import MODEL_SPECS
from lob.book import OrderBook
from lob.feeds import (BREAK_REASONS, COUNTS, Break, BookRows, Event, EventStream, Fill, PriceLevelBook,
                       lobster_events, run)
from lob.lobster import DAY_START, LOBSTER_TICK, MESSAGE_COLUMNS, book_columns, simulated_lobster
from lob.order import Side, cancel_order, limit_order, market_order, reset_order_id_counter
from lob.simulation import Simulation, SimulationConfig
from lob.studies import (ARRIVAL_LIMIT, ARRIVAL_MARKET, CANCEL, CVD, OFI, VPIN, BookImbalance, OrderToTrade,
                         Resilience, Spread, Study, StudySet, TradeDelta, book_quotes)

BID, ASK = Side.BID, Side.ASK
NONE = (None, 0, None, 0)
ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "data" / "itch" / "derived" / "03272019" / "frames"
NO_DAY = "needs the frames of data/itch/derived/03272019/frames (made by lob.itch.lobster_frames; ignored by git)"


@pytest.fixture(autouse=True)
def reset_ids():
    reset_order_id_counter()
    yield
    reset_order_id_counter()


# ---- Hand-made frames ----

def book_after(bids=(), asks=(), levels=3) -> list:
    """One row of a book frame from (cents, lots) pairs best first: a cent is 100 of LOBSTER's price units."""
    row = []
    for i in range(levels):
        ap, aq = asks[i] if i < len(asks) else (None, 0)
        bp, bq = bids[i] if i < len(bids) else (None, 0)
        row += [np.nan if ap is None else ap * 100, aq, np.nan if bp is None else bp * 100, bq]
    return row


def make(rows, hidden=(), levels=3):
    """Frames from rows ``(time, type, order id, lots, cents, direction, (bids, asks)[, extras])``.

    ``extras`` is a dict that may hold ``match``, ``group``, ``replace``, ``paired``, ``priced`` and ``ns``; if any row
    has one, every row gets ``ns`` (from its time), ``match`` (0), ``replace`` (False), ``group`` (-1), ``paired``
    and ``priced`` (False) columns, as ``lob.itch``'s do.
    ``hidden`` is rows ``(time, group, match, lots, cents, aggressor)``.
    """
    messages, books, extras = [], [], []
    for row in rows:
        t, kind, oid, size, cents, direction, (bids, asks) = row[:7]
        extra = row[7] if len(row) > 7 else {}
        messages.append((t, kind, oid, size, round(cents * 100), direction))
        books.append(book_after(bids, asks, levels))
        extras.append(extra)
    frame = pd.DataFrame(messages, columns=["time", "type", "order_id", "size", "price", "direction"])
    if any(extras):
        frame["ns"] = [e.get("ns", round(t * 1e9)) for e, (t, *_) in zip(extras, messages, strict=True)]
        frame["match"] = [e.get("match", 0) for e in extras]
        frame["replace"] = [e.get("replace", False) for e in extras]
        frame["group"] = [e.get("group", -1) for e in extras]
        frame["paired"] = [e.get("paired", False) for e in extras]
        frame["priced"] = [e.get("priced", False) for e in extras]
    book = pd.DataFrame(books, columns=book_columns(levels))
    if not hidden:
        return frame, book, None
    table = pd.DataFrame([(round(t * 1e9), t, size, round(cents * 100), match, group, aggressor)
                          for t, group, match, size, cents, aggressor in hidden],
                         columns=["ns", "time", "size", "price", "match", "group", "aggressor"])
    return frame, book, table


def events_of(frames, **options):
    stream = lobster_events(*frames, **options)
    return list(stream), stream


def shape(ev):
    """What an event says, without its book: (kind, side, price, size, fills, aggressor)."""
    return (ev.kind, ev.side, ev.price, ev.size, tuple((f.price, f.size) for f in ev.fills), ev.aggressor)


def assert_tiled(items, frames):
    """Every message row is in exactly one event or break, in order; an event's quotes after are the touch of the
    book row of its last message (none if that is crossed) and its quotes before are the last item's after."""
    messages, book = frames[0], frames[1]
    expect, previous = 0, None
    bid, ask = book_columns(1)[2], book_columns(1)[0]
    bid_size, ask_size = book_columns(1)[3], book_columns(1)[1]
    for item in items:
        if isinstance(item, Break):
            assert item.row == expect
            expect, last = item.row + 1, item.row
        else:
            assert item.first == expect
            expect, last = item.last + 1, item.last
        row = book.iloc[last]
        touch = (None if row[bid] != row[bid] else row[bid] / LOBSTER_TICK, int(row[bid_size]),
                 None if row[ask] != row[ask] else row[ask] / LOBSTER_TICK, int(row[ask_size]))
        if touch[0] is not None and touch[2] is not None and touch[0] >= touch[2]:
            touch = NONE
        if isinstance(item, Event):
            assert item.quotes_after == touch
            if previous is not None:
                assert item.quotes_before == previous
        previous = touch
    assert expect == len(messages)


# ---- The book ----

def test_a_price_level_book_answers_what_the_studies_read_of_a_book_as_an_order_book_does():
    # The same orders in an OrderBook and as a row of a book frame (ticks are cents here).
    orders = {BID: [(100, 70), (100, 30), (99, 300), (97, 40)], ASK: [(102, 150), (103, 250), (105, 10)]}
    reference = OrderBook()
    for side, entries in orders.items():
        for price, size in entries:
            reference.add_order(limit_order(price, size, 0.0, side))
    frame, book, _ = make([(1.0, 1, 1, 10, 100, 1, ([(100, 100), (99, 300), (97, 40)],
                                                       [(102, 150), (103, 250), (105, 10)]))])
    shown = PriceLevelBook().read(BookRows(book), 0)
    assert shown.best_bid == reference.best_bid == 100
    assert shown.best_ask == reference.best_ask == 102
    assert shown.spread == reference.spread == 2
    assert shown.mid_price == reference.mid_price == 101.0
    assert shown.tick_size == 1
    for side in (BID, ASK):
        for n in (1, 2, 3, 10):
            assert shown.depth_profile(side, n) == reference.depth_profile(side, n)
    assert shown._bid_depth_at == {100: 100, 99: 300, 97: 40} and shown._ask_depth_at == {102: 150, 103: 250, 105: 10}
    assert book_quotes(shown) == book_quotes(reference) == (100, 100, 102, 150)


def test_a_book_with_an_empty_side_has_no_mid_and_no_spread_and_an_empty_one_has_no_quotes():
    frame, book, _ = make([(1.0, 1, 1, 10, 100, 1, ([(100, 100)], [])), (2.0, 3, 1, 10, 100, 1, ([], []))])
    rows = BookRows(book)
    one_sided, empty = PriceLevelBook().read(rows, 0), PriceLevelBook().read(rows, 1)
    assert (one_sided.best_bid, one_sided.best_ask, one_sided.mid_price, one_sided.spread) == (100, None, None, None)
    assert one_sided.depth_profile(ASK) == [] and one_sided.depth_profile(BID) == [(100, 100)]
    assert book_quotes(one_sided) == (100, 100, None, 0) and book_quotes(empty) == NONE
    assert (empty.best_bid, empty.best_ask, empty.mid_price, empty.spread) == (None,) * 4
    assert BookRows(book).quotes(0) == (100, 100, None, 0) and BookRows(book).quotes(1) == NONE


def test_a_crossed_or_locked_book_has_no_quotes_and_no_levels():
    frame, book, _ = make([(1.0, 1, 1, 10, 100, 1, ([(103, 5)], [(102, 7)])),
                           (2.0, 1, 2, 10, 100, 1, ([(102, 5)], [(102, 7)])),
                           (3.0, 1, 3, 10, 100, 1, ([(101, 5)], [(102, 7)]))])
    rows = BookRows(book)
    assert [rows.crossed(i) for i in range(3)] == [True, True, False]
    for i in (0, 1):
        shown = PriceLevelBook().read(rows, i)
        assert (shown.best_bid, shown.best_ask, shown.mid_price, shown.spread) == (None,) * 4
        assert shown.depth_profile(BID) == [] and shown.depth_profile(ASK) == []
        assert book_quotes(shown) == NONE
    assert PriceLevelBook().read(rows, 2).spread == 1


def test_a_price_level_book_points_at_a_row_without_building_it_and_can_move_or_take_levels():
    frame, book, _ = make([(1.0, 1, 1, 10, 100, 1, ([(100, 5)], [(102, 7)])),
                           (2.0, 1, 2, 10, 100, 1, ([(101, 6)], [(102, 7), (104, 1)]))])
    rows = BookRows(book)
    shown = PriceLevelBook().read(rows, 0)
    assert shown.row == 0 and not shown._ready
    assert shown.best_bid == 100 and shown._ready
    assert shown.at(1).best_bid == 101 and shown.row == 1 and shown.depth_profile(ASK) == [(102, 7), (104, 1)]
    with pytest.raises(IndexError):
        shown.read(rows, 2)
    loaded = PriceLevelBook([(100, 3)], [(101, 4), (103, 5)])
    assert (loaded.best_bid, loaded.best_ask, loaded.spread, loaded.row) == (100, 101, 1, -1)
    assert loaded.load([], [(7, 1)]).best_bid is None
    with pytest.raises(ValueError, match="loaded with levels"):
        loaded.at(0)


def test_empty_levels_may_be_nan_or_the_placeholders_of_a_lobster_file_and_off_grid_prices_stay_fractions():
    book = pd.DataFrame([[9_999_999_999, 0, -9_999_999_999, 0, 9_999_999_999, 0, -9_999_999_999, 0],
                         [1_020_050, 7, 1_000_000, 5, np.nan, 0, np.nan, 0],
                         [1_020_000, 7, 999_900, 5, 1_030_000, 2, 999_800, 0]], columns=book_columns(2))
    rows = BookRows(book)
    assert rows.quotes(0) == NONE
    assert rows.quotes(1) == (10000, 5, 10200.5, 7)            # a half tick stays a half tick
    assert PriceLevelBook().read(rows, 1).depth_profile(ASK) == [(10200.5, 7)]
    assert PriceLevelBook().read(rows, 2).depth_profile(BID) == [(9999, 5)]   # a level of size 0 is empty
    assert isinstance(rows.quotes(2)[0], int) and isinstance(rows.quotes(2)[2], int)


# ---- What a message becomes ----

def test_an_add_is_a_limit_arrival_a_cancel_of_part_takes_out_only_what_it_cancels_and_a_deletion_the_rest():
    frames = make([
        (1.0, 1, 1, 100, 100, 1, ([(100, 100)], [])),
        (1.5, 1, 2, 50, 102, -1, ([(100, 100)], [(102, 50)])),
        (2.0, 2, 1, 40, 100, 1, ([(100, 60)], [(102, 50)])),
        (3.0, 3, 2, 50, 102, -1, ([(100, 60)], [])),
    ])
    events, stream = events_of(frames, from_empty=True)
    assert [shape(e) for e in events] == [
        (ARRIVAL_LIMIT, BID, 100, 100, (), 1),
        (ARRIVAL_LIMIT, ASK, 102, 50, (), -1),
        (CANCEL, BID, 100, 40, (), 0),
        (CANCEL, ASK, 102, 50, (), 0),
    ]
    assert [e.quotes_before for e in events] == [NONE, (100, 100, None, 0), (100, 100, 102, 50), (100, 60, 102, 50)]
    assert [e.quotes_after for e in events] == [(100, 100, None, 0), (100, 100, 102, 50), (100, 60, 102, 50),
                                                (100, 60, None, 0)]
    assert [e.t for e in events] == [1.0, 1.5, 2.0, 3.0]
    assert [(e.first, e.last) for e in events] == [(i, i) for i in range(4)]
    assert all(e.replace == 0 and e.hidden == 0 for e in events)
    counts = stream.counts
    assert (counts["events"], counts["limit"], counts["cancel"], counts["market"]) == (4, 2, 2, 0)
    assert_tiled(events, frames)


def test_messages_at_one_time_are_separate_events_and_executions_against_the_two_sides_are_two_orders():
    frames = make([
        (1.0, 1, 1, 10, 99, 1, ([(99, 10)], [(103, 10)])),
        (1.0, 1, 2, 10, 98, 1, ([(99, 10), (98, 10)], [(103, 10)])),
        (2.0, 4, 1, 10, 99, 1, ([(98, 10)], [(103, 10)])),           # a seller hits the bid
        (2.0, 4, 3, 10, 103, -1, ([(98, 10)], [])),                  # a buyer lifts the ask, at the same time
        (2.0, 1, 4, 10, 97, 1, ([(98, 10), (97, 10)], [])),
        (2.0, 1, 5, 10, 97, 1, ([(98, 10), (97, 20)], [])),
    ])
    events, _ = events_of(frames, from_empty=True)
    assert [shape(e) for e in events] == [
        (ARRIVAL_LIMIT, BID, 99, 10, (), 1),
        (ARRIVAL_LIMIT, BID, 98, 10, (), 1),
        (ARRIVAL_MARKET, ASK, 0, 10, ((99, 10),), -1),
        (ARRIVAL_MARKET, BID, 0, 10, ((103, 10),), 1),
        (ARRIVAL_LIMIT, BID, 97, 10, (), 1),
        (ARRIVAL_LIMIT, BID, 97, 10, (), 1),
    ]
    assert events[1].quotes_before == (99, 10, 103, 10) and events[2].quotes_before == (99, 10, 103, 10)
    assert events[3].quotes_before == (98, 10, 103, 10) and events[4].quotes_before == (98, 10, None, 0)
    assert_tiled(events, frames)


def test_executions_at_one_time_against_one_side_are_one_aggressive_order_with_a_fill_each():
    frames = make([
        (1.0, 1, 1, 100, 100, 1, ([(100, 100)], [(102, 50), (103, 60), (104, 80)])),
        (5.0, 4, 2, 50, 102, -1, ([(100, 100)], [(103, 60), (104, 80)])),
        (5.0, 4, 3, 60, 103, -1, ([(100, 100)], [(104, 80)])),
        (5.0, 4, 4, 10, 104, -1, ([(100, 100)], [(104, 70)])),
        (6.0, 4, 4, 10, 104, -1, ([(100, 100)], [(104, 60)])),       # another order: another time
    ])
    events, stream = events_of(frames)
    first = events[1]
    assert shape(first) == (ARRIVAL_MARKET, BID, 0, 120, ((102, 50), (103, 60), (104, 10)), 1)
    assert first.quotes_before == (100, 100, 102, 50) and first.quotes_after == (100, 100, 104, 70)
    assert (first.first, first.last, first.t) == (1, 3, 5.0)
    assert shape(events[2])[1:] == (BID, 0, 10, ((104, 10),), 1) and events[2].quotes_before == (100, 100, 104, 70)
    assert stream.counts["aggressive"] == 2 and stream.counts["fills"] == 4 and stream.counts["market"] == 2
    assert_tiled(events, frames)


def test_the_rest_of_a_limit_order_that_sweeps_belongs_to_its_event_and_gives_it_a_price():
    frames = make([
        (1.0, 1, 1, 100, 100, 1, ([(100, 100)], [(102, 50), (103, 60)])),
        (5.0, 4, 2, 50, 102, -1, ([(100, 100)], [(103, 60)])),
        (5.0, 4, 3, 60, 103, -1, ([(100, 100)], [])),
        (5.0, 1, 9, 20, 103, 1, ([(103, 20), (100, 100)], [])),      # a buyer who paid up to 103 and rests the rest
    ])
    events, stream = events_of(frames)
    assert len(events) == 2
    swept = events[1]
    assert shape(swept) == (ARRIVAL_LIMIT, BID, 103, 130, ((102, 50), (103, 60)), 1)
    assert (swept.first, swept.last) == (1, 3) and swept.quotes_after == (103, 20, None, 0)
    assert stream.counts["remainders"] == 1 and stream.counts["limit"] == 2 and stream.counts["market"] == 0
    # the same for a seller, whose rest is priced at or below its worst fill
    frames = make([
        (1.0, 1, 1, 100, 100, 1, ([(100, 10), (99, 10)], [(105, 100)])),
        (5.0, 4, 2, 10, 100, 1, ([(99, 10)], [(105, 100)])),
        (5.0, 4, 3, 10, 99, 1, ([], [(105, 100)])),
        (5.0, 1, 9, 5, 99, -1, ([], [(99, 5), (105, 100)])),
    ])
    sold = events_of(frames)[0][1]
    assert shape(sold) == (ARRIVAL_LIMIT, ASK, 99, 25, ((100, 10), (99, 10)), -1)


def test_an_add_priced_behind_the_fills_is_not_a_remainder_but_one_priced_at_the_worst_fill_is():
    behind = make([
        (1.0, 1, 1, 100, 100, 1, ([(100, 100)], [(102, 50)])),
        (5.0, 4, 2, 50, 102, -1, ([(100, 100)], [])),
        (5.0, 1, 9, 20, 101, 1, ([(101, 20), (100, 100)], [])),       # a bid below the fill it just paid: another order
    ])
    events, stream = events_of(behind)
    assert [shape(e) for e in events[1:]] == [(ARRIVAL_MARKET, BID, 0, 50, ((102, 50),), 1),
                                              (ARRIVAL_LIMIT, BID, 101, 20, (), 1)]
    assert stream.counts["adds_behind_the_fills"] == 1 and stream.counts["remainders"] == 0
    assert events[2].quotes_before == events[1].quotes_after
    assert_tiled(events, behind)
    at_worst = make([
        (1.0, 1, 1, 100, 100, 1, ([(100, 100)], [(102, 50)])),
        (5.0, 4, 2, 50, 102, -1, ([(100, 100)], [])),
        (5.0, 1, 9, 20, 102, 1, ([(102, 20), (100, 100)], [])),
    ])
    events, stream = events_of(at_worst)
    assert len(events) == 2 and shape(events[1]) == (ARRIVAL_LIMIT, BID, 102, 70, ((102, 50),), 1)
    assert stream.counts["adds_behind_the_fills"] == 0 and stream.counts["remainders"] == 1
    # an add of the other side, or at another time, is no remainder either
    other = make([
        (1.0, 1, 1, 100, 100, 1, ([(100, 100)], [(102, 50)])),
        (5.0, 4, 2, 50, 102, -1, ([(100, 100)], [])),
        (5.0, 1, 9, 20, 103, -1, ([(100, 100)], [(103, 20)])),        # a sell: the aggressor was a buyer
        (5.1, 4, 3, 10, 100, 1, ([(100, 90)], [(103, 20)])),
        (5.2, 1, 8, 20, 101, -1, ([(100, 90)], [(101, 20), (103, 20)])),
    ])
    events, stream = events_of(other)
    assert [e.kind for e in events] == [ARRIVAL_LIMIT, ARRIVAL_MARKET, ARRIVAL_LIMIT, ARRIVAL_MARKET, ARRIVAL_LIMIT]
    assert stream.counts["remainders"] == 0 and stream.counts["adds_behind_the_fills"] == 0


# ---- What lob.itch's columns add ----

def test_rows_of_one_group_are_one_event_whatever_their_times_and_next_groups_are_not_joined():
    rows = [
        (1.0, 1, 1, 100, 100, 1, ([(100, 100)], [(102, 50), (103, 60)])),
        (5.0, 4, 2, 50, 102, -1, ([(100, 100)], [(103, 60)])),
        (5.00002, 4, 3, 30, 103, -1, ([(100, 100)], [(103, 30)])),        # 20 microseconds later
        (5.00003, 4, 3, 30, 103, -1, ([(100, 100)], [])),                 # and 10 more, in the next group
    ]
    grouped = make([(*row, {"group": g, "match": m}) for row, g, m in zip(rows, (-1, 0, 0, 1), (0, 11, 12, 13),
                                                                           strict=True)])
    events, stream = events_of(grouped)
    assert [shape(e) for e in events[1:]] == [(ARRIVAL_MARKET, BID, 0, 80, ((102, 50), (103, 30)), 1),
                                              (ARRIVAL_MARKET, BID, 0, 30, ((103, 30),), 1)]
    assert [(e.first, e.last, e.ns) for e in events[1:]] == [(1, 2, 5_000_000_000), (3, 3, 5_000_030_000)]
    assert_tiled(events, grouped)
    # without the group column the time decides: three separate orders
    assert [e.kind for e in events_of(make(rows))[0]] == [ARRIVAL_LIMIT, ARRIVAL_MARKET, ARRIVAL_MARKET,
                                                          ARRIVAL_MARKET]


def test_a_hidden_execution_inside_a_group_is_a_fill_of_its_event_in_match_number_order():
    frames = make([
        (1.0, 1, 1, 100, 100, 1, ([(100, 100)], [(102, 50), (103, 60)])),
        (5.0, 4, 2, 50, 102, -1, ([(100, 100)], [(103, 60)]), {"group": 0, "match": 10}),
        (5.0, 4, 3, 30, 103, -1, ([(100, 100)], [(103, 30)]), {"group": 0, "match": 12}),
    ], hidden=[(5.0, 0, 11, 40, 102.5, 1)])      # a trade against hidden liquidity at the mid, between the two
    events, stream = events_of(frames)
    swept = events[1]
    assert shape(swept) == (ARRIVAL_MARKET, BID, 0, 120, ((102, 50), (102.5, 40), (103, 30)), 1)
    assert swept.hidden == 1 and (swept.first, swept.last) == (1, 2)
    assert swept.quotes_before == (100, 100, 102, 50) and swept.quotes_after == (100, 100, 103, 30)
    counts = stream.counts
    assert (counts["hidden_fills"], counts["fills"], counts["hidden_unsigned_runs"]) == (1, 3, 0)
    assert_tiled(events, frames)
    # the studies see the hidden lots: CVD counts all 120
    cvd = run(events, [CVD()], origin=0.0).studies["cvd"].value()
    assert cvd == 120


def test_a_run_of_hidden_executions_alone_is_no_event_is_counted_and_changes_nothing():
    frames = make([
        (1.0, 1, 1, 100, 100, 1, ([(100, 100)], [(102, 50)]), {}),
        (3.0, 1, 2, 10, 101, 1, ([(101, 10), (100, 100)], [(102, 50)]), {"match": 0}),
    ], hidden=[(2.0, 7, 31, 100, 101, 0), (2.0, 7, 32, 60, 101, 0), (2.5, 8, 40, 5, 101.5, 0)])
    with_hidden, stream = events_of(frames)
    without = events_of((frames[0], frames[1]))[0]
    assert [shape(e) for e in with_hidden] == [shape(e) for e in without] and len(with_hidden) == 2
    assert [e.quotes_before for e in with_hidden] == [e.quotes_before for e in without]
    counts = stream.counts
    assert (counts["hidden_trades"], counts["hidden_unsigned_runs"], counts["hidden_unsigned_trades"],
            counts["hidden_unsigned_shares"], counts["hidden_orphans"], counts["fills"]) == (3, 2, 3, 165, 0, 0)


def test_a_signed_hidden_run_with_no_displayed_executions_in_the_frame_is_an_orphan_and_not_a_fill():
    frames = make([(1.0, 1, 1, 100, 100, 1, ([(100, 100)], [(102, 50)]), {"match": 0})],
                  hidden=[(2.0, 7, 31, 100, 101, 1)])
    events, stream = events_of(frames)
    assert len(events) == 1 and stream.counts["hidden_orphans"] == 1 and stream.counts["hidden_unsigned_runs"] == 0


def test_a_replace_is_two_events_flagged_one_and_two_and_a_count_of_messages_can_take_it_as_one():
    rep = {"replace": True}
    frames = make([
        (1.0, 1, 1, 100, 100, 1, ([(100, 100)], [(105, 10)])),
        (2.0, 3, 1, 100, 100, 1, ([], [(105, 10)]), rep),
        (2.0, 1, 2, 100, 101, 1, ([(101, 100)], [(105, 10)]), rep),
        (3.0, 1, 3, 10, 99, 1, ([(101, 100), (99, 10)], [(105, 10)])),
    ])
    events, stream = events_of(frames)
    assert [e.replace for e in events] == [0, 1, 2, 0]
    assert [e.kind for e in events] == [ARRIVAL_LIMIT, CANCEL, ARRIVAL_LIMIT, ARRIVAL_LIMIT]
    assert events[1].quotes_after == (None, 0, 105, 10)          # the state between the two, which never existed
    assert events[2].quotes_before == (None, 0, 105, 10) and events[2].quotes_after == (101, 100, 105, 10)
    assert (stream.counts["replace_deletions"], stream.counts["replace_adds"]) == (1, 1)

    def messages(scope, **options):
        studies = StudySet([OrderToTrade(window=3.0, scope=scope)], auto_sample=False)
        return run(events, studies, origin=0.0, **options).studies["otr"].values(3.0)["messages"]

    assert messages("model") == 4                                  # a replace as two messages, by default
    assert messages("model", replaces_as_one=True) == 3            # as one
    assert messages("everyone", replaces_as_one=True) == 4         # a scope that counts every order counts both


def test_the_new_order_of_a_replace_is_never_the_rest_of_an_aggressive_order():
    rep = {"replace": True}
    frames = make([
        (1.0, 1, 1, 100, 100, 1, ([(100, 100)], [(102, 50)])),
        (1.0, 1, 5, 100, 99, 1, ([(100, 100), (99, 100)], [(102, 50)])),
        (5.0, 4, 2, 50, 102, -1, ([(100, 100), (99, 100)], [])),
        (5.0, 3, 5, 100, 99, 1, ([(100, 100)], []), rep),
        (5.0, 1, 6, 100, 103, 1, ([(103, 100), (100, 100)], []), rep),   # at the same time, priced past the fills
    ])
    events, stream = events_of(frames)
    assert [(e.kind, e.replace) for e in events] == [(ARRIVAL_LIMIT, 0), (ARRIVAL_LIMIT, 0), (ARRIVAL_MARKET, 0),
                                                     (CANCEL, 1), (ARRIVAL_LIMIT, 2)]
    assert stream.counts["remainders"] == 0 and stream.counts["adds_behind_the_fills"] == 0


# ---- Crossed rows, breaks and the start of a file ----

def test_a_crossed_or_locked_touch_is_passed_as_no_quote_and_counted_once_a_row():
    frames = make([
        (1.0, 1, 1, 10, 100, 1, ([(100, 10)], [(102, 10)])),
        (2.0, 1, 2, 10, 103, 1, ([(103, 10), (100, 10)], [(102, 10)])),      # crossed: bid 103, ask 102
        (3.0, 3, 2, 10, 103, 1, ([(100, 10)], [(102, 10)])),
        (4.0, 1, 3, 10, 102, 1, ([(102, 10), (100, 10)], [(102, 10)])),      # locked
    ])
    events, stream = events_of(frames)
    assert [e.quotes_after for e in events] == [(100, 10, 102, 10), NONE, (100, 10, 102, 10), NONE]
    assert [e.quotes_before for e in events[1:]] == [(100, 10, 102, 10), NONE, (100, 10, 102, 10)]
    assert stream.counts["crossed_quotes"] == 2
    assert PriceLevelBook().read(BookRows(frames[1]), 1).spread is None
    # the studies read it as a book with no quotes: a crossed state changes no order flow imbalance
    ofi = run(events, [OFI(window=1.0)], origin=0.0).studies["ofi"]
    assert ofi.values(4.0)["ofi"] == 0


def test_a_cross_is_a_break_whose_row_sets_the_quotes_before_the_next_event_and_the_runner_starts_afresh():
    frames = make([
        (1.0, 1, 1, 100, 100, 1, ([(100, 100)], [(102, 50)])),
        (2.0, 6, 0, 500, 101, 0, ([(100, 100)], [(102, 50)]), {"match": 5}),               # a cross trade (Q)
        (2.5, 6, 1, 40, 100, 1, ([(100, 60)], [(102, 50)]), {"match": 6}),                 # an order in a cross
        (3.0, 1, 2, 10, 99, 1, ([(100, 60), (99, 10)], [(102, 50)])),
        (4.0, 7, 0, 0, 0, 1, ([(100, 60), (99, 10)], [(102, 50)])),                         # a trading action
        (5.0, 3, 1, 60, 100, 1, ([(99, 10)], [(102, 50)])),
    ])
    items, stream = events_of(frames)
    assert [type(i).__name__ for i in items] == ["Event", "Break", "Break", "Event", "Break", "Event"]
    assert [i.reason for i in items if isinstance(i, Break)] == ["cross", "cross execution", "halt"]
    assert set(BREAK_REASONS) == {"cross", "cross execution", "halt"}
    assert items[3].quotes_before == (100, 60, 102, 50)       # the touch the cross execution left, not the add's
    assert items[5].quotes_before == (100, 60, 102, 50) and items[5].quotes_after == (99, 10, 102, 50)
    assert stream.counts["breaks"] == 3 and stream.counts["break_cross"] == 1 and stream.counts["break_halt"] == 1
    assert stream.counts["break_cross_execution"] == 1 and stream.counts["events"] == 3
    assert_tiled(items, frames)

    class Resets(Study):
        key, headline = "resets", "n"

        def _clear(self):
            pass

        def observe(self, *args, **kwargs):
            pass

        def reset(self, index=0, t=None):
            self.log = getattr(self, "log", []) + [(index, t)]
            super().reset(index, t)

        def values(self, t=None, book=None):
            return {"n": len(self.log)}

    spy = Resets()  # every break reason, which was the default until the review of C-N rows
    result = run(items, StudySet([spy], auto_sample=False), unit=1.0, origin=0.0, reset_on=BREAK_REASONS)
    assert spy.log[1:] == [(2, 2.0), (2, 2.5), (4, 4.0)] and result.counts["resets"] == 3
    spy = Resets()  # the default: a cross trade and a trading action
    result = run(items, StudySet([spy], auto_sample=False), unit=1.0, origin=0.0)
    assert spy.log[1:] == [(2, 2.0), (4, 4.0)] and result.counts["breaks"] == 3 and result.counts["resets"] == 2
    with pytest.raises(ValueError, match="no such break reason"):
        run(items, StudySet([Resets()], auto_sample=False), reset_on=("auction",))


def test_a_break_clears_a_window_so_the_value_is_undefined_until_it_has_filled_again():
    frames = make([
        (0.1, 1, 1, 10, 100, 1, ([(100, 10)], [(102, 10)])),
        (1.5, 1, 2, 10, 101, 1, ([(101, 10), (100, 10)], [(102, 10)])),
        (2.5, 6, 0, 500, 101, 0, ([(101, 10), (100, 10)], [(102, 10)])),
        (2.6, 1, 3, 10, 101, 1, ([(101, 20), (100, 10)], [(102, 10)])),
    ])
    items = events_of(frames, from_empty=True)[0]
    ofi = run(items, [OFI(window=2.0)], origin=0.0)
    assert ofi.studies["ofi"].values(2.6)["ofi"] is None and ofi.counts["resets"] == 1
    assert run(items[:2], [OFI(window=2.0)], origin=0.0).studies["ofi"].values(2.0)["ofi"] == 10


def test_the_first_events_quotes_before_are_unknown_unless_the_file_starts_from_an_empty_book():
    frames = make([(1.0, 1, 1, 10, 100, 1, ([(100, 10)], [(102, 10)])),
                   (2.0, 3, 1, 10, 100, 1, ([], [(102, 10)]))])
    real, stream = events_of(frames)
    assert real[0].quotes_before is real[0].quotes_after and stream.counts["unknown_before"] == 1
    assert real[1].quotes_before == real[0].quotes_after
    simulated, stream = events_of(frames, from_empty=True)
    assert simulated[0].quotes_before == NONE and stream.counts["unknown_before"] == 0
    assert ofi_of(real) == 0 and ofi_of(simulated) == 0


def ofi_of(events):
    return run(events, [OFI(window=1.0)], origin=0.0).studies["ofi"].values(2.0)["ofi"]


def test_frames_with_and_without_the_extra_columns_give_the_same_events_where_a_group_is_a_time_and_a_side():
    rows = [
        (1.0, 1, 1, 100, 100, 1, ([(100, 100)], [(102, 50), (103, 60)])),
        (2.0, 4, 2, 50, 102, -1, ([(100, 100)], [(103, 60)])),
        (2.0, 4, 3, 30, 103, -1, ([(100, 100)], [(103, 30)])),
        (2.0, 1, 9, 5, 103, 1, ([(103, 5), (100, 100)], [(103, 30)])),
        (3.0, 2, 1, 40, 100, 1, ([(103, 5), (100, 60)], [(103, 30)])),
        (4.0, 4, 3, 30, 103, -1, ([(103, 5), (100, 60)], [])),
    ]
    plain = events_of(make(rows))[0]
    grouped = events_of(make([(*row, {"group": 0 if i in (1, 2) else 1 if i == 5 else -1, "match": i})
                              for i, row in enumerate(rows)]))[0]
    assert [shape(e) for e in plain] == [shape(e) for e in grouped]
    assert [(e.quotes_before, e.quotes_after) for e in plain] == [(e.quotes_before, e.quotes_after) for e in grouped]


@pytest.mark.parametrize("name, change, message", [
    ("lengths", lambda m, b, h: (m.iloc[:-1], b, h), "one row after each message"),
    ("a column", lambda m, b, h: (m.drop(columns="size"), b, h), "no column 'size'"),
    ("time", lambda m, b, h: (m.assign(time=m["time"][::-1].to_numpy(), ns=m["ns"][::-1].to_numpy()), b, h),
     "go backwards"),
    ("a type", lambda m, b, h: (m.assign(type=[1, 8]), b, h), "message type 8"),
    ("a direction", lambda m, b, h: (m.assign(direction=[1, 0]), b, h), "direction"),
    ("a book column", lambda m, b, h: (m, b.drop(columns="bid_size_2"), h), "4 columns a level"),
    ("a hidden column", lambda m, b, h: (m, b, h.drop(columns="aggressor")), "no column 'aggressor'"),
    ("hidden without groups", lambda m, b, h: (m.drop(columns="group"), b, h), "group and match"),
])
def test_frames_that_are_not_a_pair_of_a_message_file_and_its_book_raise(name, change, message):
    frames = make([(1.0, 1, 1, 10, 100, 1, ([(100, 10)], [(102, 10)]), {"match": 0}),
                   (2.0, 3, 1, 10, 100, 1, ([], [(102, 10)]))], hidden=[(1.5, 0, 9, 5, 101, 0)], levels=2)
    with pytest.raises(ValueError, match=message):
        lobster_events(*change(*frames))


def test_a_group_of_executions_against_both_sides_raises_and_so_does_a_tick_that_is_not_positive():
    frames = make([
        (1.0, 1, 1, 100, 100, 1, ([(100, 100)], [(102, 50)])),
        (2.0, 4, 2, 50, 102, -1, ([(100, 100)], []), {"group": 0, "match": 1}),
        (2.0, 4, 1, 50, 100, 1, ([], []), {"group": 0, "match": 2}),
    ])
    with pytest.raises(ValueError, match="both sides"):
        list(lobster_events(*frames))
    with pytest.raises(ValueError, match="positive"):
        lobster_events(*frames, tick=0)


def test_the_counts_a_stream_keeps_are_the_documented_ones_and_start_again_with_each_iteration():
    frames = make([(1.0, 1, 1, 10, 100, 1, ([(100, 10)], [(102, 10)])), (2.0, 3, 1, 10, 100, 1, ([], [(102, 10)]))])
    stream = lobster_events(*frames)
    assert set(stream.counts) == set(COUNTS) and stream.counts["events"] == 0 and stream.counts["messages"] == 2
    first = list(stream)
    assert stream.counts["events"] == 2
    second = list(stream)
    assert stream.counts["events"] == 2 and [shape(e) for e in first] == [shape(e) for e in second]


# ---- The runner ----

def spread_frames():
    return make([
        (0.5, 1, 1, 10, 100, 1, ([(100, 10)], [])),
        (0.6, 1, 2, 10, 102, -1, ([(100, 10)], [(102, 10)])),
        (1.2, 1, 3, 10, 101, -1, ([(100, 10)], [(101, 10), (102, 10)])),      # this event opens unit 1
        (1.4, 1, 4, 10, 99, 1, ([(100, 10), (99, 10)], [(101, 10), (102, 10)])),
        (4.5, 1, 5, 10, 100, 1, ([(100, 20), (99, 10)], [(101, 10), (102, 10)])),   # this one opens unit 4
    ])


def test_the_sample_of_a_unit_is_taken_from_the_book_before_the_event_that_opens_it():
    frames = spread_frames()
    events = events_of(frames, from_empty=True)[0]
    result = run(events, [Spread(), BookImbalance(levels=1)], unit=1.0, origin=0.0)
    first = dict(result.samples)[1]
    assert first["spread"] == 2          # the book before the event at 1.2, whose own spread is 1
    assert first["imbalance"] == 0.0     # 10 lots at the bid, 10 at the ask, before it: not 10 against 20
    # units the clock jumps over get the book that stood throughout, and each unit is sampled once
    assert [u for u, _ in result.samples] == [1, 2, 3, 4]
    assert [row["spread"] for _, row in result.samples] == [2, 1, 1, 1]
    assert result.counts["samples"] == 4 and result.counts["events"] == 5
    # a set that samples from the book after each event is refused: that is not the convention
    with pytest.raises(ValueError, match="auto_sample"):
        run(events, StudySet([Spread()]), origin=0.0)


def test_the_clock_is_time_less_origin_over_unit_from_the_nanoseconds_when_the_frame_has_them():
    class Clock(Study):
        key, headline = "clock", "n"

        def _clear(self):
            self.seen = []

        def observe(self, t, *rest, **kw):
            self.seen.append(t)

        def values(self, t=None, book=None):
            return {"n": len(self.seen)}

    rows = [(100.0 + 0.5 * k, 1, k + 1, 10, 100 + k, 1, ([(100, 10)], [])) for k in range(7)]
    frames = make([(*row, {}) for row in rows[:6]] + [(*rows[6], {"ns": 103_000_000_000})])
    study = Clock()
    result = run(events_of(frames)[0], StudySet([study], auto_sample=False), unit=2.0, origin=100.0)
    assert study.seen == [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5] and result.counts["events"] == 7
    study = Clock()
    run(events_of(make(rows))[0], StudySet([study], auto_sample=False), unit=0.5, origin=99.0)   # no nanoseconds
    assert study.seen == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    with pytest.raises(ValueError, match="before the origin"):
        run(events_of(make(rows))[0], [Clock()], origin=100.2)
    with pytest.raises(ValueError, match="positive"):
        run([], [Clock()], unit=0)
    # a boundary event is in the unit it opens, from the nanoseconds
    boundary = make([(k + 0.0, 1, k + 1, 10, 100 + k, 1, ([(100, 10)], []), {"ns": (34_200 + k) * 1_000_000_000})
                     for k in range(3)])
    samples = run(events_of(boundary)[0], [Spread()], origin=34_200.0).samples
    assert [u for u, _ in samples] == [1, 2]


def test_listeners_see_each_event_after_the_studies_and_never_a_break():
    frames = make([(1.0, 1, 1, 10, 100, 1, ([(100, 10)], [(102, 10)])),
                   (2.0, 6, 0, 500, 101, 0, ([(100, 10)], [(102, 10)])),
                   (3.0, 3, 1, 10, 100, 1, ([], [(102, 10)]))])
    seen = []
    result = run(events_of(frames)[0], [CVD()], origin=0.0, listeners=[seen.append])
    assert [e.t for e in seen] == [1.0, 3.0] and all(isinstance(e, Event) for e in seen)
    assert result.counts["breaks"] == 1


# ---- The identity with the simulator ----

class Stop(Exception):
    """Raised from a model to end a ``Simulation`` after a given number of orders."""


class Tap:
    """A model seen through its events, as notebook 14's ``Tap`` sees it: the unit samples of each StudySet are
    taken from the book before the event that opens the unit (``on_pre``), then the event is observed.

    It also keeps every event the observer is given, and stops the run after ``limit`` orders (``simulated_lobster``
    counts orders where ``Simulation`` counts loop iterations, and a model may return no order).
    """

    def __init__(self, model, sets, limit):
        self.model, self.sets, self.limit, self.orders = model, sets, limit, 0
        self.samples = [[] for _ in sets]
        self.events = []

    def generate_event(self, book, t, rng):
        order = self.model.generate_event(book, t, rng)
        if order is not None:
            if self.orders == self.limit:
                raise Stop
            self.orders += 1
            for k, studies in enumerate(self.sets):
                if int(order.timestamp) > studies.unit:
                    self.samples[k].extend(studies.sample_to(int(order.timestamp), book, rows=True))
        return order

    def observe(self, t, kind, side, price, size, fills, aggressor, before, after, book, own=False):
        self.events.append((t, kind, side, price, size, tuple((f.price, f.size) for f in fills), aggressor, before,
                            after))
        for studies in self.sets:
            studies.observe(t, kind, side, price, size, fills, aggressor, before, after, book, own)

    def __getattr__(self, name):
        return getattr(self.model, name)


def study_sets():
    """Four sets that between them have every side rule, lag, weighting and onset of the studies' settings."""
    return [
        StudySet(auto_sample=False),
        StudySet([OFI(), CVD(), TradeDelta(side="quote", quote_lag=1), VPIN(side="bvc"), OrderToTrade(scope="everyone"),
                  BookImbalance(levels=5, weighting="touch"), Spread(), Resilience(onset="crossing")],
                 auto_sample=False),
        StudySet([OFI(window=2.0), CVD(), TradeDelta(side="tick"), VPIN(side="tick"), OrderToTrade(),
                  BookImbalance(levels=1), Spread(), Resilience(ticks=5)], auto_sample=False),
        StudySet([OFI(), CVD(), TradeDelta(side="quote", quote_lag=0), VPIN(side="mid", quote_lag=0), OrderToTrade(),
                  BookImbalance(), Spread(), Resilience()], auto_sample=False),
    ]


def close(a, b, tolerance=1e-9):
    if a is None or b is None:
        return a is b
    if isinstance(a, (bool, str)) or isinstance(b, (bool, str)):
        return a == b
    return math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)


def both_ways(make_model, n_events, seed):
    """One run through Simulation's observer and one through simulated_lobster and lobster_events.

    Returns what each saw: the observer's events and the stream's, and every set's unit samples both ways.
    """
    reset_order_id_counter()
    tap = Tap(make_model(), study_sets(), n_events)
    sim = Simulation(model=tap, book=OrderBook(tick_size=1, lot_size=1),
                     config=SimulationConfig(n_events=10**9, warmup_events=0, seed=seed), observer=tap.observe)
    with pytest.raises(Stop):
        sim.run(progress=False)
    reset_order_id_counter()
    messages, book = simulated_lobster(make_model, n_events, seed)
    events = list(lobster_events(messages, book, from_empty=True))
    fed = [run(events, studies, origin=DAY_START) for studies in study_sets()]
    return tap, events, fed


def assert_identity(tap, events, fed, full_limit_orders=0, looser=None):
    """The stream's events are the observer's, and every set's unit samples and final values are the same.

    ``looser`` is a dict from a study's key to the tolerance for that study (1e-9 for any other): the nanosecond
    clock of an ITCH file moves a time by up to 5e-10 s, which a study of the recovery times can show.

    A limit order that crosses the spread and fills in full leaves only its executions in the file, so the stream
    takes it for a market order: for those events, and no others, the kind may differ (``full_limit_orders`` is
    how many the run has, which the test states so that the allowance cannot grow unnoticed).
    """
    looser = looser or {}
    assert len(events) == len(tap.events)
    relaxed = 0
    for ev, (t, kind, side, price, size, fills, aggressor, before, after) in zip(events, tap.events, strict=True):
        assert math.isclose(ev.t - DAY_START, t, abs_tol=1e-9)
        if kind == ARRIVAL_LIMIT and ev.kind == ARRIVAL_MARKET and fills:
            assert price != 0 and ev.price == 0 and size == ev.size
            kind, price, relaxed = ev.kind, ev.price, relaxed + 1
        assert (ev.kind, ev.side, ev.price, ev.size, tuple((f.price, f.size) for f in ev.fills), ev.aggressor,
                ev.quotes_before, ev.quotes_after) == (kind, side, price, size, fills, aggressor, before, after)
    assert relaxed == full_limit_orders
    for direct, fed_run in zip(tap.samples, fed, strict=True):
        adapted = fed_run.samples
        assert len(direct) == len(adapted) > 500
        for (unit_a, row_a), (unit_b, row_b) in zip(direct, adapted, strict=True):
            assert unit_a == unit_b
            for key in row_a:
                assert close(row_a[key], row_b[key], looser.get(key, 1e-9)), (unit_a, key, row_a[key], row_b[key])
    for studies, run_ in zip(tap.sets, fed, strict=True):
        direct, adapted = studies.values(), run_.studies.values()
        for key in direct:
            for name, value in direct[key].items():
                assert close(value, adapted[key][name], looser.get(key, 1e-9)), (key, name, value, adapted[key][name])


def assert_not_vacuous(tap, fed):
    """The identity is about something: trades, quote moves, buckets, episodes and defined samples all occur."""
    assert sum(1 for e in tap.events if e[5]) > 1_000
    default = fed[0].studies.values()
    assert default["vpin"]["buckets"] > 50 and default["resilience"]["started"] > 5
    assert default["resilience"]["recovered"] > 0 and default["resilience"]["quote_moves"] > 100
    defined = {key: sum(row[key] is not None for _, row in fed[0].samples) for key in fed[0].samples[-1][1]}
    assert all(count > 100 for count in defined.values()), defined
    assert fed[3].studies.values()["delta"]["accuracy"] is not None


MODELS = ["smith_zi", "hawkes_zi", "mike_farmer_2008", "parlour", "informed_zi"]


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("key", MODELS)
def test_the_studies_give_the_same_unit_samples_fed_by_the_simulator_and_by_its_lobster_files(key, seed):
    """Each of notebook 14's five models, the dashboard's defaults at a mid of 10,000, 60,000 orders: the studies
    fed by ``Simulation``'s own observer and by ``lobster_events`` on ``simulated_lobster``'s frames of the same
    run (the same seed) see the same events and take the same unit samples, to 1e-9, with the sampling convention
    of notebook 14 (from the book before the event that opens the unit). Four sets of studies between them
    cover every rule and setting. The identity holds under these conditions, which the simulator keeps and a real
    file does not:

    - the simulator writes no hidden, priced (``C``), replaced, crossed or cross message, so none of the rules for
      them is used here (the hand-made frames test them);
    - an aggressive order's fills and what rests of it are written consecutively at one time, so a group is the
      time-and-side rule, and no two aggressive orders share a time and a side (times are continuous). In these
      models every aggressive order also fills once and rests nothing, which is why the sweeping model below is
      run as well;
    - every row of the book frame is the book after the whole event, so the touch after a fill is the touch
      after the event (a real file shows the book after each message and the stream reads the row of the last);
    - a market order is filled in full and meets a non-empty side (otherwise the simulator's observer sees an
      event of the whole size that the file does not show at all, which the test of that name below makes plain),
      and a cancel of an order that is not there is no event in either;
    - a limit order that crosses the spread and fills in full is a limit arrival for the simulator and, since only
      its executions are in the file, a market arrival for the stream (Mike and Farmer's model sends only such
      orders). The kind is read by one branch of ``Resilience``, which does nothing when the order did not rest,
      so the samples agree; the test says how many events differ in kind and allows no others;
    - the book frame holds ten levels and the studies read no more (``BookImbalance`` at most 10 levels,
      ``Resilience`` within the levels the book shows near the touch);
    - the clock is the same, to within the rounding of ``DAY_START + t - DAY_START``.
    """
    spec = MODEL_SPECS[key]
    tap, events, fed = both_ways(lambda: spec.build(spec.defaults(), 10_000), 60_000, seed)
    # Mike and Farmer's orders are all limit orders, and the aggressive ones fill in full: the stream calls them
    # market orders (the file does not say), which only Resilience's own-side branch could notice, and does not here
    full = sum(1 for e in tap.events if e[5]) if key == "mike_farmer_2008" else 0
    assert_identity(tap, events, fed, full_limit_orders=full)
    assert_not_vacuous(tap, fed)


class SweepModel:
    """Limit orders near the touch, market orders of several lots, limit orders that cross the spread and rest what
    is left, and cancels: aggressive orders that fill many times and leave a remainder, which no model of the
    dashboard does. A market order never asks for more than the other side holds, and a crossing limit order always
    rests something."""

    def generate_event(self, book, t, rng):
        t = t + rng.exponential(0.05)
        u = rng.random()
        buy = rng.random() < 0.5
        side, other = (BID, ASK) if buy else (ASK, BID)
        mid = 10_000
        bid, ask = book.best_bid, book.best_ask
        if u < 0.25 and book.num_orders() > 10:
            return cancel_order(book.random_order_id(rng), t)
        depth = book.total_depth(other)
        quote = ask if buy else bid
        if u < 0.40 and depth > 0:
            return market_order(int(rng.integers(1, min(depth, 80) + 1)), t, side)
        if u < 0.55 and quote is not None:
            reach = int(rng.integers(0, 6))
            price = quote + reach if buy else quote - reach
            levels = book.depth_profile(other, 100)
            within = sum(lots for level, lots in levels if (level <= price if buy else level >= price))
            return limit_order(price, within + int(rng.integers(1, 40)), t, side)   # and something rests
        reference = (ask - 1 if ask is not None else mid + 1) if buy else (bid + 1 if bid is not None else mid - 1)
        offset = int(rng.integers(0, 6))
        return limit_order(reference - offset if buy else reference + offset, int(rng.integers(5, 25)), t, side)


@pytest.mark.parametrize("seed", [1, 2])
def test_the_identity_holds_for_orders_that_sweep_several_levels_and_rest_what_is_left(seed):
    """The same two runs for a model whose aggressive orders fill several resting orders at several prices and rest
    the remainder, so a group of rows and the rest of a limit order are exercised against the simulator's own
    events: the fills, the whole size on arrival, the quotes after the last row, and the studies' samples."""
    tap, events, fed = both_ways(SweepModel, 30_000, seed)
    assert_identity(tap, events, fed)
    assert sum(1 for e in tap.events if len(e[5]) > 1) > 500                        # sweeps
    assert sum(1 for e in tap.events if e[5] and e[1] == ARRIVAL_LIMIT) > 500       # a limit order that traded, rested
    assert sum(1 for e in tap.events if e[5] and e[1] == ARRIVAL_MARKET) > 500
    assert_not_vacuous(tap, fed)


def test_a_market_order_that_meets_an_empty_side_is_an_event_of_the_simulator_that_its_file_does_not_show():
    """The first condition of the identity that fails, shown: the observer sees the order (and the order-to-trade
    ratio counts it), the LOBSTER rows hold nothing of it, so the stream has one event fewer."""
    class Orders:
        def __init__(self):
            self.queue = [limit_order(100, 10, 0.5, BID), market_order(5, 1.0, BID), limit_order(102, 10, 1.5, ASK)]

        def generate_event(self, book, t, rng):
            return self.queue.pop(0) if self.queue else limit_order(101, 1, t + 1.0, BID)

    seen = []
    Simulation(model=Orders(), observer=lambda *event: seen.append(event[:7]),
               config=SimulationConfig(n_events=3, warmup_events=0)).run(progress=False)
    assert [(e[1], bool(e[5])) for e in seen] == [(ARRIVAL_LIMIT, False), (ARRIVAL_MARKET, False),
                                                  (ARRIVAL_LIMIT, False)]
    reset_order_id_counter()
    messages, book = simulated_lobster(Orders, 3, seed=1)
    assert len(messages) == 2 and len(list(lobster_events(messages, book, from_empty=True))) == 2


def test_a_limit_order_that_fills_in_full_is_a_limit_arrival_for_the_simulator_and_a_market_arrival_in_its_file():
    """The second condition of the identity that fails, shown: a marketable limit order that leaves nothing to rest
    cannot be told from a market order in LOBSTER's rows, so the stream calls it a market order (price 0)."""
    class Orders:
        def __init__(self):
            self.queue = [limit_order(102, 10, 0.5, ASK), limit_order(103, 10, 1.0, BID)]

        def generate_event(self, book, t, rng):
            return self.queue.pop(0) if self.queue else limit_order(101, 1, t + 1.0, BID)

    seen = []
    Simulation(model=Orders(), observer=lambda *event: seen.append(event),
               config=SimulationConfig(n_events=2, warmup_events=0)).run(progress=False)
    reset_order_id_counter()
    messages, book = simulated_lobster(Orders, 2, seed=1)
    stream = list(lobster_events(messages, book, from_empty=True))
    direct, adapted = seen[1], stream[1]
    assert (direct[1], direct[3], direct[4], [(f.price, f.size) for f in direct[5]]) \
        == (ARRIVAL_LIMIT, 103, 10, [(102, 10)])
    assert (adapted.kind, adapted.price, adapted.size, [(f.price, f.size) for f in adapted.fills]) \
        == (ARRIVAL_MARKET, 0, 10, [(102, 10)])
    assert (direct[7], direct[8]) == (adapted.quotes_before, adapted.quotes_after) == ((None, 0, 102, 10), NONE)


def test_importing_lob_does_not_import_the_feeds():
    import subprocess
    import sys
    code = "import sys, lob, lob.studies, lob.simulation, lob.lobster; print('lob.feeds' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                          check=True).stdout.strip() == "False"


# ---- The real day ----

@pytest.mark.skipif(not (FRAMES / "NVDA_messages.parquet").exists(), reason=NO_DAY)
def test_nvdas_frames_are_tiled_by_the_stream_and_its_hidden_runs_take_their_groups_aggressors():
    from lob import itch
    messages, book, hidden = itch.load_frames(FRAMES, "NVDA")
    rows = 150_000
    messages, book = messages.iloc[:rows], book.iloc[:rows]
    hidden = hidden[hidden["ns"] <= messages["ns"].iloc[-1]]
    stream = lobster_events(messages, book, hidden)
    items = list(stream)
    assert_tiled(items, (messages, book, hidden))
    breaks = [i for i in items if isinstance(i, Break)]
    assert [(b.reason, b.row) for b in breaks] == [("cross", 57)]            # the opening cross
    fills = [i for i in items if isinstance(i, Event) and i.fills]
    group = messages["group"].to_numpy()
    aggressor = {}
    for g, a in zip(hidden["group"], hidden["aggressor"], strict=True):
        aggressor.setdefault(g, set()).add(int(a))
    mixed = 0
    for ev in fills:
        owned = aggressor.get(int(group[ev.first]))
        if owned:
            mixed += 1
            assert owned == {ev.aggressor}
        assert ev.aggressor == -int(messages["direction"].iloc[ev.first])
        assert ev.size == sum(f.size for f in ev.fills) + (int(messages["size"].iloc[ev.last])
                                                           if messages["type"].iloc[ev.last] == 1 else 0)
    executions = int((messages["type"] == 4).sum())
    assert stream.counts["fills"] - stream.counts["hidden_fills"] == executions > 500 and mixed > 20
    # each group of execution rows is one event and each remainder is absorbed into its group's
    counts = stream.counts
    assert counts["events"] + counts["breaks"] == rows - executions + counts["aggressive"] - counts["remainders"]
    assert stream.counts["crossed_quotes"] == 0
    result = run(items, origin=34_200.0)
    assert result.counts["samples"] > 100 and result.counts["resets"] == 1


# ---- Added by the audit: what the mechanical sweep showed no test held ----

def test_a_book_frame_of_one_unit_ticks_no_columns_no_rows_and_levels_with_no_size_are_read_as_they_say():
    # a tick of 1 (prices already in ticks) is a tick; 0 and below are not
    book = pd.DataFrame([[102, 50, 100, 70]], columns=book_columns(1))
    messages = pd.DataFrame([(1.0, 1, 1, 70, 100, 1)], columns=MESSAGE_COLUMNS)
    rows = BookRows(book, tick=1)
    assert rows.quotes(0) == (100, 70, 102, 50) and rows.tick == 1
    (event,) = list(lobster_events(messages, book, tick=1))
    assert (event.price, event.quotes_after) == (100, (100, 70, 102, 50))
    for bad in (0, -1):
        with pytest.raises(ValueError, match="tick must be positive"):
            BookRows(book, tick=bad)
    # a frame with no columns is not a book frame
    with pytest.raises(ValueError, match="4 columns a level .ask and bid price and size.; this one has 0"):
        BookRows(pd.DataFrame())
    # a frame with no rows has no row to read, and arrays with the levels' columns
    nothing = BookRows(pd.DataFrame(columns=book_columns(2)))
    assert nothing.n == 0 and nothing.ask_price.shape == nothing.bid_size.shape == (0, 2)
    # a level of no lots is empty whatever its price says (at the touch too)
    sized = BookRows(pd.DataFrame([[1_020_000, 0, 1_000_000, 0], [1_020_000, 7, 1_000_000, 0]],
                                  columns=book_columns(1)))
    assert sized.quotes(0) == NONE and sized.quotes(1) == (None, 0, 10200, 7) and not sized.crossed(0)


def test_a_price_level_book_shows_ten_levels_unless_asked_for_more():
    levels = [(100 - i, 10 + i) for i in range(12)]
    book = PriceLevelBook(levels, [(101 + i, 5) for i in range(12)])
    assert book.depth_profile(BID) == levels[:10] and len(book.depth_profile(ASK)) == 10
    assert book.depth_profile(BID, 12) == levels and book.depth_profile(BID, 3) == levels[:3]
    reference = OrderBook()
    for price, size in levels:
        reference.add_order(limit_order(price, size, 0.0, BID))
    assert book.depth_profile(BID) == reference.depth_profile(BID)


def test_the_first_ten_fields_of_an_event_are_the_arguments_of_observe_in_their_order():
    import inspect
    assert Event._fields[:10] == tuple(inspect.signature(Study.observe).parameters)[1:11]
    assert Event._fields[:10] == ("t", "kind", "side", "price", "size", "fills", "aggressor", "quotes_before",
                                  "quotes_after", "book")

def test_a_price_level_book_has_every_attribute_the_studies_read_of_a_book_and_no_other_is_read():
    import inspect
    import re

    import lob.studies
    read = set(re.findall(r"\bbook\.(\w+)", inspect.getsource(lob.studies)))
    assert read == {"best_bid", "best_ask", "mid_price", "spread", "tick_size", "depth_profile", "_bid_depth_at",
                    "_ask_depth_at"}  # what the sheet lists; a study that reads more must be added here and to the book
    assert all(hasattr(PriceLevelBook, name) for name in read)

def test_an_event_stream_made_directly_has_the_defaults_of_lobster_events():
    frames = make([
        (1.0, 1, 1, 100, 100, 1, ([(100, 100)], [])),
        (2.0, 1, 2, 50, 102, -1, ([(100, 100)], [(102, 50)])),
    ])
    direct = list(EventStream(frames[0], frames[1]))  # no hidden frame; the book is not empty before the first row
    assert [shape(e) for e in direct] == [shape(e) for e in lobster_events(*frames)]
    assert direct[0].quotes_before is direct[0].quotes_after and direct[0].quotes_before != NONE


# The whole of each symbol's 09:30 to 16:00 frames, counted by a stream-free recount written for the fix from the
# audit's own decoder and replay of the ITCH file (``build/fix/recount.py``: it shares no code with lob.itch or
# lob.feeds): an event is an add, a cancel of part, a deletion or an aggressive order (the displayed executions of one
# group, with the paired reductions that lead them, the deletions between them, and the rest of its limit order when an
# add of its side at the same nanosecond directly follows, priced at or beyond the worst fill, hidden fills counted).
WHOLE_DAY = {
    "NVDA": dict(events=1_032_162, limit=523_971, market=17_556, cancel=490_635, aggressive=17_607, remainders=51,
                 fills=25_026, hidden_fills=1_532, replace_deletions=86_406, replace_adds=86_406,
                 adds_behind_the_fills=8, breaks=1, break_cross=1, break_cross_execution=0, hidden_trades=5_404,
                 hidden_signed_trades=1_532, hidden_signed_shares=130_264, hidden_unsigned_runs=3_462,
                 hidden_unsigned_trades=3_872, hidden_unsigned_shares=294_600, hidden_orphans=0, crossed_quotes=0,
                 displayed_shares=1_977_873, hidden_fill_shares=130_264, paired_rows=0, paired_events=0,
                 interleaved_rows=8),
    "CMCSA": dict(events=912_212, limit=461_351, market=6_675, cancel=444_186, aggressive=6_874, remainders=199,
                  fills=19_404, hidden_fills=586, replace_deletions=46_461, replace_adds=46_461,
                  adds_behind_the_fills=21, breaks=2, break_cross=1, break_cross_execution=1, hidden_trades=2_311,
                  hidden_signed_trades=586, hidden_signed_shares=105_243, hidden_unsigned_runs=1_539,
                  hidden_unsigned_trades=1_725, hidden_unsigned_shares=236_369, hidden_orphans=0, crossed_quotes=0,
                  displayed_shares=3_166_822, hidden_fill_shares=105_243, paired_rows=14, paired_events=0,
                  interleaved_rows=122),
}


@pytest.mark.parametrize("symbol", sorted(WHOLE_DAY))
def test_the_stream_of_a_whole_real_session_has_the_counts_of_an_independent_recount(symbol):
    if not (FRAMES / f"{symbol}_messages.parquet").exists():
        pytest.skip(NO_DAY)
    from lob import itch
    messages, book, hidden = itch.load_frames(FRAMES, symbol)
    stream = lobster_events(messages, book, hidden)
    kinds = {"events": 0, "limit": 0, "market": 0, "cancel": 0}
    for item in stream:
        if isinstance(item, Event):
            kinds["events"] += 1
            kinds["limit" if item.kind == ARRIVAL_LIMIT else "market" if item.kind == ARRIVAL_MARKET else "cancel"] += 1
    want = WHOLE_DAY[symbol]
    assert {key: stream.counts[key] for key in want} == want
    assert kinds == {key: want[key] for key in kinds}
    assert stream.counts["messages"] == len(messages)


def test_an_execution_with_no_side_raises_and_so_does_hidden_executions_with_no_match_column():
    frames = make([(1.0, 1, 1, 10, 100, 1, ([(100, 10)], [(102, 10)]), {"match": 0}),
                   (2.0, 4, 1, 10, 100, 0, ([], [(102, 10)]), {"match": 1, "group": 0})],
                  hidden=[(1.5, 0, 9, 5, 101, 0)], levels=2)
    with pytest.raises(ValueError, match="direction"):
        lobster_events(frames[0], frames[1])
    fixed = (frames[0].assign(direction=[1, 1]), frames[1], frames[2])
    assert len(list(lobster_events(*fixed))) == 2
    with pytest.raises(ValueError, match="group and match"):
        lobster_events(fixed[0].drop(columns="match"), fixed[1], fixed[2])


def test_a_stream_with_no_hidden_executions_counts_none_of_them_whether_the_frame_is_missing_or_empty():
    frames = make([(1.0, 1, 1, 10, 100, 1, ([(100, 10)], [(102, 10)]), {"match": 0})], hidden=[(1.5, 0, 9, 5, 101, 0)])
    hidden_counts = ("hidden_trades", "hidden_unsigned_runs", "hidden_unsigned_trades", "hidden_unsigned_shares",
                     "hidden_orphans")
    for hidden in (None, frames[2].iloc[:0]):
        stream = lobster_events(frames[0], frames[1], hidden)
        list(stream)
        assert {name: stream.counts[name] for name in hidden_counts} == dict.fromkeys(hidden_counts, 0)
    with_one = lobster_events(*frames)
    assert (with_one.counts["hidden_trades"], with_one.counts["hidden_unsigned_runs"]) == (1, 1)


def test_the_fills_of_a_seller_who_trades_against_hidden_liquidity_are_in_match_order_not_price_order():
    # a seller hits the bids: 100 (displayed, match 10), 99.5 (hidden, 11), 99 (displayed, 12): prices fall as the
    # matches rise, so sorting the fills by their price would turn the order round
    frames = make([
        (1.0, 1, 1, 100, 100, 1, ([(100, 10), (99, 10)], [(105, 100)])),
        (5.0, 4, 1, 10, 100, 1, ([(99, 10)], [(105, 100)]), {"group": 0, "match": 10}),
        (5.0, 4, 2, 10, 99, 1, ([], [(105, 100)]), {"group": 0, "match": 12}),
    ], hidden=[(5.0, 0, 11, 5, 99.5, -1)])
    event = events_of(frames)[0][1]
    assert shape(event) == (ARRIVAL_MARKET, ASK, 0, 25, ((100, 10), (99.5, 5), (99, 10)), -1)
    assert event.hidden == 1 and event.replace == 0 and (event.first, event.last) == (1, 2)


def test_a_file_that_starts_with_an_aggressive_order_has_unknown_quotes_before_it_and_counts_it():
    for resting in (-1, 1):  # a buyer takes an ask, a seller hits a bid
        book = ([(100, 100)], [(103, 20)]) if resting == -1 else ([(99, 20)], [(102, 50)])
        frames = make([(5.0, 4, 1, 30, 102 if resting == -1 else 100, resting, book)])
        events, stream = events_of(frames)
        assert len(events) == 1 and events[0].quotes_before is events[0].quotes_after
        assert stream.counts["unknown_before"] == 1 and stream.counts["aggressive"] == 1
        assert events[0].replace == 0 and events[0].hidden == 0 and events[0].ns is None
        empty, stream = events_of(frames, from_empty=True)
        assert empty[0].quotes_before == NONE and stream.counts["unknown_before"] == 0


def test_run_starts_its_clock_at_zero_unless_told_otherwise():
    frames = make([(0.5, 1, 1, 10, 100, 1, ([(100, 10)], [(102, 10)])),
                   (1.2, 1, 2, 10, 101, -1, ([(100, 10)], [(101, 10), (102, 10)]))])
    result = run(events_of(frames, from_empty=True)[0], [Spread()])
    assert [u for u, _ in result.samples] == [1] and result.samples[0][1]["spread"] == 2
    with pytest.raises(ValueError, match="before the origin"):
        run(events_of(frames)[0], [Spread()], origin=1.0)


def test_a_unit_opened_by_the_second_row_of_a_file_is_sampled_from_the_book_the_first_row_left():
    frames = make([(0.5, 1, 1, 10, 100, 1, ([(100, 10)], [(103, 10)])),
                   (1.2, 1, 2, 10, 101, -1, ([(100, 10)], [(101, 10), (103, 10)]))])
    result = run(events_of(frames)[0], [Spread()], origin=0.0)
    # the spread of row 0's book, 3: not that of a book that is empty (none), nor row 1's own (2)
    assert result.samples == [(1, {"spread": 3})]


def test_a_unit_opened_by_the_first_row_is_sampled_from_the_book_after_it_unless_the_file_started_empty():
    frames = make([(1.5, 1, 1, 10, 100, 1, ([(100, 10)], [(102, 10)])),
                   (2.5, 1, 2, 10, 99, 1, ([(100, 10), (99, 10)], [(102, 10)]))])
    unknown = run(events_of(frames)[0], [Spread()], origin=0.0)
    assert unknown.samples == [(1, {"spread": 2}), (2, {"spread": 2})]  # the book already full: row 0's is all we know
    empty = run(events_of(frames, from_empty=True)[0], [Spread()], origin=0.0)
    assert empty.samples == [(1, {"spread": None}), (2, {"spread": 2})]  # a file from an empty book: nothing before it


# ---- After the judgment review (build/fix_report.md): each finding with the smallest case that shows it ----

def paired_rows():
    """A buyer's order (1, displayed at 100) executes at 101 against a seller's (2): the C-N takes 30 lots out of the
    buyer's order, then the E takes them out of the seller's, at one time with one match number (CMCSA's 14)."""
    return [
        (1.0, 1, 1, 100, 100, 1, ([(100, 100)], [])),
        (1.0, 1, 2, 100, 101, -1, ([(100, 100)], [(101, 100)])),
        (1.0, 1, 3, 100, 101, -1, ([(100, 100)], [(101, 200)])),
        (5.0, 2, 1, 30, 100, 1, ([(100, 70)], [(101, 200)]), {"paired": True, "match": 7}),
        (5.0, 4, 2, 30, 101, -1, ([(100, 70)], [(101, 170)]), {"group": 0, "match": 7}),
    ]


def test_the_default_resets_are_at_a_cross_and_a_trading_action_and_not_at_a_cross_execution():
    """A C-N of a displayed order is a trade between two orders unless it is at a cross (the review: MSFT's 211
    would have reset the studies 123 times in a session, and VPIN is defined in half of CMCSA's session)."""
    from lob.feeds import DEFAULT_RESET
    assert BREAK_REASONS == ("cross", "cross execution", "halt") and DEFAULT_RESET == ("cross", "halt")
    frames = make([
        (1.0, 1, 1, 100, 100, 1, ([(100, 100)], [(102, 50)])),
        (2.0, 6, 1, 40, 100, 1, ([(100, 60)], [(102, 50)]), {"match": 6}),                 # a C-N at a cross
        (3.0, 1, 2, 10, 99, 1, ([(100, 60), (99, 10)], [(102, 50)])),
    ])
    items, _ = events_of(frames)

    class Resets(Study):
        key, headline = "resets", "n"

        def _clear(self):
            self.log = getattr(self, "log", [])

        def observe(self, *args, **kwargs):
            pass

        def reset(self, index=0, t=None):
            super().reset(index, t)
            self.log.append((index, t))

        def values(self, t=None, book=None):
            return {"n": len(self.log)}

    spy = Resets()
    result = run(items, StudySet([spy], auto_sample=False), origin=0.0)
    assert result.counts["breaks"] == 1 and result.counts["resets"] == 0 and spy.log == [(0, None)]
    spy = Resets()
    result = run(items, StudySet([spy], auto_sample=False), origin=0.0, reset_on=("cross execution",))
    assert result.counts["resets"] == 1 and spy.log[-1] == (2, 2.0)


def test_paired_reductions_before_an_aggressive_orders_executions_belong_to_its_event():
    frames = make(paired_rows())
    events, stream = events_of(frames)
    assert [shape(e) for e in events[2:]] == [(ARRIVAL_LIMIT, ASK, 101, 100, (), -1),
                                              (ARRIVAL_MARKET, BID, 0, 30, ((101, 30),), 1)]
    trade = events[3]
    assert (trade.first, trade.last, trade.replace, trade.paired) == (3, 4, 0, 0)
    assert trade.quotes_before == (100, 100, 101, 200) and trade.quotes_after == (100, 70, 101, 170)
    counts = stream.counts
    assert (counts["paired_rows"], counts["paired_events"], counts["interleaved_rows"]) == (1, 0, 0)
    assert counts["breaks"] == 0
    assert (counts["events"], counts["aggressive"], counts["cancel"]) == (4, 1, 0)
    assert_tiled(events, frames)


def test_a_paired_reduction_with_no_execution_after_it_is_a_cancel_flagged_paired():
    """Its trade is a hidden one (INTC 3, MSFT 7): displayed lots left the book, and no aggressive order is shown."""
    rows = paired_rows()[:3] + [
        (5.0, 2, 1, 30, 100, 1, ([(100, 70)], [(101, 200)]), {"paired": True, "match": 7}),
        (6.0, 1, 4, 10, 99, 1, ([(100, 70), (99, 10)], [(101, 200)])),
    ]
    frames = make(rows, hidden=[(5.0, 0, 7, 30, 100.5, 0)])
    events, stream = events_of(frames)
    standing = events[3]
    assert (standing.kind, standing.side, standing.price, standing.size, standing.fills) == (CANCEL, BID, 100, 30, ())
    assert (standing.paired, standing.first, standing.last, standing.replace) == (1, 3, 3, 0)
    assert standing.quotes_after == (100, 70, 101, 200) and events[4].paired == 0
    assert (stream.counts["paired_rows"], stream.counts["paired_events"], stream.counts["cancel"]) == (1, 1, 1)
    assert_tiled(events, frames)


def test_a_paired_reduction_leads_an_event_only_when_it_is_at_the_nanosecond_of_the_executions():
    rows = paired_rows()
    rows[3] = (4.0, *rows[3][1:])           # the C-N a second earlier: its trade is not the one that follows
    frames = make(rows)
    events, stream = events_of(frames)
    assert [(e.kind, e.paired, e.first, e.last) for e in events[3:]] == [
        (CANCEL, 1, 3, 3), (ARRIVAL_MARKET, 0, 4, 4)]
    assert stream.counts["paired_events"] == 1 and stream.counts["paired_rows"] == 1
    assert_tiled(events, frames)


def test_a_group_with_deletions_and_paired_reductions_between_its_rows_is_one_event():
    """lob.itch joins executions across them (CMCSA: 85 aggressive orders of 5,926 were cut in two by a deletion)."""
    rows = paired_rows()[:3] + [
        (1.0, 1, 4, 100, 102, -1, ([(100, 100)], [(101, 200), (102, 100)])),
        (5.0, 4, 2, 30, 101, -1, ([(100, 100)], [(101, 170), (102, 100)]), {"group": 0, "match": 10}),
        (5.0, 3, 3, 100, 101, -1, ([(100, 100)], [(101, 70), (102, 100)])),
        (5.0, 2, 1, 20, 100, 1, ([(100, 80)], [(101, 70), (102, 100)]), {"paired": True, "match": 11}),
        (5.0, 4, 2, 20, 101, -1, ([(100, 80)], [(101, 50), (102, 100)]), {"group": 0, "match": 11}),
    ]
    frames = make(rows)
    events, stream = events_of(frames)
    assert len(events) == 5
    swept = events[4]
    assert shape(swept) == (ARRIVAL_MARKET, BID, 0, 50, ((101, 30), (101, 20)), 1)
    assert (swept.first, swept.last) == (4, 7) and swept.quotes_before == (100, 100, 101, 200)
    assert swept.quotes_after == (100, 80, 101, 50)
    assert (stream.counts["interleaved_rows"], stream.counts["paired_rows"], stream.counts["aggressive"]) == (2, 1, 1)
    assert_tiled(events, frames)


def test_a_group_with_an_add_or_another_groups_execution_between_its_rows_raises():
    base = [(1.0, 1, 1, 100, 100, 1, ([(100, 100)], [])), (1.0, 1, 2, 100, 101, -1, ([(100, 100)], [(101, 100)])),
            (1.0, 1, 3, 100, 102, -1, ([(100, 100)], [(101, 100), (102, 100)]))]
    interrupted = make(base + [
        (5.0, 4, 2, 30, 101, -1, ([(100, 100)], [(101, 70), (102, 100)]), {"group": 0, "match": 10}),
        (5.0, 1, 4, 10, 99, 1, ([(100, 100), (99, 10)], [(101, 70), (102, 100)])),
        (5.0, 4, 2, 20, 101, -1, ([(100, 100), (99, 10)], [(101, 50), (102, 100)]), {"group": 0, "match": 11}),
    ])
    with pytest.raises(ValueError, match="group 0 .* row 4 .* type 1"):
        list(lobster_events(*interrupted))
    interleaved = make(base + [
        (5.0, 4, 2, 30, 101, -1, ([(100, 100)], [(101, 70), (102, 100)]), {"group": 0, "match": 10}),
        (5.0, 4, 3, 30, 102, -1, ([(100, 100)], [(101, 70), (102, 70)]), {"group": 1, "match": 11}),
        (5.0, 4, 2, 20, 101, -1, ([(100, 100)], [(101, 50), (102, 70)]), {"group": 0, "match": 12}),
    ])
    with pytest.raises(ValueError, match="group 0 .* row 4 .* group 1"):
        list(lobster_events(*interleaved))


def hidden_frames():
    """A displayed execution with a signed hidden trade in it, and a hidden trade alone later."""
    rows = [
        (1.0, 1, 1, 100, 100, 1, ([(100, 100)], [(102, 50), (103, 60)])),
        (5.0, 4, 2, 50, 102, -1, ([(100, 100)], [(103, 60)]), {"group": 0, "match": 10, "priced": True}),
        (5.0, 4, 3, 30, 103, -1, ([(100, 100)], [(103, 30)]), {"group": 0, "match": 12}),
        (7.0, 1, 4, 10, 99, 1, ([(100, 100), (99, 10)], [(103, 30)])),
    ]
    return make(rows, hidden=[(5.0, 0, 11, 40, 102.5, 1), (6.0, 1, 20, 25, 102.5, 0)])


def test_hidden_fills_are_marked_and_so_are_fills_of_priced_executions():
    events, stream = events_of(hidden_frames())
    swept = events[1]
    assert [(f.price, f.size, f.hidden, f.priced) for f in swept.fills] == [
        (102, 50, False, True), (102.5, 40, True, False), (103, 30, False, False)]
    assert swept.hidden == 1 and (swept.fills[0].price, swept.fills[0].size) == (102, 50)
    assert Fill(5, 3) == (5, 3, False, False)        # a fill made by hand has no flags


def test_the_stream_can_leave_hidden_executions_out_or_show_unsigned_runs_as_trades_of_unknown_side():
    signed, stream = events_of(hidden_frames())                    # the default, as before
    assert [e.kind for e in signed] == [ARRIVAL_LIMIT, ARRIVAL_MARKET, ARRIVAL_LIMIT]
    assert (stream.counts["hidden_fills"], stream.counts["hidden_unsigned_runs"], stream.counts["hidden_unknown_events"]
            ) == (1, 1, 0)
    assert (stream.counts["displayed_shares"], stream.counts["hidden_fill_shares"]) == (80, 40)
    assert (stream.counts["hidden_signed_trades"], stream.counts["hidden_signed_shares"]) == (1, 40)
    assert stream.counts["hidden_unsigned_shares"] == 25 and stream.counts["hidden_trades"] == 2

    none, stream = events_of(hidden_frames(), hidden_mode="none")  # displayed executions only
    assert [shape(e) for e in none[1:2]] == [(ARRIVAL_MARKET, BID, 0, 80, ((102, 50), (103, 30)), 1)]
    assert [e.kind for e in none] == [ARRIVAL_LIMIT, ARRIVAL_MARKET, ARRIVAL_LIMIT]
    assert (stream.counts["hidden_fills"], stream.counts["hidden_fill_shares"], stream.counts["displayed_shares"]
            ) == (0, 0, 80) and stream.counts["hidden_trades"] == 2

    frames = hidden_frames()
    unknown, stream = events_of(frames, hidden_mode="unknown")     # runs of hidden trades alone are events
    assert [e.kind for e in unknown] == [ARRIVAL_LIMIT, ARRIVAL_MARKET, ARRIVAL_MARKET, ARRIVAL_LIMIT]
    alone = unknown[2]
    assert (alone.aggressor, alone.price, alone.size, alone.hidden, alone.t, alone.ns) == (0, 0, 25, 1, 6.0, 6 * 10**9)
    assert [(f.price, f.size, f.hidden) for f in alone.fills] == [(102.5, 25, True)]
    assert (alone.first, alone.last, alone.replace, alone.paired) == (3, 2, 0, 0)    # between the rows 2 and 3
    assert alone.quotes_before == alone.quotes_after == (100, 100, 103, 30) == unknown[1].quotes_after
    assert unknown[3].quotes_before == (100, 100, 103, 30)
    assert stream.counts["hidden_unknown_events"] == 1 and stream.counts["events"] == 4
    assert stream.counts["hidden_fills"] == 2 and stream.counts["hidden_fill_shares"] == 65
    assert stream.counts["market"] == 2 and stream.counts["aggressive"] == 2
    assert_tiled(unknown, frames)
    # run reads such an event without trouble, and VPIN's bucket takes its volume with no side
    done = run(unknown, [VPIN(volume=10, buckets=2, side="true")], origin=0.0)
    assert done.counts["events"] == 4
    with pytest.raises(ValueError, match="hidden_mode"):
        lobster_events(*frames, hidden_mode="all")


def test_an_unknown_side_run_before_the_first_message_and_after_the_last_has_a_place_in_the_stream():
    frames = make([
        (2.0, 1, 1, 100, 100, 1, ([(100, 100)], [(102, 50)]), {"match": 0}),
        (3.0, 1, 2, 10, 99, 1, ([(100, 100), (99, 10)], [(102, 50)])),
    ], hidden=[(1.0, 0, 5, 7, 100.5, 0), (9.0, 1, 6, 9, 100.5, 0)])
    events, stream = events_of(frames, hidden_mode="unknown")
    assert [(e.kind, e.size, e.first, e.last) for e in events] == [
        (ARRIVAL_MARKET, 7, 0, -1), (ARRIVAL_LIMIT, 100, 0, 0), (ARRIVAL_LIMIT, 10, 1, 1), (ARRIVAL_MARKET, 9, 2, 1)]
    assert events[0].quotes_before is events[0].quotes_after and stream.counts["unknown_before"] >= 1
    assert events[3].quotes_before == events[3].quotes_after == (100, 100, 102, 50)
    assert_tiled(events, frames)
    result = run(events, [CVD(), Spread()], origin=0.0)
    assert result.counts["events"] == 4


def test_run_copies_the_streams_counts_when_given_the_stream():
    frames = hidden_frames()
    stream = lobster_events(*frames)
    result = run(stream, [OFI()], origin=0.0)
    assert result.stream_counts == stream.counts and result.stream_counts["hidden_unsigned_runs"] == 1
    assert result.stream_counts["events"] == result.counts["events"] == 3
    assert run(list(stream), [OFI()], origin=0.0).stream_counts == {}       # a list has none to give
    assert result.stream_counts is not stream.counts                         # a copy


def replace_rows():
    rep = {"replace": True}
    return [
        (1.0, 1, 1, 100, 100, 1, ([(100, 100)], [(105, 10)])),
        (2.0, 3, 1, 100, 100, 1, ([], [(105, 10)]), rep),
        (2.0, 1, 2, 100, 101, 1, ([(101, 100)], [(105, 10)]), rep),
        (3.0, 1, 3, 10, 99, 1, ([(101, 100), (99, 10)], [(105, 10)])),
    ]


class Spy(Study):
    """Records what ``observe`` is given."""

    key, headline = "spy", "n"

    def _clear(self):
        self.seen = []

    def observe(self, t, kind, side, price, size, fills, aggressor, quotes_before, quotes_after, book, own=False):
        self.seen.append((t, kind, quotes_before, quotes_after, book.best_bid, book.best_ask, own,
                          quotes_before is quotes_after))

    def values(self, t=None, book=None):
        return {"n": len(self.seen)}


def test_with_replaces_as_one_every_study_sees_the_deletion_with_the_touch_after_the_new_order():
    """The state between a replace's two rows never existed (the review: 4 zero-length recoveries and 195 more
    episode starts in 11,235 on NVDA come from it). The deletion carries the touch and the book after the new order
    and the new order, then, changes nothing."""
    events, _ = events_of(make(replace_rows()))
    was, now = Spy(), Spy()
    run(events, StudySet([was], auto_sample=False), origin=0.0)
    run(events, StudySet([now], auto_sample=False), origin=0.0, replaces_as_one=True)
    assert was.seen[1] == (2.0, CANCEL, (100, 100, 105, 10), (None, 0, 105, 10), None, 105, False, False)
    assert was.seen[2][2:6] == ((None, 0, 105, 10), (101, 100, 105, 10), 101, 105)
    assert now.seen[1] == (2.0, CANCEL, (100, 100, 105, 10), (101, 100, 105, 10), 101, 105, False, False)
    assert now.seen[2] == (2.0, ARRIVAL_LIMIT, (101, 100, 105, 10), (101, 100, 105, 10), 101, 105, True, True)
    assert len(now.seen) == len(was.seen) == 4 and now.seen[3][2] == (101, 100, 105, 10)
    # order flow imbalance sees the replace as a move of the bid from 100 to 101, not a loss and a gain of the bid
    for as_one, expected in ((False, 0), (True, 100)):
        ofi = run(events, [OFI(window=5.0)], origin=0.0, replaces_as_one=as_one).studies["ofi"]
        assert ofi.values(5.0)["ofi"] == expected
    # a deletion with no new order after it (the stream ends, or a break follows) is passed as it is
    lone = [e for e in events[:2]]
    spy = Spy()
    run(lone, StudySet([spy], auto_sample=False), origin=0.0, replaces_as_one=True)
    assert spy.seen[1][3] == (None, 0, 105, 10)


def test_a_group_never_has_its_hidden_fill_counted_in_two_events():
    """What the audit found for a hand-made frame whose group is split: the stream used to hand the hidden fill of
    the group to each part. Now the stream raises (above), and lob.itch's groups are contiguous apart from
    deletions and paired rows, which the group's event holds."""
    rows = [(1.0, 1, 1, 100, 100, 1, ([(100, 100)], [(102, 50), (103, 60)])),
            (5.0, 4, 2, 50, 102, -1, ([(100, 100)], [(103, 60)]), {"group": 0, "match": 10}),
            (5.0, 1, 4, 10, 99, 1, ([(100, 100), (99, 10)], [(103, 60)])),
            (5.0, 4, 3, 30, 103, -1, ([(100, 100), (99, 10)], [(103, 30)]), {"group": 0, "match": 12})]
    frames = make(rows, hidden=[(5.0, 0, 11, 40, 102.5, 1)])
    with pytest.raises(ValueError, match="group 0"):
        list(lobster_events(*frames))


# ---- The identity on the path real frames take ----

def both_ways_itch(make_model, n_events, seed, directory):
    """``both_ways`` with the frames made the way a real day's are: ``simulated_lobster``'s rows written as ITCH
    bytes, read by ``lob.itch.read_messages`` and replayed by ``lobster_frames`` (so with ``ns``, ``match``, ``group``,
    ``replace``, ``paired`` and ``priced`` columns and a hidden table), and fed through ``lobster_events``."""
    from lob import itch
    from tests.test_itch import as_itch
    reset_order_id_counter()
    tap = Tap(make_model(), study_sets(), n_events)
    sim = Simulation(model=tap, book=OrderBook(tick_size=1, lot_size=1),
                     config=SimulationConfig(n_events=10**9, warmup_events=0, seed=seed), observer=tap.observe)
    with pytest.raises(Stop):
        sim.run(progress=False)
    reset_order_id_counter()
    messages, book = simulated_lobster(make_model, n_events, seed)
    path = Path(directory) / "SIM.itch"
    path.write_bytes(as_itch(messages))
    out = itch.lobster_frames(itch.read_messages(path), start=0.0, end=86_400.0)
    assert len(out.messages) == len(messages) and len(out.hidden) == 0
    events = list(lobster_events(out.messages, out.book, out.hidden, from_empty=True))
    fed = [run(events, studies, origin=DAY_START) for studies in study_sets()]
    return tap, events, fed, out


@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("key", MODELS)
def test_the_identity_holds_on_the_path_real_frames_take(key, seed, tmp_path):
    """The identity above, with the frames of a real day's path: ITCH bytes, ``lob.itch``, ``lobster_events``. The
    stream then reads ``group`` and ``ns`` (the group rule and the clock of nanoseconds, not the time-and-side rule and
    the float clock) and carries a ``paired``/``priced`` column and an empty hidden table. Same conditions as
    above, and what it does not reach: a group of more than one row exists only in the sweeping model below (each
    aggressive order of the five models fills once); hidden, priced, replaced, paired and crossed executions are not
    written by the simulator, so their rules have hand-made frames and no identity."""
    spec = MODEL_SPECS[key]
    tap, events, fed, out = both_ways_itch(lambda: spec.build(spec.defaults(), 10_000), 60_000, seed, tmp_path)
    full = sum(1 for e in tap.events if e[5]) if key == "mike_farmer_2008" else 0
    assert_identity(tap, events, fed, full_limit_orders=full)
    assert_not_vacuous(tap, fed)
    assert int((out.messages["group"] >= 0).sum()) > 1_000 and (out.messages["ns"] > 0).all()


@pytest.mark.parametrize("seed", [1, 2])
def test_the_identity_holds_on_the_real_path_for_orders_that_sweep_and_resilience_to_the_nanosecond(seed, tmp_path):
    """The sweeping model through ITCH: groups of several rows are exercised. Every event and every sample is the
    simulator's to 1e-9 except the resilience score, which reads the times of recoveries: ITCH holds a time to the
    nanosecond, which moves it by up to 5.1e-10 s, and the score then differs by up to 1.6e-8 (relative). With the
    simulator's own times substituted every study agrees within 5.2e-11 (the review's ``identity_tolerance.py``)."""
    tap, events, fed, out = both_ways_itch(SweepModel, 30_000, seed, tmp_path)
    assert_identity(tap, events, fed, looser={"resilience": 1e-7})
    assert_not_vacuous(tap, fed)
    rows = out.messages.loc[out.messages["group"] >= 0, "group"].value_counts()
    assert int((rows > 1).sum()) > 500


def test_an_unknown_side_run_goes_after_the_rows_at_its_own_nanosecond():
    """The frames do not say where a hidden trade sat among the rows of its nanosecond; the stream puts it after them,
    with the quotes of the last of them, and documents it."""
    frames = make([
        (1.0, 1, 1, 100, 100, 1, ([(100, 100)], [(102, 50)]), {"match": 0}),
        (5.0, 1, 2, 10, 99, 1, ([(100, 100), (99, 10)], [(102, 50)])),
        (5.0, 1, 3, 20, 98, 1, ([(100, 100), (99, 10), (98, 20)], [(102, 50)])),
        (7.0, 1, 4, 10, 97, 1, ([(100, 100), (99, 10), (98, 20), (97, 10)], [(102, 50)])),
    ], hidden=[(5.0, 0, 6, 9, 101, 0)])
    events, _ = events_of(frames, hidden_mode="unknown")
    assert [(e.kind, e.first, e.last) for e in events] == [
        (ARRIVAL_LIMIT, 0, 0), (ARRIVAL_LIMIT, 1, 1), (ARRIVAL_LIMIT, 2, 2), (ARRIVAL_MARKET, 3, 2),
        (ARRIVAL_LIMIT, 3, 3)]
    assert events[3].quotes_before == events[3].quotes_after == events[2].quotes_after
    assert_tiled(events, frames)


def test_a_reset_marks_the_units_up_to_its_own_as_sampled_and_a_gap_of_over_a_hundred_units_keeps_the_last_100():
    """Two limits of ``StudySet.sample_to`` that ``run`` documents: a reset marks every unit up to its own as sampled
    (units 2 and 3 are never sampled here: the cross is in unit 3 and no event opened unit 2 before it), and a jump of
    more than 100 units samples only the last 100 (the jump to unit 300 samples 201 to 300; 5 to 200 are dropped)."""
    frames = make([
        (1.5, 1, 1, 100, 100, 1, ([(100, 100)], [(102, 50)]), {"match": 0}),
        (3.5, 6, 0, 500, 101, 0, ([(100, 100)], [(102, 50)]), {"match": 5}),               # a cross, in unit 3
        (4.5, 1, 2, 10, 99, 1, ([(100, 100), (99, 10)], [(102, 50)])),
        (300.5, 1, 3, 10, 98, 1, ([(100, 100), (99, 10), (98, 10)], [(102, 50)])),
    ])
    items = events_of(frames)[0]
    units = [unit for unit, _ in run(items, [Spread()], origin=0.0).samples]
    assert units == [1, 4] + list(range(201, 301))


@pytest.mark.skipif(not (FRAMES / "CMCSA_messages.parquet").exists(), reason=NO_DAY)
def test_cmcsas_non_printable_executions_in_continuous_trading_reset_nothing():
    """CMCSA's 15 C-N rows in 09:30 to 16:00: 14 are the other half of a printed trade (a reduction, never a break) and
    one is at the opening cross. Before the fix the stream had 16 breaks and the default run reset 16 times, VPIN
    defined at 11,231 of 22,200 units; now 2 breaks (the cross trade and the C-N at its nanosecond) and one reset."""
    from lob import itch
    messages, book, hidden = itch.load_frames(FRAMES, "CMCSA")
    stream = lobster_events(messages, book, hidden)
    result = run(stream, [Spread()], origin=34_200.0)
    assert result.counts["breaks"] == 2 and result.counts["resets"] == 1
    assert (stream.counts["break_cross"], stream.counts["break_cross_execution"], stream.counts["paired_rows"]) == (
        1, 1, 14)
    assert result.stream_counts["hidden_unsigned_runs"] == 1_539 and result.stream_counts["paired_events"] == 0
    assert int(messages["paired"].sum()) == 14 and int(messages["priced"].sum()) == 451
