"""Tests for the depth chart under the order book (app/depth_chart.py)."""

import re

import pytest

from app.depth_chart import (
    HEIGHT, MAX_HALF_WIDTH, MIN_HALF_WIDTH, PAD_BOTTOM, PAD_TOP, PAD_X, WIDTH, depth_chart, nice_ceiling,
)
from lob.book import OrderBook
from lob.order import Side, limit_order


def book_with(bids=(), asks=()):
    book = OrderBook()
    for price, size in bids:
        book.add_order(limit_order(price, size, 0.0, Side.BID))
    for price, size in asks:
        book.add_order(limit_order(price, size, 0.0, Side.ASK))
    return book


def coordinates(path):
    """Every number in an SVG path of M and L commands, in order."""
    return [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", path)]


def test_nice_ceiling():
    values = [0, 1, 3, 21, 99, 101, 2_400, 2_600]
    assert [nice_ceiling(v) for v in values] == [1, 1, 5, 25, 100, 200, 2_500, 5_000]


def test_curves_accumulate_from_the_mid_outward():
    chart = depth_chart(book_with(bids=[(99, 10), (97, 30)], asks=[(101, 20)]))
    assert chart["mid"] == 100 and chart["half_width"] == MIN_HALF_WIDTH
    assert (chart["lo"], chart["hi"], chart["lo_label"], chart["hi_label"]) == (90, 110, 90, 110)
    assert (chart["bid_total"], chart["ask_total"], chart["y_max"]) == (40, 20, 50)

    def x(price):
        return PAD_X + (price - 90) / 20 * (WIDTH - 2 * PAD_X)

    def y(lots):
        return PAD_TOP + (1 - lots / 50) * (HEIGHT - PAD_TOP - PAD_BOTTOM)

    # From the mid along zero to the best bid, up by 10 lots there, across to 97, up to 40, out to the edge.
    steps = [(100, 0), (99, 0), (99, 10), (97, 10), (97, 40), (90, 40)]
    assert coordinates(chart["bid_line"]) == pytest.approx([v for p, n in steps for v in (x(p), y(n))], abs=0.051)
    assert chart["base"] == y(0)
    # The area closes the line down to the axis.
    assert coordinates(chart["bid_area"])[-2:] == pytest.approx([x(90), y(0)], abs=0.051)
    steps = [(100, 0), (101, 0), (101, 20), (110, 20)]
    assert coordinates(chart["ask_line"]) == pytest.approx([v for p, n in steps for v in (x(p), y(n))], abs=0.051)


def test_the_chart_reaches_the_30th_level_of_the_wider_side_within_limits():
    bids = [(p, 5) for p in range(99, 59, -1)]  # 40 levels, one a tick
    chart = depth_chart(book_with(bids=bids, asks=[(101, 5)]))
    assert chart["half_width"] == 30  # the 30th bid, 70, is 30 ticks from the mid
    assert chart["bid_total"] == 30 * 5  # the levels from 99 down to 70
    # A far-off order on one side would stretch the chart, so the half width stops at the limit;
    # what lies beyond is left out.
    chart = depth_chart(book_with(bids=[(99, 5)], asks=[(101, 5), (900, 7)]))
    assert chart["half_width"] == MAX_HALF_WIDTH
    assert chart["ask_total"] == 5


def test_with_one_side_empty_the_chart_centers_on_the_other():
    chart = depth_chart(book_with(asks=[(105, 8), (107, 2)]))
    assert chart["mid"] == 105
    assert chart["bid_line"] == chart["bid_area"] == ""
    assert chart["ask_total"] == 10
    assert depth_chart(OrderBook()) is None


def test_your_orders_inside_the_chart_are_marked():
    bids = [(p, 5) for p in range(99, 59, -1)] + [(50, 9)]  # the chart reaches 70, the 30th level
    chart = depth_chart(book_with(bids=bids, asks=[(101, 20)]),
                        own=[("bid", 99), ("bid", 99), ("bid", 50), ("ask", 101)])
    assert [(o["side"], o["price"]) for o in chart["own"]] == [("ask", 101), ("bid", 99)]  # 50 is outside



def test_a_spread_wider_than_the_limit_keeps_both_best_quotes_in_view():
    chart = depth_chart(book_with(bids=[(10, 4)], asks=[(260, 6)]))
    assert chart["half_width"] > MAX_HALF_WIDTH
    assert chart["lo"] <= 10 and chart["hi"] >= 260
    assert (chart["bid_total"], chart["ask_total"]) == (4, 6)
