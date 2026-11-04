"""Tests for metaorder impact against a twin run (lob/impact.py)."""

import math

import numpy as np
import pytest

from lob.book import OrderBook
from lob.impact import Market, paired_metaorders, warmed_market
from lob.matching import PriceTimeEngine
from lob.models.zero_intelligence import SmithFarmerModel
from lob.order import Side, limit_order


class FarAway:
    """A model whose orders rest far from the market (at price 1), so only the metaorder moves the price."""

    def generate_event(self, book, t, rng):
        return limit_order(1, 1, t + 1.0, Side.BID)


def static_market():
    book = OrderBook()
    for price in (99, 98):
        book.add_order(limit_order(price, 10, 0.0, Side.BID))
    for price in (101, 102, 103, 104):
        book.add_order(limit_order(price, 10, 0.0, Side.ASK))
    return Market(model=FarAway(), book=book, engine=PriceTimeEngine(), rng=np.random.default_rng(0))


def test_a_metaorder_on_a_book_that_stays_put():
    (run,) = paired_metaorders(static_market(), sizes=[20], children=2, spacing=1, after=(1, 3))
    # Two buys of 10 take the asks at 101 and 102: the best ask is then 103, the mid 101.
    assert (run.filled, run.cost, run.impact_end) == (20, 1.5, 1.0)
    assert run.impact_after == {1: 1.0, 3: 1.0}  # nothing else moves the price
    (sell,) = paired_metaorders(static_market(), sizes=[10], children=1, spacing=2, side=Side.ASK, after=(1,))
    assert (sell.filled, sell.cost, sell.impact_end) == (10, 1.0, 0.5)  # the bid falls from 99 to 98


def test_sizes_must_split_evenly():
    with pytest.raises(ValueError):
        paired_metaorders(static_market(), sizes=[15], children=2, spacing=1)


def test_the_twin_runs_start_from_the_same_market_and_repeat_exactly():
    make = lambda: SmithFarmerModel(mu=1.0, lam=0.5, delta=0.1, price_range=50, initial_mid=10000)  # noqa: E731
    start = warmed_market(make, seed=3, warmup_events=2_000)
    first = paired_metaorders(start, sizes=[20, 100], children=10, spacing=20, after=(200,))
    again = paired_metaorders(start, sizes=[20, 100], children=10, spacing=20, after=(200,))
    assert [(r.impact_end, r.cost) for r in first] == [(r.impact_end, r.cost) for r in again]
    assert all(r.filled == r.size for r in first) and first[0].market_volume > 0
    assert all(math.isfinite(r.impact_end) for r in first)
