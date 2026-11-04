"""The price impact of a metaorder: a large order executed as a sequence of smaller market orders.

A metaorder of ``size`` lots is split into ``children`` equal market orders, one every ``spacing``
model events (an even schedule in event time). Its impact is measured against a twin run that
starts from the same book, model and random state but sends nothing: the difference between the
two mid-prices at the same model event is the metaorder's effect, with most of the market's own
price noise taken out (common random numbers). The runs drift apart as the metaorder changes the
book and with it the model's later decisions, so the pairing reduces the noise without removing it.

Used by notebook 09 to test the square-root law of impact (impact grows like the square root of
the metaorder's size).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from lob.book import OrderBook
from lob.matching import MatchingEngine, PriceTimeEngine
from lob.order import Side, market_order


@dataclass
class Market:
    """A model, its book, engine and random state, and its clock: everything a run continues from."""
    model: object
    book: OrderBook
    engine: MatchingEngine
    rng: np.random.Generator
    t: float = 0.0

    def step(self) -> tuple[bool, int]:
        """One model event. Returns whether there was one, and the lots it traded."""
        order = self.model.generate_event(self.book, self.t, self.rng)
        if order is None:
            return False, 0
        self.t = max(self.t, order.timestamp)
        fills = self.engine.process(self.book, order)
        return True, sum(f.size for f in fills)


def warmed_market(make_model: Callable[[], object], seed: int, warmup_events: int = 5_000,
                  engine: Optional[MatchingEngine] = None) -> Market:
    """A market after ``warmup_events`` model events from an empty book."""
    market = Market(model=make_model(), book=OrderBook(), engine=engine or PriceTimeEngine(),
                    rng=np.random.default_rng(seed))
    events = 0
    while events < warmup_events:
        happened, _ = market.step()
        events += happened
    return market


@dataclass
class MetaorderRun:
    """What one metaorder did, against its twin run without it (all in ticks; ``side`` signs them)."""
    size: int
    filled: int
    cost: float  # side times (the fills' average price - the mid at the start)
    impact_end: float  # side times (mid with - mid without) just after the last child order
    impact_after: dict[int, float] = field(default_factory=dict)  # the same, this many events later
    market_volume: int = 0  # lots the market traded during the execution, in the twin run


def _mids(market: Market, events: int, checkpoints: list[int],
          children: dict[int, int], side: Side) -> tuple[dict[int, float], int, float, int]:
    """Run ``events`` model events, sending ``children[k]`` lots at market after the k-th.

    Returns the mid-price at each checkpoint (after that many model events and any child order
    sent then; NaN with a side of the book empty), the lots and the cost the children filled
    (price times lots), and the lots the model's own orders traded up to the last child.
    """
    mids: dict[int, float] = {}
    wanted = set(checkpoints)
    filled, cost, volume = 0, 0.0, 0
    last_child = max(children) if children else 0
    events_done = 0
    while events_done < events:
        happened, traded = market.step()
        if not happened:
            continue
        events_done += 1
        if events_done <= last_child:
            volume += traded
        lots = children.get(events_done)
        if lots:
            fills = market.engine.process(market.book, market_order(lots, market.t, side))
            filled += sum(f.size for f in fills)
            cost += sum(f.price * f.size for f in fills)
        if events_done in wanted:
            mid = market.book.mid_price
            mids[events_done] = float(mid) if mid is not None else float("nan")
    return mids, filled, cost, volume


def paired_metaorders(start: Market, sizes: list[int], children: int, spacing: int,
                      after: tuple[int, ...] = (1_000, 5_000), side: Side = Side.BID) -> list[MetaorderRun]:
    """Execute a metaorder of each size from the same starting market, each against one twin run.

    Each size runs from its own copy of ``start``; the twin, which sends nothing, is shared. Sizes
    must be multiples of ``children``.
    """
    sign = 1 if side == Side.BID else -1
    end = children * spacing
    checkpoints = [end] + [end + a for a in after]
    total = checkpoints[-1]
    start_mid = start.book.mid_price
    twin_mids, _, _, market_volume = _mids(copy.deepcopy(start), total, checkpoints,
                                           {k * spacing: 0 for k in range(1, children + 1)}, side)
    runs = []
    for size in sizes:
        if size % children:
            raise ValueError(f"size {size} is not a multiple of {children} children")
        schedule = {k * spacing: size // children for k in range(1, children + 1)}
        mids, filled, cost, _ = _mids(copy.deepcopy(start), total, checkpoints, schedule, side)
        diff = {c: sign * (mids[c] - twin_mids[c]) for c in checkpoints}
        runs.append(MetaorderRun(
            size=size, filled=filled,
            cost=sign * (cost / filled - start_mid) if filled and start_mid is not None else float("nan"),
            impact_end=diff[end], impact_after={a: diff[end + a] for a in after},
            market_volume=market_volume,
        ))
    return runs
