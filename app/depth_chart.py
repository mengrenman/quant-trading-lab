"""The depth chart under the order book: cumulative visible depth on each side of the mid, as SVG paths.

Bids accumulate from the best bid down and asks from the best ask up, so each curve steps up at
every price level and the gap between them at the bottom is the spread. The chart spans the same
number of ticks either side of the mid: far enough to take in the best ``LEVELS`` levels of the
side whose levels spread further, within limits. The page draws the paths
(components/ladder.html); a large resting order shows as a cliff in its side's curve.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional

from lob.book import OrderBook
from lob.order import Side

LEVELS = 30  # levels of the wider side that the chart takes in
MIN_HALF_WIDTH = 10  # ticks either side of the mid
MAX_HALF_WIDTH = 100
WIDTH, HEIGHT = 400, 110  # the SVG's viewBox
PAD_X, PAD_TOP, PAD_BOTTOM = 4, 14, 16


def nice_ceiling(value: float) -> float:
    """The smallest of 1, 2, 2.5 or 5 times a power of ten that is at least ``value`` (1 for 0)."""
    if value <= 0:
        return 1
    power = 10 ** math.floor(math.log10(value))
    for step in (1, 2, 2.5, 5, 10):
        if step * power >= value:
            return step * power
    return 10 * power  # not reached


def _point(x: float, y: float) -> str:
    return f"{x:.1f} {y:.1f}"


def depth_chart(book: OrderBook, own: Iterable[tuple[str, int]] = ()) -> Optional[dict[str, Any]]:
    """Paths and labels for the depth chart, or None when the book is empty.

    ``own`` holds the (side, price) pairs where the user has orders resting; those inside the
    chart are marked on its price axis. With one side empty, the chart centers on the other
    side's best price.
    """
    best_bid, best_ask = book.best_bid, book.best_ask
    if best_bid is None and best_ask is None:
        return None
    if best_bid is not None and best_ask is not None:
        mid = (best_bid + best_ask) / 2
    else:
        mid = float(best_bid if best_bid is not None else best_ask)

    # Half width: to the LEVELS-th level of whichever side reaches further.
    reach = 0.0
    for side in (Side.BID, Side.ASK):
        levels = book.depth_profile(side, n_levels=LEVELS)
        if levels:
            reach = max(reach, abs(levels[-1][0] - mid))
    half = int(min(MAX_HALF_WIDTH, max(MIN_HALF_WIDTH, math.ceil(reach))))
    lo, hi = mid - half, mid + half

    # Every level inside the chart (with one-tick prices there are at most half + 1 a side).
    inside = {
        side: [(p, d) for p, d in book.depth_profile(side, n_levels=half + 1) if lo <= p <= hi]
        for side in (Side.BID, Side.ASK)
    }
    totals = {side: sum(d for _, d in levels) for side, levels in inside.items()}
    y_max = nice_ceiling(max(totals.values()))
    if float(y_max).is_integer():  # always, for whole lots
        y_max = int(y_max)

    plot_w = WIDTH - 2 * PAD_X
    plot_h = HEIGHT - PAD_TOP - PAD_BOTTOM
    base = PAD_TOP + plot_h

    def x(price: float) -> float:
        return PAD_X + (price - lo) / (hi - lo) * plot_w

    def y(lots: float) -> float:
        return PAD_TOP + (1 - lots / y_max) * plot_h

    paths: dict[str, str] = {}
    for side, edge in ((Side.BID, lo), (Side.ASK, hi)):
        points = [_point(x(mid), base)]
        cum = 0
        for price, depth in inside[side]:
            points.append(_point(x(price), y(cum)))
            cum += depth
            points.append(_point(x(price), y(cum)))
        points.append(_point(x(edge), y(cum)))
        line = "M" + " L".join(points)
        name = side.value
        paths[f"{name}_line"] = line if inside[side] else ""
        paths[f"{name}_area"] = f"{line} L{_point(x(edge), base)} Z" if inside[side] else ""

    return {
        "width": WIDTH,
        "height": HEIGHT,
        "base": base,
        "half_width": half,
        "lo": lo,
        "hi": hi,
        "lo_label": math.ceil(lo),  # whole prices for the axis (the edges are half ticks when the mid is)
        "hi_label": math.floor(hi),
        "mid": mid,
        "mid_x": round(x(mid), 1),
        "y_max": y_max,
        "bid_total": totals[Side.BID],
        "ask_total": totals[Side.ASK],
        **paths,
        "own": [{"side": side, "price": price, "x": round(x(price), 1)}
                for side, price in sorted(set(own)) if lo <= price <= hi],
    }
