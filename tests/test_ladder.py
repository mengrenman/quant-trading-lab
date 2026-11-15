"""Tests for lob.ladder: the degradations of an order-level stream to what a price-level feed gives.

Three kinds of test. The hand-made frames show each rule of a rung on a book small enough to read: the replace as one
change, the best price first, the fused trade that is not also a cancel, what a batch's net change loses, the stamp
of a batch, the view a trade has of the book and what a shift does to it, a break that ends a batch. The simulator's
frames (seeds 1 and 2, a sweeping model and one of notebook 14's) give every rung a stream with a truth: each rung
runs a full set of studies, and the finest batch is the reference's book after each nanosecond. The last tests read the
real day's frames of the two exploration stocks (skipped without the downloaded data) and check what the exploration's
review reproduced: the finest batch is the book after each nanosecond, a replace as one gives the per-nanosecond
reading (CMCSA's 3,488 one-tick and 2,991 two-tick readings), and ``levels`` ends on the reference's touch after every
event (2 events of NVDA it cannot split).
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from app.model_specs import MODEL_SPECS
from lob import ladder as ladder_module
from lob.feeds import Break, Event, lobster_events
from lob.ladder import Day, batches, level_changes, levels, reference, run_rung, shifted
from lob.lobster import DAY_START, simulated_lobster
from lob.order import Side, reset_order_id_counter
from lob.studies import (ARRIVAL_LIMIT, ARRIVAL_MARKET, CANCEL, CVD, OFI, VPIN, BookImbalance, OrderToTrade,
                         Resilience, Spread, Study, StudySet, TradeDelta)
from tests.test_feeds import SweepModel, make

BID, ASK = Side.BID, Side.ASK
ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "data" / "itch" / "derived" / "03272019" / "frames"
NO_DAY = "needs the frames of data/itch/derived/03272019/frames (made by lob.itch.lobster_frames; ignored by git)"
FINEST = 1e-6          # milliseconds: one nanosecond
NONE = (None, 0, None, 0)


@pytest.fixture(autouse=True)
def reset_ids():
    reset_order_id_counter()
    yield
    reset_order_id_counter()


# ---- Hand-made frames ----

def day_of(rows, levels=3, end=10.0, origin=0.0):
    frames = make(rows, levels=levels)
    return Day(frames[0], frames[1], origin=origin, end=end, from_empty=True)


def sweep_rows():
    """Four asks of 10 at 101 to 104, two bids at 99 and 98; at 2.0 a market buy of 25 takes 101, 102 and 5 of 103;
    at 3.0 the order at 104 is cancelled. Frames of four levels."""
    asks = [(101, 10), (102, 10), (103, 10), (104, 10)]
    bids = [(99, 10), (98, 10)]
    return [
        (1.0, 1, 1, 10, 101, -1, ([], asks[:1])),
        (1.1, 1, 2, 10, 102, -1, ([], asks[:2])),
        (1.2, 1, 3, 10, 103, -1, ([], asks[:3])),
        (1.3, 1, 4, 10, 104, -1, ([], asks)),
        (1.4, 1, 5, 10, 99, 1, (bids[:1], asks)),
        (1.5, 1, 6, 10, 98, 1, (bids, asks)),
        (2.0, 4, 1, 10, 101, -1, (bids, asks[1:])),
        (2.0, 4, 2, 10, 102, -1, (bids, asks[2:])),
        (2.0, 4, 3, 5, 103, -1, (bids, [(103, 5), (104, 10)])),
        (3.0, 3, 4, 10, 104, -1, (bids, [(103, 5)])),
    ]


def replace_rows():
    """Bids at 99, 98 and 100 and an ask at 105; at 2.0 the order at 100 is replaced by one at 101."""
    rep = {"replace": True}
    return [
        (1.0, 1, 1, 10, 105, -1, ([], [(105, 10)])),
        (1.1, 1, 2, 10, 99, 1, ([(99, 10)], [(105, 10)])),
        (1.2, 1, 3, 10, 98, 1, ([(99, 10), (98, 10)], [(105, 10)])),
        (1.3, 1, 7, 10, 100, 1, ([(100, 10), (99, 10), (98, 10)], [(105, 10)])),
        (2.0, 3, 7, 10, 100, 1, ([(99, 10), (98, 10)], [(105, 10)]), rep),
        (2.0, 1, 8, 10, 101, 1, ([(101, 10), (99, 10), (98, 10)], [(105, 10)]), rep),
    ]


def sweep_and_rest_rows():
    """Three asks of 10 (and a fourth, deeper than the frames' three levels), a bid at 99; at 2.0 a limit buy of 35 at
    103 takes the three and rests 5 at 103: the asks it leaves show a level that was out of view before."""
    bids = [(99, 10)]
    return [
        (1.0, 1, 1, 10, 101, -1, ([], [(101, 10)])),
        (1.1, 1, 2, 10, 102, -1, ([], [(101, 10), (102, 10)])),
        (1.2, 1, 3, 10, 103, -1, ([], [(101, 10), (102, 10), (103, 10)])),
        (1.3, 1, 4, 10, 104, -1, ([], [(101, 10), (102, 10), (103, 10)])),
        (1.4, 1, 5, 10, 99, 1, (bids, [(101, 10), (102, 10), (103, 10)])),
        (2.0, 4, 1, 10, 101, -1, (bids, [(102, 10), (103, 10), (104, 10)])),
        (2.0, 4, 2, 10, 102, -1, (bids, [(103, 10), (104, 10)])),
        (2.0, 4, 3, 10, 103, -1, (bids, [(104, 10)])),
        (2.0, 1, 6, 5, 103, 1, ([(103, 5), (99, 10)], [(104, 10)])),
    ]


def limit_rows():
    """Three asks of 10 at 101 to 103 and a deeper one at 104 (out of the frames' three levels), three bids of 10 at 99
    to 97 and a deeper one at 96. At 2.0 a market buy of 25 takes 101, 102 and 5 of 103 (the level the window ends on);
    at 3.0 a market sell of 25 takes 99, 98 and 5 of 97."""
    asks = [(101, 10), (102, 10), (103, 10)]
    bids = [(99, 10), (98, 10), (97, 10)]
    return [
        (1.0, 1, 1, 10, 101, -1, ([], asks[:1])),
        (1.1, 1, 2, 10, 102, -1, ([], asks[:2])),
        (1.2, 1, 3, 10, 103, -1, ([], asks)),
        (1.3, 1, 4, 10, 104, -1, ([], asks)),
        (1.4, 1, 5, 10, 99, 1, (bids[:1], asks)),
        (1.5, 1, 6, 10, 98, 1, (bids[:2], asks)),
        (1.6, 1, 7, 10, 97, 1, (bids, asks)),
        (1.7, 1, 8, 10, 96, 1, (bids, asks)),
        (2.0, 4, 1, 10, 101, -1, (bids, asks[1:] + [(104, 10)])),
        (2.0, 4, 2, 10, 102, -1, (bids, asks[2:] + [(104, 10)])),
        (2.0, 4, 3, 5, 103, -1, (bids, [(103, 5), (104, 10)])),
        (3.0, 4, 5, 10, 99, 1, (bids[1:] + [(96, 10)], [(103, 5), (104, 10)])),
        (3.0, 4, 6, 10, 98, 1, (bids[2:] + [(96, 10)], [(103, 5), (104, 10)])),
        (3.0, 4, 7, 5, 97, 1, ([(97, 5), (96, 10)], [(103, 5), (104, 10)])),
    ]


def reveal_rows():
    """Five asks of 10 at 101 to 105 (the frames show three), a bid at 99; at 2.0 a limit buy of 25 at 102 takes 101
    and 102 and rests 5 at 102: the asks it leaves show two levels that were out of view, which the state between
    its updates does not."""
    asks = [(101, 10), (102, 10), (103, 10), (104, 10), (105, 10)]
    rows = [(1.0 + 0.1 * i, 1, i + 1, 10, 101 + i, -1, ([], asks[:min(i + 1, 3)])) for i in range(5)]
    rows[3] = (1.3, 1, 4, 10, 104, -1, ([], asks[:3]))
    rows[4] = (1.4, 1, 5, 10, 105, -1, ([], asks[:3]))
    return rows + [
        (1.5, 1, 6, 10, 99, 1, ([(99, 10)], asks[:3])),
        (2.0, 4, 1, 10, 101, -1, ([(99, 10)], asks[1:4])),
        (2.0, 4, 2, 10, 102, -1, ([(99, 10)], asks[2:5])),
        (2.0, 1, 7, 5, 102, 1, ([(102, 5), (99, 10)], asks[2:5])),
    ]


def beyond_rows():
    """Two asks at 101 and 102 that the frames of two levels show and a third at 103 that they do not, a bid at 99; at
    2.0 a limit buy of 35 at 103 takes the three and rests 5 at 103: its last fill, at 103, is at a level the frames
    never showed, and the level it rests on is at the same price."""
    return [
        (1.0, 1, 1, 10, 101, -1, ([], [(101, 10)])),
        (1.1, 1, 2, 10, 102, -1, ([], [(101, 10), (102, 10)])),
        (1.2, 1, 3, 10, 103, -1, ([], [(101, 10), (102, 10)])),
        (1.3, 1, 4, 10, 99, 1, ([(99, 10)], [(101, 10), (102, 10)])),
        (2.0, 4, 1, 10, 101, -1, ([(99, 10)], [(102, 10), (103, 10)])),
        (2.0, 4, 2, 10, 102, -1, ([(99, 10)], [(103, 10)])),
        (2.0, 4, 3, 10, 103, -1, ([(99, 10)], [])),
        (2.0, 1, 5, 5, 103, 1, ([(103, 5), (99, 10)], [])),
    ]


def priced_rows(cents, cross=False):
    """A bid at 99 and an ask at 105; at 1.10 s an ask of 10 is added at 104 and at 1.12 s, in the same 100 ms window,
    it is executed in full: by a printable C (``priced``) whose price, ``cents``, is a cent off its order's, or by a
    plain execution at 104. lob.itch writes a printable C at its own price and reduces the book at the order's (on the
    two exploration days the price of an ask is a cent below its level and of a bid a cent above), so the book after
    the execution is the same whatever price the row carries. With ``cross`` the execution is an order executed in a
    cross (a row of type 6 with an order), which lob.itch also writes at its own price, the cross's."""
    last = (1.12, 6, 3, 10, cents, -1, ([(99, 10)], [(105, 10)]), {"match": 8}) if cross else (
        1.12, 4, 3, 10, cents, -1, ([(99, 10)], [(105, 10)]), {"priced": cents != 104})
    return [
        (0.10, 1, 1, 10, 99, 1, ([(99, 10)], [])),
        (0.11, 1, 2, 10, 105, -1, ([(99, 10)], [(105, 10)])),
        (1.10, 1, 3, 10, 104, -1, ([(99, 10)], [(104, 10), (105, 10)])),
        last,
    ]


def priced_sweep_rows():
    """Bids of 10 at 100 and 99 and an ask at 105; at 2.0 a market sell of 20 takes the bid at 100 (an execution) and
    the bid at 99 by a printable C at 100: a cent above its order's price, as a bid's is on the exploration days.
    Two levels fall, each by a trade."""
    return [
        (1.0, 1, 1, 10, 100, 1, ([(100, 10)], [])),
        (1.1, 1, 2, 10, 99, 1, ([(100, 10), (99, 10)], [])),
        (1.2, 1, 3, 10, 105, -1, ([(100, 10), (99, 10)], [(105, 10)])),
        (2.0, 4, 1, 10, 100, 1, ([(99, 10)], [(105, 10)])),
        (2.0, 4, 2, 10, 100, 1, ([], [(105, 10)]), {"priced": True}),
    ]


def priced_two_levels_rows():
    """Bids of 10 at 100 and 99 and an ask at 105; at 2.0 a market sell of 20 takes both bids, each by a printable C at
    a cent above its order's price (as a bid's is on the exploration days): fills at 101 and 100, from levels 100 and
    99. Two levels fall, each by its own trade."""
    return [
        (1.0, 1, 1, 10, 100, 1, ([(100, 10)], [])),
        (1.1, 1, 2, 10, 99, 1, ([(100, 10), (99, 10)], [])),
        (1.2, 1, 3, 10, 105, -1, ([(100, 10), (99, 10)], [(105, 10)])),
        (2.0, 4, 1, 10, 101, 1, ([(99, 10)], [(105, 10)]), {"priced": True}),
        (2.0, 4, 2, 10, 100, 1, ([], [(105, 10)]), {"priced": True}),
    ]


def partial_priced_rows(cents):
    """A bid at 99 and an ask at 105; at 1.10 s an ask of 10 is added at 104 and at 1.12 s, in the same 100 ms window,
    4 of it are executed, by a printable C at ``cents`` (a cent off its order's level when it is not 104). The book
    keeps 6 at 104, so the window's net change is the 6, at 104, whatever price the execution printed at."""
    return [
        (0.10, 1, 1, 10, 99, 1, ([(99, 10)], [])),
        (0.11, 1, 2, 10, 105, -1, ([(99, 10)], [(105, 10)])),
        (1.10, 1, 3, 10, 104, -1, ([(99, 10)], [(104, 10), (105, 10)])),
        (1.12, 4, 3, 4, cents, -1, ([(99, 10)], [(104, 6), (105, 10)]), {"priced": cents != 104}),
    ]


def placed_before_rows(cents):
    """The frames begin at 1.0 s with a book that already holds an ask of 10 at 104, an order (77) placed before them,
    whose add is no row of the frames. At 1.10 s another order adds 10 at 104 and at 1.12 s, in the same 100 ms window,
    order 77 is executed in full, by a printable C at ``cents`` (a cent off its level when it is not 104): the level
    is back at 10 and the window nets to nothing."""
    return [
        (1.00, 1, 1, 10, 99, 1, ([(99, 10)], [(104, 10), (105, 10)])),
        (1.10, 1, 2, 10, 104, -1, ([(99, 10)], [(104, 20), (105, 10)])),
        (1.12, 4, 77, 10, cents, -1, ([(99, 10)], [(104, 10), (105, 10)]), {"priced": cents != 104}),
    ]


def unrelated_change_rows(cents):
    """A bid at 99 and an ask at 105; an ask of 10 is added at 104 (order 3) at 1.0 s. In the window of 1.2 s another
    order adds 10 at 103 and order 3 is executed in full, by a printable C at ``cents`` (103: its own price is the level
    of the other order's add, a cent off its level). 104 falls by 10 and 103 rises by 10, two changes that no other
    level makes up."""
    return [
        (0.10, 1, 1, 10, 99, 1, ([(99, 10)], [])),
        (0.11, 1, 2, 10, 105, -1, ([(99, 10)], [(105, 10)])),
        (1.00, 1, 3, 10, 104, -1, ([(99, 10)], [(104, 10), (105, 10)])),
        (1.20, 1, 4, 10, 103, -1, ([(99, 10)], [(103, 10), (104, 10), (105, 10)])),
        (1.22, 4, 3, 10, cents, -1, ([(99, 10)], [(103, 10), (105, 10)]), {"priced": cents != 104}),
    ]


def mirrored(rows):
    """The same day with the two sides swapped: a price p becomes 200 - p, a buy a sell, the bids the asks."""
    out = []
    for t, kind, oid, size, cents, direction, (bids, asks), *extra in rows:
        flip = [(200 - p, q) for p, q in asks], [(200 - p, q) for p, q in bids]
        out.append((t, kind, oid, size, 200 - cents, -direction, flip, *extra))
    return out


def cross_execution_rows():
    """A bid and an ask, a cross trade at 1.05, and in a window of its own an order of 4 executed in a cross at 1.5."""
    return [
        (1.00, 1, 1, 10, 100, 1, ([(100, 10)], [])),
        (1.01, 1, 2, 10, 105, -1, ([(100, 10)], [(105, 10)])),
        (1.05, 6, 0, 500, 102, 0, ([(100, 10)], [(105, 10)]), {"match": 7}),
        (1.50, 6, 1, 4, 100, 1, ([(100, 6)], [(105, 10)]), {"match": 8}),
    ]


def batch_rows():
    """A bid at 100 and an ask at 105 at 0.1 s. In the window of 1.1 s a bid of 5 arrives at 101 and is cancelled; in
    the window of 1.2 s the 10 lots at 100 are replaced by another order of 10 (an arrival against a cancel); at 1.3 s
    an ask of 7 arrives at 104 and stays."""
    return [
        (0.10, 1, 1, 10, 100, 1, ([(100, 10)], [])),
        (0.11, 1, 2, 10, 105, -1, ([(100, 10)], [(105, 10)])),
        (1.10, 1, 3, 5, 101, 1, ([(101, 5), (100, 10)], [(105, 10)])),
        (1.12, 3, 3, 5, 101, 1, ([(100, 10)], [(105, 10)])),
        (1.20, 1, 4, 10, 100, 1, ([(100, 20)], [(105, 10)])),
        (1.22, 3, 1, 10, 100, 1, ([(100, 10)], [(105, 10)])),
        (1.30, 1, 5, 7, 104, -1, ([(100, 10)], [(104, 7), (105, 10)])),
    ]


def trade_rows():
    """A bid at 99, asks at 101 and 105 (at 0.1 s); at 1.05 s a buy of 10 takes the level at 101. Before the trade the
    mid is 100, after it 102: a fill at 101 is above the first and below the second."""
    return [
        (0.10, 1, 1, 10, 99, 1, ([(99, 10)], [])),
        (0.12, 1, 2, 10, 101, -1, ([(99, 10)], [(101, 10)])),
        (0.14, 1, 3, 10, 105, -1, ([(99, 10)], [(101, 10), (105, 10)])),
        (1.05, 4, 2, 10, 101, -1, ([(99, 10)], [(105, 10)])),
    ]


def break_rows(same_nanosecond=False):
    """Two rows, a cross trade at 1.05 s, an add after it (at the cross's nanosecond, or a little later), an order
    executed in a cross (which resets nothing) and an add; all in the window of 1.0 s."""
    return [
        (1.00, 1, 1, 10, 100, 1, ([(100, 10)], [])),
        (1.01, 1, 2, 10, 105, -1, ([(100, 10)], [(105, 10)])),
        (1.05, 6, 0, 500, 102, 0, ([(100, 10)], [(105, 10)]), {"match": 7}),
        (1.05 if same_nanosecond else 1.06, 1, 3, 10, 104, -1, ([(100, 10)], [(104, 10), (105, 10)]), {}),
        (1.08, 6, 1, 4, 100, 1, ([(100, 6)], [(104, 10), (105, 10)]), {"match": 8}),
        (1.09, 1, 4, 10, 99, 1, ([(100, 6), (99, 10)], [(104, 10), (105, 10)]), {}),
    ]


class Spy(Study):
    """Records what ``observe`` is given: (kind, quotes before, quotes after, own)."""

    key, headline = "spy", "n"

    def _clear(self):
        self.seen = []

    def observe(self, t, kind, side, price, size, fills, aggressor, before, after, book, own=False):
        self.seen.append((kind, before, after, own))

    def values(self, t=None, book=None):
        return {"n": len(self.seen)}


def seen(rung):
    """What a rung gives, flattened: ("Event", ns, kind, side, price, size, fills, aggressor, before, after) or
    ("Break", ns, reason)."""
    out = []
    for item in rung:
        if isinstance(item, Break):
            out.append(("Break", item.ns, item.reason))
        else:
            out.append(("Event", item.ns, item.kind, item.side, item.price, item.size,
                        tuple((f.price, f.size) for f in item.fills), item.aggressor, item.quotes_before,
                        item.quotes_after))
    return out


def is_trade(e):
    """A trade of a stream of its own: it has fills and holds the book, so its quotes before and after are one
    object."""
    return isinstance(e, Event) and bool(e.fills) and e.quotes_before is e.quotes_after


def states(rung):
    """The touch after each update of the rung (an event that is no trade of a stream of its own), with its stamp."""
    return [(e.ns, e.quotes_after) for e in rung if isinstance(e, Event) and not is_trade(e)]


def trades_of(rung):
    return [e for e in rung if is_trade(e)]


def collapse(items):
    """A run of equal touches kept once."""
    out = []
    for ns, touch in items:
        if not out or out[-1][1] != touch:
            out.append((ns, touch))
    return out


def book_after_each_nanosecond(day, every=False):
    """From the frames alone: the touch after the last row of each nanosecond, a run of equal touches kept once (every
    nanosecond with a row if ``every``). The row of a cross or a halt is not in any group and ends the one before it,
    even if it is at the same nanosecond as the next."""
    rows = day.rows
    reset = np.zeros(day.n, bool)
    reset[day.table.brk_row[[why in ("cross", "halt") for why in day.table.brk_reason]]] = True
    keep = np.flatnonzero(~reset)
    ns, segment = day.ns[keep], np.cumsum(reset)[keep]
    ends = keep[np.flatnonzero(np.r_[(ns[1:] != ns[:-1]) | (segment[1:] != segment[:-1]), True])]
    out = [(int(day.ns[r]), NONE if rows.crossed(r) else rows.quotes(r)) for r in ends.tolist()]
    return out if every else collapse(out)


def reference_changes(day):
    """The reference's events with a replace counted as one change: (before, after, fills) of each."""
    out, held = [], None
    for e in reference(day):
        if isinstance(e, Break):
            if held:
                out.append(held)
                held = None
            continue
        if held:
            if e.replace == 2:
                out.append((held[0], e.quotes_after, ()))
                held = None
                continue
            out.append(held)
            held = None
        if e.replace == 1:
            held = (e.quotes_before, e.quotes_after, e.fills)
        else:
            out.append((e.quotes_before, e.quotes_after, e.fills))
    if held:
        out.append(held)
    return out


def assert_levels_end_on_the_references_touch(day):
    """Each reference event (a replace counted as one) is a run of consecutive updates of ``levels`` that starts at its
    touch before and ends on its touch after, whose fills are its fills, and that chain: each update starts where
    the one before it ended."""
    rung = levels(day)
    wanted = reference_changes(day)
    updates = [e for e in rung if isinstance(e, Event)]
    owners = [o for o in rung.owner.tolist() if o >= 0]
    assert len(owners) == len(updates) and rung.counts["events"] == len(wanted) == max(owners) + 1
    by_event: dict = {}
    for owner, e in zip(owners, updates, strict=True):
        by_event.setdefault(owner, []).append(e)
    assert sorted(by_event) == list(range(len(wanted)))
    for j, (before, after, fills) in enumerate(wanted):
        run_ = by_event[j]
        assert run_[0].quotes_before == before and run_[-1].quotes_after == after
        for a, b in zip(run_, run_[1:], strict=False):
            assert b.quotes_before == a.quotes_after
        assert sorted((f.price, f.size) for e in run_ for f in e.fills) == sorted((f.price, f.size) for f in fills)


def test_the_reference_is_the_displayed_streams_events_and_runs_with_a_replace_as_one_change():
    day = day_of(replace_rows())
    events = list(reference(day))
    plain = list(lobster_events(*make(replace_rows(), levels=3), from_empty=True, hidden_mode="none"))
    assert [(e.t, e.kind, e.quotes_before, e.quotes_after, e.replace) for e in events] == [
        (e.t, e.kind, e.quotes_before, e.quotes_after, e.replace) for e in plain]
    assert [e.replace for e in events] == [0, 0, 0, 0, 1, 2]
    assert reference(day).replaces_as_one and not reference(day, replaces_as_one=False).replaces_as_one
    spy = Spy()
    run_rung(reference(day), [spy])
    deletion, new = spy.seen[-2:]
    assert deletion == (CANCEL, (100, 10, 105, 10), (101, 10, 105, 10), False)     # the touch after the new order
    assert new[3] is True and new[1] is new[2]                                        # the new order changes nothing
    spy = Spy()
    run_rung(reference(day, replaces_as_one=False), [spy])
    assert spy.seen[-2][2] == (99, 10, 105, 10)                                      # a state that never existed
    assert all(not own for *_, own in spy.seen)
    assert reference(day).counts == {"events": 6, "aggressive": 0, "fills": 0, "breaks": 0, "replace_deletions": 1}


def test_the_finest_batch_is_the_book_after_each_nanosecond_and_a_replace_leaves_no_state_between_its_rows():
    rep = {"replace": True}
    frames = make([
        (1.0, 1, 1, 10, 100, 1, ([(100, 10)], [])),
        (2.0, 1, 2, 10, 105, -1, ([(100, 10)], [(105, 10)])),
        (2.0, 1, 3, 10, 106, -1, ([(100, 10)], [(105, 10), (106, 10)])),                    # two rows at one nanosecond
        (3.0, 3, 1, 10, 100, 1, ([], [(105, 10), (106, 10)]), rep),
        (3.0, 1, 4, 10, 101, 1, ([(101, 10)], [(105, 10), (106, 10)]), rep),
    ])
    day = Day(frames[0], frames[1], origin=0.0, from_empty=True)
    finest = states(batches(day, FINEST))
    assert [(ns / 1e9, q) for ns, q in finest] == [(1.0, (100, 10, None, 0)), (2.0, (100, 10, 105, 10)),
                                                    (3.0, (101, 10, 105, 10))]
    assert collapse(finest) == book_after_each_nanosecond(day)
    first = next(e for e in batches(day, FINEST) if isinstance(e, Event))            # from an empty book
    assert first.quotes_before == NONE and first.quotes_before is not first.quotes_after and first.first == 1
    # what a study is given: with the replace as one the state between its rows is in no event, as in no batch
    spy = Spy()
    run_rung(reference(day), [spy])
    assert [after for _, _, after, own in spy.seen if not own] == [
        (100, 10, None, 0), (100, 10, 105, 10), (100, 10, 105, 10), (101, 10, 105, 10)]
    spy = Spy()
    run_rung(reference(day, replaces_as_one=False), [spy])
    assert (None, 0, 105, 10) in [after for _, _, after, _ in spy.seen]
    assert (None, 0, 105, 10) not in [q for _, q in finest]


def test_levels_splits_an_event_one_update_a_changed_level_best_price_first_and_fuses_the_fills():
    day = day_of(sweep_rows(), levels=4)
    rung = levels(day)
    updates = seen(rung)
    assert len(updates) == 10 and rung.counts["split_events"] == 1 and rung.counts["fallback_events"] == 0
    assert rung.counts["trades"] == 3
    assert rung.counts["level_changes"] == 9      # one level an update, but the file's first row has no book before it
    # the sweep: three levels, the best first, each the trade at its price (a market arrival with its own fills)
    sweep = updates[6:9]
    assert [(u[2], u[3], u[4], u[5], u[6], u[7]) for u in sweep] == [
        (ARRIVAL_MARKET, BID, 0, 10, ((101, 10),), 1), (ARRIVAL_MARKET, BID, 0, 10, ((102, 10),), 1),
        (ARRIVAL_MARKET, BID, 0, 5, ((103, 5),), 1)]
    assert [u[8] for u in sweep] == [(99, 10, 101, 10), (99, 10, 102, 10), (99, 10, 103, 10)]    # each starts where
    assert [u[9] for u in sweep] == [(99, 10, 102, 10), (99, 10, 103, 10), (99, 10, 103, 5)]     # the one before ended
    # an event that changes one level is one update, as it is: the deep cancel keeps its kind, price and size
    assert updates[9][2:8] == (CANCEL, ASK, 104, 10, (), 0)
    assert [o for o in rung.owner.tolist() if o >= 0] == [0, 1, 2, 3, 4, 5, 6, 6, 6, 7]     # the reference's events
    # what the consumer holds after each update of the sweep: the levels before it that it has not changed
    events = [e for e in rung if isinstance(e, Event)]
    after = [events[k].book.at(events[k].last).depth_profile(ASK, 4) for k in (6, 7, 8)]
    assert after == [[(102, 10), (103, 10), (104, 10)], [(103, 10), (104, 10)], [(103, 5), (104, 10)]]


def test_a_replace_to_another_price_is_a_rise_and_a_fall_best_price_first_or_two_updates_as_two_events():
    updates = seen(levels(day_of(replace_rows())))
    assert len(updates) == 6                                                    # four adds, then the replace in two
    rise, fall = updates[4:]
    assert rise[2:7] == (ARRIVAL_LIMIT, BID, 101, 10, ()) and rise[8:] == ((100, 10, 105, 10), (101, 10, 105, 10))
    assert fall[2:7] == (CANCEL, BID, 100, 10, ()) and fall[8:] == ((101, 10, 105, 10), (101, 10, 105, 10))
    # as two events of the stream, a deletion and a new order, each is the one level it changes
    rung = levels(day_of(replace_rows()), replaces_as_one=False)
    deletion, add = seen(rung)[4:]
    assert deletion[2:7] == (CANCEL, BID, 100, 10, ()) and deletion[9] == (99, 10, 105, 10)
    assert add[2:7] == (ARRIVAL_LIMIT, BID, 101, 10, ()) and add[9] == (101, 10, 105, 10)
    assert rung.counts["split_events"] == 0 and rung.counts["updates"] == 6


def test_a_rung_made_from_the_book_carries_no_replace_flags_and_says_so_however_it_was_built():
    day = day_of(replace_rows())
    for rung in (levels(day), levels(day, replaces_as_one=False), batches(day, 100), batches(day, FINEST),
                 shifted(day, 100, 50)):
        assert rung.replaces_as_one is False, rung.name
        assert all(e.replace == 0 for e in rung if isinstance(e, Event)), rung.name
    assert reference(day).replaces_as_one is True and reference(day, replaces_as_one=False).replaces_as_one is False


def test_a_trade_is_not_also_a_cancel_the_fall_of_a_level_at_an_execution_is_the_trade():
    day = day_of(trade_rows())
    updates = seen(levels(day))
    assert CANCEL not in [u[2] for u in updates]
    at_trade = [u for u in updates if u[1] == round(1.05e9)]
    assert len(at_trade) == 1                                       # not a cancel of the level and a trade beside it
    assert at_trade[0][2:7] == (ARRIVAL_MARKET, BID, 0, 10, ((101, 10),))
    # in a batch the fall of the level is the book's update and the trade is its own event: neither is a cancel
    rung = batches(day, 100)
    assert {(e.kind, e.side, e.price, e.size, e.aggressor) for e in rung if isinstance(e, Event) and not e.fills} == {
        (ARRIVAL_LIMIT, BID, 0, 0, 0)}                           # a batch update's kind, side and size are placeholders
    book_update = [e for e in rung if isinstance(e, Event) and not e.fills][-1]
    assert book_update.quotes_after == (99, 10, 105, 10) and book_update.fills == ()
    (trade,) = trades_of(rung)
    assert [(f.price, f.size) for f in trade.fills] == [(101, 10)] and trade.aggressor == 1


def test_levels_ends_on_the_references_touch_and_a_split_that_cannot_end_there_keeps_the_event_whole():
    day = day_of(sweep_and_rest_rows())
    rung = levels(day)
    assert rung.counts["fallback_events"] == 1 and rung.counts["split_events"] == 0
    event = list(reference(day))[-1]
    last = seen(rung)[-1]
    assert (last[8], last[9]) == (event.quotes_before, event.quotes_after) == ((99, 10, 101, 10), (103, 5, 104, 10))
    assert last[6] == ((101, 10), (102, 10), (103, 10)) and last[2] == ARRIVAL_LIMIT       # kept whole, as it is
    assert_levels_end_on_the_references_touch(day)
    assert_levels_end_on_the_references_touch(day_of(sweep_rows(), levels=4))
    assert_levels_end_on_the_references_touch(day_of(replace_rows()))


def test_a_batch_is_the_net_change_of_its_levels_and_loses_what_a_net_change_cannot_show():
    day = day_of(batch_rows())
    fine = states(batches(day, FINEST))
    assert len(fine) == 7                                       # every row of the frames changes some level's net size
    rung = batches(day, 100)
    assert [(ns / 1e9, q) for ns, q in states(rung)] == [(0.199999999, (100, 10, 105, 10)),
                                                          (1.399999999, (100, 10, 104, 7))]
    assert rung.counts["windows"] == 4 and rung.counts["empty_net_change"] == 2 and rung.counts["batches"] == 2
    assert rung.counts["level_changes"] == 3 and rung.owner is None          # levels 100 and 105, then 104
    # the bid of 5 that came and went inside one window (a better best price that is never seen), and the order
    # replaced by one of its size (an arrival against a cancel) leave no update, though the reference has four events
    assert [q for ns, q in fine if 1.0e9 < ns < 1.3e9] == [(101, 5, 105, 10), (100, 10, 105, 10), (100, 20, 105, 10),
                                                            (100, 10, 105, 10)]
    assert len(list(reference(day))) == 7
    # a window wide enough to hold the whole day is one update, the net state
    assert [q for _, q in states(batches(day, 10_000))] == [(100, 10, 104, 7)]
    with pytest.raises(ValueError, match="at least a nanosecond"):
        batches(day, 1e-9)


def test_a_batch_is_stamped_a_nanosecond_before_its_windows_end_so_a_whole_second_never_splits_one():
    day = day_of(batch_rows())
    for width in (1, 50, 100, 250, 500):
        assert all((ns + 1) % (width * 10**6) == 0 for ns, _ in states(batches(day, width))), width
    # the sample of unit 2 is the state at 2.0 on every width that divides a second, as the reference's is
    wanted = dict(run_rung(reference(day), [BookImbalance(levels=1)]).samples)
    for width in (1, 100, 500):
        assert dict(run_rung(batches(day, width), [BookImbalance(levels=1)]).samples) == wanted, width


def test_a_trade_sees_the_last_batch_stamped_at_or_before_it_and_a_shift_moves_it_against_the_book():
    day = day_of(trade_rows())
    after, before = (99, 10, 105, 10), (99, 10, 101, 10)
    for shift, quotes in ((0, before), (-100, before), (49, before), (50, after), (100, after), (150, after)):
        rung = shifted(day, 100, shift)
        (trade,) = trades_of(rung)
        assert trade.quotes_before is trade.quotes_after and trade.quotes_after == quotes, shift
        assert trade.ns == round(1.05e9) + shift * 10**6 and trade.aggressor == 1 and trade.side == BID
        accuracy = run_rung(rung, [TradeDelta(side="quote", quote_lag=0)]).studies["delta"].values()["accuracy"]
        assert accuracy == (1.0 if quotes == before else 0.0)          # the fill at 101 against a mid of 100 and 102
    # a shift of 0 is the unshifted rung, every field of every event
    assert seen(shifted(day, 100, 0)) == seen(batches(day, 100))
    assert shifted(day, 100, 0).rows.n == batches(day, 100).rows.n
    # the finest batch holds the trade's own effect, so a trade sees its own batch there
    (trade,) = trades_of(batches(day, FINEST))
    assert trade.quotes_after == after
    # a trade moved to the session's end or past it is gone, and one moved before the origin is stamped at it
    short = day_of(trade_rows(), end=1.1)
    assert len(trades_of(shifted(short, 100, 0))) == 1 and trades_of(shifted(short, 100, 100)) == []
    assert shifted(short, 100, 100).counts["trades_after_the_end"] == 1
    at_the_end = day_of(trade_rows(), end=1.15)                    # the end is the first second not in the session
    assert trades_of(shifted(at_the_end, 100, 99)) != [] and trades_of(shifted(at_the_end, 100, 100)) == []
    early = trades_of(shifted(day, 100, -1_100))
    assert early[0].ns == 0 and early[0].quotes_after == NONE


@pytest.mark.parametrize("same_nanosecond", [False, True])
def test_a_reset_break_ends_a_batch_and_a_cross_execution_does_not(same_nanosecond):
    day = day_of(break_rows(same_nanosecond))
    items = seen(batches(day, 100))
    assert [(i[0], i[2]) for i in items if i[0] == "Break"] == [("Break", "cross"), ("Break", "cross execution")]
    assert [i[0] for i in items] == ["Event", "Break", "Break", "Event"]
    pre, cross, execution, post = items
    assert cross[1] == round(1.05e9) and pre[1] == cross[1] and pre[9] == (100, 10, 105, 10)    # ended by the cross
    assert execution[1] == round(1.08e9)
    # the batch after the cross starts from the book of the cross's row and holds the order executed in the cross
    assert post[8] == (100, 10, 105, 10) and post[9] == (100, 6, 104, 10) and post[1] == round(1.1e9) - 1
    assert states(batches(day, 100_000))[-1][1] == (100, 6, 104, 10) and len(states(batches(day, 100_000))) == 2
    # at the finest the first update after the cross may be at the cross's own nanosecond, and still follows it
    fine = list(batches(day, FINEST))
    at = next(i for i, e in enumerate(fine) if isinstance(e, Break) and e.reason == "cross")
    assert fine[at + 1].quotes_before == (100, 10, 105, 10) and fine[at - 1].quotes_after == (100, 10, 105, 10)
    result = run_rung(batches(day, 100), [Spread()])
    assert result.counts["breaks"] == 2 and result.counts["resets"] == 1
    # a break names the row of the day's frames it is, whichever rung gives it
    for rung in (batches(day, 100), levels(day), reference(day)):
        assert [(b.row, b.reason) for b in rung if isinstance(b, Break)] == [(2, "cross"), (4, "cross execution")]


def test_an_order_executed_in_a_cross_is_a_change_of_the_book_in_the_window_it_stands_alone_in():
    day = day_of(cross_execution_rows())
    for width, stamp in ((100, 1_599_999_999), (FINEST, 1_500_000_000)):
        rung = batches(day, width)
        assert [(ns, q) for ns, q in states(rung)][-1] == (stamp, (100, 6, 105, 10)), width
        assert rung.counts["level_changes"] == 3                                 # 100 and 105 at first, then 100 again
    assert [i[0] for i in seen(batches(day, 100))] == ["Event", "Break", "Break", "Event"]


def test_a_hidden_execution_among_the_messages_is_a_trade_and_changes_no_level_so_its_window_is_no_update():
    # a row of type 5 (a file of LOBSTER's has them; lob.itch's frames do not) leaves the displayed book as it was
    day = day_of([
        (0.10, 1, 1, 10, 99, 1, ([(99, 10)], [])),
        (0.11, 1, 2, 10, 105, -1, ([(99, 10)], [(105, 10)])),
        (1.10, 5, 0, 3, 104, -1, ([(99, 10)], [(105, 10)])),
        (2.10, 1, 3, 10, 98, 1, ([(99, 10), (98, 10)], [(105, 10)])),
    ])
    rung = batches(day, 100)
    assert [rung.counts[k] for k in ("windows", "batches", "empty_net_change", "trades", "level_changes")] == [
        3, 2, 1, 1, 3]
    assert [ns for ns, _ in states(rung)] == [199_999_999, 2_199_999_999]
    (trade,) = trades_of(rung)
    assert [(f.price, f.size) for f in trade.fills] == [(104, 3)] and trade.quotes_after == (99, 10, 105, 10)
    fine = batches(day, FINEST)
    assert [fine.counts[k] for k in ("windows", "batches", "empty_net_change")] == [4, 3, 1]


def test_a_priced_execution_takes_its_shares_from_its_orders_level_in_a_batchs_net_change():
    # the window of 1.1 s holds an add and its execution in full: the book at its end is the book at its start, so it
    # nets to nothing and is no update, whatever price the execution printed at
    plain, priced = batches(day_of(priced_rows(104)), 100), batches(day_of(priced_rows(103)), 100)
    names = ("batches", "empty_net_change", "level_changes")
    assert [plain.counts[k] for k in names] == [1, 1, 2]
    assert [priced.counts[k] for k in names] == [plain.counts[k] for k in names]
    assert states(priced) == states(plain)


def test_an_order_executed_in_a_cross_takes_its_shares_from_its_orders_level_in_a_batchs_net_change():
    plain, crossed = (batches(day_of(priced_rows(cents, cross=True)), 100) for cents in (104, 103))
    names = ("batches", "empty_net_change", "level_changes")
    assert [plain.counts[k] for k in names] == [1, 1, 2]
    assert [crossed.counts[k] for k in names] == [plain.counts[k] for k in names]
    assert states(crossed) == states(plain)


def mirrored_if(flip, rows):
    return mirrored(rows) if flip else rows


@pytest.mark.parametrize("flip", [False, True])
def test_a_partial_priced_execution_takes_its_shares_from_its_orders_level_and_leaves_the_rest_there(flip):
    plain, priced = (batches(day_of(mirrored_if(flip, partial_priced_rows(cents))), 100) for cents in (104, 103))
    names = ("batches", "empty_net_change", "level_changes")
    # the window of 1.1 s adds 10 at the level and takes 4 of them: a net change of 6 at one level
    assert [plain.counts[k] for k in names] == [2, 0, 3]
    assert [priced.counts[k] for k in names] == [plain.counts[k] for k in names]
    assert states(priced) == states(plain)
    assert states(priced)[-1][1] == ((99, 10, 104, 6) if not flip else (96, 6, 101, 10))


@pytest.mark.parametrize("flip", [False, True])
def test_a_priced_execution_of_an_order_placed_before_the_frames_takes_its_shares_from_the_level_the_book_shows(flip):
    days = [Day(*make(mirrored_if(flip, placed_before_rows(cents)), levels=3)[:2], origin=0.0, end=10.0)
            for cents in (104, 103)]
    plain, priced = days
    assert 77 not in priced.order_id[priced.kind == 1]              # the order's add is not among the messages
    assert priced.price[2] != plain.price[2] == priced.level[2]
    names = ("batches", "empty_net_change", "level_changes")
    # the window of 1.1 s adds 10 where 10 of the older order go: the level is as it was, and the window is no update
    assert [batches(plain, 100).counts[k] for k in names] == [1, 1, 1]
    assert [batches(priced, 100).counts[k] for k in names] == [1, 1, 1]
    assert states(batches(priced, 100)) == states(batches(plain, 100))


@pytest.mark.parametrize("flip", [False, True])
def test_a_priced_execution_and_an_unrelated_change_at_its_own_price_do_not_net_in_a_batch(flip):
    plain, priced = (batches(day_of(mirrored_if(flip, unrelated_change_rows(cents))), 100) for cents in (104, 103))
    names = ("batches", "empty_net_change", "level_changes")
    # the window of 1.2 s: one level rises by 10 (another order's add) and the level of the executed order falls by 10;
    # the execution's price is the rising level's, which must not hide either
    assert [plain.counts[k] for k in names] == [3, 0, 5]
    assert [priced.counts[k] for k in names] == [plain.counts[k] for k in names]
    assert states(priced) == states(plain)
    assert states(priced)[-1][1] == ((99, 10, 103, 10) if not flip else (97, 10, 101, 10))


def test_the_level_a_row_changed_is_its_own_price_but_for_a_priced_execution_and_an_order_executed_in_a_cross():
    for flip in (False, True):
        for cross in (False, True):
            plain = day_of(mirrored_if(flip, priced_rows(104, cross=False)))
            day = day_of(mirrored_if(flip, priced_rows(103, cross=cross)))
            assert day.level.tolist() == plain.price.tolist() != day.price.tolist(), (flip, cross)
            assert day.level[:3].tolist() == day.price[:3].tolist()
    # a file with no ``priced`` column (the simulator's, a LOBSTER file) and every row that is none of the two
    for day in (day_of(sweep_rows(), levels=4), day_of(break_rows()), simulated_day("sweeping", 1)):
        assert day.level.tolist() == day.price.tolist()
    # where the book does not show the level, the row keeps its price: the first row of a file that starts from a book
    # it does not show (no book before it), and a level the frames' two levels do not reach (the book does not change)
    first = Day(*make([(1.0, 4, 77, 10, 103, -1, ([(99, 10)], [(105, 10)]), {"priced": True}),
                       (1.1, 1, 1, 10, 104, -1, ([(99, 10)], [(104, 10), (105, 10)]))])[:2], origin=0.0, end=10.0)
    assert first.level.tolist() == first.price.tolist()       # not the level the last row's book has and this lacks
    deep = day_of([
        (1.0, 1, 1, 10, 101, -1, ([], [(101, 10)])),
        (1.1, 1, 2, 10, 102, -1, ([], [(101, 10), (102, 10)])),
        (1.2, 1, 3, 10, 103, -1, ([], [(101, 10), (102, 10)])),
        (1.3, 4, 3, 10, 102, -1, ([], [(101, 10), (102, 10)]), {"priced": True}),
    ], levels=2)
    assert deep.level.tolist() == deep.price.tolist()   # pins the fallback: its level is 103 (its add is a row)


def test_the_units_every_rung_samples_are_the_same_and_the_last_is_the_state_at_the_session_end():
    day = day_of(sweep_rows(), levels=4, end=6.0)
    rungs = [reference(day), levels(day), batches(day, FINEST), batches(day, 100), shifted(day, 100, 50),
             batches(day, 2_000)]
    spread = {}
    for rung in rungs:
        result = run_rung(rung, [Spread()])
        assert [u for u, _ in result.samples] == [1, 2, 3, 4, 5, 6], rung.name
        spread[rung.name] = [row["spread"] for _, row in result.samples]
        assert result.counts["samples"] == 6
    assert spread["reference"] == [None, 2, 4, 4, 4, 4]                  # unit 3 is the book before the cancel
    assert spread["levels"] == spread["reference"] == spread["batches(1e-06 ms)"] == spread["batches(100 ms)"]
    assert spread["batches(2000 ms)"][1:3] == [2, 2]                    # a batch of 2 s: unit 3 is still the old book
    assert spread["batches(2000 ms)"][-1] == 4                         # and the last unit is the state at the end
    # without the final samples a stream that ends early samples fewer units: the units no event opened are missing
    short = run_rung(reference(day), [Spread()], final_samples=False)
    assert [u for u, _ in short.samples] == [1, 2, 3]
    # the rung's counts reach the run, as a stream's do
    assert run_rung(levels(day), [Spread()]).stream_counts["updates"] == 10


def test_a_day_with_an_origin_after_midnight_samples_the_same_units_as_the_same_day_moved_to_midnight():
    # the units are counted from the origin and the last one is at the session's end, whatever the origin is
    rows = sweep_rows()
    early = day_of(rows, levels=4, end=6.0)
    late = day_of([(t + 5.0, *rest) for t, *rest in rows], levels=4, end=11.0, origin=5.0)
    for make_rung in (reference, levels, lambda d: batches(d, 100), lambda d: shifted(d, 100, 50)):
        wanted, result = run_rung(make_rung(early), [Spread()]), run_rung(make_rung(late), [Spread()])
        assert [u for u, _ in wanted.samples] == [1, 2, 3, 4, 5, 6]
        assert result.samples == wanted.samples and result.counts["samples"] == wanted.counts["samples"] == 6


def test_level_changes_counts_a_level_that_is_new_gone_or_of_another_size_inside_the_common_window_only():
    frames = make([
        (1.0, 1, 1, 10, 101, -1, ([], [(101, 10)])),
        (1.1, 1, 2, 10, 102, -1, ([], [(101, 10), (102, 10)])),
        (1.2, 1, 3, 10, 103, -1, ([], [(101, 10), (102, 10), (103, 10)])),
        (1.3, 3, 1, 10, 101, -1, ([], [(102, 10), (103, 10)])),
        (1.4, 2, 2, 4, 102, -1, ([], [(102, 6), (103, 10)])),
    ], levels=3)
    rows = Day(*frames[:2], from_empty=True).rows
    # rows 2 to 3: the first level goes (one change); 3 to 4: 102 is cut by 4 (one); 0 to 4: all three; 2 to 2: none
    assert level_changes(rows, np.array([2, 3, 0, 2]), np.array([3, 4, 4, 2])).tolist() == [1, 1, 3, 0]
    # a change at the window's last level counts, on the ask side and on the bid side
    ends = make([
        (1.0, 1, 1, 10, 101, -1, ([(99, 10)], [(101, 10), (102, 10)])),
        (1.1, 2, 1, 4, 102, -1, ([(99, 10)], [(101, 10), (102, 6)])),
        (1.2, 1, 2, 10, 98, 1, ([(99, 10), (98, 10)], [(101, 10), (102, 6)])),
        (1.3, 2, 2, 4, 98, 1, ([(99, 10), (98, 6)], [(101, 10), (102, 6)])),
    ], levels=2)
    assert level_changes(Day(*ends[:2], from_empty=True).rows, np.array([0, 2]), np.array([1, 3])).tolist() == [1, 1]
    # a level that is new at the window's last place is a change, and the one it pushes out of the window is none
    pushed = make([
        (1.0, 1, 1, 10, 101, -1, ([(99, 10), (97, 10)], [(101, 10), (103, 10)])),
        (1.1, 1, 2, 10, 98, 1, ([(99, 10), (98, 10)], [(101, 10), (103, 10)])),
        (1.2, 1, 3, 10, 102, -1, ([(99, 10), (98, 10)], [(101, 10), (102, 10)])),
    ], levels=2)
    assert level_changes(Day(*pushed[:2], from_empty=True).rows, np.array([0, 1]), np.array([1, 2])).tolist() == [1, 1]
    # a deeper level that enters the window because a better one has gone is no change
    deeper = make([
        (1.0, 1, 1, 10, 101, -1, ([], [(101, 10), (102, 10)])),
        (1.1, 3, 1, 10, 101, -1, ([], [(102, 10), (103, 10)])),
    ], levels=2)
    assert level_changes(Day(*deeper[:2], from_empty=True).rows, np.array([0]), np.array([1])).tolist() == [1]


def test_level_changes_counts_the_same_whichever_size_of_chunk_it_counts_in():
    # the counts are made in chunks of 50,000 pairs (a day has a million events); here the chunks are of one to five
    # pairs. Row 5 is the whole book (four asks, two bids): the rows before it lack levels it has, the rows after it
    # have lost some of them
    day = day_of(sweep_rows(), levels=4)
    r1 = np.arange(1, day.n)
    r0 = np.full(len(r1), 5)
    wanted = [4, 3, 2, 1, 0, 1, 2, 3, 4]
    assert level_changes(day.rows, r0, r1).tolist() == wanted
    for chunk in (1, 2, 3, 5):
        assert level_changes(day.rows, r0, r1, chunk=chunk).tolist() == wanted, chunk


def test_a_sweep_that_ends_on_the_last_level_of_the_window_is_split_on_both_sides_best_price_first():
    day = day_of(limit_rows())
    rung = levels(day)
    assert rung.counts["split_events"] == 2 and rung.counts["fallback_events"] == 0
    updates = seen(rung)
    buy, sell = updates[8:11], updates[11:14]
    assert [(u[6], u[8][2:], u[9][2:]) for u in buy] == [
        (((101, 10),), (101, 10), (102, 10)), (((102, 10),), (102, 10), (103, 10)), (((103, 5),), (103, 10), (103, 5))]
    assert [(u[6], u[7], u[8][:2], u[9][:2]) for u in sell] == [
        (((99, 10),), -1, (99, 10), (98, 10)), (((98, 10),), -1, (98, 10), (97, 10)),
        (((97, 5),), -1, (97, 10), (97, 5))]                       # a seller hits the bids, the best first
    assert_levels_end_on_the_references_touch(day)


@pytest.mark.parametrize("flip", [False, True])
def test_the_state_after_the_last_update_of_a_split_event_is_the_true_one_and_those_before_it_show_fewer_levels(flip):
    rows = mirrored(reveal_rows()) if flip else reveal_rows()
    day = day_of(rows)
    rung = levels(day)
    assert rung.counts["split_events"] == 1 and rung.counts["fallback_events"] == 0 and rung.counts["updates"] == 9
    aggressor, own, hit, rest = (-1, ASK, BID, ASK) if flip else (1, BID, ASK, BID)
    price = lambda p: 200 - p if flip else p                                          # noqa: E731
    updates = seen(rung)
    assert [(u[2], u[3], u[4], u[5], u[6], u[7]) for u in updates[-3:]] == [
        (ARRIVAL_MARKET, own, 0, 10, ((price(101), 10),), aggressor),
        (ARRIVAL_MARKET, own, 0, 10, ((price(102), 10),), aggressor),
        (ARRIVAL_LIMIT, rest, price(102), 5, (), aggressor)]     # the rest of the order is a rise of a level, no trade
    events = [e for e in rung if isinstance(e, Event)]
    depth = [events[k].book.at(events[k].last).depth_profile(hit, 3) for k in (-3, -2, -1)]   # one book, moved
    assert depth == [[(price(p), 10) for p in ps] for ps in ((102, 103), (103,), (103, 104, 105))]
    assert events[-1].book.at(events[-1].last).depth_profile(own, 3) == [(price(102), 5), (price(99), 10)]
    assert_levels_end_on_the_references_touch(day)


@pytest.mark.parametrize("flip", [False, True])
def test_a_fill_at_a_level_the_frames_never_showed_goes_to_the_side_it_hit_and_not_to_the_rest_of_its_own_order(flip):
    rows = mirrored(beyond_rows()) if flip else beyond_rows()
    day = day_of(rows, levels=2)
    rung = levels(day)
    assert rung.counts["split_events"] == 1 and rung.counts["fallback_events"] == 0 and rung.counts["updates"] == 7
    aggressor, own = (-1, ASK) if flip else (1, BID)
    price = lambda p: 200 - p if flip else p                                          # noqa: E731
    # two levels fall, and the order's rest rises at 103: the fill at 103, a level that is in no update of the side it
    # hit, goes to the nearest one that is (102), and not to the rise at its own price
    assert [(u[2], u[3], u[4], u[5], u[6], u[7]) for u in seen(rung)[-3:]] == [
        (ARRIVAL_MARKET, own, 0, 10, ((price(101), 10),), aggressor),
        (ARRIVAL_MARKET, own, 0, 20, ((price(102), 10), (price(103), 10)), aggressor),
        (ARRIVAL_LIMIT, own, price(103), 5, (), aggressor)]
    assert rung.counts["trades"] == 2
    assert_levels_end_on_the_references_touch(day)


def test_a_priced_fill_of_a_sweep_goes_with_the_fall_of_its_own_level_and_leaves_no_cancel_beside_the_trade():
    day = day_of(priced_sweep_rows())
    rung = levels(day)
    assert rung.counts["split_events"] == 1 and rung.counts["fallback_events"] == 0
    sweep = seen(rung)[-2:]
    assert CANCEL not in [u[2] for u in sweep]
    assert [(u[2], u[5], u[6]) for u in sweep] == [(ARRIVAL_MARKET, 10, ((100, 10),))] * 2
    assert_levels_end_on_the_references_touch(day)


@pytest.mark.parametrize("flip", [False, True])
def test_two_priced_fills_of_a_sweep_go_each_with_the_fall_of_the_level_it_took(flip):
    day = day_of(mirrored_if(flip, priced_two_levels_rows()))
    rung = levels(day)
    price = lambda p: 200 - p if flip else p                                          # noqa: E731
    assert rung.counts["split_events"] == 1 and rung.counts["fallback_events"] == 0 and rung.counts["trades"] == 2
    sweep = seen(rung)[-2:]
    # the fill at 101 took the level of 100, the best, and the fill at 100 the level of 99. Each price is the nearest
    # to the level of 100 (101 by one, 100 by none), so by its price both fills would go to it, and the level of 99
    # would fall with no fill: a cancel beside the trades
    assert CANCEL not in [u[2] for u in sweep]
    assert [(u[2], u[5], u[6]) for u in sweep] == [(ARRIVAL_MARKET, 10, ((price(101), 10),)),
                                                   (ARRIVAL_MARKET, 10, ((price(100), 10),))]
    assert [u[9] for u in sweep] == ([(95, 10, 101, 10), (95, 10, None, 0)] if flip
                                     else [(99, 10, 105, 10), (None, 0, 105, 10)])
    assert_levels_end_on_the_references_touch(day)


def test_a_change_deeper_than_the_frames_show_is_an_update_with_an_unchanged_touch():
    day = day_of([
        (1.0, 1, 1, 10, 101, -1, ([], [(101, 10)])),
        (1.1, 1, 2, 10, 102, -1, ([], [(101, 10), (102, 10)])),
        (1.2, 1, 3, 10, 103, -1, ([], [(101, 10), (102, 10), (103, 10)])),
        (1.3, 1, 4, 10, 104, -1, ([], [(101, 10), (102, 10), (103, 10)])),       # the fourth level is out of view
    ])
    (deep,) = [u for u in seen(levels(day)) if u[1] == round(1.3e9)]
    assert deep[2:8] == (ARRIVAL_LIMIT, ASK, 104, 10, (), -1) and deep[8] == deep[9] == (None, 0, 101, 10)
    assert levels(day).counts["level_changes"] == 2          # the two the frames show: the first row has no book before
    fine = batches(day, FINEST)
    assert len(states(fine)) == 4 and fine.counts["level_changes"] == 4     # the batch counts it from the messages


def test_a_sweep_that_empties_a_side_ends_on_the_touch_the_event_leaves_and_is_not_refused():
    asks = [(101, 10), (102, 10)]
    day = day_of([
        (1.0, 1, 1, 10, 101, -1, ([], asks[:1])),
        (1.1, 1, 2, 10, 102, -1, ([], asks)),
        (1.2, 1, 3, 10, 99, 1, ([(99, 10)], asks)),
        (2.0, 4, 1, 10, 101, -1, ([(99, 10)], asks[1:])),
        (2.0, 4, 2, 10, 102, -1, ([(99, 10)], [])),
    ])
    rung = levels(day)
    assert rung.counts["split_events"] == 1 and rung.counts["fallback_events"] == 0
    assert [u[9] for u in seen(rung)[-2:]] == [(99, 10, 102, 10), (99, 10, None, 0)]


def test_an_event_with_no_aggressor_that_changes_both_sides_is_one_update_a_side_that_chain_to_the_true_touch():
    # a replace of the bid at 99 by an ask at 101. lob.itch's replace keeps its side (lob/itch.py 767-769), so no feed
    # gives such an event, and which side comes first is no rule of the rung. That each side is one update, that they
    # chain and that the last ends on the true touch is
    rep = {"replace": True}
    day = day_of([
        (1.0, 1, 1, 10, 99, 1, ([(99, 10)], [])),
        (1.1, 1, 2, 10, 105, -1, ([(99, 10)], [(105, 10)])),
        (2.0, 3, 1, 10, 99, 1, ([], [(105, 10)]), rep),
        (2.0, 1, 3, 10, 101, -1, ([], [(101, 10), (105, 10)]), rep),
    ])
    rung = levels(day)
    assert rung.counts["split_events"] == 1 and rung.counts["fallback_events"] == 0 and rung.counts["updates"] == 4
    pair = seen(rung)[-2:]
    assert {u[2:8] for u in pair} == {(ARRIVAL_LIMIT, ASK, 101, 10, (), -1), (CANCEL, BID, 99, 10, (), 0)}
    assert pair[0][8] == (99, 10, 105, 10) and pair[1][9] == (None, 0, 101, 10) and pair[1][8] == pair[0][9]
    assert_levels_end_on_the_references_touch(day)


def test_a_break_between_the_two_halves_of_a_replace_means_they_are_two_changes_and_a_break_may_be_last():
    rep = {"replace": True}
    rows = [
        (1.0, 1, 1, 10, 99, 1, ([(99, 10)], []), {}),
        (2.0, 3, 1, 10, 99, 1, ([], []), rep),
        (2.5, 6, 0, 500, 102, 0, ([], []), {"match": 7}),
        (3.0, 1, 2, 10, 100, 1, ([(100, 10)], []), rep),
        (4.0, 6, 0, 500, 102, 0, ([(100, 10)], []), {"match": 8}),
    ]
    day = day_of(rows)
    assert [(e.replace, e.quotes_after) for e in reference(day) if isinstance(e, Event)] == [
        (0, (99, 10, None, 0)), (1, NONE), (2, (100, 10, None, 0))]
    assert_levels_end_on_the_references_touch(day)
    rung = levels(day)
    assert rung.counts["updates"] == 3 and rung.counts["events"] == 3 and rung.counts["breaks"] == 2
    assert [type(i).__name__ for i in rung] == ["Event", "Event", "Break", "Event", "Break"]


def test_a_row_at_the_sessions_end_is_not_in_it_and_a_state_with_no_book_yet_has_no_quotes():
    day = day_of(sweep_rows(), levels=4, end=3.0)
    assert day.n == 9 and len(list(reference(day))) == 7 and day.end_ns == round(3.0e9)
    for rung in (reference(day), levels(day), batches(day, 100)):
        assert [e for e in rung if isinstance(e, Event)][-1].quotes_after == (99, 10, 103, 5)
        assert rung.final_book().depth_profile(ASK, 2) == [(103, 5), (104, 10)]
    # a trade shifted before the first state of a file that starts from a book it does not show sees no book
    day = Day(*make(trade_rows())[:2], origin=0.0, end=10.0)
    (early,) = trades_of(shifted(day, 100, -1_100))
    assert early.quotes_after is early.quotes_before and early.quotes_after == NONE
    assert (early.first, early.last) == (0, -1)
    (later,) = trades_of(shifted(day, 100, 100))
    assert later.quotes_after == (99, 10, 105, 10) and later.last >= 0
    # a day of one row has that row's book at its end
    one = Day(*make(trade_rows()[:1])[:2], origin=0.0, end=10.0, from_empty=True)
    assert reference(one).final_book().depth_profile(BID, 1) == [(99, 10)]


def test_prices_are_whole_ticks_or_fractions_as_the_stream_has_them():
    day = day_of([
        (1.0, 1, 1, 10, 101.5, -1, ([], [(101.5, 10)])),
        (1.1, 1, 2, 10, 99, 1, ([(99, 10)], [(101.5, 10)])),
    ])
    updates = seen(levels(day))
    assert [u[4] for u in updates] == [101.5, 99] and [type(u[4]) for u in updates] == [float, int]


def test_a_day_validates_its_frames_as_the_stream_does_and_takes_a_frame_with_no_nanosecond_column():
    frames = make(sweep_rows(), levels=4)
    assert "ns" not in frames[0].columns
    day = Day(*frames[:2], from_empty=True)
    assert day.ns[0] == 1_000_000_000 and day.n == len(frames[0]) and day.origin_ns == round(DAY_START * 1e9)
    assert day.end_ns == round(3.0 * 1e9) + 1                         # the whole frame by default
    assert all(e.ns == round(e.t * 1e9) for e in reference(day))      # the reference runs on the same clock
    with pytest.raises(ValueError, match="one row after each message"):
        Day(frames[0].iloc[:-1], frames[1], from_empty=True)
    with pytest.raises(ValueError, match="message times go backwards"):
        Day(frames[0].iloc[::-1].reset_index(drop=True), frames[1], from_empty=True)


def test_a_file_that_starts_from_a_book_it_does_not_show_has_no_quotes_before_its_first_update():
    frames = make(batch_rows())
    day = Day(*frames[:2], origin=0.0, end=10.0)
    for rung in (reference(day), levels(day), batches(day, 100)):
        first = next(e for e in rung if isinstance(e, Event))
        assert first.quotes_before is first.quotes_after and first.first == 0
        assert first.quotes_after != NONE                       # the book after the first update is its own state
        assert run_rung(rung, [Spread()]).counts["samples"] == 10
    # and a session with no end ends its last window at its last row
    whole = Day(*frames[:2], origin=0.0, from_empty=True)
    assert states(batches(whole, 100))[-1][0] == round(1.3e9) and states(batches(day, 100))[-1][0] == 1_399_999_999


def test_importing_lob_does_not_import_the_ladder():
    code = "import sys, lob, lob.studies, lob.simulation, lob.lobster, lob.feeds; print('lob.ladder' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                          check=True).stdout.strip() == "False"
    assert set(ladder_module.__all__) == {"Day", "Rung", "batches", "level_changes", "levels", "reference", "run_rung",
                                          "shifted"}


@pytest.mark.parametrize("chunk", [1, 3, 5, 10, 11])
def test_a_rung_is_the_same_stream_whichever_size_of_chunk_it_is_read_in(chunk, monkeypatch):
    # a rung is read in chunks of 65,536 items (a day has a million); here they are at and about the 10 items of these
    # rungs, so that the first update after a chunk's end reads the touch the last one before it held
    cases = [(sweep_rows(), 4, (levels, lambda d: batches(d, FINEST), lambda d: shifted(d, 100, 50))),
             (break_rows(), 3, (levels, lambda d: batches(d, 100), lambda d: batches(d, FINEST))),
             (batch_rows(), 3, (levels, lambda d: batches(d, 100), lambda d: batches(d, FINEST)))]
    rungs = [make_rung(day_of(rows, levels=depth)) for rows, depth, makers in cases for make_rung in makers]
    wanted = [seen(rung) for rung in rungs]
    assert max(len(w) for w in wanted) >= 10
    monkeypatch.setattr(ladder_module, "_CHUNK", chunk)
    assert [seen(rung) for rung in rungs] == wanted


# ---- The simulator's stream through every rung ----

def study_set():
    return StudySet([OFI(), CVD(), TradeDelta(side="quote", quote_lag=0), VPIN(side="true"), OrderToTrade(),
                     BookImbalance(levels=3), Spread(), Resilience()], auto_sample=False)


def simulated_day(key, seed, n_events=6_000):
    if key == "sweeping":
        make_model = SweepModel
    else:
        spec = MODEL_SPECS[key]

        def make_model():
            return spec.build(spec.defaults(), 10_000)
    reset_order_id_counter()
    messages, book = simulated_lobster(make_model, n_events, seed)
    return Day(messages, book, from_empty=True)


@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("key", ["sweeping", "smith_zi"])
def test_every_rung_runs_the_studies_on_the_simulators_stream_and_the_finest_batch_equals_the_reference(key, seed):
    day = simulated_day(key, seed)
    ref = reference(day)
    events = [e for e in ref if isinstance(e, Event)]
    fills = [(f.price, f.size) for e in events for f in e.fills]
    assert len(fills) > 200 and any(len(e.fills) > 1 for e in events) == (key == "sweeping")
    # the finest batch: the book the consumer holds after each nanosecond is the frames' book there, and the last
    # event of each nanosecond of the reference ends on it
    fine = batches(day, FINEST)
    last_of_nanosecond = {}
    for e in events:
        last_of_nanosecond[e.ns] = e.quotes_after
    assert collapse(states(fine)) == book_after_each_nanosecond(day) == collapse(sorted(last_of_nanosecond.items()))
    assert [(f.price, f.size) for e in trades_of(fine) for f in e.fills] == fills
    # and the studies take the same unit samples from it as from the reference
    for studies in (lambda: [Spread(), BookImbalance(levels=3)], lambda: [Spread(), BookImbalance(levels=1)]):
        assert run_rung(fine, studies()).samples == run_rung(ref, studies()).samples
    # every rung runs the whole set of studies, on the same units
    wanted = run_rung(ref, study_set())
    for rung in (levels(day), batches(day, 1), batches(day, 50), batches(day, 100), shifted(day, 100, -50),
                 shifted(day, 100, 100)):
        result = run_rung(rung, study_set())
        assert [u for u, _ in result.samples] == [u for u, _ in wanted.samples], rung.name
        assert result.counts["resets"] == 0 and result.studies["vpin"].values() is not None
    # levels: each reference event ends on its touch, and its fills are the fills of its updates
    assert_levels_end_on_the_references_touch(day)
    lv = levels(day)
    assert sorted((f.price, f.size) for e in lv if isinstance(e, Event) for f in e.fills) == sorted(fills)
    assert (lv.counts["split_events"] > 50) == (key == "sweeping")


# ---- The real day ----

real_day = pytest.mark.skipif(not (FRAMES / "CMCSA_messages.parquet").exists(), reason=NO_DAY)
_DAYS: dict = {}


def real(symbol):
    """The day of one exploration stock, built once; the other is let go, so that a test run holds one (about 2 GB)."""
    if symbol not in _DAYS:
        from lob import itch
        _DAYS.clear()
        messages, book, _ = itch.load_frames(FRAMES, symbol)
        _DAYS[symbol] = Day(messages, book, origin=34_200.0, end=57_000.0)
    return _DAYS[symbol]


class Reading(Study):
    """Notebook 14's strict reading of the queue imbalance at the touch (one level, equal weights), over the session:
    after each change of the mid, the first state that leaves the mid alone and whose imbalance is not zero is read
    and scored against the next change of the mid (dropped if it comes later than 30 units), by the spread at the
    reading in ticks (1, 2, 3 or more). It skips the new order of a replace (``own``) and the trades of a stream of
    their own: neither changes the book."""

    key, headline = "reading", "scored"

    def _clear(self):
        self.armed, self.reading, self.last_mid, self.t_change, self.last_move = False, None, None, 0.0, 0
        self.scored = {1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0]}

    def observe(self, t, kind, side, price, size, fills, aggressor, before, after, book, own=False):
        if own or (fills and before is after) or t >= 22_800:
            return
        if after[0] is None or after[2] is None:
            self.last_mid, self.armed, self.reading = None, False, None
            return
        mid = (after[0] + after[2]) / 2.0
        previous, self.last_mid = self.last_mid, mid
        if t < 600:
            return
        changed = previous is not None and mid != previous
        if changed:
            move = 1 if mid > previous else -1
            if self.reading is not None:
                value, spread, t0, last = self.reading
                if t - t0 <= 30.0:
                    cell = self.scored[min(max(round(spread), 1), 3)]
                    cell[0] += 1
                    cell[1] += (value > 0) == (move > 0)
                    cell[2] += move == -last
            self.armed, self.reading, self.t_change, self.last_move = True, None, t, move
        if self.armed and self.reading is None and not changed:
            total = after[1] + after[3]
            value = (after[1] - after[3]) / total if total > 0 else 0.0
            if value:
                self.reading, self.armed = (value, after[2] - after[0], self.t_change, self.last_move), False

    def values(self, t=None, book=None):
        return {"scored": sum(c[0] for c in self.scored.values())}


def reading_counts(rung):
    study = Reading()
    run_rung(rung, [study])
    return {k: tuple(v) for k, v in study.scored.items()}


def reading_on_nanoseconds(day):
    """The reading over the book after the last row of every nanosecond that has a row, whatever its net change."""
    study = Reading()
    for ns, touch in book_after_each_nanosecond(day, every=True):
        study.observe((ns - day.origin_ns) / 1e9, ARRIVAL_LIMIT, BID, 0, 0, (), 0, touch, touch, None)
    return {k: tuple(v) for k, v in study.scored.items()}


def finest_batch_is_the_book_after_each_nanosecond(symbol):
    day = real(symbol)
    fine = batches(day, FINEST)
    wanted = book_after_each_nanosecond(day)
    assert collapse(states(fine)) == wanted and len(wanted) > 100_000
    assert fine.counts["trades"] == len(day.table.fills)
    return fine


@real_day
def test_the_finest_batch_of_nvdas_day_is_the_book_after_each_nanosecond_of_its_frames():
    fine = finest_batch_is_the_book_after_each_nanosecond("NVDA")
    assert fine.counts["windows"] - fine.counts["batches"] == 4                  # nanoseconds with a row, no net change


@real_day
def test_levels_of_nvdas_day_ends_on_the_references_touch_after_every_event_and_counts_what_it_cannot_split():
    day = real("NVDA")
    rung = levels(day)
    assert rung.counts["fallback_events"] == 2 and rung.counts["split_events"] > 10_000
    assert rung.counts["trades"] > 17_000
    assert_levels_end_on_the_references_touch(day)
    # built from the stream with the replace as two events (as the exploration did) it splits 330 events, 329 of them
    # after the opening cross, and falls back on the same 2
    two = levels(day, replaces_as_one=False)
    assert (two.counts["split_events"], two.counts["fallback_events"], two.counts["updates"]) == (330, 2, 996_660)


@real_day
def test_the_finest_batch_of_cmcsas_day_is_the_book_after_each_nanosecond_of_its_frames():
    fine = finest_batch_is_the_book_after_each_nanosecond("CMCSA")
    assert fine.counts["windows"] - fine.counts["batches"] == 1_610


@real_day
def test_a_replace_as_one_gives_the_per_nanosecond_reading_and_as_two_adds_states_that_never_existed():
    day = real("CMCSA")
    as_one = reading_counts(reference(day))
    assert [as_one[k][0] for k in (1, 2, 3)] == [3_488, 2_991, 7]
    assert reading_on_nanoseconds(day) == as_one
    as_two = reading_counts(reference(day, replaces_as_one=False))
    assert [as_two[k][0] for k in (1, 2, 3)] == [3_488, 3_145, 7]           # 154 more, at states that never existed
    # the finest batch has the same states but not the same events: a nanosecond whose rows net to nothing (a replace
    # to the same price and size, an order cancelled as another arrives) is an event of the reference and no update
    # of the batch, and the reading is taken at the first event that leaves the mid alone
    fine = reading_counts(batches(day, FINEST))
    assert [fine[k][0] for k in (1, 2, 3)] == [3_488, 2_983, 7]


@real_day
def test_the_batched_rungs_of_a_real_day_have_the_explorations_windows_readings_and_quote_rule():
    day = real("CMCSA")
    # The count as lob.ladder first built it was 2,785 empty windows at 50 ms, and the right count is 2,786. It keyed
    # each of the day's 451 printable C rows (409 in the session) by the execution's price and not by the level the
    # book fell at, a cent away, so the one window at 50 ms in which another order's add put back, at the level, the
    # shares a priced execution took from it (10:27:06.55, the ask at 40.29, printed at 40.28) did not net to nothing
    # and was counted as an update. The windows, the 100 ms counts and both scored counts are the same either way (a
    # spurious update leaves the book as it was)
    for width, windows, empty, scored in ((50, 126_765, 2_786, 2_412), (100, 101_792, 2_533, 2_301)):
        rung = batches(day, width)
        assert (rung.counts["windows"], rung.counts["empty_net_change"]) == (windows, empty)
        assert sum(c[0] for c in reading_counts(rung).values()) == scored
    cells = reading_counts(batches(day, 100))
    assert (cells[1][0], cells[2][0]) == (2_269, 32)
    accuracy = {}
    for shift in (0, 50, 100):
        accuracy[shift] = run_rung(shifted(day, 100, shift), [TradeDelta(side="quote", quote_lag=0)]
                                   ).studies["delta"].values()["accuracy"]
    assert accuracy[0] > 0.80 > accuracy[50] > 0.55 > accuracy[100]
