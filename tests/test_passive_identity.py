"""With no trader given an execution, a ``PassiveBookMarket`` is a ``BookMarket`` to the last digit (lob/passive.py).

Resting orders live in a new file, and nothing that was there changes: the nine files they could touch are those of
commit b1d2fdd, read out of git. What ``PassiveBookMarket`` adds must also be invisible while no trader has an
execution (or one with a patience of 0): so a market of each class, run in the same state, is compared in every number
a run leaves, as ``tests/test_stage4_identity.py`` does it: the frame, the events and the time, both generators'
states, the flow and the volume, the fair values and their parts, the rates, every resting order in the order the
cancel draw indexes them, each trader's account, positions, states and history, and ``score_holdings``. Order ids come
from a global counter, so it is reset before each run and the number of orders each made is compared as well: an
order more or less anywhere would show.
"""

import dataclasses
import shutil
import subprocess
from functools import cache
from pathlib import Path

import numpy as np
import pytest

import lob.crosssection as cross_now
import lob.order as order_module
import lob.statarb as statarb_now
from lob.deal_traders import DealTrader, NewsAwareTrader
from lob.deals import Deals
from lob.indexarb import IndexTrader
from lob.passive import Execution, PassiveBookMarket, score_execution
from lob.statarb import BookMarket, BookTrader, FactorValues, score_holdings, sector_loadings
from tests.test_passive import adds_up
from tests.test_stage4_identity import BARE_UNITS, KINDS, MARKETS, SPAN, TOGETHER, observed, same

BASELINE = "b1d2fdd"  # the commit before this stage: what every file it does not own must still be, byte for byte
EXECUTION_KEYS = ("paper", "shortfall", "asked_lots", "shortfall_a_lot", "earned", "passive_lots", "crossed_lots",
                  "urgent_lots", "orders_placed", "orders_withdrawn", "orders_crossed", "wait", "markout_1_market",
                  "markout_5_market", "markout_20_market", "markout_1_limit", "markout_5_limit", "markout_20_limit")
EXECUTION_SCORES = {}  # what ``score_execution`` adds to ``score_holdings``' keys, by the arguments of ``played``
ROOT = Path(__file__).resolve().parents[1]
FILES = ("statarb", "crosssection", "book", "matching", "order", "pairs", "deals", "indexarb", "deal_traders")

FOUR = sector_loadings(2, 2)
SIX = sector_loadings(2, 3)
BUSY = dict(rate=0.15, premium=60.0, length=30.0, breaks=0.3, priced=0.4, cool=10.0)  # a deal about every 7 units


def pairs_of(books):
    """Two pairs, books 0 and 1 and books 2 and 3, in a table of this many books."""
    rows = np.zeros((2, books))
    rows[0, :2], rows[1, 2:4] = (1.0, -1.0), (1.0, -1.0)
    return rows


# What the stage 4 test does not have: the markets of an index, of deals and of a trend, and the traders that need them
# (``size`` and ``window`` as the tests of those traders use them, so that each trades inside a few hundred units).
MORE_MARKETS = {
    "an index": dict(table=FOUR, fair=dict(index=True, index_volatility=0.8), seed=6),
    "deals": dict(table=FOUR, fair=dict(deals=Deals(**BUSY)), seed=7),
    "a trend": dict(table=SIX, fair=dict(trend=0.1), seed=8),
    "an index, deals and a trend": dict(
        table=SIX, fair=dict(index=True, index_volatility=0.8, deals=Deals(**BUSY), trend=0.1), seed=2),
}
MORE_KINDS = {
    "index": lambda n: IndexTrader(window=20, entry=0.5, size=4, interval=1.0),
    "index sample, not hedged": lambda n: IndexTrader(window=20, entry=0.5, size=4, interval=1.0, legs=2,
                                                      hedged=False),
    "deals": lambda n: DealTrader(entry=8.0, size=5),
    "news-aware pairs": lambda n: NewsAwareTrader(weights=pairs_of(n), window=10, entry=0.5, size=6),
    "momentum": lambda n: cross_now.MomentumTrader(lookback=5, skip=2, factors=1, pca_window=40, pca_every=10, hold=3,
                                                   interval=2.0),
}
ALL_MARKETS = MARKETS | MORE_MARKETS
BOOKS = {"an index": 5, "deals": 4, "a trend": 6, "an index, deals and a trend": 7}
TOGETHER_MORE = {
    "an index, deals and a trend": ("an index, deals and a trend",
                                    ["index", "deals", "news-aware pairs", "momentum"], 2),
}


def trader_of(kind, market):
    if kind in KINDS:
        return KINDS[kind](statarb_now, cross_now)
    return MORE_KINDS[kind](BOOKS[market])


def make(passive, how, market, kinds, seed=None):
    """The market and its traders (each recording its decisions): a ``BookMarket``, or a ``PassiveBookMarket`` with
    the traders given at construction (``how`` "at construction") or added one by one with a patience of 0 (a trader
    that does not net its orders cannot be given one, and is added with none)."""
    spec = ALL_MARKETS[market]
    fair = FactorValues(spec["table"], **spec["fair"])
    traders = [trader_of(kind, market) for kind in kinds]
    for trader in traders:
        trader.record = True
    seed = spec["seed"] if seed is None else seed
    if not passive:
        return BookMarket(fair, seed=seed, traders=traders, **spec.get("market", {})), traders
    if how == "at construction":
        return PassiveBookMarket(fair, seed=seed, traders=traders, **spec.get("market", {})), traders
    made = PassiveBookMarket(fair, seed=seed, **spec.get("market", {}))
    for trader in traders:
        made.add(trader, Execution(patience=0.0, reprice=None) if trader.netted else None)
    return made, traders


def more_of(market, traders):
    """What ``observed`` leaves out: the index, the deals and the trend, the books' cash, and what each trader did."""
    fair = market.fair
    out = {"premium": fair.premium, "drift": np.array(fair.drift), "gapped": list(fair.gapped),
           "deals": [dataclasses.astuple(deal) for deal in fair.deals],
           "ended": [dataclasses.astuple(deal) for deal in fair.ended], "scales": fair.scales}
    for i, trader in enumerate(traders):
        out[f"trader {i} book_cash"] = trader.book_cash.copy()
        out[f"trader {i} history"] = list(trader.history)
        out[f"trader {i} outcomes"] = [dataclasses.astuple(o) for o in getattr(trader, "outcomes", [])]
        out[f"trader {i} not flat"] = getattr(trader, "not_flat_at_entry", None)
    return out


@cache
def played(passive, how, market, kinds=(), seed=None):
    """The numbers of one run, from a ``BookMarket`` or a ``PassiveBookMarket``. Bare, it runs ``BARE_UNITS``; with
    traders it runs through the warm-up of the slowest and 250 units beyond it, and scores the first of them: with
    ``score_holdings`` in a ``BookMarket`` and with ``score_execution`` in a ``PassiveBookMarket``, of which the keys
    ``score_holdings`` has are compared and the others are kept in ``EXECUTION_SCORES``. The order counter is reset
    before the run and put back after it."""
    saved = order_module._next_order_id
    order_module.reset_order_id_counter()
    try:
        m, traders = make(passive, how, market, kinds, seed)
        assert (type(m) is PassiveBookMarket) == passive
        if not traders:
            out = observed(m, traders, m.run(BARE_UNITS))
        else:
            frame = m.run(max(t.warm_up for t in traders) + 251.0)
            if passive:
                scored = score_execution(m, traders[0], m.time, SPAN)
                assert [key for key in scored if key in EXECUTION_KEYS] == list(EXECUTION_KEYS)  # the sheet's, in order
                EXECUTION_SCORES[(passive, how, market, kinds, seed)] = {key: scored.pop(key) for key in EXECUTION_KEYS}
                score = scored
            else:
                score = score_holdings(m, traders[0], m.time, SPAN)
            out = observed(m, traders, frame, score)
        out |= more_of(m, traders)
        out["orders made"] = order_module._next_order_id
    finally:
        order_module._next_order_id = saved
    return out, m, traders


def assert_as_a_book_market(market, kinds=(), seed=None, how="at construction"):
    """The run of a ``PassiveBookMarket`` is that of a ``BookMarket`` in every number, and was not a dull one."""
    (old, old_market, _), (new, new_market, new_traders) = (
        played(False, "", market, kinds, seed), played(True, how, market, kinds, seed))
    assert type(old_market) is BookMarket and type(new_market) is PassiveBookMarket
    assert old.keys() == new.keys()
    for key in old:
        assert same(old[key], new[key]), key
    assert new["events"] > 10_000 and new["frame"].shape[0] > 100 and new["orders made"] > 1000
    for trader in new_traders:
        assert trader.lots > 20 and trader.entries > 0, type(trader).__name__  # each traded: a dull run proves little
    if new_traders:
        assert new["score"]["lots"] > 0 and new["score"].keys() == old["score"].keys()
        # what score_execution adds, for a trader that crossed all it did: nothing passive, and a shortfall that is
        # what the paper account (what the signal asked, at each decision's mids) says crossing cost
        extra = EXECUTION_SCORES[(True, how, market, kinds, seed)]
        assert extra["crossed_lots"] == new["score"]["lots"] and extra["asked_lots"] > 0
        assert extra["shortfall"] == pytest.approx(extra["paper"] - new["score"]["pnl"], rel=1e-12, abs=1e-9)
        assert extra["shortfall_a_lot"] == pytest.approx(extra["shortfall"] / extra["asked_lots"])
        for key in ("passive_lots", "urgent_lots", "orders_placed", "orders_withdrawn", "orders_crossed"):
            assert extra[key] == 0, key
        for key in ("earned", "wait") + EXECUTION_KEYS[-6:]:
            assert extra[key] == 0.0, key
    return new_market, new_traders


# ---- The files that must not change ----

@pytest.mark.parametrize("name", FILES)
def test_the_files_resting_orders_could_touch_are_those_of_b1d2fdd_byte_for_byte(name):
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    try:
        shown = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASELINE}:lob/{name}.py"], capture_output=True,
                               text=True)
    except OSError:
        pytest.skip("git cannot be run")
    if shown.returncode:
        pytest.skip(f"git cannot show lob/{name}.py at {BASELINE}")
    assert (ROOT / "lob" / f"{name}.py").read_bytes() == shown.stdout.encode(), \
        f"lob/{name}.py is not as it was at {BASELINE}"


# ---- Off is off ----

@pytest.mark.parametrize("market", ALL_MARKETS)
def test_a_passive_market_with_no_trader_is_the_market_of_a_book_market_to_the_tick(market):
    new_market, _ = assert_as_a_book_market(market)
    if market == "at the floor":
        assert new_market.fair.pushed > 0  # the floor was met: the run held something


@pytest.mark.parametrize("kind", list(KINDS) + list(MORE_KINDS))
def test_with_no_execution_each_kind_of_trader_alone_does_what_it_does_in_a_book_market_to_the_tick(kind):
    market = {"index": "an index", "index sample, not hedged": "an index", "deals": "deals",
              "news-aware pairs": "deals", "momentum": "a trend"}.get(kind, "sectors at level 3000")
    assert_as_a_book_market(market, (kind,))


@pytest.mark.parametrize("name", list(TOGETHER) + list(TOGETHER_MORE))
def test_with_no_execution_several_traders_together_do_what_they_do_in_a_book_market_to_the_tick(name):
    market, kinds, seed = (TOGETHER | TOGETHER_MORE)[name]
    new_market, _ = assert_as_a_book_market(market, tuple(kinds), seed)
    assert len(new_market.traders) == len(kinds)


@pytest.mark.parametrize("name", list(TOGETHER) + list(TOGETHER_MORE))
def test_with_a_patience_of_0_several_traders_together_do_what_they_do_in_a_book_market_to_the_tick(name):
    # an execution of patience 0 is no execution: the trader is kept as one with none, and nothing of it is worked
    market, kinds, seed = (TOGETHER | TOGETHER_MORE)[name]
    new_market, new_traders = assert_as_a_book_market(market, tuple(kinds), seed, how="patience 0")
    assert not new_market._works and not any(new_market._working)
    assert all(new_market.account(trader).orders_placed == 0 for trader in new_traders)


@pytest.mark.parametrize("kind", ["pairs", "reversal", "flow of queues", "deals"])
def test_with_a_patience_of_0_a_trader_alone_does_what_it_does_in_a_book_market_to_the_tick(kind):
    market = "deals" if kind == "deals" else "sectors at level 3000"
    assert_as_a_book_market(market, (kind,), how="patience 0")


@dataclasses.dataclass
class Idle(BookTrader):
    """Asks for nothing, at every decision."""

    def decide(self, mids):
        return []


def test_a_trader_with_an_execution_that_needs_nothing_leaves_the_market_as_it_would_be_without_it():
    # it decides, and observes through its view, and rests nothing: so nothing moves, down to the generators
    runs = {}
    for name, trader in (("alone", None), ("idle", Idle(interval=1.0, record=True))):
        order_module.reset_order_id_counter()
        m = PassiveBookMarket(FactorValues(sector_loadings(3, 4)), seed=3)
        if trader is not None:
            m.add(trader, Execution(patience=20.0, reprice=1.0))
        frame = m.run(300.0)
        runs[name] = (observed(m, [], frame), order_module._next_order_id, m, trader)
    alone, idle = runs["alone"][0], runs["idle"][0]
    assert alone.keys() == idle.keys()
    assert idle["columns"][-1] == "value_0" and not idle["frame"][:, -1].any()  # the trader's own column: nothing
    idle["columns"], idle["frame"] = idle["columns"][:-1], idle["frame"][:, :-1]
    for key in alone:
        if key != "traders":
            assert same(alone[key], idle[key]), key
    assert runs["alone"][1] == runs["idle"][1] and idle["events"] > 10_000
    idle_market, idle_trader = runs["idle"][2], runs["idle"][3]
    assert len(idle_trader.history) > 200 and idle_trader.next_time > 299.0  # it did decide
    assert idle_market.account(idle_trader).orders_placed == 0 and not idle_market.account(idle_trader).fills


# ---- What an account of a trader with no execution holds ----

@pytest.mark.parametrize("name", list(TOGETHER) + list(TOGETHER_MORE))
def test_with_no_execution_each_traders_account_adds_up_to_what_it_crossed_and_places_nothing(name):
    market, kinds, seed = (TOGETHER | TOGETHER_MORE)[name]
    _, new_market, traders = played(True, "at construction", market, tuple(kinds), seed)
    for trader in traders:
        account = new_market.account(trader)
        adds_up(new_market, trader)
        assert account.crossed_lots == trader.lots > 20 and account.passive_lots == 0
        assert account.urgent_lots == 0 and account.orders_placed == account.orders_withdrawn == 0
        assert account.spread_earned == 0.0 and not account.waits
        assert {fill[5] for fill in account.fills} == {"cross"}
        assert sum(abs(fill[2]) for fill in account.fills) == trader.lots
        assert account.asked_lots > 0 and len(account.asked) == new_market.n
