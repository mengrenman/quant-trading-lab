"""The dashboard's Stat-arb view: many books and traders of many relations in them (app/statarb.py)."""

import asyncio
import html
import json
import math
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from starlette.testclient import TestClient

from app import main as main_module
from app import statarb as statarb_module
from app.main import app, registry, render_statarb_panels
from app.pairs import MAX_TRADERS, PairsSession, Switch
from app.sessions import COOKIE_NAME
from app.statarb import (COLORS, CROSS_SECTIONAL, LEVEL, MARKET_PARAMS, MAX_SIGNALS, MAX_STEP, MIDS_KEPT, SAMPLES,
                         SIGNAL_SLOTS, TRADER_KINDS, StatArbSession, book_names, loadings_of, market_from, market_lacks,
                         parse_market, relation_text, too_few_books, what_the_tests_see)
from lob.crosssection import FlowTrader, LeadLagTrader, MomentumTrader, ReversalTrader
from lob.deal_traders import DealTrader
from lob.deals import Deals
from lob.indexarb import IndexTrader
from lob.pairs import FLOOR
from lob.sparse_baskets import sparse_baskets
from lob.statarb import (BookMarket, CopulaPortfolio, FactorValues, RelationTrader, best_baskets, johansen_relations,
                         pairs_by_distance, pairs_by_pvalue, sector_loadings)
from tests.test_pairs_view import read_stream as read_pairs_stream

DEFAULTS = {kind: {p.name: p.default for p in spec.params} for kind, spec in TRADER_KINDS.items()}
MARKET = {p.name: p.default for p in MARKET_PARAMS} | {"shape": "sectors"}
QUICK = {
    "pairs": {**DEFAULTS["pairs"], "count": 2, "formation": 60, "window": 20},
    "baskets": {**DEFAULTS["baskets"], "legs": 3, "count": 2, "formation": 200, "window": 20},
    "residuals": {**DEFAULTS["residuals"], "window": 30, "pca_window": 60},
    "copula": {**DEFAULTS["copula"], "pairs": 2, "window": 60, "trading": 20},
    # A PCA window of 40 decisions and a hold of 5: the first order after 41 decisions, some 200 units (leadlag: 41)
    "reversal": {**DEFAULTS["reversal"], "lookback": 5, "hold": 5, "pca_window": 40},
    "momentum": {**DEFAULTS["momentum"], "lookback": 5, "skip": 1, "hold": 5, "pca_window": 40},
    "leadlag": {**DEFAULTS["leadlag"], "hold": 5, "pca_window": 40},
    "flow": {**DEFAULTS["flow"], "hold": 5, "pca_window": 40},
    "index": {**DEFAULTS["index"], "window": 20},  # in a market with an index
    "deals": {**DEFAULTS["deals"]},  # in a market with deals, and it has nothing to wait for
}
# Deals often enough (one in a hundred units, as long as that) that a few hundred units see some, half of them broken.
DEALS = {**MARKET, "deals": True, "deal_rate": 10.0, "deal_length": 100.0, "deal_breaks": 50.0, "deal_priced": 40.0}
# The market each of the two kinds that trade something the default market has not is started in: the other eight
# start in the default one.
KIND_MARKETS = {"index": {**MARKET, "index": True}, "deals": DEALS}
# The kinds whose summaries quote what notebook 15 found: the two it is about, momentum (its trend) and pairs (its
# deals).
NOTEBOOK_15_KINDS = ("pairs", "momentum", "index", "deals")


@pytest.fixture
def client():
    client = TestClient(app)
    client.get("/")
    return client


@pytest.fixture
def statarb(client):
    return registry.get(client.cookies[COOKIE_NAME]).statarb


def form(values):
    """As the page sends them: a switch only when on."""
    return {k: v for k, v in values.items() if v is not False}


def in_a_market_for(client, kind):
    """Apply the market settings a kind trades in, if it needs more than the default market has."""
    if kind in KIND_MARKETS:
        assert client.post("/api/statarb/config", data=form(KIND_MARKETS[kind])).status_code == 200


def read_stream(client, wanted):
    """First payload of each event type on /statarb/events, until every wanted type has arrived."""
    seen = {}
    with client.stream("GET", "/statarb/events") as resp:
        assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/event-stream")
        event = None
        for line in resp.iter_lines():
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: ") and event:
                seen.setdefault(event, line[len("data: "):])
                event = None
                if wanted <= set(seen):
                    break
    return seen


@pytest.mark.parametrize("session_class, event, read, panels", [
    (PairsSession, "pairs_board", read_pairs_stream, "render_pairs_panels"),
    (StatArbSession, "statarb_board", read_stream, "render_statarb_panels"),
], ids=["pairs", "statarb"])
def test_a_stream_reads_the_market_on_the_event_loop_and_renders_what_it_read_on_a_worker(
        client, monkeypatch, session_class, event, read, panels):
    """The loop steps the market, so what the panels show of it is read there (a worker thread reading it as it is
    stepped can find a deque or a book changed under it); only the rendering is left to a thread."""
    def on_a_loop():
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        return True

    where = {}
    context, render = session_class.context, getattr(main_module, panels)

    def spy_context(self):
        where.setdefault("read", on_a_loop())
        return context(self)

    def spy_render(ctx):
        where.setdefault("rendered", on_a_loop())
        return render(ctx)

    monkeypatch.setattr(session_class, "context", spy_context)
    monkeypatch.setattr(main_module, panels, spy_render)
    assert event in read(client, {event})
    assert where == {"read": True, "rendered": False}


# ---- The market ----

def test_the_view_runs_a_market_of_its_own(client, statarb):
    m = registry.get(client.cookies[COOKIE_NAME])
    page = client.get("/").text
    assert 'id="view-statarb"' in page and 'id="tab-statarb"' in page and "js/lob-statarb.js" in page
    assert len(statarb.samples) == 1 and statarb.samples[0]["t"] == 0
    before, pairs_time = m.sim_time, m.pairs.market.time
    reply = client.post("/api/statarb/control", data={"action": "step", "units": 120}).json()
    assert reply == {"running": False, "time": pytest.approx(statarb.market.time)} and 120 <= statarb.market.time < 121
    assert m.sim_time == before and m.pairs.market.time == pairs_time  # the other two markets stood still
    assert [s["t"] for s in statarb.samples] == list(range(121))  # one sample a unit of time
    assert len(statarb.samples[-1]["mids"]) == 12 and len(statarb.mids) == 121
    assert statarb.samples[-1]["mids"] == [round(float(x), 1) for x in statarb.mids[-1]]  # half a tick shows
    assert any(x % 1 for sample in statarb.samples for x in sample["mids"])


def test_the_settings_make_the_market_they_say():
    session = StatArbSession()
    fair = session.market.fair
    assert fair.n == 12 and fair.rank == 9 and (fair.level == LEVEL).all() and fair.half_life == 50.0
    assert session.names[:5] == ["A1", "A2", "A3", "A4", "B1"] and session.groups()[3:5] == [0, 1]
    session.configure({**MARKET, "shape": "mixed", "books": 7, "factors": 2, "table_seed": 5, "volatility": 2.0,
                       "own_volatility": 1.5, "impact": 0.2, "reverting": False, "drifting": True, "loading_sd": 0.3,
                       "loading_half_life": 500.0})
    fair = session.market.fair
    assert fair.n == 7 and fair.k == 2 and fair.rank == 5 and fair.half_life is None and fair.loading_sd == 0.3
    assert fair.own_volatility == 1.5 and fair.impact == 0.2 and fair.loading_half_life == 500.0
    assert (fair.volatility == 2.0).all() and session.names == ["1", "2", "3", "4", "5", "6", "7"]
    assert session.groups() == [0] * 7
    assert (loadings_of({**MARKET, "shape": "mixed", "books": 7, "factors": 2, "table_seed": 5})
            == fair.mean_loadings).all()


def test_a_half_life_only_counts_while_the_own_parts_revert():
    session = StatArbSession()
    session.configure({**MARKET, "half_life": 30.0})
    assert session.market.fair.half_life == 30.0
    session.configure({**MARKET, "half_life": 30.0, "reverting": False})
    assert session.market.fair.half_life is None


def test_a_reset_starts_the_market_and_every_trader_over():
    session = StatArbSession()
    runner, _ = session.start_trader({**QUICK["pairs"], "formation": 20, "window": 10}, "pairs")
    while runner.trader.fits == 0:  # choosing happens elsewhere, so it takes as long as it takes
        session.run_for(100)
    old, factors = runner.trader, session.market.fair.factors.copy()
    session.reset()
    assert runner.trader is not old and runner.trader.fits == 0 and session.market.traders == [runner.trader]
    assert runner.trader.position.tolist() == [0] * 12 and session.version == 2 and session.market.time == 0
    session.reset()
    assert session.market.fair.factors.tolist() == [0.0, 0.0, 0.0, 0.0] and factors.any()


def test_the_same_seed_gives_the_same_factors_with_a_trader_or_without():
    alone, busy = StatArbSession(), StatArbSession()
    busy.start_trader(QUICK["residuals"], "residuals")
    alone.run_for(400)
    busy.run_for(400)
    alone.market.fair.advance_to(400.0)
    busy.market.fair.advance_to(400.0)
    assert (alone.market.fair.factors == busy.market.fair.factors).all()


def test_the_seed_in_the_settings_is_the_markets():
    def factors(seed):
        session = StatArbSession()
        session.configure({**MARKET, "seed": seed})
        session.run_for(200)
        session.market.fair.advance_to(200.0)
        return session.market.fair.factors

    assert (factors(7) == factors(7)).all() and not (factors(7) == factors(8)).all()


def test_the_clock_starts_and_stops_and_the_page_hears_of_it():
    async def main():
        session = StatArbSession()
        await session.stop()  # nothing to stop yet
        assert not session.is_running
        changes = session.changes
        await session.start()
        assert session.is_running and session._task is not None and session.changes != changes
        await asyncio.sleep(0.2)
        assert session.market.time > 0
        changes = session.changes
        await session.stop()
        assert not session.is_running and session._task is None and session.changes != changes
        stopped_at = session.market.time
        await asyncio.sleep(0.1)
        assert session.market.time == stopped_at

    asyncio.run(main())


def test_the_clock_takes_up_from_wherever_a_step_left_the_market():
    async def main():
        session = StatArbSession()
        await session.start()
        await session.advance(500)  # far past where the clock had got to
        moved_to = session.market.time
        await asyncio.sleep(0.3)  # at 50 units a second
        now = session.market.time
        await session.stop()
        assert now > moved_to + 3  # it runs on from there, without waiting for its target to catch up

    asyncio.run(main())


def test_a_run_ended_by_an_error_shows_it_paused():
    async def main():
        session = StatArbSession()

        def broken():
            raise RuntimeError("broken")

        session._event = broken
        changes = session.changes
        await session.start()
        task = session._task
        await asyncio.wait({task})
        assert isinstance(task.exception(), RuntimeError)
        assert not session.is_running and session.changes != changes + 1  # one for starting, one for this

    asyncio.run(main())


def test_a_reset_keeps_the_mids_of_the_new_market_only():
    session = StatArbSession()
    session.run_for(50)
    session.reset()
    assert len(session.mids) == 1 and len(session.samples) == 1 and session.test is None


def test_the_past_kept_is_bounded():
    session = StatArbSession()
    assert session.mids.maxlen == MIDS_KEPT and session.samples.maxlen == SAMPLES


@pytest.mark.parametrize("change, name, message", [
    ({"sectors": 9}, "sectors", "between"), ({"books_each": 1}, "books_each", "between"),
    ({"books": 2}, "books", "between"), ({"half_life": 0}, "half_life", "between"),
    ({"volatility": "x"}, "volatility", "number"), ({"shape": "cube"}, "shape", "No such kind"),
    ({"shape": "mixed", "books": 3, "factors": 3}, "factors", "Fewer factors"),
    ({"seed": -1}, "seed", "between"),
    # an index, deals and a trend: a chance of 100 percent is a hazard of 1 over 0, and no rate is 0 or below
    ({"deal_priced": 100}, "deal_priced", "between"), ({"deals": True, "deal_breaks": 100}, "deal_breaks", "between"),
    ({"deal_priced": 96}, "deal_priced", "between"), ({"deal_breaks": -5}, "deal_breaks", "between"),
    ({"deal_rate": 0}, "deal_rate", "between"), ({"deal_premium": 0}, "deal_premium", "between"),
    ({"deal_length": 10}, "deal_length", "between"), ({"deal_length": "long"}, "deal_length", "number"),
    ({"trend": 0.06}, "trend", "between"), ({"trend": -0.01}, "trend", "between"),
    ({"index_volatility": 3.5}, "index_volatility", "between"),
    ({"index_volatility": "x"}, "index_volatility", "number"),
    ({"shape": "mixed", "books": 5, "factors": 1, "trend": 0.02}, "trend", "two factors or more"),  # no second factor
])
def test_settings_that_cannot_work_are_refused_with_the_field_at_fault(client, statarb, change, name, message):
    version = statarb.version
    reply = client.post("/api/statarb/config", data=form({**MARKET, **change}))
    assert reply.status_code == 422 and message in reply.json()["errors"][name] and statarb.version == version


def test_new_settings_apply_through_the_route(client, statarb):
    reply = client.post("/api/statarb/config", data=form({**MARKET, "shape": "mixed", "books": 6, "factors": 2}))
    assert reply.status_code == 200 and "6 books, 4 relations" in reply.json()["message"]
    assert statarb.market.n == 6 and statarb.settings["shape"] == "mixed" and "from time 0" in reply.json()["message"]


def test_a_market_left_running_stays_above_the_floor(monkeypatch):
    monkeypatch.setattr(statarb_module, "LEVEL", 11.0)  # just above it, so that a market gets there in moments
    session = StatArbSession()
    session.configure({**MARKET, "volatility": 5.0})
    session.run_for(60)
    assert session.market.fair.pushed > 0 and all(b.mid_price is not None for b in session.market.books)
    assert min(session.market.fair.fairs) >= FLOOR - 2
    status = render_statarb_panels(session.context())["statarb_status"]
    assert "data-statarb-floor" in status and f"{session.market.fair.pushed:,.1f}" in status


# ---- Traders ----

@pytest.mark.parametrize("kind", TRADER_KINDS)
def test_every_kind_starts_trades_and_shows(client, statarb, kind):
    """The two kinds that trade an index or deals run in a market that has one; the rows a deal trader shows come and
    go with its deals, so what it shows in a row is for its own test (further down), as its message is."""
    in_a_market_for(client, kind)
    reply = client.post("/api/statarb/traders/start", data=form({"kind": kind, **QUICK[kind]}))
    assert reply.status_code == 200 and reply.json()["name"] == f"{TRADER_KINDS[kind].label} 1"
    message = reply.json()["message"]
    if kind == "flow":  # it takes in the mids for its factors, and has no flow of the past to learn from
        assert "order flow only from now on" in message and "learned" not in message
    elif kind == "deals":  # it has none to learn from, and nothing to choose
        assert message == "Deals 1 has nothing to learn from the past: it trades from its next decision."
    else:
        assert "watching" in message or "learned" in message
    runner = statarb.runners[0]
    assert runner.kind == kind and statarb.market.traders == [runner.trader]
    for _ in range(10):
        client.post("/api/statarb/control", data={"action": "step", "units": 300})
        if runner.trader.lots:
            break
        choosing = getattr(runner.trader, "_pending", None)
        if choosing is not None:  # it happens elsewhere, and a market stepped flat out can starve it for many seconds
            choosing.result()
    t = runner.trader
    assert t.lots > 0 and t.warming is None
    row = statarb.board()[0]
    assert row["pnl"] == pytest.approx(row["before"] - row["cost"]) and row["lots"] == t.lots
    assert row["describe"] and row["gross"] == np.abs(t.position).sum()
    sample = statarb.samples[-1]
    assert sample["pnl"][runner.id] == pytest.approx(t.value(statarb.market.marks()), abs=50)
    assert runner.id in sample["sig"] and sample["gross"][runner.id] >= 0
    panels = render_statarb_panels(statarb.context())
    assert f'data-statarb-id="{runner.id}"' in panels["statarb_board"]
    if kind != "deals":  # a deal trader has a row only while a deal is on
        assert row["relations"] and "data-statarb-relations" in panels["statarb_board"]
    assert panels["statarb_books"].count("data-statarb-book=") == statarb.market.n == (13 if kind == "index" else 12)


@pytest.mark.parametrize("kind", CROSS_SECTIONAL)
def test_a_trader_that_scores_every_book_shows_a_line_and_a_relation_to_each_and_no_levels(client, statarb, kind):
    client.post("/api/statarb/traders/start", data=form({"kind": kind, **QUICK[kind]}))
    runner = statarb.runners[0]
    t = runner.trader
    traders = statarb.chart_traders()
    assert traders[0]["kind"] == kind and traders[0]["entry"] == 0 and traders[0]["exit"] == 0  # none to draw
    relations = statarb.board()[0]["relations"]  # each book is a relation of its own, flat until it has a score
    assert [r["text"] for r in relations] == statarb.names and [r["state"] for r in relations] == [0] * 12
    assert [r["signal"] for r in relations] == [None] * 12 and runner.id not in statarb.samples[-1]["sig"]
    while not t.lots:
        statarb.run_for(100)
    assert t.warming is None and len(statarb.names) <= MAX_SIGNALS  # a line each: twelve books
    signal = t.signal()
    assert statarb.samples[-1]["sig"][runner.id] == [round(float(x), 3) for x in signal]  # a line a book
    row = statarb.board()[0]
    assert [r["text"] for r in row["relations"]] == statarb.names
    assert [r["state"] for r in row["relations"]] == t.states.tolist()  # the sign of what it wants of each
    assert [r["signal"] for r in row["relations"]] == pytest.approx([float(x) for x in signal])
    assert row["holding"] == np.count_nonzero(t.states) > 0
    board = render_statarb_panels(statarb.context())["statarb_board"]
    assert board.count("data-statarb-relations") == 1 and f"holding {row['holding']} of 12" in board
    assert board.count('<span class="px-1 rounded border') == 12
    assert 'title="Flat, with a signal of none yet"' not in board


def test_the_board_counts_what_it_holds_against_all_the_books_though_it_lists_only_so_many(client, statarb):
    sixteen = {**MARKET, "sectors": 4, "books_each": 4}
    assert client.post("/api/statarb/config", data=form(sixteen)).status_code == 200
    client.post("/api/statarb/traders/start", data=form({"kind": "reversal", **QUICK["reversal"], "size": 20.0}))
    t = statarb.runners[0].trader
    while np.count_nonzero(t.states) <= MAX_SIGNALS:  # more held than the board lists
        statarb.run_for(100)
        assert statarb.market.time < 2_000
    row = statarb.board()[0]
    assert row["total"] == 16 and len(row["relations"]) == MAX_SIGNALS
    assert row["holding"] == np.count_nonzero(t.states) > MAX_SIGNALS
    board = render_statarb_panels(statarb.context())["statarb_board"]
    assert f"holding {row['holding']} of 16" in board
    assert board.count('<span class="px-1 rounded border') == MAX_SIGNALS == 12  # of sixteen, as it always was


def a_reversal_trader_with_lots(client, statarb, settings):
    """A reversal trader started in a market of these settings, run until it holds something; its runner."""
    assert client.post("/api/statarb/config", data=form(settings)).status_code == 200
    runner, _ = statarb.start_trader(QUICK["reversal"], "reversal")
    while not runner.trader.lots:
        statarb.run_for(100)
        assert statarb.market.time < 3_000
    return runner


def test_sixteen_books_without_an_index_keep_the_twelve_lines_and_chips_they_always_had(client, statarb):
    """Stage 4 had made it thirteen (twelve books and an index) in every market: with the new settings off the view is
    what it was, so a market of sixteen shows twelve of them."""
    runner = a_reversal_trader_with_lots(client, statarb, {**MARKET, "sectors": 4, "books_each": 4})
    assert statarb.market.n == 16 and statarb.signal_cap() == MAX_SIGNALS == 12
    assert len(statarb.samples[-1]["sig"][runner.id]) == 12  # a line each, of sixteen books
    row = statarb.board()[0]
    assert row["total"] == 16 and len(row["relations"]) == 12
    board = render_statarb_panels(statarb.context())["statarb_board"]
    assert board.count('<span class="px-1 rounded border') == 12 and "of 16" in board


def test_sixteen_books_and_an_index_keep_a_line_and_a_chip_for_all_seventeen_and_the_index_is_the_last(client, statarb):
    """The index, last, is not cut off: with one the lines and chips are of every book, and the chart has a slot for
    each (the most books a market has: four sectors of four, and an index)."""
    sixteen = {**MARKET, "sectors": 4, "books_each": 4, "index": True}
    runner = a_reversal_trader_with_lots(client, statarb, sixteen)
    assert statarb.market.n == 17 == len(statarb.names) == statarb.signal_cap() == SIGNAL_SLOTS
    assert SIGNAL_SLOTS == len(book_names(sixteen)) > MAX_SIGNALS
    assert len(statarb.samples[-1]["sig"][runner.id]) == 17
    row = statarb.board()[0]
    assert row["total"] == 17 and len(row["relations"]) == 17 and row["relations"][-1]["text"] == "IDX"
    board = render_statarb_panels(statarb.context())["statarb_board"]
    assert board.count('<span class="px-1 rounded border') == 17 and "of 17" in board
    assert f'data-signals="{SIGNAL_SLOTS}"' in client.get("/").text  # a series for each, on the chart


def test_no_market_with_an_index_has_more_books_than_the_chart_has_slots_for():
    for shape in ({"shape": "sectors", "sectors": 4, "books_each": 4}, {"shape": "mixed", "books": 12, "factors": 3}):
        assert len(book_names({**MARKET, **shape, "index": True})) <= SIGNAL_SLOTS
    assert max(p.high for p in MARKET_PARAMS if p.name == "sectors") * max(
        p.high for p in MARKET_PARAMS if p.name == "books_each") + 1 == SIGNAL_SLOTS


def test_a_pair_of_a_book_and_the_index_trades_the_book_alone_at_a_small_size_and_the_page_says_so(client):
    """Such a pair is chosen in a market with an index (in the mixed one, by both methods): its leg of the index is
    about a twelfth of the book's, and whole lots round it to none. The library's rounding, not the view's; the
    summary of Pairs and the guide say it, where a user turning the index on reads of it."""
    spec = TRADER_KINDS["pairs"]
    trader = spec.build({**DEFAULTS["pairs"], "size": 5.0})
    trader.attach(13)
    trader.weights = np.zeros((1, 13))
    trader.weights[0, 6], trader.weights[0, 12] = 1.0, -0.088  # 7 - 0.09 IDX, as a mixed market chose it
    trader.states = np.zeros(1, dtype=int)
    (move,) = trader._become(np.array([1]))
    assert move[6] == 5 and move[12] == 0 and np.count_nonzero(move) == 1  # five lots of the book, none of the index
    page = html.unescape(client.get("/").text)
    assert "that rounds to none, and the pair then trades its book alone" in spec.summary
    assert page.count("rounds to none") == 2  # in Pairs' fieldset and in the guide
    guide = page[page.index("<span class=\"text-slate-200\">Stat-arb</span>"):]
    assert "a pair of a book and IDX has a leg of about a twelfth of a lot for each lot of the book" in guide


def test_with_the_new_settings_off_the_panels_and_the_pairs_page_are_what_they_were_to_the_space(client, statarb):
    """The new blocks of the templates make nothing, not even a space, while they are off (the full comparison with
    42ae383 is in tests/test_stage4_identity.py for the market; this is for the markup)."""
    # The status panel: the template without its block for deals renders the same.
    path = Path(main_module.__file__).parent / "templates" / "components" / "statarb_status.html"
    source = path.read_text()
    start, end = source.index("\n    {%- if p.deals %}"), source.index("{% endif %}", source.index("{%- if p.deals %}"))
    old = source[:start] + source[end + len("{% endif %}"):]
    assert "p.deals" not in old
    statarb.run_for(30)
    ctx = statarb.context()
    assert ctx["deals"] is None
    assert render_statarb_panels(ctx)["statarb_status"] == main_module.templates.env.from_string(old).render(p=ctx)
    # The chips of a trader of every book: one separator between them, as there was.
    client.post("/api/statarb/traders/start", data=form({"kind": "flow", **QUICK["flow"]}))
    board = render_statarb_panels(statarb.context())["statarb_board"]
    gaps = re.findall(r'</span>(\s*)<span class="px-1 rounded border', board)
    assert len(gaps) == 11 and set(gaps) == {"\n" + " " * 28 + "\n" + " " * 28}
    assert re.findall(r"</span>(\s*)</div>\s*</td>", board) == ["\n" + " " * 28 + "\n" + " " * 24]
    # The Pairs page, which has no kind that is not fitted: no wrapper for its fit controls, one space in a fieldset.
    page = client.get("/").text
    pairs = page[page.index('id="pairs-fit-panel"'):page.index('id="pairs-fit-count"')]
    assert "data-pairs-fit-controls" not in page and "data-no-fit" not in pairs
    assert re.findall(r"<fieldset[^>]*>", pairs) == ['<fieldset data-pairs-fit-kind="zscore" >',
                                                     '<fieldset data-pairs-fit-kind="kalman" hidden>',
                                                     '<fieldset data-pairs-fit-kind="copula" hidden>']
    stat = page[page.index('id="statarb-fit-panel"'):page.index('id="statarb-fit-count"')]
    assert "data-statarb-fit-controls" in stat and stat.count("data-no-fit ") == 1  # its deals kind, and so the wrapper
    assert '<fieldset data-statarb-fit-kind="pairs" >' in stat
    assert 'data-statarb-fit-kind="deals" data-no-fit hidden>' in stat


def test_a_relation_chip_shows_its_score_in_two_figures_whatever_the_size_of_the_scores(client, statarb):
    client.post("/api/statarb/traders/start", data=form({"kind": "leadlag", **QUICK["leadlag"]}))
    scores = [0.031, -0.012, 0.049, -0.104, 0.0004, 0.5, -1.26, 11.94, -2.91, 0.003, -0.0449, 0.2]
    statarb.runners[0].trader._scores = np.array(scores)
    board = render_statarb_panels(statarb.context())["statarb_board"]
    chips = re.findall(r'<span class="text-slate-500">([^<]*)</span></span>', board)
    # a lead-lag score of 0.03 is not "+0.0"; a sign is shown only on a figure
    assert chips == ["+0.03", "-0.01", "+0.05", "-0.10", "0.00", "+0.50", "-1.3", "+11.9", "-2.9", "0.00", "-0.04",
                     "+0.20"]


@pytest.mark.parametrize("measure", ["queues", "trades"])
def test_a_flow_trader_with_no_mids_learned_counts_up_to_one_total_on_the_board(statarb, measure):
    runner, _ = statarb.start_trader({**QUICK["flow"], "measure": measure, "warm": False}, "flow")
    assert runner.trader.warming == (0, 41)  # not (0, 1) or (0, 6) first: the factors need 41 mids
    assert "learning 0/41" in render_statarb_panels(statarb.context())["statarb_board"]
    totals = set()
    while runner.trader.warming:
        totals.add(runner.trader.warming[1])
        statarb.run_for(5)
    assert totals == {41}


def test_a_trader_learns_the_latest_mids_one_every_interval_oldest_first():
    session = StatArbSession()
    session.run_for(100)
    runner, learned = session.start_trader({**QUICK["pairs"], "formation": 10, "window": 5, "interval": 5.0}, "pairs")
    assert learned == 10
    assert [m.tolist() for m in runner.trader._mids] == [m.tolist() for m in list(session.mids)[-46::5]]


def test_the_settings_make_the_trader_they_say():
    def build(kind, **change):
        return TRADER_KINDS[kind].build({**DEFAULTS[kind], **change})

    by_distance, by_pvalue = build("pairs", count=3), build("pairs", count=3, by_pvalue=True)
    assert by_distance.choose.func is pairs_by_distance and by_distance.choose.keywords["count"] == 3
    assert by_pvalue.choose.func is pairs_by_pvalue and by_pvalue.choose.keywords["count"] == 3
    few, every_book = build("baskets", legs=4, count=2), build("baskets", legs=0, count=2)
    assert few.choose.func is best_baskets and few.choose.keywords["legs"] == 4 and few.choose.keywords["count"] == 2
    assert every_book.choose.func is johansen_relations and every_book.choose.keywords["count"] == 2
    for t in (by_pvalue, few, every_book):
        assert t.choose.keywords["every"] == 1  # what it is given is already one row a decision
    z = {"formation": 300, "window": 50, "entry": 1.5, "exit": 0.5, "refit": 100, "size": 2.0, "interval": 3.0}
    for t in (build("pairs", **z), build("baskets", **z)):
        assert (t.formation, t.window, t.entry, t.exit, t.refit, t.size, t.interval) == tuple(z.values())
    assert build("pairs", refit=0).refit is None and build("baskets", refit=0).refit is None
    r = {"factors": 4, "window": 50, "pca_window": 100, "open_at": 2.0, "close_long": 0.25, "close_short": 0.5,
         "size": 2.0, "interval": 3.0}
    t = build("residuals", **r)
    assert ((t.factors, t.window, t.pca_window, t.open_at, t.close_long, t.close_short, t.size, t.interval)
            == tuple(r.values()))
    c = {"pairs": 3, "window": 100, "trading": 50, "entry": 2.0, "exit": 0.5, "size": 1.5, "interval": 2.0}
    t = build("copula", **c)
    assert (t.pairs, t.window, t.trading, t.entry, t.exit, t.size, t.interval) == tuple(c.values())
    x = {"lookback": 7, "hold": 9, "factors": 2, "pca_window": 80, "size": 2.0, "interval": 3.0}
    for kind, extra, cls in (("reversal", {}, ReversalTrader), ("momentum", {"skip": 2}, MomentumTrader),
                             ("leadlag", {"ridge": 0.5}, LeadLagTrader), ("flow", {"measure": "trades"}, FlowTrader)):
        t = build(kind, **x, **extra)
        assert type(t) is cls and t.lookback == (1 if kind == "leadlag" else 7)  # a lead-lag score has no lookback
        assert (t.hold, t.factors, t.pca_window, t.size, t.interval) == (9, 2, 80, 2.0, 3.0)
        assert all(getattr(t, name) == value for name, value in extra.items())
    assert build("flow").measure == "queues" and build("flow", measure="queues").measure == "queues"
    assert build("leadlag").interval == 1.0 and build("reversal").interval == 5.0


def test_each_kind_says_what_it_scores_how_far_back_how_often_it_rebalances_its_factors_and_its_lots():
    def say(kind, **change):
        return TRADER_KINDS[kind].describe({**DEFAULTS[kind], **change})

    assert say("reversal") == (  # notebook 13's section 5: a look back of 50, at every decision, a tenth of the way
        "Buys the books that fell over the last 50 decisions and sells those that rose, neutral to the dollar and 3 "
        "PCA factors from 500 decisions: 5 lots of the largest, rebalanced at every decision, deciding every 5 "
        "units, going 0.1 of the way to its target at each decision.")
    assert say("reversal", lookback=20, hold=100, rate=1.0) == (
        "Buys the books that fell over the last 20 decisions and sells those that rose, neutral to the dollar and 3 "
        "PCA factors from 500 decisions: 5 lots of the largest, rebalanced every 100 decisions, deciding every 5 "
        "units.")
    assert "the dollar alone: 2.5 lots" in say("reversal", factors=0, size=2.5)
    assert "the dollar and 1 PCA factor from 500" in say("reversal", factors=1)
    assert "over the last 100 decisions and sells those that fell" in say("momentum")
    assert "over the 100 decisions before the last 5 and sells" in say("momentum", skip=5)
    assert say("leadlag") == (
        "Buys the books that the other books' last moves say will rise, by a ridge regression (penalty 1) over 500 "
        "decisions, neutral to the dollar and 3 PCA factors: 5 lots of the largest, rebalanced every 20 decisions, "
        "deciding every 1 unit.")
    assert "the lots at the best bid against those at the best ask" in say("flow")
    assert "the share of what traded over the last 5 decisions that was bought" in say("flow", measure="trades")
    assert "last 5 decisions" not in say("flow") and "rebalanced every 25 decisions" in say("flow")  # queues: now
    for kind in CROSS_SECTIONAL:
        assert "\n" not in say(kind) and say(kind).count(". ") == 0 and say(kind).endswith(".")  # one sentence


@pytest.mark.parametrize("kind", TRADER_KINDS)
def test_every_kind_but_reversal_and_baskets_goes_all_the_way_by_default_and_each_trades_less_as_told(kind):
    spec = TRADER_KINDS[kind]
    assert [p.name for p in spec.numbers][-4:] == ["size", "rate", "band", "interval"]
    t = spec.build(DEFAULTS[kind])
    if kind == "reversal":  # notebook 13's section 5 best: at every decision, a tenth of the way
        assert (t.lookback, t.hold, t.rate, t.band) == (50, 1, 0.1, 0.0) and t._cost_aware
    elif kind == "baskets":  # section 6's best for Johansen's baskets of 3: an entry of 1, a quarter of the way
        assert (t.entry, t.rate, t.band) == (1.0, 0.25, 0.0) and t._cost_aware and t.netted
        assert DEFAULTS[kind]["method"] == "johansen" and (t.choose.func, t.choose.keywords) == (
            best_baskets, {"count": 4, "legs": 3, "every": 1})  # the chooser and its baskets are as they were
        assert (t.formation, t.window, t.exit, t.refit, t.size, t.interval) == (2000, 400, 0.0, None, 5.0, 5.0)
    else:
        assert (t.rate, t.band) == (1.0, 0.0) and not t._cost_aware
    if kind == "pairs":  # which shares its z-score settings with baskets, and keeps entering at 2 and going all the way
        assert (t.entry, t.rate, t.band) == (2.0, 1.0, 0.0) and DEFAULTS[kind]["entry"] == 2.0
    t = spec.build({**DEFAULTS[kind], "rate": 0.25, "band": 1.5})
    assert (t.rate, t.band) == (0.25, 1.5) and t._cost_aware and t.netted  # cost-aware trading needs netting
    if spec.no_fit:  # the deals kind has no fit, so no grid of rates and bands to try (and its own test says so)
        assert spec.fit_grid == {} and spec.fit_default == ()
        return
    rates, bands = ((0.02, 0.05, 0.1, 1.0), (0.0, 1.0, 2.0, 4.0)) if kind in CROSS_SECTIONAL else (
        (0.1, 0.25, 0.5, 1.0), (0.0, 0.5, 1.0, 2.0))
    assert spec.fit_grid["rate"] == rates and spec.fit_grid["band"] == bands
    assert "band" not in spec.fit_default  # a fit tries a band only when told
    assert ("rate" in spec.fit_default) == (kind == "reversal")  # with reversal's look back, at its hold of 1


def test_each_kind_says_how_it_trades_less_only_when_it_does():
    def say(kind, **change):
        return TRADER_KINDS[kind].describe({**DEFAULTS[kind], **change})

    for kind in TRADER_KINDS:
        whole = say(kind, rate=1.0)  # reversal goes a tenth of the way, baskets a quarter, the rest all the way
        assert ("of the way" in say(kind)) == (kind in ("reversal", "baskets"))
        assert "of the way" not in whole and "within" not in whole
        both = say(kind, rate=0.25, band=1.0)
        assert both.startswith(whole[:-1]) and both.endswith(
            "0.25 of the way to its target at each decision and not trading within 1 lot of that, except to close a "
            "book." if kind in CROSS_SECTIONAL else
            " It goes 0.25 of the way to its target at each decision and does not trade within 1 lot of that, except "
            "to close a book.")
        assert say(kind, rate=0.5).endswith("0.5 of the way to its target at each decision.")
        band = say(kind, rate=1.0, band=2.0)
        assert band.endswith("within 2 lots of its target, except to close a book.") and "of the way" not in band
    assert say("reversal", rate=0.5) == say("reversal", rate=1.0)[:-1] + (
        ", going 0.5 of the way to its target at each decision.")
    alone = " It does not trade within 2 lots of its target, except to close a book."
    assert say("pairs", band=2.0) == say("pairs") + alone
    assert say("baskets", rate=1.0, band=2.0) == say("baskets", rate=1.0) + alone
    assert say("baskets") == say("baskets", rate=1.0) + " It goes 0.25 of the way to its target at each decision."
    assert say("baskets", band=2.0) == say("baskets", rate=1.0) + (  # its default rate is there when only a band is set
        " It goes 0.25 of the way to its target at each decision and does not trade within 2 lots of that, except to "
        "close a book.")
    for kind in CROSS_SECTIONAL:  # still one sentence
        assert say(kind, rate=0.25, band=1.0).count(". ") == 0


@pytest.mark.parametrize("kind", TRADER_KINDS)
def test_every_kind_trades_part_of_the_way_on_the_view(client, statarb, kind):
    in_a_market_for(client, kind)  # the two that trade an index or deals, in a market that has them
    # One basket is a lot of the index, and going half the way to it is half a lot: inside a band of 1 lot, so that it
    # would never trade. Four baskets are two lots to aim at after the first decision, and one beyond the band.
    more = {"size": 4} if kind == "index" else {}
    reply = client.post("/api/statarb/traders/start", data=form({"kind": kind, **QUICK[kind], "rate": 0.5,
                                                                 "band": 1.0, **more}))
    assert reply.status_code == 200
    t = statarb.runners[0].trader
    assert (t.rate, t.band) == (0.5, 1.0)
    for _ in range(10):
        client.post("/api/statarb/control", data={"action": "step", "units": 300})
        if t.lots:
            break
        choosing = getattr(t, "_pending", None)
        if choosing is not None:
            choosing.result()
    assert t.lots > 0 and statarb.board()[0]["describe"].endswith("except to close a book.")


def test_the_summaries_of_momentum_and_lead_lag_agree_with_their_defaults_and_with_each_other():
    leadlag, momentum = TRADER_KINDS["leadlag"].summary, TRADER_KINDS["momentum"].summary
    assert DEFAULTS["leadlag"]["hold"] == 20 and "held for 20 decisions, as it is by default" in leadlag
    assert "every decision" not in leadlag  # it rebalances every 20, as its description says
    assert DEFAULTS["momentum"]["lookback"] == 100 and "at its look back of 100 it loses before costs" in momentum
    assert "right where reversal is wrong (over 25 units of time)" in momentum
    assert "nothing to find" not in momentum  # there is something, over 25 units of time, though not enough


def test_the_board_ranks_the_traders_and_says_what_each_holds_of_the_factors():
    session = StatArbSession()
    first, _ = session.start_trader(QUICK["pairs"], "pairs")
    second, _ = session.start_trader(QUICK["pairs"], "pairs")
    session.set_active(first, False)
    session.set_active(second, False)
    marks = session.market.marks()
    first.trader.position[:2] = [5, -5]  # within a sector: none of the factors
    first.trader.cash = -float(first.trader.position @ marks)
    second.trader.position[0] = 5  # one book: its market and its sector's factors, five of each
    second.trader.cash = 100.0 - float(second.trader.position @ marks)
    second.trader.spread_paid, second.trader.lots = 12.0, 4
    rows = session.board()
    assert [r["id"] for r in rows] == [second.id, first.id]  # the best first, though it started later
    assert rows[0]["pnl"] == pytest.approx(100.0) and rows[1]["pnl"] == pytest.approx(0.0, abs=1e-9)
    assert rows[0]["gross"] == 5 and rows[0]["exposure"] == pytest.approx(5 * math.sqrt(2))
    assert rows[1]["gross"] == 10 and rows[1]["exposure"] == pytest.approx(0.0)
    assert rows[0]["before"] == pytest.approx(112.0) and rows[0]["cost_a_lot"] == 3.0 and rows[1]["cost_a_lot"] is None
    session.run_for(2)
    sample = session.samples[-1]
    assert sample["gross"] == {first.id: 10, second.id: 5}
    assert sample["exposure"][first.id] == 0
    assert sample["exposure"][second.id] == pytest.approx(5 * math.sqrt(2), abs=0.01)


def test_a_relation_without_a_signal_says_so():
    session = StatArbSession()
    runner, _ = session.start_trader(QUICK["pairs"], "pairs")
    t = runner.trader
    t.weights, t.states = np.eye(12)[:3], np.array([1, 0, 0])
    t.signal = lambda: np.array([np.nan, 1.5])
    relations = session.relations_of(t)
    assert [r["signal"] for r in relations] == [None, 1.5, None] and [r["state"] for r in relations] == [1, 0, 0]
    assert session.board()[0]["holding"] == 1
    t.signal = lambda: None  # its window not full yet
    assert [r["signal"] for r in session.relations_of(t)] == [None, None, None]
    t.signal = lambda: np.array([np.nan, 1.5])
    session.set_active(runner, False)
    session.run_for(2)
    assert session.samples[-1]["sig"][runner.id] == [None, 1.5]


def test_what_changes_the_page_outside_the_clock_tells_the_stream():
    session = StatArbSession()

    def tells(action):
        before = session.changes
        action()
        assert session.changes != before

    made = []
    tells(lambda: made.append(session.start_trader(QUICK["pairs"], "pairs")[0]))
    tells(lambda: session.set_active(made[0], False))
    tells(lambda: session.remove(made[0]))
    tells(session.reset)
    session.run_for(1_100)
    tells(lambda: asyncio.run(session.test_relations()))


def test_new_settings_are_refused_only_for_a_residual_trader_with_too_many_factors():
    session = StatArbSession()
    pairs, _ = session.start_trader(QUICK["pairs"], "pairs")
    residuals, _ = session.start_trader(QUICK["residuals"], "residuals")  # three factors
    session.configure({**MARKET, "sectors": 2, "books_each": 2})
    assert session.market.n == 4
    assert pairs.trader.position.tolist() == [0] * 4 and residuals.trader.position.tolist() == [0] * 4


def test_a_stream_gets_each_sample_once():
    session = StatArbSession()
    session.run_for(5)
    assert [s["t"] for s in session.samples_since(None)] == [0, 1, 2, 3, 4, 5]
    assert [s["t"] for s in session.samples_since(3)] == [4, 5]
    assert session.samples_since(5) == []


def test_a_trader_learns_from_the_past_when_it_can():
    session = StatArbSession()
    session.run_for(1_000)
    runner, learned = session.start_trader({**QUICK["pairs"], "formation": 100, "window": 20, "interval": 5.0}, "pairs")
    assert learned == 100 and runner.trader.warming == (100, 100)
    young = StatArbSession()
    young.run_for(100)
    runner, learned = young.start_trader({**QUICK["pairs"], "formation": 100, "interval": 5.0}, "pairs")
    assert learned == 21 and runner.trader.warming == (21, 100)  # one mid every 5 units
    cold, learned = session.start_trader({**QUICK["pairs"], "warm": False}, "pairs")
    assert learned == 0 and cold.trader.warming == (0, QUICK["pairs"]["formation"])


def test_settings_that_do_not_suit_the_books_are_refused(client, statarb):
    eight_factors = {**DEFAULTS["residuals"], "factors": 8}
    reply = client.post("/api/statarb/traders/start", data=form({"kind": "residuals", **eight_factors}))
    assert reply.status_code == 200  # twelve books: fine
    # New settings cannot strand it either: they are refused, and the market goes on as it was.
    version = statarb.version
    few = form({**MARKET, "sectors": 1, "books_each": 4})
    reply = client.post("/api/statarb/config", data=few)
    assert reply.status_code == 422 and "Residuals 1 takes 8 factors" in reply.json()["error"]
    assert statarb.version == version and statarb.market.n == 12
    client.post(f"/api/statarb/traders/{statarb.runners[0].id}/remove")
    assert client.post("/api/statarb/config", data=few).status_code == 200
    reply = client.post("/api/statarb/traders/start", data=form({"kind": "residuals", **eight_factors}))
    assert reply.status_code == 422 and "fewer factors" in reply.json()["error"]
    assert not statarb.runners and statarb.market.traders == []


@pytest.mark.parametrize("kind", CROSS_SECTIONAL)
def test_a_trader_that_scores_every_book_needs_more_books_than_its_factors_and_one(client, statarb, kind):
    assert too_few_books(kind, {"factors": 3}, 5) is None and too_few_books(kind, {"factors": 0}, 2) is None
    assert too_few_books(kind, {"factors": 3}, 4) == "fewer factors than the 4 books less one"
    four = form({**MARKET, "sectors": 1, "books_each": 4})
    assert client.post("/api/statarb/config", data=four).status_code == 200
    reply = client.post("/api/statarb/traders/start", data=form({"kind": kind, **QUICK[kind], "factors": 3}))
    assert reply.status_code == 422 and reply.json() == {"error": "Not started: fewer factors than the books less one."}
    assert not statarb.runners and statarb.market.traders == [] and statarb._seq == 0  # and it took no number
    reply = client.post("/api/statarb/traders/start", data=form({"kind": kind, **QUICK[kind], "factors": 2}))
    assert reply.status_code == 200 and reply.json()["name"] == f"{TRADER_KINDS[kind].label} 1"
    # New settings cannot strand it either: three books take one factor at most.
    version = statarb.version
    three = form({**MARKET, "shape": "mixed", "books": 3, "factors": 1})
    reply = client.post("/api/statarb/config", data=three)
    assert reply.status_code == 422 and reply.json() == {
        "error": f"Not applied: {TRADER_KINDS[kind].label} 1 takes 2 factors, and needs at least 4 books: remove it, "
                 "or apply that many or more."}
    assert statarb.version == version and statarb.market.n == 4
    four_mixed = form({**MARKET, "shape": "mixed", "books": 4, "factors": 1})
    assert client.post("/api/statarb/config", data=four_mixed).status_code == 200 and statarb.market.n == 4
    client.post(f"/api/statarb/traders/{statarb.runners[0].id}/remove")
    assert client.post("/api/statarb/config", data=three).status_code == 200


def test_a_flow_trader_says_it_took_in_mids_for_its_factors_and_sees_the_flow_only_from_now_on(client, statarb):
    statarb.run_for(300)

    def message(**change):
        reply = client.post("/api/statarb/traders/start", data=form({"kind": "flow", **QUICK["flow"], **change}))
        client.post(f"/api/statarb/traders/{reply.json()['id']}/remove")
        return reply.json()["message"]

    sees = "sees the order flow only from now on: it chooses once it has seen"
    assert message() == f"Order flow 1 took in 41 points of the past mids for its factors, but {sees} 1 more decision."
    assert message(measure="trades") == (
        f"Order flow 2 took in 41 points of the past mids for its factors, but {sees} 6 more decisions.")
    assert message(warm=False) == f"Order flow 3 {sees} 41 decisions."  # its mids too: for the factors
    assert message(factors=0) == f"Order flow 4 {sees} 1 decision."
    assert message(factors=0, measure="trades") == f"Order flow 5 {sees} 6 decisions."
    assert message(factors=0, measure="trades", lookback=1) == f"Order flow 6 {sees} 2 decisions."


def test_a_flow_trader_reads_trades_or_queues_and_nothing_else(client, statarb):
    for measure, lookback in (("queues", 5), ("trades", 5)):
        reply = client.post("/api/statarb/traders/start", data=form({"kind": "flow", **QUICK["flow"],
                                                                      "measure": measure}))
        assert reply.status_code == 200 and statarb.runners[-1].trader.measure == measure
        assert statarb.runners[-1].values["measure"] == measure and statarb.runners[-1].trader.lookback == lookback
    reply = client.post("/api/statarb/traders/start", data=form({"kind": "flow", **QUICK["flow"], "measure": "tea"}))
    assert reply.status_code == 422 and reply.json() == {"errors": {"measure": "Reads: one of queues, trades"}}
    assert len(statarb.runners) == 2
    default = {k: v for k, v in form(QUICK["flow"]).items() if k != "measure"}
    reply = client.post("/api/statarb/traders/start", data={"kind": "flow", **default})
    assert reply.status_code == 200 and statarb.runners[-1].trader.measure == "queues"  # left out: its default


def test_the_page_offers_the_four_kinds_and_a_flow_trader_a_choice_of_what_it_reads(client):
    page = client.get("/").text
    for kind in CROSS_SECTIONAL:
        spec = TRADER_KINDS[kind]
        assert f'<input type="radio" name="kind" value="{kind}"' in page and f">{spec.label}</span>" in page
        start = page.index(f'data-statarb-kind="{kind}"')
        fieldset = page[start:page.index("</fieldset>", start)]
        # the page escapes the & of P&L and apostrophes
        assert spec.summary in html.unescape(fieldset) and 'name="warm"' in fieldset
        for q in spec.numbers:
            assert f'name="{q.name}" value="{q.default}"' in fieldset
        assert ('<select name="measure"' in fieldset) == (kind == "flow")
        assert ('name="lookback"' in fieldset) == (kind != "leadlag")  # its score uses the last decision alone
        assert ('name="skip"' in fieldset) == (kind == "momentum")
        assert ('name="ridge"' in fieldset) == (kind == "leadlag")
    start = page.index('data-statarb-kind="flow"')
    select = page[page.index('<select name="measure"', start):]
    select = select[:select.index("</select>")]
    assert re.findall(r'<option value="(\w+)" (selected)?', select) == [("queues", "selected"), ("trades", "")]


# ---- How baskets are chosen ----

def test_baskets_are_chosen_by_johansen_unless_box_tiao_is_asked_for():
    def build(**change):
        return TRADER_KINDS["baskets"].build({**DEFAULTS["baskets"], **change})

    assert DEFAULTS["baskets"]["method"] == "johansen"
    for legs, func in ((3, best_baskets), (4, best_baskets), (0, johansen_relations)):
        t = build(legs=legs, count=2)
        keywords = {"count": 2, "every": 1, **({"legs": legs} if legs else {})}
        assert t.choose.func is func and t.choose.keywords == keywords
    for legs in (2, 3, 4, 0):  # legs 0: the portfolios on all the books, as sparse_baskets takes it
        t = build(method="box-tiao", legs=legs, count=3)
        assert t.choose.func is sparse_baskets and t.choose.keywords == {"count": 3, "legs": legs, "every": 1}
        assert isinstance(t, RelationTrader) and t.choose.args == ()
    z = {"formation": 300, "window": 50, "entry": 1.5, "exit": 0.5, "refit": 100, "size": 2.0, "interval": 3.0}
    t = build(method="box-tiao", **z)
    assert (t.formation, t.window, t.entry, t.exit, t.refit, t.size, t.interval) == tuple(z.values())
    assert build(method="box-tiao", refit=0).refit is None
    for method in ("johansen", "box-tiao"):
        with pytest.raises(ValueError, match="baskets of two books or more, or 0"):
            build(method=method, legs=1)


def test_a_baskets_trader_says_how_it_chooses_its_baskets():
    def say(**change):
        return TRADER_KINDS["baskets"].describe({**DEFAULTS["baskets"], **change})

    def whole(**change):  # entering beyond z of 2 and going all the way, as the default did before section 6's best
        return say(entry=2.0, rate=1.0, **change)

    # The defaults are section 6's best: an entry of 1, and a quarter of the way to the target at each decision.
    quarter = " It goes 0.25 of the way to its target at each decision."
    best = "5 lots of the largest leg beyond z of 1 over 400 decisions, every 5 units"
    assert say() == (
        f"Up to 4 Johansen baskets of 3 books, by their trace statistics, from 2,000 decisions; {best}.{quarter}")
    assert say(method="box-tiao") == (
        "Up to 4 baskets of 3 books grown greedily from every pair by Box-Tiao predictability (d'Aspremont 2011), the "
        f"least predictable first and none rejected, from 2,000 decisions; {best}; there is no test: it always gives "
        f"baskets.{quarter}")
    assert say(legs=0, refit=500) == (
        f"Up to 4 Johansen vectors on all the books, from 2,000 decisions, again every 500; {best}.{quarter}")
    # What it said when the default was an entry of 2 and a rate of 1, still said for them.
    trading = "5 lots of the largest leg beyond z of 2 over 400 decisions, every 5 units"
    assert whole() == (
        f"Up to 4 Johansen baskets of 3 books, by their trace statistics, from 2,000 decisions; {trading}.")
    assert whole(legs=0, refit=500) == (
        f"Up to 4 Johansen vectors on all the books, from 2,000 decisions, again every 500; {trading}.")
    assert whole(method="box-tiao") == (
        "Up to 4 baskets of 3 books grown greedily from every pair by Box-Tiao predictability (d'Aspremont 2011), the "
        f"least predictable first and none rejected, from 2,000 decisions; {trading}; there is no test: it always "
        "gives baskets.")
    assert whole(method="box-tiao", legs=0) == (  # "up to": no more than there are books
        "Up to 4 of the least predictable portfolios of all the books (Box and Tiao 1977), none rejected, from 2,000 "
        f"decisions; {trading}; there is no test: it always gives baskets.")
    assert whole(method="box-tiao", legs=0, count=1).startswith("The least predictable portfolio of all the books (Box")
    assert whole(method="box-tiao", legs=4, count=1).startswith("Up to 1 basket of 4 books grown greedily")
    assert whole(method="box-tiao", legs=2) == (  # nothing is grown from a pair: the pairs are ranked
        "Up to 4 pairs ranked by Box-Tiao predictability (d'Aspremont 2011), the least predictable first and none "
        f"rejected, from 2,000 decisions; {trading}; there is no test: it always gives baskets.")
    assert whole(method="box-tiao", legs=2, count=1).startswith("Up to 1 pair ranked by Box-Tiao predictability")
    assert "grown" not in whole(method="box-tiao", legs=2)
    assert whole(method="box-tiao", refit=500, formation=1500).count("from 1,500 decisions, again every 500; ") == 1
    for method in ("johansen", "box-tiao"):  # trading less ends it, as it ends the others
        assert say(method=method, rate=0.25, band=1.0).endswith(
            " It goes 0.25 of the way to its target at each decision and does not trade within 1 lot of that, except "
            "to close a book.")
        assert whole(method=method, band=2.0).endswith(" It does not trade within 2 lots of its target, except to "
                                                       "close a book.")
        assert say(method=method, band=2.0).endswith(  # the default's quarter is there when only a band is asked for
            " It goes 0.25 of the way to its target at each decision and does not trade within 2 lots of that, except "
            "to close a book.")
        assert "of the way" not in say(method=method, rate=1.0)  # and nothing of it when asked for the whole way
    assert say(method="box-tiao", entry=2.0, rate=0.5) == whole(method="box-tiao") + (
        " It goes 0.5 of the way to its target at each decision.")


def test_a_baskets_trader_chooses_by_johansen_or_box_tiao_and_nothing_else(client, statarb):
    for method, func in (("johansen", best_baskets), ("box-tiao", sparse_baskets)):
        reply = client.post("/api/statarb/traders/start", data=form({"kind": "baskets", **QUICK["baskets"],
                                                                      "method": method}))
        assert reply.status_code == 200 and statarb.runners[-1].trader.choose.func is func
        assert statarb.runners[-1].values["method"] == method
    reply = client.post("/api/statarb/traders/start", data=form({"kind": "baskets", **QUICK["baskets"],
                                                                  "method": "tea"}))
    assert reply.status_code == 422 and reply.json() == {"errors": {"method": "Chosen by: one of johansen, box-tiao"}}
    assert len(statarb.runners) == 2 and statarb._seq == 2  # it took no number
    default = {k: v for k, v in form(QUICK["baskets"]).items() if k != "method"}
    reply = client.post("/api/statarb/traders/start", data={"kind": "baskets", **default})
    assert reply.status_code == 200 and statarb.runners[-1].trader.choose.func is best_baskets  # left out: Johansen
    for kind in TRADER_KINDS:  # no other kind has the choice
        assert ("method" in DEFAULTS[kind]) == (kind == "baskets")


def test_the_summary_of_baskets_agrees_with_its_defaults_and_pairs_are_as_they_were():
    baskets, pairs = TRADER_KINDS["baskets"].summary, TRADER_KINDS["pairs"].summary
    assert DEFAULTS["baskets"]["entry"] == 1.0 and DEFAULTS["baskets"]["rate"] == 0.25
    assert "an entry of 1, going a quarter of the way to the target at each decision (a trade rate of 0.25)" in baskets
    assert "+2,044 (se 132) and +2,012 (se 176) more than an entry of 2 traded the whole way" in baskets
    assert "at these defaults" not in baskets  # it was the old ones that it was compared with
    assert (DEFAULTS["pairs"]["entry"], DEFAULTS["pairs"]["rate"]) == (2.0, 1.0)  # the z-score settings they share
    assert "+5,706 at these defaults" in pairs  # whose summary still compares with them


def test_the_page_offers_baskets_a_choice_of_how_they_are_chosen_ahead_of_their_numbers(client):
    page = client.get("/").text
    for kind, spec in TRADER_KINDS.items():
        start = page.index(f'data-statarb-kind="{kind}"')
        fieldset = page[start:page.index("</fieldset>", start)]
        assert ('<select name="method"' in fieldset) == (kind == "baskets")
        assert ("Box-Tiao predictability" in html.unescape(fieldset)) == (kind == "baskets")
        for hint in ("by the trace test at 95%", "whether or not any is a relation"):  # the select's hover hint
            assert (hint in html.unescape(fieldset)) == (kind == "baskets")
        assert spec.summary in html.unescape(fieldset)
    start = page.index('data-statarb-kind="baskets"')
    fieldset = page[start:page.index("</fieldset>", start)]
    assert fieldset.index('<select name="method"') < fieldset.index('name="legs"')  # first, as the flow's measure is
    select = fieldset[fieldset.index('<select name="method"'):]
    select = select[:select.index("</select>")]
    assert re.findall(r'<option value="([\w-]+)" (selected)?', select) == [("johansen", "selected"), ("box-tiao", "")]
    assert ">Chosen by</span>" in fieldset
    assert 'name="entry" value="1.0"' in fieldset and 'name="rate" value="0.25"' in fieldset  # section 6's best
    pairs = page[page.index('data-statarb-kind="pairs"'):]
    pairs = pairs[:pairs.index("</fieldset>")]  # whose z-score settings baskets share, and which are as they were
    assert 'name="entry" value="2.0"' in pairs and 'name="rate" value="1.0"' in pairs


@pytest.mark.parametrize("legs", [3, 0])
def test_a_box_tiao_baskets_trader_starts_chooses_trades_and_shows(client, statarb, legs):
    values = {**QUICK["baskets"], "method": "box-tiao", "legs": legs}
    reply = client.post("/api/statarb/traders/start", data=form({"kind": "baskets", **values}))
    assert reply.status_code == 200 and reply.json()["name"] == "Baskets 1"
    runner = statarb.runners[0]
    assert runner.kind == "baskets" and runner.trader.choose.func is sparse_baskets
    for _ in range(10):
        client.post("/api/statarb/control", data={"action": "step", "units": 300})
        if runner.trader.lots:
            break
        choosing = getattr(runner.trader, "_pending", None)
        if choosing is not None:  # it happens elsewhere, and a market stepped flat out can starve it for many seconds
            choosing.result()
    t = runner.trader
    assert t.lots > 0 and t.warming is None and t.fits >= 1
    assert (t.entry, t.rate) == (1.0, 0.25) and runner.values["entry"] == 1.0  # the default's, which is cost-aware
    assert t.weights.shape == (2, 12) and np.count_nonzero(t.weights, axis=1).tolist() == [legs or 12] * 2
    assert np.abs(t.weights).max(axis=1).tolist() == [1.0, 1.0]  # the largest leg 1, as the others give them
    row = statarb.board()[0]
    assert row["pnl"] == pytest.approx(row["before"] - row["cost"]) and row["lots"] == t.lots and row["relations"]
    assert len(row["relations"]) == 2 and row["describe"].endswith(
        "there is no test: it always gives baskets. It goes 0.25 of the way to its target at each decision.")
    assert ("Box and Tiao 1977" if legs == 0 else "d'Aspremont 2011") in row["describe"]
    assert runner.id in statarb.samples[-1]["sig"] and statarb.samples[-1]["gross"][runner.id] >= 0
    panels = render_statarb_panels(statarb.context())
    assert f'data-statarb-id="{runner.id}"' in panels["statarb_board"]
    assert "data-statarb-relations" in panels["statarb_board"] and "no test: it always gives baskets" in html.unescape(
        panels["statarb_board"])


@pytest.mark.parametrize("change, name", [({"window": 1}, "window"), ({"size": 0}, "size"), ({"count": 99}, "count"),
                                          ({"rate": 0}, "rate"), ({"rate": 1.5}, "rate"), ({"band": -1}, "band")])
def test_a_trader_with_a_bad_setting_is_refused_with_the_field_at_fault(client, statarb, change, name):
    reply = client.post("/api/statarb/traders/start", data=form({"kind": "pairs", **DEFAULTS["pairs"], **change}))
    assert reply.status_code == 422 and name in reply.json()["errors"] and not statarb.runners


@pytest.mark.parametrize("kind, change, words", [
    ("copula", {"entry": 1.0, "exit": 1.0}, "exit < entry"),
    ("residuals", {"open_at": 1.0, "close_long": 1.0}, "closing < opening"),
    ("residuals", {"open_at": 1.0, "close_short": 2.0}, "closing < opening"),
])
def test_levels_that_cross_are_refused_before_the_trader_is_in_the_market(client, statarb, kind, change, words):
    reply = client.post("/api/statarb/traders/start", data=form({"kind": kind, **DEFAULTS[kind], **change}))
    assert reply.status_code == 422 and words in reply.json()["error"]
    assert not statarb.runners and statarb.market.traders == []


def test_an_unknown_kind_is_refused(client):
    reply = client.post("/api/statarb/traders/start", data={"kind": "tea_leaves"})
    assert reply.status_code == 422 and "No such kind" in reply.json()["errors"]["kind"]


def test_the_board_holds_so_many_and_stops_resumes_and_removes(client, statarb):
    assert not statarb.context()["full"]
    for _ in range(MAX_TRADERS):
        reply = client.post("/api/statarb/traders/start", data=form({"kind": "pairs", **QUICK["pairs"]}))
        assert reply.status_code == 200
    reply = client.post("/api/statarb/traders/start", data=form({"kind": "pairs", **QUICK["pairs"]}))
    assert reply.status_code == 409 and "At most" in reply.json()["error"]
    assert statarb.context()["full"]
    assert render_statarb_panels(statarb.context())["statarb_board"].count("data-statarb-id=") == MAX_TRADERS
    first = statarb.runners[0]
    assert client.post(f"/api/statarb/traders/{first.id}/stop").json() == {"id": first.id, "action": "stop"}
    assert not first.trader.active and not first.active
    client.post("/api/statarb/control", data={"action": "step", "units": 200})
    assert first.trader.lots == 0
    client.post(f"/api/statarb/traders/{first.id}/resume")
    assert first.trader.active and first.trader.next_time == statarb.market.time
    client.post(f"/api/statarb/traders/{first.id}/remove")
    assert statarb.runner(first.id) is None and first.trader not in statarb.market.traders
    assert client.post(f"/api/statarb/traders/{first.id}/stop").status_code == 404


@pytest.mark.parametrize("kind", TRADER_KINDS)
def test_removing_the_second_of_two_traders_started_alike_leaves_the_first_in_the_market(client, statarb, kind):
    in_a_market_for(client, kind)
    for _ in range(2):  # at the same time, so that nothing yet tells them apart
        assert client.post("/api/statarb/traders/start", data=form({"kind": kind, **QUICK[kind]})).status_code == 200
    first, second = statarb.runners
    assert client.post(f"/api/statarb/traders/{second.id}/remove").json() == {"id": second.id, "action": "remove"}
    assert [r.id for r in statarb.runners] == [first.id]
    assert len(statarb.market.traders) == 1 and statarb.market.traders[0] is first.trader  # no trader out of sight


def test_each_trader_has_a_line_and_a_color_and_its_levels():
    session = StatArbSession()
    session.start_trader(QUICK["residuals"], "residuals")
    session.start_trader({**QUICK["pairs"], "entry": 1.5, "exit": 0.5}, "pairs")
    traders = session.chart_traders()
    assert [t["id"] for t in traders] == [1, 2] and traders[0]["color"] != traders[1]["color"]
    assert traders[0]["entry"] == 1.5 and traders[0]["exit"] == 0.0 and traders[1]["exit"] == 0.5
    assert traders[0]["color"] == COLORS[0] and traders[1]["color"] == COLORS[1]


def test_a_relation_reads_in_lots_of_each_book():
    names = ["A1", "A2", "A3", "B1"]
    assert relation_text(np.array([1.0, -0.98, 0.0, 0.001]), names) == "A1 - 0.98 A2"
    assert relation_text(np.array([-1.0, 0.5, 0.25, 0.0]), names) == "-A1 + 0.50 A2 + 0.25 A3"
    assert relation_text(np.array([1.0, 0.007, 0.0, 0.0]), names) == "A1 + 0.01 A2"
    many = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    assert (relation_text(many, [str(i) for i in range(8)])
            == "0 + 0.90 1 + 0.80 2 + 0.70 3 + 0.60 4 + 0.50 5 and 2 more")
    assert book_names({"shape": "sectors", "sectors": 2, "books_each": 2}) == ["A1", "A2", "B1", "B2"]


# ---- What the tests see ----

def test_the_tests_say_what_they_find_against_the_truth(client, statarb):
    assert not client.post("/api/statarb/test").json()["enough"]
    assert "Too few mids" in render_statarb_panels(statarb.context())["statarb_status"]
    client.post("/api/statarb/control", data={"action": "step", "units": 1500})
    reply = client.post("/api/statarb/test").json()
    assert reply["enough"] and reply["true_rank"] == 9 and reply["pairs"] == 66 and reply["truly"] == 18
    assert 0 <= reply["rank"] <= 12 and 0 <= reply["called"] <= 66 and reply["units"] == 1501
    status = render_statarb_panels(statarb.context())["statarb_status"]
    assert f"Johansen finds <span class=\"text-slate-100 font-semibold\">{reply['rank']}</span> of 9" in status


def test_the_tests_run_as_they_say_and_count_what_passes(monkeypatch):
    def johansen(data, det_order, k_ar_diff):
        assert len(data) == 400 and det_order == 0 and k_ar_diff == 2  # one mid in five
        return SimpleNamespace(lr1=np.array([40.0, 20.0, 1.0]),
                               cvt=np.array([[25.0, 27.0, 30.0], [12.0, 15.0, 20.0], [1.0, 2.0, 3.0]]))

    p_values = iter([0.01, 0.04, 0.06])

    def coint(x, y, maxlag, autolag):
        assert len(x) == len(y) == 400 and maxlag == 5 and autolag is None
        return 0.0, next(p_values), None

    monkeypatch.setattr("statsmodels.tsa.vector_ar.vecm.coint_johansen", johansen)
    monkeypatch.setattr("statsmodels.tsa.stattools.coint", coint)
    mids, loadings = np.zeros((2_000, 3)), np.array([[1.0], [1.0], [1.0]])
    assert what_the_tests_see(mids, loadings) == {"units": 2_000, "enough": True, "rank": 2, "true_rank": 2,
                                                  "pairs": 3, "called": 2, "truly": 3}
    few = what_the_tests_see(mids[:10], loadings)
    assert not few["enough"] and few["true_rank"] == 2 and few["truly"] == 3


def test_the_button_tests_the_last_ten_thousand_units(monkeypatch):
    session = StatArbSession()
    session.mids.extend(np.full(12, float(i)) for i in range(12_000))
    seen = []
    monkeypatch.setattr(statarb_module, "what_the_tests_see",
                        lambda mids, loadings: seen.append(mids) or {"enough": False})
    asyncio.run(session.test_relations())
    assert len(seen[0]) == 10_000 and seen[0][-1, 0] == 11_999 and seen[0][0, 0] == 2_000


def test_the_tests_run_off_the_event_loop_and_a_reset_forgets_them():
    session = StatArbSession()
    session.run_for(1_100)

    async def main():
        ticks = []

        async def beat():
            while True:
                ticks.append(1)
                await asyncio.sleep(0.005)

        beating = asyncio.create_task(beat())
        result = await session.test_relations()
        beating.cancel()
        return result, len(ticks)

    result, ticks = asyncio.run(main())
    assert result["enough"] and ticks > 1 and session.test == result
    session.reset()
    assert session.test is None


# ---- The stream ----

def test_the_stream_stands_in_for_events_and_brings_the_view_its_news(client, statarb):
    wanted = {"hello", "statarb_history", "statarb_state", "statarb_status", "statarb_books", "statarb_board"}
    seen = read_stream(client, wanted)
    assert json.loads(seen["hello"])["instance"] == registry.get(client.cookies[COOKIE_NAME]).instance_id
    history = json.loads(seen["statarb_history"])
    assert history["names"][0] == "A1" and history["groups"][-1] == 2 and len(history["samples"]) == 1
    assert json.loads(seen["statarb_state"])["settings"]["shape"] == "sectors"


def test_an_evicted_session_stops_its_clock():
    session = StatArbSession()

    async def main():
        await session.start()
        await asyncio.sleep(0.15)
        running = session.market.time
        session.shutdown()
        await asyncio.sleep(0.1)
        assert running > 0 and not session.is_running and session._task is None
        return session.market.time

    stopped = asyncio.run(main())
    assert stopped == session.market.time


def test_the_controls_limits_hold(client):
    assert client.post("/api/statarb/control", data={"action": "step", "units": MAX_STEP + 1}).status_code == 422
    assert client.post("/api/statarb/control", data={"action": "sing"}).status_code == 422
    assert client.post("/api/statarb/speed", data={"speed": 0}).status_code == 422
    assert client.post("/api/statarb/speed", data={"speed": 200}).json() == {"speed": 200}


def test_a_form_says_what_the_market_settings_are():
    values, errors = parse_market({"shape": "mixed", "books": "5", "factors": "2"})
    assert not errors and values["books"] == 5 and values["factors"] == 2 and values["shape"] == "mixed"
    assert values["reverting"] is False  # a switch is on only when in the form


def test_the_status_counts_one_unit_kept_in_the_singular(statarb):
    assert len(statarb.mids) == 1
    text = render_statarb_panels(statarb.context())["statarb_status"]
    assert "1 unit kept" in text and "units kept" not in text
    statarb.run_for(3)
    assert f"{len(statarb.mids)} units kept" in render_statarb_panels(statarb.context())["statarb_status"]


def test_a_copula_trader_is_built_by_the_chooser_thread_and_not_while_the_market_stands_still(monkeypatch):
    session = StatArbSession()
    session.configure({**MARKET, "shape": "mixed", "books": 3, "factors": 1})
    runner, _ = session.start_trader({**QUICK["copula"], "warm": False}, "copula")
    assert runner.trader.defer is statarb_module._choose_elsewhere
    built_on = []
    build = CopulaPortfolio._build

    def spy_build(self, history):
        built_on.append(threading.current_thread().name)
        return build(self, history)

    monkeypatch.setattr(CopulaPortfolio, "_build", spy_build)
    deadline = time.monotonic() + 60
    while not runner.trader.traders:
        assert time.monotonic() < deadline
        session.run_for(20.0)
        time.sleep(0.01)
    assert len(built_on) == 1 and built_on[0].startswith("statarb-choose")


# ---- Johansen's test takes twelve books at most ----
# statsmodels has no critical values past twelve books: its trace test would read a rank from nan. The sectors market
# of four sectors of four books has sixteen.

SIXTEEN = {**MARKET, "sectors": 4, "books_each": 4}
JOHANSEN_OF_ALL = {**DEFAULTS["baskets"], "method": "johansen", "legs": 0}


def test_baskets_of_all_the_books_by_johansen_are_refused_past_twelve_books_and_say_what_to_do_instead():
    for books in (13, 16):
        reason = too_few_books("baskets", JOHANSEN_OF_ALL, books)
        assert reason == (f"baskets of a few books, or chosen by Box-Tiao: Johansen's test takes 12 books at most, "
                          f"and there are {books}")
    assert too_few_books("baskets", JOHANSEN_OF_ALL, 12) is None
    for change in ({"legs": 3}, {"legs": 4}, {"method": "box-tiao"}):  # few books each, or no table to read
        assert too_few_books("baskets", {**JOHANSEN_OF_ALL, **change}, 16) is None
    assert too_few_books("baskets", {**JOHANSEN_OF_ALL, "legs": 20}, 16) == "baskets of at most the 16 books"
    assert too_few_books("pairs", DEFAULTS["pairs"], 16) is None  # and no other kind is asked


def test_a_trader_of_baskets_of_all_the_books_by_johansen_is_not_started_in_sixteen_books_and_takes_no_number(client,
                                                                                                         statarb):
    assert client.post("/api/statarb/config", data=form(SIXTEEN)).status_code == 200 and statarb.market.n == 16
    reply = client.post("/api/statarb/traders/start", data=form({"kind": "baskets", **QUICK["baskets"], "legs": 0}))
    assert reply.status_code == 422 and reply.json() == {
        "error": "Not started: baskets of a few books, or chosen by Box-Tiao: Johansen's test takes 12 books at most, "
                 "and there are 16."}
    assert not statarb.runners and statarb.market.traders == [] and statarb._seq == 0
    for change in ({"legs": 3}, {"method": "box-tiao", "legs": 0}):
        reply = client.post("/api/statarb/traders/start", data=form({"kind": "baskets", **QUICK["baskets"], **change}))
        assert reply.status_code == 200, change
    assert len(statarb.runners) == 2 and statarb.market.n == 16


def test_new_settings_of_thirteen_books_or_more_are_refused_while_a_johansen_trader_of_all_the_books_is_on_the_board(
        client, statarb):
    reply = client.post("/api/statarb/traders/start", data=form({"kind": "baskets", **QUICK["baskets"], "legs": 0}))
    assert reply.status_code == 200 and statarb.market.n == 12
    version = statarb.version
    reply = client.post("/api/statarb/config", data=form(SIXTEEN))
    assert reply.status_code == 422 and reply.json() == {
        "error": "Not applied: Baskets 1 chooses baskets of all the books by Johansen's test, which takes 12 books at "
                 "most: remove it, or keep 12 books or fewer."}
    assert statarb.version == version and statarb.market.n == 12 and len(statarb.runners) == 1
    client.post(f"/api/statarb/traders/{statarb.runners[0].id}/remove")
    assert client.post("/api/statarb/config", data=form(SIXTEEN)).status_code == 200


def test_a_fit_in_sixteen_books_tries_no_johansen_baskets_of_all_the_books_and_refuses_settings_that_are():
    sixteen, twelve = StatArbSession(), StatArbSession()
    sixteen.configure(SIXTEEN)
    base = {**QUICK["baskets"], "legs": 3}
    tried = twelve.plan("baskets", base, {"legs": [0, 3, 4]}, 300, 2, 1)[0]
    assert sorted(c["legs"] for c in tried) == [0, 3, 4]
    tried = sixteen.plan("baskets", base, {"legs": [0, 3, 4]}, 300, 2, 1)[0]
    assert sorted(c["legs"] for c in tried) == [3, 4]
    with pytest.raises(ValueError, match="^your settings need baskets of a few books, or chosen by Box-Tiao: Johan"):
        sixteen.plan("baskets", {**base, "legs": 0}, {"legs": [3]}, 300, 2, 1)
    assert sixteen.plan("baskets", {**base, "legs": 0, "method": "box-tiao"}, {"legs": [0, 3]}, 300, 2, 1)[0]


def test_what_the_tests_see_says_johansen_takes_twelve_books_at_most_where_it_has_no_table_to_read_a_rank_from(
        monkeypatch):
    def johansen(data, det_order, k_ar_diff):
        assert len(data) == 400 and det_order == 0 and k_ar_diff == 2
        return SimpleNamespace(lr1=np.array([40.0, 20.0, 1.0]),
                               cvt=np.array([[25.0, 27.0, 30.0], [12.0, 15.0, 20.0], [1.0, 2.0, 3.0]]))

    monkeypatch.setattr("statsmodels.tsa.vector_ar.vecm.coint_johansen", johansen)
    monkeypatch.setattr("statsmodels.tsa.stattools.coint", lambda x, y, maxlag, autolag: (0.0, 0.01, None))
    twelve = what_the_tests_see(np.zeros((2_000, 12)), sector_loadings(3, 4))
    assert twelve["rank"] == 2 and "johansen" not in twelve and twelve["pairs"] == 66 and twelve["called"] == 66
    monkeypatch.setattr("statsmodels.tsa.vector_ar.vecm.coint_johansen",
                        lambda *args, **kwargs: pytest.fail("it has no critical values for more than 12 books"))
    mids, loadings = np.zeros((2_000, 16)), sector_loadings(4, 4)
    sixteen = what_the_tests_see(mids, loadings)
    assert sixteen == {"units": 2_000, "enough": True, "rank": None, "true_rank": 12, "pairs": 120, "called": 120,
                       "truly": 24, "johansen": "Johansen's test takes 12 books at most, and there are 16"}
    thirteen = what_the_tests_see(np.zeros((2_000, 13)), np.ones((13, 2)) + np.eye(13)[:, :2])
    assert thirteen["rank"] is None and "13" in thirteen["johansen"]
    few = what_the_tests_see(mids[:10], loadings)  # too few mids to say anything: as it was
    assert not few["enough"] and "johansen" not in few and few["true_rank"] == 12


def test_the_status_of_sixteen_books_says_johansens_test_takes_twelve_at_most_and_does_not_print_none(client, statarb):
    assert client.post("/api/statarb/config", data=form(SIXTEEN)).status_code == 200 and statarb.market.n == 16
    client.post("/api/statarb/control", data={"action": "step", "units": 1500})
    reply = client.post("/api/statarb/test").json()
    assert reply["enough"] and reply["rank"] is None and reply["true_rank"] == 12 and reply["pairs"] == 120
    status = html.unescape(render_statarb_panels(statarb.context())["statarb_status"])  # the apostrophe is escaped
    assert "Johansen's test takes 12 books at most, and there are 16" in status
    assert "None" not in status and "Johansen finds" not in status
    called = f"<span class=\"text-slate-100 font-semibold\">{reply['called']}</span>"
    assert f"Engle-Granger calls {called} of 120 pairs" in status


# ---- An index, deals and a trend ----
# Three things a market can have besides its books, each off until a setting turns it on (lob.statarb.FactorValues; the
# library's own tests are test_index_book.py, test_deals.py and test_trend.py, and tests/test_stage4_identity.py holds
# the view's default market to the library's at 42ae383).

NEW = ("index", "index_volatility", "deals", "deal_rate", "deal_premium", "deal_length", "deal_priced", "deal_breaks",
       "trend")
OLD = [p.name for p in MARKET_PARAMS if p.name not in NEW]
# How the view called FactorValues before them (after the loadings), and so must with them off.
OLD_CALL = {"level": 1000.0, "volatility": 1.0, "own_volatility": 0.8, "half_life": 50.0, "impact": 0.05,
            "loading_sd": 0.0, "loading_half_life": 2000.0}
# All three on, away from their defaults, in the sectors market of three by four (a factor of the market and three of
# sectors, so that four factors have a trend of 0 and three of 0.02), with deals often enough for a few hundred units.
EVERYTHING = {**MARKET, "index": True, "index_volatility": 0.7, "deals": True, "deal_rate": 10.0, "deal_premium": 80.0,
              "deal_length": 100.0, "deal_priced": 35.0, "deal_breaks": 15.0, "trend": 0.02, "seed": 4}


@pytest.fixture
def made(monkeypatch):
    """The calls the view makes to ``FactorValues`` while a test runs, as (positional arguments, keywords), each made
    for real."""
    calls, real = [], statarb_module.FactorValues

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(statarb_module, "FactorValues", record)
    return calls


def tags_off(html_text):
    """What a panel says, in words and figures one space apart."""
    return " ".join(re.sub(r"<[^>]+>", " ", html.unescape(html_text)).split())


def start(client, kind, **change):
    """Start a trader of a kind at its quick settings through the page's route."""
    return client.post("/api/statarb/traders/start", data=form({"kind": kind, **QUICK[kind], **change}))


def config_of(page):
    """The settings form, from its opening tag to its closing."""
    start = page.index('<form id="statarb-config"')
    return page[start:page.index("</form>", start)]


def run_until_a_deal_is_on(session, limit=4_000):
    while not session.market.fair.deals:
        session.run_for(10)
        assert session.market.time < limit


def test_the_settings_for_an_index_deals_and_a_trend_are_off_by_default_and_make_the_market_they_say(made):
    assert (MARKET["index"], MARKET["deals"], MARKET["trend"]) == (False, False, 0.0)
    assert {name: MARKET[name] for name in ("index_volatility", "deal_rate", "deal_premium", "deal_length",
                                            "deal_priced", "deal_breaks")} == {
        "index_volatility": 0.4, "deal_rate": 4.0, "deal_premium": 60.0, "deal_length": 500.0, "deal_priced": 40.0,
        "deal_breaks": 20.0}
    market = market_from(MARKET, 1)
    (args, kwargs), = made  # the call made before them, argument by argument
    assert len(args) == 1 and np.array_equal(args[0], sector_loadings(3, 4)) and kwargs == OLD_CALL
    fair = market.fair
    assert fair.n == 12 and fair.index_weights is None and fair.deal_settings is None and not fair.trend.any()
    assert StatArbSession().names == book_names(MARKET) and len(book_names(MARKET)) == 12

    def kwargs_of(**change):
        made.clear()
        market_from({**MARKET, **change}, 1)
        return made[0][1]

    # Each when it is on, and only then: the numbers of one that is off are not passed.
    assert kwargs_of(index=True, index_volatility=0.7) == {**OLD_CALL, "index": True, "index_volatility": 0.7}
    assert kwargs_of(index_volatility=2.0, deal_rate=9.0, deal_priced=60.0) == OLD_CALL
    deals = kwargs_of(deals=True, deal_rate=2.5, deal_premium=80.0, deal_length=300.0, deal_priced=35.0,
                      deal_breaks=15.0)["deals"]  # the rate a thousand units, the chances in percent
    assert deals == Deals(rate=0.0025, premium=80.0, length=300.0, breaks=0.15, priced=0.35, cool=100.0, within=True)
    assert set(kwargs_of(deals=True)) == set(OLD_CALL) | {"deals"}
    assert kwargs_of(trend=0.02) == {**OLD_CALL, "trend": (0.0, 0.02, 0.02, 0.02), "trend_half_life": 500.0}
    made.clear()
    market_from({**MARKET, "shape": "mixed", "books": 6, "factors": 2, "trend": 0.01}, 1)
    assert made[0][1]["trend"] == (0.0, 0.01) and made[0][1]["trend_half_life"] == 500.0  # every factor but the first
    assert kwargs_of(**{k: EVERYTHING[k] for k in NEW}) == {
        **OLD_CALL, "index": True, "index_volatility": 0.7, "trend": (0.0, 0.02, 0.02, 0.02), "trend_half_life": 500.0,
        "deals": Deals(rate=0.01, premium=80.0, length=100.0, breaks=0.15, priced=0.35, cool=100.0, within=True)}


def test_the_market_the_settings_make_is_the_one_the_library_makes_by_hand_over_a_few_hundred_units():
    fair = FactorValues(sector_loadings(3, 4), level=1000.0, volatility=1.0, own_volatility=0.8, half_life=50.0,
                        impact=0.05, index=True, index_volatility=0.7, trend=(0.0, 0.02, 0.02, 0.02),
                        deals=Deals(rate=0.01, premium=80.0, length=100.0, breaks=0.15, priced=0.35))
    by_hand, view = BookMarket(fair, seed=4), market_from(EVERYTHING, 4)
    a, b = by_hand.run(400.0), view.run(400.0)
    assert a.shape == (400, 1 + 2 * 13) and np.array_equal(a.to_numpy(float), b.to_numpy(float))
    assert view.fair.fairs == by_hand.fair.fairs and view.events == by_hand.events and view.n == 13
    assert len(view.fair.ended) + len(view.fair.deals) >= 1  # a deal came
    assert view.fair.drift[0] == 0.0 and view.fair.drift[1:].all()  # the trend: every factor but the first
    for name, off in (("index", False), ("deals", False), ("trend", 0.0)):  # and each is in what the market did
        other = market_from({**EVERYTHING, name: off}, 4)
        assert not np.array_equal(other.run(400.0).to_numpy(float)[:, 1:13], a.to_numpy(float)[:, 1:13]), name


def test_a_form_without_the_new_fields_gives_the_old_market(client, statarb, made):
    old_form = form({name: MARKET[name] for name in OLD} | {"shape": "sectors"})  # what a page of before sent
    values, errors = parse_market(old_form)
    assert not errors and values == MARKET
    empty, errors = parse_market({})  # nothing at all: every number its default, and each switch off, as a form has it
    assert not errors and {name: empty[name] for name in NEW} == {name: MARKET[name] for name in NEW}
    assert {name: values[name] for name in NEW} == {p.name: p.default for p in MARKET_PARAMS if p.name in NEW}
    made.clear()
    reply = client.post("/api/statarb/config", data=old_form)
    assert reply.status_code == 200 and "12 books, 9 relations" in reply.json()["message"]
    assert statarb.settings == MARKET and statarb.market.n == 12 and statarb.names == book_names(MARKET)
    assert made[-1][1] == OLD_CALL
    old_settings = {name: MARKET[name] for name in OLD} | {"shape": "sectors"}  # a dict that lacks the new keys
    assert book_names(old_settings) == book_names(MARKET)
    assert np.array_equal(market_from(old_settings, 1).run(60.0).to_numpy(float),
                          market_from(MARKET, 1).run(60.0).to_numpy(float))
    assert not any(statarb.market.fair.deals) and statarb.groups() == [0] * 4 + [1] * 4 + [2] * 4


def test_the_context_and_the_history_are_what_they_were_but_for_their_new_keys(client, statarb):
    assert set(statarb.context()) == {
        "time", "events", "running", "speed", "books", "rows", "settings", "n", "factors", "relations", "floor",
        "pushed", "half_life", "drifting", "full", "test", "kept", "fitting", "deals"}
    assert statarb.context()["deals"] is None and statarb.context()["settings"] == MARKET
    seen = read_stream(client, {"statarb_history", "statarb_state"})
    history = json.loads(seen["statarb_history"])
    assert set(history) == {"version", "samples", "traders", "marks", "mark_seq", "settings", "speed", "running",
                            "names", "groups", "index"}
    assert history["index"] is None and len(history["names"]) == 12 and history["groups"][-1] == 2
    assert set(history["settings"]) == {p.name for p in MARKET_PARAMS} | {"shape"} and history["settings"] == MARKET
    assert set(json.loads(seen["statarb_state"])) == {"running", "speed", "version", "traders", "settings", "changes",
                                                      "fitting"}


def test_an_index_adds_a_book_named_idx_with_a_group_of_its_own_and_everything_that_lists_the_books_counts_thirteen(
        client, statarb):
    assert "IDX" not in statarb.names and statarb.history()["index"] is None
    reply = client.post("/api/statarb/config", data=form({**MARKET, "index": True}))
    assert reply.status_code == 200 and "13 books, 10 relations" in reply.json()["message"]
    m = statarb.market
    assert m.n == 13 and len(m.books) == 13 and m.fair.m == 12 and len(statarb.names) == 13
    assert statarb.names == [*book_names(MARKET), "IDX"] and statarb.names[-1] == "IDX"
    assert statarb.groups() == [0] * 4 + [1] * 4 + [2] * 4 + [3]  # a group of its own, not a sector's
    history = statarb.history()
    assert history["names"] == statarb.names and history["groups"] == statarb.groups() and history["index"] == 12
    statarb.run_for(200)
    sample = statarb.samples[-1]
    assert len(sample["mids"]) == 13 and len(statarb.mids[-1]) == 13
    assert 0.97 < sample["mids"][12] / sum(sample["mids"][:12]) < 1.03  # its own price: theirs summed, and a premium
    rows = statarb.book_rows()
    assert [r["name"] for r in rows][-2:] == ["C4", "IDX"] and rows[12]["mid"] == m.books[12].mid_price > 10_000
    ctx = statarb.context()
    assert (ctx["n"], ctx["factors"], ctx["relations"]) == (13, 3, 10)
    panels = render_statarb_panels(ctx)
    assert panels["statarb_books"].count("data-statarb-book=") == 13 and 'data-statarb-book="IDX"' in panels[
        "statarb_books"]
    status = tags_off(panels["statarb_status"])
    assert 'data-books="13"' in panels["statarb_status"] and "13 books, 3 factors, 10 relations" in status
    # A trader that scores every book has a line, a relation and a chip for each of the thirteen, the index last.
    runner, _ = statarb.start_trader(QUICK["reversal"], "reversal")
    while not runner.trader.lots:
        statarb.run_for(100)
        assert m.time < 3_000
    row = statarb.board()[0]
    assert row["total"] == 13 and len(row["relations"]) == 13 <= statarb.signal_cap()  # all of them
    assert [r["text"] for r in row["relations"]] == statarb.names and row["relations"][-1]["text"] == "IDX"
    assert len(statarb.samples[-1]["sig"][runner.id]) == 13 and len(statarb.book_rows()[0]["held"]) == 1
    board = render_statarb_panels(statarb.context())["statarb_board"]
    assert f"holding {row['holding']} of 13" in board and board.count('<span class="px-1 rounded border') == 13
    assert len(runner.trader.position) == 13 and statarb.book_rows()[12]["held"][0]["lots"] == int(
        runner.trader.position[12])


def test_without_sectors_an_index_is_a_group_of_its_own_after_the_one_there_is_and_it_goes_when_it_is_turned_off():
    session = StatArbSession()
    session.configure({**MARKET, "shape": "mixed", "books": 7, "factors": 2, "index": True})
    assert session.names == ["1", "2", "3", "4", "5", "6", "7", "IDX"] and session.market.n == 8
    assert session.groups() == [0] * 7 + [1] and session.history()["index"] == 7
    session.configure({**MARKET, "shape": "mixed", "books": 7, "factors": 2})
    assert session.names[-1] == "7" and session.groups() == [0] * 7 and session.history()["index"] is None
    session.configure({**MARKET, "sectors": 4, "books_each": 4, "index": True})  # a fourth sector: the index is a fifth
    assert session.groups() == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4 + [4] and session.names[-1] == "IDX"


def test_the_tests_say_johansens_takes_twelve_books_at_most_in_thirteen_with_an_index():
    session = StatArbSession()
    session.configure({**MARKET, "index": True})
    session.run_for(1_100)
    result = asyncio.run(session.test_relations())
    assert result["enough"] and result["rank"] is None and result["true_rank"] == 10 and result["pairs"] == 78
    assert result["johansen"] == "Johansen's test takes 12 books at most, and there are 13"
    status = tags_off(render_statarb_panels(session.context())["statarb_status"])
    assert "Johansen's test takes 12 books at most, and there are 13" in status and "None" not in status
    assert "of 78 pairs cointegrated" in status


def test_the_status_says_which_deals_are_on_with_target_acquirer_gap_and_time_left():
    session = StatArbSession()
    assert session.context()["deals"] is None
    assert "data-statarb-deals" not in render_statarb_panels(session.context())["statarb_status"]
    session.configure(DEALS)
    assert session.context()["deals"] == {"on": [], "completed": 0, "broken": 0}
    status = render_statarb_panels(session.context())["statarb_status"]
    assert "data-statarb-deals" in status and "Deals 0 on, 0 completed, 0 broken" in tags_off(status)
    run_until_a_deal_is_on(session)
    m, fair = session.market, session.market.fair
    line = session.context()["deals"]
    json.dumps(line)  # plain data: the panels are rendered off the event loop
    assert len(line["on"]) == len(fair.deals) >= 1
    for shown, deal in zip(line["on"], fair.deals, strict=True):
        assert (shown["target"], shown["acquirer"]) == (session.names[deal.target], session.names[deal.acquirer])
        gap = deal.ratio * m.books[deal.acquirer].mid_price - m.books[deal.target].mid_price  # r * mid_A - mid_T
        assert shown["gap"] == pytest.approx(gap, abs=0.051) and shown["left"] == round(deal.ends - m.time)
        assert 0 < shown["left"] <= 100 and shown["ratio"] == pytest.approx(deal.ratio, abs=5e-5)
    status = render_statarb_panels(session.context())
    text = tags_off(status["statarb_status"])
    assert f'data-on="{len(line["on"])}"' in status["statarb_status"]
    assert status["statarb_status"].count("data-statarb-deal ") == len(line["on"])
    for shown in line["on"]:
        assert f"{shown['target']} bought by {shown['acquirer']} gap {shown['gap']:+.1f} left {shown['left']}" in text
    # How many ended, completed and broken, counted from the market's own record, and again after more of them.
    for _ in range(2):
        while True:
            session.run_for(100)
            assert m.time < 8_000
            outcomes = [d.outcome for d in fair.ended]
            if outcomes.count("completed") >= 2 and outcomes.count("broken") >= 2:
                break
        line = session.context()["deals"]
        outcomes = [d.outcome for d in fair.ended]
        assert (line["completed"], line["broken"]) == (outcomes.count("completed"), outcomes.count("broken"))
        assert len(outcomes) == line["completed"] + line["broken"]
        assert session.context()["deals"] == line  # a second look counts nothing twice
        shown = tags_off(render_statarb_panels(session.context())["statarb_status"])
        assert f"{line['completed']} completed, {line['broken']} broken" in shown
        session.run_for(300)
    session.reset()  # a new market has ended none, whatever the old one had
    assert session.context()["deals"] == {"on": [], "completed": 0, "broken": 0}
    run_until_a_deal_is_on(session)
    session.run_for(500)
    line = session.context()["deals"]
    outcomes = [d.outcome for d in session.market.fair.ended]
    assert (line["completed"], line["broken"]) == (outcomes.count("completed"), outcomes.count("broken"))
    session.configure(MARKET)  # and none at all when the deals are turned off
    assert session.context()["deals"] is None


def test_the_board_and_the_samples_read_exposure_through_the_deals_in_the_fair_values():
    session = StatArbSession()
    session.configure(DEALS)
    first, _ = session.start_trader(QUICK["pairs"], "pairs")
    session.set_active(first, False)
    run_until_a_deal_is_on(session)
    fair, trader = session.market.fair, first.trader
    deal = fair.deals[0]
    trader.position[deal.target] = 5  # five lots of a target in a deal: it holds what the offer's worth of its acquirer
    trader.position[(deal.target + 1) % 12] += 3
    exposures = fair.exposures()
    assert exposures is not fair.loadings and not np.array_equal(exposures, fair.loadings)  # the deal changed a row
    wanted = float(np.linalg.norm(trader.position @ exposures))
    assert wanted != pytest.approx(float(np.linalg.norm(trader.position @ fair.loadings)), rel=1e-6)
    assert session.board()[0]["exposure"] == pytest.approx(wanted)
    session.market.step()
    session._sample()
    sample = session.samples[-1]
    assert sample["exposure"][first.id] == pytest.approx(
        float(np.linalg.norm(trader.position @ fair.exposures())), abs=0.01)
    while fair.deals:  # with none on it is the loadings, as it was
        session.run_for(50)
        assert session.market.time < 6_000
    assert fair.exposures() is fair.loadings
    assert session.board()[0]["exposure"] == pytest.approx(float(np.linalg.norm(trader.position @ fair.loadings)))


def test_the_trend_is_on_every_factor_but_the_first():
    session = StatArbSession()
    assert not session.market.fair.trend.any()
    session.configure({**MARKET, "trend": 0.03})
    fair = session.market.fair
    assert fair.k == 4 and fair.trend.tolist() == [0.0, 0.03, 0.03, 0.03] and fair.trend_half_life == 500.0
    session.run_for(100)
    assert fair.drift[0] == 0.0 and fair.drift[1:].all()
    session.configure({**MARKET, "shape": "mixed", "books": 6, "factors": 3, "trend": 0.05})
    assert session.market.fair.trend.tolist() == [0.0, 0.05, 0.05]
    session.configure({**MARKET, "shape": "mixed", "books": 6, "factors": 1})  # one factor, and no trend asked of it
    assert session.market.fair.trend.tolist() == [0.0]


@pytest.mark.parametrize("change", [
    {"index_volatility": 0.0}, {"index_volatility": 3.0, "deal_priced": 0.0, "deal_breaks": 0.0},
    {"deal_priced": 95.0, "deal_breaks": 95.0, "deal_premium": 200.0}, {"deal_rate": 0.5, "deal_length": 2_000.0},
    {"deal_rate": 10.0, "deal_length": 50.0, "deal_premium": 5.0, "trend": 0.05}])
def test_every_setting_at_the_ends_of_its_range_makes_a_market_that_runs(client, statarb, change):
    reply = client.post("/api/statarb/config", data=form({**MARKET, "index": True, "deals": True, **change}))
    assert reply.status_code == 200 and statarb.market.n == 13
    statarb.run_for(120)
    assert all(b.mid_price is not None for b in statarb.market.books) and len(statarb.samples[-1]["mids"]) == 13


def test_the_page_offers_the_new_settings_off_inside_the_one_form(client, statarb):
    page = client.get("/").text
    config = config_of(page)
    assert page.count('id="statarb-config"') == 1 and config.count("<form") == 1  # one form, and no other inside it
    start = config.index('<details id="statarb-more"')
    details = config[start:config.index("</details>", start)]
    assert " open" not in details[:details.index(">")]  # collapsed
    assert config.index("name=\"loading_half_life\"") < start and config.index('type="submit"') > start
    for p in MARKET_PARAMS:
        assert (f'name="{p.name}"' in details) == (p.name in NEW), p.name  # these inside it, the others outside
        assert (f'name="{p.name}"' in config) and f'name="{p.name}"' in page
        if p.name in NEW:
            assert len(p.label) <= 26 and "mixed" not in p.label.lower()  # one line in a narrow column
            assert re.search(rf">\s*{re.escape(p.label)}\s*<", html.unescape(details)), p.label  # shown, as a label
    for p in MARKET_PARAMS:
        tag = re.search(rf'<input[^>]*name="{p.name}"[^>]*>', config).group(0)
        if isinstance(p, Switch):  # on only when it is on in the settings: these are the defaults
            assert 'type="checkbox"' in tag and ("checked" in tag) == p.default
        else:
            assert f'value="{p.default}"' in tag and f'min="{p.low}"' in tag and f'max="{p.high}"' in tag
    for name in ("index", "deals"):
        assert 'type="checkbox"' in re.search(rf'<input[^>]*name="{name}"[^>]*>', details).group(0)
        assert "checked" not in re.search(rf'<input[^>]*name="{name}"[^>]*>', details).group(0)
    assert details.count("grid-cols-2") >= 2  # two columns
    text = html.unescape(details)
    for hint in ("so one lot of it is a lot of each book", "the order flow's own impact still makes a premium",
                 "closes by the deadline unless the deal breaks", "is a risk premium",
                 "a premium of its own that mean-reverts as the own parts do",  # with them off it wanders
                 "Announced over all the books in a thousand units of time, never more than one",
                 "above about 0.04 (0.02 in a mixed market) can reach the floor, which then pushes the first factor"):
        assert hint in text, hint
    # Notebook 15 turns each of the three up from nothing: the old pointer is gone, and the paragraph sends the reader
    # to the hover texts of the three dials, which have its figures (test_the_summaries_and_hints_say_what_notebook_15
    # _printed_word_for_word)
    assert "Notebook 15 has what they do" not in text and "what they do to the traders" not in text
    assert ("Notebook 15 turns each up from nothing and says where a trader breaks even: the hover texts of Its "
            "premium's volatility, Break in truth and Factor trend have its figures.") in text
    assert "premium of its own that mean-reverts," not in text and "all the books, a thousand" not in text
    # With them on, as the server has them, they show on, and the details are open.
    assert client.post("/api/statarb/config", data=form({**MARKET, "index": True, "trend": 0.02})).status_code == 200
    page = client.get("/").text
    config = config_of(page)
    assert re.search(r'<details id="statarb-more"[^>]* open', config)
    assert "checked" in re.search(r'<input[^>]*name="index"[^>]*>', config).group(0)
    assert "checked" not in re.search(r'<input[^>]*name="deals"[^>]*>', config).group(0)
    assert re.search(r'<input[^>]*name="trend"[^>]*value="0.02"', config)


# ---- A kind that trades what the market may lack ----
# Two kinds trade something the default market has not (``KIND_TRADES``): index arbitrage the index book, and the deals
# kind deals. These tests name a kind of the eight there as well, so that they hold the rule and not only the two kinds
# that follow it today: refused at a start, in a fit's plan and by new settings, with nothing started and no number
# taken.

LACKS_INDEX = "this market lacks an index book: turn it on in the market's settings and apply"


def test_what_a_kind_trades_that_the_market_lacks_is_said_in_one_place_and_only_the_two_kinds_have_any(monkeypatch):
    session = StatArbSession()
    assert statarb_module.KIND_TRADES == {"index": ("index",), "deals": ("deals",)}
    assert list(TRADER_KINDS)[-2:] == ["index", "deals"] and list(TRADER_KINDS)[0] == "pairs"  # after the eight
    for kind in TRADER_KINDS:
        has_none = kind not in ("index", "deals")
        assert (market_lacks(kind, session.market.fair) is None) == has_none and (market_lacks(kind, {}) is None) == (
            has_none)
        assert (market_lacks(kind, session.settings) is None) == has_none
    assert market_lacks("index", MARKET) == "an index book" and market_lacks("deals", MARKET) == "deals"
    assert market_lacks("index", session.market.fair) == "an index book"
    assert market_lacks("deals", session.market.fair) == "deals"
    with_index, with_deals = market_from({**MARKET, "index": True}, 1).fair, market_from(DEALS, 1).fair
    assert market_lacks("index", with_index) is None and market_lacks("index", with_deals) == "an index book"
    assert market_lacks("deals", with_deals) is None and market_lacks("deals", with_index) == "deals"
    assert market_lacks("index", {**MARKET, "index": True}) is None and market_lacks("deals", DEALS) is None
    assert market_lacks("deals", {**MARKET, "index": True}) == "deals" and market_lacks("index", DEALS) == (
        "an index book")
    monkeypatch.setitem(statarb_module.KIND_TRADES, "pairs", ("index",))
    monkeypatch.setitem(statarb_module.KIND_TRADES, "baskets", ("deals", "index"))
    assert market_lacks("pairs", session.market.fair) == "an index book" == market_lacks("pairs", MARKET)
    assert market_lacks("pairs", {}) == "an index book"  # settings without it, as a form of before has them
    assert market_lacks("baskets", MARKET) == "deals" and market_lacks("residuals", MARKET) is None
    assert market_lacks("pairs", with_index) is None and market_lacks("pairs", {**MARKET, "index": True}) is None
    assert market_lacks("baskets", with_index) == "deals"
    assert market_lacks("baskets", {**MARKET, "index": True}) == "deals"
    assert market_lacks("baskets", with_deals) == "an index book" and market_lacks("baskets", DEALS) == "an index book"
    both = {**MARKET, "index": True, "deals": True}
    assert market_lacks("baskets", both) is None and market_lacks("baskets", market_from(both, 1).fair) is None


def test_a_kind_that_trades_what_the_market_lacks_is_not_started_and_takes_no_number(client, statarb, monkeypatch):
    monkeypatch.setitem(statarb_module.KIND_TRADES, "pairs", ("index",))
    reply = client.post("/api/statarb/traders/start", data=form({"kind": "pairs", **QUICK["pairs"]}))
    assert reply.status_code == 422 and reply.json() == {"error": f"Not started: {LACKS_INDEX}."}
    assert not statarb.runners and statarb.market.traders == [] and statarb._seq == 0
    assert start(client, "baskets").status_code == 200
    changes = statarb.changes
    with pytest.raises(ValueError, match="^" + LACKS_INDEX + "$"):
        statarb.start_trader(QUICK["pairs"], "pairs")
    assert statarb.changes == changes and len(statarb.runners) == 1
    client.post("/api/statarb/config", data=form({**MARKET, "index": True}))  # turned on, as it says, and applied
    reply = client.post("/api/statarb/traders/start", data=form({"kind": "pairs", **QUICK["pairs"]}))
    assert reply.status_code == 200 and reply.json()["name"] == "Pairs 2" and statarb._seq == 2  # no number was lost


def test_a_fit_of_a_kind_that_trades_what_the_market_lacks_is_refused_before_it_runs(client, statarb, monkeypatch):
    monkeypatch.setitem(statarb_module.KIND_TRADES, "pairs", ("index",))
    grid = {"entry": [1.5, 2.0]}
    with pytest.raises(ValueError, match="^" + LACKS_INDEX + "$"):
        statarb.plan("pairs", QUICK["pairs"], grid, 300, 2, 1)
    data = {"kind": "pairs", **form(QUICK["pairs"]), "fit_entry": "on", "grid_entry": "1.5, 2", "span": 500, "runs": 2,
            "objective": "pnl", "fit_seed": 1}
    reply = client.post("/api/statarb/traders/fit", data=data)
    assert reply.status_code == 422 and reply.json() == {"error": f"Not fitted: {LACKS_INDEX}."}
    assert statarb.fitting is None and statarb.last_fit is None
    client.post("/api/statarb/config", data=form({**MARKET, "index": True}))
    assert statarb.plan("pairs", QUICK["pairs"], grid, 300, 2, 1)[0]  # in a market with one, it plans


@pytest.mark.parametrize("ticked, asks", [
    ({}, "fit"), ({"fit_window": "on", "grid_window": "many"}, "grid_window"),
    ({"fit_entry": "on", "grid_entry": ""}, "grid_entry")], ids=["nothing ticked", "a bad value", "no value"])
def test_a_fit_of_the_index_kind_in_a_market_without_one_says_so_before_it_asks_for_ticks_or_values(
        client, statarb, ticked, asks):
    """Ticking settings, or mending the grid, would not make the market one with an index: say that first, so that the
    user is not sent to tick a setting and then told the market has no index."""
    data = {"kind": "index", **form(QUICK["index"]), **ticked, "span": 500, "runs": 2, "objective": "pnl",
            "fit_seed": 1}
    reply = client.post("/api/statarb/traders/fit", data=data)
    assert reply.status_code == 422
    assert reply.json() == {"error": "Not fitted: this market has no index book: turn it on in the market's settings "
                                     "and apply."}
    assert statarb.fitting is None and statarb.last_fit is None
    client.post("/api/statarb/config", data=form({**MARKET, "index": True}))  # in a market with one, the grid is asked
    reply = client.post("/api/statarb/traders/fit", data=data)
    assert reply.status_code == 422 and list(reply.json()["errors"]) == [asks]
    assert statarb.fitting is None and statarb.last_fit is None


def test_new_settings_that_take_what_a_trader_on_the_board_trades_are_refused_and_change_nothing(client, statarb,
                                                                                                monkeypatch):
    monkeypatch.setitem(statarb_module.KIND_TRADES, "pairs", ("index",))
    monkeypatch.setitem(statarb_module.KIND_TRADES, "copula", ("deals",))
    client.post("/api/statarb/config", data=form({**MARKET, "index": True}))
    assert client.post("/api/statarb/traders/start", data=form({"kind": "pairs", **QUICK["pairs"]})).status_code == 200
    version, settings = statarb.version, dict(statarb.settings)
    reply = client.post("/api/statarb/config", data=form(MARKET))  # the index taken away
    assert reply.status_code == 422 and reply.json() == {
        "error": "Not applied: Pairs 1 needs an index book, which these settings leave out: remove it, or keep an "
                 "index book in the market."}
    assert statarb.version == version and statarb.settings == settings and statarb.market.n == 13
    assert client.post("/api/statarb/config", data=form({**MARKET, "index": True, "seed": 9})).status_code == 200
    assert statarb.market.n == 13 and statarb.runners[0].trader in statarb.market.traders  # kept, and it goes on
    # The deals, for a kind that trades them: the market that has the index but not them refuses to start it, and
    # settings that turn them on are applied when it is on the board, and refused when they take them away.
    assert start(client, "copula").status_code == 422
    assert client.post("/api/statarb/config", data=form({**MARKET, "index": True, "deals": True})).status_code == 200
    assert start(client, "copula").status_code == 200
    reply = client.post("/api/statarb/config", data=form({**MARKET, "index": True}))
    assert reply.status_code == 422 and "Copula pairs 2 needs deals, which these settings leave out" in reply.json()[
        "error"]
    assert statarb.market.fair.deal_settings is not None and len(statarb.runners) == 2


def test_a_trader_is_checked_against_the_market_before_it_joins_it_and_what_it_says_ends_as_not_started(
        client, statarb, monkeypatch):
    seen = []

    def check(self, market):
        seen.append((market is statarb.market, self.position is None))  # before it was attached
        raise ValueError("this market has no deals")

    monkeypatch.setattr(RelationTrader, "check", check)
    reply = client.post("/api/statarb/traders/start", data=form({"kind": "pairs", **QUICK["pairs"]}))
    assert reply.status_code == 422 and reply.json() == {"error": "Not started: this market has no deals."}
    assert seen == [(True, True)] and not statarb.runners and statarb.market.traders == [] and statarb._seq == 0
    with pytest.raises(ValueError, match="your settings need|this market has no deals"):
        statarb.plan("pairs", QUICK["pairs"], {"entry": [1.5, 2.0]}, 300, 2, 1)
    with pytest.raises(ValueError, match="^this market has no deals$"):  # its own sentence, as it is
        statarb.plan("pairs", QUICK["pairs"], {"entry": [1.5, 2.0]}, 300, 2, 1)
    ok = client.post("/api/statarb/traders/start", data=form({"kind": "residuals", **QUICK["residuals"]}))
    assert ok.status_code == 200 and ok.json()["name"] == "Residuals 1"  # another kind, whose check says nothing
    monkeypatch.undo()
    assert client.post("/api/statarb/traders/start", data=form({"kind": "pairs", **QUICK["pairs"]})).status_code == 200


@pytest.mark.parametrize("kind", TRADER_KINDS)
def test_every_kind_starts_in_a_market_with_an_index_deals_and_a_trend_and_thirteen_books_show(client, statarb, kind):
    assert client.post("/api/statarb/config", data=form({**EVERYTHING, "seed": 1})).status_code == 200
    values = QUICK[kind] if kind != "baskets" else {**QUICK[kind], "formation": 100, "window": 20}
    reply = client.post("/api/statarb/traders/start", data=form({"kind": kind, **values}))
    assert reply.status_code == 200, reply.json()
    runner = statarb.runners[0]
    assert len(runner.trader.position) == 13 and statarb.market.traders == [runner.trader]
    statarb.run_for(150)
    row = statarb.board()[0]
    assert row["pnl"] == pytest.approx(row["before"] - row["cost"]) and row["exposure"] >= 0
    sample = statarb.samples[-1]
    assert len(sample["mids"]) == 13 and sample["exposure"][runner.id] >= 0
    panels = render_statarb_panels(statarb.context())
    assert panels["statarb_books"].count("data-statarb-book=") == 13
    assert "data-statarb-deals" in panels["statarb_status"]


# ---- Two kinds of trader that trade what the extra settings add: index arbitrage and deals ----

NO_INDEX = "this market has no index book: turn it on in the market's settings and apply"
NO_DEALS = "this market has no deals: turn it on in the market's settings and apply"


def test_each_new_kind_says_what_it_does_with_its_defaults_word_for_word():
    say = TRADER_KINDS["index"].describe, TRADER_KINDS["deals"].describe
    assert say[0](DEFAULTS["index"]) == (
        "The spread of the index IDX over the books it is the sum of, its z-score over the last 400 decisions, every 5 "
        "units. Above 2: short 1 basket, selling a lot of the index and buying a lot of every book for each; below "
        "−2 the reverse; flat back at its mean.")
    assert say[1](DEFAULTS["deals"]) == (
        "Each stock deal that is on, entered as soon as the offer is worth at least the target in the books' mids: "
        "buys 5 lots of its target and sells the offer's ratio of that in lots of its acquirer, holds them until the "
        "deal is over, and then closes both, deciding every 5 units.")
    # what each setting changes in the sentence, and the trading less that every kind says in the same words
    assert say[0]({**DEFAULTS["index"], "size": 3, "exit": 0.5, "entry": 1.5, "window": 200, "interval": 1.0}) == (
        "The spread of the index IDX over the books it is the sum of, its z-score over the last 200 decisions, every 1 "
        "unit. Above 1.5: short 3 baskets, selling a lot of the index and buying a lot of every book for each; below "
        "−1.5 the reverse; flat within 0.5 of its mean.")
    assert say[0]({**DEFAULTS["index"], "hedged": False, "size": 2, "exit": 0.5}) == (
        "The spread of the index IDX over the books it is the sum of, its z-score over the last 400 decisions, every 5 "
        "units. Above 2: sell 2 lots of the index; below −2 buy them; flat within 0.5 of its mean. None of the "
        "books, so it holds what the index holds of the factors.")
    assert "sell 1 lot of the index; below −2 buy it;" in say[0]({**DEFAULTS["index"], "hedged": False})
    assert say[1]({**DEFAULTS["deals"], "entry": 8.0, "limit": 2, "size": 2.5, "interval": 1.0}) == (
        "Each stock deal that is on, entered once the offer is worth 8 ticks more than the target in the books' mids: "
        "buys 2.5 lots of its target and sells the offer's ratio of that in lots of its acquirer, holds them until the "
        "deal is over, and then closes both, holding at most 2 deals at once, the widest gap first, deciding every 1 "
        "unit.")
    assert "once the offer is worth 1 tick more than" in say[1]({**DEFAULTS["deals"], "entry": 1.0})
    assert "holding at most 1 deal at once" in say[1]({**DEFAULTS["deals"], "limit": 1})
    for kind in ("index", "deals"):  # and trading less, as each kind says it
        assert TRADER_KINDS[kind].describe({**DEFAULTS[kind], "rate": 0.5, "band": 1.0}).endswith(
            " It goes 0.5 of the way to its target at each decision and does not trade within 1 lot of that, except "
            "to close a book.")


def test_the_two_new_kinds_have_the_settings_the_view_says_and_the_others_have_none_of_them():
    index, deals = TRADER_KINDS["index"], TRADER_KINDS["deals"]
    assert DEFAULTS["index"] == {"window": 400, "entry": 2.0, "exit": 0.0, "hedged": True, "size": 1, "rate": 1.0,
                                 "band": 0.0, "interval": 5.0, "warm": True}
    assert DEFAULTS["deals"] == {"entry": 0.0, "limit": 0, "size": 5.0, "rate": 1.0, "band": 0.0, "interval": 5.0}
    limits = {p.name: (p.low, p.high, p.integer) for p in index.numbers}
    assert limits["window"] == (3, 2000, True) and limits["entry"] == (0.1, 10.0, False)
    assert limits["exit"] == (0.0, 9.9, False) and limits["size"] == (1, 10, True)
    limits = {p.name: (p.low, p.high, p.integer) for p in deals.numbers}
    assert limits["entry"] == (0.0, 100.0, False) and limits["limit"] == (0, 6, True)
    assert limits["size"][:2] == (0.5, 50)
    assert [p.label for p in index.params if isinstance(p, Switch)] == [
        "Hedge with the books", "Learn from the past first"]
    assert [p.name for p in deals.params if isinstance(p, Switch)] == []  # nothing to learn, so no switch for it
    assert [p.label for p in deals.numbers][:3] == ["Enter at a gap of (ticks)", "Deals at once (0: any)",
                                                    "Lots of the target"]
    assert index.fit_default == ("window", "entry") and index.fit_grid["window"] == (200, 400, 800)
    assert index.fit_grid["entry"] == (1.0, 1.5, 2.0) and not index.no_fit and deals.no_fit
    for kind, spec in TRADER_KINDS.items():  # only these two have no relation in the sense of the others'
        assert ("hedged" in DEFAULTS[kind]) == (kind == "index") and ("limit" in DEFAULTS[kind]) == (kind == "deals")
        assert ("warm" in DEFAULTS[kind]) == (kind != "deals") and bool(spec.no_fit) == (kind == "deals")
        assert spec.summary and ("notebook 15" in spec.summary.lower()) == (kind in NOTEBOOK_15_KINDS)
    # and what they quote of it is the sum index's, the one the view builds, and not the 1/12 index's, which it does
    # not offer: only a clause says that an index quoted like one more book needs about twice the premium
    assert "1/12" not in index.summary and "32.5" not in index.summary and "twelfth" not in index.summary
    assert "An index quoted like one more book needed a premium about twice as large to break even." in index.summary


def test_the_index_kind_makes_the_trader_the_settings_say():
    t = TRADER_KINDS["index"].build({**DEFAULTS["index"], "window": 50, "entry": 1.5, "exit": 0.5, "size": 3,
                                     "rate": 0.5, "band": 1.0, "interval": 2.0, "hedged": False})
    assert type(t) is IndexTrader and (t.legs, t.window, t.entry, t.exit, t.size) == (0, 50, 1.5, 0.5, 3)
    assert (t.rate, t.band, t.interval, t.hedged, t.formation) == (0.5, 1.0, 2.0, False, 50)
    assert TRADER_KINDS["index"].build(DEFAULTS["index"]).hedged is True
    for change in ({"exit": 2.0}, {"window": 2}, {"size": 0}):  # the library's own refusals, which the form passes on
        assert not TRADER_KINDS["index"].valid({**DEFAULTS["index"], **change}), change
    t = TRADER_KINDS["deals"].build({**DEFAULTS["deals"], "entry": 12.0, "limit": 3, "size": 2.5, "rate": 0.5,
                                     "band": 1.0, "interval": 2.0})
    assert type(t) is DealTrader and (t.entry, t.limit, t.size, t.rate, t.band, t.interval) == (
        12.0, 3, 2.5, 0.5, 1.0, 2.0)
    for change in ({"size": 0.0}, {"entry": -1.0}, {"limit": -1}):
        assert not TRADER_KINDS["deals"].valid({**DEFAULTS["deals"], **change}), change


@pytest.mark.parametrize("kind, lacks, settings", [("index", NO_INDEX, {**MARKET, "index": True}),
                                                   ("deals", NO_DEALS, DEALS)])
def test_each_new_kind_is_refused_a_market_without_what_it_trades_with_the_librarys_sentence_and_takes_no_number(
        client, statarb, kind, lacks, settings):
    reply = client.post("/api/statarb/traders/start", data=form({"kind": kind, **QUICK[kind]}))
    assert reply.status_code == 422 and reply.json() == {"error": f"Not started: {lacks}."}
    assert not statarb.runners and statarb.market.traders == [] and statarb._seq == 0
    changes = statarb.changes
    with pytest.raises(ValueError, match="^" + lacks + "$"):
        statarb.start_trader(QUICK[kind], kind)
    assert statarb.changes == changes and statarb._seq == 0
    for other in ("residuals", "pairs"):  # the eight go on starting in this market
        assert start(client, other).status_code == 200
    assert client.post("/api/statarb/config", data=form(settings)).status_code == 200  # turned on, as it says
    reply = client.post("/api/statarb/traders/start", data=form({"kind": kind, **QUICK[kind]}))
    assert reply.status_code == 200 and reply.json()["name"] == f"{TRADER_KINDS[kind].label} 3"  # no number was lost
    assert any(t is statarb.runners[-1].trader for t in statarb.market.traders) and statarb._seq == 3


@pytest.mark.parametrize("kind, needs, settings, takes_away", [
    ("index", "an index book", {**MARKET, "index": True}, MARKET),
    ("deals", "deals", DEALS, {**DEALS, "deals": False})])
def test_new_settings_that_take_the_index_or_the_deals_from_a_trader_on_the_board_are_refused_and_change_nothing(
        client, statarb, kind, needs, settings, takes_away):
    assert client.post("/api/statarb/config", data=form(settings)).status_code == 200
    assert start(client, kind).status_code == 200
    statarb.run_for(40)
    version, applied, time = statarb.version, dict(statarb.settings), statarb.market.time
    reply = client.post("/api/statarb/config", data=form(takes_away))
    assert reply.status_code == 422 and reply.json() == {
        "error": f"Not applied: {TRADER_KINDS[kind].label} 1 needs {needs}, which these settings leave out: remove it, "
                 f"or keep {needs} in the market."}
    assert statarb.version == version and statarb.settings == applied and statarb.market.time == time
    assert [r.kind for r in statarb.runners] == [kind] and statarb.market.traders[0] is statarb.runners[0].trader
    assert client.post("/api/statarb/config", data=form({**settings, "seed": 5})).status_code == 200  # kept: it goes on
    assert statarb.market.time < time and statarb.market.traders[0] is statarb.runners[0].trader
    client.post(f"/api/statarb/traders/{statarb.runners[0].id}/remove")  # taken off the board, they are applied
    assert client.post("/api/statarb/config", data=form(takes_away)).status_code == 200
    assert market_lacks(kind, statarb.market.fair) == needs


def test_the_message_of_a_start_is_true_for_a_trader_with_nothing_to_learn_or_choose(client, statarb):
    client.post("/api/statarb/config", data=form({**DEALS, "index": True}))
    said = client.post("/api/statarb/traders/start", data=form({"kind": "deals", **QUICK["deals"]})).json()["message"]
    assert said == "Deals 1 has nothing to learn from the past: it trades from its next decision."
    assert "learned" not in said and "0 points" not in said and "choose" not in said
    # the index trader chooses nothing either (its relation is the market's), and learns the mids its window takes
    cold = client.post("/api/statarb/traders/start", data=form({"kind": "index", **QUICK["index"], "warm": False}))
    assert cold.json()["message"] == "Index arbitrage 2 is watching: it trades once it has seen 20."
    statarb.run_for(500)
    warm = client.post("/api/statarb/traders/start", data=form({"kind": "index", **QUICK["index"]}))
    assert warm.json()["message"] == "Index arbitrage 3 learned from the last 20 points and trades from now on."
    assert warm.json()["learned"] == 20
    young = StatArbSession()
    young.configure({**MARKET, "index": True})
    young.run_for(50)
    runner, learned = young.start_trader({**QUICK["index"], "window": 100}, "index")
    assert main_module.statarb_started(runner, learned) == (
        f"Index arbitrage 1 learned from {learned} points of the past; it trades once it has seen "
        f"{100 - learned} more.")
    assert 0 < learned < 100
    # the kinds that do choose still say so
    pairs, learned = young.start_trader({**QUICK["pairs"], "formation": 100, "warm": False}, "pairs")
    assert main_module.statarb_started(pairs, learned) == "Pairs 2 is watching: it chooses once it has seen 100."


def test_an_index_trader_trades_the_index_against_every_book_and_unhedged_the_index_alone(client, statarb):
    assert client.post("/api/statarb/config", data=form({**MARKET, "index": True})).status_code == 200
    reply = client.post("/api/statarb/traders/start", data=form({"kind": "index", **QUICK["index"], "entry": 1.0}))
    assert reply.status_code == 200
    hedged = statarb.runners[0]
    assert statarb.market.n == 13 and len(hedged.trader.position) == 13
    assert hedged.trader.choose.weights.tolist() == [1.0] * 12  # the sum of the books: a lot of each for the index
    for _ in range(40):
        client.post("/api/statarb/control", data={"action": "step", "units": 25})
        if hedged.trader.lots:
            break
    t = hedged.trader
    assert t.lots > 0 and t.warming is None and t.weights.tolist() == [[-1.0] * 12 + [1.0]]
    row = statarb.board()[0]
    assert row["total"] == 1 and [x["text"] for x in row["relations"]] == ["IDX less each of the 12 books"]
    assert row["relations"][0]["state"] in (-1, 0, 1) and row["relations"][0]["signal"] == pytest.approx(float(t.z[0]))
    assert row["warming"] is None and row["holding"] == (1 if t.states[0] else 0)
    statarb._sample()  # one taken now, of this state
    assert statarb.samples[-1]["sig"][hedged.id] == [round(float(t.z[0]), 3)]  # one line on the signal pane: its z
    entry = statarb.chart_traders()[0]
    assert (entry["entry"], entry["exit"], entry["kind"]) == (1.0, 0.0, "index")
    html_board = render_statarb_panels(statarb.context())["statarb_board"]
    assert "IDX less each of the 12 books" in html_board and "holding" in html_board
    # unhedged: the same spread and the same z-score in a market of its own, and the index alone in the lots it holds
    other = StatArbSession()
    other.configure({**MARKET, "index": True})
    runner, _ = other.start_trader({**QUICK["index"], "entry": 1.0, "hedged": False}, "index")
    held = []
    for _ in range(40):
        other.run_for(25)
        held.append(runner.trader.position.copy())
        if runner.trader.lots and len(held) > 8:
            break
    positions = np.array(held)
    assert runner.trader.lots > 0 and not positions[:, :-1].any() and positions[:, -1].any()  # no book, ever
    assert runner.trader.weights.tolist() == [[-1.0] * 12 + [1.0]]  # reads the same relation
    assert runner.trader.entries > 0 and "None of the books" in other.board()[0]["describe"]
    book = other.board()[0]
    assert book["gross"] == abs(int(runner.trader.position[-1])) and (book["exposure"] > 0) == (book["gross"] > 0)


def test_a_deal_trader_has_a_row_a_deal_on_that_comes_and_goes_with_the_gap_in_ticks(client, statarb):
    assert client.post("/api/statarb/config", data=form({**DEALS, "seed": 2})).status_code == 200
    assert start(client, "deals").status_code == 200
    runner = statarb.runners[0]
    t = runner.trader
    seen, counts, texts, had_none = set(), [], [], False
    for _ in range(400):
        statarb.run_for(10)
        row = statarb.board()[0]
        rows = row["relations"]
        counts.append(len(rows))
        assert row["total"] == len(rows) == len(t.weights) == len(t.states) == len(t.signal())  # one in step
        assert row["holding"] == sum(x["state"] for x in rows) and all(x["state"] in (0, 1) for x in rows)
        for x in rows:
            assert re.fullmatch(r"[A-C][1-4] bought by [A-C][1-4]", x["text"]) and x["signal"] is not None
            seen.add(x["text"])
        texts.append([x["text"] for x in rows])
        statarb._sample()  # one taken now, of this state
        lines = statarb.samples[-1]["sig"][runner.id]  # each deal's gap in ticks, in a slot of its own
        assert sorted(g for g in lines if g is not None) == sorted(round(x["signal"], 3) for x in rows)
        had_none = had_none or (not rows and t.outcomes)
        if t.outcomes and had_none and len(set(counts)) > 1 and len(seen) >= 3:
            break
    assert t.outcomes and had_none and len(seen) >= 3  # deals announced, entered and ended, and the board followed
    steps = list(zip(counts, counts[1:], strict=False))
    assert 0 in counts and max(counts) >= 1 and any(a == 0 and b > 0 for a, b in steps)  # they come
    assert any(a > 0 and b == 0 for a, b in steps)  # and they go
    assert all(r.outcome in ("completed", "broken") and r.flat >= r.ended for r in t.outcomes)
    names = set(statarb.names)
    fair = statarb.market.fair
    announced = {f"{statarb.names[d.target]} bought by {statarb.names[d.acquirer]}" for d in [*fair.ended, *fair.deals]}
    for text in seen:  # the target first, and the acquirer after it, as the market's deals have them
        target, acquirer = text.split(" bought by ")
        assert target in names and acquirer in names and target != acquirer and text in announced, text
    # the page says it with the same words, and no deal on is said as that
    panels = render_statarb_panels(statarb.context())["statarb_board"]
    if statarb.board()[0]["relations"]:
        assert "data-statarb-deal" in panels and " bought by " in panels and re.search(r"in \d of \d deals? on", panels)
    for _ in range(400):
        if not statarb.board()[0]["relations"]:
            break
        statarb.run_for(5)
    empty = render_statarb_panels(statarb.context())["statarb_board"]
    assert "no deal on" in empty and "data-statarb-deal " not in empty and "data-statarb-relations" not in empty
    assert statarb.chart_traders()[0]["entry"] == 0.0 and statarb.chart_traders()[0]["kind"] == "deals"


def test_a_deal_traders_rows_are_the_deals_it_saw_at_its_last_decision_and_the_tooltip_says_so(client, statarb):
    """Not the deals on now: one announced or ended since its last decision (an interval ago, here 50 units) is not
    counted yet, while the status line is current. The tooltip of "in n of m deals on" says that."""
    assert client.post("/api/statarb/config", data=form({**DEALS, "seed": 2})).status_code == 200
    assert start(client, "deals", interval=50.0).status_code == 200
    apart = []
    for _ in range(800):
        statarb.run_for(1)
        rows, on = statarb.board()[0]["relations"], statarb.deals_line()["on"]
        if len(rows) != len(on):
            apart.append(statarb.market.time)
            board = render_statarb_panels(statarb.context())["statarb_board"]
            title = re.search(r'<span[^>]*data-statarb-deals-held[^>]*title="([^"]*)"', board).group(1)
            assert "last decision" in title and "not counted yet" in title and "status has the ones on now" in title
            break
    assert apart  # the rows lag the market's deals by up to an interval: the premise of the wording
    # and the status line, which reads the market's own deals, is the one that is current
    assert len(statarb.deals_line()["on"]) == len(statarb.market.fair.deals)


def test_a_deal_has_one_line_on_the_signal_pane_while_it_is_on_and_a_line_stops_where_a_deal_ends(client, statarb):
    assert client.post("/api/statarb/config", data=form({**DEALS, "seed": 2})).status_code == 200
    assert start(client, "deals").status_code == 200
    runner = statarb.runners[0]
    was, used, seen_slots = {}, [], set()
    for _ in range(2_000):  # a sample at a time
        statarb.run_for(1)
        now = dict(statarb._slots.get(runner.id, {}))
        lines = statarb.samples[-1]["sig"][runner.id]
        assert len(lines) == (max(now.values()) + 1 if now else 0) and len(set(now.values())) == len(now)  # one each
        assert all((lines[slot] is not None) for slot in now.values()) and sum(g is not None for g in lines) == len(now)
        for key, slot in now.items():
            assert was.get(key, slot) == slot  # a deal keeps its slot while it is on
            seen_slots.add(slot)
        for slot in set(was.values()) - set(now.values()):  # a slot just left empty is not taken in the same sample
            assert slot not in now.values()
        for key in set(was) - set(now):  # a deal that ended is not back
            assert key not in now
        used.append({slot: key for key, slot in now.items()})
        was = now
    assert len(seen_slots) >= 2 and max(seen_slots) <= MAX_SIGNALS - 1
    # Between a deal that ends and the next in its slot there is a sample with none there: the chart's line stops.
    gaps_in_a_line = 0
    for slot in seen_slots:
        deals_in_it = [i for i, u in enumerate(used) if slot in u]
        for before, after_ in zip(deals_in_it, deals_in_it[1:], strict=False):
            if used[before][slot] != used[after_][slot]:  # two deals in a slot, one after the other
                assert after_ - before >= 2
                gaps_in_a_line += 1
    assert gaps_in_a_line >= 1


def test_a_deal_trader_enters_a_deal_at_the_gap_it_is_given_and_holds_at_most_its_limit(client, statarb):
    long_deals = {**DEALS, "deal_length": 300.0, "seed": 3}
    assert client.post("/api/statarb/config", data=form(long_deals)).status_code == 200
    assert start(client, "deals", limit=1, entry=15.0).status_code == 200
    t = statarb.runners[0].trader
    most_on = most_held = 0
    for _ in range(120):
        statarb.run_for(10)
        row = statarb.board()[0]
        most_on, most_held = max(most_on, row["total"]), max(most_held, row["holding"])
        assert row["holding"] <= 1
        if not row["holding"]:  # with room, none whose gap was 15 ticks at its last decision is left out
            assert all(x["signal"] < 15.0 for x in row["relations"])
    assert most_held == 1 and most_on >= 2  # more on than held: the limit held it to one
    assert "holding at most 1 deal at once" in statarb.board()[0]["describe"]
    assert (statarb.chart_traders()[0]["entry"], t.entry, t.limit) == (15.0, 15.0, 1)
    assert statarb.board()[0]["describe"].count("once the offer is worth 15 ticks more") == 1


def test_the_page_offers_the_two_new_kinds_after_the_eight_with_their_settings_and_the_hedge_switch_on(client):
    page = client.get("/").text
    form_ = page[page.index('<form id="statarb-trader-form"'):]
    radios = re.findall(r'<input type="radio" name="kind" value="(\w+)"([^>]*)>', form_[:form_.index("</form>")])
    assert [k for k, _ in radios] == list(TRADER_KINDS) and [k for k, rest in radios if "checked" in rest] == ["pairs"]
    assert [k for k, _ in radios][-2:] == ["index", "deals"]
    for kind in ("index", "deals"):
        spec = TRADER_KINDS[kind]
        assert f">{spec.label}</span>" in page
        start_ = page.index(f'data-statarb-kind="{kind}"')
        fieldset = page[start_:page.index("</fieldset>", start_)]
        assert spec.summary in html.unescape(fieldset)
        assert f'<fieldset data-statarb-kind="{kind}" hidden>' in page  # shown when it is chosen
        for q in spec.numbers:
            assert f'name="{q.name}" value="{q.default}"' in fieldset
        assert ('name="warm"' in fieldset) == (kind == "index") and ('name="hedged"' in fieldset) == (kind == "index")
    fieldset = page[page.index('data-statarb-kind="index"'):]
    fieldset = fieldset[:fieldset.index("</fieldset>")]
    assert re.search(r'<input type="checkbox" name="hedged" value="on" checked', fieldset)  # on, as it is by default
    assert re.search(r'<input type="checkbox" name="warm" value="on" checked', fieldset)
    hint = re.search(r'<label[^>]*title="([^"]*)"[^>]*>\s*<input type="checkbox" name="hedged"', fieldset).group(1)
    assert "the index alone" in html.unescape(hint) and "pays one book's spread, not all" in html.unescape(hint)
    deals = page[page.index('data-statarb-kind="deals"'):]
    deals = html.unescape(deals[:deals.index("</fieldset>")])
    assert "Enter at a gap of (ticks)" in deals and "Deals at once (0: any)" in deals and "Lots of the target" in deals
    assert "the widest gap first. 0: as many as there are" in deals and "A deal is entered when the offer" in deals
    assert 'name="warm"' not in deals and 'type="checkbox"' not in deals  # no switch among the deal trader's settings


def test_a_form_without_the_switches_of_the_two_kinds_starts_them_unhedged_and_cold_as_every_switch_is(client, statarb):
    client.post("/api/statarb/config", data=form({**MARKET, "index": True}))
    reply = client.post("/api/statarb/traders/start", data={"kind": "index", "window": 20})  # no switch: both off
    assert reply.status_code == 200
    t = statarb.runners[0].trader
    assert t.hedged is False and t.warming == (0, 20) and statarb.runners[0].values["warm"] is False
    assert statarb.runners[0].values["hedged"] is False


# ---- What a sweep of changes to the view's new lines found that no test held ----
# The audit of the view's part of stage 4 changed each operator, comparison, boolean and number of the lines of
# app/statarb.py and app/main.py that differ from 42ae383, one at a time. These hold what those changes found missing.
# The figures are VIEW.md's, typed here and not read from the module they hold (a page held to the settings it was made
# from follows them wherever they go).

SHEET_SETTINGS = {  # VIEW.md section B: default, low, high, step
    "index_volatility": (0.4, 0.0, 3.0, 0.1), "deal_rate": (4.0, 0.5, 10.0, 0.5),
    "deal_premium": (60.0, 5.0, 200.0, 5.0), "deal_length": (500.0, 50.0, 2000.0, 50.0),
    "deal_priced": (40.0, 0.0, 95.0, 5.0), "deal_breaks": (20.0, 0.0, 95.0, 5.0), "trend": (0.0, 0.0, 0.05, 0.005)}


def input_of(page_part, name):
    """The value, least, most and step an input of this name has in a part of the page, as numbers."""
    tag = re.search(rf'<input[^>]*name="{name}"[^>]*>', page_part).group(0)
    return tuple(float(re.search(rf'\b{attr}="([^"]*)"', tag).group(1)) for attr in ("value", "min", "max", "step"))


def test_the_new_settings_have_the_defaults_ranges_and_steps_the_sheet_says_in_the_module_and_on_the_page(client):
    page = client.get("/").text
    config = config_of(page)
    by_name = {p.name: p for p in MARKET_PARAMS}
    for name, sheet in SHEET_SETTINGS.items():
        p = by_name[name]
        assert (p.default, p.low, p.high, p.step) == sheet, name
        assert input_of(config, name) == sheet, name
    # the two kinds' z-scores and gaps step as the other kinds' do (a whole number is stepped by 1, whatever its step)
    for kind, steps in (("index", {"entry": 0.1, "exit": 0.1}), ("deals", {"entry": 1.0})):
        start_ = page.index(f'data-statarb-kind="{kind}"')
        fieldset = page[start_:page.index("</fieldset>", start_)]
        for name, step in steps.items():
            assert input_of(fieldset, name)[3] == step, (kind, name)
    assert [input_of(page[page.index('data-statarb-kind="index"'):], n)[3] for n in ("window", "size")] == [1.0, 1.0]


def test_a_trend_in_a_mixed_market_needs_two_factors_and_two_are_enough():
    mixed = {**MARKET, "shape": "mixed", "books": 6, "trend": 0.01}
    values, errors = parse_market(form({**mixed, "factors": 1}))
    assert errors == {"trend": "A trend is on every factor but the first, so it needs two factors or more."}
    for factors in (2, 3, 4):
        assert parse_market(form({**mixed, "factors": factors}))[1] == {}, factors
    assert parse_market(form({**mixed, "factors": 1, "trend": 0.0}))[1] == {}  # no trend asked, no factor to put it on
    assert parse_market(form({**MARKET, "factors": 1, "trend": 0.01}))[1] == {}  # sectors have their own factors


def test_an_index_trader_chooses_its_one_relation_at_once_and_trades_as_the_library_does_in_a_market_made_by_hand():
    """No thread chooses for it (``StatArbRunner.fresh``): the same decisions as an ``IndexTrader`` in a ``BookMarket``
    of the same settings and seed, number for number, where a relation that came a decision late would part from it."""
    session = StatArbSession()
    session.configure({**MARKET, "index": True, "seed": 3})
    runner, _ = session.start_trader({**DEFAULTS["index"], "window": 20, "warm": False}, "index")
    assert runner.trader.defer is None
    for other in ("pairs", "baskets", "copula"):  # the others that choose still do, off the event loop
        assert runner_of(other).trader.defer is statarb_module._choose_elsewhere
    by_hand = IndexTrader(legs=0, hedged=True, window=20, entry=2.0, exit=0.0, size=1, rate=1.0, band=0.0, interval=5.0)
    hand = BookMarket(FactorValues(sector_loadings(3, 4), level=1000.0, volatility=1.0, own_volatility=0.8,
                                   half_life=50.0, impact=0.05, loading_sd=0.0, loading_half_life=2000.0, index=True,
                                   index_volatility=0.4), seed=3, traders=[by_hand])
    session.run_for(1_200)
    while hand.time < session.market.time:
        hand.step()
    t = runner.trader
    assert t.lots > 0 and t.entries > 0  # it traded, so that the comparison says something
    assert hand.time == session.market.time and (t.lots, t.entries) == (by_hand.lots, by_hand.entries)
    assert t.position.tolist() == by_hand.position.tolist() and t.spread_paid == by_hand.spread_paid


def runner_of(kind):
    """A runner of a kind at its quick settings, as the board has it (it makes its trader, and needs no market)."""
    return statarb_module.StatArbRunner(1, kind, dict(QUICK[kind]), kind)


def test_the_status_counts_the_deals_that_had_ended_before_it_was_first_read():
    """The tally begins at 0 whenever the first look comes (the market ran on before one: a tab opened late, or a reset
    that nobody was watching), and the first deal to have ended is one of those it counts."""
    session = StatArbSession()
    session.configure(DEALS)
    session.run_for(2_000)  # not read: nothing has called ``context`` since the market was made
    ended = session.market.fair.ended
    assert len(ended) >= 5
    line = session.context()["deals"]
    outcomes = [d.outcome for d in ended]
    assert (line["completed"], line["broken"]) == (outcomes.count("completed"), outcomes.count("broken"))
    assert line["completed"] + line["broken"] == len(ended)


def test_new_settings_of_twelve_books_are_applied_with_a_johansen_trader_of_all_the_books_on_the_board_and_thirteen_not(
        client, statarb):
    reply = client.post("/api/statarb/traders/start", data=form({"kind": "baskets", **QUICK["baskets"], "legs": 0}))
    assert reply.status_code == 200 and statarb.market.n == 12
    twelve = {**MARKET, "shape": "mixed", "books": 12, "factors": 3}
    reply = client.post("/api/statarb/config", data=form(twelve))  # twelve is what Johansen's test takes
    assert reply.status_code == 200 and statarb.market.n == 12 and len(statarb.runners) == 1
    version = statarb.version
    reply = client.post("/api/statarb/config", data=form({**twelve, "index": True}))  # and the index makes thirteen
    assert reply.status_code == 422 and "keep 12 books or fewer" in reply.json()["error"]
    assert statarb.market.n == 12 and statarb.version == version and statarb.settings["shape"] == "mixed"


def test_the_status_gives_a_deals_gap_in_tenths_of_a_tick_and_a_time_left_of_0_in_the_last_half_unit_of_the_deal():
    session = StatArbSession()
    session.configure(DEALS)
    m, fair = session.market, session.market.fair
    run_until_a_deal_is_on(session)
    compared = 0
    while compared < 40:  # a tenth of a tick, to the figure: a gap to the unit, or in hundredths, is not what is said
        session.run_for(4)
        for shown, deal in zip(session.deals_line()["on"], fair.deals, strict=True):
            gap = deal.ratio * m.mark(deal.acquirer) - m.mark(deal.target)
            assert shown["gap"] == round(gap, 1)
            compared += 1
        assert m.time < 6_000
    while True:  # a deal in the last half unit of its time has 0 left, and not 1
        run_until_a_deal_is_on(session)
        deal = min(fair.deals, key=lambda d: d.ends)
        if deal.ends - m.time > 2:
            session.run_for(deal.ends - m.time - 1.0)
        while deal.outcome is None and deal.ends - m.time > 0.4:
            session._event()
        if deal.outcome is None:  # it did not break on the way
            break
        assert m.time < 6_000
    shown = next(d for d in session.deals_line()["on"]
                 if (d["target"], d["acquirer"]) == (session.names[deal.target], session.names[deal.acquirer]))
    assert 0 < deal.ends - m.time <= 0.4 and shown["left"] == 0


def test_a_start_says_one_point_of_the_past_and_not_one_points_and_where_it_trades_for_the_index_kind(client, statarb):
    client.post("/api/statarb/config", data=form({**MARKET, "index": True}))
    reply = client.post("/api/statarb/traders/start", data=form({"kind": "index", **QUICK["index"]}))  # one mid kept
    assert reply.json()["message"] == ("Index arbitrage 1 learned from 1 point of the past; it trades once it has seen "
                                       "19 more.")
    client.post("/api/statarb/control", data={"action": "step", "units": 60})  # a point each interval of 5, 13 by now
    reply = client.post("/api/statarb/traders/start", data=form({"kind": "index", **QUICK["index"]}))
    assert reply.json()["message"] == ("Index arbitrage 2 learned from 13 points of the past; it trades once it has "
                                       "seen 7 more.")


# ---- What the summaries and hints say of notebook 15 ----
# Notebook 15 was run on seeds 101 and up (every setting chosen beforehand on seeds 1 to 8). Four kinds quote it (index
# arbitrage, deals, momentum for its trend, pairs for its deals) and the three dials of the market's extra settings
# quote one break-even each. Each figure is pinned word for word, with the unit and the market it belongs to, so that a
# figure changed on its own, or one that loses its market, fails here. The notebook's tables are not in the repository;
# these are copied from what it printed (its section 1, 2 and 3 outputs and its conclusions).

NOTEBOOK_15_SAYS = {
    "index": (
        "In notebook 15's sectors market (the 12 books and the index, at level 3000 where this view's is 1000), at "
        "these defaults, it made +7 (se 24, 5 of 8 seeds positive) with the premium's volatility at 0, the lowest it "
        "goes, where the order flow alone makes a premium of 14.7 ticks of a basket (a standard deviation), so it "
        "broke even about there; it was positive on every seed from 0.2 (+467, se 42), and made +1,214 (se 48, 8 of 8 "
        "seeds positive) at this view's 0.4.",
        "Without the books' legs (the switch Hedge with the books off) it made +1,109 (se 196, 8 of 8 seeds "
        "positive) more at 0.4: the legs cost about 930 (se 107) and cut the risk over 100 units from 6.91 to 1.92.",
        "An index quoted like one more book needed a premium about twice as large to break even.",
        "These are means of 8 seeds of 20,000 units, in ticks times lots after costs."),
    "deals": (
        "In notebook 15's sectors market (level 3000, where this view's is 1000), with this view's defaults for "
        "deals (4 announced in 1,000 units, a premium of 60, 500 units long, a break priced at 0.40) and 5 lots of "
        "the target, it made -2,379 (se 348, 0 of 12 seeds positive) when deals truly broke at the price, 0.40, and "
        "+2,747 (se 346, 12 of 12 seeds positive) at this view's default of 0.20, and it broke even at a true chance "
        "of 0.310 (95% 0.290 to 0.331 by a bootstrap over the seeds, though the dial's own arithmetic gives an sd of "
        "0.014 over 12 fresh seeds, the spread to read it with).",
        "At 0.20 a completed deal made +18.57 (sd 3.22) a lot pair (a lot of the target and its acquirer's lots) and "
        "a broken one, 19.6% of 966 deals, lost 41.72 (sd 8.52), which is 2.2 completed ones; it won 80.4% of its "
        "deals.",
        "These are means of 12 seeds of 20,000 units, in ticks times lots after costs."),
    "momentum": (
        "In notebook 15's sectors market (level 3000, where this view's is 1000), at these defaults but with the "
        "dollar alone taken out (0 factors), it broke even at a factor trend of 0.0269 ticks a unit of time (0.0251 "
        "to 0.0289, 5% to 95%), and in the mixed market at 0.0364 (0.0352 to 0.0374), though this view warns that "
        "above 0.02 a long run in its mixed market can reach the floor (see the trend setting).",
        "With 3 factors taken out, as here, a trend of 0.05 took it only from -5,061 (se 148) to -4,574 (se 230) in "
        "the sectors market, and it lost on every seed there and in the mixed market (-4,113, se 183), keeping 1.8% "
        "and 3.7% of the dollar-neutral trader's factor part.",
        "The P&Ls are means of 8 seeds of 20,000 units, in ticks times lots after costs; a break-even is a long-run "
        "figure, from the market's 8 windows and the same trader's factor part in fair values on more seeds (36 at the "
        "two points that bracket the sectors market's, 8 in the mixed market)."),
    "pairs": (
        "Deals in the market cost it: in notebook 15's sectors market (level 3000, where this view's is 1000), with "
        "deals announced at 0.002 a unit (half this view's default), it made +2,044 (se 272, 8 of 8 seeds positive) "
        "at a fair price (a chance of 0.40 of a break, in truth and priced) against +5,703 (se 118, these 8 seeds) "
        "with none, 3,659 less (se 344, on all 8 seeds), and +837 (se 268, 6 of 8 seeds positive) at a true chance "
        "of 0.20, this view's default.",),
}


@pytest.mark.parametrize("kind", NOTEBOOK_15_KINDS)
def test_the_summaries_say_what_notebook_15_printed_word_for_word(kind):
    summary = TRADER_KINDS[kind].summary
    for sentence in NOTEBOOK_15_SAYS[kind]:
        assert sentence in summary, sentence
    assert "notebook 15 has" not in summary.lower() and "for notebook 15 to say" not in summary  # the old pointers
    assert "seeds 1 to 8" not in summary  # the seeds the settings were chosen on are not reported


def test_what_the_summaries_call_this_views_defaults_are_its_defaults():
    """A sentence that says 'at these defaults' or 'this view's default' is true of the settings it names, and the
    level it says the notebook's market had is the library's."""
    import inspect
    assert inspect.signature(FactorValues).parameters["level"].default == 3000.0 and LEVEL == 1000.0
    # the index kind, as notebook 15's section 1 ran it: the sum index, a window of 400, an entry of 2, one basket
    index = {k: DEFAULTS["index"][k] for k in ("window", "entry", "exit", "hedged", "size", "rate", "band", "interval")}
    assert index == {"window": 400, "entry": 2.0, "exit": 0.0, "hedged": True, "size": 1, "rate": 1.0, "band": 0.0,
                     "interval": 5.0}
    assert MARKET["index_volatility"] == 0.4  # "at this view's 0.4"
    # the deals kind, as section 2 ran it: 5 lots, at the first decision, any number of deals at once, a deal rate
    # of 0.004 a unit (4 in 1,000), a premium of 60, 500 units, priced at 0.40 and a true chance of 0.20
    assert (DEFAULTS["deals"]["size"], DEFAULTS["deals"]["entry"], DEFAULTS["deals"]["limit"]) == (5.0, 0.0, 0)
    assert (DEFAULTS["deals"]["rate"], DEFAULTS["deals"]["band"], DEFAULTS["deals"]["interval"]) == (1.0, 0.0, 5.0)
    assert (MARKET["deal_rate"], MARKET["deal_premium"], MARKET["deal_length"]) == (4.0, 60.0, 500.0)
    assert (MARKET["deal_priced"], MARKET["deal_breaks"]) == (40.0, 20.0)
    assert MARKET["deal_rate"] / 1000 / 2 == 0.002  # "announced at 0.002 a unit (half this view's default)"
    # momentum, as section 3 ran it: a look back and a hold of 100 decisions of 5 units, 5 lots, all the way, with 0
    # factors (the dollar alone) or 3 (this view's default); the drift's half-life is the view's
    momentum = {k: DEFAULTS["momentum"][k] for k in ("lookback", "skip", "hold", "factors", "size", "rate", "interval")}
    assert momentum == {"lookback": 100, "skip": 0, "hold": 100, "factors": 3, "size": 5.0, "rate": 1.0,
                        "interval": 5.0}
    assert statarb_module.TREND_HALF_LIFE == 500.0
    # pairs, as notebook 13 and section 2 ran them: eight by distance, 2.5 lots, 2,000 to choose from, 400, entry 2
    assert {k: DEFAULTS["pairs"][k] for k in ("count", "by_pvalue", "formation", "window", "entry", "exit", "size",
                                              "rate", "band", "interval")} == {
        "count": 8, "by_pvalue": False, "formation": 2000, "window": 400, "entry": 2.0, "exit": 0.0, "size": 2.5,
        "rate": 1.0, "band": 0.0, "interval": 5.0}
    # the sectors market of the notebook: three sectors of four, which is the view's
    assert (MARKET["sectors"], MARKET["books_each"], MARKET["shape"]) == (3, 4, "sectors")


def title_of(config, name):
    """The hover text (title) of the label of the setting ``name``, unescaped."""
    label = re.search(rf'<label[^>]*title="([^"]*)"[^>]*>(?:(?!</label>).)*?name="{name}"', config, flags=re.S)
    assert label, name
    return html.unescape(label.group(1))


def test_the_hints_of_the_three_dials_have_one_break_even_each_and_the_others_have_none(client, statarb):
    config = config_of(client.get("/").text)
    said = {
        "index_volatility": (
            "With 0 the order flow's own impact still makes a premium. In notebook 15's sectors market (level 3000, "
            "where this view's is 1000), index arbitrage at its defaults made +7 (se 24, 5 of 8 seeds positive) at 0, "
            "the lowest it goes, so it breaks even about there, and was positive on every seed from 0.2 (+467, se "
            "42)"),
        "deal_breaks": (
            "The share of deals that break before they complete, in percent. In notebook 15's sectors market (level "
            "3000, where this view's is 1000), with the other deal settings at their defaults, the deals kind broke "
            "even at a true chance of 31% (0.310; 95% 0.290 to 0.331 by a bootstrap over 12 seeds, though the dial's "
            "own arithmetic gives an sd of 0.014, the spread to read it with) and "
            "was positive on every seed from 25% (+1,652, se 329)"),
        "trend": (
            "can reach the floor, which then pushes the first factor. In notebook 15's sectors market (level 3000), "
            "momentum at its defaults with the dollar alone taken out broke even at 0.0269 (0.0251 to 0.0289, 5% to "
            "95%), and in its mixed market at 0.0364 (0.0352 to 0.0374), where the floor note above gives 0.02")}
    for name, sentence in said.items():
        assert sentence in title_of(config, name), name
    # the floor note stays, whole, before the figure, and the mixed market's 0.02 is the summary's
    assert ("a long run at a trend above about 0.04 (0.02 in a mixed market) can reach the floor, which then pushes "
            "the first factor. In notebook 15's") in title_of(config, "trend")
    names = re.findall(r'<label[^>]*title="[^"]*"[^>]*>(?:(?!</label>).)*?name="(\w+)"', config, flags=re.S)
    quoting = {n for n in names if "notebook 15" in title_of(config, n).lower()}
    assert quoting == set(said)  # and the premium, length, rate and price of a deal and the two switches quote none
    for name in set(NEW) - set(said):
        assert not re.search(r"\d+(\.\d+)? \(se", title_of(config, name)), name
    # the two switches' hints (the index book and deals) say what they are, and quote none
    for name in ("index", "deals"):
        label = re.search(rf'<label[^>]*title="([^"]*)"[^>]*>\s*<input type="checkbox" name="{name}"', config)
        assert label and "notebook 15" not in label.group(1).lower()


def test_the_guide_says_where_the_three_dials_break_even_in_one_passage(client):
    page = html.unescape(client.get("/").text)
    guide = page[page.index('<span class="text-slate-200">Stat-arb</span>'):]
    guide = guide[:guide.index("</li>")]
    assert ("Notebook 15 turns each up from nothing, in its sectors market at level 3000 (this view's is 1000), and "
            "says where a trader breaks even: index arbitrage at 0, the lowest the premium's volatility goes, where "
            "the order flow alone makes a premium (+7, se 24, 5 of 8 seeds positive; every seed is positive from "
            "0.2); deals at a true chance of a break of 0.310 where the market prices 0.40; momentum with the dollar "
            "alone taken out at a factor trend of 0.0269.") in guide
    assert "Notebook 15 says what they do" not in guide
    assert guide.lower().count("notebook 15") == 1 and "Notebook 12 says what to expect" in guide  # and 12's stays


def test_every_kinds_summary_is_on_the_page_whole_in_its_own_fieldset_and_the_form_still_names_every_kind(client):
    page = html.unescape(client.get("/").text)
    form_ = page[page.index('<form id="statarb-trader-form"'):]
    form_ = form_[:form_.index("</form>")]
    kinds = re.findall(r'<fieldset data-statarb-kind="(\w+)"', form_)
    assert kinds == list(TRADER_KINDS) and len(kinds) == 10
    for kind, spec in TRADER_KINDS.items():
        start = form_.index(f'<fieldset data-statarb-kind="{kind}"')
        fieldset = form_[start:form_.index("</fieldset>", start)]
        assert spec.summary in fieldset, kind
        assert fieldset.count(spec.summary) == 1, kind  # once, and not another kind's as well
        assert len(spec.summary) < 1500, kind  # a paragraph in the form's small text, not an essay
