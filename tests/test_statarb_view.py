"""The dashboard's Stat-arb view: many books and traders of many relations in them (app/statarb.py)."""

import asyncio
import html
import json
import math
import re
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from starlette.testclient import TestClient

from app import main as main_module
from app import statarb as statarb_module
from app.main import app, registry, render_statarb_panels
from app.pairs import MAX_TRADERS, PairsSession
from app.sessions import COOKIE_NAME
from app.statarb import (COLORS, CROSS_SECTIONAL, LEVEL, MARKET_PARAMS, MAX_SIGNALS, MAX_STEP, MIDS_KEPT, SAMPLES,
                         TRADER_KINDS, StatArbSession, book_names, loadings_of, parse_market, relation_text,
                         too_few_books, what_the_tests_see)
from lob.crosssection import FlowTrader, LeadLagTrader, MomentumTrader, ReversalTrader
from lob.pairs import FLOOR
from lob.sparse_baskets import sparse_baskets
from lob.statarb import (CopulaPortfolio, RelationTrader, best_baskets, johansen_relations, pairs_by_distance,
                         pairs_by_pvalue)
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
}


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
    reply = client.post("/api/statarb/traders/start", data=form({"kind": kind, **QUICK[kind]}))
    assert reply.status_code == 200 and reply.json()["name"] == f"{TRADER_KINDS[kind].label} 1"
    message = reply.json()["message"]
    if kind == "flow":  # it takes in the mids for its factors, and has no flow of the past to learn from
        assert "order flow only from now on" in message and "learned" not in message
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
    assert row["pnl"] == pytest.approx(row["before"] - row["cost"]) and row["lots"] == t.lots and row["relations"]
    assert row["describe"] and row["gross"] == np.abs(t.position).sum()
    sample = statarb.samples[-1]
    assert sample["pnl"][runner.id] == pytest.approx(t.value(statarb.market.marks()), abs=50)
    assert runner.id in sample["sig"] and sample["gross"][runner.id] >= 0
    panels = render_statarb_panels(statarb.context())
    assert f'data-statarb-id="{runner.id}"' in panels["statarb_board"]
    assert "data-statarb-relations" in panels["statarb_board"]
    assert panels["statarb_books"].count("data-statarb-book=") == 12


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
    assert t.warming is None and len(statarb.names) == MAX_SIGNALS
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


def test_the_board_counts_what_it_holds_against_all_the_books_though_it_lists_only_twelve(client, statarb):
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
    assert f"holding {row['holding']} of 16" in board and board.count('<span class="px-1 rounded border') == 12


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
    reply = client.post("/api/statarb/traders/start", data=form({"kind": kind, **QUICK[kind], "rate": 0.5,
                                                                 "band": 1.0}))
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
