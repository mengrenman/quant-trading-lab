"""The dashboard's Stat-arb view: many books and traders of many relations in them (app/statarb.py)."""

import asyncio
import json
import math
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
from app.statarb import (COLORS, LEVEL, MARKET_PARAMS, MAX_STEP, MIDS_KEPT, SAMPLES, TRADER_KINDS, StatArbSession,
                         book_names, loadings_of, parse_market, relation_text, what_the_tests_see)
from lob.pairs import FLOOR
from lob.statarb import CopulaPortfolio, best_baskets, johansen_relations, pairs_by_distance, pairs_by_pvalue
from tests.test_pairs_view import read_stream as read_pairs_stream

DEFAULTS = {kind: {p.name: p.default for p in spec.params} for kind, spec in TRADER_KINDS.items()}
MARKET = {p.name: p.default for p in MARKET_PARAMS} | {"shape": "sectors"}
QUICK = {
    "pairs": {**DEFAULTS["pairs"], "count": 2, "formation": 60, "window": 20},
    "baskets": {**DEFAULTS["baskets"], "legs": 3, "count": 2, "formation": 200, "window": 20},
    "residuals": {**DEFAULTS["residuals"], "window": 30, "pca_window": 60},
    "copula": {**DEFAULTS["copula"], "pairs": 2, "window": 60, "trading": 20},
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
    assert "watching" in reply.json()["message"] or "learned" in reply.json()["message"]
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


@pytest.mark.parametrize("change, name", [({"window": 1}, "window"), ({"size": 0}, "size"), ({"count": 99}, "count")])
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
