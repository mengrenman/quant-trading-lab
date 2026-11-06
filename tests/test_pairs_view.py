"""The dashboard's Pairs view: two books whose gap mean-reverts, and pairs traders in them (app/pairs.py)."""

import asyncio
import json
import math

import pytest
from starlette.testclient import TestClient

from app import pairs as pairs_module
from app.main import app, registry, render_pairs_panels
from app.pairs import LEVEL, MARKET_PARAMS, MAX_STEP, MAX_TRADERS, TRADER_KINDS, PairsSession, parse
from app.sessions import COOKIE_NAME
from lob.pairs import A, B, FLOOR

TRADER = {p.name: p.default for p in TRADER_KINDS["zscore"].params}
KALMAN = {p.name: p.default for p in TRADER_KINDS["kalman"].params}
COPULA = {p.name: p.default for p in TRADER_KINDS["copula"].params}


@pytest.fixture
def client():
    client = TestClient(app)
    client.get("/")
    return client


@pytest.fixture
def pairs(client):
    return registry.get(client.cookies[COOKIE_NAME]).pairs


def trader(**settings):
    return {**TRADER, **settings}


def read_stream(client, wanted):
    """First payload of each event type on /pairs/events, until every wanted type has arrived."""
    seen = {}
    with client.stream("GET", "/pairs/events") as resp:
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


def test_the_pairs_view_runs_a_market_of_its_own(client, pairs):
    m = registry.get(client.cookies[COOKIE_NAME])
    page = client.get("/").text
    assert 'id="view-pairs"' in page and 'id="tab-pairs"' in page and "js/lob-pairs.js" in page
    assert len(pairs.samples) == 1 and pairs.samples[0]["t"] == 0
    before = m.sim_time
    reply = client.post("/api/pairs/control", data={"action": "step", "units": 250}).json()
    assert reply == {"running": False, "time": pytest.approx(pairs.market.time)} and 250 <= pairs.market.time < 251
    assert m.sim_time == before  # the Trade view's market did not move
    assert [s["t"] for s in pairs.samples] == list(range(251))  # one sample a unit of time


def test_a_reset_starts_the_books_and_every_trader_over_and_a_stopped_one_stays_stopped():
    s = PairsSession()
    s.run_for(300)
    one, _ = s.start_trader(trader(window=50))
    two, _ = s.start_trader(trader(window=50, entry=1.0))
    s.run_for(400)
    s.set_active(two, False)
    held = list(two.trader.position)
    s.run_for(200)
    assert two.trader.position == held  # stopped: it holds what it had and trades no more
    assert one.trader.lots > 0
    s.reset()
    assert s.market.time == 0 and [r.id for r in s.runners] == [1, 2]
    assert s.market.traders == [one.trader, two.trader]  # fresh traders, in the market
    assert one.trader.lots == 0 and one.trader.position == [0, 0] and one.active and not two.active
    s.run_for(300)
    assert one.trader.lots > 0 and two.trader.lots == 0


def test_a_trader_can_learn_the_gap_from_the_chart_first():
    s = PairsSession()
    s.run_for(1000)
    warm, learned = s.start_trader(trader(window=400, interval=2.0))
    cold, none = s.start_trader(trader(window=400, warm=False))
    assert learned == 400 and none == 0 and len(cold.trader._gaps) == 0
    every_other = [p for p in list(s.samples)[::-1][::2] if p["a"] is not None and p["b"] is not None]
    expected = [p["b"] - 1.0 * p["a"] for p in every_other[:400]][::-1]  # one gap every 2 units, oldest first
    assert warm.trader.warming is None
    assert list(warm.trader._gaps) == pytest.approx(expected)
    s.run_for(5)
    assert warm.trader.z is not None and cold.trader.z is None  # one trades at once; the other is still watching
    _, n = s.start_trader(trader(window=5_000))
    assert n == len([p for p in s.samples if p["a"] is not None and p["b"] is not None]) < 5_000


def test_marks_fall_on_the_first_sample_after_each_change_of_target():
    s = PairsSession()
    r, _ = s.start_trader(trader(window=60, entry=1.0))
    r.trader.record = True
    s.run_for(3000)
    changes, target = [], 0
    for t, *_, want in r.trader.history:
        if want != target:
            changes.append((t, want))
            target = want
    marks = [(k, want) for _, i, k, want in s.marks if i == r.id]
    assert len(marks) == len(changes) > 10
    for (k, want), (t, wanted) in zip(marks, changes, strict=True):
        assert want == wanted and abs(k - t) < 1  # the sample taken next after the decision
    assert [k for k, _ in marks] == sorted(k for k, _ in marks)


def test_the_gap_crosses_a_traders_bands_where_its_z_score_crosses_its_entry_level():
    """Its bands are its own gap's (B - hedge A) moved into the chart's terms (B - level_B - beta (A - level_A))."""
    s = PairsSession()
    s.configure({**s.settings, "beta": 1.5})
    r, _ = s.start_trader(trader(window=100, hedge=1.2, entry=1.5))
    s.run_for(600)
    t = r.trader
    s._sample()  # with its mean and sd as last measured
    p = s.samples[-1]
    lo, mean, hi = p["bands"][r.id]
    assert hi - mean == pytest.approx(t.entry * t.sd, abs=2e-3) and mean - lo == pytest.approx(hi - mean, abs=2e-3)
    z_chart = t.entry * (p["gap"] - mean) / (hi - mean)
    z_own = (p["b"] - t.hedge * p["a"] - t.mean) / t.sd
    assert z_chart == pytest.approx(z_own, abs=2e-3)


def test_the_board_adds_up():
    s = PairsSession()
    s.configure({**s.settings, "arbitrageur": True, "reverting": False})
    r, _ = s.start_trader(trader(window=100))
    s.run_for(2000)
    rows, arb = s.board()
    row = rows[0]
    t, m = r.trader, s.market
    assert row["pnl"] == pytest.approx(t.cash + t.position[A] * m.mark(A) + t.position[B] * m.mark(B))
    assert row["before"] == pytest.approx(row["pnl"] + t.spread_paid) and row["cost_a_lot"] == t.spread_paid / t.lots
    assert "short 5 B, long 5 A" in row["describe"]
    hedged, _ = s.start_trader(trader(size=5, hedge=1.3))
    assert "short 5 B, long 6 A" in hedged.describe()  # the whole lots it trades
    assert arb["trades"] > 50 and arb["position"][A] == -arb["position"][B]
    assert s.context()["half_life"] is None and s.gap_sds()["theory"] is None


def test_market_settings_are_checked_and_applied_with_a_reset(client, pairs):
    client.post("/api/pairs/control", data={"action": "step", "units": 100})
    bad = client.post("/api/pairs/config", data={"half_life": 1, "beta": "x"})
    assert bad.status_code == 422 and set(bad.json()["errors"]) == {"half_life", "beta"}
    assert pairs.market.time >= 100  # nothing applied
    form = {p.name: p.default for p in MARKET_PARAMS if not isinstance(p.default, bool)}
    good = client.post("/api/pairs/config", data={**form, "beta": 2.0, "seed": 9, "arbitrageur": "on"})
    assert good.status_code == 200 and "from time 0" in good.json()["message"]
    assert pairs.market.time == 0 and pairs.market.fair.beta == 2.0 and pairs.market.arbitrageur is not None
    assert pairs.market.fair.half_life is None and not pairs.settings["reverting"]  # its box was not ticked
    values, errors = parse(MARKET_PARAMS, {"reverting": "on"})
    assert not errors and values["reverting"] and values["half_life"] == 50.0


def test_traders_start_stop_and_leave_through_the_endpoints(client, pairs):
    bad = client.post("/api/pairs/traders/start", data=trader(entry=1.0, exit=1.0))
    assert bad.status_code == 422 and "exit" in bad.json()["errors"]
    started = client.post("/api/pairs/traders/start", data=trader(window=200)).json()
    assert started["name"] == "Z-score 1" and started["learned"] == 1
    assert started["message"] == "Z-score 1 learned from the chart's 1 point; it trades once it has seen 199 more."
    assert client.post(f"/api/pairs/traders/{started['id']}/stop").json() == {"id": 1, "action": "stop"}
    assert not pairs.runner(1).active
    client.post(f"/api/pairs/traders/{started['id']}/resume")
    assert pairs.runner(1).active and pairs.runner(1).trader.next_time == pairs.market.time
    for _ in range(MAX_TRADERS - 1):
        client.post("/api/pairs/traders/start", data=trader())
    full = client.post("/api/pairs/traders/start", data=trader())
    assert full.status_code == 409 and "remove one first" in full.json()["error"]
    client.post("/api/pairs/traders/1/remove")
    assert pairs.runner(1) is None and len(pairs.market.traders) == MAX_TRADERS - 1
    assert client.post("/api/pairs/traders/1/stop").status_code == 404
    assert client.post("/api/pairs/traders/2/sell").status_code == 422


def test_the_controls_refuse_what_they_cannot_do(client):
    assert client.post("/api/pairs/control", data={"action": "step", "units": MAX_STEP + 1}).status_code == 422
    assert client.post("/api/pairs/control", data={"action": "step", "units": "nan"}).status_code == 422
    assert client.post("/api/pairs/speed", data={"speed": 0}).status_code == 422
    assert client.post("/api/pairs/speed", data={"speed": 50}).json() == {"speed": 50}


def test_the_event_stream_sends_the_history_the_state_and_the_panels(client, pairs):
    client.post("/api/pairs/control", data={"action": "step", "units": 30})
    client.post("/api/pairs/traders/start", data=trader(window=100, warm=False))
    seen = read_stream(client, {"hello", "pairs_history", "pairs_state", "pairs_status", "pairs_books", "pairs_board"})
    history = json.loads(seen["pairs_history"])
    assert [s["t"] for s in history["samples"]] == list(range(31))
    assert history["traders"] == [{"id": 1, "name": "Z-score 1", "color": "lime-400", "active": True, "kind": "zscore",
                                   "entry": 2.0, "exit": 0.0}]
    assert json.loads(seen["pairs_state"])["running"] is False
    assert 'id="pairs-status"' in seen["pairs_status"] and 'data-running="false"' in seen["pairs_status"]
    assert seen["pairs_books"].count('data-book=') == 2
    assert 'data-pairs-id="1"' in seen["pairs_board"] and "learning 0/100" in seen["pairs_board"]


def test_the_stream_stands_in_for_events_on_this_view(client, pairs):
    """A tab on this view holds this stream in place of /events: it brings the notices, keeps the session alive,
    and has nothing to say when only the other market moves."""
    m = registry.get(client.cookies[COOKIE_NAME])
    with client.stream("GET", "/pairs/events") as resp:
        lines = resp.iter_lines()

        def next_event():
            event = None
            for line in lines:
                if line.startswith("event: "):
                    event = line[len("event: "):]
                elif line.startswith("data: ") and event:
                    return event, line[len("data: "):]

        while next_event()[0] != "pairs_board":  # the panels, on connect
            pass
        assert m.has_subscribers and pairs._subscribers  # eviction asks the single-book market
        inside = m.book.best_bid + 1
        assert inside < m.book.best_ask
        client.post("/api/order/limit", data={"price": inside, "size": 5, "side": "bid"})  # wakes it: nothing to say
        client.post("/api/order/market", data={"size": 5, "side": "ask"})  # fills that order: a notice
        event, data = next_event()  # no panels first: this market has not moved
        assert event == "notices" and "now filled in full" in data
        client.post("/api/pairs/control", data={"action": "step", "units": 3})
        assert next_event()[0] == "pairs_samples" and next_event()[0] == "pairs_status"


def test_the_clock_follows_the_speed():
    async def run():
        s = PairsSession()
        s.speed = 100
        await s.start()
        await asyncio.sleep(0.6)
        await s.stop()
        return s.market.time

    assert 40 < asyncio.run(run()) < 80  # about 60: 100 units a second for 0.6 s


def test_a_session_evicted_stops_its_pairs_market():
    async def run():
        client = TestClient(app)
        client.get("/")
        m = registry.get(client.cookies[COOKIE_NAME])
        await m.pairs.start()
        task = m.pairs._task
        m.shutdown()
        await asyncio.sleep(0.01)
        return m.pairs.is_running, task.cancelled() or task.done()

    assert asyncio.run(run()) == (False, True)


def test_the_gap_standard_deviation_on_the_page_is_measured_over_the_chart():
    s = PairsSession()
    s.run_for(2500)
    sds = s.gap_sds()
    gaps = [p["true"] for p in s.samples]
    mean = sum(gaps) / len(gaps)
    assert sds["fair"] == pytest.approx(math.sqrt(sum((g - mean) ** 2 for g in gaps) / len(gaps)))
    assert sds["theory"] == pytest.approx(0.8 * math.sqrt(2 / (2 * math.log(2) / 50)))
    assert sds["units"] == 2501


# ---- The three kinds, and the market's options ----

def test_each_kind_of_trader_starts_learns_and_shows_on_the_board_and_the_chart():
    s = PairsSession()
    s.run_for(2500)
    z, learned_z = s.start_trader(trader(window=200))
    kalman, learned_k = s.start_trader({**KALMAN, "window": 200, "memory": 500.0}, "kalman")
    copula, learned_c = s.start_trader({**COPULA, "window": 100, "trading": 50, "interval": 5.0}, "copula")
    assert (z.name, kalman.name, copula.name) == ("Z-score 1", "Kalman 2", "Copula 3")
    assert learned_z == 200 and learned_k == 2501 and learned_c == 101  # as much as each can use
    assert all(r.trader.warming is None for r in (z, kalman, copula))
    assert copula.trader.fits == 1 and copula.trader.flag_a == 0 == copula.trader.flag_b  # its fit is fresh
    s.run_for(1500)
    rows = {row["kind"]: row for row in s.board()[0]}
    assert rows["kalman"]["detail"].startswith(f"hedge {kalman.trader.beta_hat:.3f} (true beta 1.000)")
    assert copula.trader.copula.describe() in rows["copula"]["detail"]
    assert rows["zscore"]["detail"] == ""
    assert len(rows["zscore"]["signals"]) == len(rows["kalman"]["signals"]) == 1 and len(rows["copula"]["signals"]) == 2
    assert "Kalman filter of B on A with a memory of 500" in rows["kalman"]["describe"]
    assert "fitted to the last 100 and again every 50" in rows["copula"]["describe"]
    p = s.samples[-1]
    assert set(p["pnl"]) == set(p["sig"]) == set(p["hedge"]) == {1, 2, 3}
    assert set(p["bands"]) == {1, 2}  # a copula trader's signal lies in its flags, not in a gap
    assert p["hedge"][1] == 1.0 and p["hedge"][2] == pytest.approx(kalman.trader.beta_hat, abs=0.01)
    assert p["sig"][3] == pytest.approx([copula.trader.flag_a, copula.trader.flag_b], abs=1e-3)
    s.reset()  # each kind starts over as it was set
    assert kalman.trader.filter is None and kalman.trader.warming == (0, 200) and copula.trader.fit is None


def test_a_kalman_traders_bands_are_its_prediction_in_the_charts_terms():
    s = PairsSession()
    s.run_for(600)
    r, _ = s.start_trader({**KALMAN, "window": 100, "memory": 300.0, "entry": 1.5}, "kalman")
    s.run_for(600)
    t = r.trader
    s._sample()
    p = s.samples[-1]
    lo, mean, hi = p["bands"][r.id]
    assert hi - mean == pytest.approx(1.5 * t.error_sd, abs=2e-3) and mean - lo == pytest.approx(hi - mean, abs=2e-3)
    assert p["gap"] - mean == pytest.approx(p["b"] - t.filter.fit(p["a"]), abs=2e-3)  # the filter's error, unchanged


def test_market_options_reach_the_fair_values_and_are_off_by_default():
    s = PairsSession()
    f = s.market.fair
    assert f.beta_sd == 0 and f.stress_share == 0 and f.level_a == LEVEL and f.level_b == LEVEL
    assert f.volatility == 1.0  # a pair whose returns move together: at 0.3 they hardly do (0.12 over a unit of time)
    s.configure({**s.settings, "beta": 1.5, "drifting": True, "beta_sd": 0.2, "beta_half_life": 100.0, "stress": True,
                 "stress_share": 0.3, "stress_volatility": 4.0, "stress_length": 20.0, "stress_drift": -0.5})
    f = s.market.fair
    assert f.level_b == 1.5 * LEVEL and f.beta_sd == 0.2 and f.beta_half_life == 100.0
    assert (f.stress_share, f.stress_volatility, f.stress_length, f.stress_drift) == (0.3, 4.0, 20.0, -0.5)
    s.run_for(500)
    context = s.context()
    assert context["drifting"] and context["stress_on"] and context["beta"] == f.beta != 1.5
    assert s.samples[-1]["beta"] == pytest.approx(f.beta, abs=0.05)


def test_kinds_are_checked_through_the_endpoint(client, pairs):
    assert client.post("/api/pairs/traders/start", data={"kind": "nope"}).status_code == 422
    bad = client.post("/api/pairs/traders/start", data={**KALMAN, "kind": "kalman", "memory": 1})
    assert bad.status_code == 422 and set(bad.json()["errors"]) == {"memory"}
    bad = client.post("/api/pairs/traders/start", data={**COPULA, "kind": "copula", "exit": 2, "entry": 1})
    assert bad.status_code == 422 and "exit" in bad.json()["errors"]
    good = client.post("/api/pairs/traders/start", data={**COPULA, "kind": "copula", "warm": ""}).json()
    assert good["name"] == "Copula 1" and good["message"] == "Copula 1 is watching: it trades once it has seen 400."
    good = client.post("/api/pairs/traders/start", data={**KALMAN, "kind": "kalman", "rebalance": ""}).json()
    assert good["name"] == "Kalman 2" and pairs.runner(2).trader.rebalance is False
    html = client.get("/").text
    for kind in TRADER_KINDS:
        assert f'data-pairs-kind="{kind}"' in html and f'name="kind" value="{kind}"' in html
    controls = ('data-pairs-toggle="signal"', 'data-pairs-vzoom="gap"', 'id="pairs-hzoom-slider"', 'id="pairs-hud"')
    assert all(control in html for control in controls)


def test_every_market_the_form_allows_starts_above_the_floor():
    beta = next(p for p in MARKET_PARAMS if p.name == "beta")
    assert LEVEL * beta.low >= FLOOR and LEVEL >= FLOOR


def test_the_view_says_when_the_floor_has_been_reached(client, pairs, monkeypatch):
    panels = lambda: render_pairs_panels(pairs.context())["pairs_status"]  # noqa: E731
    assert "data-pairs-floor" not in panels() and pairs.context()["pushed"] == 0
    assert f"Neither fair value goes below {FLOOR:g} ticks" in client.get("/").text
    monkeypatch.setattr(pairs_module, "LEVEL", 11.0)  # just above it, so that a market gets there in moments
    pairs.configure({**pairs.settings, "volatility": 5.0})
    pairs.run_for(100)
    assert pairs.market.fair.pushed > 0 and pairs.market.mid(A) is not None
    status = panels()
    assert "data-pairs-floor" in status and f"{pairs.market.fair.pushed:,.1f}" in status


def test_a_long_fit_at_a_high_volatility_stays_above_the_floor(monkeypatch):
    # A fit's runs are as long as they like: without the floor they could end in books with one side.
    monkeypatch.setattr(pairs_module, "LEVEL", 20.0)
    session = PairsSession()
    session.configure({**session.settings, "volatility": 5.0})
    values = {**TRADER, "window": 20}
    trader = TRADER_KINDS["zscore"].build(values)
    market = pairs_module.market_from(session.settings, 3, (trader,))
    while market.time < 500:
        market.step()
    assert market.fair.pushed > 0 and market.mid(A) is not None and market.mid(B) is not None
    assert min(market.fair.fair(A), market.fair.fair(B)) >= FLOOR - 2
