"""Whether the order flow pays liquidity providers (LOBManager.regime), and fills held against a null with no skill."""

import math

from starlette.testclient import TestClient

import app.manager as manager_module
from app.main import app, get_view_context, registry, render_panels
from app.manager import LOBManager
from app.sessions import COOKIE_NAME
from app.strategy_specs import STRATEGY_SPECS


def start(m, kind, **settings):
    spec = STRATEGY_SPECS[kind]
    return m.start_strategy(spec, {**spec.defaults(), **settings})


def test_the_regime_is_the_spread_against_the_mid_move_per_trade(monkeypatch):
    monkeypatch.setattr(manager_module, "REGIME_TRADES", 10**12)  # weights all but equal: plain averages
    m = LOBManager()
    m.seed = 4
    m.reset()
    seen = []  # the spread and mid just before each trade, as Wyart et al. (2008) measure them
    process = m._process

    def watched(order):
        if m._takes(order) and m.book.mid_price is not None:
            seen.append((m.book.spread, m.book.mid_price))
        return process(order)

    m._process = watched
    m.step(8000)
    regime = m.regime()
    spread = sum(s for s, _ in seen) / len(seen)
    sigma1 = math.sqrt(sum((b - a) ** 2 for (_, a), (_, b) in zip(seen, seen[1:], strict=False)) / (len(seen) - 1))
    assert regime["trades"] == len(seen) > 500
    assert math.isclose(regime["spread"], spread, rel_tol=1e-6) and math.isclose(regime["sigma1"], sigma1, rel_tol=1e-6)
    assert regime["ratio"] > 2  # the ZI flow: a spread several times the mid's move per trade
    assert regime["make"] > 0.3 > -0.3 > regime["take"]  # resting pays here, taking costs
    assert regime["join"] is None and not regime["joiner"]


def test_fills_are_held_against_a_null_with_no_skill():
    m = LOBManager()
    m.seed = 2
    m.reset()
    m.step(3000)
    joiner = start(m, "quote_joiner")
    random = start(m, "random", switch=0.2)  # takes only: against the model's market orders, which know nothing
    maker = start(m, "avellaneda_stoikov")  # rests only: against the joiner
    m.step(30_000)
    mark = m._mark()
    j, r, a = (s.account.summary(mark) for s in (joiner, random, maker))
    assert j["null_lots"] == 0 and j["markout_lots"] > 1000  # the null itself
    assert abs(r["vs_null"]) < 0.3 and r["null_lots"] == r["markout_lots"]  # an uninformed taker is about even
    assert 0 < a["null_lots"] <= a["markout_lots"]
    regime = m.regime()
    assert regime["joiner"] and regime["join"] < regime["make"]  # the naive joiner earns less than an average order
    # With the joiner stopped, resting fills have no null; taking fills still do.
    m.stop_strategy(joiner)
    before = (a["null_lots"], r["null_lots"])
    m.step(5000)
    a2, r2 = maker.account.summary(m._mark()), random.account.summary(m._mark())
    assert a2["null_lots"] == before[0] < a2["markout_lots"] and r2["null_lots"] > before[1]


def test_without_a_joiner_resting_fills_have_no_null():
    m = LOBManager()
    m.step(2000)
    mm = start(m, "market_making", half_spread=1)
    m.step(20_000)
    summary = mm.account.summary(m._mark())
    assert summary["markout_lots"] > 0 and summary["null_lots"] == 0 and summary["vs_null"] is None


def test_the_board_shows_the_regime_and_the_null():
    client = TestClient(app)
    client.get("/")
    m = registry.get(client.cookies[COOKIE_NAME])
    assert 'id="regime-line"' not in render_panels(get_view_context(m))["strategies"]  # too few trades yet
    start(m, "penny_jumping")
    m.step(8000)
    board = render_panels(get_view_context(m))["strategies"]
    assert 'data-verdict="pays"' in board and "The order flow pays resting orders here." in board
    assert "&times; the mid's move per trade" in board
    assert 'data-joiner="false"' in board and "Vs null" in board
    assert '<div id="add-null" >' in client.get("/").text  # offered
    client.post("/api/strategies/start", data={"kind": "quote_joiner", "size": 5, "limit": 50, "interval": 0.5})
    m.step(3000)
    board = render_panels(get_view_context(m))["strategies"]
    assert 'data-joiner="true"' in board and ">the null<" in board and "the quote joiner" in board
    assert '<div id="add-null" hidden>' in client.get("/").text  # no longer offered
