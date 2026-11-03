"""Tests for FastAPI web application endpoints."""

import asyncio
import json
import math
import re

import pytest
from starlette.testclient import TestClient
from app.main import app, get_view_context, registry, render_panels
from app.manager import MANUAL_ACTION_DT
from app.sessions import COOKIE_NAME, SessionRegistry
from lob.order import Side


@pytest.fixture
def client():
    """A browser with its own session: the first request issues the session cookie."""
    client = TestClient(app)
    client.get("/")
    return client


@pytest.fixture
def manager(client):
    """The simulation behind the client's session."""
    return registry.get(client.cookies[COOKIE_NAME])


def candle_volume(candles):
    return sum(c["volume"] for c in candles)


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Limit Order Book" in response.text
    assert "Depth of Market" in response.text
    assert "new EventSource('/events')" in response.text
    assert 'data-instance="' in response.text


def test_page_loads_nothing_from_the_internet(client):
    html = client.get("/").text
    external = re.findall(r'(?:src|href)="(https?://[^"]+)"', html)
    assert external == ["https://arxiv.org/abs/1012.0349"]  # a link to the paper, not a resource
    local = re.findall(r'(?:src|href)="(/static/[^"]+)"', html)
    assert len(local) == 7  # favicon, fonts.css, themes.css, app.css, htmx, Lightweight Charts, lob-ladder.js
    for url in local:
        assert client.get(url).status_code == 200, url
    fonts = client.get("/static/css/fonts.css").text
    for url in set(re.findall(r"url\((/static/fonts/[^)]+)\)", fonts)):
        assert client.get(url).status_code == 200, url


def test_submit_limit_order(client, manager):
    resp = client.post("/api/order/limit", data={"price": 50, "size": 10, "side": "bid"})
    assert resp.status_code == 200
    oid = next(iter(manager.manual_order_ids))
    assert f"Buy order #{oid} for 10 lots at 50 is resting in the book." in resp.text
    assert manager.event_count == 1
    assert manager.book.depth_at_price(50, Side.BID) == 10
    assert len(manager.manual_order_ids) == 1


def test_submit_market_order(client, manager):
    best_ask = manager.book.best_ask
    resp = client.post("/api/order/market", data={"size": 5, "side": "bid"})
    assert resp.status_code == 200
    assert f"Bought 5 lots at {best_ask}" in resp.text
    assert len(manager.recent_trades) > 0
    assert manager.recent_trades[0].side == "bid"


def test_limit_order_result_reports_fills_and_what_rests(client, manager):
    best_ask = manager.book.best_ask
    depth = manager.book.depth_at_price(best_ask, Side.ASK)
    resp = client.post("/api/order/limit", data={"price": best_ask, "size": depth + 3, "side": "bid"})
    oid = next(iter(manager.manual_order_ids))
    assert f"Bought {depth} at {best_ask} on arrival; the other 3 rest at {best_ask} as order #{oid}." in resp.text

    best_bid = manager.book.best_bid  # now the resting remainder
    resp = client.post("/api/order/limit", data={"price": best_bid, "size": 1, "side": "ask"})
    assert f"Sold 1 lot at {best_bid} on arrival; nothing rests." in resp.text
    assert len(manager.manual_order_ids) == 1  # an order filled on arrival is not listed as active


def test_ladder_marks_own_resting_orders(client, manager):
    best_bid = manager.book.best_bid
    client.post("/api/order/limit", data={"price": best_bid, "size": 7, "side": "bid"})
    client.post("/api/order/limit", data={"price": best_bid, "size": 3, "side": "bid"})
    ctx = get_view_context(manager)
    marked = {lv["price"]: lv["own"] for lv in ctx["bids_display"] + ctx["asks_display"] if lv["own"]}
    assert marked == {best_bid: 10}
    assert "you 10" in render_panels(ctx)["ladder"]


def test_market_order_walks_at_most_the_protection_and_the_rest_rests(client, manager):
    best_ask = manager.book.best_ask
    resp = client.post("/api/order/market", data={"size": 1_000_000, "side": "bid", "protection": 3})
    limit = best_ask + 3
    oid = next(iter(manager.manual_order_ids))
    prices = {t.price for t in manager.recent_trades}
    assert prices == set(range(best_ask, limit + 1))  # walked exactly up to the protection limit
    rest = manager.book.get_order(oid)
    assert rest.price == limit and rest.side == Side.BID
    assert manager.book.best_bid == limit and manager.book.best_ask > limit
    filled = 1_000_000 - rest.size
    assert f"Partly filled: bought {filled:,} of 1,000,000 lots at {best_ask}\u2013{limit}" in resp.text
    rest_text = f"The other {rest.size:,} rest as a buy order at {limit}, the protection limit, as order #{oid}."
    assert rest_text in resp.text


def test_market_order_with_zero_protection_fills_only_at_the_best_price(client, manager):
    best_ask = manager.book.best_ask
    depth = manager.book.depth_at_price(best_ask, Side.ASK)
    client.post("/api/order/market", data={"size": depth + 7, "side": "bid", "protection": 0})
    assert {t.price for t in manager.recent_trades} == {best_ask}
    assert manager.book.depth_at_price(best_ask, Side.BID) == 7  # the rest waits at the best price


def test_market_sell_protection_stops_at_the_lowest_price(client, manager):
    resp = client.post("/api/order/market", data={"size": 1_000_000, "side": "ask", "protection": 1000})
    oid = next(iter(manager.manual_order_ids))
    assert manager.book.get_order(oid).price == 1  # never below one tick
    assert manager.book.best_bid is None  # it sold to every bid on the way down
    assert "rest as a sell order at 1, the protection limit" in resp.text


def test_market_order_with_no_opposite_side_places_nothing(client, manager):
    client.post("/api/order/market", data={"size": 1_000_000, "side": "bid", "protection": 1000})
    client.post("/api/order/cancel", data={"order_id": next(iter(manager.manual_order_ids))})
    assert manager.book.best_ask is None and not manager.manual_order_ids
    events = manager.event_count
    resp = client.post("/api/order/market", data={"size": 5, "side": "bid"})
    assert "Nothing filled: there are no asks in the book" in resp.text
    assert not manager.manual_order_ids and manager.event_count == events


@pytest.mark.parametrize("path, data", [
    ("/api/order/limit", {"price": 100, "size": 0, "side": "bid"}),
    ("/api/order/limit", {"price": 0, "size": 5, "side": "bid"}),
    ("/api/order/limit", {"price": 100, "size": 5, "side": "buy"}),
    ("/api/order/market", {"size": -1, "side": "ask"}),
    ("/api/order/market", {"size": 5, "side": "bid", "protection": -1}),
    ("/api/order/market", {"size": 5, "side": "bid", "protection": 1001}),
    ("/api/sim/config", {"model": "random_walk"}),
    ("/api/sim/control", {"action": "rewind"}),
    ("/api/sim/control", {"action": "step", "n": 10**9}),
    ("/api/engine", {"engine": "fastest"}),
])
def test_invalid_input_is_rejected(client, path, data, manager):
    resp = client.post(path, data=data)
    assert resp.status_code == 422
    assert manager.event_count == 0


def test_cancel_order(client, manager):
    # Submit resting limit order far out
    client.post("/api/order/limit", data={"price": 50, "size": 10, "side": "bid"})
    assert len(manager.manual_order_ids) > 0
    oid = list(manager.manual_order_ids)[0]

    resp = client.post("/api/order/cancel", data={"order_id": oid})
    assert resp.status_code == 204
    assert oid not in manager.manual_order_ids
    assert manager.book.get_order(oid) is None


def test_sim_control_step_and_reset(client, manager):
    init_events = manager.event_count
    resp = client.post("/api/sim/control", data={"action": "step", "n": 5})
    assert resp.status_code == 204
    assert manager.event_count > init_events

    resp = client.post("/api/sim/control", data={"action": "reset"})
    assert resp.status_code == 200
    assert "Start Sim" in resp.text
    assert manager.event_count == 0
    assert manager.sim_time == 0.0


def test_sim_config(client, manager):
    resp = client.post("/api/sim/config", data={"model": "mike_farmer", "speed": 15})
    assert resp.status_code == 204
    assert manager.model_type == "mike_farmer"
    assert manager.speed == 15


def test_engine_toggle_round_trip(client, manager):
    resp = client.post("/api/engine")
    assert resp.status_code == 200
    assert manager.engine_type == "pro_rata"
    assert "Pro-Rata" in resp.text and 'id="engine-toggle"' in resp.text

    resp = client.post("/api/engine")
    assert manager.engine_type == "price_time"
    assert "Price-Time" in resp.text


def test_engine_can_be_set_explicitly(client, manager):
    resp = client.post("/api/engine", data={"engine": "pro_rata"})
    assert resp.status_code == 200
    assert manager.engine_type == "pro_rata"


def test_snapshot_api(client):
    resp = client.get("/api/snapshot")
    assert resp.status_code == 200
    data = resp.json()
    assert "best_bid" in data
    assert "best_ask" in data
    assert "spread" in data


def test_candlesticks_api(client):
    resp = client.get("/api/candlesticks")
    assert resp.status_code == 200
    data = resp.json()
    assert "interval" in data
    assert "candles" in data
    assert len(data["candles"]) > 0
    first_candle = data["candles"][0]
    for key in ("time", "open", "high", "low", "close", "volume", "buy_volume", "sell_volume"):
        assert key in first_candle


def test_no_candle_history_before_any_trade(client):
    candles = client.get("/api/candlesticks").json()["candles"]
    assert len(candles) == 1
    assert candles[0]["time"] == 0
    assert candle_volume(candles) == 0
    assert candles[0]["open"] == candles[0]["close"] == 100.0


def test_candles_are_built_from_trades(client, manager):
    client.post("/api/order/market", data={"size": 10, "side": "bid"})
    client.post("/api/order/market", data={"size": 4, "side": "ask"})
    candles = client.get("/api/candlesticks").json()["candles"]
    assert candle_volume(candles) == manager.traded_volume == 14
    assert sum(c["buy_volume"] for c in candles) == 10
    assert sum(c["sell_volume"] for c in candles) == 4
    traded_prices = {t.price for t in manager.recent_trades}
    assert {c["high"] for c in candles if c["volume"]} <= traded_prices


def test_interval_change_reaggregates_history(client, manager):
    client.post("/api/sim/control", data={"action": "step", "n": 2000})
    traded = manager.traded_volume
    assert traded > 0
    version = manager.chart_version
    totals = {}
    for interval in [1, 2, 5, 10, 15, 30, 60, 180, 300]:
        resp = client.post("/api/chart/interval", data={"interval": interval})
        assert resp.status_code == 204
        data = client.get("/api/candlesticks").json()
        assert data["interval"] == interval
        times = [c["time"] for c in data["candles"]]
        assert times == sorted(set(times)) and all(t % interval == 0 for t in times)
        totals[interval] = candle_volume(data["candles"])
    assert set(totals.values()) == {traded}
    assert manager.chart_version == version + 9


def test_candles_follow_the_simulation_clock(client, manager):
    client.post("/api/sim/control", data={"action": "step", "n": 500})
    last = client.get("/api/candlesticks").json()["candles"][-1]
    n = manager.bar_interval
    assert last["time"] == math.floor(manager.sim_time / n) * n


def test_sim_time_never_runs_backwards(client, manager):
    times = [manager.sim_time]
    client.post("/api/sim/control", data={"action": "step", "n": 200})
    times.append(manager.sim_time)
    for price in (60, 61, 62):
        client.post("/api/order/limit", data={"price": price, "size": 1, "side": "bid"})
        times.append(manager.sim_time)
    assert times[-1] == pytest.approx(times[1] + 3 * MANUAL_ACTION_DT)
    client.post("/api/sim/control", data={"action": "step", "n": 1})
    times.append(manager.sim_time)
    client.post("/api/sim/config", data={"model": "parlour"})
    client.post("/api/sim/control", data={"action": "step", "n": 50})
    times.append(manager.sim_time)
    assert times == sorted(times)
    assert times[-1] > times[-3]


def read_stream(client, wanted):
    """First payload of each event type on /events, until every wanted type has arrived."""
    seen = {}
    with client.stream("GET", "/events") as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
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


def test_event_stream_carries_panels_and_chart(client, manager):
    panels = {"ladder", "metrics", "trades", "active_orders", "sim_status"}
    seen = read_stream(client, {"hello", "candles"} | panels)
    assert json.loads(seen["hello"]) == {"instance": manager.instance_id}
    assert json.loads(seen["candles"])[-1]["time"] == 0
    assert 'id="ladder-container"' in seen["ladder"]
    assert 'data-running="false"' in seen["sim_status"]


def test_separate_chart_stream_is_gone(client):
    assert client.get("/events/chart").status_code == 404


def test_speed_up_to_1000(client, manager):
    resp = client.post("/api/sim/config", data={"speed": 1000})
    assert resp.status_code == 204
    assert manager.speed == 1000


def test_play_and_pause_simulation(client, manager):
    async def _verify():
        await manager.start_sim_loop()
        assert manager.is_running is True
        await manager.stop_sim_loop()
        assert manager.is_running is False

    asyncio.run(_verify())

    resp_play = client.post("/api/sim/control", data={"action": "play"})
    assert resp_play.status_code == 200
    assert "Pause Sim" in resp_play.text

    resp_pause = client.post("/api/sim/control", data={"action": "pause"})
    assert resp_pause.status_code == 200
    assert "Start Sim" in resp_pause.text


def test_stochastic_mid_price_evolution(client, manager):
    initial_mid = manager.book.mid_price
    assert initial_mid == 100.0

    # Run steps and collect mid-price values
    mids = set()
    for _ in range(300):
        client.post("/api/sim/control", data={"action": "step"})
        if manager.book.mid_price is not None:
            mids.add(manager.book.mid_price)

    # Limit orders landing inside the spread and market orders consuming the
    # best quotes must move the mid-price across several levels.
    assert len(mids) > 1


# ---- One simulation per browser session ----

def test_each_browser_gets_its_own_simulation(client, manager):
    other = TestClient(app)
    other.get("/")
    assert other.cookies[COOKIE_NAME] != client.cookies[COOKIE_NAME]
    client.post("/api/order/market", data={"size": 5, "side": "bid"})
    assert manager.traded_volume == 5
    assert registry.get(other.cookies[COOKIE_NAME]).traded_volume == 0


def test_requests_with_the_cookie_reuse_the_session(client, manager):
    sessions = len(registry)
    client.post("/api/sim/control", data={"action": "step", "n": 10})
    assert len(registry) == sessions
    assert manager.event_count > 0


def test_unknown_session_id_is_replaced_by_a_server_issued_one():
    resp = TestClient(app).get("/", headers={"cookie": f"{COOKIE_NAME}=chosen-by-client"})
    issued = resp.cookies.get(COOKIE_NAME)
    assert issued and issued != "chosen-by-client"
    assert issued in registry and "chosen-by-client" not in registry


def test_static_files_do_not_create_sessions():
    client = TestClient(app)
    sessions = len(registry)
    resp = client.get("/static/js/lightweight-charts.standalone.production.js")
    assert resp.status_code == 200
    assert "set-cookie" not in resp.headers
    assert len(registry) == sessions


def test_registry_evicts_least_recently_seen_session_at_capacity():
    reg = SessionRegistry(max_sessions=2)
    first, first_manager, _ = reg.get_or_create(None)
    second, _, _ = reg.get_or_create(None)
    reg.get_or_create(first)  # first is now the more recent
    first_manager.is_running = True
    third, _, created = reg.get_or_create(None)
    assert created and len(reg) == 2
    assert second not in reg and first in reg and third in reg


def test_registry_evicts_idle_sessions_and_stops_them():
    reg = SessionRegistry(idle_timeout=0.0)
    idle_id, idle_manager, _ = reg.get_or_create(None)

    async def evict_while_running():
        await idle_manager.start_sim_loop()
        task = idle_manager._bg_task
        reg.get_or_create(None)  # any request sweeps idle sessions
        await asyncio.sleep(0)
        return task

    task = asyncio.run(evict_while_running())
    assert idle_id not in reg
    assert idle_manager.is_running is False and task.cancelled()


def test_registry_keeps_idle_sessions_with_open_streams():
    reg = SessionRegistry(idle_timeout=0.0)
    watched_id, watched, _ = reg.get_or_create(None)
    watched.subscribe()
    reg.get_or_create(None)
    assert watched_id in reg


# ---- Model parameters and seed ----

def zi_form(**overrides):
    form = {"model": "smith_zi", "mu": 6.0, "lam": 1.0, "delta": 0.03, "price_range": 15, "lot_size": 5, "seed": 42}
    form.update(overrides)
    return form


def trade_tape(manager):
    return [(t.price, t.size, t.side) for t in manager.recent_trades]


def test_index_offers_every_model_and_its_parameters(client):
    from app.model_specs import MODEL_SPECS

    html = client.get("/").text
    for spec in MODEL_SPECS.values():
        assert f'value="{spec.key}"' in html
    assert 'id="model-params"' in html and 'name="lam"' in html and 'name="seed"' in html


def test_switching_model_returns_its_parameter_panel(client, manager):
    resp = client.post("/api/sim/model", data={"model": "hawkes_zi"})
    assert resp.status_code == 200
    assert manager.model_type == "hawkes_zi"
    assert 'name="branching"' in resp.text and 'name="decay"' in resp.text


def test_applying_parameters_rebuilds_the_model(client, manager):
    resp = client.post("/api/sim/params", data=zi_form(mu=3.0, price_range=30, seed=7))
    assert resp.status_code == 200 and "Applied" in resp.text
    assert manager.model.mu == 3.0 and manager.model.price_range == 30
    assert manager.seed == 7


@pytest.mark.parametrize("field, value", [
    ("mu", "-1"), ("mu", "abc"), ("price_range", "2.5"), ("lot_size", "0"), ("seed", "-3"),
])
def test_invalid_parameters_change_nothing(client, manager, field, value):
    before = (dict(manager.model_params["smith_zi"]), manager.seed, manager.model)
    resp = client.post("/api/sim/params", data=zi_form(**{field: value}))
    assert resp.status_code == 422
    assert "Not applied" in resp.text and "border-rose-500" in resp.text
    assert (manager.model_params["smith_zi"], manager.seed, manager.model) == before


def test_parameters_are_kept_per_model(client, manager):
    client.post("/api/sim/params", data=zi_form(mu=2.0))
    client.post("/api/sim/model", data={"model": "parlour"})
    client.post("/api/sim/model", data={"model": "smith_zi"})
    assert manager.model.mu == 2.0


def test_same_seed_replays_the_same_run_after_reset(client, manager):
    def run(seed):
        client.post("/api/sim/params", data=zi_form(seed=seed))
        client.post("/api/sim/control", data={"action": "reset"})
        client.post("/api/sim/control", data={"action": "step", "n": 3000})
        return trade_tape(manager), manager.sim_time

    first = run(11)
    assert first == run(11)
    assert first != run(12)


@pytest.mark.parametrize("model", ["smith_zi", "hawkes_zi", "mike_farmer", "mike_farmer_2008", "parlour"])
def test_every_model_runs_in_the_dashboard(client, manager, model):
    client.post("/api/sim/model", data={"model": model})
    client.post("/api/sim/control", data={"action": "step", "n": 3000})
    assert manager.event_count > 0 and manager.traded_volume > 0
    assert manager.book.best_bid is not None and manager.book.best_ask is not None


# ---- Keeping tabs of one session in sync ----

def test_control_changes_bump_the_config_version(client, manager):
    changes = [
        lambda: client.post("/api/sim/model", data={"model": "parlour"}),
        lambda: client.post("/api/engine"),
        lambda: client.post("/api/sim/config", data={"speed": 300}),
        lambda: client.post("/api/sim/model", data={"model": "smith_zi"}),
        lambda: client.post("/api/sim/params", data=zi_form(mu=2.0)),
    ]
    for change in changes:
        before = manager.config_version
        change()
        assert manager.config_version == before + 1
    for no_change in (lambda: client.post("/api/sim/params", data=zi_form(mu=-1)),  # invalid
                      lambda: client.post("/api/sim/control", data={"action": "reset"}),
                      lambda: client.post("/api/chart/interval", data={"interval": 5})):
        before = manager.config_version
        no_change()
        assert manager.config_version == before


def test_stream_sends_controls_on_connect(client, manager):
    controls = json.loads(read_stream(client, {"controls"})["controls"])
    assert (controls["model"], controls["engine"], controls["speed"]) == ("smith_zi", "price_time", 50)
    assert f'data-signature="{controls["params_signature"]}"' in controls["params_html"]
    assert 'name="lam"' in controls["params_html"]
    assert 'data-engine="price_time"' in controls["engine_html"]


def test_stream_pushes_control_changes_made_by_another_tab(client, manager):
    def next_controls(lines):
        event = None
        for line in lines:
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: ") and event == "controls":
                return json.loads(line[len("data: "):])

    with client.stream("GET", "/events") as resp:
        lines = resp.iter_lines()
        assert next_controls(lines)["model"] == "smith_zi"
        client.post("/api/sim/model", data={"model": "hawkes_zi"})  # same session, another request
        update = next_controls(lines)
    assert update["model"] == "hawkes_zi"
    assert 'name="branching"' in update["params_html"]
    assert update["params_signature"].startswith("hawkes_zi|")


# ---- Stylized facts panel ----

def test_facts_panel_on_the_page_and_in_the_stream(client, manager):
    html = client.get("/").text
    assert 'id="facts-container"' in html and "Warming up" in html
    seen = read_stream(client, {"hello", "facts"})
    assert 'id="facts-container"' in seen["facts"]


def test_collect_gathers_a_sample_and_the_panel_reads_it(client, manager):
    resp = client.post("/api/sim/collect", data={"n": 60_000})
    assert resp.status_code == 204
    assert manager.facts.steps >= 60_000 and manager.collecting is None and not manager.is_running
    seen = read_stream(client, {"facts"})
    assert 'data-fact="Bid-ask bounce" data-call="Yes"' in seen["facts"]
    assert client.post("/api/sim/collect", data={"n": 10}).status_code == 422  # below the minimum


def test_only_one_collection_at_a_time_and_play_waits_for_it(client, manager):
    async def _two_at_once():
        first = asyncio.create_task(manager.collect(200_000))
        await asyncio.sleep(0)  # let the first one start
        second = await manager.collect(10_000)
        assert await first is True
        return second

    assert asyncio.run(_two_at_once()) is False

    manager.collecting = (0, 1)  # as if a collection were running
    try:
        resp = client.post("/api/sim/control", data={"action": "play"})
        assert "Start Sim" in resp.text and not manager.is_running
    finally:
        manager.collecting = None


def test_new_model_settings_engine_or_book_start_a_new_sample(client, manager):
    def generation_after(action):
        before = manager.facts.generation
        action()
        return manager.facts.generation - before

    assert generation_after(lambda: manager.set_model("hawkes_zi")) == 1
    assert generation_after(lambda: manager.configure("hawkes_zi", manager.model_params["hawkes_zi"], 7)) == 1
    assert generation_after(lambda: manager.configure("parlour", manager.model_params["parlour"], 7)) == 0
    assert generation_after(manager.toggle_engine) == 1
    assert generation_after(lambda: manager.reset(mid=100)) == 1
    assert generation_after(lambda: manager.set_speed(500)) == 0


# ---- The user's resting orders ----

def test_a_model_cancel_aimed_at_your_order_is_skipped(manager, monkeypatch):
    from lob.order import cancel_order

    order, _ = manager.submit_limit_order(price=20, size=5, side=Side.BID)
    events, clock = manager.event_count, manager.sim_time
    monkeypatch.setattr(manager.model, "generate_event",
                        lambda book, t, rng: cancel_order(order_id=order.order_id, timestamp=t + 1.0))
    manager.step(3)
    assert manager.book.get_order(order.order_id) is not None
    assert order.order_id in manager.manual_order_ids
    assert manager.event_count == events  # skipped, not processed
    assert manager.sim_time == clock + 3.0  # the model's clock still moves on


def test_models_never_cancel_your_orders_but_still_cancel_their_own(manager):
    from lob.order import limit_order

    mine, _ = manager.submit_limit_order(price=20, size=5, side=Side.BID)
    theirs = limit_order(price=21, size=5, timestamp=manager.sim_time, side=Side.BID)
    manager.book.add_order(theirs)  # the same kind of order, but the model's
    for model in ("smith_zi", "hawkes_zi", "mike_farmer", "mike_farmer_2008", "parlour"):
        manager.set_model(model)
        manager.step(20_000)
        assert manager.book.get_order(mine.order_id) is not None, model
    assert mine.order_id in manager.manual_order_ids
    assert manager.book.get_order(theirs.order_id) is None  # canceled by a model along the way


# ---- Iceberg orders in the dashboard ----

def test_an_iceberg_order_shows_only_its_peak(client, manager):
    best_bid = manager.book.best_bid
    depth = manager.book.depth_at_price(best_bid, Side.BID)
    resp = client.post("/api/order/limit", data={"price": best_bid, "size": 100, "side": "bid", "display": "10"})
    oid = next(iter(manager.manual_order_ids))
    assert f"Buy order #{oid} for 100 lots at {best_bid} is resting in the book, showing 10 at a time." in resp.text
    assert manager.book.depth_at_price(best_bid, Side.BID) == depth + 10
    ctx = get_view_context(manager)
    assert ctx["active_orders"] == [{"order_id": oid, "side": "bid", "price": best_bid, "size": 10, "hidden": 90,
                                     "timestamp": manager.book.get_order(oid).timestamp}]
    panels = render_panels(ctx)
    assert "you 10 +90 hidden" in panels["ladder"]
    assert "+90" in panels["active_orders"]


@pytest.mark.parametrize("display", ["0", "-5", "abc", "2.5"])
def test_invalid_iceberg_peaks_are_rejected(client, manager, display):
    resp = client.post("/api/order/limit", data={"price": 50, "size": 100, "side": "bid", "display": display})
    assert resp.status_code == 422 and manager.event_count == 0


def test_an_empty_peak_means_an_ordinary_order(client, manager):
    client.post("/api/order/limit", data={"price": 50, "size": 100, "side": "bid", "display": ""})
    assert manager.book.depth_at_price(50, Side.BID) == 100


def test_order_flow_card_fills_in_after_a_collection(client, manager):
    html = client.get("/").text
    assert 'id="ofi-container"' in html and "Warming up with the stylized facts" in html
    client.post("/api/sim/collect", data={"n": 60_000})
    seen = read_stream(client, {"ofi"})
    assert "ticks per lot" in seen["ofi"] and seen["ofi"].count("<circle") >= 50


# ---- VWAP ----

def test_session_vwap_on_candles_and_in_order_results(client, manager):
    assert manager.session_vwap() is None and manager.get_candles()[-1]["vwap"] is None
    best_ask, best_bid = manager.book.best_ask, manager.book.best_bid
    resp = client.post("/api/order/market", data={"size": 10, "side": "bid"})
    assert manager.session_vwap() == best_ask
    assert "Your average was" not in resp.text  # no trades before it, so no benchmark
    resp = client.post("/api/order/market", data={"size": 10, "side": "ask"})
    assert manager.session_vwap() == pytest.approx((best_ask + best_bid) / 2)
    below = (best_ask - best_bid) / best_ask * 100
    assert f"Your average was {below:.1f}% below the session VWAP of {best_ask:.2f}." in resp.text
    assert manager.get_candles()[-1]["vwap"] == pytest.approx((best_ask + best_bid) / 2)
    manager.sim_time += 20  # bars without trades carry the VWAP forward
    assert manager.get_candles()[-1]["vwap"] == pytest.approx((best_ask + best_bid) / 2)


# ---- Export ----

def test_export_trades_and_bars(client, manager):
    import io

    import pandas as pd

    client.post("/api/order/market", data={"size": 30, "side": "bid"})
    manager.step(2_000)
    resp = client.get("/api/export/trades.csv")
    assert resp.status_code == 200 and resp.headers["content-disposition"] == 'attachment; filename="lob-trades.csv"'
    trades = pd.read_csv(io.StringIO(resp.text))
    assert list(trades.columns) == ["time", "price", "size", "aggressor", "maker_order_id", "taker_order_id", "yours"]
    assert trades["size"].sum() == manager.traded_volume
    assert trades.loc[trades["yours"], "size"].sum() == 30  # the user's market buy
    assert set(trades["aggressor"]) == {"buy", "sell"}

    bars = pd.read_csv(io.StringIO(client.get("/api/export/bars.csv").text))
    assert list(bars.columns) == ["time", "open", "high", "low", "close", "volume", "buy_volume", "sell_volume", "vwap"]
    assert bars["volume"].sum() == manager.traded_volume
    assert bars["vwap"].iloc[-1] == pytest.approx(manager.session_vwap())

    parquet = pd.read_parquet(io.BytesIO(client.get("/api/export/trades.parquet").content))
    pd.testing.assert_frame_equal(parquet, trades, check_dtype=False)
    assert client.get("/api/export/orders.csv").status_code == 422
    assert client.get("/api/export/trades.xlsx").status_code == 422


def test_trade_log_keeps_the_most_recent_fills(client, manager, monkeypatch):
    import app.manager as manager_module

    monkeypatch.setattr(manager_module, "MAX_TRADE_LOG", 50)
    manager.step(3_000)
    frame = manager.trades_frame()
    assert len(frame) == 50 and len(manager._trade_log["time"]) <= 100
    assert frame["time"].is_monotonic_increasing


# ---- Fills of the user's resting orders ----

def test_fills_of_your_resting_orders_are_reported(client, manager):
    from lob.order import market_order

    inside = manager.book.best_bid + 1  # inside the spread, so the first sell market order hits it
    client.post("/api/order/limit", data={"price": inside, "size": 8, "side": "bid"})
    oid = next(iter(manager.manual_order_ids))
    manager._process(market_order(5, manager.sim_time, Side.ASK))  # someone else's sell
    manager._process(market_order(5, manager.sim_time, Side.ASK))
    texts = [f.describe() for f in manager.own_fills]
    assert texts == [f"Your buy order #{oid} filled 5 lots at {inside}, 3 left.",
                     f"Your buy order #{oid} filled 3 lots at {inside}, now filled in full."]
    panel = render_panels(get_view_context(manager))["active_orders"]
    assert "Recent fills of your resting orders" in panel
    assert panel.index("now filled in full") < panel.index("3 left")  # newest first


# ---- Split orders (TWAP) ----

def test_sweep_estimate_walks_the_visible_book():
    from app.manager import sweep_estimate
    from lob.book import OrderBook
    from lob.order import limit_order

    book = OrderBook()
    for price, size in ((101, 25), (102, 30), (103, 40)):
        book.add_order(limit_order(price, size, 0.0, Side.ASK))
    assert sweep_estimate(book, Side.BID, 40, limit=110) == (40, (25 * 101 + 15 * 102) / 40)
    assert sweep_estimate(book, Side.BID, 500, limit=102) == (55, (25 * 101 + 30 * 102) / 55)  # stops at the limit


def test_a_split_order_sends_its_slices_on_schedule_and_reports(client, manager):
    resp = client.post("/api/order/market", data={"size": 40, "side": "bid", "slices": 4, "interval": 2.0})
    assert "Splitting a buy of 40 lots into 4 market orders of about 10, one every 2 time units" in resp.text
    assert "One market order now would buy 40 at an average of" in resp.text
    split = manager.split_orders[0]
    assert split.sent == 1 and split.filled == 10
    while manager.split_orders:
        manager.step(1)
    summary = manager.split_history[-1]
    assert summary.startswith("Split buy done: bought 40 of 40 lots in 4 orders, at an average of")
    assert "One market order at the start would have bought 40 at an average of" in summary
    assert "The market's VWAP over the period was" in summary
    assert manager.notices[-1].text == summary
    assert not any(manager.book.get_order(oid) for oid in manager.manual_order_ids)  # no slice left resting
    mine = manager.trades_frame().query("yours")
    assert mine["size"].sum() == 40
    times = sorted(set(mine["time"]))
    assert len(times) == 4 and all(t >= split.start_time + i * 2.0 for i, t in enumerate(times))


def test_stopping_a_split_order(client, manager):
    client.post("/api/order/market", data={"size": 50, "side": "ask", "slices": 5, "interval": 100})
    split_id = manager.split_orders[0].split_id
    assert client.post("/api/order/split/cancel", data={"split_id": split_id}).status_code == 204
    assert not manager.split_orders
    assert manager.split_history[-1].startswith("Split sell canceled: sold 10 of 50 lots in 1 order,")
    assert client.post("/api/order/split/cancel", data={"split_id": split_id}).status_code == 404


@pytest.mark.parametrize("data", [{"slices": 0}, {"slices": 101}, {"slices": 3, "interval": 0}])
def test_invalid_split_settings_are_rejected(client, manager, data):
    resp = client.post("/api/order/market", data={"size": 30, "side": "bid", **data})
    assert resp.status_code == 422 and not manager.split_orders


# ---- Your orders and fills on the chart ----

def test_your_fills_are_marked_per_bar_as_maker_and_taker(client, manager):
    from lob.order import market_order

    inside = manager.book.best_bid + 1  # inside the spread, so the next sell market order meets it first
    client.post("/api/order/limit", data={"price": inside, "size": 8, "side": "bid"})  # at time 0.5
    manager._process(market_order(5, manager.sim_time, Side.ASK))  # someone sells 5 into it: you bought as maker
    best_ask = manager.book.best_ask
    client.post("/api/order/market", data={"size": 3, "side": "bid"})  # at time 1.0, you buy 3 as taker
    assert manager.book.best_ask == best_ask  # all 3 at the best ask

    marks = client.get("/api/candlesticks").json()["own_trades"]  # 2-unit bars: both fills in the first
    assert marks == [{"time": 0, "bought": 8, "buy_avg": pytest.approx((5 * inside + 3 * best_ask) / 8),
                      "sold": 0, "sell_avg": None}]
    client.post("/api/chart/interval", data={"interval": 1})
    marks = client.get("/api/candlesticks").json()["own_trades"]
    assert [(m["time"], m["bought"]) for m in marks] == [(0, 5), (1, 3)]
    assert manager.own_trade_seq == 2  # one per fill


def test_a_fill_against_your_own_order_marks_both_sides(client, manager):
    inside = manager.book.best_bid + 1
    client.post("/api/order/limit", data={"price": inside, "size": 4, "side": "bid"})
    client.post("/api/order/market", data={"size": 4, "side": "ask", "protection": 0})  # sells into your own bid
    (mark,) = manager.own_trade_marks()
    assert (mark["bought"], mark["buy_avg"], mark["sold"], mark["sell_avg"]) == (4, inside, 4, inside)


def test_resting_orders_are_listed_per_price_for_the_chart(client, manager):
    client.post("/api/order/limit", data={"price": 50, "size": 10, "side": "bid"})
    client.post("/api/order/limit", data={"price": 50, "size": 30, "side": "bid", "display": "5"})
    client.post("/api/order/limit", data={"price": 150, "size": 7, "side": "ask"})
    levels = [{"side": "ask", "price": 150, "size": 7, "hidden": 0},
              {"side": "bid", "price": 50, "size": 15, "hidden": 25}]
    ctx = get_view_context(manager)
    assert ctx["own_levels"] == levels
    panel = render_panels(ctx)["active_orders"]
    assert json.loads(re.search(r"data-levels='([^']*)'", panel).group(1)) == levels


def test_stream_sends_your_fills_on_connect_and_after_each_fill(client, manager):
    def next_own_trades(lines):
        event = None
        for line in lines:
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: ") and event == "own_trades":
                return json.loads(line[len("data: "):])

    client.post("/api/order/market", data={"size": 2, "side": "bid"})
    with client.stream("GET", "/events") as resp:
        lines = resp.iter_lines()
        assert [m["bought"] for m in next_own_trades(lines)] == [2]
        client.post("/api/order/market", data={"size": 3, "side": "ask"})
        (mark,) = next_own_trades(lines)
    assert (mark["bought"], mark["sold"]) == (2, 3)


# ---- Depth chart ----

def test_the_ladder_panel_draws_the_depth_chart(client, manager):
    html = render_panels(get_view_context(manager))["ladder"]
    assert 'id="depth-chart"' in html
    assert "cumulative lots within 16 ticks of the mid" in html  # the seeded book has 16 levels a side
    assert html[html.index('id="depth-chart"'):].count("<path") == 4  # two areas and two lines

    client.post("/api/order/limit", data={"price": 95, "size": 300, "side": "bid"})
    html = render_panels(get_view_context(manager))["ladder"]
    assert "Your buy orders at 95" in html  # a marker on the price axis


# ---- Trading from the order book: moving your orders, canceling them all ----

def test_moving_your_orders_keeps_their_sizes_peaks_and_order(client, manager):
    client.post("/api/order/limit", data={"price": 95, "size": 10, "side": "bid"})
    client.post("/api/order/limit", data={"price": 95, "size": 30, "side": "bid", "display": "5"})
    first, second = (manager.book.get_order(oid) for oid in sorted(manager.manual_order_ids))
    time_before, events_before = manager.sim_time, manager.event_count
    resp = client.post("/api/order/move", data={"side": "bid", "from_price": 95, "to_price": 93})
    assert resp.status_code == 200
    assert resp.json() == {"moved": 2, "filled": 0, "resting": 40,
                           "text": "Moved your 2 buy orders (40 lots) from 95 to 93, at the back of the queue there."}
    assert manager.sim_time == time_before + MANUAL_ACTION_DT  # one manual action
    assert manager.event_count == events_before + 4  # two cancellations and two new orders
    assert not [o for o in manager.book.orders_at(95, Side.BID) if o.order_id in manager.manual_order_ids]
    mine = [o for o in manager.book.orders_at(93, Side.BID) if o.order_id in manager.manual_order_ids]
    assert [(o.size, o.hidden, o.display_size) for o in mine] == [(10, 0, None), (5, 25, 5)]  # in their old order
    assert mine[-1] is manager.book.orders_at(93, Side.BID)[-1]  # at the back of the queue
    assert {o.order_id for o in mine}.isdisjoint({first.order_id, second.order_id})  # new orders


def test_moving_across_the_spread_trades_on_arrival(client, manager):
    best_bid = manager.book.best_bid
    depth = manager.book.depth_at_price(best_bid, Side.BID)
    client.post("/api/order/limit", data={"price": manager.book.best_ask + 5, "size": depth + 4, "side": "ask"})
    resp = client.post("/api/order/move", data={"side": "ask", "from_price": manager.book.best_ask + 5,
                                                 "to_price": best_bid})
    body = resp.json()
    assert (body["filled"], body["resting"]) == (depth, 4)
    assert body["text"].endswith(
        f"across the spread, so it sold {depth} on arrival at {best_bid}; 4 rest at {best_bid}.")
    assert manager.book.best_ask == best_bid  # the rest is now the best ask


def test_moving_nothing_is_rejected(client, manager):
    assert client.post("/api/order/move", data={"side": "bid", "from_price": 95, "to_price": 93}).status_code == 404
    client.post("/api/order/limit", data={"price": 95, "size": 10, "side": "bid"})
    assert client.post("/api/order/move", data={"side": "ask", "from_price": 95, "to_price": 93}).status_code == 404
    assert client.post("/api/order/move", data={"side": "bid", "from_price": 95, "to_price": 95}).status_code == 422
    assert client.post("/api/order/move", data={"side": "bid", "from_price": 95, "to_price": 0}).status_code == 422


def test_cancel_all_cancels_your_orders_and_only_yours(client, manager):
    client.post("/api/order/limit", data={"price": 95, "size": 10, "side": "bid"})
    client.post("/api/order/limit", data={"price": 96, "size": 10, "side": "bid"})
    client.post("/api/order/limit", data={"price": 105, "size": 10, "side": "ask"})
    seeded = manager.book.num_orders() - 3
    time_before = manager.sim_time
    assert client.post("/api/order/cancel_all", data={"side": "bid", "price": 96}).json() == {"canceled": 1}
    assert client.post("/api/order/cancel_all", data={"side": "bid"}).json() == {"canceled": 1}
    assert manager.sim_time == time_before + 2 * MANUAL_ACTION_DT
    assert client.post("/api/order/cancel_all").json() == {"canceled": 1}
    assert client.post("/api/order/cancel_all").json() == {"canceled": 0}
    assert not manager.manual_order_ids
    assert manager.book.num_orders() == seeded  # the models' orders are all still there
    panel = render_panels(get_view_context(manager))["active_orders"]
    assert "Cancel all" not in panel  # shown only while you have orders resting
