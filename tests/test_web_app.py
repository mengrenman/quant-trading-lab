"""Tests for FastAPI web application endpoints."""

import math

import pytest
from starlette.testclient import TestClient
from app.main import app
from app.manager import manager, MANUAL_ACTION_DT
from lob.order import Side


@pytest.fixture
def client():
    # Reset manager state (including configuration) before each test
    manager.set_engine("price_time")
    manager.set_model("smith_zi")
    manager.set_speed(50)
    manager.bar_interval = 2
    manager.reset(mid=100)
    return TestClient(app)


def candle_volume(candles):
    return sum(c["volume"] for c in candles)


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Limit Order Book" in response.text
    assert "Depth of Market" in response.text
    assert 'hx-ext="sse"' in response.text
    assert 'sse-connect="/events"' in response.text


def test_submit_limit_order(client):
    resp = client.post("/api/order/limit", data={"price": 50, "size": 10, "side": "bid"})
    assert resp.status_code == 204
    assert manager.event_count == 1
    assert manager.book.depth_at_price(50, Side.BID) == 10
    assert len(manager.manual_order_ids) == 1


def test_submit_market_order(client):
    resp = client.post("/api/order/market", data={"size": 5, "side": "bid"})
    assert resp.status_code == 204
    assert len(manager.recent_trades) > 0
    assert manager.recent_trades[0].side == "bid"


@pytest.mark.parametrize("path, data", [
    ("/api/order/limit", {"price": 100, "size": 0, "side": "bid"}),
    ("/api/order/limit", {"price": 0, "size": 5, "side": "bid"}),
    ("/api/order/limit", {"price": 100, "size": 5, "side": "buy"}),
    ("/api/order/market", {"size": -1, "side": "ask"}),
    ("/api/sim/config", {"model": "random_walk"}),
    ("/api/sim/control", {"action": "rewind"}),
    ("/api/sim/control", {"action": "step", "n": 10**9}),
    ("/api/engine", {"engine": "fastest"}),
])
def test_invalid_input_is_rejected(client, path, data):
    resp = client.post(path, data=data)
    assert resp.status_code == 422
    assert manager.event_count == 0


def test_cancel_order(client):
    # Submit resting limit order far out
    client.post("/api/order/limit", data={"price": 50, "size": 10, "side": "bid"})
    assert len(manager.manual_order_ids) > 0
    oid = list(manager.manual_order_ids)[0]

    resp = client.post("/api/order/cancel", data={"order_id": oid})
    assert resp.status_code == 204
    assert oid not in manager.manual_order_ids
    assert manager.book.get_order(oid) is None


def test_sim_control_step_and_reset(client):
    init_events = manager.event_count
    resp = client.post("/api/sim/control", data={"action": "step", "n": 5})
    assert resp.status_code == 204
    assert manager.event_count > init_events

    resp = client.post("/api/sim/control", data={"action": "reset"})
    assert resp.status_code == 200
    assert "Start Sim" in resp.text
    assert manager.event_count == 0
    assert manager.sim_time == 0.0


def test_sim_config(client):
    resp = client.post("/api/sim/config", data={"model": "mike_farmer", "speed": 15})
    assert resp.status_code == 204
    assert manager.model_type == "mike_farmer"
    assert manager.speed == 15


def test_engine_toggle_round_trip(client):
    resp = client.post("/api/engine")
    assert resp.status_code == 200
    assert manager.engine_type == "pro_rata"
    assert "Pro-Rata" in resp.text and 'id="engine-toggle"' in resp.text

    resp = client.post("/api/engine")
    assert manager.engine_type == "price_time"
    assert "Price-Time" in resp.text


def test_engine_can_be_set_explicitly(client):
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


def test_candles_are_built_from_trades(client):
    client.post("/api/order/market", data={"size": 10, "side": "bid"})
    client.post("/api/order/market", data={"size": 4, "side": "ask"})
    candles = client.get("/api/candlesticks").json()["candles"]
    assert candle_volume(candles) == manager.traded_volume == 14
    assert sum(c["buy_volume"] for c in candles) == 10
    assert sum(c["sell_volume"] for c in candles) == 4
    traded_prices = {t.price for t in manager.recent_trades}
    assert {c["high"] for c in candles if c["volume"]} <= traded_prices


def test_interval_change_reaggregates_history(client):
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


def test_candles_follow_the_simulation_clock(client):
    client.post("/api/sim/control", data={"action": "step", "n": 500})
    last = client.get("/api/candlesticks").json()["candles"][-1]
    n = manager.bar_interval
    assert last["time"] == math.floor(manager.sim_time / n) * n


def test_sim_time_never_runs_backwards(client):
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


def test_chart_events_stream(client):
    with client.stream("GET", "/events/chart") as resp:
        assert resp.status_code == 200
        first_event = None
        for line in resp.iter_lines():
            if line.startswith("event:"):
                first_event = line
                break
        assert first_event == "event: candles"


def test_speed_up_to_1000(client):
    resp = client.post("/api/sim/config", data={"speed": 1000})
    assert resp.status_code == 204
    assert manager.speed == 1000


def test_play_and_pause_simulation(client):
    import asyncio

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


def test_stochastic_mid_price_evolution(client):
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
