"""Tests for FastAPI web application endpoints."""

import pytest
from starlette.testclient import TestClient
from app.main import app
from app.manager import manager


@pytest.fixture
def client():
    # Reset manager state before each test
    manager.reset(mid=100)
    return TestClient(app)


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Limit Order Book" in response.text
    assert "Depth of Market" in response.text
    assert 'hx-ext="sse"' in response.text
    assert 'sse-connect="/events"' in response.text


def test_submit_limit_order(client):
    init_bid = manager.book.best_bid
    resp = client.post("/api/order/limit", data={"price": 105, "size": 10, "side": "bid"})
    assert resp.status_code == 204
    # Bid at 105 crossed ask or updated best bid
    assert manager.event_count > 0


def test_submit_market_order(client):
    resp = client.post("/api/order/market", data={"size": 5, "side": "bid"})
    assert resp.status_code == 204
    assert len(manager.recent_trades) > 0


def test_cancel_order(client):
    # Submit resting limit order far out
    client.post("/api/order/limit", data={"price": 50, "size": 10, "side": "bid"})
    assert len(manager.manual_order_ids) > 0
    oid = list(manager.manual_order_ids)[0]

    resp = client.post("/api/order/cancel", data={"order_id": oid})
    assert resp.status_code == 204
    assert oid not in manager.manual_order_ids


def test_sim_control_step_and_reset(client):
    init_events = manager.event_count
    resp = client.post("/api/sim/control", data={"action": "step", "n": 5})
    assert resp.status_code == 204
    assert manager.event_count >= init_events + 5

    resp = client.post("/api/sim/control", data={"action": "reset"})
    assert resp.status_code == 200
    assert "Start Sim" in resp.text
    assert manager.event_count == 0


def test_sim_config(client):
    resp = client.post("/api/sim/config", data={"model": "mike_farmer", "speed": 15})
    assert resp.status_code == 204
    assert manager.model_type == "mike_farmer"
    assert manager.speed == 15


def test_toggle_engine(client):
    resp = client.post("/api/engine", data={"engine": "pro_rata"})
    assert resp.status_code == 204
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
    for key in ("time", "open", "high", "low", "close", "volume"):
        assert key in first_candle


def test_chart_interval_change(client):
    for interval in [1, 2, 5, 10, 15, 30, 60, 180, 300]:
        resp = client.post("/api/chart/interval", data={"interval": interval})
        assert resp.status_code == 204
        assert manager.bar_interval == interval
        resp_candlesticks = client.get("/api/candlesticks")
        assert resp_candlesticks.status_code == 200
        assert resp_candlesticks.json()["interval"] == interval
    # Reset back to default 2s
    client.post("/api/chart/interval", data={"interval": 2})


def test_candle_updates_on_orders_and_steps(client):
    prev_close = manager.candles[-1].close
    # Submit crossing buy market order
    client.post("/api/order/market", data={"size": 10, "side": "bid"})
    latest = manager.get_latest_candle()
    assert latest is not None
    assert latest["volume"] > 0


def test_chart_events_stream(client):
    with client.stream("GET", "/events/chart") as resp:
        assert resp.status_code == 200
        first_event = None
        for line in resp.iter_lines():
            if line.startswith("event:"):
                first_event = line
                break
        assert first_event == "event: candle"


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
    manager.set_model("smith_zi")
    manager.reset(mid=100)
    initial_mid = manager.book.mid_price
    assert initial_mid == 100.0

    # Run steps and collect mid-price values
    mids = set()
    for _ in range(300):
        client.post("/api/sim/control", data={"action": "step"})
        if manager.book.mid_price is not None:
            mids.add(manager.book.mid_price)

    # Over 300 steps with inside-spread limit orders and market orders,
    # mid price must dynamically evolve across multiple distinct price levels
    assert len(mids) > 1




