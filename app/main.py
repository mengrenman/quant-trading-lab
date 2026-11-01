"""FastAPI web application for interactive Limit Order Book simulation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time
from typing import Any, Literal, Optional

from fastapi import FastAPI, Form, Request, Response, status
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles

from lob.order import Side
from app.manager import manager

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="Limit Order Book Simulator")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

LOG_FILE = Path("/tmp/lob_app.log")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    client = request.client.port if request.client else "unknown"
    with open(LOG_FILE, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] REQ {request.method} {request.url.path} from port {client}\n")
    response = await call_next(request)
    with open(LOG_FILE, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] RES {request.method} {request.url.path} -> {response.status_code}\n")
    return response


def get_view_context() -> dict[str, Any]:
    """Calculate and format full view context from the current book state.

    Templates are rendered on worker threads while the simulation keeps
    mutating the book, so the context holds only copied values, never the
    live book.
    """
    book = manager.book
    bids = book.depth_profile(Side.BID, n_levels=20)
    asks = book.depth_profile(Side.ASK, n_levels=20)

    # Fixed window height: exactly 10 levels for asks and 10 levels for bids (20 levels total)
    # This prevents the window height from jumping or shifting during simulation
    FIXED_DOM_LEVELS = 10
    bids_subset = bids[:FIXED_DOM_LEVELS]
    asks_subset = asks[:FIXED_DOM_LEVELS]

    # Calculate accumulated depth and relative percentages for bars
    max_bid_d = max([d for _, d in bids_subset], default=1)
    max_ask_d = max([d for _, d in asks_subset], default=1)
    max_d = max(max_bid_d, max_ask_d, 1)

    bids_display = []
    cum = 0
    for p, d in bids_subset:
        cum += d
        bids_display.append({
            "price": p,
            "depth": d,
            "cum_depth": cum,
            "pct": min(100, int((d / max_d) * 100)),
            "empty": False,
        })
    # Pad bids to FIXED_DOM_LEVELS if book is thin so DOM height never shifts
    while len(bids_display) < FIXED_DOM_LEVELS:
        bids_display.append({
            "price": None,
            "depth": None,
            "cum_depth": None,
            "pct": 0,
            "empty": True,
        })

    # For asks: display reversed so highest price is at top, down to best ask at bottom
    temp_asks = []
    cum = 0
    for p, d in asks_subset:
        cum += d
        temp_asks.append({
            "price": p,
            "depth": d,
            "cum_depth": cum,
            "pct": min(100, int((d / max_d) * 100)),
            "empty": False,
        })
    # Pad asks to FIXED_DOM_LEVELS (outer/higher price edge)
    while len(temp_asks) < FIXED_DOM_LEVELS:
        temp_asks.append({
            "price": None,
            "depth": None,
            "cum_depth": None,
            "pct": 0,
            "empty": True,
        })
    asks_display = list(reversed(temp_asks))

    # Top-of-book depth
    best_bid = book.best_bid
    best_ask = book.best_ask
    best_bid_depth = book.depth_at_price(best_bid, Side.BID) if best_bid is not None else 0
    best_ask_depth = book.depth_at_price(best_ask, Side.ASK) if best_ask is not None else 0

    # Imbalance top 5
    bid_vol_5 = sum(d for _, d in bids[:5])
    ask_vol_5 = sum(d for _, d in asks[:5])
    tot_5 = bid_vol_5 + ask_vol_5
    imbalance = (bid_vol_5 - ask_vol_5) / tot_5 if tot_5 > 0 else 0.0

    bid_ratio = int((bid_vol_5 / tot_5) * 100) if tot_5 > 0 else 50
    ask_ratio = 100 - bid_ratio

    # Active manual orders
    active_orders = []
    for oid in sorted(manager.manual_order_ids):
        o = book.get_order(oid)
        if o is not None:
            active_orders.append({
                "order_id": o.order_id,
                "side": o.side.value,
                "price": o.price,
                "size": o.size,
                "timestamp": o.timestamp,
            })

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": book.spread,
        "mid_price": book.mid_price,
        "best_bid_depth": best_bid_depth,
        "best_ask_depth": best_ask_depth,
        "bids_display": bids_display,
        "asks_display": asks_display,
        "imbalance": imbalance,
        "bid_ratio": bid_ratio,
        "ask_ratio": ask_ratio,
        "trades": list(manager.recent_trades),
        "active_orders": active_orders,
        "is_running": manager.is_running,
        "speed": manager.speed,
        "model_type": manager.model_type,
        "engine_type": manager.engine_type,
        "event_count": manager.event_count,
        "sim_time": manager.sim_time,
        "bar_interval": manager.bar_interval,
    }


def format_sse(event: str, data: str) -> str:
    cleaned = data.replace("\r", "").replace("\n", " ")
    return f"event: {event}\ndata: {cleaned}\n\n"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    ctx = get_view_context()
    return templates.TemplateResponse(request=request, name="index.html", context=ctx)


@app.get("/events")
async def events_stream(request: Request):
    """Server-Sent Events endpoint pushing live updates to HTMX."""
    queue = manager.subscribe()

    async def event_generator():
        try:
            # Send initial full state immediately upon connection
            ctx = get_view_context()

            def _render_sim_badge(c: dict) -> str:
                if c["is_running"]:
                    status_pill = (
                        '<button type="button" onclick="togglePlayback()" id="sim-status-toggle-pill" '
                        'class="cursor-pointer transition hover:opacity-80 flex items-center gap-1.5 text-[10px] font-semibold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20" '
                        'title="Click to pause simulation">'
                        '<span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span> RUNNING</button>'
                    )
                else:
                    status_pill = (
                        '<button type="button" onclick="togglePlayback()" id="sim-status-toggle-pill" '
                        'class="cursor-pointer transition hover:opacity-80 flex items-center gap-1.5 text-[10px] font-semibold text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20" '
                        'title="Click to start simulation">'
                        '<span class="w-1.5 h-1.5 rounded-full bg-amber-400"></span> PAUSED</button>'
                    )
                return (
                    f'<div id="sim-stats-badge" class="flex items-center gap-3 text-slate-400 font-mono text-[11px] bg-slate-900/80 px-3 py-1.5 rounded-lg border border-slate-800">'
                    f'{status_pill}'
                    f'<div>Events: <span class="text-slate-200 font-bold">{c["event_count"]}</span></div>'
                    f'<div>Sim Time: <span class="text-slate-200 font-bold">{c["sim_time"]:.2f}s</span></div>'
                    f'</div>'
                )

            def _render_all_templates(c: dict) -> dict[str, str]:
                """Run template rendering on worker threadpool to free the asyncio event loop."""
                return {
                    "ladder": templates.get_template("components/ladder.html").render(c),
                    "metrics": templates.get_template("components/metrics.html").render(c),
                    "trades": templates.get_template("components/trades.html").render(c),
                    "active_orders": templates.get_template("components/active_orders.html").render(c),
                    "sim_status": _render_sim_badge(c),
                }

            rendered = await asyncio.to_thread(_render_all_templates, ctx)
            yield format_sse("ladder", rendered["ladder"])
            yield format_sse("metrics", rendered["metrics"])
            yield format_sse("trades", rendered["trades"])
            yield format_sse("active_orders", rendered["active_orders"])
            yield format_sse("sim_status", rendered["sim_status"])

            last_dom_push = 0.0
            last_order_count = len(ctx["active_orders"])

            # Stream updates whenever manager broadcasts
            while True:
                if await request.is_disconnected():
                    break

                try:
                    await asyncio.wait_for(queue.get(), timeout=15.0)
                    while not queue.empty():
                        try:
                            queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                except asyncio.TimeoutError:
                    # Heartbeat comment to keep HTTP connection alive
                    yield ": ping\n\n"
                    continue

                now = time.monotonic()
                # Optimized DOM push rate: 5 FPS (200ms) while running.
                # Human eyes process at ~5-10Hz; this cuts browser reflow load by 60% and eliminates stuttering.
                if manager.is_running and (now - last_dom_push < 0.2):
                    continue

                last_dom_push = now
                ctx = get_view_context()

                # Offload HTML template generation to parallel threadpool
                rendered = await asyncio.to_thread(_render_all_templates, ctx)

                yield format_sse("ladder", rendered["ladder"])
                yield format_sse("metrics", rendered["metrics"])
                yield format_sse("trades", rendered["trades"])
                yield format_sse("sim_status", rendered["sim_status"])

                # Only push active_orders if the active order list actually changed
                curr_order_count = len(ctx["active_orders"])
                if curr_order_count != last_order_count or not manager.is_running:
                    yield format_sse("active_orders", rendered["active_orders"])
                    last_order_count = curr_order_count

        finally:
            manager.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/events/chart")
async def chart_events_stream(request: Request):
    """Dedicated Server-Sent Events stream for real-time Candlestick Chart updates."""
    queue = manager.subscribe()

    async def event_generator():
        # Each connection tracks the chart version it has seen, so a reset or an
        # interval change reaches every open chart, not just the first one.
        version = manager.chart_version
        last_time: Optional[int] = None
        try:
            # Send the current bar immediately upon connection; the client loads
            # the history from /api/candlesticks.
            bars = manager.get_candles(limit=1)
            if bars:
                last_time = bars[-1]["time"]
                yield format_sse("candles", json.dumps(bars))

            while True:
                if await request.is_disconnected():
                    break

                try:
                    await asyncio.wait_for(queue.get(), timeout=15.0)
                    while not queue.empty():
                        try:
                            queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue

                if manager.chart_version != version:
                    version = manager.chart_version
                    last_time = None
                    yield format_sse("chart_reset", "true")
                    continue

                # Every bar since the last one sent (which may have changed since),
                # so fast simulations do not skip bars between pushes.
                if last_time is None:
                    bars = manager.get_candles(limit=1)
                else:
                    bars = manager.get_candles(since=last_time)
                if bars:
                    last_time = bars[-1]["time"]
                    yield format_sse("candles", json.dumps(bars))

        finally:
            manager.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/candlesticks")
async def get_candlesticks():
    """Return the OHLCV bars for the current interval, built from executed trades."""
    candles = manager.get_candles()
    return {
        "interval": manager.bar_interval,
        "candles": candles,
        "latest": candles[-1] if candles else None,
    }


@app.post("/api/chart/interval")
async def post_chart_interval(interval: int = Form(..., ge=1, le=3600)):
    async with manager._lock:
        manager.set_bar_interval(interval)
    await manager.broadcast()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/order/limit")
async def post_limit_order(
    price: int = Form(..., ge=1),
    size: int = Form(..., ge=1),
    side: Literal["bid", "ask"] = Form(...),
):
    async with manager._lock:
        manager.submit_limit_order(price=price, size=size, side=Side(side))
    await manager.broadcast()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/order/market")
async def post_market_order(
    size: int = Form(..., ge=1),
    side: Literal["bid", "ask"] = Form(...),
):
    async with manager._lock:
        manager.submit_market_order(size=size, side=Side(side))
    await manager.broadcast()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/order/cancel")
async def post_cancel_order(
    order_id: int = Form(...),
):
    async with manager._lock:
        manager.cancel_order_by_id(order_id)
    await manager.broadcast()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def render_playback_btn(is_running: bool) -> str:
    if is_running:
        return (
            '<button id="sim-play-btn" type="button" onclick="togglePlayback()" '
            'class="flex items-center gap-2 px-5 py-2.5 rounded-lg font-bold text-xs uppercase tracking-wider transition cursor-pointer shadow-lg bg-amber-500 hover:bg-amber-600 text-slate-950 shadow-amber-500/20">'
            '<span id="sim-play-icon"><svg class="w-4 h-4 fill-current" viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg></span>'
            '<span id="sim-play-text">Pause Sim</span>'
            '</button>'
        )
    else:
        return (
            '<button id="sim-play-btn" type="button" onclick="togglePlayback()" '
            'class="flex items-center gap-2 px-5 py-2.5 rounded-lg font-bold text-xs uppercase tracking-wider transition cursor-pointer shadow-lg bg-emerald-500 hover:bg-emerald-600 text-slate-950 shadow-emerald-500/20">'
            '<span id="sim-play-icon"><svg class="w-4 h-4 fill-current" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg></span>'
            '<span id="sim-play-text">Start Sim</span>'
            '</button>'
        )


@app.post("/api/sim/control")
async def post_sim_control(
    action: Literal["play", "pause", "toggle", "step", "reset"] = Form(...),
    n: int = Form(1, ge=1, le=10_000),
):
    if action == "play":
        await manager.start_sim_loop()
        return HTMLResponse(render_playback_btn(True))
    elif action == "pause":
        await manager.stop_sim_loop()
        return HTMLResponse(render_playback_btn(False))
    elif action == "toggle":
        if manager.is_running:
            await manager.stop_sim_loop()
            return HTMLResponse(render_playback_btn(False))
        else:
            await manager.start_sim_loop()
            return HTMLResponse(render_playback_btn(True))
    elif action == "step":
        async with manager._lock:
            manager.step(n or 1)
        await manager.broadcast()
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    elif action == "reset":
        if manager.is_running:
            await manager.stop_sim_loop()
        async with manager._lock:
            manager.reset(mid=100)
        await manager.broadcast()
        return HTMLResponse(render_playback_btn(False))

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/sim/config")
async def post_sim_config(
    model: Optional[Literal["smith_zi", "mike_farmer", "parlour", "manual"]] = Form(None),
    speed: Optional[int] = Form(None, ge=1, le=2000),
):
    if model:
        async with manager._lock:
            manager.set_model(model)
    if speed is not None:
        manager.set_speed(speed)
    await manager.broadcast()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/engine", response_class=HTMLResponse)
async def post_engine(
    request: Request,
    engine: Optional[Literal["price_time", "pro_rata"]] = Form(None),
):
    """Set the matching engine, or toggle it when none is given.

    Returns the re-rendered toggle button so the page always shows the
    engine actually in use.
    """
    async with manager._lock:
        if engine is None:
            manager.toggle_engine()
        else:
            manager.set_engine(engine)
    await manager.broadcast()
    return templates.TemplateResponse(request=request, name="components/engine_toggle.html",
                                      context={"engine_type": manager.engine_type})


@app.get("/api/snapshot")
async def get_snapshot():
    return manager.book.snapshot()
