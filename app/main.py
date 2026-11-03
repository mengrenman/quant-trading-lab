"""FastAPI web application for interactive Limit Order Book simulation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time
from typing import Annotated, Any, Literal, Optional

from fastapi import Depends, FastAPI, Form, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles

from lob.order import Fill, Order, Side
from app.facts import WARMUP_STEPS, summarize as summarize_facts, summarize_order_flow
from app.manager import DEFAULT_PROTECTION_TICKS, MAX_PROTECTION_TICKS, LOBManager
from app.model_specs import MODEL_SPECS
from app.sessions import SessionMiddleware, SessionRegistry

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def format_price(value: Optional[float]) -> str:
    """A price in ticks for display: whole numbers without decimals (the mid can be a half tick)."""
    if value is None:
        return "\u2014"
    return f"{value:.0f}" if float(value).is_integer() else f"{value:.2f}".rstrip("0")


templates.env.filters["price"] = format_price


def asset(path: str) -> str:
    """URL of a static file with its modification time appended, so browsers fetch it again after a change."""
    try:
        version = int((BASE_DIR / "static" / path).stat().st_mtime)
    except OSError:
        version = 0
    return f"/static/{path}?v={version}"


templates.env.globals["asset"] = asset
templates.env.globals["protection_default"] = DEFAULT_PROTECTION_TICKS
templates.env.globals["protection_max"] = MAX_PROTECTION_TICKS

# One simulation per browser session (see app/sessions.py).
registry = SessionRegistry()

app = FastAPI(title="Limit Order Book Simulator")
app.add_middleware(SessionMiddleware, registry=registry)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def get_manager(request: Request) -> LOBManager:
    """The simulation belonging to the request's session."""
    return registry.get(request.state.session_id)


SessionManager = Annotated[LOBManager, Depends(get_manager)]


def get_view_context(manager: LOBManager) -> dict[str, Any]:
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

    # Active manual orders
    active_orders = []
    for oid in sorted(manager.manual_order_ids):
        o = book.get_order(oid)
        if o is not None:
            active_orders.append({
                "order_id": o.order_id,
                "side": o.side.value,
                "price": o.price,
                "size": o.size,  # showing (an iceberg's peak)
                "hidden": o.hidden,  # an iceberg's reserve
                "timestamp": o.timestamp,
            })

    # Lots of the user's own resting orders at each (side, price), marked in the ladder:
    # showing, and hidden in icebergs
    own_size: dict[tuple[str, int], int] = {}
    own_hidden: dict[tuple[str, int], int] = {}
    for o in active_orders:
        key = (o["side"], o["price"])
        own_size[key] = own_size.get(key, 0) + o["size"]
        own_hidden[key] = own_hidden.get(key, 0) + o["hidden"]

    bids_display = []
    cum = 0
    for p, d in bids_subset:
        cum += d
        bids_display.append({
            "price": p,
            "depth": d,
            "cum_depth": cum,
            "pct": min(100, int((d / max_d) * 100)),
            "own": own_size.get(("bid", p), 0),
            "own_hidden": own_hidden.get(("bid", p), 0),
            "empty": False,
        })
    # Pad bids to FIXED_DOM_LEVELS if book is thin so DOM height never shifts
    while len(bids_display) < FIXED_DOM_LEVELS:
        bids_display.append({
            "price": None,
            "depth": None,
            "cum_depth": None,
            "pct": 0,
            "own": 0,
            "own_hidden": 0,
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
            "own": own_size.get(("ask", p), 0),
            "own_hidden": own_hidden.get(("ask", p), 0),
            "empty": False,
        })
    # Pad asks to FIXED_DOM_LEVELS (outer/higher price edge)
    while len(temp_asks) < FIXED_DOM_LEVELS:
        temp_asks.append({
            "price": None,
            "depth": None,
            "cum_depth": None,
            "pct": 0,
            "own": 0,
            "own_hidden": 0,
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
        "instance_id": manager.instance_id,
        **model_params_context(manager),
    }


def model_params_context(manager: LOBManager, *, values: Optional[dict] = None,
                         errors: Optional[dict] = None, message: Optional[str] = None,
                         open_panel: bool = False) -> dict[str, Any]:
    """Context for the model parameter panel (components/model_params.html)."""
    return {
        "model_specs": MODEL_SPECS,
        "model_spec": MODEL_SPECS[manager.model_type],
        "model_values": manager.model_params[manager.model_type],
        "seed": manager.seed,
        "params_signature": params_signature(manager),
        "param_values": values or {},
        "param_errors": errors or {},
        "params_message": message,
        "params_open": open_panel,
    }


def params_signature(manager: LOBManager) -> str:
    """Identifies the applied model, parameters and seed, so a tab can tell whether its panel is current."""
    values = manager.model_params[manager.model_type]
    return "|".join([manager.model_type, str(manager.seed), *(f"{k}={values[k]}" for k in sorted(values))])


def controls_state(manager: LOBManager) -> dict[str, Any]:
    """The page controls that tabs of one session keep in sync (sent as the ``controls`` event)."""
    return {
        "model": manager.model_type,
        "engine": manager.engine_type,
        "speed": manager.speed,
        "params_signature": params_signature(manager),
        "params_html": templates.get_template("components/model_params.html").render(model_params_context(manager)),
        "engine_html": templates.get_template("components/engine_toggle.html").render(engine_type=manager.engine_type),
    }


def render_model_params(request: Request, manager: LOBManager, status_code: int = 200,
                        **kwargs: Any) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="components/model_params.html",
                                      context=model_params_context(manager, **kwargs), status_code=status_code)


def format_sse(event: str, data: str) -> str:
    cleaned = data.replace("\r", "").replace("\n", " ")
    return f"event: {event}\ndata: {cleaned}\n\n"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, manager: SessionManager):
    ctx = get_view_context(manager)
    data = manager.facts.data()
    ctx["facts"] = summarize_facts(data, manager.model_type)
    ctx["ofi"] = summarize_order_flow(data)
    ctx["collecting"] = manager.collecting
    return templates.TemplateResponse(request=request, name="index.html", context=ctx)


FACTS_PUSH_INTERVAL = 1.0  # seconds between stylized-facts pushes
COLLECTING_PUSH_INTERVAL = 0.3  # the same while a fast collection shows its progress


def facts_interval(manager: LOBManager) -> float:
    return COLLECTING_PUSH_INTERVAL if manager.collecting is not None else FACTS_PUSH_INTERVAL


def facts_key(manager: LOBManager) -> tuple:
    """Identifies what the stylized-facts panel shows, so it is recomputed only when that changes."""
    facts = manager.facts
    return (facts.generation, facts.version, min(facts.steps, WARMUP_STEPS) // 250,
            manager.model_type, manager.collecting)


async def facts_html(manager: LOBManager) -> dict[str, str]:
    """The stylized-facts and order-flow panels, summarized on a worker thread and cached until the sample changes."""
    key = facts_key(manager)
    if manager.facts_cache is not None and manager.facts_cache[0] == key:
        return manager.facts_cache[1]
    # Copied on the event loop, between simulation steps; summarized and rendered off it.
    data, model_type, collecting = manager.facts.data(), manager.model_type, manager.collecting

    def render() -> dict[str, str]:
        return {
            "facts": templates.get_template("components/facts.html").render(
                facts=summarize_facts(data, model_type), collecting=collecting, model_type=model_type),
            "ofi": templates.get_template("components/ofi.html").render(ofi=summarize_order_flow(data)),
        }

    html = await asyncio.to_thread(render)
    manager.facts_cache = (key, html)
    return html


PANELS = ("ladder", "metrics", "trades", "active_orders", "sim_status")
PANEL_TEMPLATES = {
    "ladder": "components/ladder.html",
    "metrics": "components/metrics.html",
    "trades": "components/trades.html",
    "active_orders": "components/active_orders.html",
    "sim_status": "components/sim_badge.html",
}
PANEL_PUSH_INTERVAL = 0.2  # seconds between panel pushes while the simulation runs


def render_panels(ctx: dict[str, Any]) -> dict[str, str]:
    """Render every live panel; called on a worker thread to keep the event loop free."""
    return {name: templates.get_template(PANEL_TEMPLATES[name]).render(ctx) for name in PANELS}


@app.get("/events")
async def events_stream(request: Request, manager: SessionManager):
    """Server-Sent Events: one stream per tab carrying both the page panels and the chart.

    Browsers allow about six connections per host over HTTP/1.1, so a single
    stream per tab leaves room for more tabs and for the page's own requests.
    Event types: ``hello`` (the simulation's public instance ID, so a page
    can tell it is talking to a different simulation after a restart), one
    per panel (rendered HTML), ``candles`` (every bar since the last push),
    ``chart_reset`` (reload the chart) and ``controls`` (model, parameters,
    seed, engine and speed, so every tab of a session stays current).
    """
    queue = manager.subscribe()
    session_id = request.state.session_id

    async def event_generator():
        version = manager.chart_version
        config_version: Optional[int] = None
        last_time: Optional[int] = None
        last_orders: Optional[tuple] = None
        last_panel_push = 0.0
        facts_pushed: Optional[tuple] = None
        last_facts_push = 0.0
        first = True
        try:
            yield format_sse("hello", json.dumps({"instance": manager.instance_id}))
            while True:
                if not first:
                    if await request.is_disconnected():
                        break
                    # Wait for news; sooner if a facts update is waiting out its interval.
                    timeout = 15.0
                    if facts_key(manager) != facts_pushed:
                        timeout = max(0.05, facts_interval(manager) - (time.monotonic() - last_facts_push))
                    try:
                        await asyncio.wait_for(queue.get(), timeout=timeout)
                        while not queue.empty():
                            queue.get_nowait()
                    except asyncio.TimeoutError:
                        if facts_key(manager) == facts_pushed:
                            registry.touch(session_id)
                            yield ": ping\n\n"  # keep the connection alive
                            continue
                registry.touch(session_id)

                # Controls: on connect, and whenever another request changed them.
                if manager.config_version != config_version:
                    config_version = manager.config_version
                    yield format_sse("controls", json.dumps(controls_state(manager)))

                # Stylized facts: at most once a second, and only when the sample changed.
                now = time.monotonic()
                key = facts_key(manager)
                if key != facts_pushed and (first or now - last_facts_push >= facts_interval(manager)):
                    facts_pushed, last_facts_push = key, now
                    panels = await facts_html(manager)
                    yield format_sse("facts", panels["facts"])
                    yield format_sse("ofi", panels["ofi"])

                # During a fast collection only its progress is shown; the chart and panels
                # catch up when it ends (rendering them meanwhile would slow it down).
                if manager.collecting is not None and not first:
                    continue

                # Chart: every bar since the last push (which may have changed since),
                # so fast simulations do not skip bars between pushes.
                if manager.chart_version != version:
                    version = manager.chart_version
                    last_time = None
                    yield format_sse("chart_reset", "true")
                else:
                    bars = manager.get_candles(limit=1) if last_time is None else manager.get_candles(since=last_time)
                    if bars:
                        last_time = bars[-1]["time"]
                        yield format_sse("candles", json.dumps(bars))

                # Panels: at most five pushes a second while running.
                now = time.monotonic()
                if not first and manager.is_running and now - last_panel_push < PANEL_PUSH_INTERVAL:
                    continue
                last_panel_push = now
                ctx = get_view_context(manager)
                rendered = await asyncio.to_thread(render_panels, ctx)
                orders = tuple((o["order_id"], o["size"]) for o in ctx["active_orders"])
                for name in PANELS:
                    # The order table is replaced only when an order appeared, changed or left.
                    if name == "active_orders" and orders == last_orders:
                        continue
                    yield format_sse(name, rendered[name])
                last_orders = orders
                first = False
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
async def get_candlesticks(manager: SessionManager):
    """Return the OHLCV bars for the current interval, built from executed trades."""
    candles = manager.get_candles()
    return {
        "interval": manager.bar_interval,
        "candles": candles,
        "latest": candles[-1] if candles else None,
    }


@app.post("/api/chart/interval")
async def post_chart_interval(manager: SessionManager, interval: int = Form(..., ge=1, le=3600)):
    async with manager._lock:
        manager.set_bar_interval(interval)
    await manager.broadcast()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def order_outcome(kind: str, order: Optional[Order], size: int, fills: list[Fill], rests: bool,
                  side: Optional[str] = None, vwap_before: Optional[float] = None) -> dict[str, Any]:
    """What happened to a manual order, for components/order_result.html (``order`` is None if none was placed).

    ``vwap_before`` is the session VWAP just before the order, the usual benchmark for its fills.
    """
    filled = sum(f.size for f in fills)
    average = sum(f.price * f.size for f in fills) / filled if filled else None
    prices = [f.price for f in fills]
    low, high = (min(prices), max(prices)) if prices else (None, None)
    return {
        "kind": kind,
        "side": order.side.value if order is not None else side,
        "size": size,
        "price": order.price if order is not None else None,
        "order_id": order.order_id if order is not None else None,
        "filled": filled,
        "rests": size - filled if rests else 0,
        "display_size": order.display_size if order is not None else None,
        "price_text": "" if low is None else (f"{low}" if low == high else f"{low}\u2013{high}"),
        "avg_text": f"{average:.2f}" if low != high else "",
        "vwap_before": vwap_before,
        "vs_vwap": (average - vwap_before) / vwap_before * 100 if average is not None and vwap_before else None,
    }


def parse_display_size(raw: str) -> Optional[int]:
    """The form's optional iceberg peak: empty for an ordinary order, else a whole number of lots >= 1."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "display must be a whole number of lots >= 1")
    return value


@app.post("/api/order/limit", response_class=HTMLResponse)
async def post_limit_order(
    request: Request,
    manager: SessionManager,
    price: int = Form(..., ge=1),
    size: int = Form(..., ge=1),
    side: Literal["bid", "ask"] = Form(...),
    display: str = Form(""),
):
    display_size = parse_display_size(display)
    async with manager._lock:
        vwap_before = manager.session_vwap()
        order, fills = manager.submit_limit_order(price=price, size=size, side=Side(side),
                                                  display_size=display_size)
        rests = manager.book.get_order(order.order_id) is not None
    await manager.broadcast()
    return templates.TemplateResponse(request=request, name="components/order_result.html",
                                      context=order_outcome("limit", order, size, fills, rests,
                                                            vwap_before=vwap_before))


@app.post("/api/order/market", response_class=HTMLResponse)
async def post_market_order(
    request: Request,
    manager: SessionManager,
    size: int = Form(..., ge=1),
    side: Literal["bid", "ask"] = Form(...),
    protection: int = Form(DEFAULT_PROTECTION_TICKS, ge=0, le=MAX_PROTECTION_TICKS),
):
    """A market order that walks at most ``protection`` ticks; the rest rests as a limit order."""
    async with manager._lock:
        vwap_before = manager.session_vwap()
        order, fills = manager.submit_market_order(size=size, side=Side(side), protection=protection)
        rests = order is not None and manager.book.get_order(order.order_id) is not None
    await manager.broadcast()
    return templates.TemplateResponse(request=request, name="components/order_result.html",
                                      context=order_outcome("market", order, size, fills, rests, side=side,
                                                            vwap_before=vwap_before))


@app.post("/api/order/cancel")
async def post_cancel_order(
    manager: SessionManager,
    order_id: int = Form(...),
):
    async with manager._lock:
        manager.cancel_order_by_id(order_id)
    await manager.broadcast()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def render_playback_btn(is_running: bool) -> str:
    return templates.get_template("components/play_button.html").render(is_running=is_running)


@app.post("/api/sim/control")
async def post_sim_control(
    manager: SessionManager,
    action: Literal["play", "pause", "toggle", "step", "reset"] = Form(...),
    n: int = Form(1, ge=1, le=10_000),
):
    if manager.collecting is not None and action in ("play", "toggle", "step"):
        # A fast collection is running (and keeps the simulation paused until it ends).
        if action == "step":
            return Response(status_code=status.HTTP_204_NO_CONTENT)
        return HTMLResponse(render_playback_btn(False))
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


@app.post("/api/sim/collect")
async def post_collect(manager: SessionManager, n: int = Form(400_000, ge=10_000, le=2_000_000)):
    """Run n model steps as fast as possible, for the stylized-facts panel (409 if one is running)."""
    if not await manager.collect(n):
        raise HTTPException(status.HTTP_409_CONFLICT, "A collection is already running")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/sim/config")
async def post_sim_config(
    manager: SessionManager,
    model: Optional[str] = Form(None),
    speed: Optional[int] = Form(None, ge=1, le=2000),
):
    if model:
        if model not in MODEL_SPECS:
            raise HTTPException(status_code=422, detail=f"Unknown model {model!r}")
        async with manager._lock:
            manager.set_model(model)
    if speed is not None:
        manager.set_speed(speed)
    await manager.broadcast()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/sim/model", response_class=HTMLResponse)
async def post_sim_model(
    request: Request,
    manager: SessionManager,
    model: str = Form(...),
    params_open: bool = Form(False),
):
    """Switch model (continuing from the current book) and return its parameter panel."""
    if model not in MODEL_SPECS:
        raise HTTPException(status_code=422, detail=f"Unknown model {model!r}")
    async with manager._lock:
        manager.set_model(model)
    await manager.broadcast()
    return render_model_params(request, manager, open_panel=params_open)


@app.post("/api/sim/params", response_class=HTMLResponse)
async def post_sim_params(request: Request, manager: SessionManager):
    """Apply the current model's parameters and the seed (form fields named after them).

    Invalid input changes nothing: the panel comes back with the messages
    and status 422.
    """
    form = dict(await request.form())
    async with manager._lock:
        errors = manager.configure(form.get("model", manager.model_type), form, form.get("seed"))
    if errors:
        return render_model_params(request, manager, status_code=422, values=form, errors=errors,
                                   open_panel=True)
    await manager.broadcast()
    return render_model_params(request, manager, message="Applied", open_panel=True)


@app.post("/api/engine", response_class=HTMLResponse)
async def post_engine(
    request: Request,
    manager: SessionManager,
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
async def get_snapshot(manager: SessionManager):
    return manager.book.snapshot()
