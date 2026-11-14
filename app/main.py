"""FastAPI web application for interactive Limit Order Book simulation."""

from __future__ import annotations

import asyncio
from html import escape
import importlib.util
import io
import json
from pathlib import Path
import time
from typing import Annotated, Any, Callable, Literal, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles

from lob.order import Fill, Order, Side
from app import recording as rec
from app.depth_chart import depth_chart
from app.facts import WARMUP_STEPS, summarize as summarize_facts, summarize_order_flow
from app.fitting import FIT_OBJECTIVES, FIT_SEED, MAX_FIT_CANDIDATES, GridError, parse_grid
from app.manager import (DEFAULT_PROTECTION_TICKS, HEAT_HALF, MAX_PROTECTION_TICKS, REGIME_TRADES, STRATEGY_SAMPLES,
                         LOBManager)
from app.model_specs import MODEL_SPECS, Param
from app.pairs import (MARKET_PARAMS as PAIRS_MARKET_PARAMS, MAX_STEP as PAIRS_MAX_STEP,
                       MAX_TRADERS as PAIRS_MAX_TRADERS, SAMPLES as PAIRS_SAMPLES, SPEED_MAX as PAIRS_SPEED_MAX,
                       TRADER_KINDS as PAIRS_TRADER_KINDS, PairsRunner, PairsSession, TraderKind,
                       parse as parse_pairs)
from app.statarb import (MARKET_PARAMS as STATARB_MARKET_PARAMS, MAX_STEP as STATARB_MAX_STEP,
                         NOTHING_TO_CHOOSE as STATARB_NOTHING_TO_CHOOSE, SHAPES as STATARB_SHAPES,
                         SIGNAL_SLOTS as STATARB_SIGNAL_SLOTS, SPEED_MAX as STATARB_SPEED_MAX,
                         TRADER_KINDS as STATARB_TRADER_KINDS, StatArbRunner, StatArbSession, parse_market)
from app.strategy_specs import EXECUTION_LABELS, FAMILIES, MAX_STRATEGIES, STRATEGY_SPECS
from app.studies import STUDY_HELP, STUDY_LABELS, STUDY_SHORT, STUDY_SPECS
from app.triggers import (ACTIONS as TRIGGER_ACTIONS, DEFAULT_COOLDOWN as TRIGGER_COOLDOWN,
                          EDGE_RULES as TRIGGER_EDGE_RULES, MAX_CONDITIONS as MAX_TRIGGER_CONDITIONS,
                          MAX_RULES as MAX_TRIGGER_RULES, NAME_MAX as TRIGGER_NAME_MAX, OPERATORS as TRIGGER_OPERATORS,
                          VALUE_GROUPS as TRIGGER_VALUE_GROUPS, parse_rule)
from app.sessions import SessionMiddleware, SessionRegistry
from app.training import CLEARLY, FIT_RUNS, FIT_SPAN, FitCanceled, TrainedMarket, same_market

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
PARQUET_AVAILABLE = importlib.util.find_spec("pyarrow") is not None  # the optional [parquet] extra
templates.env.globals["parquet_available"] = PARQUET_AVAILABLE
templates.env.globals["protection_max"] = MAX_PROTECTION_TICKS
templates.env.globals["heat_half"] = HEAT_HALF
templates.env.globals["strategy_specs"] = STRATEGY_SPECS
templates.env.globals["strategy_tags"] = {spec.tag: spec.label for spec in STRATEGY_SPECS.values()}
templates.env.globals["max_strategies"] = MAX_STRATEGIES
templates.env.globals["strategy_samples"] = STRATEGY_SAMPLES
templates.env.globals["max_recordings"] = rec.MAX_RECORDINGS
templates.env.globals["strategy_families"] = FAMILIES
templates.env.globals["study_specs"] = STUDY_SPECS
templates.env.globals["study_labels"] = STUDY_LABELS
templates.env.globals["study_short"] = STUDY_SHORT
templates.env.globals["study_help"] = STUDY_HELP
templates.env.globals["trigger_value_groups"] = TRIGGER_VALUE_GROUPS
templates.env.globals["trigger_operators"] = TRIGGER_OPERATORS
templates.env.globals["trigger_actions"] = TRIGGER_ACTIONS
templates.env.globals["trigger_edge_rules"] = TRIGGER_EDGE_RULES
templates.env.globals["trigger_cooldown"] = TRIGGER_COOLDOWN
templates.env.globals["trigger_name_max"] = TRIGGER_NAME_MAX
templates.env.globals["max_trigger_rules"] = MAX_TRIGGER_RULES
templates.env.globals["max_trigger_conditions"] = MAX_TRIGGER_CONDITIONS
templates.env.globals["fit_objectives"] = FIT_OBJECTIVES
templates.env.globals["max_fit_candidates"] = MAX_FIT_CANDIDATES
templates.env.globals["regime_trades"] = REGIME_TRADES
templates.env.globals["execution_labels"] = EXECUTION_LABELS
templates.env.globals["max_recording_events"] = rec.MAX_EVENTS
templates.env.globals["pairs_market_params"] = {p.name: p for p in PAIRS_MARKET_PARAMS}
templates.env.globals["pairs_trader_kinds"] = PAIRS_TRADER_KINDS
templates.env.globals["pairs_samples"] = PAIRS_SAMPLES
templates.env.globals["pairs_speed_max"] = PAIRS_SPEED_MAX
templates.env.globals["pairs_max_traders"] = PAIRS_MAX_TRADERS
templates.env.globals["pairs_max_step"] = PAIRS_MAX_STEP
templates.env.globals["pairs_fit_span"] = FIT_SPAN
templates.env.globals["pairs_fit_runs"] = FIT_RUNS
templates.env.globals["pairs_clearly"] = CLEARLY
templates.env.globals["statarb_market_params"] = {p.name: p for p in STATARB_MARKET_PARAMS}
templates.env.globals["statarb_trader_kinds"] = STATARB_TRADER_KINDS
templates.env.globals["statarb_shapes"] = STATARB_SHAPES
templates.env.globals["statarb_speed_max"] = STATARB_SPEED_MAX
templates.env.globals["statarb_max_step"] = STATARB_MAX_STEP
templates.env.globals["statarb_signal_slots"] = STATARB_SIGNAL_SLOTS

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
            # The order's place in its price level's queue: the visible lots in front of it
            # (what must trade before it does under price-time priority), and its share of the
            # level (what pro-rata matching allocates by).
            ahead = ahead_orders = 0
            for queued in book.orders_at(o.price, o.side):
                if queued.order_id == oid:
                    break
                ahead += queued.size
                ahead_orders += 1
            level = book.depth_at_price(o.price, o.side)
            active_orders.append({
                "order_id": o.order_id,
                "side": o.side.value,
                "price": o.price,
                "size": o.size,  # showing (an iceberg's peak)
                "hidden": o.hidden,  # an iceberg's reserve
                "timestamp": o.timestamp,
                "ahead": ahead,
                "ahead_orders": ahead_orders,
                "share": o.size / level if level else 1.0,
            })

    # Lots of the user's own resting orders at each (side, price), marked in the ladder:
    # showing, and hidden in icebergs
    own_size: dict[tuple[str, int], int] = {}
    own_hidden: dict[tuple[str, int], int] = {}
    for o in active_orders:
        key = (o["side"], o["price"])
        own_size[key] = own_size.get(key, 0) + o["size"]
        own_hidden[key] = own_hidden.get(key, 0) + o["hidden"]
    # ... and of the user's strategies' orders, by (side, price) and the strategy's tag.
    strategy_size: dict[tuple[str, int], dict[str, int]] = {}
    for s in manager.strategies:
        for oid in s.order_ids:
            o = book.get_order(oid)
            if o is not None:
                tags = strategy_size.setdefault((o.side.value, o.price), {})
                tags[s.spec.tag] = tags.get(s.spec.tag, 0) + o.size

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
            "strategies": sorted(strategy_size.get(("bid", p), {}).items()),
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
            "strategies": sorted(strategy_size.get(("ask", p), {}).items()),
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
        "depth": depth_chart(book, own=own_size.keys()),  # the depth chart under the ladder
        "imbalance": imbalance,
        "bid_ratio": bid_ratio,
        "ask_ratio": ask_ratio,
        "account": manager.account_summary(),  # the user's position and P&L
        "active_orders": active_orders,
        # The same per price and side, for the dashed lines on the chart.
        "own_levels": [{"side": side, "price": price, "size": size, "hidden": own_hidden[(side, price)]}
                       for (side, price), size in sorted(own_size.items())],
        "strategy_levels": [[side, price, size, tag] for (side, price), tags in sorted(strategy_size.items())
                            for tag, size in sorted(tags.items())],
        "strategies": manager.strategies_summary(),  # the leaderboard
        "regime": manager.regime(),  # whether the order flow pays resting orders here
        "null_running": any(s.running and s.spec.key == "quote_joiner" for s in manager.strategies),
        "recording": manager.recording_status(),  # a recording in progress
        "playback": manager.playback_status(),
        "fitting": manager.fitting,  # (candidates done, candidates) while a strategy is fitted
        "recordings": [r.describe() for r in manager.recordings],
        "recordings_version": manager.recordings_version,
        # The latest fills of the user's resting orders, newest first.
        "own_fills": [{"text": f.describe(), "time": f.time} for f in reversed(list(manager.own_fills)[-5:])],
        "notice_seq": manager.notice_seq,
        "split_orders": [{
            "split_id": s.split_id, "side": s.side.value, "total": s.total, "slices": s.slices,
            "interval": s.interval, "sent": s.sent, "filled": s.filled, "average": s.average,
            "next_time": s.next_time,
        } for s in manager.split_orders],
        "split_history": list(reversed(manager.split_history)),
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
        "informed": manager.informed_status(),  # the live ground truth line (components/model_params.html)
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
    ctx["pairs"] = get_pairs(manager).context()
    ctx["statarb"] = get_statarb(manager).context()
    ctx["triggers"] = manager.triggers_status()  # the rules when the page is made (components/triggers_card.html)
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
                facts=summarize_facts(data, model_type), collecting=collecting, model_type=model_type,
                model_spec=MODEL_SPECS[model_type]),
            "ofi": templates.get_template("components/ofi.html").render(ofi=summarize_order_flow(data)),
        }

    html = await asyncio.to_thread(render)
    manager.facts_cache = (key, html)
    return html


PANELS = ("ladder", "depth", "metrics", "active_orders", "sim_status", "pnl", "strategies", "recordings")
PANEL_TEMPLATES = {
    "recordings": "components/recordings.html",
    "strategies": "components/strategies.html",
    "ladder": "components/ladder.html",
    "depth": "components/depth_chart.html",
    "pnl": "components/pnl.html",
    "metrics": "components/metrics.html",
    "active_orders": "components/active_orders.html",
    "sim_status": "components/sim_badge.html",
}
PANEL_PUSH_INTERVAL = 0.2  # seconds between panel pushes while the simulation runs


TICKS_HALF = 60  # ticks either side of the mid in the price ladder's data (it shows 40)


def ticks_view(manager: LOBManager, own_levels: list[dict[str, Any]],
               strategy_levels: Optional[list[list[Any]]] = None) -> dict[str, Any]:
    """The book at every tick near the mid, for the tick-by-tick price ladder (static/js/lob-ticks.js).

    Visible lots bid and offered at each price from ``lo`` up, the user's resting lots
    at each (side, price) as [side, price, showing, hidden], their strategies' as
    [side, price, lots, tag], and the best quotes.
    """
    book = manager.book
    mid = book.mid_price
    center = round(mid) if mid is not None else (book.best_bid or book.best_ask or round(manager._reference_price))
    lo = max(1, center - TICKS_HALF)
    prices = range(lo, lo + 2 * TICKS_HALF + 1)
    return {
        "lo": lo, "mid": mid, "best_bid": book.best_bid, "best_ask": book.best_ask,
        "bids": [book.depth_at_price(p, Side.BID) for p in prices],
        "asks": [book.depth_at_price(p, Side.ASK) for p in prices],
        "own": [[lv["side"], lv["price"], lv["size"], lv["hidden"]] for lv in own_levels if lv["price"] in prices],
        "strategies": [lv for lv in strategy_levels or [] if lv[1] in prices],
    }


def render_panels(ctx: dict[str, Any]) -> dict[str, str]:
    """Render every live panel; called on a worker thread to keep the event loop free."""
    return {name: templates.get_template(PANEL_TEMPLATES[name]).render(ctx) for name in PANELS}


@app.get("/events")
async def events_stream(request: Request, manager: SessionManager):
    """Server-Sent Events: one stream per tab carrying both the page panels and the chart.

    Browsers allow about six connections per host over HTTP/1.1, so a single
    stream per tab leaves room for more tabs and for the page's own requests
    (on the Pairs view a tab holds /pairs/events instead of this one).
    Event types: ``hello`` (the simulation's public instance ID, so a page
    can tell it is talking to a different simulation after a restart), one
    per panel (rendered HTML), ``candles`` (every bar since the last push),
    ``chart_reset`` (reload the chart), ``own_trades`` (the user's fills per
    bar, for the chart's markers), ``heatmap`` (its new columns), ``ticks``
    (the book at every tick near the mid, for the price ladder), ``studies``
    (the microstructure studies: ``LOBManager.studies_snapshot``, with the samples
    since the last push), ``tape`` (the fills since the last push, or the whole
    ring: ``LOBManager.tape_since``), ``triggers`` (the trigger rules and what each has
    seen: ``LOBManager.triggers_status``, when it changed), ``informed`` (the informed
    traders' ground truth and the Reveal switch: ``LOBManager.informed_status``, when it changed) and
    ``controls`` (model, parameters, seed, engine and speed, so every tab of a
    session stays current).
    """
    queue = manager.subscribe()
    session_id = request.state.session_id

    async def event_generator():
        version = manager.chart_version
        config_version: Optional[int] = None
        last_time: Optional[int] = None
        last_orders: Optional[tuple] = None
        recordings_seen: Optional[int] = None
        last_panel_push = 0.0
        facts_pushed: Optional[tuple] = None
        last_facts_push = 0.0
        notices_seen = manager.notice_seq  # show only notices from now on
        own_trades_seen: Optional[int] = None  # sent on connect too: the chart may have loaded before a fill
        heat_seen: Optional[int] = None  # the heatmap's change count and version when last pushed
        heat_version: Optional[int] = None
        heat_last: Optional[int] = None  # the index of the last column pushed
        studies_version: Optional[int] = None  # what the studies event and the tape event last carried
        studies_unit: Optional[int] = None
        tape_seen: Optional[int] = None
        triggers_seen: Optional[str] = None  # the rules' status as last pushed
        informed_seen: Optional[str] = None  # the informed traders' ground truth as last pushed
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

                # Notices since the last push: fills of the user's resting orders, finished split
                # orders (each tab shows them once, as toasts).
                if manager.notice_seq != notices_seen:
                    news = [n.text for n in manager.notices if n.seq > notices_seen]
                    notices_seen = manager.notice_seq
                    if news:
                        yield format_sse("notices", json.dumps(news))

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
                    cells, fair = manager.footprint_on, manager.reveal_active
                    bars = manager.get_candles(limit=1, footprint=cells, reveal=fair) if last_time is None \
                        else manager.get_candles(since=last_time, footprint=cells, reveal=fair)
                    if bars:
                        last_time = bars[-1]["time"]
                        # Each bar says which version of the chart it belongs to: a page that has loaded another
                        # (its reload's answer and these pushes come on different connections, in no fixed order)
                        # drops it.
                        yield format_sse("candles", json.dumps([{**bar, "v": version} for bar in bars]))

                # The user's fills per bar (after the bars, so each marker has its bar), when they change.
                if manager.own_trade_seq != own_trades_seen:
                    own_trades_seen = manager.own_trade_seq
                    yield format_sse("own_trades", json.dumps(manager.own_trade_marks()))

                # Panels: at most five pushes a second while running.
                now = time.monotonic()
                if not first and manager.is_running and now - last_panel_push < PANEL_PUSH_INTERVAL:
                    continue
                last_panel_push = now
                ctx = get_view_context(manager)
                rendered = await asyncio.to_thread(render_panels, ctx)
                orders = (tuple((o["order_id"], o["size"], o["hidden"], o["ahead"], round(o["share"], 3))
                                for o in ctx["active_orders"]),
                          ctx["engine_type"], ctx["notice_seq"],
                          tuple((s["split_id"], s["sent"], s["filled"]) for s in ctx["split_orders"]))
                for name in PANELS:
                    # The order table is replaced only when an order appeared, changed or left, and the
                    # recordings only when one did (both have buttons a replacement would take away).
                    if name == "active_orders" and orders == last_orders:
                        continue
                    if name == "recordings" and ctx["recordings_version"] == recordings_seen:
                        continue
                    yield format_sse(name, rendered[name])
                last_orders = orders
                recordings_seen = ctx["recordings_version"]
                yield format_sse("ticks", json.dumps(ticks_view(manager, ctx["own_levels"], ctx["strategy_levels"])))

                # The studies: their samples since the last push, all of them when a page has none or what it has
                # is out of date (a new version); the tape: the fills since the last push, or the ring when behind.
                same = studies_version == manager.studies_version
                studies = manager.studies_snapshot(studies_unit if same else None)
                studies_unit = studies["unit"]
                yield format_sse("studies", json.dumps(studies))
                tape = manager.tape_since(tape_seen if same else None)
                if tape["reload"] or tape["fills"]:
                    yield format_sse("tape", json.dumps(tape))
                studies_version, tape_seen = studies["version"], tape["seq"]

                # The rules' status (definitions, what each has seen, the firings for the chart's markers), only
                # when it changed: while paused, or with no rule armed, nothing is sent.
                rules = json.dumps(manager.triggers_status())
                if rules != triggers_seen:
                    triggers_seen = rules
                    yield format_sse("triggers", rules)

                # The informed traders' ground truth (the share of their lots, what makers lose to them) and the
                # Reveal switch, when it changed: with another model, or in a playback, it says so once.
                truth = json.dumps(manager.informed_status())
                if truth != informed_seen:
                    informed_seen = truth
                    yield format_sse("informed", truth)

                # The heatmap: columns from the last one pushed on (its trades may have grown since).
                # A page loads the history from /api/heatmap and reloads it when the version changes.
                if manager.heat_changes != heat_seen:
                    heat_seen = manager.heat_changes
                    if manager.heat_version != heat_version:
                        heat_version, heat_last = manager.heat_version, None
                    columns = manager.heat_columns(since=heat_last) if heat_last is not None \
                        else [manager.latest_heat_column()]
                    if columns:
                        heat_last = columns[-1]["t"]
                        yield format_sse("heatmap", json.dumps({"version": heat_version, "columns": columns}))
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
    """Return the OHLCV bars for the current interval, built from executed trades.

    With the footprint on (``POST /api/chart/footprint``) each bar also has ``cells``: [price, lots sold into the
    bid, lots bought from the ask] by rising price, or null for a bar that does not keep them. With Reveal on
    (``POST /api/chart/reveal``) and a model that has a hidden value, each bar has ``fair``: that value at the bar's
    last sampled unit, or null.
    """
    candles = manager.get_candles(footprint=manager.footprint_on, reveal=manager.reveal_active)
    return {
        "version": manager.chart_version,  # the stream's bars carry it too (``v``): the page drops those of another
        "interval": manager.bar_interval,
        "footprint": manager.footprint_on,
        "reveal": manager.informed_status()["reveal"],
        "candles": candles,
        "latest": candles[-1] if candles else None,
        "own_trades": manager.own_trade_marks(),  # the user's fills per bar, for the markers
    }


@app.get("/api/studies")
async def get_studies(manager: SessionManager, since: Optional[int] = None):
    """The microstructure studies: version, settings, latest values and the unit samples after unit ``since``.

    What the ``studies`` event of the stream carries (``LOBManager.studies_snapshot``); a value that is not defined
    is null. A page that has the samples up to a unit and the same version asks for those after it.
    """
    return manager.studies_snapshot(since)


@app.get("/api/tape")
async def get_tape(manager: SessionManager, since: Optional[int] = None):
    """The fills after number ``since`` from the last 2,000 (all of them without ``since``), and how to use them.

    The same as the ``tape`` event of the stream (``LOBManager.tape_since``): ``reload`` says the fills replace
    what the page holds (it had none, or fell behind the ring), ``version`` that it should drop what it holds if it
    is not the one it has. A fill is [number, simulation time, price, size, side], side 1 when the buyer was the
    aggressor. Numbers are never reused, not even after a reset.
    """
    return manager.tape_since(since)


@app.get("/api/heatmap")
async def get_heatmap(manager: SessionManager):
    """The order-book heatmap's columns: one per unit of simulation time, the most recent 600.

    Each has its unit's index ``t``, the lowest price ``lo``, the mid, the visible depth at
    each price from ``lo`` up (bids positive, asks negative), and [price, lots] pairs of
    iceberg reserves (``hidden``) and of lots traded during the unit (``traded``).
    """
    return {"version": manager.heat_version, "half": HEAT_HALF, "columns": manager.heat_columns()}


@app.get("/api/export/{what}.{fmt}")
async def export_data(manager: SessionManager, what: Literal["trades", "bars"], fmt: Literal["csv", "parquet"]):
    """Download the session's fills or its bars at the current interval, as CSV or Parquet."""
    if fmt == "parquet" and not PARQUET_AVAILABLE:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, "Parquet export needs pyarrow: pip install pyarrow")
    async with manager._lock:  # copied between simulation steps
        frame = manager.trades_frame() if what == "trades" else manager.bars_frame()
    name = "lob-trades" if what == "trades" else f"lob-bars-{manager.bar_interval}s"

    def serialize() -> bytes:
        if fmt == "csv":
            return frame.to_csv(index=False).encode()
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        return buffer.getvalue()

    return Response(await asyncio.to_thread(serialize),
                    media_type="text/csv" if fmt == "csv" else "application/vnd.apache.parquet",
                    headers={"Content-Disposition": f'attachment; filename="{name}.{fmt}"'})


# ---- Recording and playback (app/recording.py) ----

@app.post("/api/recording/start")
async def post_start_recording(manager: SessionManager):
    """Record from now on; 409 while recording already, or with the most recordings kept."""
    async with manager._lock:
        try:
            manager.start_recording()
        except RuntimeError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, f"Not recording: {exc}.") from None
    await manager.broadcast()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/recording/stop")
async def post_stop_recording(manager: SessionManager):
    """Stop recording and keep the recording; 409 when not recording."""
    async with manager._lock:
        try:
            recording = manager.stop_recording()
        except RuntimeError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, f"Nothing to stop: {exc}.") from None
    await manager.broadcast()
    return {"id": recording.id, "name": recording.name, "events": len(recording.tape)}


def find_recording(manager: LOBManager, recording_id: int) -> rec.Recording:
    recording = manager.recording(recording_id)
    if recording is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such recording in this session")
    return recording


@app.get("/api/recordings/{recording_id}/download")
async def get_recording_file(manager: SessionManager, recording_id: int):
    """A recording as a NumPy .npz file, to keep or to play back later (upload it again)."""
    recording = find_recording(manager, recording_id)
    data = await asyncio.to_thread(rec.to_bytes, recording)  # a finished recording does not change
    filename = "lob-" + "".join(c if c.isalnum() else "-" for c in recording.name.lower()).strip("-") + ".npz"
    return Response(data, media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.post("/api/recordings/upload")
async def post_upload_recording(manager: SessionManager, file: Annotated[UploadFile, File()]):
    """Keep a recording from a file; 422 saying why a file is not one, 409 with the most kept, 413 if too large."""
    data = await file.read(rec.MAX_FILE_BYTES + 1)
    if len(data) > rec.MAX_FILE_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE,
                            f"The file is larger than {rec.MAX_FILE_BYTES // 2**20} MB.")
    async with manager._lock:
        try:
            recording = manager.add_recording(data)
        except RuntimeError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, f"Not added: {exc}.") from None
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                                f"Not a recording this simulator can play: {exc}.") from None
    await manager.broadcast()
    return {"id": recording.id, "name": recording.name, "events": len(recording.tape)}


@app.post("/api/recordings/{recording_id}/delete")
async def post_delete_recording(manager: SessionManager, recording_id: int):
    async with manager._lock:
        recording = find_recording(manager, recording_id)
        try:
            manager.delete_recording(recording)
        except RuntimeError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, f"Not deleted: {exc}.") from None
    await manager.broadcast()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/playback/start")
async def post_start_playback(manager: SessionManager, recording_id: int = Form(...)):
    """Play a recording back from its start, paused (the session becomes the recording's)."""
    recording = find_recording(manager, recording_id)
    await manager.stop_sim_loop()
    async with manager._lock:
        manager.start_playback(recording)
    await manager.broadcast()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/playback/seek")
async def post_seek_playback(manager: SessionManager, time: float = Form(..., allow_inf_nan=False)):
    """Move the playback to a simulation time (going back replays the tape from its start); 409 without one."""
    if manager.playback is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Nothing is playing back")
    await manager.seek_playback(time)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/playback/exit")
async def post_exit_playback(manager: SessionManager):
    """End the playback and carry on live from here: the selected model takes over the book."""
    async with manager._lock:
        manager.exit_playback()
    await manager.broadcast()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/chart/interval")
async def post_chart_interval(manager: SessionManager, interval: int = Form(..., ge=1, le=3600)):
    async with manager._lock:
        manager.set_bar_interval(interval)
    await manager.broadcast()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/chart/footprint")
async def post_chart_footprint(manager: SessionManager, on: bool = Form(...)):
    """Carry the footprint's cells in this session's candles (or stop); every chart reloads."""
    async with manager._lock:
        manager.set_footprint(on)
    await manager.broadcast()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/chart/reveal")
async def post_chart_reveal(manager: SessionManager, on: bool = Form(...)):
    """Carry the informed-traders model's hidden value in this session's candles (or stop); every chart reloads.

    Only that model has one: turning it on with another answers 422.
    """
    async with manager._lock:
        try:
            manager.set_reveal(on)
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from None
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
    slices: int = Form(1, ge=1, le=100),
    interval: float = Form(5.0, gt=0, le=1_000),
):
    """A market order that walks at most ``protection`` ticks; the rest rests as a limit order.

    With ``slices`` above 1 it is split into that many market orders, one every ``interval``
    of simulation time (a TWAP schedule), and the reply describes the plan.
    """
    if slices > 1:
        async with manager._lock:
            split = manager.start_split_order(side=Side(side), total=size, slices=slices, interval=interval,
                                              protection=protection)
        await manager.broadcast()
        return templates.TemplateResponse(request=request, name="components/order_result.html", context={
            "kind": "split", "side": side, "size": size, "slices": slices, "interval": interval,
            "protection": protection, "slice_size": -(-size // slices), "sweep_filled": split.sweep_filled,
            "sweep_avg": split.sweep_avg, "filled": split.filled, "finished": split.finished,
        })
    async with manager._lock:
        vwap_before = manager.session_vwap()
        order, fills = manager.submit_market_order(size=size, side=Side(side), protection=protection)
        rests = order is not None and manager.book.get_order(order.order_id) is not None
    await manager.broadcast()
    return templates.TemplateResponse(request=request, name="components/order_result.html",
                                      context=order_outcome("market", order, size, fills, rests, side=side,
                                                            vwap_before=vwap_before))


@app.post("/api/order/split/cancel")
async def post_cancel_split_order(manager: SessionManager, split_id: int = Form(...)):
    """Stop a split order: no more slices go out (what filled stays filled)."""
    async with manager._lock:
        found = manager.cancel_split_order(split_id)
    await manager.broadcast()
    if not found:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No working split order with that ID")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def move_text(side: Side, from_price: int, to_price: int, old: list[Order], new: list[Order],
              fills: list[Fill], resting: int) -> str:
    """What moving the user's orders did, for the page's notice."""
    kind = "buy" if side == Side.BID else "sell"
    what = f"your {kind} order" if len(old) == 1 else f"your {len(old)} {kind} orders"
    lots = sum(o.size + o.hidden for o in old)
    text = f"Moved {what} ({lots:,} lots) from {from_price} to {to_price}"
    filled = sum(f.size for f in fills)
    if not filled:
        return text + ", at the back of the queue there."
    prices = [f.price for f in fills]
    where = f"{min(prices)}" if min(prices) == max(prices) else f"{min(prices)}\u2013{max(prices)}"
    verb = "bought" if side == Side.BID else "sold"
    rest = f"{resting:,} rest at {to_price}" if resting else "nothing rests"
    return f"{text}: across the spread, so it {verb} {filled:,} on arrival at {where}; {rest}."


@app.post("/api/order/move")
async def post_move_orders(
    manager: SessionManager,
    side: Literal["bid", "ask"] = Form(...),
    from_price: int = Form(..., ge=1),
    to_price: int = Form(..., ge=1),
):
    """Move your resting orders at one price and side to another price (dragged in the order book).

    They are canceled and entered again, so they lose their place in the queue; 404 if you
    have no orders there. Answers with what happened, for a notice.
    """
    if from_price == to_price:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "from_price and to_price are the same")
    async with manager._lock:
        old, new, fills = manager.move_orders(Side(side), from_price, to_price)
        # What rests, from the book: an order that trades on arrival rests as a copy with what is left.
        resting = sum(r.size + r.hidden for r in (manager.book.get_order(o.order_id) for o in new) if r is not None)
    if not old:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"You have no {side} orders resting at {from_price}")
    await manager.broadcast()
    return {"moved": len(old), "filled": sum(f.size for f in fills), "resting": resting,
            "text": move_text(Side(side), from_price, to_price, old, new, fills, resting)}


@app.post("/api/order/cancel_all")
async def post_cancel_all_orders(
    manager: SessionManager,
    side: Optional[Literal["bid", "ask"]] = Form(None),
    price: Optional[int] = Form(None, ge=1),
):
    """Cancel your resting orders: all of them, or those on one side or at one price."""
    async with manager._lock:
        canceled = manager.cancel_all_orders(side=Side(side) if side else None, price=price)
    await manager.broadcast()
    return {"canceled": canceled}


def strategy_result(text: str, tone: str = "rose", strategy_id: Optional[int] = None) -> str:
    """The Strategies card's message under its form (#strategy-result)."""
    data = f' data-strategy-id="{strategy_id}"' if strategy_id is not None else ""
    return f'<span class="text-{tone}-400"{data}>{text}</span>'


@app.post("/api/strategies/start", response_class=HTMLResponse)
async def post_start_strategy(request: Request, manager: SessionManager):
    """Start a strategy (app/strategy_specs.py): ``kind`` and its settings, each defaulting to the spec's.

    Answers with a message for the card; the new strategy's id is its ``data-strategy-id``.
    422 for a setting out of range or settings that contradict each other, 409 when the user
    already has the most strategies allowed.
    """
    form = await request.form()
    if form.get("fit_id") is not None:  # a candidate of the last fit (its result table's Start)
        try:
            fit_id, rank = int(str(form.get("fit_id"))), int(str(form.get("rank", "")))
        except ValueError:
            return HTMLResponse(strategy_result("A fit's candidate is a fit and a rank."),
                                status.HTTP_422_UNPROCESSABLE_CONTENT)
        async with manager._lock:
            try:
                s = manager.start_fitted(fit_id, rank)
            except (LookupError, RuntimeError) as exc:
                return HTMLResponse(strategy_result(escape(f"Not started: {exc}.")), status.HTTP_409_CONFLICT)
        await manager.broadcast()
        return HTMLResponse(strategy_result(
            f"Started <b>{escape(s.name)}</b>, as fitted: {escape(s.strategy.describe())}.", tone="emerald",
            strategy_id=s.id))
    spec = STRATEGY_SPECS.get(str(form.get("kind", "")))
    if spec is None:
        return HTMLResponse(strategy_result(f"Choose one of: {', '.join(STRATEGY_SPECS)}."),
                            status.HTTP_422_UNPROCESSABLE_CONTENT)
    values, errors = spec.parse(dict(form))
    if errors:
        return HTMLResponse(strategy_result(escape(" ".join(errors.values()))), status.HTTP_422_UNPROCESSABLE_CONTENT)
    async with manager._lock:
        try:
            s = manager.start_strategy(spec, values)
        except ValueError as exc:
            return HTMLResponse(strategy_result(escape(f"{spec.label}: {exc}.")), status.HTTP_422_UNPROCESSABLE_CONTENT)
        except RuntimeError as exc:
            return HTMLResponse(strategy_result(escape(f"Not started: {exc}.")), status.HTTP_409_CONFLICT)
    await manager.broadcast()
    hint = "" if manager.is_running else " It trades as the simulation runs."
    return HTMLResponse(strategy_result(f"Started <b>{escape(s.name)}</b>: {escape(s.strategy.describe())}.{hint}",
                                        tone="emerald", strategy_id=s.id))


def study_result(text: str, tone: str = "rose") -> str:
    """The message under a study's settings (#studies-result)."""
    return f'<span class="text-{tone}-400">{text}</span>'


@app.post("/api/studies/config", response_class=HTMLResponse)
async def post_studies_config(request: Request, manager: SessionManager):
    """Change one study's settings (form fields named after them, the study's key in ``study``) or switch it.

    Only the fields given change; ``enabled`` (on or off) switches the study. The study starts afresh and every
    page reloads its samples. Answers with a message for the page; 422, with the message, for a setting out of range,
    settings that contradict each other, or a study that is not there.
    """
    form = dict(await request.form())
    key = str(form.get("study", ""))
    if manager.studies is None or key not in STUDY_SPECS:
        return HTMLResponse(study_result(escape(f"Choose one of: {', '.join(STUDY_LABELS)}.")),
                            status.HTTP_422_UNPROCESSABLE_CONTENT)
    values, errors = manager.studies.parse(key, form)
    enabled = {"on": True, "true": True, "1": True, "off": False, "false": False, "0": False}.get(
        str(form.get("enabled", "")).lower())
    if "enabled" in form and enabled is None:
        errors["enabled"] = f"{STUDY_LABELS[key]}: choose on or off"
    if errors:
        return HTMLResponse(study_result(escape(" ".join(errors.values()))), status.HTTP_422_UNPROCESSABLE_CONTENT)
    async with manager._lock:
        try:
            changed = manager.configure_study(key, values, enabled)
        except ValueError as exc:
            return HTMLResponse(study_result(escape(f"{STUDY_LABELS[key]}: {exc}.")),
                                status.HTTP_422_UNPROCESSABLE_CONTENT)
    if changed:
        await manager.broadcast()
    return HTMLResponse(study_result(f"{escape(STUDY_LABELS[key])}: " + ("applied" if changed else "no change"),
                                     tone="emerald"))


def trigger_result(text: str, tone: str = "rose", rule_id: Optional[int] = None) -> str:
    """The message under the trigger editor (#triggers-result); a rule's id is its ``data-trigger-id``."""
    data = f' data-trigger-id="{rule_id}"' if rule_id is not None else ""
    return f'<span class="text-{tone}-400"{data}>{text}</span>'


def trigger_note(manager: LOBManager) -> str:
    """What to know about a rule that was just made or changed, for the message that says so."""
    notes = []
    if manager.model_type == "manual":
        notes.append("The manual model takes no steps, so it is checked only after an order of yours.")
    if manager.studies is None or manager.studies.failed is not None:
        notes.append("The studies are off.")
    return " ".join(notes)


@app.get("/api/triggers")
async def get_triggers(manager: SessionManager):
    """The trigger rules: each one's definition and what it has seen, and the firings that mark the chart.

    What the ``triggers`` event of the stream carries (``LOBManager.triggers_status``); every number is finite or null.
    """
    return manager.triggers_status()


@app.post("/api/triggers", response_class=HTMLResponse)
async def post_trigger(request: Request, manager: SessionManager):
    """Make a trigger rule (app/triggers.py ``parse_rule``: the fields of the editor; each condition is one of the
    repeated ``value``, ``op`` and ``threshold``, the actions are ``action`` repeated).

    Answers with a message for the editor; the new rule's id is its ``data-trigger-id``. 422, with the message, for
    what is wrong with it (a crossing twice in an all-of rule, an unknown value, no action, too many rules).
    """
    if manager.triggers is None or manager.studies is None:
        return HTMLResponse(trigger_result("This simulation has no studies to read."),
                            status.HTTP_422_UNPROCESSABLE_CONTENT)
    try:
        arguments = parse_rule(await request.form())
        arguments["armed"] = arguments["armed"] is not False
        async with manager._lock:
            rule = manager.triggers.create(**arguments)
    except ValueError as exc:
        return HTMLResponse(trigger_result(escape(str(exc))), status.HTTP_422_UNPROCESSABLE_CONTENT)
    await manager.broadcast()
    return HTMLResponse(trigger_result(f"Added <b>{escape(rule.name)}</b>: {escape(rule.describe())}. "
                                       f"{escape(trigger_note(manager))}",
                                       tone="emerald", rule_id=rule.id))


@app.post("/api/triggers/{rule_id}", response_class=HTMLResponse)
async def post_trigger_update(request: Request, manager: SessionManager, rule_id: int):
    """Change a rule: the same fields as making one. What it has seen is cleared. 404 for a rule that is not there."""
    if manager.triggers is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such rule")
    try:
        arguments = parse_rule(await request.form())
        async with manager._lock:
            rule = manager.triggers.update(rule_id, **arguments)
    except LookupError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such rule") from None
    except ValueError as exc:
        return HTMLResponse(trigger_result(escape(str(exc))), status.HTTP_422_UNPROCESSABLE_CONTENT)
    await manager.broadcast()
    return HTMLResponse(trigger_result(f"Saved <b>{escape(rule.name)}</b>: {escape(rule.describe())}. "
                                       f"{escape(trigger_note(manager))}",
                                       tone="emerald", rule_id=rule.id))


@app.post("/api/triggers/{rule_id}/{action}")
async def post_trigger_action(manager: SessionManager, rule_id: int, action: Literal["arm", "disarm", "delete"]):
    """Arm a rule (from now: what it had seen is forgotten), disarm it, or delete it. 404 for an unknown rule."""
    try:
        async with manager._lock:
            if manager.triggers is None:
                raise LookupError("no triggers")
            if action == "delete":
                manager.triggers.delete(rule_id)
            else:
                manager.triggers.arm(rule_id, action == "arm")
    except LookupError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such rule") from None
    await manager.broadcast()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


FIT_EVENTS = Param("events", "Events", 10_000, 1_000, 50_000, 1_000, integer=True)  # a fresh training run


@app.post("/api/strategies/fit", response_class=HTMLResponse)
async def post_fit_strategy(request: Request, manager: SessionManager):
    """Fit a strategy over a training period (LOBManager.fit_strategy): every combination of the values given.

    The form has the strategy's ``kind`` and settings, the settings to search (``fit_<name>``) with
    their values (``grid_<name>``, separated by commas), the ``source`` (``fresh``: a run of
    ``events`` model steps from now, seeded by ``fit_seed``; or ``recording:<id>``) and the
    ``objective`` (``pnl`` or ``sharpe``). With more combinations than a fit tries, a random sample
    of them, drawn from ``fit_seed`` too. Answers with the candidates, best first; 422 saying what
    is wrong, 409 while another fit runs.
    """
    form = await request.form()

    def refuse(text: str, code: int = status.HTTP_422_UNPROCESSABLE_CONTENT) -> HTMLResponse:
        return HTMLResponse(strategy_result(escape(text)), code)

    spec = STRATEGY_SPECS.get(str(form.get("kind", "")))
    if spec is None:
        return refuse(f"Choose one of: {', '.join(STRATEGY_SPECS)}.")
    base, errors = spec.parse(dict(form))
    if errors:
        return refuse(" ".join(errors.values()))
    try:
        grid = parse_grid(spec.params, form)
    except ValueError as exc:
        return refuse(f"Not fitted: {exc}.")
    try:
        seed = FIT_SEED.parse(form.get("fit_seed", FIT_SEED.default))
    except ValueError as exc:
        return refuse(f"Not fitted: {exc}.")
    if not spec.candidates(base, grid, budget=1, seed=seed)[0]:
        return refuse("Not fitted: none of these combinations of settings is valid.")
    objective = str(form.get("objective", "pnl"))
    if objective not in FIT_OBJECTIVES:
        return refuse(f"Not fitted: best by one of {', '.join(FIT_OBJECTIVES.values())}.")
    if manager.fitting is not None:
        return refuse("Not fitted: another fit is running.", status.HTTP_409_CONFLICT)
    source = str(form.get("source", "fresh"))
    try:
        if source == "fresh":
            events = FIT_EVENTS.parse(form.get("events", FIT_EVENTS.default))
            label = f"a fresh run of {MODEL_SPECS[manager.model_type].label} from {manager.sim_time:.1f}, seed {seed}"
            training = await manager.training_tape(events, seed)
        else:
            recording = manager.recording(int(source.removeprefix("recording:"))) \
                if source.startswith("recording:") and source[10:].isdigit() else None
            if recording is None:
                return refuse("Not fitted: that recording is no longer here.")
            training, label = recording, recording.name
        fit = await manager.fit_strategy(spec, base, grid, training, label, objective, seed)
    except ValueError as exc:
        return refuse(f"Not fitted: {exc}.")
    except RuntimeError as exc:
        return refuse(f"Not fitted: {exc}.", status.HTTP_409_CONFLICT)
    return HTMLResponse(templates.get_template("components/fit_result.html").render(
        fit=fit, spec=spec, objective_label=FIT_OBJECTIVES[fit.objective]))


@app.post("/api/strategies/{strategy_id}/{action}")
async def post_strategy_action(manager: SessionManager, strategy_id: int,
                               action: Literal["stop", "resume", "remove"]):
    """Stop a strategy (its orders withdrawn, its position kept), resume it, or remove it from the board."""
    async with manager._lock:
        s = manager.strategy(strategy_id)
        if s is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No such strategy")
        {"stop": manager.stop_strategy, "resume": manager.resume_strategy, "remove": manager.remove_strategy}[action](s)
    await manager.broadcast()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/order/cancel")
async def post_cancel_order(
    manager: SessionManager,
    order_id: int = Form(...),
):
    """Cancel one of your resting orders (204 also when it has just filled); 404 for someone else's order."""
    async with manager._lock:
        canceled = manager.cancel_order_by_id(order_id)
        someone_elses = canceled is None and manager.book.get_order(order_id) is not None
    if someone_elses:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That order is not one of yours")
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
    elif action == "reset":  # in a playback, back to the start of the recording
        if manager.is_running:
            await manager.stop_sim_loop()
        async with manager._lock:
            if manager.playback is not None:
                manager.rewind_playback()
            else:
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


# ---- The Pairs view: two order books whose gap mean-reverts, and pairs traders (app/pairs.py) ----

def get_pairs(manager: SessionManager) -> PairsSession:
    """The session's two-book market, made the first time it is needed."""
    if manager.pairs is None:
        manager.pairs = PairsSession()
    return manager.pairs


Pairs = Annotated[PairsSession, Depends(get_pairs)]
PAIRS_PANELS = {
    "pairs_status": "components/pairs_status.html",
    "pairs_books": "components/pairs_books.html",
    "pairs_board": "components/pairs_board.html",
}


def render_pairs_panels(ctx: dict[str, Any]) -> dict[str, str]:
    """The Pairs view's pushed panels; called on a worker thread with plain data (PairsSession.context)."""
    return {name: templates.get_template(path).render(p=ctx) for name, path in PAIRS_PANELS.items()}


def pairs_state(pairs: PairsSession) -> dict[str, Any]:
    return {"running": pairs.is_running, "speed": pairs.speed, "version": pairs.version,
            "traders": pairs.chart_traders(), "settings": pairs.settings, "changes": pairs.changes,
            "fitting": pairs.fitting}


def render_pairs_fit(pairs: PairsSession) -> str:
    """The last fit's panel, or nothing before the first."""
    fit = pairs.last_fit
    if fit is None:
        return ""
    seed_note = ("Runs of one seed start alike, but the first trade sends each its own way, so that difference carries "
                 "both runs' luck; it allows for that, not for having picked the best of "
                 f"{len(fit.candidates)}, so expect less from it afterwards.")
    return templates.get_template("components/fit_runs_result.html").render(
        prefix="pairs", fit=fit, kind=PAIRS_TRADER_KINDS[fit.kind], objective_label=FIT_OBJECTIVES[fit.objective],
        same_market=same_market(fit.settings, pairs.settings), seed_note=seed_note)


def fit_panel(session: TrainedMarket, event: str, render: Callable[[Any], str]) -> Callable[[], list[str]]:
    """What a view's stream sends besides its panels (``market_stream``'s ``more``): the last fit's panel, as ``event``,
    on connect and when it changes (apart from the panels, so that a push never takes its buttons)."""
    seen = 0

    def panel() -> list[str]:
        nonlocal seen
        if session.fit_changes == seen:
            return []
        seen = session.fit_changes
        return [format_sse(event, render(session))]

    return panel


def market_stream(request: Request, manager: LOBManager, session: Any, prefix: str,
                  panels: Callable[[dict[str, Any]], dict[str, str]], state: Callable[[], dict[str, Any]],
                  more: Optional[Callable[[], list[str]]] = None) -> StreamingResponse:
    """Server-Sent Events for a view with a market of its own, which a tab holds in place of /events while it is shown.

    Event types, each name after ``prefix``: ``hello`` (the simulation's instance ID, as on /events) and ``notices``
    (as on /events, which is closed meanwhile) have none; ``history`` (every sample kept and the traders: on connect
    and after each reset), ``samples`` (the samples and marks since the last push), ``state`` (for the controls of every
    tab), and one per panel (rendered HTML, named by ``panels``), at most five times a second while running.
    ``panels`` renders them from ``session.context()``, which is taken here on the event loop, where the market runs,
    and rendered on a worker thread. ``more`` gives any other events to send at a wake-up.
    """
    queue = manager.subscribe(session.subscribe())  # either market's news wakes it, for the notices
    session_id = request.state.session_id

    async def event_generator():
        version: Optional[int] = None
        last_t: Optional[int] = None
        marks_seen = 0
        changes_seen: Optional[int] = None
        clock_seen: Optional[float] = None
        last_panel_push = 0.0
        notices_seen = manager.notice_seq  # show only notices from now on
        first = True
        try:
            yield format_sse("hello", json.dumps({"instance": manager.instance_id}))
            while True:
                if not first:
                    if await request.is_disconnected():
                        break
                    try:
                        await asyncio.wait_for(queue.get(), timeout=15.0)
                        while not queue.empty():
                            queue.get_nowait()
                    except asyncio.TimeoutError:
                        registry.touch(session_id)
                        yield ": ping\n\n"  # keep the connection alive
                        continue
                registry.touch(session_id)

                if manager.notice_seq != notices_seen:
                    news = [n.text for n in manager.notices if n.seq > notices_seen]
                    notices_seen = manager.notice_seq
                    if news:
                        yield format_sse("notices", json.dumps(news))

                if session.version != version:  # on connect, and after a reset: the chart starts over
                    version = session.version
                    history = session.history()
                    last_t = history["samples"][-1]["t"] if history["samples"] else None
                    marks_seen = history["mark_seq"]
                    yield format_sse(f"{prefix}_history", json.dumps(history))
                else:
                    samples, marks = session.samples_since(last_t), session.marks_since(marks_seen)
                    if samples or marks:
                        if samples:
                            last_t = samples[-1]["t"]
                        marks_seen = session.mark_seq
                        yield format_sse(f"{prefix}_samples", json.dumps({"version": version, "samples": samples,
                                                                           "marks": marks}))

                if more is not None:
                    for event in more():
                        yield event

                # Panels: only if this market moved (the other's news wakes this stream too), at most five
                # pushes a second while running, and at once after any other change.
                now = time.monotonic()
                if not first and session.changes == changes_seen and (
                        session.market.time == clock_seen
                        or (session.is_running and now - last_panel_push < PANEL_PUSH_INTERVAL)):
                    continue
                last_panel_push, clock_seen = now, session.market.time
                if session.changes != changes_seen:
                    changes_seen = session.changes
                    yield format_sse(f"{prefix}_state", json.dumps(state()))
                rendered = await asyncio.to_thread(panels, session.context())
                for name, html in rendered.items():
                    yield format_sse(name, html)
                first = False
        finally:
            session.unsubscribe(queue)
            manager.unsubscribe(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                                      "X-Accel-Buffering": "no"})


@app.get("/pairs/events")
async def pairs_events(request: Request, manager: SessionManager, pairs: Pairs):
    """The Pairs view's stream (market_stream), with ``pairs_fit``, the last fit's panel: on connect, and when it
    changes (apart from the panels, so that a push never takes its buttons)."""
    return market_stream(request, manager, pairs, "pairs", render_pairs_panels, lambda: pairs_state(pairs),
                         fit_panel(pairs, "pairs_fit", render_pairs_fit))


@app.post("/api/pairs/control")
async def post_pairs_control(
    pairs: Pairs,
    action: Literal["play", "pause", "toggle", "step", "reset"] = Form(...),
    units: float = Form(1.0, gt=0, le=PAIRS_MAX_STEP, allow_inf_nan=False),
):
    """Start, pause, run on by some units of time, or reset (which pauses, as on the Trade view)."""
    if action == "play" or (action == "toggle" and not pairs.is_running):
        await pairs.start()
    elif action in ("pause", "toggle"):
        await pairs.stop()
    elif action == "step":
        await pairs.advance(units)
    elif action == "reset":
        if pairs.is_running:
            await pairs.stop()
        async with pairs.lock:
            pairs.reset()
        await pairs.broadcast()
    return {"running": pairs.is_running, "time": pairs.market.time}


@app.post("/api/pairs/speed")
async def post_pairs_speed(pairs: Pairs, speed: int = Form(..., ge=1, le=PAIRS_SPEED_MAX)):
    pairs.speed = speed
    pairs.changes += 1
    await pairs.broadcast()
    return {"speed": speed}


@app.post("/api/pairs/config")
async def post_pairs_config(request: Request, pairs: Pairs):
    """New settings for the two books: the market starts over with them (running on if it was)."""
    values, errors = parse_pairs(PAIRS_MARKET_PARAMS, await request.form())
    if errors:
        return JSONResponse({"errors": errors}, status_code=422)
    async with pairs.lock:
        pairs.configure(values)
    await pairs.broadcast()
    n = len(pairs.runners)
    kept = f" and its {n} trader{'' if n == 1 else 's'} start" if n else " starts"
    return {"message": f"Applied: the market{kept} over from time 0."}


def started_message(runner: PairsRunner, learned: int) -> str:
    warming = runner.trader.warming
    if warming is None:
        return f"{runner.name} learned from the chart's last {learned:,} points and trades from now on."
    if learned:
        return (f"{runner.name} learned from the chart's {learned:,} point{'' if learned == 1 else 's'}; it trades "
                f"once it has seen {warming[1] - warming[0]:,} more.")
    return f"{runner.name} is watching: it trades once it has seen {warming[1]:,}."


@app.post("/api/pairs/traders/start")
async def post_pairs_trader(request: Request, pairs: Pairs):
    """Start a pairs trader: the kind and settings in the form, or a candidate of a fit (``fit_id``, ``rank``)."""
    form = await request.form()
    if form.get("fit_id") is not None:
        try:
            fit_id, rank = int(str(form.get("fit_id"))), int(str(form.get("rank", "")))
        except ValueError:
            return JSONResponse({"error": "A fit's candidate is a fit and a rank."}, status_code=422)
        try:
            async with pairs.lock:
                runner, learned = pairs.start_fitted(fit_id, rank)
        except (LookupError, RuntimeError) as exc:
            return JSONResponse({"error": f"Not started: {str(exc).rstrip('.')}."}, status_code=409)
    else:
        kind = form.get("kind", "zscore")
        if kind not in PAIRS_TRADER_KINDS:
            return JSONResponse({"errors": {"kind": f"No such kind of trader: {kind}."}}, status_code=422)
        values, errors = parse_pairs(PAIRS_TRADER_KINDS[kind].params, form)
        if errors:
            return JSONResponse({"errors": errors}, status_code=422)
        try:
            async with pairs.lock:
                runner, learned = pairs.start_trader(values, kind)
        except ValueError as exc:
            return JSONResponse({"errors": {"exit": str(exc)}}, status_code=422)
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
    await pairs.broadcast()
    return {"id": runner.id, "name": runner.name, "learned": learned, "message": started_message(runner, learned)}


async def fit_response(session: TrainedMarket, form: Any, kinds: dict[str, TraderKind], default: str,
                       unfit: Optional[tuple[str, str]] = None) -> Any:
    """A view's fit route (TrainedMarket.fit_trader): its form's kind, settings and what to search, and the answer.

    With ``unfit`` (a setting and what to say of it), settings the kind refuses are turned down as that setting's
    error; otherwise the session's plan says so. What the market lacks of what the kind trades (``lacks``) comes before
    the grid's errors, since no ticks would mend it. Answers 422 saying what is wrong, 409 while another fit runs.
    """
    kind = form.get("kind", default)
    if kind not in kinds:
        return JSONResponse({"errors": {"kind": f"No such kind of trader: {kind}."}}, status_code=422)
    spec = kinds[kind]
    if spec.no_fit:  # a kind that is not fitted, whatever else is wrong with the request: the panel says why
        return JSONResponse({"error": f"Not fitted: {spec.no_fit}."}, status_code=422)
    base, errors = parse_pairs(spec.params, form)
    more, more_errors = parse_pairs((FIT_SPAN, FIT_RUNS, FIT_SEED), form)
    errors |= more_errors
    objective = str(form.get("objective", "pnl"))
    if objective not in FIT_OBJECTIVES:
        errors["objective"] = f"Best by one of {', '.join(FIT_OBJECTIVES.values())}."
    if unfit is not None and not errors and not spec.valid(base):
        errors[unfit[0]] = unfit[1]
    if not errors and (lacks := session.lacks(kind, base)) is not None:  # before the grid: no ticks would mend a market
        return JSONResponse({"error": f"Not fitted: {lacks}."}, status_code=422)
    grid: dict[str, list[Any]] = {}
    try:
        grid = parse_grid(spec.numbers, form)
    except GridError as exc:
        errors[f"grid_{exc.name}" if exc.name else "fit"] = f"{str(exc)[0].upper()}{str(exc)[1:]}."
    if errors:
        return JSONResponse({"errors": errors}, status_code=422)
    try:
        fit = await session.fit_trader(kind, base, grid, more["span"], more["runs"], objective, more["fit_seed"])
    except FitCanceled:
        return {"canceled": True, "message": "Canceled: nothing was kept of those runs."}
    except ValueError as exc:
        return JSONResponse({"error": f"Not fitted: {exc}."}, status_code=422)
    except RuntimeError as exc:
        return JSONResponse({"error": f"Not fitted: {exc}."}, status_code=409)
    tried = len(fit.candidates)
    each = f"{fit.runs} run{'' if fit.runs == 1 else 's'} each"
    return {"id": fit.id, "verdict": fit.verdict,
            "message": f"Tried {tried} setting{'' if tried == 1 else 's'}, {each}, in {fit.seconds:.0f} s: "
                       "the results are above the traders."}


@app.post("/api/pairs/traders/fit")
async def post_pairs_fit(request: Request, pairs: Pairs):
    """Fit a pairs trader over training runs (PairsSession.fit_trader), holding the request until they are done.

    The form has the trader's ``kind`` and settings, the settings to search (``fit_<name>``) with their values
    (``grid_<name>``, separated by commas), each run's length after its warm-up (``span``), how many runs each
    gets (``runs``), the ``objective`` (``pnl`` or ``sharpe``) and ``fit_seed``. The candidates come by the event
    stream (``pairs_fit``), to every tab. Answers 422 saying what is wrong, 409 while another fit runs.
    """
    return await fit_response(pairs, await request.form(), PAIRS_TRADER_KINDS, "zscore",
                              ("exit", "The flat level must be below the entry level."))


@app.post("/api/pairs/traders/fit/cancel")
async def post_pairs_fit_cancel(pairs: Pairs):
    """Stop the fit that is running, if one is."""
    return {"canceled": pairs.cancel_fit()}


@app.post("/api/pairs/traders/{runner_id}/{action}")
async def post_pairs_trader_action(pairs: Pairs, runner_id: int, action: Literal["stop", "resume", "remove"]):
    runner = pairs.runner(runner_id)
    if runner is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such pairs trader")
    async with pairs.lock:
        if action == "remove":
            pairs.remove(runner)
        else:
            pairs.set_active(runner, action == "resume")
    await pairs.broadcast()
    return {"id": runner_id, "action": action}


# ---- The Stat-arb view: many books with several common factors (app/statarb.py) ----

def get_statarb(manager: SessionManager) -> StatArbSession:
    """The session's many books, made the first time they are needed."""
    if manager.statarb is None:
        manager.statarb = StatArbSession()
    return manager.statarb


StatArb = Annotated[StatArbSession, Depends(get_statarb)]
STATARB_PANELS = {
    "statarb_status": "components/statarb_status.html",
    "statarb_books": "components/statarb_books.html",
    "statarb_board": "components/statarb_board.html",
}


def render_statarb_panels(ctx: dict[str, Any]) -> dict[str, str]:
    """The Stat-arb view's pushed panels; called on a worker thread with plain data (StatArbSession.context)."""
    return {name: templates.get_template(path).render(p=ctx) for name, path in STATARB_PANELS.items()}


def render_statarb_fit(statarb: StatArbSession) -> str:
    """The last fit's panel, or nothing before the first."""
    fit = statarb.last_fit
    if fit is None:
        return ""
    seed_note = ("Runs of one seed share their factors and shocks, whatever trades, though the order flow parts at the "
                 "first trade, so comparing on the same seeds takes out some of the luck. The standard error, measured "
                 "from the differences, says how much is left, but not for having picked the best of "
                 f"{len(fit.candidates)}, so expect less from it afterwards.")
    return templates.get_template("components/fit_runs_result.html").render(
        prefix="statarb", fit=fit, kind=STATARB_TRADER_KINDS[fit.kind], objective_label=FIT_OBJECTIVES[fit.objective],
        same_market=same_market(fit.settings, statarb.settings), seed_note=seed_note)


@app.get("/statarb/events")
async def statarb_events(request: Request, manager: SessionManager, statarb: StatArb):
    """The Stat-arb view's stream (market_stream), with ``statarb_fit``, the last fit's panel (as on /pairs/events)."""
    def state() -> dict[str, Any]:
        return {"running": statarb.is_running, "speed": statarb.speed, "version": statarb.version,
                "traders": statarb.chart_traders(), "settings": statarb.settings, "changes": statarb.changes,
                "fitting": statarb.fitting}

    return market_stream(request, manager, statarb, "statarb", render_statarb_panels, state,
                         fit_panel(statarb, "statarb_fit", render_statarb_fit))


@app.post("/api/statarb/control")
async def post_statarb_control(
    statarb: StatArb,
    action: Literal["play", "pause", "toggle", "step", "reset"] = Form(...),
    units: float = Form(1.0, gt=0, le=STATARB_MAX_STEP, allow_inf_nan=False),
):
    """Start, pause, run on by some units of time, or reset (which pauses, as on the Trade view)."""
    if action == "play" or (action == "toggle" and not statarb.is_running):
        await statarb.start()
    elif action in ("pause", "toggle"):
        await statarb.stop()
    elif action == "step":
        await statarb.advance(units)
    elif action == "reset":
        if statarb.is_running:
            await statarb.stop()
        async with statarb.lock:
            statarb.reset()
        await statarb.broadcast()
    return {"running": statarb.is_running, "time": statarb.market.time}


@app.post("/api/statarb/speed")
async def post_statarb_speed(statarb: StatArb, speed: int = Form(..., ge=1, le=STATARB_SPEED_MAX)):
    statarb.speed = speed
    statarb.changes += 1
    await statarb.broadcast()
    return {"speed": speed}


@app.post("/api/statarb/config")
async def post_statarb_config(request: Request, statarb: StatArb):
    """New settings for the books: the market starts over with them (running on if it was)."""
    values, errors = parse_market(await request.form())
    if errors:
        return JSONResponse({"errors": errors}, status_code=422)
    try:
        async with statarb.lock:
            statarb.configure(values)
    except ValueError as exc:
        return JSONResponse({"error": f"Not applied: {exc}."}, status_code=422)
    await statarb.broadcast()
    n = len(statarb.runners)
    kept = f" and its {n} trader{'' if n == 1 else 's'} start" if n else " starts"
    fair = statarb.market.fair
    return {"message": f"Applied: {fair.n} books, {fair.rank} relations. The market{kept} over from time 0."}


def statarb_started(runner: StatArbRunner, learned: int) -> str:
    warming = runner.trader.warming
    if runner.kind == "flow":  # it learns the mids its factors use, but not the flow, which it sees from now on
        left = warming[1] - warming[0]
        points = f"{learned:,} point{'' if learned == 1 else 's'}"
        took = f"took in {points} of the past mids for its factors, but " if learned else ""
        return (f"{runner.name} {took}sees the order flow only from now on: it chooses once it has seen "
                f"{left:,}{' more' if learned else ''} decision{'' if left == 1 else 's'}.")
    if warming is None and not learned:  # a trader with no past to learn from: the deals kind, with no switch for it
        return f"{runner.name} has nothing to learn from the past: it trades from its next decision."
    if warming is None or warming[0] >= warming[1]:
        return f"{runner.name} learned from the last {learned:,} points and trades from now on."
    does = "trades" if runner.kind in STATARB_NOTHING_TO_CHOOSE else "chooses"  # the index trader's relation is known
    if learned:
        return (f"{runner.name} learned from {learned:,} point{'' if learned == 1 else 's'} of the past; it {does} "
                f"once it has seen {warming[1] - warming[0]:,} more.")
    return f"{runner.name} is watching: it {does} once it has seen {warming[1]:,}."


@app.post("/api/statarb/traders/start")
async def post_statarb_trader(request: Request, statarb: StatArb):
    """Start a trader: the kind and settings in the form, or a candidate of a fit (``fit_id``, ``rank``)."""
    form = await request.form()
    if form.get("fit_id") is not None:
        try:
            fit_id, rank = int(str(form.get("fit_id"))), int(str(form.get("rank", "")))
        except ValueError:
            return JSONResponse({"error": "A fit's candidate is a fit and a rank."}, status_code=422)
        try:
            async with statarb.lock:
                runner, learned = statarb.start_fitted(fit_id, rank)
        except (LookupError, RuntimeError) as exc:
            return JSONResponse({"error": f"Not started: {str(exc).rstrip('.')}."}, status_code=409)
        except ValueError as exc:  # settings that no longer suit the books
            return JSONResponse({"error": f"Not started: {exc}."}, status_code=422)
    else:
        kind = form.get("kind", "pairs")
        if kind not in STATARB_TRADER_KINDS:
            return JSONResponse({"errors": {"kind": f"No such kind of trader: {kind}."}}, status_code=422)
        values, errors = parse_pairs(STATARB_TRADER_KINDS[kind].params, form)
        if errors:
            return JSONResponse({"errors": errors}, status_code=422)
        try:
            async with statarb.lock:
                runner, learned = statarb.start_trader(values, kind)
        except ValueError as exc:
            return JSONResponse({"error": f"Not started: {exc}."}, status_code=422)
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
    await statarb.broadcast()
    return {"id": runner.id, "name": runner.name, "learned": learned, "message": statarb_started(runner, learned)}


@app.post("/api/statarb/traders/fit")
async def post_statarb_fit(request: Request, statarb: StatArb):
    """Fit a trader over training runs (StatArbSession.fit_trader), as ``/api/pairs/traders/fit`` does."""
    return await fit_response(statarb, await request.form(), STATARB_TRADER_KINDS, "pairs")


@app.post("/api/statarb/traders/fit/cancel")
async def post_statarb_fit_cancel(statarb: StatArb):
    """Stop the fit that is running, if one is."""
    return {"canceled": statarb.cancel_fit()}


@app.post("/api/statarb/traders/{runner_id}/{action}")
async def post_statarb_trader_action(statarb: StatArb, runner_id: int, action: Literal["stop", "resume", "remove"]):
    runner = statarb.runner(runner_id)
    if runner is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such trader")
    async with statarb.lock:
        if action == "remove":
            statarb.remove(runner)
        else:
            statarb.set_active(runner, action == "resume")
    await statarb.broadcast()
    return {"id": runner_id, "action": action}


@app.post("/api/statarb/test")
async def post_statarb_test(statarb: StatArb):
    """What the Johansen test and Engle-Granger find in the mids kept: it comes back on the status panel too."""
    return await statarb.test_relations()
