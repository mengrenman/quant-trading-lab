# Project Handoff: Limit Order Book (LOB) Simulation

## 1. Executive Summary

This project is an interactive, educational **Limit Order Book (LOB) Simulation & Visualization Platform** built in Python and modern web technologies. It is based on the foundational research paper:

> **Gould, M.D., Porter, M.A., Williams, S., McDonald, M., Fenn, D.J. and Howison, S.D.** (2013).  
> *Limit order books.* Quantitative Finance, 13(11), 1709–1742.  
> [doi:10.1080/14697688.2013.803148](https://doi.org/10.1080/14697688.2013.803148) | [arXiv:1012.0349](https://arxiv.org/abs/1012.0349)

The platform pairs a pure-Python matching engine and stochastic market simulation models with nine teaching notebooks and a real-time web dashboard (FastAPI, Server-Sent Events, HTMX, Tailwind CSS, and TradingView Lightweight Charts). It lets researchers and students study market microstructure, order queues, emergent bid-ask spreads, and stylized facts (heavy tails, volatility clustering, price impact) through interactive experimentation.

The project is under git; Section 6 lists the changes since the baseline snapshot of 2026-11-01.

---

## 2. System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Web Browser Frontend                            │
│  - Tailwind CSS, four themes   - TradingView Lightweight Charts        │
│  - One EventSource per tab     - Trade, Analyze and Learn views        │
│  - DOM ladder, depth chart     - Heatmap canvas, guided lessons        │
│  - One-click trading, drag     - Model parameter panel (HTMX)          │
└───────────────────────────────────▲────────────────────────────────────┘
                                    │  /events: panels (HTML) + bars (JSON)
┌───────────────────────────────────┴────────────────────────────────────┐
│                          FastAPI Web Layer                             │
│  - app/main.py: Routes, validation, the event stream, threadpool       │
│  - app/sessions.py: One simulation per browser session (cookie)        │
│  - app/manager.py: LOBManager, simulation clock, trade-based bars      │
│  - app/model_specs.py: Dashboard models, parameters and ranges         │
└───────────────────────────────────▲────────────────────────────────────┘
                                    │
┌───────────────────────────────────┴────────────────────────────────────┐
│                       Core Simulation & Engine                         │
│  - lob/book.py: Limit order book, validation, cached depths            │
│  - lob/matching.py: PriceTimeEngine (FIFO) & ProRataEngine             │
│  - lob/models/: ZI, Hawkes ZI, two Mike-Farmer variants, Parlour rule  │
│  - lob/simulation.py: Event loop, event log, per-fill trade log        │
└────────────────────────────────────────────────────────────────────────┘
```

### Key Modules & Files

- **`lob/book.py`**:
  - `OrderBook`: Dual-sided order book with tick size and lot size validation.
  - Price levels live in `SortedDict`s (bid keys negated so iteration runs best-first); each level is a FIFO `deque`.
  - O(1) order lookup, depth-at-price queries and uniform random order sampling. Canceling removes the order from its level's queue, which is linear in that queue's length.
  - `validate(order)` checks tick size, lot size and order-ID uniqueness. Both engines call it *before* matching, so a rejected order never leaves the book half-changed.
  - `orders_at(price, side)` returns a level's queue in priority order.
  - Iceberg orders (`display_size`): while an order rests, `size` is its visible part, which is all that depths, depth profiles, snapshots and matching see, and `hidden` is the reserve (`hidden_depth_at_price` sums reserves at a price). An arriving iceberg trades its full size and is split when it comes to rest.
- **`lob/matching.py`**:
  - `PriceTimeEngine`: Price-time priority (FIFO within each price level).
  - `ProRataEngine`: Allocates proportionally to resting order sizes in whole lots, rounding down and handing leftover lots out in time priority, so fills and remainders stay multiples of the lot size.
  - Engines reduce resting orders' `size` in place as they fill.
  - When an iceberg's peak is filled, the next comes from its reserve and joins the back of its price level (losing time priority), so an incoming order can fill more than the visible depth; pro-rata shares follow visible sizes and the level is allocated again after refills. Ordinary orders behave exactly as before (every model's output hashes identically).
- **`lob/models/`** (all models continue from the caller's clock when it is ahead of their own):
  - `zero_intelligence.SmithFarmerModel`: Zero-intelligence model of Daniels et al. (2003) / Smith et al. (2003), Gould et al. Section 5.2.4. Limit orders land uniformly on the `price_range` ticks next to the *opposite* best quote, so they fall inside the spread with probability proportional to its width; market orders and per-order cancellations are Poisson. `_sample_side` / `_sample_size` hooks let ablations change signs and sizes before an order is priced.
  - `hawkes.HawkesZIModel`: The ZI model with market orders arriving as a self-exciting (Hawkes) process. The baseline rate is scaled to `mu * (1 - branching)` so the long-run arrival rate stays `mu`; every arrival excites, even one that finds the opposite side empty. Simulated exactly by thinning.
  - `mike_farmer.MikeFarmerModel`: Simplified Mike & Farmer (2008): power-law relative prices measured from the mid-price (always at least one tick), long-memory order signs, log-normal sizes, constant-rate cancellation.
  - `mike_farmer.MikeFarmer2008Model`: Mike and Farmer's structure: prices set relative to the same-side best quote with Student's t offsets (orders can land inside the spread or cross it and execute as effective market orders; no separate market order process), long-memory signs, and state-dependent cancellation simulated by thinning. Parameter values are illustrative, not their LSE calibration.
  - `long_memory.LongMemorySigns`: A continuous stream of long-memory signs (signs of FARIMA(0, H − 1/2, 0) noise, applied by overlap-save FFT convolution) whose memory never resets.
  - `strategic.ParlourModel`: A hand-tuned rule inspired by Parlour (1998): the probability of a market order rises with a narrow spread and a long queue at the best quotes. It is not a game-theoretic equilibrium model.
- **`lob/simulation.py`**: `Simulation.run()` returns the event log as a DataFrame (stored internally in typed arrays: a 400k-event run takes about 190 MB); `Simulation.trades` has one row per fill; `Simulation.snapshots` holds book snapshots. The log's `bid_size` and `ask_size` are the visible depth at the best quotes.
- **`lob/calibration.py`**: maximum-likelihood estimates of every `MikeFarmer2008Model` parameter from an event log that starts from an empty book (arrival rate; rounded Student's t offsets; the signs' Hurst exponent; the state-dependent cancellation rate and imbalance offset, from replaying the log through a book). Notebook 08 checks them by recovering known parameters.
- **`lob/lobster.py`**: LOBSTER (Nasdaq) data: `read_lobster`/`save_lobster` for the message and order-book files, `lobster_orders` (messages to orders: executions at one time against one side are one aggressive order, a new order of its side at the same time is its resting part), and `calibrate_lobster`, the same estimates allowing for what such data hold back. An order that executes in full leaves no limit price, only a bound (its worst fill), so the price fit treats its offset as censored (`fit_rounded_t_censored`, exact for the model); the cancellation replay follows only orders the file shows placed, with quotes from the order-book file, after a burn-in (`last_unknown` says how long one needs). `simulated_lobster` writes the model's runs in LOBSTER's format, hiding what Nasdaq's feed hides.
- **`lob/impact.py`**: a metaorder's impact against a twin run from the same book, model and random state that sends nothing (common random numbers): `warmed_market`, `paired_metaorders` (notebook 09).
- **`lob/metrics.py`**, **`lob/analysis/`**: returns (`event_time_returns`), per-order VWAP (`trades_by_order`), virtual price impact on book snapshots, ACF with white-noise bands, rank-based volatility ACF, Hurst (R/S), Hill and power-law estimators, time-averaged depth profiles, order flow imbalance (`order_flow_imbalance`, Cont, Kukanov and Stoikov 2014) with `ofi_regression`, and plotting helpers.
- **`app/sessions.py`**: `SessionRegistry` holds one `LOBManager` per browser session, found through a random ID in an HTTP-only, SameSite=Lax cookie (tabs of one browser share it). IDs are issued only by the server. Sessions without an open stream are evicted after 30 minutes idle, and the least recently seen one beyond 50 sessions; eviction stops the session's simulation.
- **`app/manager.py`**:
  - `LOBManager`: One session's book, engine, model, parameters, seed, background runner and manual orders.
  - One simulation clock (`sim_time`): model events advance it by the model's own event times; each manual action advances it by `MANUAL_ACTION_DT` (0.5).
  - Executed trades are stored as one-unit bars (`deque(maxlen=100_000)`); `get_candles()` rebuilds OHLCV bars, with buy and sell volume, for any interval.
  - `reset()` restarts the random generator from the session's seed, so the same seed and parameters replay a run exactly.
  - Every move of the clock goes through `_advance`, which starts an order-book heatmap column (`HeatColumn`) at each whole unit of time it reaches: the visible depth 30 ticks either side of the mid just before the event that reached it (bids positive, asks negative), the user's iceberg reserves, and the lots traded during the unit. The most recent 600 are kept; none are taken during a collection, after which the heatmap starts over (as after a reset, with a new `heat_version`).
  - The user's fills, as maker or taker (a fill against their own order counts on both sides), are kept per unit of time (`OwnBar`); `own_trade_marks` aggregates them to the chart's interval for its markers.
  - `move_orders` moves the user's orders at one price to another as one manual action (each canceled and entered again with what it had left and the same iceberg peak, in its old queue order); `cancel_all_orders` cancels theirs, optionally on one side or at one price. `cancel_order_by_id` cancels only the user's own orders.
  - `Account` books a trader's fills by average cost (realized P&L on reducing a position, the rest marked to the mid) and scores each fill against the mid before it (edge) and 100 events later (markout, `MARKOUT_EVENTS`; in events so that models with different event rates compare like with like). The user has one (`account`); the market-making bot another (`bot_account`).
  - `MarketMaker` (settings) and `start_bot`/`stop_bot`/`_run_bot`: the user's quoting bot (see 3M). Like the user's orders, its quotes (`bot_order_ids`) are never canceled by the models.
- **`app/model_specs.py`**: The models the dashboard offers, their labels, one-line summaries, default parameters, allowed ranges and notebook 05's values; the model menu and parameter panel are generated from it.
- **`app/depth_chart.py`**: the depth chart under the order book: cumulative visible depth from the mid outward as SVG paths, over the same number of ticks either side of the mid (to the 30th level of whichever side reaches further, 10 to 100 ticks).
- **`app/facts.py`**: Live stylized facts. `FactsRecorder` samples each session's event stream as notebook 05 samples a `Simulation`; `summarize` computes the notebook's statistics with single-run standard errors (see 3F).
- **`app/main.py`**: FastAPI endpoints (`/api/sim/*`, `/api/order/*` including `move` and `cancel_all`, `/api/bot/start` and `/api/bot/stop`, `/api/chart/*`, `/api/engine`, `/api/candlesticks` with the user's fills per bar (and running position and cash, for the P&L line), `/api/heatmap`, `/api/export/*`, `/api/snapshot`) with validated inputs (invalid values return 422), and the `/events` stream (panels, bars, the user's fills per bar, heatmap columns, `ticks` (the book at every tick near the mid, from `ticks_view`, for the tick view), controls and notices). Templates render on worker threads from copied values only.
- **`app/templates/`**: `index.html` (the views, the layout and the event-stream script), `components/chart.html` (trade-based candlesticks with stacked buy/sell volume, the VWAP, and the user's orders and fills), `ladder.html` (the order book by levels; `ladder_mode.html` its Levels/Ticks switch, `ticks_ladder.html` the tick view's frame), `depth_chart.html`, `pnl.html` (your P&L), `bot_card.html` and `bot.html` (the market maker's settings and status), `model_params.html` (parameter panel), `facts.html` (stylized facts), `ofi.html`, `heatmap.html`, `lessons.html`, `guide.html` (how to read the page), `order_result.html`, `sim_controls.html`, `play_button.html`, `sim_badge.html`, `engine_toggle.html`, `order_form.html`, `active_orders.html`, `trades.html`, `metrics.html`.
- **`app/static/js/`**: the page's own scripts (the rest are vendored): `lob-ticks.js` (the order book's tick view), `lob-ladder.js` (one-click trading, drag-to-move and the keyboard, in both views of the order book), `lob-heatmap.js` (draws the heatmap) and `lob-lessons.js` (the lessons and their dock).
- **`scripts/build_themes.py`**: generates `app/static/css/themes.css`, the color themes, as CSS variables (see 3I).

---

## 3. Web Dashboard Features

### A. Sessions and Live Updates
1. **One simulation per browser.** Each browser gets its own simulation; tabs of the same browser share it.
2. **One event stream per tab** carries the panels, the chart and the controls. Browsers allow about six connections per host over HTTP/1.1, so a tab used to need two streams and three tabs froze the page; four tabs now work.
3. **Tabs stay in sync**: a change to the model, parameters, seed, engine or speed in one tab appears in the session's other tabs. A tab that already shows the change keeps its "Applied" message, a parameter form someone is typing in updates once focus leaves it, and the speed slider is not moved while dragged. A panel with a request on its way (the parameter panel, the engine toggle) is not replaced by a push until the request ends, or the reply would have nowhere to go.
4. **Connection indicator** in the header: Connecting, Live or Reconnecting. If the stream comes back attached to a different simulation (server restart or expired session), the page reloads itself.

### B. Models and Parameters
1. Five models plus manual placement: Smith et al. ZI, ZI with self-exciting market orders, Mike & Farmer simplified, Mike & Farmer (2008 structure), Parlour-inspired rule.
2. **Model parameters panel**: a one-line summary of the current model (shown in the collapsed bar), every parameter plus the random seed. Apply validates every field and either applies all of them (continuing from the current book) or changes nothing and shows per-field messages. Parameters are kept per model.
3. **Reset Book** restarts from the seed, so the same seed and parameters replay the same run.

### C. Candlestick Chart & Volume Dual-Panel
1. **Trade-based bars on the simulation clock**: bars aggregate executed trades only. Intervals without trades are drawn flat at the last trade price (the mid-price before the first trade) with zero volume. There is no synthetic history.
2. **Intervals**: `1s` to `5m` in units of simulation time (each model's own clock, not wall-clock time). Changing the interval rebuilds every bar from the stored trades.
3. **Stacked volume**: green is buy-initiated volume, red sell-initiated.
4. **Zoom**: horizontal zoom from 2 to 32 px per bar with presets, in the toolbar between the two panels. `Fit` shows every bar (at most 32 px wide) and also recenters both charts vertically, switching their price axes back to auto-scaling after a drag. Both panels stay horizontally aligned.
5. **Whole-tick prices**: the price axis and the OHLC readout show whole ticks (102, not 102.00).
6. **Session VWAP**: an amber line on the price chart, and in the OHLC readout to two decimals. Each candle carries the VWAP at its close; bars keep the session's running volume and turnover, so the value survives truncation. No technical indicators: the models' prices are built to be unpredictable.
7. **Your orders and fills**: a dashed line in its side's color at each price where the user has orders resting ("you buy 40", "you sell 10 +90" for an iceberg's hidden rest), and an arrow on each bar in which they traded, as maker or taker: up below the bar for buying, down above it for selling, with the lots and average price. The markers follow interval changes; a small legend appears once there are any.

### D. Depth of Market, Controls and Performance
1. Fixed-height DOM with 10 ask and 10 bid slots, padded so the layout never shifts; mid-price and spread banner pinned in the middle. Levels holding the user's resting orders show "you N".
2. **Trading from the book**: a click on a level copies its price into the order form. With **1-click** on (the toggle in the DOM header, which rings the card in amber), it places a limit order there instead, for the size in the limit form: a buy on a bid level, a sell on an ask level, reported in a notice. Dragging a "you" badge onto another level moves those orders there (`POST /api/order/move`): they go to the back of the new level's queue, as a price change does on an exchange, and a price across the spread trades on arrival. The ladder lists only levels with orders and is pushed five times a second, so its rows shift as levels empty and fill: the book holds still while a level is pressed or dragged, and while the pointer aims at it with one-click on (refreshing every two seconds), so the level clicked is the level seen.
3. **Depth chart** under the order book (a panel of its own): the visible lots bid and offered, added up from the mid outward, as two step curves over the same number of ticks either side of the mid (to the 30th level of whichever side reaches further, always taking in both best quotes). A large resting order is a cliff in its side's curve; the user's orders are marked on its price axis.
3a. **Tick view**: Levels | Ticks in the book's header switches to a row for every price 40 ticks either side of the mid, bids left and asks right, fixed in place and updated where they are (`lob-ticks.js`, from the stream's `ticks` events). One-click trading and dragging work at any price there, empty ones included; the rows are built around the mid again by Center, or by themselves once the mid has moved 20 ticks, never while the pointer is on them. The choice is kept in this browser.
3b. **Keyboard**: the book takes the focus with Tab; the arrows move a cursor over its prices (and between bids and asks in the tick view), Enter does what a click would there, M picks up your orders at the cursor (arrows choose a price, Enter puts them there, Escape leaves them). The hint says what Enter or M would do; screen readers follow the cursor (`aria-activedescendant`). The right arrow does not step the simulation while the book has focus.
4. One compact control card: Play/Pause (optimistic), Step, Reset and a Running/Paused status on one row with the model menu and the engine toggle (switches on the server and shows the engine in use). At 1440x900 both charts are visible without scrolling.
5. Speed from 1 to 2000 events per second (default 50) on a logarithmic slider; the preset matching the current speed is highlighted.
6. Keyboard shortcuts: Space starts or pauses, the right arrow steps one event (Shift: ten), F fits the chart, 1 to 3 switch views. They are off while typing in a field.
7. Template rendering runs on worker threads; panels are pushed at most five times a second while running; the chart receives every bar since its last push.

### E. Order Entry
1. Limit and market orders. The Bid, Mid and Ask quick-price buttons follow the book (Mid rounds down for a buy, up for a sell).
2. Every order reports what happened under the button: resting as order #id, filled at what prices (with the average when it walked the book), what rests, or nothing filled. `/api/order/limit` and `/api/order/market` return that line as HTML (status 200); rejected input gets a message too.
3. The user's resting orders stay until they fill or the user cancels them. The models' cancellations stand for other traders withdrawing their own orders, so `LOBManager.step` skips one aimed at a manual order (counted as a step without an event). The models choose the order to cancel uniformly from the book, so this is exact thinning: their own orders keep their cancellation rate.
4. Manual market orders carry price protection, as on CME Globex: they walk the book at most N ticks (default 10, from 0 to 1,000, set in the form) past the best opposite price on arrival, and whatever is left rests there as the user's limit order, the paper's rule for an order that runs out of matches (Gould et al. p. 1712). Protection 0 fills only at the best price, like a market-to-limit order; with no opposite side nothing is placed. The result line says what was filled and what rests. The models' own market orders are unchanged: what they cannot fill is discarded.
5. Iceberg limit orders: an optional "show at most (lots)" field. The order book shows only the peak, marked "you 10 +90 hidden"; My Active Orders shows the reserve.
6. An order that trades is compared with the session VWAP from just before it: "Your average was 1.0% below the session VWAP of 101.00."
7. Fills of the user's resting orders are announced: a notice in each open tab ("Your buy order #583 filled 5 lots at 100, 3 left.") and the latest five under My Active Orders.
8. Split orders (TWAP): a market order can go out as N market orders, one every T units of simulation time. Each slice is protected; what it cannot fill rolls into the next. At the start the manager records what one market order for the whole size would have done (`sweep_estimate`); the summary compares the split's average with it and with the market's VWAP over the period. Slices wait while the simulation is paused; Stop ends a split.
9. **Cancel all** in My Active Orders cancels every resting order of the user's (`POST /api/order/cancel_all`, which can also take a side or a price) and leaves the models' alone.

---

### F. Stylized Facts (this run)
1. **What it measures**: notebook 05's seven facts on the session's own run: heavy tails (excess kurtosis of mid-price returns over 1,000 events, bar 1), volatility clustering (rank ACF of |r| at lag 1, bar 0.1), long memory in order signs (ACF at lag 100), bid-ask bounce (lag-1 ACF of trade-price changes), a hump-shaped depth profile, concave impact (log-log slope of virtual impact, bar 0.95) and power-law relative prices (a property of each model, shown as such).
2. **Same sampling as the notebook**: after 5,000 warm-up steps, the mid-price every 1,000 events with both sides quoted, the sign of every limit and market order, the last fill price of every aggressive order, and a book snapshot every 400 steps. A test replays a `Simulation` through the recorder and requires identical samples and statistics.
3. **Uncertainty from one run**: a moving-block bootstrap for the return statistics and batch means for signs, trades and snapshots, graded with the notebook's rule (3 SE "Yes", 2-3 SE "Weak evidence"). These errors are approximate and too small for series with long memory, so every reading is labeled "this run"; the notebook uses 32 independent seeds.
4. **Collect 400k / 2M**: runs that many steps as fast as possible (about 2 s and 10 s), in chunks so the server stays responsive; only the panel updates meanwhile, and the chart reloads afterwards. 400k events (one notebook run) often leave clustering open; 2M usually settle it.
5. **A new sample** starts when the model, its parameters, the engine or the book changes. Memory per session is capped (the most recent 500k signs, 100k trades, 2,500 snapshots).
6. **Notebook 05's values**: the parameter panel can fill in the values notebook 05 runs each model with (a test reads them from the notebook). The dashboard's defaults trade far more (about 29 lots per time unit against 1 for the notebook's ZI rates), which keeps the page lively but changes the facts: with the default rates, the Hawkes model's clustering sits on the bar (0.12 against 0.1 after 2M events), while with the notebook's values a 2M-event run gives the notebook's verdict on all seven facts.

### G. Order Flow Imbalance (this run)
1. From the same sample as the stylized facts, OFI (Cont, Kukanov and Stoikov 2014) is summed over intervals of 100 events: lots added at the bid or taken from the ask count as buying pressure.
2. A card plots the mid-price change against OFI for the latest 400 intervals, with the fitted line, the slope in ticks per lot, R2, the depth at the best quotes and the latest interval. On the default ZI model after 400k events: slope 0.031, R2 0.33, depth 29.7 lots. A test requires the recorder's OFI and price changes to equal `lob.metrics.order_flow_imbalance` on a `Simulation` replay.
3. Unlike the top-5 depth imbalance in the metrics strip, which one large resting order can dominate, OFI measures what changed at the best quotes.

### H. Export
1. `GET /api/export/{trades|bars}.{csv|parquet}`: every fill (the most recent 100,000; time, price, size, aggressor, maker and taker IDs, whether it was the user's order), or every bar at the current interval with its volume split and VWAP. Links on the trades card and under the volume chart.
2. Parquet needs `pyarrow` (the optional `[parquet]` extra, included in `[dev]`); without it the links are hidden and the endpoint answers 501.

### I. Views, Layout, Text Size and Themes
1. **Three views**, picked from tabs in the header: Trade (chart, order entry, your orders, the order book and the tape), Analyze (the heatmap, order flow imbalance and stylized facts) and Learn (lessons and the reading guide). The simulation controls and the top of the book stay in view on all three. The view is kept in the address (`#analyze`) and in this browser; keys 1 to 3 switch, and the arrow keys move along the tab list.
2. **Full width, in columns**: the Trade view takes as many columns as the window has room for: one on a phone; from 1024 px two (the chart over order entry | the book over the tape); from 1280 px three (order entry and your orders | the chart | the book and the tape); from 1600 px four, the tape in a column of its own. Wrappers dissolve (`display: contents`) when their children become columns.
3. **Text size**: Small (the default), Medium or Large in the header sets the root font size to 75%, 87.5% or 100% of the browser's own. Every size on the page is in rem, so this one value scales everything, as zooming would; the charts' text follows (11 px at Large). Large is the original look.
4. **Themes**: Midnight (the original), Daylight (light, for projectors and screenshots) and Terminal (black, amber accents, JetBrains Mono), and a color-blind-safe palette that shows buying in blue and selling in orange. Every shade of the color families the templates use is a CSS variable (`--c-<family>-<shade>`, an "R G B" triplet so opacity modifiers work), generated by `scripts/build_themes.py`; a theme is one attribute on `<html>` (`data-theme`, `data-palette`). Script-drawn colors (the charts, the heatmap) read the variables through `themeColor()` and redraw on a change. All four keep text and buy/sell colors at 4.5:1 contrast or more. Text size and theme are kept in this browser and applied before the page paints.

### J. Order-Book Heatmap (Analyze)
1. The visible depth at each price near the mid over simulation time, one column per unit of time (the most recent 600; see `HeatColumn` above), with the trades as circles and the mid-price as a line. Brightness grows with depth up to twice the median in view, then moves away from the background (toward white, or black on Daylight), so a wall stands out as a bright band the price runs into; an iceberg is a level that keeps trading while what shows of it stays small. **Reveal hidden** outlines the iceberg reserves, which the visible depth leaves out; hovering gives the numbers.
2. The page loads the history from `/api/heatmap` when the view is first shown, and again when the heatmap starts over or the stream reconnects; the stream's `heatmap` events bring the new columns (throttled like the panels). Taking columns costs about 0.07 microseconds per event.

### K. Guided Lessons (Learn)
1. Six lessons of five or six steps: price impact (a market order walking the book), walls (a large resting bid holding the price up), hidden liquidity (an iceberg trading more than the book shows), order flow imbalance (across two models), splitting a large order (TWAP against one sweep and the VWAP), and making a market (the bot's edge, markout and adverse selection, and how requoting more often costs queue priority).
2. A lesson runs in a dock in the bottom-left corner that stays across views. Each step explains something, with numbers read from the current book; **Do it** sends the same requests the page's own controls do, **Show me** switches view and outlines the part of the page the step is about (an overlay that follows the element, so pushes that replace it do not lose the outline), and steps that can check their result mark Next once done. The lesson, the step and what the steps noted are kept in this browser, so a reload resumes; finished lessons are marked done. The lessons are written in `app/static/js/lob-lessons.js`.

---

### L. Your Position and P&L
1. A Your P&L card under My Active Orders and a card in the metrics strip (so in every view): position, its average price, the realized P&L (from selling out of a position at more or less than its average price) and the unrealized part marked to the mid, fills, and the edge against the mid of the user's resting and aggressive fills. Money is in ticks times lots (one lot gaining one tick is 1); there are no fees.
2. A P&L line under the volume chart, shown once the user has traded: cash plus position at each bar's close, exact however a bar's fills were ordered. It follows the price chart and never leads (it is made hidden, and a chart without a width has no range to pass on); the three charts' price axes take the widest one's width, so their bars line up.
3. My Active Orders shows each resting order's place in its queue: the lots and orders ahead of it under price-time priority, or its share of the level under pro-rata matching.

### M. Market Maker
1. A card in the Trade view runs a quoting bot of the user's: every interval of simulation time it withdraws its quotes and posts a bid and an ask a set number of ticks from the mid, leaning both against its inventory and never across the spread; at its inventory limit it quotes only the side that reduces it. It quotes only while the simulation runs; its quotes show as "bot" in both order-book views; its fills go to its own account.
2. Its card shows inventory, P&L, its edge against the mid at each fill, the markout 100 events later, and the difference, adverse selection. Checked over 8 seeds of 60,000 events in the ZI model: requoting every 15 events instead of 58 raised adverse selection from 0.20 ± 0.05 to 0.40 ± 0.04 ticks a lot, on 262 instead of 472 lots (each new quote joins the back of its queue). An earlier comparison across models, which suggested that long-memory signs raise adverse selection, did not survive matching the requote interval in events.

## 4. Environment & Execution Guide

### Python Environment
- **Path**: `/Users/mengren/anaconda3/envs/lob-sim/bin/python` (Python 3.14.3)
- **Install**: `make install` (`pip install -e ".[dev]"`). Dependencies live in `pyproject.toml`: core, plus `[web]`, `[notebooks]` (includes `joblib`), `[parquet]` (`pyarrow`, for Parquet downloads) and `[dev]` (all of them, plus `pytest`, `httpx`, `ruff`, `pytailwindcss`, `playwright`) extras. Python ≥ 3.10.

Run `make` targets from an environment with the project installed, or pass the interpreter, e.g. `make check PYTHON=/Users/mengren/anaconda3/envs/lob-sim/bin/python`.

### Checks
```bash
make lint             # ruff
make css-check        # fails if app/static/css/app.css is out of date with the templates
make test             # pytest (249 tests, ~20 s)
make notebooks-check  # executes all nine notebooks into a temp dir; fails on the first error
make check            # all four (~5 min)
make test-browser     # drives the dashboard in Chrome with Playwright (29 tests, ~45 s); not part of make check
make test-browsers    # the same in Chrome, Firefox and WebKit (Safari's engine), ~2.5 min
```

The browser tests (`tests/browser`, marked `browser` and skipped by `make test`) start their own server on a free port and use the installed Google Chrome, or Playwright's Chromium (`python -m playwright install chromium`). Each test gets a fresh browser profile, hence a fresh session, and fails on any console error. They cover what the server-side tests cannot: playback buttons and keyboard shortcuts, the logarithmic speed slider, live quick prices and order results, own-order markers, cross-tab control sync (including a form being edited), Fit after dragging the price axis, the stylized-facts panel and notebook 05's values, the views, the themes and the text size, the columns at four window widths and on a phone, your orders and fills on the chart, one-click trading and drag-to-move (with the simulation running), the heatmap (also when it starts over during its load), guided lessons, your P&L and its line, the tick view, the market maker, and the order book from the keyboard. Tests that use the mouse on the order book first scroll its rows to the middle of the window: the sticky header covers the top; they click radio buttons' visible labels (a forced click on the hidden input misses in Firefox). `LOB_BROWSERS` (chrome, firefox, webkit) picks the engines, each from `python -m playwright install`; WebKit found two races Chrome's timing hid (see 3A.3 and 3D.3b), so run `make test-browsers` after changing the page's scripts. When driving htmx forms, wait for the swap to settle (`settle()` in the tests): htmx binds new content only after a short delay.

### Front-end Assets
The page loads nothing from the internet: htmx, Lightweight Charts and the Inter and JetBrains Mono fonts are in `app/static` (licenses in `app/static/LICENSES.md`). `app/static/css/themes.css` (the color themes) is generated by `scripts/build_themes.py`, and `app/static/css/app.css` is built by Tailwind CSS v3.4.17 from the classes used in the templates and in the page's own scripts (`app/static/js/lob-*.js`). **After adding or changing classes, run `make css`** (the `pytailwindcss` dev dependency fetches the Tailwind binary on first use); `make check` fails while `app.css` is stale. Static URLs carry the file's modification time (`asset()` in `app/main.py`), so browsers pick up a rebuilt stylesheet without a hard refresh.

### Re-running the Notebooks
`make notebooks` re-executes all nine in place, refreshing their stored outputs (`NOTEBOOKS=...` selects a subset). Notebooks 03–09 take about 14, 17, 58, 18, 21, 28 and 28 seconds on this machine; 05 (32 seeds per model) and 06 (8) run their seeds in parallel. Their conclusions are computed from the runs, so re-running regenerates them.

### Launching the Server
```bash
make serve            # 127.0.0.1:8000; HOST=... PORT=... to change
```
This runs `uvicorn app.main:app --host 127.0.0.1 --port 8000 --timeout-graceful-shutdown 5`. The timeout matters: open dashboard tabs hold event streams, and without it a server waits for them indefinitely when asked to stop. To stop a server, kill only the listener (a plain `lsof -t -i:8000` also matches browsers connected to the port):
```bash
lsof -t -iTCP:8000 -sTCP:LISTEN | xargs kill
```

Sessions live in the server process's memory: each uvicorn process has its own sessions, and a restart starts every visitor afresh (open pages reload themselves). The API has no authentication; keep the default localhost binding unless the network is trusted.

---

## 5. Test Suite Summary

| Test File | Focus Areas | Tests |
|---|---|---|
| `tests/test_book.py` | Book mechanics, depth profiles, tick/lot validation before matching, duplicate IDs, snapshots, icebergs (peak only in the depth, validation) | 21 |
| `tests/test_matching.py` | FIFO execution, sweeps, partial fills, pro-rata allocation in whole lots, icebergs (fills beyond the peak, lost priority, pro-rata refills), randomized invariant checks for both engines at lot sizes 1 and 5, with and without icebergs (quantity conserved) | 25 |
| `tests/test_order.py` | Limit/market/cancel orders, positive size checks, unique IDs, fill records | 6 |
| `tests/test_models.py` | ZI placement and time-unit invariance, Hawkes ZI (reduces to ZI, arrival rate, clustering), long-memory signs, fractional noise, both Mike-Farmer models, Parlour rule, shared clock, simulation harness, event and trade logs | 39 |
| `tests/test_analysis.py` | Estimators (ACF, Hill, power law, Hurst), virtual impact, depth profiles, event-time returns, per-order VWAP, order flow imbalance and its regression, plotting helpers | 21 |
| `tests/test_depth_chart.py` | The depth chart's geometry: step curves from the mid outward, the reach and its limits, a very wide spread, one side empty, your orders marked | 6 |
| `tests/test_impact.py` | Metaorder impact against a twin run: an exact case on a book that stays put, sizes that must split evenly, repeatability | 3 |
| `tests/test_lobster.py` | LOBSTER data: messages to orders (by hand), the files' round trip, the censored price fit, every parameter recovered from simulated files, a file that starts on a full book | 5 |
| `tests/test_calibration.py` | Mike & Farmer (2008) estimators: FARIMA and sample autocorrelations, the rounded-t fit, the signs' Hurst exponent, offsets, recovery of a full parameter set from a simulated run | 5 |
| `tests/test_facts.py` | Live stylized facts and OFI: the recorder reproduces notebook 05's sample and statistics, and the library's OFI, from a `Simulation`; grading on synthetic data; caps and resets; notebook presets read from notebook 05 | 8 |
| `tests/test_web_app.py` | Routes, local-only assets, validation, order results, own-order markers, stylized-facts panel and collection, engine toggle, trade-based candles, interval re-aggregation, simulation clock, event stream, sessions (isolation, server-issued IDs, eviction), parameters and seed, every model runs, cross-tab control sync, models never cancel the user's orders, market-order protection, icebergs, VWAP, order-flow card, export, fill notices, split orders, your fills per bar and resting levels for the chart, the depth chart panel, moving and canceling your orders (only yours), heatmap columns (timing, a column for every unit, trades, reserves, restarts, the stream), the account and P&L panels, queue positions, the tick view's data, the market maker (quotes, lean, its own account and markouts, never canceled by the models, start and stop) | 110 |
| `tests/browser/test_dashboard.py` | In Chrome (`make test-browser`; Firefox and WebKit too with `make test-browsers`): offline assets and styling, playback and shortcuts, speed slider, quick prices and order results, own-order markers, cross-tab sync, Fit, layout, stylized facts, notebook values, the user's orders surviving the model, a protected market order, VWAP line and order-flow card, an iceberg and the downloads, a fill notice, a split order, views, themes, text size, columns at four widths, orders and fills on the chart, one-click trading, drag-to-move, the heatmap and its recovery, guided lessons, your P&L and its line, the tick view, the market maker, the order book from the keyboard | 29 |
| **Total** | | **249 + 29 browser** |

---

## 6. Changes on the `review-fixes` Branch (2026-11-02)

**Review corrections.**
- **Models**: the ZI model follows the paper's placement rule; the fractional-noise generator no longer halves every autocorrelation; Mike-Farmer rounds bids and asks symmetrically (it had biased 55% of mid-price moves downward).
- **Notebooks**: all six re-executed; conclusions are computed with stated criteria rather than typed in; the concave-impact, bid-ask-bounce and long-memory-sign ablation cells were redone; quotations checked against the paper.
- **Dashboard**: fabricated candle history removed; bars built from trades on one simulation clock; engine toggle and multi-tab chart reset fixed; input validation added.

**Improvements.**
1. **Order book**: orders are validated before matching (an off-tick crossing order used to execute its fills and then crash), and duplicate order IDs are rejected (they corrupted the order index).
2. **Dashboard robustness**: one event stream per tab instead of two; a real connection indicator; pages reload when their simulation is gone; no more synchronous request logging to `/tmp/lob_app.log`; `make serve` binds to localhost with a graceful-shutdown timeout.
3. **Packaging**: `pyproject.toml` with extras, `ruff`, and `make check`.
4. **Notebooks**: seeds run in parallel (joblib). Notebook 05 uses 32 seeds per model and shows a 95% confidence interval in every scorecard cell, with a computed note on any verdict left open; notebook 06 uses 8.
5. **Simulation**: `Simulation.trades` records every fill; the event log uses typed arrays.
6. **Mike-Farmer signs**: a continuous long-memory stream instead of independent 50,000-sign blocks.
7. **Dashboard parameters**: users set each model's parameters and the seed; Reset replays a run exactly.
8. **Mike-Farmer (2008 structure) model**: reproduces 6 of the 7 stylized facts in notebook 05, including heavy tails, the hump-shaped depth profile and concave impact; its volatility clustering sits on the bar.
9. **Hawkes ZI model**: self-exciting market orders produce volatility clustering (rank ACF of |r| 0.24 vs 0.05 for ZI in notebook 05) and heavy tails; notebook 06 shows it is the only tested ingredient that raises clustering.
10. **One simulation per browser session.**

**Follow-ups.**
- Pro-rata matching allocates in whole lots (with lots above 1 it used to leave resting orders that were not whole lots).
- Model, parameters, seed, engine and speed stay in sync across tabs of a session.
- Notebook 05's verdicts: 32 seeds, confidence intervals, and computed explanations for open verdicts.

**Dashboard revisions.**
- Inaccurate text fixed: the notes card (three of six models, an unverified ">150k events/sec with O(1) depth tracking" claim) is now a reading guide; the header links to the paper instead of a generic GitHub topic; trades are labeled BUY/SELL by aggressor.
- Order entry: quick prices follow the book (they showed page-load prices); every order reports its result.
- Layout: compact controls and metrics bring the price chart above the fold; rows wrap on phones instead of overflowing; the chart header no longer repeats the zoom toolbar's buttons.
- Logarithmic speed slider with the active preset highlighted; keyboard shortcuts; own resting orders marked in the DOM.

**Remaining roadmap items (2026-11-03).** Fill notices; split orders (TWAP) compared with one sweep and the market's VWAP; notebook 07 (OFI across models and depths: OFI explains 10% to 61% of 100-event price changes; in the ZI model the slope falls as depth^-2.3); `lob.calibration` and notebook 08 (every Mike & Farmer parameter recovered from simulated runs).

**Dashboard interface (2026-11-03 and 04).** Color themes (Midnight, Daylight, Terminal, color-blind-safe buy/sell); Trade, Analyze and Learn views; the user's resting orders and fills on the price chart; a depth chart under the order book; one-click trading and drag-to-move in the book, and Cancel all; a smaller default text size with a Text setting, and the full width in up to four columns; an order-book heatmap; five guided lessons (Section 3).

**Six more (2026-11-04).** A review of the interface work and its fixes (a heatmap that froze after a reset during its load, gaps in the heatmap's columns, the cancel endpoint canceling the models' orders, and smaller ones); your position, P&L and place in the queue (3L); a tick view of the order book (3D); a market-making bot and its lesson (3M); notebook 09 (metaorder impact: about linear and permanent in every model, not the square root); `lob.lobster` and notebook 08's section 3 (the estimates from data in LOBSTER's format); the order book from the keyboard; and the browser tests in Firefox and WebKit, which found two races.

**Roadmap items (2026-11-03).** Order flow imbalance (library, simulation columns, live card); iceberg orders (both engines, dashboard form); the session VWAP (chart line, order benchmark); CSV and Parquet export of trades and bars. Also: the models never cancel the user's orders, and manual market orders carry price protection.

## 7. Known Limitations

- Sessions are held in memory: lost on restart, separate per server process, capped at 50 (least recently seen evicted) and evicted after 30 minutes idle without an open tab.
- The simplified Mike-Farmer model concentrates depth at the best quote and its book is too thin to measure impact; use the 2008-structure model for those questions.
- Two verdicts in notebook 05 stay open because their values sit next to the bar. Mike-Farmer (2008 structure) volatility clustering is 0.099 against a bar of 0.1, so more seeds will not settle it. Parlour's excess kurtosis is 0.95 against a bar of 1; settling that would take about 915 seeds, 29 times the 32 used. The notebook explains both.
- The stylized-facts panel measures one run: its standard errors are approximate (too small under long memory), and with the dashboard's default rates the Hawkes model's clustering sits on the bar. Use notebook 05's values and 2M events to compare with the notebook.
- The Mike & Farmer (2008) estimators are checked on simulated data only, also in LOBSTER's format: the repository has no order-level market data (LOBSTER's free samples come through a request form). Notebook 08 fits any LOBSTER files put in `data/lobster/` (ignored by git); real data also have varying sizes, hidden orders, partial cancellations and intraday seasonality, which the model lacks.
- The order book's levels view lists only levels with orders; the tick view covers 40 ticks either side of the mid. One-click orders are plain limit orders (no iceberg peak).
- No model reproduces the square-root law of metaorder impact: impact grows about linearly (faster in Mike & Farmer's) and does not revert (notebook 09). Their order flow does not react to a metaorder.
- The market maker requotes only as model events arrive, at its interval, and in these models nobody trades on information, so stale quotes cost little; P&L is in ticks times lots, without fees.
- The heatmap covers 30 ticks either side of the mid and the last 600 units of simulation time, and starts over after a reset or a collection.
- The notebooks filter joblib's "worker stopped" warning: its workers restart when their memory has grown 300 MB since their first task, which happens after a few large simulations. No work is lost.

## 8. Recommendations & Future Enhancements

1. **Calibration on market data**: request LOBSTER's free samples, put them in `data/lobster/` and run notebook 08's last cell (`calibrate_lobster`, with a burn-in of at least its `last_unknown`); then offer the result as a parameter set in the dashboard.
2. **Concave metaorder impact**: add latent liquidity (orders held back and revealed as the price approaches, as in Tóth et al. 2011) to a model and test it with notebook 09's design.
