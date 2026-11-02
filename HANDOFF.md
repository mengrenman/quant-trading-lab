# Project Handoff: Limit Order Book (LOB) Simulation

## 1. Executive Summary

This project is an interactive, educational **Limit Order Book (LOB) Simulation & Visualization Platform** built in Python and modern web technologies. It is based on the foundational research paper:

> **Gould, M.D., Porter, M.A., Williams, S., McDonald, M., Fenn, D.J. and Howison, S.D.** (2013).  
> *Limit order books.* Quantitative Finance, 13(11), 1709–1742.  
> [doi:10.1080/14697688.2013.803148](https://doi.org/10.1080/14697688.2013.803148) | [arXiv:1012.0349](https://arxiv.org/abs/1012.0349)

The platform pairs a pure-Python matching engine and stochastic market simulation models with six teaching notebooks and a real-time web dashboard (FastAPI, Server-Sent Events, HTMX, Tailwind CSS, and TradingView Lightweight Charts). It lets researchers and students study market microstructure, order queues, emergent bid-ask spreads, and stylized facts (heavy tails, volatility clustering, price impact) through interactive experimentation.

The project is under git. `main` holds the baseline snapshot of 2026-11-01; the `review-fixes` branch holds the corrections and improvements described in Section 6.

---

## 2. System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Web Browser Frontend                            │
│  - Tailwind CSS Dark UI        - TradingView Lightweight Charts        │
│  - One EventSource per tab     - Model parameter panel (HTMX forms)    │
│  - Fixed-Height DOM Ladder     - Connection indicator                  │
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
- **`lob/matching.py`**:
  - `PriceTimeEngine`: Price-time priority (FIFO within each price level).
  - `ProRataEngine`: Allocates proportionally to resting order sizes in whole lots, rounding down and handing leftover lots out in time priority, so fills and remainders stay multiples of the lot size.
  - Engines reduce resting orders' `size` in place as they fill.
- **`lob/models/`** (all models continue from the caller's clock when it is ahead of their own):
  - `zero_intelligence.SmithFarmerModel`: Zero-intelligence model of Daniels et al. (2003) / Smith et al. (2003), Gould et al. Section 5.2.4. Limit orders land uniformly on the `price_range` ticks next to the *opposite* best quote, so they fall inside the spread with probability proportional to its width; market orders and per-order cancellations are Poisson. `_sample_side` / `_sample_size` hooks let ablations change signs and sizes before an order is priced.
  - `hawkes.HawkesZIModel`: The ZI model with market orders arriving as a self-exciting (Hawkes) process. The baseline rate is scaled to `mu * (1 - branching)` so the long-run arrival rate stays `mu`; every arrival excites, even one that finds the opposite side empty. Simulated exactly by thinning.
  - `mike_farmer.MikeFarmerModel`: Simplified Mike & Farmer (2008): power-law relative prices measured from the mid-price (always at least one tick), long-memory order signs, log-normal sizes, constant-rate cancellation.
  - `mike_farmer.MikeFarmer2008Model`: Mike and Farmer's structure: prices set relative to the same-side best quote with Student's t offsets (orders can land inside the spread or cross it and execute as effective market orders; no separate market order process), long-memory signs, and state-dependent cancellation simulated by thinning. Parameter values are illustrative, not their LSE calibration.
  - `long_memory.LongMemorySigns`: A continuous stream of long-memory signs (signs of FARIMA(0, H − 1/2, 0) noise, applied by overlap-save FFT convolution) whose memory never resets.
  - `strategic.ParlourModel`: A hand-tuned rule inspired by Parlour (1998): the probability of a market order rises with a narrow spread and a long queue at the best quotes. It is not a game-theoretic equilibrium model.
- **`lob/simulation.py`**: `Simulation.run()` returns the event log as a DataFrame (stored internally in typed arrays: a 400k-event run takes about 190 MB); `Simulation.trades` has one row per fill; `Simulation.snapshots` holds book snapshots.
- **`lob/metrics.py`**, **`lob/analysis/`**: returns (`event_time_returns`), per-order VWAP (`trades_by_order`), virtual price impact on book snapshots, ACF with white-noise bands, rank-based volatility ACF, Hurst (R/S), Hill and power-law estimators, time-averaged depth profiles and plotting helpers.
- **`app/sessions.py`**: `SessionRegistry` holds one `LOBManager` per browser session, found through a random ID in an HTTP-only, SameSite=Lax cookie (tabs of one browser share it). IDs are issued only by the server. Sessions without an open stream are evicted after 30 minutes idle, and the least recently seen one beyond 50 sessions; eviction stops the session's simulation.
- **`app/manager.py`**:
  - `LOBManager`: One session's book, engine, model, parameters, seed, background runner and manual orders.
  - One simulation clock (`sim_time`): model events advance it by the model's own event times; each manual action advances it by `MANUAL_ACTION_DT` (0.5).
  - Executed trades are stored as one-unit bars (`deque(maxlen=100_000)`); `get_candles()` rebuilds OHLCV bars, with buy and sell volume, for any interval.
  - `reset()` restarts the random generator from the session's seed, so the same seed and parameters replay a run exactly.
- **`app/model_specs.py`**: The models the dashboard offers, their labels, default parameters and allowed ranges; the model menu and parameter panel are generated from it.
- **`app/main.py`**: FastAPI endpoints (`/api/sim/*`, `/api/order/*`, `/api/chart/*`, `/api/engine`, `/api/candlesticks`, `/api/snapshot`) with validated inputs (invalid values return 422), and the `/events` stream. Templates render on worker threads from copied values only.
- **`app/templates/`**: `index.html` (layout and the event-stream script), `components/chart.html` (trade-based candlesticks with stacked buy/sell volume), `model_params.html` (parameter panel), `sim_controls.html`, `play_button.html`, `sim_badge.html`, `engine_toggle.html`, `ladder.html`, `order_form.html`, `active_orders.html`, `trades.html`, `metrics.html`.

---

## 3. Web Dashboard Features

### A. Sessions and Live Updates
1. **One simulation per browser.** Each browser gets its own simulation; tabs of the same browser share it.
2. **One event stream per tab** carries the panels, the chart and the controls. Browsers allow about six connections per host over HTTP/1.1, so a tab used to need two streams and three tabs froze the page; four tabs now work.
3. **Tabs stay in sync**: a change to the model, parameters, seed, engine or speed in one tab appears in the session's other tabs. A tab that already shows the change keeps its "Applied" message, a parameter form someone is typing in updates once focus leaves it, and the speed slider is not moved while dragged.
4. **Connection indicator** in the header: Connecting, Live or Reconnecting. If the stream comes back attached to a different simulation (server restart or expired session), the page reloads itself.

### B. Models and Parameters
1. Five models plus manual placement: Smith et al. ZI, ZI with self-exciting market orders, Mike & Farmer simplified, Mike & Farmer (2008 structure), Parlour-inspired rule.
2. **Model parameters panel**: every parameter of the current model plus the random seed. Apply validates every field and either applies all of them (continuing from the current book) or changes nothing and shows per-field messages. Parameters are kept per model.
3. **Reset Book** restarts from the seed, so the same seed and parameters replay the same run.

### C. Candlestick Chart & Volume Dual-Panel
1. **Trade-based bars on the simulation clock**: bars aggregate executed trades only. Intervals without trades are drawn flat at the last trade price (the mid-price before the first trade) with zero volume. There is no synthetic history.
2. **Intervals**: `1s` to `5m` in units of simulation time (each model's own clock, not wall-clock time). Changing the interval rebuilds every bar from the stored trades.
3. **Stacked volume**: green is buy-initiated volume, red sell-initiated.
4. **Zoom**: horizontal zoom from 2 to 32 px per bar with presets. `Fit` shows every bar (at most 32 px wide) and also recenters both charts vertically, switching their price axes back to auto-scaling after a drag. Both panels stay horizontally aligned.

### D. Depth of Market, Controls and Performance
1. Fixed-height DOM with 10 ask and 10 bid slots, padded so the layout never shifts; mid-price and spread banner pinned in the middle.
2. Play/Pause (optimistic), Step, Reset, engine toggle (switches on the server and shows the engine in use), speed from 1 to 2000 events per second (default 50).
3. Template rendering runs on worker threads; panels are pushed at most five times a second while running; the chart receives every bar since its last push.

---

## 4. Environment & Execution Guide

### Python Environment
- **Path**: `/Users/mengren/anaconda3/envs/lob-sim/bin/python` (Python 3.14.3)
- **Install**: `make install` (`pip install -e ".[dev]"`). Dependencies live in `pyproject.toml`: core, plus `[web]`, `[notebooks]` (includes `joblib`) and `[dev]` (`pytest`, `httpx`, `ruff`) extras. Python ≥ 3.10.

Run `make` targets from an environment with the project installed, or pass the interpreter, e.g. `make check PYTHON=/Users/mengren/anaconda3/envs/lob-sim/bin/python`.

### Checks
```bash
make lint             # ruff
make test             # pytest (152 tests, ~5 s)
make notebooks-check  # executes all six notebooks into a temp dir; fails on the first error
make check            # all three (~2 min)
```

### Re-running the Notebooks
`make notebooks` re-executes all six in place, refreshing their stored outputs (`NOTEBOOKS=...` selects a subset). Notebooks 03–06 take about 14, 17, 58 and 18 seconds on this machine; 05 (32 seeds per model) and 06 (8) run their seeds in parallel. Their conclusions are computed from the runs, so re-running regenerates them.

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
| `tests/test_book.py` | Book mechanics, depth profiles, tick/lot validation before matching, duplicate IDs, snapshots | 19 |
| `tests/test_matching.py` | FIFO execution, sweeps, partial fills, pro-rata allocation in whole lots, randomized invariant checks for both engines at lot sizes 1 and 5 | 16 |
| `tests/test_order.py` | Limit/market/cancel orders, positive size checks, unique IDs, fill records | 6 |
| `tests/test_models.py` | ZI placement and time-unit invariance, Hawkes ZI (reduces to ZI, arrival rate, clustering), long-memory signs, fractional noise, both Mike-Farmer models, Parlour rule, shared clock, simulation harness, event and trade logs | 39 |
| `tests/test_analysis.py` | Estimators (ACF, Hill, power law, Hurst), virtual impact, depth profiles, event-time returns, per-order VWAP, plotting helpers | 19 |
| `tests/test_web_app.py` | Routes, validation, engine toggle, trade-based candles, interval re-aggregation, simulation clock, event stream, sessions (isolation, server-issued IDs, eviction), parameters and seed, every model runs, cross-tab control sync | 53 |
| **Total** | | **152** |

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
9. **Hawkes ZI model**: self-exciting market orders produce volatility clustering (rank ACF of |r| 0.27 vs 0.04 for ZI) and heavy tails; notebook 06 shows it is the only tested ingredient that raises clustering.
10. **One simulation per browser session.**

**Follow-ups.**
- Pro-rata matching allocates in whole lots (with lots above 1 it used to leave resting orders that were not whole lots).
- Model, parameters, seed, engine and speed stay in sync across tabs of a session.
- Notebook 05's verdicts: 32 seeds, confidence intervals, and computed explanations for open verdicts.

## 7. Known Limitations

- Sessions are held in memory: lost on restart, separate per server process, capped at 50 (least recently seen evicted) and evicted after 30 minutes idle without an open tab.
- The simplified Mike-Farmer model concentrates depth at the best quote and its book is too thin to measure impact; use the 2008-structure model for those questions.
- Two verdicts in notebook 05 sit on their bars and will not resolve with more seeds: Mike-Farmer (2008 structure) volatility clustering (about 0.10 against 0.1) and Parlour's tails (excess kurtosis about 0.9 against 1). The notebook says so, with the seeds a verdict would take.
- The notebooks filter joblib's "worker stopped" warning: its workers restart when their memory has grown 300 MB since their first task, which happens after a few large simulations. No work is lost.

## 8. Recommendations & Future Enhancements

1. **VWAP & technical overlays**: VWAP and EMA line series on the candlestick panel (the trade log already has what VWAP needs).
2. **Order Flow Imbalance (OFI) indicator**: live OFI (Cont et al. 2014) in the volume pane.
3. **Tick data export / replay**: export the trade log and book snapshots to CSV/Parquet for offline research in the notebooks.
4. **Iceberg & hidden orders**: extend the order types with peak/display quantity matching.
5. **Calibration**: fit the Mike-Farmer (2008 structure) parameters to data rather than illustrative values.
