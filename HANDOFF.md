# Project Handoff: Limit Order Book (LOB) Simulation

## 1. Executive Summary

This project is an interactive, educational **Limit Order Book (LOB) Simulation & Visualization Platform** built in Python and modern web technologies. It is based on the foundational research paper:

> **Gould, M.D., Porter, M.A., Williams, S., McDonald, M., Fenn, D.J. and Howison, S.D.** (2013).  
> *Limit order books.* Quantitative Finance, 13(11), 1709–1742.  
> [doi:10.1080/14697688.2013.803148](https://doi.org/10.1080/14697688.2013.803148) | [arXiv:1012.0349](https://arxiv.org/abs/1012.0349)

The platform pairs a pure-Python matching engine and stochastic market simulation models with six teaching notebooks and a real-time web dashboard (FastAPI, Server-Sent Events, HTMX, Tailwind CSS, and TradingView Lightweight Charts). It lets researchers and students study market microstructure, order queues, emergent bid-ask spreads, and stylized facts (heavy tails, volatility clustering, price impact) through interactive experimentation.

The project is under git. `main` holds the baseline snapshot; the `review-fixes` branch holds the corrections described in Section 6.

---

## 2. System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Web Browser Frontend                            │
│  - Tailwind CSS Dark UI        - TradingView Lightweight Charts        │
│  - HTMX SSE Listeners          - Dual-Panel Time-Scale Synchronizer    │
│  - Fixed-Height DOM Ladder     - Optimistic Playback & Speed Toolbar   │
└───────────────────▲──────────────────────────────▲─────────────────────┘
                    │                              │
         /events (DOM / Tape / Metrics)   /events/chart (OHLCV bars)
                    │                              │
┌───────────────────┴──────────────────────────────┴─────────────────────┐
│                          FastAPI Web Layer                             │
│  - app/main.py: Routes, input validation, SSE channels, threadpool     │
│  - app/manager.py: LOBManager, simulation clock, trade-based bars      │
└───────────────────────────────────▲────────────────────────────────────┘
                                    │
┌───────────────────────────────────┴────────────────────────────────────┐
│                       Core Simulation & Engine                         │
│  - lob/book.py: Limit order book, cached depths, random order sampling │
│  - lob/matching.py: PriceTimeEngine (FIFO) & ProRataEngine             │
│  - lob/models/: Smith et al. ZI, simplified Mike-Farmer, Parlour rule  │
│  - lob/simulation.py: Event loop with event log and book snapshots     │
└────────────────────────────────────────────────────────────────────────┘
```

### Key Modules & Files

- **`lob/book.py`**:
  - `OrderBook`: Dual-sided order book with tick size and lot size validation.
  - Price levels live in `SortedDict`s (bid keys negated so iteration runs best-first); each level is a FIFO `deque`.
  - O(1) order lookup, depth-at-price queries and uniform random order sampling. Canceling removes the order from its level's queue, which is linear in that queue's length.
  - `orders_at(price, side)` returns a level's queue in priority order.
- **`lob/matching.py`**:
  - `MatchingEngine`: Abstract base class.
  - `PriceTimeEngine`: Price-time priority (FIFO within each price level).
  - `ProRataEngine`: Allocates proportionally to resting order sizes, rounding down and handing leftover lots out in time priority.
  - Engines reduce resting orders' `size` in place as they fill.
- **`lob/models/`** (all three continue from the caller's clock when it is ahead of their own):
  - `SmithFarmerModel`: Zero-intelligence model of Daniels et al. (2003) / Smith et al. (2003), Gould et al. Section 5.2.4. Limit orders land uniformly on the `price_range` ticks next to the *opposite* best quote, so they fall inside the spread with probability proportional to its width; market orders and per-order cancellations are Poisson. `_sample_side` / `_sample_size` hooks let ablations change signs and sizes before an order is priced.
  - `MikeFarmerModel`: Simplified Mike & Farmer (2008): power-law relative prices measured from the mid-price (always at least one tick), long-memory order signs from fractional Gaussian noise (Davies-Harte), log-normal sizes. Cancellations are uniform at a constant rate, unlike the original's state-dependent cancellation.
  - `ParlourModel`: A hand-tuned rule inspired by Parlour (1998): the probability of a market order rises with a narrow spread and a long queue at the best quotes. It is not a game-theoretic equilibrium model.
- **`lob/metrics.py`**, **`lob/analysis/`**: returns (`event_time_returns`), virtual price impact on book snapshots, ACF with white-noise bands, rank-based volatility ACF, Hurst (R/S), Hill and power-law estimators, time-averaged depth profiles and plotting helpers.
- **`app/manager.py`**:
  - `LOBManager`: Coordinates the book, engine, model, background runner and manual orders.
  - One simulation clock (`sim_time`): model events advance it by the model's own event times; each manual action advances it by `MANUAL_ACTION_DT` (0.5).
  - Executed trades are stored as one-unit bars (`deque(maxlen=100_000)`); `get_candles()` rebuilds OHLCV bars, with buy and sell volume, for any interval.
  - `chart_version` increments on reset and interval changes so every open chart reloads.
- **`app/main.py`**:
  - FastAPI endpoints (`/api/sim/*`, `/api/order/*`, `/api/chart/*`, `/api/engine`, `/api/candlesticks`, `/api/snapshot`). Form inputs are validated (invalid values return 422).
  - Two Server-Sent Events channels. Templates render on worker threads from copied values only; the live book never enters the template context.
- **`app/templates/components/`**:
  - `chart.html`: Trade-based candlesticks plus stacked buy/sell volume, with zoom and separation controls; axis labels show elapsed simulation time.
  - `engine_toggle.html`: Matching-engine button, re-rendered by `/api/engine` after each switch.
  - `ladder.html`: Fixed-height Depth of Market (DOM) price ladder.
  - `sim_controls.html`: Play/pause, step, reset, model select, engine toggle, and speed slider (1–2000 events per second).
  - `order_form.html`, `active_orders.html`, `trades.html`, `metrics.html`: order entry, manual orders, time and sales, top-of-book metrics.

---

## 3. Web Dashboard Features

### A. Candlestick Chart & Volume Dual-Panel
1. **Trade-based bars on the simulation clock**: bars aggregate executed trades only. Intervals without trades are drawn flat at the last trade price (the mid-price before the first trade) with zero volume. A fresh session starts with a single empty bar; there is no synthetic history.
2. **Intervals**: `1s` to `5m` in units of simulation time (each model's own clock, not wall-clock time). Changing the interval rebuilds every bar from the stored trades.
3. **Stacked volume**: green is buy-initiated volume, red sell-initiated.
4. **Horizontal zoom**: slider (2–32 px per bar), steppers, presets (`Dense`, `1.0x`, `2.0x`, `Wide`) and `Fit` (capped at 32 px per bar). Both panels stay horizontally aligned.
5. **Adjustable vertical separation** between the price and volume panels.

### B. Depth of Market (DOM)
1. Fixed `h-[600px]` window with 10 ask and 10 bid slots; thin books are padded with `—` rows so the layout does not shift.
2. Mid-price and spread banner pinned between best ask and best bid.

### C. Simulation Controls
1. Optimistic Play/Pause button, synced with the server over SSE.
2. Speed from 1 to 2000 events per second (default 50). Above 20 eps the server batches steps into 20 frames per second.
3. The engine toggle switches engines on the server and shows the engine actually in use.

### D. Performance
1. Template rendering runs on worker threads (`asyncio.to_thread`), keeping the event loop responsive.
2. DOM pushes are throttled to 5 per second while the simulation runs.
3. The chart channel sends every bar since its last push, so fast simulations do not drop bars.

---

## 4. Environment & Execution Guide

### Python Environment
- **Path**: `/Users/mengren/anaconda3/envs/lob-sim/bin/python`
- **Python Version**: 3.14.3
- **Dependencies**: see `requirements.txt` (numpy, pandas, scipy, sortedcontainers, matplotlib, jupyter, fastapi, uvicorn, jinja2, python-multipart, httpx, pytest, ...)

### Running the Test Suite
```bash
/Users/mengren/anaconda3/envs/lob-sim/bin/python -m pytest tests/ -v
```

### Re-running the Notebooks
```bash
/Users/mengren/anaconda3/envs/lob-sim/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```
On this machine notebooks 03–06 take about 14, 17, 33 and 48 seconds, and 01–02 about 3 seconds each (two minutes in total). Their conclusions are computed from the runs, so re-running regenerates them.

### Launching the Server
```bash
# Stop servers already listening on these ports (listeners only; a plain
# `lsof -t -i:8000` also matches browsers connected to the port)
lsof -t -iTCP:8000 -sTCP:LISTEN | xargs kill

nohup /Users/mengren/anaconda3/envs/lob-sim/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 > /tmp/lob_8000.log 2>&1 &
```

Each uvicorn process holds its own simulation (the manager is a per-process singleton), so servers on two ports are two independent sessions. The API has no authentication; binding to `0.0.0.0` exposes the simulation controls to the local network.

Check server health:
```bash
curl -s http://localhost:8000/api/candlesticks | python3 -m json.tool
```

---

## 5. Test Suite Summary

| Test File | Focus Areas | Tests |
|---|---|---|
| `tests/test_book.py` | Book emptiness, bid/ask sorting, spreads, mid-price, cancels, depth profiles, tick/lot validation, snapshots | 16 |
| `tests/test_matching.py` | FIFO execution, sweeps, partial fills, pro-rata allocation, randomized invariant checks for both engines | 13 |
| `tests/test_order.py` | Limit/market/cancel orders, positive size checks, unique IDs, fill records | 6 |
| `tests/test_models.py` | ZI placement and time-unit invariance, side hook, shared clock, fractional-noise autocorrelation, Mike-Farmer rounding symmetry, Parlour rule, simulation harness | 20 |
| `tests/test_analysis.py` | Estimators (ACF, Hill, power law, Hurst), virtual impact, depth profiles, event-time returns, plotting helpers | 18 |
| `tests/test_web_app.py` | Routes, validation, engine toggle, trade-based candles, interval re-aggregation, simulation clock, SSE | 27 |
| **Total** | | **100** |

---

## 6. Changes on the `review-fixes` Branch (2026-11-02)

A review found correctness problems in the models, notebooks and dashboard. Each fix below has tests or re-executed notebook output behind it.

- **Models**: the ZI model now follows the paper's placement rule; it used to place orders behind the same-side quote, plus a flat 40% chance of an inside-spread order. The fractional-noise generator now uses a symmetric circulant embedding; before, every autocorrelation came out halved. Mike-Farmer rounds bids and asks symmetrically; bids used to sit a tick farther from the mid when the spread was odd, so with the notebook parameters 55% of mid-price moves were downward.
- **Notebooks**: all six re-executed. The stored outputs dated from April 2026 and predated the October model changes. Printed conclusions are now computed with stated criteria rather than typed in. The concave-impact, bid-ask-bounce and long-memory-sign ablation cells were measuring the wrong thing and have been redone. Quotations were checked against the paper and corrected.
- **Dashboard**: fabricated candle history removed; bars built from trades on one simulation clock; engine toggle and multi-tab chart reset fixed; input validation added.

## 7. Known Limitations

- The simplified Mike-Farmer model concentrates depth at the best quote (no hump) and its book is too thin to measure price impact beyond a couple of lots.
- At the 1,000-event scale, the ZI and Mike-Farmer models show only weak volatility clustering (lag-1 rank autocorrelation of |r| around 0.04–0.05, measured with 8 seeds × 1M events), far below real markets.
- The "Live Connected" indicator in the header is static text; only the chart's status dot reacts to connection loss.
- The request-logging middleware appends every request to `/tmp/lob_app.log` synchronously and without rotation.

## 8. Recommendations & Future Enhancements

1. **VWAP & technical overlays**: VWAP and EMA line series on the candlestick panel.
2. **Order Flow Imbalance (OFI) indicator**: live OFI (Cont et al. 2014) in the volume pane.
3. **Tick data export / replay**: export the trade log and book snapshots to CSV/Parquet for offline research in the notebooks.
4. **Iceberg & hidden orders**: extend the order types with peak/display quantity matching.
5. **Mike-Farmer fidelity**: same-side relative prices with inside-spread placement and state-dependent cancellation, as in the original model.
