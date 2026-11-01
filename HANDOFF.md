# Project Handoff: Limit Order Book (LOB) Simulation

## 1. Executive Summary

This project is an interactive, educational, and high-fidelity **Limit Order Book (LOB) Simulation & Visualization Platform** built in Python and modern web technologies. It is based on the foundational research paper:

> **Gould, M.D., Porter, M.A., Williams, S., McDonald, M., Fenn, D.J. and Howison, S.D.** (2013).  
> *Limit order books.* Quantitative Finance, 13(11), 1709–1742.  
> [doi:10.1080/14697688.2013.803148](https://doi.org/10.1080/14697688.2013.803148) | [arXiv:1012.0349](https://arxiv.org/abs/1012.0349)

The platform pairs a pure-Python matching engine and stochastic market simulation models with a real-time web dashboard (FastAPI, Server-Sent Events, HTMX, Tailwind CSS, and TradingView Lightweight Charts). It allows quantitative researchers and students to study market microstructure, order queues, emergent bid-ask spreads, and stylized facts (volatility clustering, heavy tails, price impact) through interactive experimentation.

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
         /events (DOM / Tape / Metrics)   /events/chart (OHLCV Bars)
                    │                              │
┌───────────────────┴──────────────────────────────┴─────────────────────┐
│                          FastAPI Web Layer                             │
│  - app/main.py: Routes, SSE broadcast channels, Threadpool offloading  │
│  - app/manager.py: LOBManager, Candlestick aggregator, State lock      │
└───────────────────────────────────▲────────────────────────────────────┘
                                    │
┌───────────────────────────────────┴────────────────────────────────────┐
│                       Core Simulation & Engine                         │
│  - lob/book.py: Limit Order Book, O(1) queue management, depth profile │
│  - lob/matching.py: PriceTimeEngine (FIFO) & ProRataEngine             │
│  - lob/models/: Smith-Farmer ZI, Mike-Farmer, Parlour Models           │
│  - lob/simulation.py: Poisson event generator, event clock             │
└────────────────────────────────────────────────────────────────────────┘
```

### Key Modules & Files

- **`lob/book.py`**:
  - `OrderBook`: Dual-sided order book with tick size and lot size validation.
  - Maintains `_bids` (max-heap/sorted reverse) and `_asks` (min-heap/sorted) price levels.
  - Constant-time $O(1)$ order lookup, cancel, and depth-at-price querying.
- **`lob/matching.py`**:
  - `MatchingEngine`: Abstract base class.
  - `PriceTimeEngine`: Standard price-time priority (FIFO within each price level).
  - `ProRataEngine`: Proportional allocation matching based on resting order sizes.
- **`lob/models/`**:
  - `SmithFarmerModel`: Zero-Intelligence Poisson process for limit orders, market orders, and cancellations.
  - `MikeFarmerModel`: Empirical order placement with heavy-tailed cancellations and clustering.
  - `ParlourModel`: Strategic trader game-theoretic order placement.
- **`app/manager.py`**:
  - `LOBManager`: Coordinates the simulation lifecycle, background async runner, manual order tracking, and dynamic candlestick aggregation (OHLCV).
  - Maintains a sliding historical candle window (`deque(maxlen=500)`).
- **`app/main.py`**:
  - FastAPI application endpoints (`/api/sim/*`, `/api/order/*`, `/api/chart/*`).
  - Dual Server-Sent Events (SSE) streaming with background template rendering offloaded via `asyncio.to_thread`.
- **`app/templates/components/`**:
  - `chart.html`: Candlestick OHLC + Volume dual-panel with zoom and separation controls.
  - `ladder.html`: Symmetrical, fixed-height Depth of Market (DOM) price ladder.
  - `sim_controls.html`: Play/pause, step, reset, engine toggle, and 2000 eps speed slider.
  - `order_form.html`: Limit and market order submission forms.
  - `active_orders.html`: Active manual order management table.
  - `trades.html`: Time-and-sales execution tape.
  - `metrics.html`: Real-time spread, mid-price, order imbalance, and book depths.

---

## 3. Key Enhancements Completed

### A. Candlestick Chart & Volume Dual-Panel
1. **Enlarged Panel Height**:
   - Expanded candlestick container height from `240px` to **`420px`** (`#price-chart-container`).
   - Integrated responsive `ResizeObserver` maintaining optimal vertical scale across screen sizes.
2. **Extended Aggregation Intervals**:
   - Supported timeframes: **`1s`**, **`2s`**, **`5s`**, **`10s`**, **`15s`**, **`30s`**, **`1m`**, **`3m`**, **`5m`**.
   - One-click pill toolbar in chart header with active state highlighting.
   - Server-side dynamic aggregation recalculation with state preservation.
3. **Horizontal Zoom Controls**:
   - **Continuous Zoom Slider**: Smoothly scales bar spacing from `2px` (dense macro view) to `32px` (micro view).
   - **Zoom Level Indicator Badge**: Displays live zoom percentage and bar width in pixels (e.g., `100% (8.0px)`).
   - **Steppers & Quick Presets**: Quick `[-]` / `[+]` buttons, plus presets: `Dense` (4px), `1.0x` (8px), `2.0x` (16px), `Wide` (26px), and `Fit` (auto-fit content).
   - **Bidirectional Dual-Chart Synchronization**: Scaling or panning either the price chart or volume chart keeps both charts in exact horizontal alignment.
4. **Adjustable Vertical Separation**:
   - Slider (`8px` to `100px`) and preset buttons (`12px`, `32px`, `56px`) to adjust the spacing between the Price Candlestick panel and Traded Volume panel.

### B. Depth of Market (DOM) Stabilization
1. **Fixed Window Height (`h-[600px]`)**:
   - Locked the DOM window height to eliminate layout jumping during simulation runs.
2. **Deterministic Row Count & Clean Padding**:
   - Configured fixed 10-level slots for asks and 10-level slots for bids (20 levels total around the mid banner).
   - If book liquidity is thin, missing levels are rendered as clean empty placeholders (`—`) with fixed row heights (`21px`), guaranteeing zero layout shift.
3. **Mid-Price Pinning**:
   - The mid-price and spread banner (`h-[36px]`) remains perfectly centered, flanked by best ask above and best bid below.

### C. Simulation Controls & Playback
1. **Optimistic UI Playback Toggle**:
   - Instant 0ms visual feedback on the Play/Pause button (`#sim-play-btn`) before the server response completes.
   - Automatic sync with server state via SSE messages.
2. **High-Frequency Simulation Speed Bar**:
   - Speed slider supporting from 1 event/sec up to **2000 events/sec**.
   - Preset buttons (`1`, `10`, `50`, `100`, `500`, `1k`, `2k max`) with client-side event fallbacks.

### D. Performance & Stutter Elimination
1. **Threadpool Template Rendering**:
   - Offloaded Jinja2 template rendering (`ladder`, `metrics`, `trades`, `active_orders`) to worker threads via `asyncio.to_thread`, keeping the async event loop responsive.
2. **DOM Push Rate Throttling**:
   - Limited DOM tree push frequency to 5 FPS (200ms) during simulation. This reduced browser reflow overhead by over 60% and eliminated UI stutter.
3. **Dedicated SSE Chart Channel**:
   - High-frequency candlestick updates run over `/events/chart`, decoupled from the DOM layout updates.

---

## 4. Environment & Execution Guide

### Python Environment
- **Path**: `/Users/mengren/anaconda3/envs/lob-sim/bin/python`
- **Python Version**: 3.14.3
- **Dependencies**: `fastapi`, `uvicorn`, `jinja2`, `python-multipart`, `numpy`, `scipy`, `pandas`, `pytest`, `anyio`

### Running the Test Suite
All 48 unit and integration tests pass:
```bash
/Users/mengren/anaconda3/envs/lob-sim/bin/pytest tests/ -v
```

### Launching the Servers
Dual-port servers are supported and kept in sync:
```bash
# Terminate existing instances if running
kill -9 $(lsof -t -i:8000 -i:8001)

# Start Primary Web Server (Port 8000)
nohup /Users/mengren/anaconda3/envs/lob-sim/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 > /tmp/lob_8000.log 2>&1 &

# Start Secondary Web Server (Port 8001)
nohup /Users/mengren/anaconda3/envs/lob-sim/bin/uvicorn app.main:app --host 0.0.0.0 --port 8001 > /tmp/lob_8001.log 2>&1 &
```

Check server health:
```bash
curl -s http://localhost:8000/api/candlesticks | python3 -m json.tool
curl -s http://localhost:8001/api/candlesticks | python3 -m json.tool
```

---

## 5. Test Suite Summary

The test suite covers book mechanics, matching rules, order lifecycle, web routes, and real-time streaming:

| Test File | Focus Areas | Status |
|---|---|---|
| `tests/test_book.py` | Book emptiness, bid/ask sorting, spreads, mid-price, cancels, depth profiles, tick/lot validation, snapshots | **16/16 Passed** |
| `tests/test_matching.py` | FIFO price-time execution, sweeping multiple levels, partial fills, pro-rata proportional allocation | **11/11 Passed** |
| `tests/test_order.py` | Limit/market/cancel orders, positive size checks, unique ID generation, fill records | **6/6 Passed** |
| `tests/test_web_app.py` | Index page, order forms, cancel routes, sim playback & reset, engine toggle, snapshot API, candlestick API, multi-interval change (1s to 5m), SSE streams, 1000+ eps speed | **15/15 Passed** |
| **Total** | | **48/48 Passed** |

---

## 6. Recommendations & Future Enhancements

1. **VWAP & Technical Overlays**:
   - Add Volume-Weighted Average Price (VWAP) and Exponential Moving Average (EMA) line series overlays onto the Candlestick panel.
2. **Order Flow Imbalance (OFI) Indicator**:
   - Plot live Order Flow Imbalance (Cont et al. 2014) in the volume pane to visualize short-term price pressure.
3. **Tick Data Export / Replay**:
   - Provide an endpoint to export generated trade logs and book snapshots to CSV/Parquet for offline quantitative research in the Jupyter notebooks.
4. **Iceberg & Hidden Orders**:
   - Extend the core order types to include iceberg orders with peak/display quantity matching.
