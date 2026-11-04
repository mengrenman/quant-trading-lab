# Limit Order Book Simulation

A Python + Jupyter notebook project illustrating the concepts from:

> **Gould, M.D., Porter, M.A., Williams, S., McDonald, M., Fenn, D.J. and Howison, S.D.** (2013). *Limit order books.* Quantitative Finance, 13(11), 1709--1742. [doi:10.1080/14697688.2013.803148](https://doi.org/10.1080/14697688.2013.803148) | [arXiv:1012.0349](https://arxiv.org/abs/1012.0349)

## Overview

Limit order books (LOBs) match buyers and sellers in more than half of the world's financial markets. This project provides:

- A **pure-Python LOB engine** (order book, matching, metrics) designed for clarity over speed
- **Five simulation models** (zero intelligence with Poisson or self-exciting market orders, two Mike-Farmer variants, a Parlour-inspired strategic rule)
- **Six Jupyter notebooks** that walk through the paper section by section, building intuition through interactive simulation and visualization
- **An interactive web dashboard** with one simulation per browser session

The code is educational -- it is a teaching tool, not a production trading system.

## Setup

Requires Python 3.10 or later.

### Option A: pip

```bash
pip install -e ".[dev]"     # or: make install
```

The extras are `[web]` (dashboard), `[notebooks]` (Jupyter, joblib) and `[dev]` (both, plus pytest and ruff).

### Option B: conda

```bash
conda env create -f environment.yml
conda activate lob-sim
```

### Launch Interactive Web App (HTMX + FastAPI)

```bash
make serve                  # uvicorn on 127.0.0.1:8000
```
The page works offline: its scripts, fonts and stylesheet are served from `app/static`. After changing classes in the templates, rebuild the stylesheet with `make css`.
Open [http://localhost:8000](http://localhost:8000) to access:
- **Three views**: Trade, Analyze and Learn, on the full width of the window in up to four columns. A Text setting (Small by default) scales everything, and four color themes include a light one and a color-blind-safe palette.
- **Real-Time Candlestick Chart (OHLCV)** powered by Lightweight Charts: bars are built from executed trades on the simulation clock, with stacked buy/sell volume and intervals from 1 to 300 units of simulation time. Your resting orders show as dashed lines and your fills as arrows.
- **Depth of Market (DOM) Ladder** with relative volume fill bars, your own resting orders marked, and a depth chart under it. A click copies a level's price into the order form, or with one-click trading on places a limit order there; drag your orders' badge to another level to move them. A tick view shows every price near the mid at fixed rows, so one-click trading and dragging work at any price; both views also work from the keyboard.
- **Your P&L**: your position, realized and unrealized P&L (marked to the mid) and your edge against the mid, in every view and as a line under the chart; each resting order's place in its queue.
- **Strategies**: market making, momentum, mean reversion and queue imbalance, up to eight at once with settings of your choice, each trading with its own account; a live leaderboard ranks them, and you, by P&L, with what each earns or pays against the mid and what the price takes back afterwards.
- **Combined Simulation & Manual Trading**: submit manual limit/market orders (each one reports what happened: resting, filled at what prices, or nothing filled), manage active orders, or run any of the five models. Your resting orders stay until they fill or you cancel them; the models cancel only their own. Market orders walk the book at most a set number of ticks (price protection, 10 by default); whatever is left waits at that price as your limit order.
- **Controls**: play, step and reset, a logarithmic speed slider from 1 to 2000 events per second, and keyboard shortcuts (Space start/pause, right arrow step, F fit the chart, 1 to 3 switch views).
- **Model parameters and seed**: adjust a model's parameters while it runs; Reset replays a run exactly from the seed.
- **Order flow imbalance, VWAP and icebergs**: a card plotting price changes against order flow imbalance (Cont, Kukanov & Stoikov 2014) for the current run; the session VWAP on the chart and as a benchmark for your fills; iceberg limit orders that show only a peak.
- **Export**: download the session's trades or bars as CSV, or as Parquet with the `[parquet]` extra.
- **Recording and playback**: record a session and play it back exactly, at any speed, stepping or dragging a timeline; trade or run strategies on the same market; carry on live from any moment; keep recordings as `.npz` files to play back later or read in a notebook.
- **Split orders and fill notices**: send a large market order as slices over simulation time and compare the result with one sweep and the market's VWAP; get a notice when one of your resting orders fills.
- **Order-book heatmap**: the visible depth near the mid over simulation time, with the trades and the mid-price; walls and icebergs stand out, and Reveal hidden outlines iceberg reserves.
- **Guided lessons**: price impact, walls, icebergs, order flow imbalance, order splitting, making a market and trading on a signal, step by step; each step can do its action for you and point at the part of the page it is about.
- **Stylized facts, live**: notebook 05's seven facts measured on the run you are watching (heavy tails, volatility clustering, sign memory, bid-ask bounce, depth profile, impact, placement), with single-run uncertainty; collect 400k or 2M events in seconds, or switch to notebook 05's parameter values to compare.
- **Live updates** over one Server-Sent Events stream per tab, with a connection indicator. Each browser gets its own simulation.

### Launch Jupyter Notebooks

```bash
jupyter notebook notebooks/
make notebooks              # re-execute all nine in place
```

## Notebooks

Run in order for the best learning experience:

| # | Notebook | Paper Section | What You'll Learn |
|---|---------|--------------|-------------------|
| 01 | [LOB Mechanics](notebooks/01_lob_mechanics.ipynb) | Section 2 | Build an LOB from scratch; bid/ask/spread/mid-price; depth profiles; returns |
| 02 | [Order Matching](notebooks/02_order_matching.ipynb) | Section 3 | Price-time vs pro-rata priority; fill sequences; state-space complexity |
| 03 | [Zero Intelligence](notebooks/03_zero_intelligence.ipynb) | Section 5 | Smith et al. (2003) ZI model; emergent spread; model limitations |
| 04 | [Stylized Facts](notebooks/04_stylized_facts.ipynb) | Section 4 | Heavy tails, volatility clustering, price impact, long memory |
| 05 | [Agent-Based Models](notebooks/05_agent_based_models.ipynb) | Section 5 | Five models compared on seven stylized facts |
| 06 | [Open Questions](notebooks/06_open_questions.ipynb) | Section 6 | Ablation studies: what causes heavy tails, volatility clustering, concave impact, long memory |
| 07 | [Order Flow Imbalance](notebooks/07_order_flow_imbalance.ipynb) | Cont, Kukanov & Stoikov (2014) | How much of the mid-price changes OFI explains in each model, and how its slope falls with depth |
| 08 | [Calibration](notebooks/08_calibration.ipynb) | Section 5.2.5; Mike & Farmer (2008) | Estimating the Mike & Farmer (2008) model's parameters from order flow, checked by recovering known parameters, also from data in LOBSTER's format (where the prices of orders that execute in full are hidden) |
| 09 | [Metaorder Impact](notebooks/09_metaorder_impact.ipynb) | Tóth et al. (2011); Bouchaud et al. (2018) | Does a large order's impact grow like the square root of its size? In these models it grows about linearly and does not revert |

The notebooks' conclusions are computed from the simulations against stated criteria, so re-running them (for example with other seeds) regenerates the verdicts.

## Package Structure

```
lob/
├── order.py          # Order dataclass, Side/OrderType enums
├── book.py           # OrderBook with SortedDict-based storage
├── matching.py       # PriceTimeEngine, ProRataEngine
├── metrics.py        # Spread, depth, returns, price impact, VWAP, order flow imbalance
├── simulation.py     # Event-driven simulation loop, event and trade logs
├── calibration.py    # Maximum-likelihood estimates of the Mike & Farmer (2008) model
├── lobster.py        # LOBSTER (Nasdaq) data: read, write, and estimate the model from it
├── impact.py         # Metaorder impact against a twin run (common random numbers)
├── strategies.py     # Market making, momentum, mean reversion and queue imbalance, for the dashboard
├── models/
│   ├── zero_intelligence.py   # Smith et al. (2003)
│   ├── hawkes.py              # ZI with self-exciting market orders
│   ├── mike_farmer.py         # Mike & Farmer (2008): simplified and structural
│   ├── long_memory.py         # Continuous long-memory order signs
│   └── strategic.py           # Parlour (1998)-inspired
└── analysis/
    ├── stylized_facts.py      # Power-law fits, ACF, Hurst exponent, depth profiles
    └── plotting.py            # Reusable visualization helpers
app/                  # FastAPI dashboard (run from the repo; not installed)
```

### Models

| Model | Paper Reference | Key Features |
|-------|----------------|-------------|
| **Smith et al. (2003)** | Section 5.2.4 | Zero intelligence: Poisson arrivals, limit orders placed uniformly on the ticks next to the opposite best quote, independent cancellations |
| **ZI + Hawkes** | Section 5.2.4 (after Zhao 2010, Toke 2011) | The ZI model with market orders arriving as a self-exciting process; same long-run rate, clustered in time |
| **Mike & Farmer (2008)**, simplified | Section 5.2.5 | Power-law relative prices (measured from the mid-price), long-memory order signs, log-normal sizes; constant-rate cancellations |
| **Mike & Farmer (2008)**, structure | Section 5.2.5 | Student's t prices relative to the same-side quote (orders can cross the spread), long-memory signs, state-dependent cancellation; illustrative parameters |
| **Parlour (1998)**-inspired | Section 5.1 | Hand-tuned rule: the chance of a market order rises with a narrow spread and a long queue at the best quotes |

## Running Tests

```bash
make test                   # pytest
make check                  # ruff + stylesheet check + pytest + execute every notebook
make test-browser           # drive the dashboard in Chrome (Playwright)
make test-browsers          # the same in Chrome, Firefox and WebKit (Safari's engine)
```

## References

- Gould, M.D. et al. (2013). Limit order books. *Quantitative Finance*, 13(11), 1709--1742.
- Smith, E. et al. (2003). Statistical theory of the continuous double auction. *Quantitative Finance*, 3, 481--514.
- Mike, S. and Farmer, J.D. (2008). An empirical behavioral model of liquidity and volatility. *J. Econ. Dyn. Control*, 32, 200--234.
- Parlour, C.A. (1998). Price dynamics in limit order markets. *Rev. Financ. Stud.*, 11, 789--816.
