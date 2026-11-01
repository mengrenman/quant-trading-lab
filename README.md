# Limit Order Book Simulation

A Python + Jupyter notebook project illustrating the concepts from:

> **Gould, M.D., Porter, M.A., Williams, S., McDonald, M., Fenn, D.J. and Howison, S.D.** (2013). *Limit order books.* Quantitative Finance, 13(11), 1709--1742. [doi:10.1080/14697688.2013.803148](https://doi.org/10.1080/14697688.2013.803148) | [arXiv:1012.0349](https://arxiv.org/abs/1012.0349)

## Overview

Limit order books (LOBs) match buyers and sellers in more than half of the world's financial markets. This project provides:

- A **pure-Python LOB engine** (order book, matching, metrics) designed for clarity over speed
- **Three simulation models** of increasing sophistication (zero-intelligence, agent-based, strategic)
- **Six Jupyter notebooks** that walk through the paper section by section, building intuition through interactive simulation and visualization

The code is educational -- it is a teaching tool, not a production trading system.

## Setup

### Option A: pip

```bash
pip install -r requirements.txt
pip install -e .
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate lob-sim
pip install -e .
```

### Launch Interactive Web App (HTMX + FastAPI)

```bash
uvicorn app.main:app --reload --port 8000
```
Open [http://localhost:8000](http://localhost:8000) to access:
- **Real-Time Candlestick Chart (OHLCV)** powered by Lightweight Charts with live volume sub-chart and timeframe selector (1s, 2s, 5s).
- **Depth of Market (DOM) Ladder** with relative volume fill bars and click-to-fill order pricing.
- **Combined Simulation & Manual Trading**: Submit manual limit/market orders, manage active orders, or run automated models (Smith ZI, Mike-Farmer, Parlour).
- **High-Frequency SSE Streaming**: Zero-flicker live updates via Server-Sent Events.

### Launch Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

## Notebooks

Run in order for the best learning experience:

| # | Notebook | Paper Section | What You'll Learn |
|---|---------|--------------|-------------------|
| 01 | [LOB Mechanics](notebooks/01_lob_mechanics.ipynb) | Section 2 | Build an LOB from scratch; bid/ask/spread/mid-price; depth profiles; returns |
| 02 | [Order Matching](notebooks/02_order_matching.ipynb) | Section 3 | Price-time vs pro-rata priority; fill sequences; state-space complexity |
| 03 | [Zero Intelligence](notebooks/03_zero_intelligence.ipynb) | Section 5 | Smith et al. (2003) ZI model; emergent spread; model limitations |
| 04 | [Stylized Facts](notebooks/04_stylized_facts.ipynb) | Section 4 | Heavy tails, volatility clustering, price impact, long memory |
| 05 | [Agent-Based Models](notebooks/05_agent_based_models.ipynb) | Section 5 | Mike-Farmer, Parlour models; side-by-side comparison |
| 06 | [Open Questions](notebooks/06_open_questions.ipynb) | Section 6 | Ablation studies: what causes heavy tails, concave impact, long memory |

## Package Structure

```
lob/
├── order.py          # Order dataclass, Side/OrderType enums
├── book.py           # OrderBook with SortedDict-based storage
├── matching.py       # PriceTimeEngine, ProRataEngine
├── metrics.py        # Spread, depth, returns, price impact
├── simulation.py     # Event-driven simulation loop
├── models/
│   ├── zero_intelligence.py   # Smith et al. (2003)
│   ├── mike_farmer.py         # Mike & Farmer (2008)
│   └── strategic.py           # Parlour (1998)-inspired
└── analysis/
    ├── stylized_facts.py      # Power-law fits, ACF, Hurst exponent
    └── plotting.py            # Reusable visualization helpers
```

### Models

| Model | Paper Reference | Key Features |
|-------|----------------|-------------|
| **Smith et al. (2003)** | Section 5.2.4 | Zero intelligence: Poisson arrivals, uniform placement, independent cancellations |
| **Mike & Farmer (2008)** | Section 5.2.5 | Power-law relative prices, long-memory order signs, heavy-tailed sizes |
| **Parlour (1998)** | Section 5.1 | Strategic placement: agents condition on spread and queue depth |

## Running Tests

```bash
pytest tests/ -v
```

## References

- Gould, M.D. et al. (2013). Limit order books. *Quantitative Finance*, 13(11), 1709--1742.
- Smith, E. et al. (2003). Statistical theory of the continuous double auction. *Quantitative Finance*, 3, 481--514.
- Mike, S. and Farmer, J.D. (2008). An empirical behavioral model of liquidity and volatility. *J. Econ. Dyn. Control*, 32, 200--234.
- Parlour, C.A. (1998). Price dynamics in limit order markets. *Rev. Financ. Stud.*, 11, 789--816.
