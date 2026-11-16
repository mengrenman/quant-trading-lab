# Package structure and models

The library is the `lob` package; the dashboard in `app/` is run from the repository and is not installed. This page lists the modules, then the simulation models.

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
├── pairs.py          # Two order books whose gap mean-reverts, an arbitrageur, and z-score, Kalman and copula traders
├── statarb.py        # Many books with several common factors (and, off by default, a trend, an index book and deals), and traders of many relations: pairs, Johansen baskets, PCA residuals, copula pairs
├── crosssection.py   # Cross-sectional long/short traders over those books: reversal, momentum, lead-lag and order flow, neutral to the dollar and to estimated factors
├── sparse_baskets.py # Baskets chosen by Box-Tiao predictability (a greedy search after d'Aspremont), the alternative to Johansen's test
├── deals.py          # Stock deals in those books: an acquirer offers shares for a target's, and the deal may break (the settings; `statarb.py` runs them)
├── indexarb.py       # Index arbitrage: the index book (the last, the sum of the others and a premium) against the books it is made of
├── deal_traders.py   # A trader of the deals, and a trader of relations (pairs, baskets) that keeps out of a deal's target
├── passive.py        # Resting limit orders for those traders: a working order at the best price, crossed once its patience runs out; paper and shortfall against the mid at the decision (not on the dashboard)
├── studies.py        # Microstructure studies of the event stream: order flow imbalance, volume delta, VPIN, order-to-trade ratios, book imbalance, spread, resilience
├── itch.py           # One day of Nasdaq TotalView-ITCH 5.0 as frames in LOBSTER's format: scan, extract, replay a day into a book (reads files that are not in the repository; not on the dashboard)
├── feeds.py          # Frames in LOBSTER's format as the stream of events the studies take, and a runner for them
├── ladder.py         # What a price-level feed loses: the real stream degraded to levels, batches of real time and shifted trades
├── copulas.py        # Bivariate copulas: conditional probabilities, fitting by canonical maximum likelihood, AIC
├── strategies.py     # Market making, momentum, mean reversion and queue imbalance, for the dashboard
├── models/
│   ├── zero_intelligence.py   # Smith et al. (2003)
│   ├── hawkes.py              # ZI with self-exciting market orders
│   ├── mike_farmer.py         # Mike & Farmer (2008): simplified and structural
│   ├── long_memory.py         # Continuous long-memory order signs
│   ├── informed.py            # ZI with informed traders who know a hidden value (the dashboard's sixth model)
│   └── strategic.py           # Parlour (1998)-inspired
└── analysis/
    ├── stylized_facts.py      # Power-law fits, ACF, Hurst exponent, depth profiles
    └── plotting.py            # Reusable visualization helpers
app/                  # FastAPI dashboard (run from the repo; not installed)
├── main.py           # Routes, validation and the event streams
├── manager.py        # One session's order book, simulation clock, strategies, recordings, studies, tape, footprint and triggers
├── pairs.py          # The Pairs view: two books of their own, and its traders
├── statarb.py        # The Stat-arb view: many books of their own, and its traders
├── live.py           # What the Pairs and Stat-arb markets share: speed, run loop, streams
├── training.py       # Fitting a trader over training runs, for the Pairs and Stat-arb views
├── fitting.py        # What a fit tries: the values to combine, a random sample when there are many
├── studies.py        # One session's board of studies: their settings, samples and helps for the page
├── triggers.py       # Trigger rules on the studies' values: notice, pause, cancel, flatten
└── ...               # sessions, model and strategy specs, recordings, stylized facts, templates, static files
```

### Models

| Model | Paper Reference | Key Features |
|-------|----------------|-------------|
| **Smith et al. (2003)** | Section 5.2.4 | Zero intelligence: Poisson arrivals, limit orders placed uniformly on the ticks next to the opposite best quote, independent cancellations |
| **ZI + Hawkes** | Section 5.2.4 (after Zhao 2010, Toke 2011) | The ZI model with market orders arriving as a self-exciting process; same long-run rate, clustered in time |
| **Mike & Farmer (2008)**, simplified | Section 5.2.5 | Power-law relative prices (measured from the mid-price), long-memory order signs, log-normal sizes; constant-rate cancellations |
| **Mike & Farmer (2008)**, structure | Section 5.2.5 | Student's t prices relative to the same-side quote (orders can cross the spread), long-memory signs, state-dependent cancellation; illustrative parameters |
| **Parlour (1998)**-inspired | Section 5.1 | Hand-tuned rule: the chance of a market order rises with a narrow spread and a long queue at the best quotes |
| **ZI with informed traders** | Glosten and Milgrom (1985) for the trader's rule; not in the paper | The ZI model plus informed traders who know a hidden value (a random walk with jumps) and send market orders only when it lies beyond the quote they hit; their share of the taker volume and what makers lose to them are known exactly, which no other model offers. Reveal draws the value on the dashboard's chart. A persistence setting makes the other market orders come in runs of one side, for a one-sided flow that is not toxic. Not in notebook 05 |

## Browser tests

```bash
make test-browser           # drive the dashboard in Chrome (Playwright)
make test-browsers          # the same in Chrome, Firefox and WebKit (Safari's engine)
```

## References

- Gould, M.D., Porter, M.A., Williams, S., McDonald, M., Fenn, D.J. and Howison, S.D. (2013). Limit order books. *Quantitative Finance*, 13(11), 1709--1742. [doi:10.1080/14697688.2013.803148](https://doi.org/10.1080/14697688.2013.803148) | [arXiv:1012.0349](https://arxiv.org/abs/1012.0349)
- Smith, E. et al. (2003). Statistical theory of the continuous double auction. *Quantitative Finance*, 3, 481--514.
- Mike, S. and Farmer, J.D. (2008). An empirical behavioral model of liquidity and volatility. *J. Econ. Dyn. Control*, 32, 200--234.
- Parlour, C.A. (1998). Price dynamics in limit order markets. *Rev. Financ. Stud.*, 11, 789--816.

Each notebook's own references are in [notebooks.md](notebooks.md).
