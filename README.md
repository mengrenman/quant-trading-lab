# Limit Order Book Simulation

A Python + Jupyter notebook project illustrating the concepts from:

> **Gould, M.D., Porter, M.A., Williams, S., McDonald, M., Fenn, D.J. and Howison, S.D.** (2013). *Limit order books.* Quantitative Finance, 13(11), 1709--1742. [doi:10.1080/14697688.2013.803148](https://doi.org/10.1080/14697688.2013.803148) | [arXiv:1012.0349](https://arxiv.org/abs/1012.0349)

## Overview

Limit order books (LOBs) match buyers and sellers in more than half of the world's financial markets. This project provides:

- A **pure-Python LOB engine** (order book, matching, metrics) designed for clarity over speed
- **Six simulation models** (zero intelligence with Poisson or self-exciting market orders, two Mike-Farmer variants, a Parlour-inspired strategic rule, and zero intelligence with informed traders who know a hidden value)
- **Fifteen Jupyter notebooks**: six walk through the paper section by section, building intuition through interactive simulation and visualization, and nine go on to order flow imbalance, calibration, metaorder impact, pairs and stat-arb, long/short strategies, microstructure signals, and an index book, stock deals and a trend
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
- **Five views**: Trade, Analyze, Learn, Pairs and Stat-arb, on the full width of the window in up to four columns. A Text setting (Small by default) scales everything, and four color themes include a light one and a color-blind-safe palette.
- **Real-Time Candlestick Chart (OHLCV)** powered by Lightweight Charts: bars are built from executed trades on the simulation clock, with stacked buy/sell volume and intervals from 1 to 300 units of simulation time. Your resting orders show as dashed lines and your fills as arrows.
- **Depth of Market (DOM) Ladder** with relative volume fill bars, your own resting orders marked, and a depth chart under it. A click copies a level's price into the order form, or with one-click trading on places a limit order there; drag your orders' badge to another level to move them. A tick view shows every price near the mid at fixed rows, so one-click trading and dragging work at any price; both views also work from the keyboard.
- **Your P&L**: your position, realized and unrealized P&L (marked to the mid) and your edge against the mid, in every view and as a line under the chart; each resting order's place in its queue.
- **Strategies**: ten to play with, up to twelve at once, each with its own account: market making, Avellaneda-Stoikov and penny jumping; momentum, mean reversion, breakout, queue imbalance and order flow imbalance; and two benchmarks, a random trader and a quote joiner. A live leaderboard ranks them, and you, by P&L; it says whether the order flow pays resting orders (the spread against the mid's move per trade, after Wyart et al. 2008) and holds every strategy's fills against a null with no skill. **Fit** any of them over a training period (a fresh run of the model, or a recording): every combination of the values you give is tried on the same period (a random sample of them when there are many), and the one you start keeps its training P&L beside its live one.
- **Combined Simulation & Manual Trading**: submit manual limit/market orders (each one reports what happened: resting, filled at what prices, or nothing filled), manage active orders, or run any of the six models. Your resting orders stay until they fill or you cancel them; the models cancel only their own. Market orders walk the book at most a set number of ticks (price protection, 10 by default); whatever is left waits at that price as your limit order.
- **Controls**: play, step and reset, a logarithmic speed slider from 1 to 2000 events per second, and keyboard shortcuts (Space start/pause, right arrow step, F fit the chart, 1 to 5 switch views).
- **Model parameters and seed**: adjust a model's parameters while it runs; Reset replays a run exactly from the seed.
- **Order flow imbalance, VWAP and icebergs**: a card plotting price changes against order flow imbalance (Cont, Kukanov & Stoikov 2014) for the current run; the session VWAP on the chart and as a benchmark for your fills; iceberg limit orders that show only a peak.
- **Export**: download the session's trades or bars as CSV, or as Parquet with the `[parquet]` extra.
- **Recording and playback**: record a session and play it back exactly, at any speed, stepping or dragging a timeline; trade or run strategies on the same market; carry on live from any moment; keep recordings as `.npz` files to play back later or read in a notebook.
- **Split orders and fill notices**: send a large market order as slices over simulation time and compare the result with one sweep and the market's VWAP; get a notice when one of your resting orders fills.
- **Order-book heatmap**: the visible depth near the mid over simulation time, with the trades and the mid-price; walls and icebergs stand out, and Reveal hidden outlines iceberg reserves.
- **Guided lessons**: price impact, walls, icebergs, order flow imbalance, order splitting, making a market and trading on a signal, step by step; each step can do its action for you and point at the part of the page it is about.
- **Stylized facts, live**: notebook 05's seven facts measured on the run you are watching (heavy tails, volatility clustering, sign memory, bid-ask bounce, depth profile, impact, placement), with single-run uncertainty; collect 400k or 2M events in seconds, or switch to notebook 05's parameter values to compare.
- **Pairs**: a market of its own, two order books whose fair values share a random walk and whose gap mean-reverts with a half-life you set (or wanders, with an arbitrageur to hold the books together if you like), as in notebooks 10 and 11; beta can drift, and the common factor can have a stressed regime. However long it runs, neither fair value goes below 10 ticks: the common factor stops there, which leaves the gap alone. Run up to six traders in it, of three kinds: a z-score at a fixed hedge, a Kalman filter that learns its hedge, and a copula on the books' returns. Watch both mids, the gap with a trader's levels and entries, its signal, its hedge against the true beta and each trader's P&L, with a horizontal zoom, a vertical scale for each pane, Center, Fit and Now, a crosshair across the panes with a readout, panes to show or hide and a choice of heights. A new trader can learn from the chart first, so it trades at once. Fit any kind over training runs: your settings and the combinations you give trade alone in new markets on seeds the chart can never have, each is shown against yours with a standard error, and the panel says whether the best is clearly better.
- **Stat-arb**: another market of its own, up to sixteen books (twelve when their loadings are in no order) with several common factors: sectors, where the pairs within a sector are cointegrated, or loadings in no order, where no pair is. Run up to six traders in it: many pairs at once, baskets (chosen by the Johansen test or by Box-Tiao predictability, after d'Aspremont 2011), stat-arb on PCA residuals after Avellaneda and Lee (2010), and many pairs through copulas; or one of four that score every book and hold a book of longs and shorts neutral to the dollar and to estimated factors: reversal, momentum, lead-lag and order flow (notebook 13 finds that, traded the whole way, only a slow reversal pays). Every kind can go only part of the way to its target at a decision (a trade rate) or not trade within a band of lots around it (a no-trade band), off by default but for reversal, whose default goes a tenth of the way at every decision, and baskets, whose default enters at a z-score of 1 and goes a quarter of the way, and a fit can try them: in notebook 13 that gives reversal, at a look back of 50, more than four times the P&L of its old default (look back 20, held for 100) with a rate and more than three times with a band or held longer, nearly doubles what the residuals make, adds little to pairs, and leaves order flow and lead-lag losing however they are traded less. Baskets chosen by Box-Tiao predictability earn about what Johansen's do at the same settings, and are chosen more than ten times as fast (notebook 13's section 6). Those settings, an entry of 1 and a trade rate of 0.25, earn Johansen's baskets of 3 about +2,000 more than an entry of 2 traded the whole way, which is why they are baskets' default. Watch every relation's signal (for those four, every book's score), what each trader holds and how much of the factors that takes on, and set what the cointegration tests find beside what is there. Fit any kind over training runs, as in Pairs: your settings and the combinations of the values the panel proposes trade alone in new markets on seeds the chart can never have, each is shown against yours with a standard error, and the panel says whether the best is clearly better; notebook 12 tests that fit on seeds none of it saw. The market's settings can also add, each off until you turn it on, an index book (`IDX`, one more book, the sum of the others and a premium of its own), stock deals (an acquirer offers shares for a target's, and the deal may break, as in merger arbitrage) and a trend in the factors, which gives momentum something to hold; the books' resting orders move with a deal's jump, so that no one trades it for free. Two more kinds of trader trade them, index arbitrage and a deals trader (after Mitchell and Pulvino 2001), and are refused, with the reason, in a market without what they trade (the deals trader has no fit: a few runs would rank its settings by how few breaks each met); notebook 15 asks how far each dial must be turned before its trade pays its costs.
- **Microstructure studies**: a band of eight tiles under the Trade view, each with a help, settings and a sparkline against the mid: order flow imbalance (Cont, Kukanov & Stoikov 2014), cumulative volume delta, delta of trades (the true side, the tick rule, or Lee and Ready's quote rule, with its coverage and its accuracy against the true side, which only a simulator can show), VPIN (true side, tick rule, against the mid, or bulk volume classification), order-to-trade ratio and trade-to-order volume, book imbalance (equal weights or weighted to the touch), the spread and its z-score, and depth depletion and recovery (after Large 2007). They run live and in a playback, cost about 3 microseconds an event, and follow the code of VisualHFT (Apache License 2.0) where their docstrings say so; where its code differs from its website (its order-to-trade ratio counts book changes, its VPIN classifies against the mid, its bands are on its website only) they say so too. Notebook 14 tests them against the truth.
- **Footprint and tape**: a toggle on the candle chart draws the volume traded at each price in each bar, sold into the bid on the left and bought from the ask on the right, with the busiest price outlined and diagonal imbalances marked (legible at intervals of about 15 units or more; Widen bars gives them room); the page checks that the cells add up to each bar's volume. The Recent Trades card is a tape with a fill number that is never reused, a size and a side filter, a pause with Jump to newest, the trade rate, and marks on the fills whose side the tick rule gets wrong.
- **Triggers**: rules on the studies' values (above, below, crosses above, crosses below; all of or any of; a cooldown in simulation time) that show a notice, pause the simulation, cancel your orders or flatten your position, checked after every event, in a playback too (where only the notice and the pause act); fired rules mark the chart.
- **Informed traders**: the sixth model has a hidden value that informed traders trade on, so toxicity has a ground truth: the model's bar says what share of the taker lots they sent and what makers lose to them, and Reveal draws the value on the chart.
- **Live updates** over one Server-Sent Events stream per tab, with a connection indicator. Each browser gets its own simulation.

### Launch Jupyter Notebooks

```bash
jupyter notebook notebooks/
make notebooks              # re-execute all fifteen in place
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
| 10 | [Pairs Trading](notebooks/10_pairs_trading.ipynb) | Engle & Granger (1987); Gatev et al. (2006) | Two order books whose gap mean-reverts: cointegration, the hedge ratio and the half-life as the books show them, and a pairs trade that pays its spreads, or does not, beside an arbitrageur |
| 11 | [A Dynamic Hedge and a Copula](notebooks/11_dynamic_hedge_and_copulas.ipynb) | Chan (2013); Liew & Wu (2013); Xie et al. (2014); Rad, Low & Faff (2016) | A Kalman hedge and a copula in the same books, with a drifting beta and a stressed regime to give them something to find: where the filter tracks and pays, what the copula finds and whether it matters, why its flags do not know where the gap's mean is, and whether fitting their settings helps on markets the fit has not seen |
| 12 | [Stat-Arb and Many Pairs](notebooks/12_stat_arb_and_many_pairs.ipynb) | Johansen (1991); Chan (2013); Avellaneda & Lee (2010); Gatev et al. (2006); Rad, Low & Faff (2016) | Many books with several common factors: how many relations the tests find, why a pair need not be cointegrated to pay, what a leg of a basket costs, what stat-arb on PCA residuals needs, how much breadth is worth at the same capital, and whether fitting a trader's settings over training runs, as the Stat-arb view does, helps on seeds the fit has not seen |
| 13 | [More Long/Short Strategies](notebooks/13_more_long_short_strategies.ipynb) | Lehmann (1990); Lo & MacKinlay (1990); Jegadeesh & Titman (1993); Hou (2007); Cont, Kukanov & Stoikov (2014); Gârleanu & Pedersen (2013); Davis & Norman (1990); Box & Tiao (1977); d'Aspremont (2011) | Stages 1 to 3 of four. What the books' moves remember (they bounce, continue, then revert, and nothing passes between the books once the factors are out), what four cross-sectional signals (reversal, momentum, lead-lag and order flow, each neutral to the dollar and to estimated factors) know apart from trading them, why in these books only a slow reversal pays after costs, and what trading a signal less does for it, by going only part of the way to its target or not trading within a band of it: more than four times the P&L for reversal with a rate (more than three times with a band or held longer), nearly twice for stat-arb on PCA residuals, little for pairs, and not enough for order flow or lead-lag to pay; and whether baskets chosen by Box-Tiao predictability (the least predictable few books, grown greedily) trade better than Johansen's (at the same settings they earn about the same, and are chosen more than ten times as fast) |
| 14 | [Microstructure Signals Against the Truth](notebooks/14_microstructure_signals.ipynb) | Cont, Kukanov & Stoikov (2014); Lee & Ready (1991); Easley, Lopez de Prado & O'Hara (2012); Andersen & Bondarenko (2014); Large (2007); Gould & Bonart (2016); Cont & de Larrard (2013); Avellaneda, Reed & Stoikov (2011); Glosten & Milgrom (1985); VisualHFT's research notes R-001 and R-002 and its code (Apache License 2.0) | Six studies of the dashboard, tested as VisualHFT's research tests its own, but where the truth is known (the true aggressor of every fill, and in the informed model a hidden value): the queue imbalance beats the reversal baseline by 5.6 to 11.1 pp in every model, and 50% flatters it; order flow imbalance explains the interval it falls in and forecasts almost nothing beyond the last change; the quote rule is exact at lag 0 by construction and bulk volume classification tracks the true buy share far worse than the tick rule; VPIN rises with one-sided flow whether or not anything is informed (the same VPIN earns makers 0.76 or 0.92 a lot); depth takes tens of units to refill (32.0 against 7.3 from the gross refill in ZI), which the net refill accounts for; a taker on the imbalance pays after costs only in the corner of its grid with the fewest entries, and only in ZI; and pulling a quote joiner's quotes on VPIN does no better than a placebo that pulls as much |
| 15 | [An Index, Deals and a Trend](notebooks/15_an_index_deals_and_a_trend.ipynb) | MacKinlay & Ramaswamy (1988); Stoll & Whaley (1990); Mitchell & Pulvino (2001); Jegadeesh & Titman (1993); Moskowitz & Grinblatt (1999); Moskowitz, Ooi & Pedersen (2012) | Stage 4 of the long/short work: three things the earlier markets lack, each off by default (with it off, notebooks 10 to 14 give the same numbers, to the tick) and added one at a time, each with a dial that starts at, or near, nothing to find: an index book (the sum of the others and a premium of its own: index arbitrage), stock deals that may break (a merger arbitrageur's many small gains and a few larger losses) and a drift in the factors that continues (momentum, which here can pay only by holding a factor). For each: how far must the dial be turned before the trade pays its costs, and what does its P&L look like when it does? On seeds 101 and up (means of eight to twelve seeds of one simulated market, with every setting chosen beforehand on seeds 1 to 8), the sum index breaks even about at the dial's lowest point, a premium of 14.7 ticks of a basket (+7, se 24; not bracketed, since the dial has no lower point), and the same position through the 1/12 index at 32.5 (31.6 to 33.4), the deal trader at a true chance of a break of about 0.31 (0.29 to 0.33) against a price of 0.40 (all three in the sectors market), and momentum, with the dollar alone taken out, at a drift of 0.0269 ticks a unit of time (0.0251 to 0.0289) in the sectors market and 0.0364 (0.0352 to 0.0374) in the mixed one. Each is for its own trader's size and costs and in its own unit, and the intervals are bootstraps over the seeds (5% to 95%, but 2.5% to 97.5% for the deal trader). |

The notebooks' conclusions are computed from the simulations against stated criteria, so re-running them (for example with other seeds) regenerates the verdicts. The backtests of notebooks 10 to 15 draw their scorecards and signal charts with [quant-plots](../../work/quant-plots), which `make install` installs as an editable package from `QUANT_PLOTS` (by default `../../work/quant-plots`).

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
├── studies.py        # Microstructure studies of the event stream: order flow imbalance, volume delta, VPIN, order-to-trade ratios, book imbalance, spread, resilience
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
