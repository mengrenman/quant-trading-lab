# Notebooks

Each section below gives a notebook's number and title (linked to the file), the paper section or references it rests on, and a description of what it asks and finds.

Run in order for the best learning experience.

The notebooks' conclusions are computed from the simulations against stated criteria, so re-running them (for example with other seeds) regenerates the verdicts. The backtests of notebooks 10 to 17 (notebook 18 measures and has none) draw their scorecards and signal charts with [quant-plots](../../../work/quant-plots), which `make install` installs as an editable package from `QUANT_PLOTS` (by default `../../work/quant-plots`).

## Contents

- [01. LOB Mechanics](#01-lob-mechanics)
- [02. Order Matching](#02-order-matching)
- [03. Zero Intelligence](#03-zero-intelligence)
- [04. Stylized Facts](#04-stylized-facts)
- [05. Agent-Based Models](#05-agent-based-models)
- [06. Open Questions](#06-open-questions)
- [07. Order Flow Imbalance](#07-order-flow-imbalance)
- [08. Calibration](#08-calibration)
- [09. Metaorder Impact](#09-metaorder-impact)
- [10. Pairs Trading](#10-pairs-trading)
- [11. A Dynamic Hedge and a Copula](#11-a-dynamic-hedge-and-a-copula)
- [12. Stat-Arb and Many Pairs](#12-stat-arb-and-many-pairs)
- [13. More Long/Short Strategies](#13-more-longshort-strategies)
- [14. Microstructure Signals Against the Truth](#14-microstructure-signals-against-the-truth)
- [15. An Index, Deals and a Trend](#15-an-index-deals-and-a-trend)
- [16. What Notebook 15 Left Open](#16-what-notebook-15-left-open)
- [17. Resting Orders](#17-resting-orders)
- [18. Real Books](#18-real-books)

## 01. [LOB Mechanics](../notebooks/01_lob_mechanics.ipynb)

**Paper section or references:** Section 2

Build an LOB from scratch; bid/ask/spread/mid-price; depth profiles; returns

## 02. [Order Matching](../notebooks/02_order_matching.ipynb)

**Paper section or references:** Section 3

Price-time vs pro-rata priority; fill sequences; state-space complexity

## 03. [Zero Intelligence](../notebooks/03_zero_intelligence.ipynb)

**Paper section or references:** Section 5

Smith et al. (2003) ZI model; emergent spread; model limitations

## 04. [Stylized Facts](../notebooks/04_stylized_facts.ipynb)

**Paper section or references:** Section 4

Heavy tails, volatility clustering, price impact, long memory

## 05. [Agent-Based Models](../notebooks/05_agent_based_models.ipynb)

**Paper section or references:** Section 5

Five models compared on seven stylized facts

## 06. [Open Questions](../notebooks/06_open_questions.ipynb)

**Paper section or references:** Section 6

Ablation studies: what causes heavy tails, volatility clustering, concave impact, long memory

## 07. [Order Flow Imbalance](../notebooks/07_order_flow_imbalance.ipynb)

**Paper section or references:** Cont, Kukanov & Stoikov (2014)

How much of the mid-price changes OFI explains in each model, and how its slope falls with depth

## 08. [Calibration](../notebooks/08_calibration.ipynb)

**Paper section or references:** Section 5.2.5; Mike & Farmer (2008)

Estimating the Mike & Farmer (2008) model's parameters from order flow, checked by recovering known parameters, also from data in LOBSTER's format (where the prices of orders that execute in full are hidden)

## 09. [Metaorder Impact](../notebooks/09_metaorder_impact.ipynb)

**Paper section or references:** Tóth et al. (2011); Bouchaud et al. (2018)

Does a large order's impact grow like the square root of its size? In these models it grows about linearly and does not revert

## 10. [Pairs Trading](../notebooks/10_pairs_trading.ipynb)

**Paper section or references:** Engle & Granger (1987); Gatev et al. (2006)

Two order books whose gap mean-reverts: cointegration, the hedge ratio and the half-life as the books show them, and a pairs trade that pays its spreads, or does not, beside an arbitrageur

## 11. [A Dynamic Hedge and a Copula](../notebooks/11_dynamic_hedge_and_copulas.ipynb)

**Paper section or references:** Chan (2013); Liew & Wu (2013); Xie et al. (2014); Rad, Low & Faff (2016)

A Kalman hedge and a copula in the same books, with a drifting beta and a stressed regime to give them something to find: where the filter tracks and pays, what the copula finds and whether it matters, why its flags do not know where the gap's mean is, and whether fitting their settings helps on markets the fit has not seen

## 12. [Stat-Arb and Many Pairs](../notebooks/12_stat_arb_and_many_pairs.ipynb)

**Paper section or references:** Johansen (1991); Chan (2013); Avellaneda & Lee (2010); Gatev et al. (2006); Rad, Low & Faff (2016)

Many books with several common factors: how many relations the tests find, why a pair need not be cointegrated to pay, what a leg of a basket costs, what stat-arb on PCA residuals needs, how much breadth is worth at the same capital, and whether fitting a trader's settings over training runs, as the Stat-arb view does, helps on seeds the fit has not seen

## 13. [More Long/Short Strategies](../notebooks/13_more_long_short_strategies.ipynb)

**Paper section or references:** Lehmann (1990); Lo & MacKinlay (1990); Jegadeesh & Titman (1993); Hou (2007); Cont, Kukanov & Stoikov (2014); Gârleanu & Pedersen (2013); Davis & Norman (1990); Box & Tiao (1977); d'Aspremont (2011)

Stages 1 to 3 of four.

What the books' moves remember (they bounce, continue, then revert, and nothing passes between the books once the factors are out), what four cross-sectional signals (reversal, momentum, lead-lag and order flow, each neutral to the dollar and to estimated factors) know apart from trading them, why in these books only a slow reversal pays after costs, and what trading a signal less does for it, by going only part of the way to its target or not trading within a band of it: more than four times the P&L for reversal with a rate (more than three times with a band or held longer), nearly twice for stat-arb on PCA residuals, little for pairs, and not enough for order flow or lead-lag to pay; and whether baskets chosen by Box-Tiao predictability (the least predictable few books, grown greedily) trade better than Johansen's (at the same settings they earn about the same, and are chosen more than ten times as fast)

## 14. [Microstructure Signals Against the Truth](../notebooks/14_microstructure_signals.ipynb)

**Paper section or references:** Cont, Kukanov & Stoikov (2014); Lee & Ready (1991); Easley, Lopez de Prado & O'Hara (2012); Andersen & Bondarenko (2014); Large (2007); Gould & Bonart (2016); Cont & de Larrard (2013); Avellaneda, Reed & Stoikov (2011); Glosten & Milgrom (1985); VisualHFT's research notes R-001 and R-002 and its code (Apache License 2.0)

Six studies of the dashboard, tested as VisualHFT's research tests its own, but where the truth is known (the true aggressor of every fill, and in the informed model a hidden value): the queue imbalance beats the reversal baseline by 5.6 to 11.1 pp in every model, and 50% flatters it; order flow imbalance explains the interval it falls in and forecasts almost nothing beyond the last change; the quote rule is exact at lag 0 by construction and bulk volume classification tracks the true buy share far worse than the tick rule; VPIN rises with one-sided flow whether or not anything is informed (the same VPIN earns makers 0.76 or 0.92 a lot); depth takes tens of units to refill (32.0 against 7.3 from the gross refill in ZI), which the net refill accounts for; a taker on the imbalance pays after costs only in the corner of its grid with the fewest entries, and only in ZI; and pulling a quote joiner's quotes on VPIN does no better than a placebo that pulls as much

## 15. [An Index, Deals and a Trend](../notebooks/15_an_index_deals_and_a_trend.ipynb)

**Paper section or references:** MacKinlay & Ramaswamy (1988); Stoll & Whaley (1990); Mitchell & Pulvino (2001); Jegadeesh & Titman (1993); Moskowitz & Grinblatt (1999); Moskowitz, Ooi & Pedersen (2012)

Stage 4 of the long/short work: three things the earlier markets lack, each off by default (with it off, notebooks 10 to 14 give the same numbers, to the tick) and added one at a time, each with a dial that starts at, or near, nothing to find: an index book (the sum of the others and a premium of its own: index arbitrage), stock deals that may break (a merger arbitrageur's many small gains and a few larger losses) and a drift in the factors that continues (momentum, which here can pay only by holding a factor).

For each: how far must the dial be turned before the trade pays its costs, and what does its P&L look like when it does?

On seeds 101 and up (means of eight to twelve seeds of one simulated market, with every setting chosen beforehand on seeds 1 to 8), the sum index breaks even about at the dial's lowest point, a premium of 14.7 ticks of a basket (+7, se 24; not bracketed, since the dial has no lower point), and the same position through the 1/12 index at 32.5 (31.6 to 33.4), the deal trader at a true chance of a break of about 0.31 (0.29 to 0.33) against a price of 0.40 (all three in the sectors market), and momentum, with the dollar alone taken out, at a drift of 0.0269 ticks a unit of time (0.0251 to 0.0289) in the sectors market and 0.0364 (0.0352 to 0.0374) in the mixed one. Each is for its own trader's size and costs and in its own unit, and the intervals are bootstraps over the seeds (5% to 95%, but 2.5% to 97.5% for the deal trader).

## 16. [What Notebook 15 Left Open](../notebooks/16_what_notebook_15_left_open.ipynb)

**Paper section or references:** Mitchell & Pulvino (2001); Gatev, Goetzmann & Rouwenhorst (2006); Avellaneda & Lee (2010)

The three things notebook 15 could not settle, on seeds it did not use (means over seeds of one simulated market, no setting chosen on any of them).

Does reading the news pay a pairs trader on its own pairs, where a break is overpriced? A test written before the run, on 24 new seeds, is met (the reader leads the blind trader by +576, se 116, t 4.98, 19 of 24 seeds positive, which is far from 0 and not a size), and the lead is the deal part that the blind trader loses in the targets of deals and the reader avoids (+911, about the expected bet of +886) less a cost of reading (-335, se 103) that does not move with the chance of a break.

Why are the pairs chosen in the market with a risk premium worth less? On a route with no order books, on 200 seeds, the completed deals' steps do it, through the choice of pairs and the hedge together (+442, se 70, of a similar size to notebook 15's +446, and +67, se 31, with both cleaned), but 24 new seeds do not show the gap in the market (+38, se 177) or on the route (-94, se 186), and no cause was found.

Would sector index books let a residual hedge with one leg, more cheaply? Not at this market's price: an index lot costs 2.98 book lots against a break-even of 2, and the cheapest mix saves 0.01% of the crossing cost (13.27% at a book's price).

## 17. [Resting Orders](../notebooks/17_resting_orders.ipynb)

**Paper section or references:** Perold (1988); Gould et al. (2013); Gârleanu & Pedersen (2013)

Every finding of notebooks 12 to 16 came down to what crossing the spread costs; here the same traders rest their orders at the best price instead (`lob/passive.py`), on seeds 101 to 112 (means over seeds of one simulated market, no setting chosen on any of them), measured against the mid at the decision: the position a trader asked for at each decision, got at once at that decision's mid, is its paper, and paper less P&L is the shortfall, which `lob/passive.py` splits into the lots got, taken back, unasked and open.

What does a lot cost when it rests? A one-lot order resting at the best price for up to 20 units cost -0.221 a lot (se 0.006) against the decision mid, where a market order cost +0.989 (se 0.005), and five lots cost -0.036 (se 0.008) against +1.240 (se 0.005), in a market built to be kind to a resting order (the order flow never reacts to it, no one steps in front of it, cancel and replace are free and instant, an order alone at the best price steps back for nothing, a jump is always seen in time, there are no fees, and the takers carry 0.05 a lot of information) and not as a prediction for a real one; at an impact of 0.5, which moves the takers' information and none of the other items, the savings are +1.027 (se 0.019) and +0.870 (se 0.028).

How much of the cost is the signal's own? The order in the signal's direction paid more than the one against it for the queues, reversal, pairs and residuals, on every seed at 0.05, and paid less for momentum, whose sign is the same at both impacts and is not explained.

What becomes of four traders when their orders rest? Every worked trader beat its crossing twin on every seed, by 0.82 to 1.19 times what the twin paid to cross, which is the identity of the gain's three terms; the queue row stayed a loss (-266, se 55), its lots taken back being why its paper does not reach its P&L; and worked pairs and baskets held about their twins' lots and 1.56 and 2.19 times their factor exposure, the excess being the gap between position and paper. 57 of 63 predictions, written from seeds 1 to 8 before the run, held.

The library leaves nine earlier files byte-identical, and a market whose traders have no execution is the old market to the last digit. The dashboard has no execution setting: the Stat-arb view still trades with market orders.

## 18. [Real Books](../notebooks/18_real_books.ipynb)

**Paper section or references:** Nasdaq TotalView-ITCH 5.0 specification; Cont, Kukanov & Stoikov (2014); Lee & Ready (1991); Easley, Lopez de Prado & O'Hara (2012); Andersen & Bondarenko (2014); Large (2007); Gould & Bonart (2016); Cont & de Larrard (2013); Avellaneda, Reed & Stoikov (2011); Gould et al. (2013)

Notebook 14's measurements, with the same study classes, on real Nasdaq order books, order by order (`lob/itch.py`, `lob/feeds.py`, `lob/ladder.py`): INTC, MSFT, CSCO, AAPL and AMZN on 2019-03-27 and again on 2019-07-30, a day nobody read until the code was frozen. It is a measurement with no strategy and no backtest, and the five stocks share each day's market factor, so no count of stocks is five independent results and no two stocks are said to differ.

**It needs the Nasdaq files, which are not in the repository (`data/itch/`, ignored by git; no terms of use are stated beside them, so they and everything derived from them are never published), and without them it prints one line a cell and finishes. `make notebook-18` runs it on the files (the four runs took 48, 47, 57 and 56 minutes on 5 workers); it is not part of `make notebooks`.**

Which of notebook 14's findings appear on real books? On INTC, MSFT and CSCO the one-tick hit rate and reversal rate are nearly one measurement, so the notebook does not say the imbalance beats the reversal rate at one tick (where the calls differ the lift is -1.49 on INTC's first day, +1.79 on MSFT's second and +6.10 on AAPL's second); order flow imbalance's median half-hour R2 at 10 s is 0.818 to 0.909 on eight stock-days and 0.473 | 0.699 on AMZN; and VPIN's excess over a fair-coin null is above 0 at every origin on all ten stock-days and is how persistent the order signs are, not informed trading.

Which are the estimator's? The quote rule at lag 0 is exact at the resting price by construction, pairing a sample with the change it has just seen agrees in sign less often than pairing it with the next unit's on all ten stock-days, bulk volume classification reads price changes and not flow, and the quote rule at lag 1 beats the tick rule on every stock-day but AMZN's two.

What does a price-level feed lose? The quote rule at lag 0, per lot, is lower at 100 ms than on the exchange's own stream by 4.7 to 12.7 points and the median half-hour R2 is lower on all ten stock-days; no direction is claimed for feeds in general.

What must the simulator's model be to look like a real day? Its fitted `imbalance_offset` is at the fit's bound on all ten stock-days, it has x6.9 to x13.1 the day's aggressive orders and it reproduces no depth: it does not look like a real day.

Ten predictions written before any reported stock was read are judged a stock and a day, never counted across tests, and a verdict that sits on its line is said to.

Two reviews of the first run found no error that changes a verdict; the frozen code's combined hash begins `2b220ac5`, and 17 of its 50 code cells were changed afterward in labels and drawing only, and `lob/ladder.py` was corrected after the third run (an execution with price was keyed at its own price and not at the level it took: ten printed counts of the reported stocks moved, and no rate, lift, R2, accuracy or verdict of a reported stock at any printed digit), as the notebook lists; the outputs it holds are the fourth run's. The freeze's manifest, the first run's notebook, the list of what changed after it and a script that checks them are in [`records/notebook_18/`](../records/notebook_18/).
