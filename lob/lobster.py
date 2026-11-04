"""Order-book data in LOBSTER's format, and the Mike and Farmer (2008) estimates from it.

LOBSTER (lobsterdata.com) reconstructs Nasdaq's order book from its ITCH messages. A day's data
come as two CSV files without headers:

- the **message file**, one row per event: time (seconds after midnight), type (1 a new limit
  order, 2 a partial cancellation, 3 a full cancellation, 4 an execution of a visible order, 5 an
  execution of a hidden order, 6 a cross trade, 7 a trading halt), order ID, size, price (dollars
  times 10,000) and direction (1 a buy order, -1 a sell order; for an execution, the side of the
  resting order, so the opposite of the side that initiated the trade);
- the **order-book file**, the book after each message: ask price 1, ask size 1, bid price 1, bid
  size 1, and so on for the file's number of levels (an empty level has the price 9999999999 or
  -9999999999).

Two things set such data apart from the simulator's event log, and the estimators here allow for
both. First, an order that executes in full on arrival leaves only its executions in the file:
its own limit price is never recorded, only that it was at least as aggressive as the worst price
it traded at. The price fit treats those offsets as censored at that bound, which is exact for the
model: with its fixed order size the fills depend on the limit price only through that bound.
Second, a day's file starts with a book full of orders placed before it, whose placement (which
the cancellation model needs) is unknown: the cancellation replay leaves them out, and counts
only after a burn-in, by when most resting orders were placed within the file.

``simulated_lobster`` writes LOBSTER-format data from a simulation, hiding what Nasdaq's feed
hides, so the whole path can be checked against known parameters (tests/test_lobster.py, notebook
08). Real files are read with ``read_lobster`` and estimated with ``calibrate_lobster``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from scipy import optimize, stats

from lob.book import OrderBook
from lob.calibration import _RestingSide, estimate_sign_hurst, fit_cancellation
from lob.matching import PriceTimeEngine
from lob.order import OrderType, Side

MESSAGE_COLUMNS = ["time", "type", "order_id", "size", "price", "direction"]
SUBMISSION, PARTIAL_CANCEL, DELETION, EXECUTION, HIDDEN_EXECUTION = 1, 2, 3, 4, 5
EMPTY_ASK, EMPTY_BID = 9_999_999_999, -9_999_999_999
LOBSTER_TICK = 100  # a cent, in LOBSTER's units of dollars times 10,000
DAY_START = 34_200.0  # 9:30, in seconds after midnight


def book_columns(levels: int) -> list[str]:
    return [f"{kind}_{i}" for i in range(1, levels + 1) for kind in ("ask_price", "ask_size", "bid_price", "bid_size")]


# ---- Reading and writing the files ----

def read_lobster(message_file: str | Path, orderbook_file: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A message file and its order-book file: the messages, and the book after each (empty levels as NaN)."""
    messages = pd.read_csv(message_file, header=None).iloc[:, :6]
    messages.columns = MESSAGE_COLUMNS
    book = pd.read_csv(orderbook_file, header=None)
    book.columns = book_columns(book.shape[1] // 4)
    for column in book.columns:
        if column.startswith("ask_price"):
            book.loc[book[column] >= EMPTY_ASK, column] = np.nan
        elif column.startswith("bid_price"):
            book.loc[book[column] <= EMPTY_BID, column] = np.nan
    return messages, book


def save_lobster(messages: pd.DataFrame, book: pd.DataFrame, directory: str | Path,
                 ticker: str = "SIM", date: str = "2012-06-21") -> tuple[Path, Path]:
    """Write the two files under LOBSTER's names (empty levels with its placeholder prices)."""
    directory = Path(directory)
    levels = book.shape[1] // 4
    stem = f"{ticker}_{date}_34200000_57600000"
    message_file = directory / f"{stem}_message_{levels}.csv"
    orderbook_file = directory / f"{stem}_orderbook_{levels}.csv"
    messages[MESSAGE_COLUMNS].to_csv(message_file, header=False, index=False)
    out = book.copy()
    for column in out.columns:
        if column.startswith("ask_price"):
            out[column] = out[column].fillna(EMPTY_ASK)
        elif column.startswith("bid_price"):
            out[column] = out[column].fillna(EMPTY_BID)
    out.astype("int64").to_csv(orderbook_file, header=False, index=False)
    return message_file, orderbook_file


def simulated_lobster(make_model: Callable[[], object], n_events: int, seed: int, levels: int = 10,
                      skip_events: int = 0, tick: int = LOBSTER_TICK) -> tuple[pd.DataFrame, pd.DataFrame]:
    """LOBSTER-format messages and book from a simulation that starts from an empty book.

    Each order's executions are written as executions of the resting orders it hit (with their
    side as the direction), followed by a new-order message for any part of it that rests, so an
    order that executes in full leaves no record of its own price, as in Nasdaq's feed. The first
    ``skip_events`` events run but are not written: the file then begins, like a real day's, with
    a book full of orders it does not show being placed. Times are seconds after 9:30. Each
    message is followed by the book after its whole event (LOBSTER's files show it after the
    message itself; the estimators read only the book before an order and after an event, where
    the two agree).
    """
    model, book, engine = make_model(), OrderBook(), PriceTimeEngine()
    rng = np.random.default_rng(seed)
    messages: list[tuple] = []
    rows: list[list[float]] = []

    def snapshot() -> list[float]:
        asks = book.depth_profile(Side.ASK, n_levels=levels)
        bids = book.depth_profile(Side.BID, n_levels=levels)
        row: list[float] = []
        for i in range(levels):
            row += [asks[i][0] * tick, asks[i][1]] if i < len(asks) else [np.nan, 0]
            row += [bids[i][0] * tick, bids[i][1]] if i < len(bids) else [np.nan, 0]
        return row

    t, events = 0.0, 0
    while events < skip_events + n_events:
        order = model.generate_event(book, t, rng)
        if order is None:
            continue
        t = max(t, order.timestamp)
        events += 1
        write = events > skip_events
        time = DAY_START + t
        if order.order_type == OrderType.CANCEL:
            resting = book.get_order(order.order_id)
            if resting is None:
                continue
            book.cancel_order(order.order_id)
            if write:
                messages.append((time, DELETION, resting.order_id, resting.size, resting.price * tick,
                                 1 if resting.side == Side.BID else -1))
                rows.append(snapshot())
            continue
        fills = engine.process(book, order)
        if not write:
            continue
        resting_direction = -1 if order.side == Side.BID else 1
        for fill in fills:
            messages.append((time, EXECUTION, fill.maker_order_id, fill.size, fill.price * tick, resting_direction))
        rest = book.get_order(order.order_id)
        if rest is not None:
            messages.append((time, SUBMISSION, order.order_id, rest.size, order.price * tick, -resting_direction))
        after = snapshot()
        rows += [after] * (len(fills) + (rest is not None))
    return (pd.DataFrame(messages, columns=MESSAGE_COLUMNS),
            pd.DataFrame(rows, columns=book_columns(levels)))


# ---- From messages to orders ----

def _quotes(book: pd.DataFrame, tick: int) -> tuple[np.ndarray, np.ndarray]:
    """The best bid and ask after each message, in ticks (NaN for an empty side)."""
    return book["bid_price_1"].to_numpy(float) / tick, book["ask_price_1"].to_numpy(float) / tick


def lobster_orders(messages: pd.DataFrame, book: pd.DataFrame, tick: int = LOBSTER_TICK) -> pd.DataFrame:
    """One row per order that arrived, in time order.

    A new limit order arrives with its price. Executions at one time against resting orders of
    one side are one aggressive order from the other side, of their total size; if a new order of
    its side follows at the same time, that is the part of it that rested, so its price is known.
    If not, it executed in full, and its price is known only to be at least as aggressive as the
    worst price it traded at.

    Columns: time, side ('bid' or 'ask'), size, price (ticks; NaN for an order that executed in
    full), worst (for an aggressive order, its worst price), same and opposite (the best quotes on
    its side and the other just before it arrived, in ticks; NaN when empty), executed (lots traded
    on arrival), order_id (of the part that rested, or -1), rest_index (that part's message index,
    or -1).
    """
    bid, ask = _quotes(book, tick)
    t = messages["time"].to_numpy(float)
    kind = messages["type"].to_numpy(int)
    oid = messages["order_id"].to_numpy(np.int64)
    size = messages["size"].to_numpy(np.int64)
    price = messages["price"].to_numpy(float) / tick
    direction = messages["direction"].to_numpy(int)
    rows = []
    i, n = 0, len(messages)
    while i < n:
        before_bid = bid[i - 1] if i > 0 else math.nan
        before_ask = ask[i - 1] if i > 0 else math.nan
        if kind[i] in (EXECUTION, HIDDEN_EXECUTION):
            resting = direction[i]
            buying = resting == -1  # sell orders executed: a buyer initiated
            j, lots, worst = i, 0, price[i]
            while j < n and kind[j] in (EXECUTION, HIDDEN_EXECUTION) and t[j] == t[i] and direction[j] == resting:
                lots += size[j]
                worst = max(worst, price[j]) if buying else min(worst, price[j])
                j += 1
            rest = j if j < n and kind[j] == SUBMISSION and t[j] == t[i] and direction[j] == -resting else -1
            rows.append((t[i], "bid" if buying else "ask", lots + (size[rest] if rest >= 0 else 0),
                         price[rest] if rest >= 0 else math.nan, worst,
                         before_bid if buying else before_ask, before_ask if buying else before_bid,
                         lots, oid[rest] if rest >= 0 else -1, rest))
            i = rest + 1 if rest >= 0 else j
        elif kind[i] == SUBMISSION:
            buying = direction[i] == 1
            rows.append((t[i], "bid" if buying else "ask", size[i], price[i], math.nan,
                         before_bid if buying else before_ask, before_ask if buying else before_bid,
                         0, oid[i], i))
            i += 1
        else:
            i += 1
    return pd.DataFrame(rows, columns=["time", "side", "size", "price", "worst", "same", "opposite",
                                       "executed", "order_id", "rest_index"])


def order_offsets(orders: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Offsets from the same-side best quote (positive: less aggressive), as in ``relative_offsets``.

    Returns the exact offsets of orders whose price is known, and for orders that executed in full
    the bound their offset is at most. Orders that arrived with their own side empty are left out.
    """
    buy = (orders["side"] == "bid").to_numpy()
    same = orders["same"].to_numpy(float)
    price = orders["price"].to_numpy(float)
    worst = orders["worst"].to_numpy(float)
    quoted = np.isfinite(same)
    known = quoted & np.isfinite(price)
    exact = np.where(buy, same - price, price - same)[known]
    full = quoted & ~np.isfinite(price)
    bound = np.where(buy, same - worst, worst - same)[full]
    # Hidden orders can trade at half a tick (the midpoint): an offset in whole ticks of at most
    # -2.5 is at most -3.
    return np.rint(exact).astype(int), np.floor(bound).astype(int)


def fit_rounded_t_censored(exact: np.ndarray, bounds: np.ndarray) -> dict[str, float]:
    """Maximum likelihood (nu, loc, scale) of round(loc + scale * t_nu), with some draws known only to be <= a bound."""
    values, counts = np.unique(exact, return_counts=True)
    limits, limit_counts = np.unique(bounds, return_counts=True)

    def neg_log_likelihood(theta: np.ndarray) -> float:
        nu, loc, scale = math.exp(theta[0]), theta[1], math.exp(theta[2])
        mass = stats.t.cdf((values + 0.5 - loc) / scale, nu) - stats.t.cdf((values - 0.5 - loc) / scale, nu)
        tail = stats.t.cdf((limits + 0.5 - loc) / scale, nu)  # P(round(X) <= b) = P(X < b + 1/2)
        return float(-(counts * np.log(np.maximum(mass, 1e-300))).sum()
                     - (limit_counts * np.log(np.maximum(tail, 1e-300))).sum())

    start = np.array([math.log(2.0), float(np.median(exact)), math.log(max(1.0, stats.iqr(exact) / 2))])
    theta = optimize.minimize(neg_log_likelihood, start, method="Nelder-Mead",
                              options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 4000}).x
    return {"nu": math.exp(theta[0]), "loc": float(theta[1]), "scale": math.exp(theta[2])}


# ---- Cancellations ----

def lobster_cancellation_statistics(messages: pd.DataFrame, book: pd.DataFrame, orders: pd.DataFrame,
                                    tick: int = LOBSTER_TICK, burn_in: float = 0.0,
                                    partial: str = "drop") -> dict[str, Any]:
    """What ``fit_cancellation`` needs, from a replay of the messages over the orders the file shows placed.

    The same sums as ``lob.calibration.cancellation_statistics``, with the best quotes taken from
    the order-book file and the resting orders followed through their own messages. Orders the
    file does not show placed are left out, and nothing is counted before ``burn_in`` seconds.
    A partial cancellation (which the model does not have) only shrinks its order with
    ``partial="drop"``, and counts as a cancellation with ``partial="full"``.
    """
    bid, ask = _quotes(book, tick)
    t = messages["time"].to_numpy(float)
    kind = messages["type"].to_numpy(int)
    oid = messages["order_id"].to_numpy(np.int64)
    size = messages["size"].to_numpy(np.int64)
    price = messages["price"].to_numpy(float) / tick
    sides = {Side.BID: _RestingSide(), Side.ASK: _RestingSide()}
    count = {Side.BID: 0, Side.ASK: 0}
    known: dict[int, list] = {}  # order ID -> [side, lots left]

    # Each resting order's placement distance, from the other side's quote before it arrived (with the
    # model's stand-ins when that side was empty), at the index of its new-order message.
    placed_at: dict[int, float] = {}
    for row in orders.itertuples(index=False):
        if row.rest_index < 0:
            continue
        opposite, same = row.opposite, row.same
        if row.side == "bid":
            if not math.isfinite(opposite):
                opposite = same + 1 if math.isfinite(same) else row.price + 1
            placed_at[row.rest_index] = max(1.0, opposite - row.price)
        else:
            if not math.isfinite(opposite):
                opposite = same - 1 if math.isfinite(same) else row.price - 1
            placed_at[row.rest_index] = max(1.0, row.price - opposite)

    def nan_none(x: float) -> Optional[float]:
        return None if math.isnan(x) else x

    start = t[0] + burn_in
    w_u = w_v = 0.0
    weights, shares = [], []
    skipped = 0
    last_unknown = math.nan  # when an order the file did not show placed last traded or left

    def drop(order_id: int) -> None:
        side, _ = known.pop(order_id)
        sides[side].remove(order_id)
        count[side] -= 1

    n = len(messages)
    for i in range(n):
        k, order_id = kind[i], int(oid[i])
        if k == SUBMISSION and i in placed_at:
            side = Side.BID if messages["direction"].iat[i] == 1 else Side.ASK
            sides[side].add(order_id, price[i], placed_at[i])
            known[order_id] = [side, int(size[i])]
            count[side] += 1
        elif k in (DELETION, PARTIAL_CANCEL):
            if order_id in known:
                side, left = known[order_id]
                full = k == DELETION or partial == "full" or size[i] >= left
                if full:
                    if t[i] >= start:
                        opposite = nan_none(ask[i - 1] if side == Side.BID else bid[i - 1]) if i > 0 else None
                        weights.append(sides[side].weight_of(order_id, opposite))
                        shares.append(count[side] / (count[Side.BID] + count[Side.ASK]))
                    drop(order_id)
                else:
                    known[order_id][1] = left - int(size[i])
            elif k == DELETION:
                last_unknown = t[i]
                if t[i] >= start:
                    skipped += 1  # placed before the file began
        elif k == EXECUTION:
            if order_id in known:
                known[order_id][1] -= int(size[i])
                if known[order_id][1] <= 0:
                    drop(order_id)
            else:
                last_unknown = t[i]
        # The state now holds until the next message.
        if i + 1 < n and t[i] >= start:
            dt = t[i + 1] - t[i]
            total = count[Side.BID] + count[Side.ASK]
            if dt > 0 and total > 0:
                y_b = sides[Side.BID].weight(nan_none(ask[i]))
                y_a = sides[Side.ASK].weight(nan_none(bid[i]))
                w_u += dt * (count[Side.BID] / total * y_b + count[Side.ASK] / total * y_a)
                w_v += dt * (y_b + y_a)
    return {"w_u": w_u, "w_v": w_v, "weight": np.array(weights), "share": np.array(shares), "skipped": skipped,
            "last_unknown": last_unknown - t[0] if math.isfinite(last_unknown) else None}


def calibrate_lobster(messages: pd.DataFrame, book: pd.DataFrame, tick: int = LOBSTER_TICK,
                      burn_in: float = 0.0, partial: str = "drop") -> dict[str, Any]:
    """Every parameter of ``MikeFarmer2008Model`` from LOBSTER data, with what the data held back.

    ``lam`` is in orders per second; prices are in ticks of ``tick`` (a cent by default).
    """
    orders = lobster_orders(messages, book, tick)
    exact, bounds = order_offsets(orders)
    cancel_stats = lobster_cancellation_statistics(messages, book, orders, tick, burn_in, partial)
    span = messages["time"].iloc[-1] - messages["time"].iloc[0]
    kinds = messages["type"]
    return {
        "lam": len(orders) / span if span > 0 else math.nan,
        **fit_cancellation(cancel_stats),
        "hurst": estimate_sign_hurst(np.where(orders["side"] == "bid", 1.0, -1.0)),
        **fit_rounded_t_censored(exact, bounds),
        "lot_size": int(orders["size"].mode().iloc[0]),
        "n_orders": int(len(orders)),
        "n_executed_in_full": int(len(bounds)),  # aggressive orders whose price the data do not show
        "n_cancels": int(len(cancel_stats["share"])),
        "n_cancels_skipped": int(cancel_stats["skipped"]),  # of orders placed before the file began
        # Seconds into the file when the last such order traded or left: a burn-in this long excludes them all.
        "last_unknown": cancel_stats["last_unknown"],
        "n_partial_cancels": int((kinds == PARTIAL_CANCEL).sum()),
        "n_hidden_executions": int((kinds == HIDDEN_EXECUTION).sum()),
    }
