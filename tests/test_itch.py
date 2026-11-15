"""Tests for reading Nasdaq ITCH 5.0 and converting it to LOBSTER's frames (lob/itch.py).

Most tests write ITCH bytes by hand (``Feed`` below) and need no data file. The same hand writes both ends of
those, so they hold the code together; they are not the reference. The reference is two independent decoders
of the real day, which the last tests compare with (they skip without the downloaded files).
"""

import gzip
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lob import itch
from lob.itch import ItchError, lobster_frames, read_messages
from lob.lobster import MESSAGE_COLUMNS, book_columns, calibrate_lobster, lobster_orders, simulated_lobster
from lob.models.mike_farmer import MikeFarmer2008Model
from lob.order import reset_order_id_counter

T0 = 34_200 * 10**9  # 09:30:00 in nanoseconds after midnight
ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "itch" / "derived" / "03272019"
NO_DAY = "needs data/itch/derived/03272019/raw/NVDA.itch and reference/nvda_rows.npz (ignored by git, not present)"


def T(microseconds: float) -> int:
    return T0 + round(microseconds * 1000)


def P(dollars: float) -> int:
    return round(dollars * 10_000)


# ---- A writer of ITCH 5.0 messages ----

def _head(kind: str, locate: int, ns: int) -> bytes:
    return kind.encode() + struct.pack(">HH", locate, 0) + ns.to_bytes(6, "big")


def _stock(name: str) -> bytes:
    return name.ljust(8).encode()


def add(ns, ref, side, shares, price, stock="TEST", locate=1, mpid=None):
    body = _head("F" if mpid else "A", locate, ns) + struct.pack(">QcI", ref, side.encode(), shares) + _stock(stock)
    return body + struct.pack(">I", price) + (mpid.ljust(4).encode() if mpid else b"")


def execute(ns, ref, shares, match, locate=1):
    return _head("E", locate, ns) + struct.pack(">QIQ", ref, shares, match)


def execute_with_price(ns, ref, shares, match, printable, price, locate=1):
    return _head("C", locate, ns) + struct.pack(">QIQcI", ref, shares, match, printable.encode(), price)


def cancel(ns, ref, shares, locate=1):
    return _head("X", locate, ns) + struct.pack(">QI", ref, shares)


def delete(ns, ref, locate=1):
    return _head("D", locate, ns) + struct.pack(">Q", ref)


def replace(ns, ref, new_ref, shares, price, locate=1):
    return _head("U", locate, ns) + struct.pack(">QQII", ref, new_ref, shares, price)


def trade(ns, shares, price, match, side="B", stock="TEST", locate=1):
    return (_head("P", locate, ns) + struct.pack(">QcI", 0, side.encode(), shares) + _stock(stock)
            + struct.pack(">IQ", price, match))


def cross(ns, shares, price, match, kind="O", stock="TEST", locate=1):
    return (_head("Q", locate, ns) + struct.pack(">Q", shares) + _stock(stock)
            + struct.pack(">IQc", price, match, kind.encode()))


def action(ns, state="T", stock="TEST", locate=1):
    return _head("H", locate, ns) + _stock(stock) + state.encode() + b" " + b"    "


def system(ns, code):
    return _head("S", 0, ns) + code.encode()


def directory(ns, stock, locate):
    return (_head("R", locate, ns) + _stock(stock)).ljust(39, b"\0")


def filler(kind, ns, locate=1):
    """A message of a type this module does not read (its body is zeros): I, L, Y, V, K and so on."""
    return _head(kind, locate, ns).ljust(itch.MESSAGE_LENGTHS[kind], b"\0")


def framed(*messages):
    return b"".join(struct.pack(">H", len(m)) + m for m in messages)


def read(tmp_path, *messages, name="TEST.itch"):
    path = tmp_path / name
    path.write_bytes(framed(*messages))
    return read_messages(path)


def frames(tmp_path, *messages, **options):
    """read_messages and lobster_frames on hand-made messages, from the start of the day's window."""
    return lobster_frames(read(tmp_path, *messages), **options)


# ---- Decoding ----

def test_every_message_type_of_a_book_is_decoded(tmp_path):
    day = read(tmp_path,
               system(T(0), "Q"), action(T(1)), add(T(2), 11, "B", 100, P(10.00)),
               add(T(3), 12, "S", 55, P(10.02), mpid="GSCO"), execute(T(4), 12, 5, 77),
               execute_with_price(T(5), 11, 20, 78, "Y", P(10.01)), execute_with_price(T(6), 11, 20, 79, "N", P(10.00)),
               cancel(T(7), 11, 3), delete(T(8), 12), replace(T(9), 11, 13, 40, P(9.99)),
               trade(T(10), 300, P(10.005), 80), cross(T(11), 138_507, P(10.01), 81, "C"),
               filler("I", T(12)), filler("L", T(13)), filler("Y", T(14)), directory(T(15), "TEST", 1))
    assert list(day.columns) == [
        "ns", "type", "ref", "new_ref", "side", "shares", "price", "match", "printable", "code"]
    assert day["type"].tolist() == list("SHAFECCXDUPQ")  # I, L, Y and R are skipped
    assert day["ns"].tolist() == [T(i) for i in range(12)]
    assert day.loc[day["type"] == "U", ["ref", "new_ref", "shares", "price"]].values.tolist() == [[11, 13, 40, P(9.99)]]
    assert day.loc[day["type"] == "F", ["ref", "side", "shares", "price"]].values.tolist() == [[12, -1, 55, P(10.02)]]
    assert day.loc[day["type"] == "A", ["ref", "side", "shares", "price"]].values.tolist() == [[11, 1, 100, P(10.00)]]
    assert day.loc[day["type"] == "E", ["ref", "shares", "match"]].values.tolist() == [[12, 5, 77]]
    assert day.loc[day["type"] == "C", ["ref", "shares", "match", "price", "printable"]].values.tolist() == [
        [11, 20, 78, P(10.01), "Y"], [11, 20, 79, P(10.00), "N"]]
    assert day.loc[day["type"] == "X", ["ref", "shares"]].values.tolist() == [[11, 3]]
    assert day.loc[day["type"] == "D", "ref"].tolist() == [12]
    assert day.loc[day["type"] == "P", ["side", "shares", "price", "match"]].values.tolist() == [
        [1, 300, P(10.005), 80]]
    assert day.loc[day["type"] == "Q", ["shares", "price", "match", "code"]].values.tolist() == [
        [138_507, P(10.01), 81, "C"]]
    assert day.loc[day["type"] == "S", "code"].tolist() == ["Q"]
    assert day.loc[day["type"] == "H", "code"].tolist() == ["T"]
    assert day.attrs == {"symbol": "TEST", "locate": 1}
    assert day.loc[day["type"] != "C", "printable"].eq("").all()


def test_values_above_the_signed_range_of_a_four_byte_field_survive(tmp_path):
    day = read(tmp_path, add(T(0), 2**40 + 5, "B", 4_000_000_000, 4_000_000_000))
    assert day.loc[0, ["ref", "shares", "price"]].tolist() == [2**40 + 5, 4_000_000_000, 4_000_000_000]
    assert (day[["ns", "ref", "shares", "price", "match", "side"]].dtypes == "int64").all()


@pytest.mark.parametrize("name, messages, match", [
    ("unknown type", [_head("Z", 1, T(0)).ljust(20, b"\0")], "type 'Z'"),
    ("wrong length for the type", [add(T(0), 1, "B", 1, 1000000)[:-1]], "length 35"),
    ("time runs backwards", [add(T(5), 1, "B", 1, 1000000), add(T(4), 2, "B", 1, 1000000)], "backwards"),
    ("two locate codes", [add(T(0), 1, "B", 1, 1000000), add(T(1), 2, "B", 1, 1000000, locate=2)],
     "more than one stock locate"),
    ("two stock names", [add(T(0), 1, "B", 1, 1000000), add(T(1), 2, "B", 1, 1000000, stock="OTHR")],
     "more than one stock:"),
    ("bad side byte", [add(T(0), 1, "X", 1, 1000000)], "side byte"),
    ("bad printable byte", [execute_with_price(T(0), 1, 1, 1, "?", 1000000)], "printable byte"),
    ("a broken trade", [filler("B", T(0))], "type 'B'"),
    ("an operational halt", [filler("h", T(0))], "type 'h'"),
])
def test_a_file_that_is_not_what_the_specification_says_raises(tmp_path, name, messages, match):
    path = tmp_path / "bad.itch"
    if name == "wrong length for the type":  # the prefix says 36 and the message has 35 bytes: the next prefix is eaten
        path.write_bytes(struct.pack(">H", 35) + messages[0])
    else:
        path.write_bytes(framed(*messages))
    with pytest.raises(ItchError, match=match):
        read_messages(path)


def test_a_file_that_ends_inside_a_message_raises(tmp_path):
    path = tmp_path / "cut.itch"
    path.write_bytes(framed(add(T(0), 1, "B", 1, 1000000))[:-3])
    with pytest.raises(ItchError, match="ends inside a message"):
        read_messages(path)
    with pytest.raises(ItchError, match="ends inside a message"):
        itch.count_messages(path)


def test_a_message_with_no_body_raises_cleanly(tmp_path):
    path = tmp_path / "empty.itch"
    path.write_bytes(framed(add(T(0), 1, "B", 1, 1000000)) + struct.pack(">H", 0))
    with pytest.raises(ItchError, match="no type byte"):
        read_messages(path)
    path.write_bytes(struct.pack(">H", 0) + framed(add(T(0), 1, "B", 1, 1000000)))
    with pytest.raises(ItchError, match="length 0; the specification"):
        read_messages(path)


def test_the_messages_of_other_types_have_their_lengths_checked_too(tmp_path):
    path = tmp_path / "other.itch"
    path.write_bytes(struct.pack(">H", 49) + filler("I", T(0))[:-1])
    with pytest.raises(ItchError, match="type 'I' and length 49"):
        read_messages(path)


def test_an_empty_file_and_a_day_with_only_system_events_give_empty_frames(tmp_path):
    empty = read(tmp_path)
    assert len(empty) == 0 and empty.attrs["symbol"] is None
    out = lobster_frames(empty)
    assert [len(f) for f in out] == [0, 0, 0] and list(out.book.columns) == book_columns(10)
    only = lobster_frames(read(tmp_path, system(T(0), "O"), system(T(1), "S")))
    assert len(only.messages) == 0 and len(only.book) == 0


# ---- The replay ----

def basic_day():
    return [
        add(T(1), 1, "B", 100, P(100.00)), add(T(2), 2, "B", 150, P(99.99)),
        add(T(3), 3, "S", 200, P(100.02)), add(T(4), 4, "S", 50, P(100.02)), add(T(5), 5, "S", 70, P(100.05)),
        execute(T(6), 3, 80, 7), cancel(T(7), 1, 30), delete(T(8), 2), delete(T(9), 4),
    ]


def test_orders_executions_cancels_and_deletions_become_lobster_rows_and_a_book(tmp_path):
    out = frames(tmp_path, *basic_day(), levels=2)
    assert list(out.messages.columns) == MESSAGE_COLUMNS + itch.EXTRA_COLUMNS
    assert out.messages[MESSAGE_COLUMNS[1:]].values.tolist() == [
        [1, 1, 100, P(100.00), 1], [1, 2, 150, P(99.99), 1], [1, 3, 200, P(100.02), -1], [1, 4, 50, P(100.02), -1],
        [1, 5, 70, P(100.05), -1],
        [4, 3, 80, P(100.02), -1],   # an execution: the resting order's side, at its price
        [2, 1, 30, P(100.00), 1],    # a cancel of part: the shares cancelled
        [3, 2, 150, P(99.99), 1],    # a deletion: the shares that were left
        [3, 4, 50, P(100.02), -1]]
    assert out.messages["time"].tolist() == [(T0 + 1000 * k) / 1e9 for k in range(1, 10)]
    assert out.messages["ns"].tolist() == [T(k) for k in range(1, 10)]
    assert out.messages["match"].tolist() == [0, 0, 0, 0, 0, 7, 0, 0, 0]
    assert not out.messages["replace"].any() and out.messages["group"].tolist() == [-1] * 5 + [0] + [-1] * 3
    nan = math.nan
    expected = [  # ask 1 price and size, bid 1, ask 2, bid 2, after each row
        [nan, 0, P(100.00), 100, nan, 0, nan, 0], [nan, 0, P(100.00), 100, nan, 0, P(99.99), 150],
        [P(100.02), 200, P(100.00), 100, nan, 0, P(99.99), 150],
        [P(100.02), 250, P(100.00), 100, nan, 0, P(99.99), 150],
        [P(100.02), 250, P(100.00), 100, P(100.05), 70, P(99.99), 150],
        [P(100.02), 170, P(100.00), 100, P(100.05), 70, P(99.99), 150],
        [P(100.02), 170, P(100.00), 70, P(100.05), 70, P(99.99), 150],
        [P(100.02), 170, P(100.00), 70, P(100.05), 70, nan, 0],
        [P(100.02), 120, P(100.00), 70, P(100.05), 70, nan, 0]]
    got = out.book[["ask_price_1", "ask_size_1", "bid_price_1", "bid_size_1", "ask_price_2", "ask_size_2",
                    "bid_price_2", "bid_size_2"]]
    pd.testing.assert_frame_equal(got, pd.DataFrame(expected, columns=got.columns).astype(
        {c: "int64" for c in got.columns if "size" in c}), check_dtype=False)
    assert list(out.book.columns) == book_columns(2)
    assert out.book.filter(like="size").dtypes.eq("int64").all()
    assert out.book.filter(like="price").dtypes.eq("float64").all()
    assert len(out.hidden) == 0 and list(out.hidden.columns) == itch.HIDDEN_COLUMNS


def test_the_book_lists_occupied_levels_best_first_with_empty_ones_as_nan(tmp_path):
    out = frames(tmp_path, add(T(1), 1, "B", 5, P(10.00)), add(T(2), 2, "B", 6, P(10.00)),
                 add(T(3), 3, "B", 7, P(9.97)), add(T(4), 4, "S", 8, P(10.03)), add(T(5), 5, "B", 9, P(9.99)),
                 levels=3)
    last = out.book.iloc[-1]
    assert last[["bid_price_1", "bid_size_1", "bid_price_2", "bid_size_2", "bid_price_3", "bid_size_3"]].tolist() == [
        P(10.00), 11, P(9.99), 9, P(9.97), 7]  # two orders at a price are one level
    assert last[["ask_price_1", "ask_size_1"]].tolist() == [P(10.03), 8]
    empty = last[["ask_price_2", "ask_size_2", "ask_price_3", "ask_size_3"]]
    assert empty.isna().tolist() == [True, False, True, False]
    assert last[["ask_size_2", "ask_size_3"]].tolist() == [0, 0]
    assert len(out.book) == len(out.messages) == 5 and out.book.iloc[0]["bid_size_2"] == 0


def test_a_level_is_an_occupied_price_not_a_tick(tmp_path):
    out = frames(tmp_path, add(T(1), 1, "S", 5, P(10.00)), add(T(2), 2, "S", 5, P(10.50)),
                 add(T(3), 3, "S", 5, P(10.90)), add(T(4), 4, "S", 5, P(10.10)), levels=2)
    assert out.book.iloc[-1][["ask_price_1", "ask_price_2"]].tolist() == [P(10.00), P(10.10)]
    assert out.book.iloc[-2][["ask_price_1", "ask_price_2"]].tolist() == [P(10.00), P(10.50)]  # before 10.10


def test_a_book_row_is_not_recomputed_when_a_deep_level_changes(tmp_path):
    """A change beyond the book frame's depth must leave the rows as they are, and one inside must change them."""
    out = frames(tmp_path, add(T(1), 1, "B", 5, P(10.00)), add(T(2), 2, "B", 5, P(9.99)), add(T(3), 3, "B", 5, P(9.98)),
                 add(T(4), 4, "B", 5, P(9.50)), add(T(5), 5, "B", 6, P(9.50)), delete(T(6), 4),
                 add(T(7), 6, "B", 1, P(9.985)), delete(T(8), 1), levels=2)
    sizes = out.book[["bid_price_1", "bid_size_1", "bid_price_2", "bid_size_2"]]
    assert sizes.iloc[2].tolist() == [P(10.00), 5, P(9.99), 5]
    assert (sizes.iloc[3:6].values == sizes.iloc[2].values).all()  # 9.50 and its changes are beyond 2 levels
    assert sizes.iloc[6].tolist() == [P(10.00), 5, P(9.99), 5]  # 9.985 sorts below 9.99: beyond
    assert sizes.iloc[7].tolist() == [P(9.99), 5, P(9.985), 1]  # the best bid goes: the third level comes in


def test_odd_lots_are_in_the_touch(tmp_path):
    out = frames(tmp_path, add(T(1), 1, "B", 100, P(10.00)), add(T(2), 2, "B", 1, P(10.01)),
                 add(T(3), 3, "S", 7, P(10.02)), levels=1)
    assert out.book.iloc[-1].tolist() == [P(10.02), 7, P(10.01), 1]


def test_a_replace_is_a_deletion_and_a_new_order_at_one_nanosecond(tmp_path):
    out = frames(tmp_path, add(T(1), 1, "B", 100, P(100.00)), add(T(2), 2, "B", 100, P(99.99)),
                 execute(T(3), 1, 40, 5), replace(T(4), 1, 9, 60, P(99.98)), execute(T(5), 9, 60, 6), levels=2)
    rows = out.messages
    assert rows.loc[3:4, ["type", "order_id", "size", "price", "direction", "ns", "replace"]].values.tolist() == [
        [3, 1, 60, P(100.00), 1, T(4), True],   # the old order, with the shares it had left
        [1, 9, 60, P(99.98), 1, T(4), True]]    # the new one, on the same side
    assert rows["replace"].tolist() == [False, False, False, True, True, False]
    assert out.book.iloc[3][["bid_price_1", "bid_size_1"]].tolist() == [P(99.99), 100]  # between the two rows
    assert out.book.iloc[4][["bid_price_1", "bid_price_2", "bid_size_2"]].tolist() == [P(99.99), P(99.98), 60]
    assert rows.iloc[5][["type", "order_id", "size", "price"]].tolist() == [4, 9, 60, P(99.98)]  # the new ref lives
    with pytest.raises(ItchError, match="not_live_execute"):
        frames(tmp_path, add(T(1), 1, "B", 100, P(100.00)), replace(T(2), 1, 9, 100, P(99.99)), execute(T(3), 1, 1, 5))


def test_a_printable_execution_with_price_is_an_execution_at_its_own_price(tmp_path):
    out = frames(tmp_path, add(T(1), 1, "B", 100, P(10.00)), add(T(2), 2, "S", 100, P(10.05)),
                 execute_with_price(T(3), 1, 30, 4, "Y", P(10.01)), levels=1)
    row = out.messages.iloc[2]
    assert row[["type", "order_id", "size", "price", "direction", "match", "group"]].tolist() == [
        4, 1, 30, P(10.01), 1, 4, 0]
    assert out.book.iloc[-1][["bid_price_1", "bid_size_1"]].tolist() == [P(10.00), 70]  # reduced at the order's price


def test_a_non_printable_execution_with_price_at_a_cross_is_a_cross_that_reduces_the_book(tmp_path):
    """Only at the nanosecond of a cross trade (Q): elsewhere a C-N is the other half of a printed trade (below), and
    one that is neither raises. (The first version of this test had a C-N in continuous trading with no printed half,
    which the review showed is not a cross.)"""
    out = frames(tmp_path, add(T(1), 1, "S", 100, P(10.05)), add(T(2), 2, "B", 100, P(10.00)),
                 execute_with_price(T(3), 1, 100, 4, "N", P(10.02)), cross(T(3), 5000, P(10.02), 9, "C"),
                 execute(T(3), 2, 10, 5), levels=1)
    row = out.messages.iloc[2]
    assert row[["type", "order_id", "size", "price", "direction", "match", "group"]].tolist() == [
        6, 1, 100, P(10.02), -1, 4, -1]
    assert out.messages.iloc[3]["type"] == 6 and out.messages.iloc[3]["order_id"] == 0
    assert out.messages.iloc[4]["group"] == 0  # the cross is no part of an aggressive order
    assert out.book.iloc[2][["ask_price_1", "ask_size_1"]].isna().tolist() == [True, False]
    assert out.book.iloc[2]["ask_size_1"] == 0 and out.book.iloc[-1]["bid_size_1"] == 90


def test_a_cross_trade_names_no_order_and_changes_nothing_displayed(tmp_path):
    out = frames(tmp_path, add(T(1), 1, "B", 100, P(10.00)), cross(T(2), 5000, P(10.01), 9, "O"), levels=1)
    row = out.messages.iloc[1]
    assert row[["type", "order_id", "size", "price", "direction", "match", "group"]].tolist() == [
        6, 0, 5000, P(10.01), 0, 9, -1]
    assert out.book.iloc[0].equals(out.book.iloc[1]) and len(out.hidden) == 0


def test_hidden_trades_are_a_table_of_their_own_and_their_side_byte_is_ignored(tmp_path):
    out = frames(tmp_path, add(T(1), 1, "B", 100, P(10.00)), trade(T(2), 40, P(10.005), 3, side="B"),
                 trade(T(3), 60, P(10.000), 4, side="B"), levels=1)
    assert out.messages["type"].tolist() == [1]  # no row for a P, and no type 5
    assert out.hidden[["ns", "size", "price", "match"]].values.tolist() == [
        [T(2), 40, 100_050, 3], [T(3), 60, 100_000, 4]]
    assert out.hidden["time"].tolist() == [T(2) / 1e9, T(3) / 1e9] and out.hidden["aggressor"].tolist() == [0, 0]
    assert out.hidden["group"].tolist() == [0, 1]  # different nanoseconds: two runs, both unsigned
    assert out.book.iloc[0]["bid_size_1"] == 100 and len(out.book) == 1


def run_book(extra):
    """Bids 100 at 10.00 and 100 at 9.99, asks 100 at 10.02 and 100 at 10.03, then ``extra`` messages."""
    return [add(T(1), 1, "B", 100, P(10.00)), add(T(2), 2, "B", 100, P(9.99)), add(T(3), 3, "S", 100, P(10.02)),
            add(T(4), 4, "S", 100, P(10.03))] + extra


def test_a_hidden_trade_inside_a_run_of_visible_executions_takes_the_runs_aggressor(tmp_path):
    # a buyer takes the asks at 10.02, trades hidden, then takes 10.03: all one nanosecond, matches 10, 11, 12
    out = frames(tmp_path, *run_book([
        execute(T(5), 3, 100, 10), trade(T(5), 30, P(10.025), 11, side="B"), execute(T(5), 4, 20, 12)]), levels=1)
    ex = out.messages[out.messages["type"] == 4]
    assert ex["group"].tolist() == [0, 0] and ex["direction"].tolist() == [-1, -1]
    assert out.hidden[["group", "aggressor"]].values.tolist() == [[0, 1]]  # buyer: minus the resting side
    # the same with the hidden trade first and last in the run
    out = frames(tmp_path, *run_book([
        trade(T(5), 30, P(10.025), 10, side="B"), execute(T(5), 3, 100, 11), trade(T(5), 5, P(10.025), 12)]), levels=1)
    assert out.hidden[["group", "aggressor"]].values.tolist() == [[0, 1], [0, 1]]
    # a seller hits the bids: the aggressor is minus +1
    out = frames(tmp_path, *run_book([execute(T(5), 1, 100, 10), trade(T(5), 5, P(10.00), 11)]), levels=1)
    assert out.hidden[["group", "aggressor"]].values.tolist() == [[0, -1]]


def test_a_run_of_hidden_trades_alone_is_unsigned_and_is_counted_as_one_group(tmp_path):
    out = frames(tmp_path, *run_book([trade(T(5), 30, P(10.01), 10), trade(T(5), 20, P(10.01), 11),
                                      execute(T(6), 3, 10, 12)]), levels=1)
    assert out.hidden[["group", "aggressor"]].values.tolist() == [[0, 0], [0, 0]]
    assert out.messages.loc[out.messages["type"] == 4, "group"].tolist() == [1]  # a later nanosecond: another order


def test_executions_are_one_aggressive_order_only_at_one_nanosecond_with_consecutive_matches(tmp_path):
    messages = [
        execute(T(5), 3, 10, 10), execute(T(5), 3, 10, 11),      # same ns, consecutive: one order (group 0)
        execute(T(5), 3, 10, 13),                                  # a gap in the match numbers: another (1)
        execute(T(6), 3, 10, 14),                                  # the next match but a later ns: another (2)
        execute(T(7), 3, 10, 15), execute(T(7), 1, 10, 16),        # the resting side changes: another (3, 4)
        execute(T(8), 3, 10, 17), add(T(8), 7, "B", 1, P(9.50)), execute(T(8), 3, 10, 18),  # an add between: (5, 6)
        execute(T(9), 3, 10, 19), delete(T(9), 2), execute(T(9), 3, 10, 20),  # a deletion between: one (7)
    ]
    out = frames(tmp_path, *run_book(messages), levels=1)
    assert out.messages.loc[out.messages["type"] == 4, "group"].tolist() == [0, 0, 1, 2, 3, 4, 5, 6, 7, 7]


def visible_groups(out):
    return out.messages.loc[out.messages["type"] == 4, "group"].tolist()


def test_merge_within_joins_successive_runs_of_one_side_that_follow_on(tmp_path):
    messages = run_book([
        execute(T(5), 3, 10, 10), execute(T(5), 3, 10, 11),
        execute(T(55), 3, 10, 12),                       # 50 microseconds later, the next match, the same side
        execute(T(500), 3, 10, 13),                      # 445 microseconds later
        execute(T(500) + 1, 1, 10, 14),                  # the other side, a nanosecond later
        trade(T(600), 5, P(10.01), 15),                  # a hidden run alone is never merged
        trade(T(600) + 1, 5, P(10.01), 16),
        execute(T(900), 3, 10, 17), add(T(901), 7, "B", 1, P(9.50)), execute(T(902), 3, 10, 18),  # an add between
    ])
    plain = frames(tmp_path, *messages, levels=1)
    merged = frames(tmp_path, *messages, levels=1, merge_within=100e-6)
    assert visible_groups(plain) == [0, 0, 1, 2, 3, 6, 7] and plain.hidden["group"].tolist() == [4, 5]
    assert visible_groups(merged) == [0, 0, 0, 1, 2, 5, 6] and merged.hidden["group"].tolist() == [3, 4]
    assert merged.messages.loc[merged.messages["type"] == 4, "direction"].tolist()[:4] == [-1, -1, -1, -1]
    assert merged.hidden["aggressor"].tolist() == [0, 0]
    # at exactly the limit they are one, a nanosecond more and they are not
    two = run_book([execute(T(5), 3, 10, 10), execute(T(5) + 100_000, 3, 10, 11)])
    assert visible_groups(frames(tmp_path, *two, levels=1, merge_within=100e-6)) == [0, 0]
    two = run_book([execute(T(5), 3, 10, 10), execute(T(5) + 100_001, 3, 10, 11)])
    assert visible_groups(frames(tmp_path, *two, levels=1, merge_within=100e-6)) == [0, 1]


def test_a_hidden_trade_in_a_merged_run_is_signed_by_it(tmp_path):
    messages = run_book([execute(T(5), 3, 10, 10), trade(T(5), 5, P(10.01), 11), execute(T(40), 3, 10, 12)])
    merged = frames(tmp_path, *messages, levels=1, merge_within=100e-6)
    assert merged.messages.loc[merged.messages["type"] == 4, "group"].tolist() == [0, 0]
    assert merged.hidden[["group", "aggressor"]].values.tolist() == [[0, 1]]


def test_the_window_holds_the_rows_inside_it_and_the_book_is_replayed_from_the_start_of_the_day(tmp_path):
    before = [add(T0 - 5_000_000, 1, "B", 100, P(10.00)), add(T0 - 4_000_000, 2, "S", 100, P(10.02)),
              trade(T0 - 3_000_000, 5, P(10.01), 1)]
    inside = [add(T(1), 3, "B", 50, P(10.00)), execute(T0 + 2_000_000_000, 2, 40, 2)]
    after = [add(T0 + 23_400_000_000_000, 4, "S", 5, P(10.02))]  # at 16:00:00 exactly: outside [start, end)
    out = frames(tmp_path, *before, *inside, *after, levels=1)
    assert out.messages["order_id"].tolist() == [3, 2] and len(out.book) == 2
    assert out.book.iloc[0].tolist() == [P(10.02), 100, P(10.00), 150]  # orders from before 09:30 are in the book
    assert out.book.iloc[1].tolist() == [P(10.02), 60, P(10.00), 150]
    assert len(out.hidden) == 0  # the hidden trade was before the window
    narrow = frames(tmp_path, *before, *inside, *after, start=34200.001, end=34201.0, levels=1)
    assert len(narrow.messages) == 0 and len(narrow.book) == 0
    one = frames(tmp_path, *before, *inside, *after, start=34200.0 + 1e-6, end=34200.5, levels=1)
    assert one.messages["order_id"].tolist() == [3] and one.messages["time"].tolist() == [T(1) / 1e9]
    assert one.book.iloc[0].tolist() == [P(10.02), 100, P(10.00), 150]
    assert one.messages.attrs["start"] == 34200.0 + 1e-6 and one.hidden.attrs["levels"] == 1


def test_group_numbers_count_from_zero_in_the_window(tmp_path):
    out = frames(tmp_path, *run_book([execute(T(5), 3, 10, 10), execute(T(6), 3, 10, 11)]),
                 start=34200.0 + 5.5e-6, levels=1)
    assert out.messages["group"].tolist() == [0]


# ---- What raises, and what is counted when it does not ----

def with_book(*extra):
    return [add(T(1), 1, "B", 100, P(100.00)), add(T(2), 2, "S", 100, P(100.05))] + list(extra)


@pytest.mark.parametrize("name, messages, match", [
    ("not_live_execute", with_book(execute(T(3), 99, 1, 1)), "reference 99 is not live"),
    ("not_live_execute", with_book(execute(T(3), 1, 100, 1), execute(T(4), 1, 1, 2)), "reference 1 is not live"),
    ("not_live_execute", with_book(execute_with_price(T(3), 99, 1, 1, "Y", P(100.01))), "reference 99"),
    ("not_live_cancel", with_book(cancel(T(3), 99, 1)), "reference 99 is not live"),
    ("not_live_delete", with_book(delete(T(3), 99)), "reference 99 is not live"),
    ("not_live_delete", with_book(delete(T(3), 1), delete(T(4), 1)), "reference 1 is not live"),
    ("not_live_replace", with_book(replace(T(3), 99, 5, 1, P(100.00))), "original 99"),
    ("over_execution", with_book(execute(T(3), 1, 101, 1)), "101 shares executed of 100"),
    ("over_cancel", with_book(cancel(T(3), 1, 101)), "101 shares cancelled of 100"),
    ("duplicate_reference", with_book(add(T(3), 1, "B", 1, P(99.00))), "reference 1 is already live"),
    ("duplicate_reference", with_book(replace(T(3), 1, 2, 1, P(99.00))), "reference 2 is already live"),
    ("empty_add", with_book(add(T(3), 5, "B", 0, P(99.00))), "0 shares"),
    ("crossed_or_locked", with_book(add(T(3), 5, "B", 1, P(100.06))), "best bid"),
    ("crossed_or_locked", with_book(add(T(3), 5, "B", 1, P(100.05))), "best bid"),
    ("crossed_or_locked", with_book(add(T(3), 5, "S", 1, P(99.99))), "best bid"),
    ("later_trading_action", [action(T(0))] + with_book(action(T(3), "H")), "trading action"),
    ("later_trading_action", [action(T(0))] + with_book(action(T(3), "T")), "trading action"),
    ("later_trading_action", [action(T(0), "H")] + with_book(), "not 'T'"),
    ("touch_under_a_dollar", [add(T(1), 1, "B", 100, P(0.50)), add(T(2), 2, "S", 100, P(0.51))], "cent tick"),
    ("live_orders_after_end_of_system_hours", with_book(system(T(3), "E")), "2 orders live"),
])
def test_the_replay_raises_on_what_cannot_happen(tmp_path, name, messages, match):
    day = read(tmp_path, *messages)
    with pytest.raises(ItchError, match=f"{name}.*{match}"):
        lobster_frames(day)


def test_what_raises_is_counted_and_skipped_without_strict(tmp_path):
    day = read(tmp_path, action(T(0)), *with_book(
        execute(T(3), 99, 1, 1), cancel(T(3), 99, 1), delete(T(3), 99), replace(T(3), 99, 5, 1, P(100.00)),
        execute(T(4), 1, 101, 2), cancel(T(5), 2, 101),
        add(T(6), 7, "B", 0, P(99.0)), add(T(6), 3, "B", 5, P(99.0)), add(T(6), 3, "B", 1, P(98.0)),
        add(T(7), 8, "S", 1, P(98.5)), action(T(8), "H"), system(T(9), "E")))
    report = {}
    out = lobster_frames(day, strict=False, report=report, levels=1)
    assert report == {
        "not_live_execute": 1, "not_live_cancel": 1, "not_live_delete": 1, "not_live_replace": 1, "over_execution": 1,
        "over_cancel": 1, "duplicate_reference": 1, "empty_add": 1, "crossed_or_locked": 1, "touch_under_a_dollar": 0,
        "later_trading_action": 1, "live_orders_after_end_of_system_hours": 2, "adds_under_a_dollar": 0,
        "adds_off_the_cent_grid": 0, "cancels_to_zero": 1, "live_orders_at_end": 2, "rows": 6, "groups": 1,
        "empty_execution": 0, "empty_cancel": 0, "unpaired_non_printable_execution": 0, "touch_off_the_cent_grid": 0,
        "p_side_not_b": 0, "p_with_reference": 0, "c_price_over_a_cent": 0}
    assert set(itch.PROBLEMS) <= set(report)
    # skipped: the messages naming no live order, the empty add and the duplicate. Cut to the shares left: the
    # execution (100 of the 101) and the cancel (100 of the 101).
    assert out.messages[["type", "order_id", "size"]].values.tolist() == [
        [1, 1, 100], [1, 2, 100], [4, 1, 100], [2, 2, 100], [1, 3, 5], [1, 8, 1]]


def test_a_crossed_book_before_the_window_does_not_raise_but_one_inside_it_does(tmp_path):
    crossed = [add(T0 - 2_000_000, 1, "B", 1, P(10.05)), add(T0 - 1_000_000, 2, "S", 1, P(10.00))]
    # the first message inside the window uncrosses the book: nothing is counted for the crossed state before it
    assert len(lobster_frames(read(tmp_path, *crossed, delete(T(1), 1)), levels=1).messages) == 1
    with pytest.raises(ItchError, match="crossed_or_locked"):  # one that leaves it crossed is
        lobster_frames(read(tmp_path, *crossed, add(T(1), 3, "B", 1, P(10.07))), levels=1)
    with pytest.raises(ItchError, match="crossed_or_locked"):  # and so is one that crosses it
        lobster_frames(read(tmp_path, add(T0 - 1_000_000, 2, "S", 1, P(10.00)), add(T(1), 3, "B", 1, P(10.00))))


def test_an_order_cancelled_down_to_nothing_leaves_the_book(tmp_path):
    report = {}
    out = frames(tmp_path, *with_book(cancel(T(3), 1, 100)), strict=False, report=report, levels=1)
    assert report["cancels_to_zero"] == 1 and report["live_orders_at_end"] == 1
    assert out.book.iloc[-1][["bid_price_1", "bid_size_1"]].isna().tolist() == [True, False]
    assert out.messages.iloc[-1][["type", "size", "price"]].tolist() == [2, 100, P(100.00)]
    with pytest.raises(ItchError, match="not_live_delete"):
        frames(tmp_path, *with_book(cancel(T(3), 1, 100), delete(T(4), 1)))


def test_prices_under_a_dollar_far_from_the_touch_are_counted_not_refused(tmp_path):
    """NVDA has 47 adds under a dollar, stub bids at a cent to 84 cents: the touch is what the cent tick concerns."""
    report = {}
    out = frames(tmp_path, *with_book(add(T(3), 5, "B", 1, P(0.84)), add(T(4), 6, "B", 100, 1),
                                      add(T(5), 7, "B", 1, P(99.995))), report=report, levels=2)
    assert report["adds_under_a_dollar"] == 2 and report["adds_off_the_cent_grid"] == 2  # P(0.84) is on the grid
    assert report["touch_under_a_dollar"] == 0 and len(out.messages) == 5
    assert out.book.iloc[-1][["bid_price_1", "bid_price_2"]].tolist() == [P(100.00), P(99.995)]


def test_a_start_of_day_trading_action_of_trading_is_accepted(tmp_path):
    out = frames(tmp_path, system(T0 - 10, "O"), action(T0 - 9, "T"), *with_book(), system(T(5), "C"), levels=1)
    assert len(out.messages) == 2


def test_the_book_must_be_empty_after_the_end_of_system_hours_only_if_the_day_got_there(tmp_path):
    lobster_frames(read(tmp_path, *with_book()))  # no end-of-hours event: a cut-off file is no failure
    done = with_book(delete(T(3), 1), delete(T(4), 2), system(T(5), "E"))
    report = {}
    lobster_frames(read(tmp_path, *done), report=report)
    assert report["live_orders_at_end"] == 0 and report["live_orders_after_end_of_system_hours"] == 0


def test_importing_lob_does_not_import_the_itch_reader():
    import subprocess
    import sys
    code = "import sys, lob; sys.exit(int('lob.itch' in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT).returncode == 0


# ---- Files ----

def test_frames_round_trip_through_parquet(tmp_path):
    pytest.importorskip("pyarrow")
    out = frames(tmp_path, *run_book([execute(T(5), 3, 10, 10), trade(T(5), 5, P(10.01), 11), execute(T(6), 3, 10, 12),
                                      replace(T(7), 2, 9, 50, P(9.98))]), levels=3, merge_within=100e-6)
    paths = itch.save_frames(out, tmp_path / "frames", "test")
    assert sorted(p.name for p in paths.values()) == [
        "TEST_book.parquet", "TEST_frames.json", "TEST_hidden.parquet", "TEST_messages.parquet"]
    again = itch.load_frames(tmp_path / "frames", "test")
    assert isinstance(again, itch.Frames)
    for was, now in zip(out, again, strict=True):
        pd.testing.assert_frame_equal(was, now)
        assert now.attrs == {"start": 34200.0, "end": 57600.0, "levels": 3, "merge_within": 100e-6, "symbol": "TEST"}
    with pytest.raises(ValueError, match="give the symbol"):
        itch.save_frames(lobster_frames(read(tmp_path, system(T(0), "O"))), tmp_path / "x")
    messages, book, hidden = again  # three frames, as lobster_frames returns
    assert len(messages) == len(book) == 8 and len(hidden) == 1


def day_of_three_stocks():
    """A small day for the pass over a file: three stocks, interleaved, and the messages that belong to none."""
    return [
        system(T0 - 10_000, "O"), directory(T0 - 9_000, "AAA", 1), directory(T0 - 8_000, "BBB", 2),
        directory(T0 - 7_000, "CCC", 3), action(T0 - 6_000, "T", "AAA", 1), action(T0 - 5_000, "T", "BBB", 2),
        system(T0 - 4_000, "Q"), add(T(1), 1, "B", 100, P(10.00), "AAA", 1), add(T(2), 11, "S", 50, P(20.00), "BBB", 2),
        add(T(3), 21, "B", 5, P(30.00), "CCC", 3), add(T(4), 2, "S", 100, P(10.02), "AAA", 1),
        execute(T(5), 1, 10, 1, 1), execute(T(5), 11, 5, 2, 2), filler("I", T(6), 2), delete(T(7), 21, 3),
        filler("V", T(8), 0), cross(T(9), 10, P(10.01), 3, "C", "AAA", 1), system(T(10), "E"), system(T(11), "C"),
    ]


def write_day(tmp_path, gz=True):
    path = tmp_path / ("day.gz" if gz else "day.itch")
    data = framed(*day_of_three_stocks())
    path.write_bytes(gzip.compress(data) if gz else data)
    return path, data


def test_scan_counts_a_gzipped_day_and_reads_its_directory_and_system_events(tmp_path):
    path, data = write_day(tmp_path)
    result = itch.scan(path)
    assert result["directory"] == {"AAA": 1, "BBB": 2, "CCC": 3}
    assert result["counts"] == {"S": 4, "R": 3, "H": 2, "A": 4, "E": 2, "I": 1, "D": 1, "V": 1, "Q": 1}
    assert result["system_events"] == [("O", T0 - 10_000), ("Q", T0 - 4_000), ("E", T(10)), ("C", T(11))]
    assert result["messages"] == 19 and result["bytes"] == len(data)
    assert itch.scan(write_day(tmp_path, gz=False)[0]) == result  # a plain file reads the same
    assert itch.count_messages(path) == result["counts"]


@pytest.mark.parametrize("chunk", [5, 37, 100])
def test_messages_that_straddle_a_block_boundary_are_read_whole(tmp_path, monkeypatch, chunk):
    path, _ = write_day(tmp_path)
    expected = itch.scan(path)
    monkeypatch.setattr(itch, "_CHUNK", chunk)
    small = itch.scan(path)
    assert small.pop("blocks") > expected.pop("blocks") == 1 and small == expected
    stats = itch.extract(path, ["BBB"], tmp_path / f"out{chunk}")
    monkeypatch.setattr(itch, "_CHUNK", 1 << 20)
    assert stats == itch.extract(path, ["BBB"], tmp_path / "again")


def test_extract_writes_each_stocks_messages_and_those_of_no_stock_in_file_order(tmp_path):
    path, data = write_day(tmp_path)
    stats = itch.extract(path, ["aaa", "CCC"], tmp_path / "out")
    everything = [data[i:j] for i, j in _spans(data)]
    for symbol, locate in (("AAA", 1), ("CCC", 3)):
        expected = b"".join(m for m in everything if int.from_bytes(m[3:5], "big") in (0, locate))
        written = (tmp_path / "out" / f"{symbol}.itch").read_bytes()
        assert written == expected
        assert stats[symbol] == {"locate": locate, "messages": len(_spans(expected)), "bytes": len(expected),
                                 "sha256": hashlib.sha256(expected).hexdigest()}
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["AAA.itch", "CCC.itch"]
    one = read_messages(tmp_path / "out" / "AAA.itch")
    assert one["type"].tolist() == ["S", "H", "S", "A", "A", "E", "Q", "S", "S"] and one.attrs["symbol"] == "AAA"
    assert read_messages(tmp_path / "out" / "CCC.itch")["type"].tolist() == ["S", "S", "A", "D", "S", "S"]


def _spans(data):
    spans, o = [], 0
    while o < len(data):
        n = 2 + int.from_bytes(data[o:o + 2], "big")
        spans.append((o, o + n))
        o += n
    return spans


def test_a_symbol_named_twice_is_extracted_once(tmp_path):
    path, _ = write_day(tmp_path)
    itch.extract(path, ["AAA", "aaa"], tmp_path / "out")
    assert (tmp_path / "out" / "AAA.itch").read_bytes() == (itch.extract(path, ["AAA"], tmp_path / "one") and
                                                            (tmp_path / "one" / "AAA.itch").read_bytes())


def test_extract_refuses_to_overwrite_and_leaves_nothing_half_written(tmp_path):
    path, _ = write_day(tmp_path)
    itch.extract(path, ["AAA"], tmp_path / "out")
    with pytest.raises(FileExistsError):
        itch.extract(path, ["AAA"], tmp_path / "out")
    assert itch.extract(path, ["AAA"], tmp_path / "out", overwrite=True)["AAA"]["messages"] == 11
    with pytest.raises(ItchError, match="not in the day's stock directory: \\['ZZZ'\\]"):
        itch.extract(path, ["BBB", "ZZZ"], tmp_path / "out2")
    assert list((tmp_path / "out2").iterdir()) == []
    bad = tmp_path / "bad.itch"
    bad.write_bytes(framed(directory(T(0), "AAA", 1), add(T(1), 1, "B", 1, 1000000, "AAA", 1))[:-2])
    with pytest.raises(ItchError, match="ends inside a message"):
        itch.extract(bad, ["AAA"], tmp_path / "out3")
    assert list((tmp_path / "out3").iterdir()) == []


def test_a_directory_that_names_a_stock_or_a_locate_twice_raises(tmp_path):
    path = tmp_path / "dup.itch"
    path.write_bytes(framed(directory(T(0), "AAA", 1), directory(T(1), "AAA", 2)))
    with pytest.raises(ItchError, match="twice"):
        itch.scan(path)
    with pytest.raises(ItchError, match="twice"):
        itch.extract(path, ["AAA"], tmp_path / "out")
    path.write_bytes(framed(directory(T(0), "AAA", 1), directory(T(1), "BBB", 1)))
    with pytest.raises(ItchError, match="twice"):
        itch.scan(path)


def test_a_corrupt_gzip_raises(tmp_path):
    path, _ = write_day(tmp_path)
    blob = bytearray(path.read_bytes())
    blob[-6] ^= 0xFF  # the checksum
    path.write_bytes(bytes(blob))
    with pytest.raises(Exception, match="CRC|crc|Not a gzipped"):
        itch.scan(path)


# ---- The simulator's events written as ITCH and parsed back ----

TRUE = dict(lam=1.0, cancel_rate=0.02, imbalance_offset=0.5, hurst=0.7, nu=2.0, loc=0.5, scale=2.5)


def as_itch(messages: pd.DataFrame) -> bytes:
    """A simulated LOBSTER message frame as ITCH: adds, deletions and executions are all the simulator writes."""
    out, match = [], 0
    for time, kind, order_id, size, price, direction in messages[MESSAGE_COLUMNS].itertuples(index=False):
        ns = round(time * 1e9)
        if kind == 1:
            out.append(add(ns, int(order_id), "B" if direction == 1 else "S", int(size), int(price)))
        elif kind == 3:
            out.append(delete(ns, int(order_id)))
        elif kind == 4:
            match += 1
            out.append(execute(ns, int(order_id), int(size), match))
        else:
            raise AssertionError(kind)
    return framed(*out)


@pytest.mark.parametrize("seed", [1, 2])
def test_the_simulators_events_written_as_itch_parse_back_to_its_frames(tmp_path, seed):
    """Holds under the simulator's conditions: it writes only adds, deletions and executions, one price at a
    time, every aggressive order's executions at one time, and the book after a whole event (so the book is
    compared at the last row of each event, where the two files agree; ITCH has a time to the nanosecond)."""
    reset_order_id_counter()
    sim_messages, sim_book = simulated_lobster(lambda: MikeFarmer2008Model(initial_mid=10000, **TRUE), 2_500, seed=seed,
                                               levels=4)
    path = tmp_path / "SIM.itch"
    path.write_bytes(as_itch(sim_messages))
    out = lobster_frames(read_messages(path), levels=4)
    assert len(out.messages) == len(sim_messages) and len(out.book) == len(sim_book) and len(out.hidden) == 0
    np.testing.assert_allclose(out.messages["time"], sim_messages["time"], rtol=0, atol=1e-9)
    pd.testing.assert_frame_equal(out.messages[MESSAGE_COLUMNS[1:]], sim_messages[MESSAGE_COLUMNS[1:]],
                                  check_dtype=False)
    last_of_event = np.append(sim_messages["time"].to_numpy()[1:] != sim_messages["time"].to_numpy()[:-1], True)
    pd.testing.assert_frame_equal(out.book[last_of_event].reset_index(drop=True),
                                  sim_book[last_of_event].reset_index(drop=True), check_dtype=False)
    # the rows of an event share a nanosecond, and one group is one aggressive order: the executions of one event
    event = np.cumsum(np.append(True, sim_messages["time"].to_numpy()[1:] != sim_messages["time"].to_numpy()[:-1]))
    ex = out.messages["type"] == 4
    assert out.messages.loc[ex, "group"].groupby(event[ex]).nunique().eq(1).all()
    assert out.messages.loc[ex, "group"].nunique() == pd.Series(event[ex]).nunique()
    # what the calibration reads comes out the same
    pd.testing.assert_frame_equal(lobster_orders(out.messages, out.book), lobster_orders(sim_messages, sim_book),
                                  check_dtype=False, rtol=1e-9)


def test_calibration_gives_the_same_estimates_from_the_itch_frames_as_from_the_simulators(tmp_path):
    reset_order_id_counter()
    sim_messages, sim_book = simulated_lobster(lambda: MikeFarmer2008Model(initial_mid=10000, **TRUE), 6_000, seed=3)
    path = tmp_path / "SIM.itch"
    path.write_bytes(as_itch(sim_messages))
    out = lobster_frames(read_messages(path))
    mine, theirs = calibrate_lobster(out.messages, out.book), calibrate_lobster(sim_messages, sim_book)
    assert mine.keys() == theirs.keys()
    for key in theirs:
        if theirs[key] is None:
            assert mine[key] is None
        else:
            assert mine[key] == pytest.approx(theirs[key], rel=1e-6, abs=1e-9), key


def test_the_lobster_readers_take_the_frames_of_a_day_with_replaces_crosses_and_hidden_trades(tmp_path):
    out = frames(tmp_path, *run_book([
        execute(T(5), 3, 100, 10), add(T(5), 5, "B", 20, P(10.02)), trade(T(6), 5, P(10.01), 11),
        replace(T(7), 2, 9, 100, P(9.98)), execute_with_price(T(8), 1, 10, 12, "N", P(10.00)),
        cross(T(8), 10, P(10.01), 13), cancel(T(10), 9, 1), delete(T(11), 5), execute(T(12), 4, 100, 14)]))
    orders = lobster_orders(out.messages, out.book)
    # four resting orders, a buy that took 100 and left 20 (one order: the remainder is at the fill's price),
    # the new order of the replace, and a buy that took 100 and left nothing. A cross, a hidden trade, a cancel
    # and a deletion are no orders.
    assert orders["side"].tolist() == ["bid", "bid", "ask", "ask", "bid", "bid", "bid"]
    assert orders["size"].tolist() == [100, 100, 100, 100, 120, 100, 100]
    assert orders["executed"].tolist() == [0, 0, 0, 0, 100, 0, 100]
    assert orders["price"].isna().tolist() == [False] * 4 + [False, False, True] and orders.loc[4, "price"] == 1002


# ---- Against the real day (skipped without the downloaded files) ----

NVDA = DERIVED / "raw" / "NVDA.itch"
REFERENCE = DERIVED / "reference" / "nvda_rows.npz"
needs_day = pytest.mark.skipif(not (NVDA.exists() and REFERENCE.exists()), reason=NO_DAY)


@pytest.fixture(scope="module")
def nvda():
    report = {}
    messages = read_messages(NVDA)
    return messages, lobster_frames(messages, 0.0, 86400.0, levels=1, report=report), report


@needs_day
def test_nvdas_touch_after_every_message_equals_the_independent_replay(nvda):
    """The review's replay (written without this module) has the touch after each of NVDA's 969,738 order messages."""
    reference = np.load(REFERENCE)
    book_message = np.isin(reference["typ"], [ord(c) for c in "AFECXDUQ"])
    _, out, _ = nvda
    # a replace is one row there, two here: take the row after the add
    row = np.flatnonzero(~(out.messages["replace"].to_numpy() & (out.messages["type"].to_numpy() == 3)))
    assert len(row) == book_message.sum() == 969_738
    book = out.book.iloc[row]
    mine = np.column_stack([np.nan_to_num(book["bid_price_1"].to_numpy()), book["bid_size_1"].to_numpy(),
                            np.nan_to_num(book["ask_price_1"].to_numpy()),
                            book["ask_size_1"].to_numpy()]).astype(np.int64)
    theirs = np.column_stack([reference[k][book_message] for k in ("bid", "bidq", "ask", "askq")])
    assert (mine == theirs).all()
    assert (out.messages["ns"].to_numpy()[row] == reference["ts"][book_message]).all()


@needs_day
def test_nvdas_replay_meets_the_invariants_and_its_counts_by_type_are_the_independent_ones(nvda):
    messages, out, report = nvda
    assert all(report[name] == 0 for name in itch.PROBLEMS) and report["live_orders_at_end"] == 0
    reference = np.load(REFERENCE)
    own = messages["type"].value_counts().to_dict()
    for letter in "AFECXDUPQH":
        assert own[letter] == int((reference["typ"] == ord(letter)).sum()), letter
    assert out.messages["replace"].sum() == 2 * own["U"]
    assert len(out.hidden) == own["P"]


@needs_day
def test_nvdas_runs_of_executions_have_the_independent_replays_counts(nvda):
    """The review counted NVDA's runs of executions from 09:31 to 15:50 with its own code: runs that follow each other
    directly, at one nanosecond, with consecutive match numbers, are 19,719 (15,447 of displayed executions only,
    3,137 of hidden trades only, 1,135 of both), from 22,057 displayed executions and 4,881 hidden trades. Nine of them
    were cut by a deletion (or a cancel of part) between two executions: counted as one run, as the sheet's rule
    says, they are 19,710 (15,438; 3,134; 1,138), by a recount of the fix from the audit's own decoder
    (``build/fix/recount.py``). With ``merge_within`` of 100 microseconds, 17,830: 1,880 joins, as the review's rule
    (the runs follow each other directly) had before."""
    messages, _, _ = nvda
    out = lobster_frames(messages, 34260.0, 57000.0, levels=1)
    shown = out.messages[out.messages["type"] == 4]
    assert len(shown) == 22_057 and len(out.hidden) == 4_881
    has_shown = set(shown["group"])
    has_hidden = set(out.hidden["group"])
    assert (len(has_shown - has_hidden), len(has_hidden - has_shown), len(has_shown & has_hidden)) == (
        15_438, 3_134, 1_138)
    merged = lobster_frames(messages, 34260.0, 57000.0, levels=1, merge_within=100e-6)
    joined = len(has_shown | has_hidden) - len(set(merged.messages.loc[merged.messages["type"] == 4, "group"])
                                                | set(merged.hidden["group"]))
    assert joined == 1_880  # the review's 1,925 adjacent pairs include some more than 100 microseconds apart


@needs_day
def test_nvdas_window_frames_feed_the_lobster_readers(nvda):
    messages, _, _ = nvda
    out = lobster_frames(messages.iloc[:60_000], 0.0, 86400.0, levels=10, strict=False)
    orders = lobster_orders(out.messages, out.book)
    assert len(orders) > 0 and orders["side"].isin(["bid", "ask"]).all()
    assert set(out.messages["type"]) <= {1, 2, 3, 4, 6}


# ---- Added by the audit: what the mechanical sweep showed no test held ----

def test_the_defaults_of_lobster_frames_and_a_depth_of_less_than_one_level_is_refused(tmp_path):
    out = frames(tmp_path, *basic_day())
    for frame in out:
        assert frame.attrs == {"start": 34200.0, "end": 57600.0, "levels": 10, "merge_within": None, "symbol": "TEST"}
    assert list(out.book.columns) == book_columns(10)
    with pytest.raises(ValueError, match="levels must be at least 1"):
        frames(tmp_path, *basic_day(), levels=0)


def test_frames_with_no_code_column_replay_as_if_every_code_were_blank(tmp_path):
    day = read(tmp_path, action(T0 - 9, "T"), *basic_day())
    assert "code" in day
    for was, now in zip(lobster_frames(day, levels=2), lobster_frames(day.drop(columns="code"), levels=2), strict=True):
        pd.testing.assert_frame_equal(was, now)


def test_a_file_that_holds_only_a_cross_has_one_row_and_an_empty_book(tmp_path):
    out = frames(tmp_path, cross(T(1), 100, P(10.00), 5, "O"), levels=2)
    assert len(out.messages) == len(out.book) == 1 and out.messages["type"].tolist() == [6]
    assert out.book.iloc[0].filter(like="price").isna().all() and (out.book.iloc[0].filter(like="size") == 0).all()


def test_the_touch_is_refused_under_a_dollar_on_either_side_and_not_at_exactly_a_dollar(tmp_path):
    ok = frames(tmp_path, add(T(1), 1, "B", 1, P(1.00)), add(T(2), 2, "S", 1, P(1.01)))
    assert len(ok.messages) == 2
    assert len(frames(tmp_path, add(T(1), 1, "B", 1, P(1.00))).messages) == 1       # a bid alone at a dollar
    assert len(frames(tmp_path, add(T(1), 1, "S", 1, P(1.00))).messages) == 1       # an ask alone at a dollar
    for messages in ([add(T(1), 1, "B", 1, P(0.99))],  # a bid alone
                     [add(T(1), 1, "S", 1, P(0.99))],  # an ask alone
                     [add(T(1), 1, "B", 1, P(0.50)), add(T(2), 2, "S", 1, P(100.00))]):  # a bid under, an ask over
        with pytest.raises(ItchError, match="touch_under_a_dollar"):
            frames(tmp_path, *messages)


def test_a_touch_under_a_dollar_is_counted_and_skipped_messages_do_not_check_the_touch_again(tmp_path):
    report = {}
    frames(tmp_path, add(T(1), 1, "B", 1, P(0.50)), add(T(2), 2, "S", 1, P(100.00)), strict=False, report=report)
    assert report["touch_under_a_dollar"] == 2 and report["adds_under_a_dollar"] == 1  # a row each: the bid is under
    # a bid at exactly a dollar is not under one (not counted as an add under a dollar either)
    report = {}
    frames(tmp_path, add(T(1), 1, "B", 1, P(1.00)), strict=False, report=report)
    assert report["touch_under_a_dollar"] == 0 and report["adds_under_a_dollar"] == 0
    # a duplicate reference and an add of no shares are skipped: the touch is not looked at for them
    report = {}
    out = frames(tmp_path, add(T(1), 1, "B", 1, P(10.00)), add(T(2), 2, "S", 1, P(9.00)), add(T(3), 1, "B", 1, P(8.00)),
                 add(T(4), 3, "B", 0, P(8.00)), strict=False, report=report, levels=1)
    assert report["crossed_or_locked"] == 1 and report["duplicate_reference"] == 1 and report["empty_add"] == 1
    assert len(out.messages) == 2
    with pytest.raises(ItchError, match="^touch_under_a_dollar at "):
        frames(tmp_path, add(T(1), 1, "B", 1, P(0.50)))


def test_the_touch_is_looked_at_after_every_kind_of_message_that_takes_orders_out_of_the_book(tmp_path):
    # the best bid goes and a stub bid under a dollar is left as the touch: a delete, an execution, a cancel of
    # all its shares, an execution with price (printable or not) each leave it so
    stub = [add(T(1), 1, "B", 100, P(100.00)), add(T(2), 2, "B", 100, P(0.50)), add(T(3), 3, "S", 100, P(100.05))]
    assert len(frames(tmp_path, *stub).messages) == 3
    for last in ((delete(T(4), 1),), (execute(T(4), 1, 100, 9),), (cancel(T(4), 1, 100),),
                 (execute_with_price(T(4), 1, 100, 9, "Y", P(100.00)),),
                 (execute_with_price(T(4), 1, 100, 9, "N", P(100.00)), execute(T(4), 3, 1, 9))):  # a C-N and its E
        with pytest.raises(ItchError, match="^touch_under_a_dollar at "):
            frames(tmp_path, *stub, *last)


def test_a_cancel_of_part_of_a_sell_order_is_on_the_sell_side(tmp_path):
    out = frames(tmp_path, add(T(1), 1, "S", 100, P(10.02)), add(T(2), 2, "B", 100, P(10.00)),
                 cancel(T(3), 1, 30), cancel(T(4), 2, 20), levels=1)
    assert out.messages.loc[2:, ["type", "order_id", "size", "price", "direction"]].values.tolist() == [
        [2, 1, 30, P(10.02), -1], [2, 2, 20, P(10.00), 1]]


def test_only_messages_that_change_the_book_look_at_the_touch_and_each_says_what_it_skipped(tmp_path):
    """Without ``strict`` a crossed book is counted once for each message that changes the book and leaves it crossed.
    Messages that name no live order, a reference already live, trades, crosses and system events change nothing, so
    they are not counted (and a trading action that is not the first, and a first that is not 'T', are)."""
    crossed = [add(T0 - 2_000_000, 1, "B", 1, P(10.05)), add(T0 - 1_000_000, 2, "S", 1, P(10.00))]
    report = {}
    out = lobster_frames(read(tmp_path, action(T0 - 3_000_000, "H"), *crossed, add(T(1), 3, "B", 1, P(10.06)),
                              delete(T(2), 99), cancel(T(2), 99, 1), replace(T(2), 99, 5, 1, P(10.00)),
                              replace(T(2), 1, 3, 1, P(10.00)), add(T(2), 3, "B", 1, P(10.00)),
                              add(T(2), 4, "B", 0, P(10.00)), trade(T(3), 5, P(10.01), 1), cross(T(3), 5, P(10.01), 2),
                              system(T(3), "C"), delete(T(4), 3)), strict=False, report=report, levels=1)
    assert report["crossed_or_locked"] == 2  # the add at T(1) and the delete at T(4) leave it crossed, nothing else
    assert out.messages["type"].tolist() == [1, 6, 3]
    assert report["not_live_delete"] == report["not_live_cancel"] == report["not_live_replace"] == 1
    assert report["duplicate_reference"] == 2 and report["empty_add"] == 1  # the replace's new reference, and the add
    assert report["later_trading_action"] == 1  # the first action says 'H'


# the length of each ITCH 5.0 message in bytes, without the 2-byte prefix, from the specification (section 4)
SPEC_LENGTHS = {"S": 12, "R": 39, "H": 25, "Y": 20, "L": 26, "V": 35, "W": 12, "K": 28, "J": 35, "h": 21, "A": 36,
                "F": 40, "E": 31, "C": 36, "X": 23, "D": 19, "U": 35, "B": 19, "P": 44, "Q": 40, "I": 50, "N": 20}


def test_the_table_of_message_lengths_is_the_specifications_and_each_length_is_enforced(tmp_path):
    assert itch.MESSAGE_LENGTHS == SPEC_LENGTHS
    for kind, length in SPEC_LENGTHS.items():
        good = tmp_path / f"good_{ord(kind)}.itch"
        good.write_bytes(struct.pack(">H", length) + _head(kind, 1, T(0)).ljust(length, b"\0"))
        assert itch.count_messages(good) == {kind: 1}
        for wrong in (length - 1, length + 1):
            bad = tmp_path / f"bad_{ord(kind)}_{wrong}.itch"
            bad.write_bytes(struct.pack(">H", wrong) + _head(kind, 1, T(0)).ljust(wrong, b"\0"))
            with pytest.raises(ItchError, match=f"type {kind!r} and length {wrong}"):
                itch.count_messages(bad)


def test_errors_name_the_byte_of_the_message_and_a_length_over_255_is_read_from_both_bytes(tmp_path):
    good = add(T(0), 1, "B", 1, 1000000)
    late = tmp_path / "late.itch"
    late.write_bytes(framed(good) + struct.pack(">H", 20) + _head("Z", 1, T(1)).ljust(20, b"\0"))
    with pytest.raises(ItchError, match="message at byte 38 has type 'Z'"):  # 2 + 36 bytes of the first message
        read_messages(late)
    long = tmp_path / "long.itch"  # a prefix of 300 is 0x012C: the high byte counts
    long.write_bytes(struct.pack(">H", 300) + _head("A", 1, T(0)).ljust(300, b"\0"))
    with pytest.raises(ItchError, match="type 'A' and length 300; the specification says 36"):
        read_messages(long)
    short = tmp_path / "short.itch"  # a last message of one byte after its prefix has a type, and the wrong length
    short.write_bytes(framed(good) + struct.pack(">H", 1) + b"A")
    with pytest.raises(ItchError, match="type 'A' and length 1;"):
        read_messages(short)


def trade_with_reference(ref):
    body = bytearray(trade(T(0), 1, P(10.00), 1))
    body[11:19] = struct.pack(">Q", ref)
    return bytes(body)


@pytest.mark.parametrize("message, what", [
    (trade_with_reference(2**63), "a reference above 2"),
    (add(T(0), 2**63, "B", 1, P(10.00)), "a reference above 2"),
    (add(T(0), 2**63, "S", 1, P(10.00), mpid="GSCO"), "a reference above 2"),
    (replace(T(0), 1, 2**63, 1, P(10.00)), "a reference above 2"),
    (replace(T(0), 2**63, 2, 1, P(10.00)), "a reference above 2"),
    (cancel(T(0), 2**63, 1), "a reference above 2"),
    (delete(T(0), 2**63), "a reference above 2"),
    (execute(T(0), 2**63, 1, 5), "a reference above 2"),
    (execute(T(0), 1, 1, 2**63), "a match above 2"),
    (execute_with_price(T(0), 2**63, 1, 5, "Y", P(10.00)), "a reference above 2"),
    (execute_with_price(T(0), 1, 1, 2**63, "Y", P(10.00)), "a match above 2"),
    (trade(T(0), 1, P(10.00), 2**63), "a match above 2"),
    (cross(T(0), 1, P(10.00), 2**63), "a match above 2"),
    (cross(T(0), 2**63, P(10.00), 1), "a share count above 2"),
])
def test_a_number_that_does_not_fit_a_signed_eight_byte_integer_raises_and_names_the_field(tmp_path, message, what):
    path = tmp_path / "big.itch"
    path.write_bytes(framed(message))
    with pytest.raises(ItchError, match=f"^{what}"):
        read_messages(path)
    assert read(tmp_path, add(T(0), 2**63 - 1, "B", 1, P(10.00))).loc[0, "ref"] == 2**63 - 1


def test_the_error_for_a_byte_that_is_not_a_side_or_a_printable_flag_names_it(tmp_path):
    with pytest.raises(ItchError, match="^side byte outside"):
        read(tmp_path, add(T(0), 1, "X", 1, 1000000))
    with pytest.raises(ItchError, match="^side byte outside"):
        read(tmp_path, trade(T(0), 1, 1000000, 1, side="X"))
    with pytest.raises(ItchError, match="^printable byte outside"):
        read(tmp_path, execute_with_price(T(0), 1, 1, 1, "?", 1000000))


def test_locate_codes_above_255_are_read_from_both_bytes(tmp_path):
    # 8712, 520 and 8 share a low byte; a reader of the low byte alone takes the three stocks for one
    day = [directory(T0 - 5, "AAA", 8712), directory(T0 - 4, "BBB", 520), directory(T0 - 3, "CCC", 8),
           add(T(1), 1, "B", 100, P(10.00), "AAA", 8712), add(T(2), 2, "B", 100, P(10.00), "BBB", 520),
           add(T(3), 3, "B", 100, P(10.00), "CCC", 8), system(T(4), "E")]
    path = tmp_path / "day.itch"
    path.write_bytes(framed(*day))
    assert itch.scan(path)["directory"] == {"AAA": 8712, "BBB": 520, "CCC": 8}
    stats = itch.extract(path, ["AAA", "BBB"], tmp_path / "out")
    assert (stats["AAA"]["locate"], stats["BBB"]["locate"]) == (8712, 520)
    one = read_messages(tmp_path / "out" / "AAA.itch")
    assert one.loc[one["type"] == "A", "ref"].tolist() == [1] and one.attrs == {"symbol": "AAA", "locate": 8712}
    other = read_messages(tmp_path / "out" / "BBB.itch")
    assert other.loc[other["type"] == "A", "ref"].tolist() == [2]


def test_extract_makes_the_directory_it_writes_to_and_its_parents(tmp_path):
    path, _ = write_day(tmp_path)
    itch.extract(path, ["AAA"], tmp_path / "a" / "b" / "out")
    assert (tmp_path / "a" / "b" / "out" / "AAA.itch").exists()


def test_a_bad_message_in_a_later_block_is_named_by_its_byte_in_the_file(tmp_path, monkeypatch):
    good = [add(T(k), k + 1, "B", 1, P(10.00)) for k in range(5)]  # 5 messages of 38 bytes with their prefixes
    bad = tmp_path / "later.itch"
    bad.write_bytes(framed(*good) + struct.pack(">H", 20) + _head("Z", 1, T(9)).ljust(20, b"\0"))
    monkeypatch.setattr(itch, "_CHUNK", 64)
    with pytest.raises(ItchError, match="message at byte 190 has type 'Z'"):
        itch.scan(bad)
    cut = tmp_path / "cut.itch"
    cut.write_bytes(framed(*good) + framed(add(T(9), 9, "B", 1, P(10.00)))[:-3])
    with pytest.raises(ItchError, match=r"ends inside a message \(35 bytes after byte 190\)"):
        itch.scan(cut)


def test_the_code_column_is_blank_except_on_system_events_trading_actions_and_crosses(tmp_path):
    day = read(tmp_path, system(T(0), "Q"), action(T(1), "T"), add(T(2), 1, "B", 100, P(10.00)),
               execute(T(3), 1, 5, 7), cross(T(4), 5, P(10.00), 8, "C"), trade(T(5), 5, P(10.00), 9),
               replace(T(6), 1, 2, 1, P(10.00)), cancel(T(7), 2, 1), delete(T(8), 2))
    assert day.loc[day["type"].isin(list("SHQ")), "code"].tolist() == ["Q", "T", "C"]
    assert day.loc[~day["type"].isin(list("SHQ")), "code"].eq("").all()


def test_a_trades_reference_and_side_are_read_whatever_they_hold(tmp_path):
    body = bytearray(trade(T(0), 300, P(10.005), 80))
    body[11:19] = struct.pack(">Q", 123_456_789)  # Nasdaq sends 0 here; the frame says what the message says
    day = read(tmp_path, bytes(body), trade(T(1), 5, P(10.00), 81), trade(T(2), 6, P(10.00), 82, side="S"))
    assert day["ref"].tolist() == [123_456_789, 0, 0] and day["shares"].tolist() == [300, 5, 6]
    assert day["side"].tolist() == [1, 1, -1]  # 'B', 'B', 'S' (Nasdaq says 'B' of every one)


def test_an_empty_file_and_one_with_only_system_events_have_no_symbol_and_no_locate(tmp_path):
    for day in (read(tmp_path), read(tmp_path, system(T(0), "O"), system(T(1), "S"))):
        assert day.attrs == {"symbol": None, "locate": None}


def test_the_names_in_all_are_the_modules_public_names():
    assert sorted(itch.__all__) == sorted([
        "BOOK_TYPES", "CROSS", "EXTRA_COLUMNS", "HIDDEN_COLUMNS", "ItchError", "Frames", "MESSAGE_LENGTHS", "PROBLEMS",
        "count_messages", "extract", "load_frames", "lobster_frames", "read_messages", "save_frames", "scan"])
    assert all(hasattr(itch, name) for name in itch.__all__)


def test_save_frames_names_its_files_after_the_symbol_given_or_else_the_frames_own_and_makes_the_directory(tmp_path):
    pytest.importorskip("pyarrow")
    out = frames(tmp_path, *run_book([execute(T(5), 3, 10, 10)]), levels=2)
    target = tmp_path / "x" / "y" / "frames"  # its parents do not exist
    paths = itch.save_frames(out, target)  # no symbol: the frames' own
    assert sorted(p.name for p in paths.values()) == [
        "TEST_book.parquet", "TEST_frames.json", "TEST_hidden.parquet", "TEST_messages.parquet"]
    paths = itch.save_frames(out, target, "other")  # an existing directory, and a symbol that is not the frames' own
    assert sorted(p.name for p in paths.values()) == [
        "OTHER_book.parquet", "OTHER_frames.json", "OTHER_hidden.parquet", "OTHER_messages.parquet"]
    for was, now in zip(out, itch.load_frames(target, "other"), strict=True):
        pd.testing.assert_frame_equal(was, now)


# ---- Against the real day, from sources written apart from this module (added by the audit) ----

CMCSA = DERIVED / "raw" / "CMCSA.itch"
SCOUT_SERIES = DERIVED / "reference" / "nvda_replay_touch_series.json"
SCOUT_COUNTS = DERIVED / "reference" / "nvda_replay.json"
needs_scout = pytest.mark.skipif(
    not (NVDA.exists() and SCOUT_SERIES.exists() and SCOUT_COUNTS.exists()),
    reason="needs raw/NVDA.itch and reference/nvda_replay_touch_series.json and nvda_replay.json (ignored by git)")
needs_cmcsa = pytest.mark.skipif(
    not CMCSA.exists(), reason="needs data/itch/derived/03272019/raw/CMCSA.itch (ignored by git, not present)")


@needs_scout
def test_nvdas_touch_every_second_equals_the_scouts_series(nvda):
    """The scout's replay of NVDA has the best bid and ask at each of the 23,400 seconds from 09:30:00 (the book as it
    stood before the second began); the frames' book at the last row before each second says the same."""
    series = json.loads(SCOUT_SERIES.read_text())
    _, out, _ = nvda
    scouts = np.array(series["series"], dtype=np.int64)
    assert len(scouts) == 23_400 and series["t0_ns"] == T0
    seconds = T0 + np.arange(len(scouts), dtype=np.int64) * 10**9
    row = np.searchsorted(out.messages["ns"].to_numpy(), seconds, side="left") - 1
    mine = np.column_stack([np.nan_to_num(out.book["bid_price_1"].to_numpy()[row]),
                            np.nan_to_num(out.book["ask_price_1"].to_numpy()[row])]).astype(np.int64)
    assert (mine == scouts[:, :2]).all() and (mine > 0).all()


@needs_scout
def test_nvdas_counts_of_messages_by_type_are_the_scouts_too(nvda):
    """The scout counted NVDA's own messages (and the eight of no stock) by type, for the whole day and for 09:30 to
    16:00; ``count_messages`` and the frame of ``read_messages`` give the same."""
    counts = json.loads(SCOUT_COUNTS.read_text())
    assert itch.count_messages(NVDA) == counts["by_type_all_day"] and sum(counts["by_type_all_day"].values()) == 975_729
    messages, _, _ = nvda
    ns = messages["ns"].to_numpy()
    window = messages[(ns >= T0) & (ns < T0 + 23_400 * 10**9)]
    continuous = {k: v for k, v in counts["by_type_continuous"].items() if k != "I"}  # I is not a message read here
    assert window["type"].value_counts().to_dict() == continuous


@needs_day
def test_nvdas_opening_and_closing_crosses_are_the_independent_replays(nvda):
    """The crosses (Q): the review's replay has their time, shares, price and match number; the frames (here of the
    whole day) carry them as type 6 rows with no order and no side."""
    messages, out, _ = nvda
    reference = np.load(REFERENCE)
    ours = messages[messages["type"] == "Q"]
    q = np.flatnonzero(reference["typ"] == ord("Q"))
    assert ours["ns"].tolist() == reference["ts"][q].tolist() == [34_200_629_926_533, 57_600_192_938_732]
    assert ours["shares"].tolist() == reference["shares"][q].tolist() == [138_507, 382_465]
    assert ours["price"].tolist() == reference["price"][q].tolist() == [1_779_100, 1_765_000]  # $177.91 and $176.50
    assert ours["match"].tolist() == reference["match"][q].tolist() and ours["code"].tolist() == ["O", "C"]
    crosses = out.messages[out.messages["type"] == 6]
    assert crosses[["ns", "size", "price", "order_id", "direction"]].values.tolist() == [
        [34_200_629_926_533, 138_507, 1_779_100, 0, 0], [57_600_192_938_732, 382_465, 1_765_000, 0, 0]]
    assert (messages["type"] == "C").sum() == 143 and (messages.loc[messages["type"] == "C", "printable"] == "Y").all()
    assert int(out.messages["priced"].sum()) == 143 and not out.messages["paired"].any()  # no C-N on NVDA
    assert (out.messages.loc[out.messages["priced"], "type"] == 4).all()


@needs_day
def test_nvdas_hidden_trades_and_their_share_of_executed_shares_are_the_independent_replays(nvda):
    """09:30 to 16:00: 5,404 non-displayed trades (P) of 424,864 shares against 1,977,873 shares executed against
    displayed orders (E and printable C), 17.68 percent of all shares executed; 3,872 of the trades are in runs of
    hidden trades alone (no side; 3,876 before executions were joined across deletions), 619 are a buyer's and 913 a
    seller's."""
    messages, _, _ = nvda
    out = lobster_frames(messages, levels=1)
    shown = int(out.messages.loc[out.messages["type"] == 4, "size"].sum())
    assert (len(out.hidden), int(out.hidden["size"].sum()), shown) == (5_404, 424_864, 1_977_873)
    assert round(100 * 424_864 / (424_864 + shown), 2) == 17.68
    assert out.hidden["aggressor"].value_counts().sort_index().to_dict() == {-1: 913, 0: 3_872, 1: 619}
    assert len(set(out.messages.loc[out.messages["type"] == 4, "group"])) == 17_607
    assert out.messages.loc[out.messages["type"] == 4, "group"].nunique() + len(
        set(out.hidden["group"]) - set(out.messages["group"])) == 21_069
    assert int(out.hidden.loc[out.hidden["aggressor"] == 0, "size"].sum()) == 294_600

@pytest.fixture(scope="module")
def cmcsa():
    report = {}
    messages = read_messages(CMCSA)
    return messages, lobster_frames(messages, levels=1, report=report), report


@needs_cmcsa
def test_cmcsas_replay_meets_the_invariants_and_has_the_counts_of_an_independent_replay(cmcsa):
    """CMCSA (the one-tick exploration symbol) has no reference decoder. These numbers are from a replay written for the
    audit from the ITCH 5.0 specification alone (it shares no code with this module), checked there against the touch
    after every row of this module's frames, 09:30 to 16:00."""
    messages, out, report = cmcsa
    assert all(report[name] == 0 for name in itch.PROBLEMS) and report["live_orders_at_end"] == 0
    assert (report["adds_under_a_dollar"], report["adds_off_the_cent_grid"]) == (7, 2)
    assert len(out.messages) == len(out.book) == 924_487 and len(out.hidden) == 2_311
    assert out.messages["type"].value_counts().sort_index().to_dict() == {
        1: 461_351, 2: 1_941, 3: 442_375, 4: 18_818, 6: 2}  # 14 of the 16 type-6 rows were paired reductions
    assert out.messages.loc[out.messages["paired"], "type"].value_counts().to_dict() == {3: 8, 2: 6}
    assert int(out.messages["priced"].sum()) == 451
    assert int(out.messages["replace"].sum()) == 2 * 46_461
    assert int(out.messages.loc[out.messages["type"] == 4, "size"].sum()) == 3_166_822
    assert int(out.hidden["size"].sum()) == 341_612
    assert out.hidden["aggressor"].value_counts().sort_index().to_dict() == {-1: 269, 0: 1_725, 1: 317}
    crossing = out.messages[out.messages["type"] == 6]
    assert int((crossing["order_id"] != 0).sum()) == 1  # a printable-N execution of a displayed order at the cross
    assert crossing[crossing["order_id"] == 0][["ns", "size", "price"]].values.tolist() == [
        [34_200_334_263_519, 148_564, 396_800]]
    assert messages["type"].value_counts().to_dict() == {
        "A": 414_573, "D": 398_804, "U": 46_467, "E": 18_372, "X": 1_937, "P": 2_328, "F": 458, "C": 476, "S": 6,
        "Q": 2, "H": 1}
    flags = messages.loc[messages["type"] == "C", "printable"].value_counts().to_dict()
    assert flags == {"Y": 451, "N": 25}
    closing = messages[(messages["type"] == "Q") & (messages["code"] == "C")]
    assert closing[["ns", "shares", "price"]].values.tolist() == [[57_600_358_616_872, 1_704_416, 402_100]]
    # the groups: aggressive orders with a displayed execution, and runs of hidden trades alone
    shown = set(out.messages.loc[out.messages["type"] == 4, "group"])
    assert (len(shown), len(set(out.hidden["group"]) - shown)) == (6_874, 1_539)


# ---- After the judgment review (build/fix_report.md): each finding with the smallest case that shows it ----

def pair_book():
    """A buyer shows 100 shares at 10.00 (order 1), a seller shows 100 at 10.01 (order 2)."""
    return [add(T(1), 1, "B", 100, P(10.00)), add(T(2), 2, "S", 100, P(10.01))]


def test_a_non_printable_execution_that_shares_its_match_with_a_printed_one_is_a_reduction_not_a_cross(tmp_path):
    """CMCSA's 14 C-N rows in continuous trading (INTC 27, MSFT 211): the C-N comes first, the E follows at the same
    nanosecond with the same match number, against an order of the other side. The buyer's order, displayed at 10.00,
    executes at 10.01 against the seller's: one trade, printed once (the E). The C-N takes shares out of its order and
    is no cross: a reduction of the order at its displayed price, flagged ``paired``."""
    out = frames(tmp_path, *pair_book(), execute_with_price(T(3), 1, 30, 7, "N", P(10.01)),
                 execute(T(3), 2, 30, 7), levels=1)
    rows = out.messages
    assert rows["type"].tolist() == [1, 1, 2, 4] and not (rows["type"] == itch.CROSS).any()
    assert rows.iloc[2][["order_id", "size", "price", "direction", "match", "group"]].tolist() == [
        1, 30, P(10.00), 1, 7, -1]
    assert rows["paired"].tolist() == [False, False, True, False]
    assert rows.iloc[3][["order_id", "direction", "group", "match"]].tolist() == [2, -1, 0, 7]  # the aggressive order
    assert out.book.iloc[2][["bid_price_1", "bid_size_1"]].tolist() == [P(10.00), 70]
    assert out.book.iloc[3].tolist() == [P(10.01), 70, P(10.00), 70]
    # a C-N that takes the rest of its order is a deletion, with the shares that were left
    out = frames(tmp_path, *pair_book(), execute_with_price(T(3), 1, 100, 7, "N", P(10.01)),
                 execute(T(3), 2, 100, 7), levels=1)
    assert out.messages.iloc[2][["type", "size", "price", "direction"]].tolist() == [3, 100, P(10.00), 1]
    assert out.messages["paired"].tolist() == [False, False, True, False]


def test_a_non_printable_execution_paired_with_a_hidden_trade_is_a_reduction_too(tmp_path):
    """INTC has 3 and MSFT 7 whose match number a P carries. The P stays in the hidden table, unsigned here."""
    out = frames(tmp_path, *pair_book(), execute_with_price(T(3), 1, 30, 7, "N", P(10.01)),
                 trade(T(3), 30, P(10.01), 7), levels=1)
    assert out.messages["type"].tolist() == [1, 1, 2] and out.messages["paired"].tolist() == [False, False, True]
    assert out.hidden[["match", "aggressor"]].values.tolist() == [[7, 0]]


def test_a_non_printable_execution_at_a_cross_is_a_cross_and_so_is_nothing_else(tmp_path):
    """The opening cross of CMCSA: its C-N share a nanosecond with the Q. Type 6, reducing the book."""
    out = frames(tmp_path, *pair_book(), execute_with_price(T(3), 1, 100, 7, "N", P(10.00)),
                 cross(T(3), 5000, P(10.00), 9, "O"), levels=1)
    assert out.messages["type"].tolist() == [1, 1, 6, 6] and not out.messages["paired"].any()
    assert out.messages.iloc[2][["order_id", "size", "price", "direction"]].tolist() == [1, 100, P(10.00), 1]
    assert out.book.iloc[2][["bid_price_1", "bid_size_1"]].isna().tolist() == [True, False]
    # not at a cross, and no printed half: neither a trade between two orders nor a cross. Raised, or counted
    for message, why in (
            (execute(T(3), 2, 30, 8), "a different match number"), (execute(T(4), 2, 30, 7), "a different nanosecond"),
            (execute_with_price(T(3), 2, 30, 7, "N", P(10.01)), "another C-N")):
        messages = [*pair_book(), execute_with_price(T(3), 1, 30, 7, "N", P(10.01)), message]
        with pytest.raises(ItchError, match="unpaired_non_printable_execution"):
            frames(tmp_path, *messages, levels=1)
        report = {}
        out = frames(tmp_path, *messages, levels=1, strict=False, report=report)
        assert report["unpaired_non_printable_execution"] >= 1 and out.messages.iloc[2]["type"] == 6, why


def test_executions_of_one_aggressive_order_are_one_group_across_deletions_cancels_and_paired_reductions(tmp_path):
    """CMCSA: 85 aggressive orders in the session (NVDA 6) were cut in two by a same-side deletion between two
    executions with consecutive match numbers at one nanosecond (a self-match-prevention cancel, reasoned). So were
    the trades of a C-N and an E in a row. Joined across deletions, cancels of part and paired reductions; an add, a
    replace, a cross or a trading action still ends a run."""
    book = run_book([add(T(4) + 500, 5, "S", 50, P(10.02)), add(T(4) + 600, 6, "S", 50, P(10.02))])
    def one(*between, second=11):
        out = frames(tmp_path, *book, execute(T(5), 3, 10, 10), *between, execute(T(5), 3, 10, second), levels=1)
        return visible_groups(out)

    assert one() == [0, 0]
    assert one(delete(T(5), 5)) == [0, 0]
    assert one(delete(T(5), 5), cancel(T(5), 6, 10)) == [0, 0]
    assert one(add(T(5), 7, "S", 5, P(10.05))) == [0, 1]                # an add of the aggressor's remainder
    assert one(replace(T(5), 5, 8, 50, P(10.02))) == [0, 1]
    assert one(cross(T(5), 5, P(10.01), 99)) == [0, 1]
    assert one(delete(T(5), 5), second=12) == [0, 1]                    # not the next match number
    # the paired reductions of a run of C-N and E pairs: C-N 10, E 10, C-N 11, E 11 (two buyers' orders, displayed at
    # 10.00, execute at 10.01 against two sellers' orders)
    day = [add(T(1), 1, "S", 100, P(10.01)), add(T(1) + 1, 2, "S", 100, P(10.01)), add(T(2), 3, "B", 100, P(10.00)),
           add(T(2) + 1, 4, "B", 100, P(10.00))]
    out = frames(tmp_path, *day, execute_with_price(T(5), 3, 10, 10, "N", P(10.01)), execute(T(5), 1, 10, 10),
                 execute_with_price(T(5), 4, 10, 11, "N", P(10.01)), execute(T(5), 2, 10, 11), levels=1)
    assert out.messages["type"].tolist()[4:] == [2, 4, 2, 4] and out.messages["paired"].tolist()[4:] == [
        True, False, True, False]
    assert visible_groups(out) == [0, 0]
    # merge_within, the sensitivity, keeps its rule that nothing lies between the runs it joins (a deletion at another
    # nanosecond is common, and says nothing of one order)
    spaced = [execute(T(5), 3, 10, 10), execute(T(55), 3, 10, 11), delete(T(56), 5), execute(T(57), 3, 10, 12)]
    assert visible_groups(frames(tmp_path, *book, *spaced, levels=1)) == [0, 1, 2]
    assert visible_groups(frames(tmp_path, *book, *spaced, levels=1, merge_within=100e-6)) == [0, 0, 1]


def test_a_priced_execution_is_marked_on_its_row(tmp_path):
    """The frames did not say which type-4 rows are printable C (executed at their own price, a cent off their
    order's displayed price): the quote rule reads that price, so a check of the rule needs to set them apart."""
    out = frames(tmp_path, add(T(1), 1, "B", 100, P(10.00)), add(T(2), 2, "S", 100, P(10.05)),
                 execute(T(3), 1, 10, 4), execute_with_price(T(4), 1, 30, 5, "Y", P(10.01)), levels=1)
    assert out.messages["priced"].tolist() == [False, False, False, True]
    assert out.messages["type"].tolist() == [1, 1, 4, 4]
    assert "priced" in itch.EXTRA_COLUMNS and "paired" in itch.EXTRA_COLUMNS


def test_an_execution_or_a_cancel_of_no_shares_raises_and_is_skipped_when_counted(tmp_path):
    """A fill of 0 shares would enter VPIN's calibration median; neither day has one."""
    for name, bad in (("empty_execution", execute(T(3), 1, 0, 5)),
                      ("empty_execution", execute_with_price(T(3), 1, 0, 5, "Y", P(100.01))),
                      ("empty_cancel", cancel(T(3), 1, 0))):
        with pytest.raises(ItchError, match=f"^{name} at "):
            frames(tmp_path, *with_book(bad), levels=1)
        report = {}
        out = frames(tmp_path, *with_book(bad), levels=1, strict=False, report=report)
        assert report[name] == 1 and len(out.messages) == 2 and out.book.iloc[-1]["bid_size_1"] == 100
    # and so does a C-N of none, whether it is the other half of a trade or at a cross
    with pytest.raises(ItchError, match="^empty_execution at "):
        frames(tmp_path, *with_book(execute_with_price(T(3), 1, 0, 5, "N", P(100.01)), execute(T(3), 2, 1, 5)))
    with pytest.raises(ItchError, match="^empty_execution at "):
        frames(tmp_path, *with_book(execute_with_price(T(3), 1, 0, 5, "N", P(100.01)), cross(T(3), 9, P(100.01), 6)))
    assert {"empty_execution", "empty_cancel"} <= set(itch.PROBLEMS)


def test_a_touch_off_the_cent_grid_inside_the_window_raises_and_one_far_from_it_is_only_counted(tmp_path):
    """The cent tick is assumed (C10). Off-grid orders away from the touch are counted (NVDA has 2); an off-grid
    touch is a book the spread in ticks would be wrong for."""
    for messages in ([add(T(1), 1, "B", 10, P(10.005))], [add(T(1), 1, "S", 10, P(10.005))],
                     [add(T(1), 1, "B", 10, P(10.00)), add(T(2), 2, "S", 10, P(10.015))]):
        with pytest.raises(ItchError, match="^touch_off_the_cent_grid at "):
            frames(tmp_path, *messages)
        report = {}
        frames(tmp_path, *messages, strict=False, report=report)
        assert report["touch_off_the_cent_grid"] >= 1
    report = {}
    frames(tmp_path, add(T(1), 1, "B", 10, P(10.00)), add(T(2), 2, "S", 10, P(10.02)), add(T(3), 3, "B", 10, P(9.505)),
           report=report)
    assert report["touch_off_the_cent_grid"] == 0 and report["adds_off_the_cent_grid"] == 1
    assert "touch_off_the_cent_grid" in itch.PROBLEMS


def test_what_would_break_the_premises_about_p_and_c_is_counted(tmp_path):
    """ITCH's P carries 'B' on every hidden trade and no reference; a C is a cent from its order's displayed price
    (NVDA's and CMCSA's are, all of them). A file where that fails is counted, not refused: it reads the same."""
    report = {}
    frames(tmp_path, *pair_book(), trade(T(3), 5, P(10.005), 1, side="B"), report=report)
    assert (report["p_side_not_b"], report["p_with_reference"], report["c_price_over_a_cent"]) == (0, 0, 0)
    sold = trade(T(3), 5, P(10.005), 1, side="S")
    referenced = sold[:11] + struct.pack(">Q", 77) + sold[19:]
    report = {}
    frames(tmp_path, *pair_book(), sold, referenced, execute_with_price(T(4), 1, 5, 2, "Y", P(10.02)),
           execute_with_price(T(5), 1, 5, 3, "Y", P(10.01)), report=report)
    assert (report["p_side_not_b"], report["p_with_reference"], report["c_price_over_a_cent"]) == (2, 1, 1)
