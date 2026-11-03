"""Tests for the dashboard's live stylized facts (app/facts.py)."""

import numpy as np
import pytest
from scipy import stats

from app import facts
from app.facts import FactsData, FactsRecorder, Snapshot, summarize
from lob.analysis.stylized_facts import (
    classify_depth_profile, compute_acf, compute_rank_volatility_acf, fit_loglog_slope, mean_depth_profile,
)
from lob.book import OrderBook
from lob.matching import PriceTimeEngine
from lob.metrics import event_time_returns, mean_virtual_impact
from lob.models.long_memory import LongMemorySigns
from lob.models.mike_farmer import MikeFarmer2008Model
from lob.order import reset_order_id_counter
from lob.simulation import Simulation, SimulationConfig


def replay(make_model, seed, n_events):
    """Feed a recorder the event stream Simulation.run produces for the same model and seed."""
    reset_order_id_counter()
    book, engine, rng, model = OrderBook(), PriceTimeEngine(), np.random.default_rng(seed), make_model()
    recorder = FactsRecorder()
    t = 0.0
    for _ in range(n_events + facts.WARMUP_STEPS):
        order = model.generate_event(book, t, rng)
        if order is None:
            recorder.observe_idle()
            continue
        t = order.timestamp
        fills = engine.process(book, order)
        recorder.observe(order.order_type, order.side, fills, book)
    return recorder


def test_recorder_measures_what_notebook_05_measures():
    """Same model and seed: the recorder's sample and statistics equal notebook 05's, computed from a Simulation."""
    def make_model():
        return MikeFarmer2008Model(initial_mid=10000)

    seed, n_events = 3, 200_000  # about 130 returns: this model often steps without an event
    reset_order_id_counter()
    config = SimulationConfig(n_events=n_events, warmup_events=facts.WARMUP_STEPS, seed=seed,
                              record_book_snapshots=True, snapshot_interval=facts.SNAPSHOT_EVERY)
    sim = Simulation(model=make_model(), config=config)
    df = sim.run(progress=False)
    valid = df.dropna(subset=["mid_price"])
    returns = event_time_returns(valid["mid_price"], facts.K_EVENTS)
    signs = np.where(df.loc[df["event_type"] != "cancel", "side"] == "bid", 1.0, -1.0)
    trade_prices = sim.trades.groupby("taker_order_id", sort=False)["price"].last().to_numpy(dtype=float)
    _, bid_depth, ask_depth = mean_depth_profile(sim.snapshots, n_ticks=facts.DEPTH_TICKS)
    impact_sizes, impact = mean_virtual_impact(sim.snapshots, facts.IMPACT_SIZES)

    data = replay(make_model, seed, n_events).data()
    assert data.events == len(df)
    np.testing.assert_allclose(np.diff(data.log_mids), returns)
    np.testing.assert_array_equal(data.signs, signs)
    np.testing.assert_array_equal(data.trade_prices, trade_prices)
    assert len(data.snapshots) == len(sim.snapshots)
    quoted = [s for s in data.snapshots if s.bid_depth is not None]
    np.testing.assert_allclose(np.mean([s.bid_depth for s in quoted], axis=0), bid_depth)
    np.testing.assert_allclose(np.mean([s.ask_depth for s in quoted], axis=0), ask_depth)

    rows = {row["fact"]: row for row in summarize(data, "mike_farmer_2008")["rows"]}
    assert rows["Heavy-tailed returns"]["value"] == pytest.approx(stats.kurtosis(returns))
    assert rows["Volatility clustering"]["value"] == pytest.approx(compute_rank_volatility_acf(returns, 1)[1])
    assert rows["Long memory in order signs"]["value"] == pytest.approx(compute_acf(signs, 100)[100])
    assert rows["Bid-ask bounce"]["value"] == pytest.approx(compute_acf(np.diff(trade_prices), 1)[1])
    assert rows["Concave price impact"]["value"] == pytest.approx(fit_loglog_slope(impact_sizes, impact))
    shape, rise, _, peak = classify_depth_profile((bid_depth + ask_depth) / 2)
    assert rows["Hump-shaped depth profile"]["call"] == ("Yes" if shape == "hump" else "No")

    # Order flow imbalance per interval of OFI_EVENTS events, as the library computes it from the log.
    from lob.metrics import order_flow_imbalance

    e = order_flow_imbalance(df["best_bid"], df["bid_size"], df["best_ask"], df["ask_size"])
    k, mid = facts.OFI_EVENTS, df["mid_price"].to_numpy()
    n = len(e) // k
    ofi = e[: n * k].reshape(n, k).sum(axis=1)
    dmid = mid[k: (n + 1) * k: k] - mid[: n * k: k]
    quoted = np.isfinite(dmid)
    np.testing.assert_allclose(data.ofi, ofi[quoted])
    np.testing.assert_allclose(data.ofi_dmid, dmid[quoted])


def synthetic(returns=(), signs=(), trade_prices=(), snapshots=(), steps=None):
    return FactsData(steps=facts.WARMUP_STEPS + 1 if steps is None else steps, events=1,
                     log_mids=np.concatenate([[0.0], np.cumsum(returns)]),
                     signs=np.asarray(signs, dtype=float), trade_prices=np.asarray(trade_prices, dtype=float),
                     snapshots=list(snapshots))


def calls(data, model="smith_zi"):
    return {row["fact"]: row["call"] for row in summarize(data, model)["rows"]}


def test_return_facts_are_graded_like_the_notebook():
    rng = np.random.default_rng(0)
    assert calls(synthetic(returns=rng.normal(size=1000)))["Heavy-tailed returns"] == "No"
    # Laplace returns have excess kurtosis 3 and finite moments, so the bootstrap error is well behaved.
    assert calls(synthetic(returns=rng.laplace(size=4000)))["Heavy-tailed returns"] == "Yes"

    # Volatility that switches between calm and busy regimes clusters; independent returns do not.
    vol = np.repeat(rng.choice([0.2, 3.0], size=100), 20)
    assert calls(synthetic(returns=rng.normal(size=2000) * vol))["Volatility clustering"] == "Yes"
    assert calls(synthetic(returns=rng.normal(size=2000)))["Volatility clustering"] == "No"


def test_sign_and_bounce_facts():
    rng = np.random.default_rng(1)
    assert calls(synthetic(signs=rng.choice([-1, 1], size=200_000)))["Long memory in order signs"] == "Not detected"
    generator = LongMemorySigns(hurst=0.8)
    long_memory = [generator.next(rng) for _ in range(200_000)]
    assert calls(synthetic(signs=long_memory))["Long memory in order signs"] == "Yes"

    # Trades alternating between the bid and the ask bounce; a random walk of trade prices does not.
    bouncing = 100 + rng.choice([0, 1], size=5000) + np.cumsum(rng.normal(scale=0.05, size=5000))
    assert calls(synthetic(trade_prices=bouncing))["Bid-ask bounce"] == "Yes"
    walk = 100 + np.cumsum(rng.choice([-1, 1], size=5000))
    assert calls(synthetic(trade_prices=walk))["Bid-ask bounce"] == "Not detected"


def snapshots_with(profile, impact):
    depth = np.asarray(profile, dtype=float)
    return [Snapshot(depth, depth, np.asarray(impact, dtype=float)) for _ in range(facts.MIN_SNAPSHOTS)]


def test_depth_profile_and_impact_facts():
    sizes = facts.IMPACT_SIZES.astype(float)
    hump = np.concatenate([np.linspace(10, 40, 6), np.linspace(40, 5, 14)])
    rows = calls(synthetic(snapshots=snapshots_with(hump, np.sqrt(sizes))))
    assert rows["Hump-shaped depth profile"] == "Yes"
    assert rows["Concave price impact"] == "Yes"

    rows = calls(synthetic(snapshots=snapshots_with(np.linspace(40, 5, 20), sizes)))
    assert rows["Hump-shaped depth profile"] == "No"
    assert rows["Concave price impact"] == "No"  # linear impact: slope 1

    thin = np.where(sizes <= 3, sizes, np.nan)
    assert calls(synthetic(snapshots=snapshots_with(hump, thin)))["Concave price impact"] == "Not measurable"


def test_readings_wait_for_enough_data_and_placement_is_by_design():
    data = synthetic(returns=np.ones(10), signs=[1] * 10, trade_prices=[100, 101], steps=facts.WARMUP_STEPS + 1)
    summary = summarize(data, "mike_farmer_2008")
    assert {row["call"] for row in summary["rows"][:6]} == {"Collecting"}
    assert summary["rows"][0]["progress"] == pytest.approx(10 / facts.MIN_RETURNS)
    assert summary["rows"][6]["call"] == "Yes" and "Student" in summary["rows"][6]["note"]
    assert summarize(data, "manual")["rows"][6]["call"] == "Not applicable"
    assert summarize(synthetic(steps=10), "smith_zi")["warming_up"]


def test_recorder_keeps_only_the_most_recent_data(monkeypatch):
    monkeypatch.setattr(facts, "MAX_SIGNS", 100)
    recorder = replay(lambda: MikeFarmer2008Model(initial_mid=10000), seed=1, n_events=2000)
    assert len(recorder.data().signs) == 100
    assert len(recorder._signs) <= 200  # trimmed as it grows


def test_reset_starts_a_new_sample():
    recorder = replay(lambda: MikeFarmer2008Model(initial_mid=10000), seed=1, n_events=2000)
    generation = recorder.generation
    recorder.reset()
    data = recorder.data()
    assert recorder.generation == generation + 1
    assert data.steps == 0 and data.events == 0 and len(data.signs) == 0


def notebook_05_models():
    """The keyword arguments notebook 05 builds each model with, read from its source."""
    import ast
    import json
    from pathlib import Path

    notebook = Path(__file__).resolve().parents[1] / "notebooks" / "05_agent_based_models.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    source = next("".join(c["source"]) for c in cells
                  if c["cell_type"] == "code" and "MODELS = {" in "".join(c["source"]))
    env, models = {}, {}
    for node in ast.parse(source).body:
        if not (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)):
            continue
        if node.targets[0].id == "ZI_PARAMS":
            env["ZI_PARAMS"] = {kw.arg: ast.literal_eval(kw.value) for kw in node.value.keywords}
        elif node.targets[0].id == "MODELS":
            for key, value in zip(node.value.keys, node.value.values, strict=True):
                call = value.body  # lambda: Model(...)
                kwargs = {}
                for kw in call.keywords:
                    kwargs.update(env[kw.value.id] if kw.arg is None else {kw.arg: ast.literal_eval(kw.value)})
                models[ast.literal_eval(key)] = (call.func.id, kwargs)
    return models


def test_notebook_presets_match_notebook_05():
    import inspect

    from app.model_specs import MODEL_SPECS
    from lob.models.hawkes import HawkesZIModel
    from lob.models.mike_farmer import MikeFarmerModel
    from lob.models.strategic import ParlourModel
    from lob.models.zero_intelligence import SmithFarmerModel

    classes = {c.__name__: c for c in (SmithFarmerModel, HawkesZIModel, MikeFarmerModel, MikeFarmer2008Model,
                                       ParlourModel)}
    keys = {"ZI (Smith et al.)": "smith_zi", "ZI + Hawkes": "hawkes_zi", "Mike-Farmer (simplified)": "mike_farmer",
            "Mike-Farmer (2008 structure)": "mike_farmer_2008", "Parlour (Strategic)": "parlour"}

    def constructor_defaults(cls):
        defaults = {}
        for klass in reversed(cls.__mro__):  # base classes first, so subclasses override
            if "__init__" in vars(klass):
                for name, param in inspect.signature(klass.__init__).parameters.items():
                    if param.default is not inspect.Parameter.empty:
                        defaults[name] = param.default
        return defaults

    models = notebook_05_models()
    assert set(models) == set(keys)
    for label, (class_name, kwargs) in models.items():
        spec = MODEL_SPECS[keys[label]]
        effective = {**constructor_defaults(classes[class_name]), **kwargs}
        assert spec.notebook == {p.name: effective[p.name] for p in spec.params}, label
        for param in spec.params:  # within the dashboard's allowed ranges
            assert param.parse(spec.notebook[param.name]) == spec.notebook[param.name]
