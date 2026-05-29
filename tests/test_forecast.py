import json

import numpy as np
import pytest

from duonlabs import Forecast


def _make_forecast(n_scenarios=4, n_steps=3, n_ctx=5):
    keys = ["binance.spot.ETHUSDT", "binance.spot.BTCUSDT"]
    columns = ["timestamp"]
    for k in keys:
        for c in ("open", "high", "low", "close", "volume"):
            columns.append(f"{k}.{c}")
    ctx = []
    for i in range(n_ctx):
        row = [1700000000 + i * 60]
        for k_idx in range(len(keys)):
            base = 100.0 if k_idx == 0 else 50_000.0
            row.extend([base + i, base + i + 1, base + i - 1, base + i + 0.5, 10.0 + i])
        ctx.append(row)
    scenarios = []
    for s in range(n_scenarios):
        scen = []
        for i in range(n_steps):
            row = [1700000000 + (n_ctx + i) * 60]
            for k_idx in range(len(keys)):
                base = 100.0 if k_idx == 0 else 50_000.0
                row.extend([base + s + i, base + s + i + 1, base + s + i - 1, base + s + i + 0.5, 10.0 + s + i])
            scen.append(row)
        scenarios.append(scen)
    return Forecast(columns=columns, context_steps=ctx, scenarios=scenarios, infos={"model": "test"})


def test_basic_shape():
    fc = _make_forecast()
    assert fc.n_scenarios == 4
    assert fc.n_steps == 3
    assert fc.keys == ["binance.spot.ETHUSDT", "binance.spot.BTCUSDT"]
    assert fc["binance.spot.BTCUSDT.close"].shape == (4, 3)
    assert fc.context["binance.spot.BTCUSDT.close"].shape == (5,)
    assert len(fc) == 4


def test_unknown_column_raises():
    fc = _make_forecast()
    with pytest.raises(KeyError):
        fc["close"]
    with pytest.raises(KeyError):
        fc.cutoff("close")


def test_cutoff():
    fc = _make_forecast()
    # Last context BTC close = 50_000 + 4 + 0.5
    assert fc.cutoff("binance.spot.BTCUSDT.close") == pytest.approx(50_004.5)


def test_scenario_int_and_slice():
    fc = _make_forecast()
    s0 = fc.scenario(0)
    assert isinstance(s0, dict)
    assert s0["binance.spot.BTCUSDT.close"].shape == (3,)
    sliced = fc.scenario(slice(0, 2))
    assert isinstance(sliced, Forecast)
    assert len(sliced) == 2
    assert sliced["binance.spot.BTCUSDT.close"].shape == (2, 3)


def test_probability_expectation_quantile():
    fc = _make_forecast()
    p = fc.probability(lambda s: bool(np.any(s["binance.spot.BTCUSDT.high"] > 50_000)))
    assert p == 1.0
    e = fc.expectation(lambda s: float(s["binance.spot.BTCUSDT.close"][-1]))
    assert e > 50_000
    q = fc.quantile(lambda s: float(s["binance.spot.BTCUSDT.close"][-1]), 0.5)
    assert q > 50_000


def test_min_max_highest_lowest_by_column():
    fc = _make_forecast()
    assert fc.min("binance.spot.BTCUSDT.low") < fc.max("binance.spot.BTCUSDT.high")
    hi = fc.highest("binance.spot.BTCUSDT.high")
    lo = fc.lowest("binance.spot.BTCUSDT.low")
    assert isinstance(hi, dict) and isinstance(lo, dict)


def test_min_max_highest_lowest_by_callable():
    fc = _make_forecast()
    by_last_close = lambda s: float(s["binance.spot.BTCUSDT.close"][-1])
    assert fc.min(by_last_close) <= fc.max(by_last_close)


def test_dump_load_roundtrip(tmp_path):
    fc = _make_forecast()
    path = tmp_path / "fc.json"
    fc.dump(path)
    data = json.loads(path.read_text())
    assert set(data.keys()) == {"columns", "context_steps", "scenarios", "infos"}
    fc2 = Forecast.load_json(path)
    assert fc2.columns == fc.columns
    np.testing.assert_array_equal(fc["binance.spot.BTCUSDT.close"], fc2["binance.spot.BTCUSDT.close"])
    np.testing.assert_array_equal(fc.context["binance.spot.BTCUSDT.close"], fc2.context["binance.spot.BTCUSDT.close"])
    assert fc2.infos["model"] == "test"
