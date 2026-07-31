import time
from unittest.mock import patch, MagicMock

import pytest

from duonlabs import DuonLabs, Forecast
from duonlabs.utils import steps_from_ccxt


def _ccxt_rows(start_ts_ms, freq_ms, n, base=100.0):
    return [
        [start_ts_ms + i * freq_ms, base + i, base + i + 1, base + i - 1, base + i + 0.5, 10.0 + i]
        for i in range(n)
    ]


def _server_response(payload):
    """Build a fake server response matching the inputs in payload."""
    cols = payload["inputs"]["columns"]
    n_steps = payload["n_steps"]
    n_scenarios = payload["n_scenarios"]
    last_ts = payload["inputs"]["steps"][-1][0]
    diff = payload["inputs"]["steps"][-1][0] - payload["inputs"]["steps"][-2][0]
    scenarios = []
    for _ in range(n_scenarios):
        scen = []
        for i in range(n_steps):
            row = [last_ts + (i + 1) * diff] + [1.0] * (len(cols) - 1)
            scen.append(row)
        scenarios.append(scen)
    return {"columns": cols, "scenarios": scenarios, "infos": {"seed": 42}}


def test_keys_str_normalized_to_list():
    end = int(time.time()) - 3600
    start_ms = (end - 9 * 60) * 1000
    steps = steps_from_ccxt({"binance.spot.BTCUSDT": _ccxt_rows(start_ms, 60_000, 10)})
    client = DuonLabs(token="x")
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["payload"] = json
        m = MagicMock()
        m.json.return_value = _server_response(json)
        m.raise_for_status.return_value = None
        return m

    with patch("duonlabs.client.requests.post", side_effect=fake_post):
        fc = client.forecast(keys="binance.spot.BTCUSDT", steps=steps)
    assert captured["payload"]["task"] == "next_candle"
    assert isinstance(fc, Forecast)


def test_multi_pair_task_inference():
    end = int(time.time()) - 3600
    start_ms = (end - 9 * 60) * 1000
    steps = steps_from_ccxt({
        "binance.spot.ETHUSDT": _ccxt_rows(start_ms, 60_000, 10, base=50.0),
        "binance.spot.BTCUSDT": _ccxt_rows(start_ms, 60_000, 10, base=100.0),
    })
    client = DuonLabs(token="x")
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["payload"] = json
        m = MagicMock()
        m.json.return_value = _server_response(json)
        m.raise_for_status.return_value = None
        return m

    with patch("duonlabs.client.requests.post", side_effect=fake_post):
        client.forecast(keys=["binance.spot.ETHUSDT", "binance.spot.BTCUSDT"], steps=steps)
    assert captured["payload"]["task"] == "multi_asset_candle"
    assert captured["payload"]["inputs"]["columns"][-5:] == [
        "binance.spot.BTCUSDT.open", "binance.spot.BTCUSDT.high",
        "binance.spot.BTCUSDT.low", "binance.spot.BTCUSDT.close", "binance.spot.BTCUSDT.volume",
    ]


def _ohlc_steps(start_ts_ms, freq_ms, n, key="pyth.Crypto.BTC-USD", base=100.0):
    """Build a volume-less OHLC {columns, steps} payload directly (no builder helper)."""
    columns = ["timestamp"] + [f"{key}.{c}" for c in ("open", "high", "low", "close")]
    steps = [
        [(start_ts_ms + i * freq_ms) // 1000, base + i, base + i + 1, base + i - 1, base + i + 0.5]
        for i in range(n)
    ]
    return {"columns": columns, "steps": steps}


def _capture_post(captured):
    def fake_post(url, headers, json, timeout):
        captured["payload"] = json
        m = MagicMock()
        m.json.return_value = _server_response(json)
        m.raise_for_status.return_value = None
        return m
    return fake_post


def test_next_price_inferred_for_volumeless_input():
    end = int(time.time()) - 3600
    start_ms = (end - 9 * 60) * 1000
    steps = _ohlc_steps(start_ms, 60_000, 10)
    client = DuonLabs(token="x")
    captured = {}
    with patch("duonlabs.client.requests.post", side_effect=_capture_post(captured)):
        fc = client.forecast(keys="pyth.Crypto.BTC-USD", steps=steps)
    assert captured["payload"]["task"] == "next_price"
    assert fc["pyth.Crypto.BTC-USD.close"].shape == (1024, 10)
    assert fc.cutoff("pyth.Crypto.BTC-USD.close") == pytest.approx(100.0 + 9 + 0.5)


def test_explicit_task_overrides_inference():
    end = int(time.time()) - 3600
    start_ms = (end - 9 * 60) * 1000
    steps = steps_from_ccxt({"binance.spot.BTCUSDT": _ccxt_rows(start_ms, 60_000, 10)})
    client = DuonLabs(token="x")
    captured = {}
    with patch("duonlabs.client.requests.post", side_effect=_capture_post(captured)):
        client.forecast(keys="binance.spot.BTCUSDT", steps=steps, task="next_price")
    assert captured["payload"]["task"] == "next_price"


def test_unknown_task_rejected():
    end = int(time.time()) - 3600
    start_ms = (end - 9 * 60) * 1000
    steps = steps_from_ccxt({"binance.spot.BTCUSDT": _ccxt_rows(start_ms, 60_000, 10)})
    client = DuonLabs(token="x")
    with pytest.raises(ValueError, match="task must be one of"):
        client.forecast(keys="binance.spot.BTCUSDT", steps=steps, task="bogus")


def test_payload_omits_optional_fields_when_none():
    end = int(time.time()) - 3600
    start_ms = (end - 9 * 60) * 1000
    steps = steps_from_ccxt({"binance.spot.BTCUSDT": _ccxt_rows(start_ms, 60_000, 10)})
    client = DuonLabs(token="x")
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["payload"] = json
        m = MagicMock()
        m.json.return_value = _server_response(json)
        m.raise_for_status.return_value = None
        return m

    with patch("duonlabs.client.requests.post", side_effect=fake_post):
        client.forecast(keys="binance.spot.BTCUSDT", steps=steps)
    payload = captured["payload"]
    assert "seed" not in payload
    assert "top_p" not in payload
    assert "tag" not in payload


def test_frequency_required_without_steps():
    client = DuonLabs(token="x")
    with pytest.raises(ValueError, match="frequency is required"):
        client.forecast(keys="binance.spot.BTCUSDT")


def test_bad_frequency_rejected():
    client = DuonLabs(token="x")
    with pytest.raises(ValueError, match="frequency must be"):
        client.forecast(keys="binance.spot.BTCUSDT", frequency="7m")


def test_fetch_steps_non_binance_spot_key_rejected():
    client = DuonLabs(token="x")
    with pytest.raises(ValueError, match="binance.spot"):
        client.fetch_steps(["binance.futures.um.BTCUSDT"], "4h")


def _shifted_klines_get(start_a_ms: int, start_b_ms: int, n: int = 200):
    """Build a requests.get double serving two symbols over differently-offset windows."""
    def fake_get(url, params, timeout):
        m = MagicMock()
        m.raise_for_status.return_value = None
        start = start_a_ms if params["symbol"] == "ETHUSDT" else start_b_ms
        m.json.return_value = [
            [start + i * 60_000, "1", "2", "0", "1", "10"] for i in range(n)
        ]
        return m
    return fake_get


def test_fetch_steps_alignment_intersects_overlapping_windows():
    """Keys whose windows only partly overlap are intersected down to the shared tail."""
    client = DuonLabs(token="x")
    end = int(time.time()) - 3600
    start_eth_ms = (end - 199 * 60) * 1000
    start_btc_ms = start_eth_ms + 60_000  # shifted by one candle

    with patch("duonlabs.client.requests.get", side_effect=_shifted_klines_get(start_eth_ms, start_btc_ms)):
        payload = client.fetch_steps(["binance.spot.ETHUSDT", "binance.spot.BTCUSDT"], "1m")
    assert len(payload["steps"]) == 199
    assert payload["steps"][0][0] == start_btc_ms // 1000


def test_fetch_steps_alignment_disjoint_windows_rejected():
    """Windows that share no timestamps cannot be aligned at all."""
    client = DuonLabs(token="x")
    end = int(time.time()) - 3600
    start_eth_ms = (end - 199 * 60) * 1000
    start_btc_ms = start_eth_ms - 1000 * 60 * 1000  # far enough back to not overlap

    with patch("duonlabs.client.requests.get", side_effect=_shifted_klines_get(start_eth_ms, start_btc_ms)):
        with pytest.raises(ValueError, match="fewer than 2 aligned timestamps"):
            client.fetch_steps(["binance.spot.ETHUSDT", "binance.spot.BTCUSDT"], "1m")


@pytest.mark.parametrize("key, coin", [
    ("hyperliquid.perp.BTC-USDC", "BTC"),
    ("xyz.perp.NVDA-USDC", "xyz:NVDA"),
    ("km.perp.AAPL-USDH", "km:AAPL"),
    ("cash.perp.TSLA-USDT0", "cash:TSLA"),
])
def test_hyperliquid_coin_parsing(key, coin):
    """Core perps address by bare symbol; builder-deployed dexes by `<dex>:<BASE>`."""
    assert DuonLabs._hyperliquid_coin(key) == coin


def test_hyperliquid_coin_rejects_non_perp_key():
    with pytest.raises(ValueError, match="<provider>.perp"):
        DuonLabs._hyperliquid_coin("hyperliquid.spot.PURR-USDC")


def test_fetch_steps_routes_non_binance_key_to_hyperliquid():
    """A hydromancer dex key is fetched from the hyperliquid info endpoint, not binance."""
    client = DuonLabs(token="x")
    end = int(time.time()) - 3600
    start_ms = (end - 199 * 60) * 1000

    def fake_post(url, json, timeout):
        assert json["req"]["coin"] == "xyz:NVDA"
        m = MagicMock()
        m.raise_for_status.return_value = None
        m.json.return_value = [
            {"t": start_ms + i * 60_000, "o": "1", "h": "2", "l": "0", "c": "1", "v": "10", "n": 3}
            for i in range(200)
        ]
        return m

    with patch("duonlabs.client.requests.post", side_effect=fake_post):
        payload = client.fetch_steps("xyz.perp.NVDA-USDC", "1m")
    assert payload["columns"] == ["timestamp"] + [f"xyz.perp.NVDA-USDC.{c}" for c in ("open", "high", "low", "close", "volume")]
    assert len(payload["steps"]) == 200
