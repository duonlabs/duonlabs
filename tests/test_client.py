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


def test_fetch_steps_alignment_mismatch():
    client = DuonLabs(token="x")
    end = int(time.time()) - 3600
    start_eth_ms = (end - 199 * 60) * 1000
    start_btc_ms = start_eth_ms + 60_000  # shifted

    def fake_get(url, params, timeout):
        m = MagicMock()
        m.raise_for_status.return_value = None
        sym = params["symbol"]
        start = start_eth_ms if sym == "ETHUSDT" else start_btc_ms
        m.json.return_value = [
            [start + i * 60_000, "1", "2", "0", "1", "10"] for i in range(200)
        ]
        return m

    with patch("duonlabs.client.requests.get", side_effect=fake_get):
        with pytest.raises(ValueError, match="misaligned"):
            client.fetch_steps(["binance.spot.ETHUSDT", "binance.spot.BTCUSDT"], "1m")
