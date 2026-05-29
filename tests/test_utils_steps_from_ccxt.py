import time

import pytest

from duonlabs.utils import steps_from_ccxt


def _rows(start_ts_ms, freq_ms, n, base=100.0):
    """Generate ccxt-style rows starting before now so the last candle is closed."""
    return [
        [start_ts_ms + i * freq_ms, base + i, base + i + 1, base + i - 1, base + i + 0.5, 10.0 + i]
        for i in range(n)
    ]


def test_single_key_ms_default():
    end = int(time.time()) - 3600
    start_ms = (end - 9 * 60) * 1000
    payload = steps_from_ccxt({"binance.spot.BTCUSDT": _rows(start_ms, 60_000, 10)})
    assert payload["columns"] == [
        "timestamp",
        "binance.spot.BTCUSDT.open", "binance.spot.BTCUSDT.high",
        "binance.spot.BTCUSDT.low", "binance.spot.BTCUSDT.close", "binance.spot.BTCUSDT.volume",
    ]
    assert len(payload["steps"]) == 10
    assert payload["steps"][0][0] == start_ms // 1000


def test_timestamp_unit_seconds():
    end = int(time.time()) - 3600
    start_s = end - 9 * 60
    rows = [[start_s + i * 60, 1.0, 2.0, 0.5, 1.5, 10.0] for i in range(10)]
    payload = steps_from_ccxt({"binance.spot.BTCUSDT": rows}, timestamp_unit="s")
    assert payload["steps"][0][0] == start_s


def test_multi_key_order_preserved_and_aligned():
    end = int(time.time()) - 3600
    start_ms = (end - 9 * 60) * 1000
    rows = {
        "binance.spot.ETHUSDT": _rows(start_ms, 60_000, 10, base=50.0),
        "binance.spot.BTCUSDT": _rows(start_ms, 60_000, 10, base=100.0),
    }
    payload = steps_from_ccxt(rows)
    # Column order: timestamp, then ETH OHLCV, then BTC OHLCV — primary (BTC) is last.
    assert payload["columns"][1:6] == [f"binance.spot.ETHUSDT.{c}" for c in ("open", "high", "low", "close", "volume")]
    assert payload["columns"][6:11] == [f"binance.spot.BTCUSDT.{c}" for c in ("open", "high", "low", "close", "volume")]


def test_misaligned_timestamps_raises():
    end = int(time.time()) - 3600
    start_ms = (end - 9 * 60) * 1000
    rows = {
        "binance.spot.ETHUSDT": _rows(start_ms, 60_000, 10),
        "binance.spot.BTCUSDT": _rows(start_ms + 60_000, 60_000, 10),
    }
    with pytest.raises(ValueError, match="misaligned"):
        steps_from_ccxt(rows)


def test_irregular_spacing_raises():
    rows = [[1700000000_000 + i * 60_000, 1, 2, 0, 1, 10] for i in range(5)]
    rows.append([rows[-1][0] + 90_000, 1, 2, 0, 1, 10])
    with pytest.raises(ValueError, match="regularly spaced"):
        steps_from_ccxt({"binance.spot.BTCUSDT": rows})


def test_ongoing_last_candle_rejected():
    now = int(time.time())
    start_ms = (now - 60) * 1000  # last candle close is in the future
    with pytest.raises(ValueError, match="not closed"):
        steps_from_ccxt({"binance.spot.BTCUSDT": _rows(start_ms, 60_000, 3)})


def test_bad_row_shape():
    with pytest.raises(ValueError, match="6-element"):
        steps_from_ccxt({"binance.spot.BTCUSDT": [[1, 2, 3]]})


def test_bad_timestamp_unit():
    with pytest.raises(ValueError, match="timestamp_unit"):
        steps_from_ccxt({"k": [[1, 1, 1, 1, 1, 1]]}, timestamp_unit="us")
