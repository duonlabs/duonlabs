import time

import pandas as pd
import pytest

from duonlabs.utils import steps_from_frames


def _frame(start_ts, freq_s, n, base=100.0, ts_as_index=True):
    ts = [start_ts + i * freq_s for i in range(n)]
    data = {
        "open":   [base + i for i in range(n)],
        "high":   [base + i + 1 for i in range(n)],
        "low":    [base + i - 1 for i in range(n)],
        "close":  [base + i + 0.5 for i in range(n)],
        "volume": [10.0 + i for i in range(n)],
    }
    if ts_as_index:
        df = pd.DataFrame(data, index=pd.Index(ts, name="timestamp"))
    else:
        df = pd.DataFrame({"timestamp": ts, **data})
    return df


def test_timestamp_as_index():
    end = int(time.time()) - 3600
    df = _frame(end - 9 * 60, 60, 10)
    payload = steps_from_frames({"binance.spot.BTCUSDT": df})
    assert payload["columns"][0] == "timestamp"
    assert len(payload["steps"]) == 10


def test_timestamp_as_column():
    end = int(time.time()) - 3600
    df = _frame(end - 9 * 60, 60, 10, ts_as_index=False)
    payload = steps_from_frames({"binance.spot.BTCUSDT": df})
    assert len(payload["steps"]) == 10


def test_key_order_preserved():
    end = int(time.time()) - 3600
    frames = {
        "binance.spot.ETHUSDT": _frame(end - 9 * 60, 60, 10, base=50.0),
        "binance.spot.BTCUSDT": _frame(end - 9 * 60, 60, 10, base=100.0),
    }
    payload = steps_from_frames(frames)
    assert payload["columns"][1].startswith("binance.spot.ETHUSDT")
    assert payload["columns"][6].startswith("binance.spot.BTCUSDT")


def test_missing_ohlcv_column_raises():
    end = int(time.time()) - 3600
    df = _frame(end - 9 * 60, 60, 10).drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing columns"):
        steps_from_frames({"binance.spot.BTCUSDT": df})


def test_misaligned_frames_raise():
    end = int(time.time()) - 3600
    frames = {
        "binance.spot.ETHUSDT": _frame(end - 9 * 60, 60, 10),
        "binance.spot.BTCUSDT": _frame(end - 8 * 60, 60, 10),
    }
    with pytest.raises(ValueError, match="misaligned"):
        steps_from_frames(frames)
