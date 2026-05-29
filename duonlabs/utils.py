"""
duonlabs.utils — input-building helpers for the v2 SDK.

Two builders convert user-held data into the handler-native wire shape
(`{columns, steps}` keyed by fully-qualified `key.column` names):

- `steps_from_frames(frames)`   — from `{key: pandas.DataFrame}`
- `steps_from_ccxt(rows_by_key)` — from `{key: list[[ts, o, h, l, c, v]]}`

Both share the validation rules in `_validate_steps_shape`.

Copyright (c) 2025 Duon labs
"""

import time

from typing import Any, Dict, List, Union

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def _validate_steps_shape(columns: List[str], steps: List[List[Union[int, float]]]) -> None:
    """Shared assertions over a freshly-built {columns, steps} payload.

    Args:
        columns: Column names, must start with "timestamp".
        steps: Rows of values matching `columns` order.

    Raises:
        ValueError: on any shape, timestamp, or spacing violation.
    """
    if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
        raise ValueError("columns must be a list of strings")
    if not columns or columns[0] != "timestamp":
        raise ValueError("columns[0] must be 'timestamp'")
    if len(set(columns)) != len(columns):
        raise ValueError("columns must be unique")
    if not isinstance(steps, list) or len(steps) < 2:
        raise ValueError("steps must be a list with at least 2 rows")
    width = len(columns)
    for i, row in enumerate(steps):
        if not isinstance(row, list) or len(row) != width:
            raise ValueError(f"steps[{i}] must be a list of length {width}")
    timestamps = [row[0] for row in steps]
    for t in timestamps:
        if not isinstance(t, (int, float)):
            raise ValueError("timestamps must be numeric")
    diffs = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    if any(d <= 0 for d in diffs):
        raise ValueError("timestamps must be strictly increasing")
    if len(set(diffs)) != 1:
        raise ValueError(f"timestamps must be regularly spaced (got {len(set(diffs))} distinct diffs)")
    freq_seconds = diffs[0]
    if timestamps[-1] + freq_seconds > time.time():
        raise ValueError(
            "the last candle is not closed: "
            f"last_timestamp ({timestamps[-1]}) + freq ({freq_seconds}s) is in the future. "
            "Drop the partial last row before forecasting."
        )


def _assemble_columns(keys: List[str]) -> List[str]:
    """Build the canonical column list for a set of keys: timestamp + key.<ohlcv> for each key in order."""
    out = ["timestamp"]
    for k in keys:
        for c in OHLCV_COLUMNS:
            out.append(f"{k}.{c}")
    return out


def steps_from_frames(frames: Dict[str, Any]) -> Dict[str, Any]:
    """Build the wire-format steps payload from a dict of pandas DataFrames.

    Each DataFrame must:
        - have a `timestamp` column (int, unix seconds) OR a `timestamp`-named index,
        - contain `open`, `high`, `low`, `close`, `volume` columns,
        - share the exact same timestamp set as every other frame.

    Args:
        frames: Mapping `{key: dataframe}`. Insertion order is preserved; the last
            key is the primary in multi-pair forecasts.

    Returns:
        `{"columns": [...], "steps": [...]}` ready to pass to `client.forecast(steps=...)`.

    Raises:
        ImportError: if pandas is not installed.
        ValueError: on missing columns, misaligned timestamps, or shape violations.
    """
    try:
        import pandas as pd  # noqa: F401
    except ImportError as e:
        raise ImportError("steps_from_frames requires pandas — install with `pip install duonlabs[pandas]`") from e
    if not isinstance(frames, dict) or not frames:
        raise ValueError("frames must be a non-empty dict of {key: DataFrame}")
    keys = list(frames.keys())
    ref_ts = None
    cleaned: Dict[str, Any] = {}
    for key, df in frames.items():
        d = df.copy()
        if "timestamp" in d.columns:
            d = d.set_index("timestamp")
        elif d.index.name != "timestamp":
            raise ValueError(f"frame for {key!r} must have 'timestamp' as a column or index name")
        missing = [c for c in OHLCV_COLUMNS if c not in d.columns]
        if missing:
            raise ValueError(f"frame for {key!r} missing columns: {missing}")
        ts = [int(t) for t in d.index.tolist()]
        if ref_ts is None:
            ref_ts = ts
        elif ts != ref_ts:
            raise ValueError(f"frame for {key!r} has misaligned timestamps vs {keys[0]!r}")
        cleaned[key] = d
    columns = _assemble_columns(keys)
    steps: List[List[Union[int, float]]] = []
    for i, t in enumerate(ref_ts):
        row: List[Union[int, float]] = [t]
        for key in keys:
            df = cleaned[key]
            for c in OHLCV_COLUMNS:
                row.append(float(df[c].iloc[i]))
        steps.append(row)
    _validate_steps_shape(columns, steps)
    return {"columns": columns, "steps": steps}


def steps_from_ccxt(
    rows_by_key: Dict[str, List[List[Union[int, float]]]],
    timestamp_unit: str = "ms",
) -> Dict[str, Any]:
    """Build the wire-format steps payload from a dict of ccxt OHLCV rows.

    Args:
        rows_by_key: `{key: [[ts, o, h, l, c, v], ...]}` as returned by `ccxt.fetch_ohlcv`.
            Insertion order is preserved; the last key is the primary.
        timestamp_unit: `"ms"` (ccxt default) or `"s"`. Converted to seconds internally.

    Returns:
        `{"columns": [...], "steps": [...]}` ready to pass to `client.forecast(steps=...)`.

    Raises:
        ValueError: on bad row shape, misaligned timestamps, or shape violations.
    """
    if timestamp_unit not in {"ms", "s"}:
        raise ValueError("timestamp_unit must be 'ms' or 's'")
    if not isinstance(rows_by_key, dict) or not rows_by_key:
        raise ValueError("rows_by_key must be a non-empty dict of {key: rows}")
    keys = list(rows_by_key.keys())
    divisor = 1000 if timestamp_unit == "ms" else 1
    ref_ts = None
    cleaned: Dict[str, List[List[float]]] = {}
    for key, rows in rows_by_key.items():
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"rows_by_key[{key!r}] must be a non-empty list")
        ts: List[int] = []
        ohlcv: List[List[float]] = []
        for j, row in enumerate(rows):
            if not isinstance(row, list) or len(row) != 6:
                raise ValueError(f"rows_by_key[{key!r}][{j}] must be a 6-element list [ts, o, h, l, c, v]")
            ts.append(int(row[0]) // divisor)
            ohlcv.append([float(x) for x in row[1:]])
        if ref_ts is None:
            ref_ts = ts
        elif ts != ref_ts:
            raise ValueError(f"rows for {key!r} have misaligned timestamps vs {keys[0]!r}")
        cleaned[key] = ohlcv
    columns = _assemble_columns(keys)
    steps: List[List[Union[int, float]]] = []
    for i, t in enumerate(ref_ts):
        row: List[Union[int, float]] = [t]
        for key in keys:
            row.extend(cleaned[key][i])
        steps.append(row)
    _validate_steps_shape(columns, steps)
    return {"columns": columns, "steps": steps}
