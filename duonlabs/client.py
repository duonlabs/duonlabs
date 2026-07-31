"""
duonlabs.client — v2 DuonLabs client.

Speaks the Voyons v2 inference dialect: fully-qualified column names,
unix-second timestamps, and a `task` field inferred from the number of keys.

Copyright (c) 2025 Duon labs
"""

import os
import requests

from typing import Any, Dict, List, Optional, Tuple, Union

from .forecast import Forecast
from .utils import _assemble_columns, _validate_steps_shape


SUPPORTED_FREQUENCIES = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "1d"]
SUPPORTED_TASKS = ["next_candle", "next_price", "multi_asset_candle"]

_FREQ_SECONDS: Dict[str, int] = {
    "1m": 60, "5m": 5 * 60, "15m": 15 * 60, "30m": 30 * 60,
    "1h": 60 * 60, "2h": 2 * 60 * 60, "4h": 4 * 60 * 60, "8h": 8 * 60 * 60,
    "1d": 24 * 60 * 60,
}

CONTEXT_SIZE = 256
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"


class DuonLabs:
    """Client for the Voyons v2 forecasting API."""

    default_base_url: str = os.getenv("DUONLABS_API_URL", "https://api.duonlabs.com/v1/")

    def __init__(self, token: str, base_url: Optional[str] = None):
        """
        Args:
            token: API token, sent as `Authorization: Token <token>`.
            base_url: Override the API root. Defaults to `$DUONLABS_API_URL` or the production URL.
        """
        self.headers: Dict[str, str] = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Token {token}",
        }
        self.base_url = base_url or self.default_base_url

    def forecast(
        self,
        keys: Union[str, List[str]],
        frequency: Optional[str] = None,
        steps: Optional[Dict[str, Any]] = None,
        model: str = "best",
        n_steps: int = 10,
        n_scenarios: int = 1024,
        seed: Optional[int] = None,
        top_p: Optional[float] = None,
        tag: Optional[str] = None,
        task: Optional[str] = None,
    ) -> Forecast:
        """Generate a forecast for one or more pair keys.

        Args:
            keys: One key (`"binance.spot.BTCUSDT"`) or a list of keys for multi-pair.
                The last key in the list is the primary in multi-pair forecasts.
            frequency: Candle frequency (e.g. `"4h"`). Required only when `steps` is not provided.
            steps: Pre-built `{columns, steps}` payload (use `duonlabs.utils.steps_from_*` helpers).
                When `None`, the SDK fetches from binance for every key.
            model: Model identifier; `"best"` resolves to a current production model.
            n_steps: Forecast horizon (number of future timesteps).
            n_scenarios: Number of parallel sampled scenarios.
            seed: RNG seed; server picks one if omitted.
            top_p: Nucleus sampling threshold; server default if omitted.
            tag: User-defined telemetry tag.
            task: Forecasting task. When omitted, inferred from the inputs: `multi_asset_candle`
                for multiple keys, else `next_candle` when a `.volume` column is present and
                `next_price` for volume-less (OHLC-only) inputs.

        Returns:
            A `Forecast` indexed by fully-qualified column names.
        """
        keys_list = [keys] if isinstance(keys, str) else list(keys)
        if not keys_list or not all(isinstance(k, str) for k in keys_list):
            raise ValueError("keys must be a non-empty str or list[str]")
        if steps is None:
            if frequency is None:
                raise ValueError("frequency is required when steps is not provided")
            if frequency not in SUPPORTED_FREQUENCIES:
                raise ValueError(f"frequency must be one of {SUPPORTED_FREQUENCIES}")
            steps = self.fetch_steps(keys_list, frequency)
        else:
            if "columns" not in steps or "steps" not in steps:
                raise ValueError("steps must be a dict with 'columns' and 'steps' keys")
            _validate_steps_shape(steps["columns"], steps["steps"])
        if task is None:
            if len(keys_list) > 1:
                task = "multi_asset_candle"
            else:
                task = "next_candle" if any(c.endswith(".volume") for c in steps["columns"]) else "next_price"
        elif task not in SUPPORTED_TASKS:
            raise ValueError(f"task must be one of {SUPPORTED_TASKS}")
        payload: Dict[str, Any] = {
            "inputs": {"columns": steps["columns"], "steps": steps["steps"]},
            "task": task,
            "n_steps": n_steps,
            "n_scenarios": n_scenarios,
            "model": model,
        }
        if seed is not None:
            payload["seed"] = seed
        if top_p is not None:
            payload["top_p"] = top_p
        if tag is not None:
            payload["tag"] = tag
        response = requests.post(
            self.base_url + "scenarios/generation",
            headers=self.headers,
            json=payload,
            timeout=360,
        )
        response.raise_for_status()
        body = response.json()
        infos = dict(body.get("infos") or {})
        infos.setdefault("model", model)
        if frequency is not None:
            infos.setdefault("frequency", frequency)
        if seed is not None:
            infos.setdefault("seed", seed)
        return Forecast(
            columns=body.get("columns", steps["columns"]),
            context_steps=steps["steps"],
            scenarios=body["scenarios"],
            infos=infos,
        )

    def fetch_steps(self, keys: Union[str, List[str]], frequency: str) -> Dict[str, Any]:
        """Fetch OHLCV candles for every key and assemble a wire-format payload.

        Each key is routed to a venue by its provider prefix: `binance.spot.*` goes to the
        binance klines endpoint, anything else is read as a hyperliquid market (core or
        builder-deployed dex). For providers neither venue serves, load data yourself and
        pass it via `steps=duonlabs.utils.steps_from_*(...)`.

        Args:
            keys: One key or a list of keys.
            frequency: Candle frequency (e.g. `"4h"`).

        Returns:
            `{"columns": [...], "steps": [...]}` aligned across all keys.
        """
        keys_list = [keys] if isinstance(keys, str) else list(keys)
        if frequency not in SUPPORTED_FREQUENCIES:
            raise ValueError(f"frequency must be one of {SUPPORTED_FREQUENCIES}")
        freq_seconds = _FREQ_SECONDS[frequency]
        per_key: Dict[str, List[List[float]]] = {}
        per_key_ts: Dict[str, List[int]] = {}
        for key in keys_list:
            per_key_ts[key], per_key[key] = self._fetch_key_ohlcv(key, frequency)
        ## Cross-key alignment: every key must have the same timestamp set
        ## Venues disagree on history depth, so intersect rather than requiring equality:
        ## a binance key and a hyperliquid key rarely return the same window.
        ref_ts = sorted(set.intersection(*(set(ts) for ts in per_key_ts.values())))
        if len(ref_ts) < 2:
            raise ValueError(
                f"keys {keys_list!r} share fewer than 2 aligned timestamps at frequency {frequency!r}; "
                "fetch data yourself and pass via duonlabs.utils.steps_from_frames"
            )
        for key in keys_list:
            index = {t: i for i, t in enumerate(per_key_ts[key])}
            per_key[key] = [per_key[key][index[t]] for t in ref_ts]
        ## Drop the partial last candle if its close-time is in the future
        import time as _time
        while ref_ts and ref_ts[-1] + freq_seconds > _time.time():
            ref_ts = ref_ts[:-1]
            for key in keys_list:
                per_key[key] = per_key[key][:-1]
        columns = _assemble_columns(keys_list)
        steps: List[List[Union[int, float]]] = []
        for i, t in enumerate(ref_ts):
            row: List[Union[int, float]] = [t]
            for key in keys_list:
                row.extend(per_key[key][i])
            steps.append(row)
        _validate_steps_shape(columns, steps)
        return {"columns": columns, "steps": steps}

    def _fetch_key_ohlcv(self, key: str, frequency: str) -> Tuple[List[int], List[List[float]]]:
        """Fetch one key's candles from whichever venue its provider prefix names.

        Args:
            key: Fully-qualified key, e.g. `binance.spot.BTCUSDT` or `xyz.perp.NVDA-USDC`.
            frequency: Candle frequency.

        Returns:
            `(timestamps_seconds, [[open, high, low, close, volume], ...])`, oldest first.
        """
        if key.startswith("binance."):
            return self._fetch_binance_ohlcv(self._binance_symbol(key), frequency)
        return self._fetch_hyperliquid_ohlcv(self._hyperliquid_coin(key), frequency)

    @staticmethod
    def _fetch_binance_ohlcv(symbol: str, frequency: str) -> Tuple[List[int], List[List[float]]]:
        """Fetch `CONTEXT_SIZE` candles for a binance symbol from the klines endpoint."""
        r = requests.get(
            BINANCE_KLINES_URL,
            params={"interval": frequency, "limit": CONTEXT_SIZE, "symbol": symbol},
            timeout=10,
        )
        r.raise_for_status()
        ts: List[int] = []
        ohlcv: List[List[float]] = []
        for row in r.json():
            ts.append(int(row[0]) // 1000)
            ohlcv.append([float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])])
        return ts, ohlcv

    @staticmethod
    def _fetch_hyperliquid_ohlcv(coin: str, frequency: str) -> Tuple[List[int], List[List[float]]]:
        """Fetch `CONTEXT_SIZE` candles for a hyperliquid coin from the info endpoint.

        `candleSnapshot` is window-addressed rather than count-addressed, so the window is
        sized from the frequency and the tail is trimmed to `CONTEXT_SIZE`.
        """
        import time as _time
        end_ms = int(_time.time() * 1000)
        start_ms = end_ms - (CONTEXT_SIZE + 1) * _FREQ_SECONDS[frequency] * 1000
        r = requests.post(
            HYPERLIQUID_INFO_URL,
            json={"type": "candleSnapshot", "req": {"coin": coin, "interval": frequency, "startTime": start_ms, "endTime": end_ms}},
            timeout=10,
        )
        r.raise_for_status()
        candles = r.json() or []
        if not candles:
            raise ValueError(f"hyperliquid returned no candles for coin {coin!r} at frequency {frequency!r}")
        ts: List[int] = []
        ohlcv: List[List[float]] = []
        for candle in candles[-CONTEXT_SIZE:]:
            ts.append(int(candle["t"]) // 1000)
            ohlcv.append([float(candle["o"]), float(candle["h"]), float(candle["l"]), float(candle["c"]), float(candle["v"])])
        return ts, ohlcv

    @staticmethod
    def _binance_symbol(key: str) -> str:
        """Parse a `binance.spot.SYMBOL` key into the symbol used by the binance klines endpoint."""
        parts = key.split(".")
        if len(parts) != 3 or parts[0] != "binance" or parts[1] != "spot":
            raise ValueError(
                f"binance fetch only supports keys of the form 'binance.spot.SYMBOL' (got {key!r}); "
                "for other providers/markets, fetch data yourself and pass via duonlabs.utils.steps_from_frames"
            )
        return parts[2]

    @staticmethod
    def _hyperliquid_coin(key: str) -> str:
        """Parse a `<provider>.perp.<BASE>-<QUOTE>` key into a hyperliquid `coin` identifier.

        `hyperliquid` is the core perp venue and addresses markets by bare base symbol;
        every other provider is a builder-deployed (HIP-3) dex, addressed as `<dex>:<BASE>`.
        """
        parts = key.split(".")
        if len(parts) != 3 or parts[1] != "perp":
            raise ValueError(
                f"hyperliquid fetch only supports keys of the form '<provider>.perp.<BASE>-<QUOTE>' (got {key!r}); "
                "for other providers/markets, fetch data yourself and pass via duonlabs.utils.steps_from_frames"
            )
        base = parts[2].split("-")[0]
        return base if parts[0] == "hyperliquid" else f"{parts[0]}:{base}"
