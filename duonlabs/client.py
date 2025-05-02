"""
duonlabs.client submodule.

Copyright (c) 2025 Duon labs
"""
import os
import time
import warnings
import requests

from typing import Dict, List, Optional

from .candles import CandleListofLists
from .forecast import Forecast


class DuonLabs:
    default_base_url: str = os.getenv("DUONLABS_API_URL", "https://api.duonlabs.com/v1/")
    headers: Dict[str, str] = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    supported_frequencies: List[str] = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "1d"]
    freq2sec: Dict[str, int] = {
        "1m": 60,
        "5m": 5 * 60,
        "15m": 15 * 60,
        "30m": 30 * 60,
        "1h": 60 * 60,
        "2h": 2 * 60 * 60,
        "4h": 4 * 60 * 60,
        "8h": 8 * 60 * 60,
        "1d": 24 * 60 * 60,
    }

    def __init__(self, token: str, base_url: Optional[str] = None):
        self.headers["Authorization"] = f"Token {token}"
        self.base_url = base_url or self.default_base_url

    def forecast(
        self,
        pair: str,
        frequency: str,
        candles: Optional[CandleListofLists] = None,
        model: str = "best",
        n_steps: int = 15,
        n_scenarios: int = 512,
        timestamp_unit: str = "s",
        last_candle: str = "auto",
        tag: Optional[str] = None,
    ) -> Forecast:
        """
        Args:
            pair: str | Name of the pair to forecast.
            frequency: str | Frequency of the candles (1m, 5m, 30m, 2h, 8h, 1d).
            candles: List[List[Union[int, float]]] (context_size, 6) | ccxt/binance format:
                [[timestamp (int), open (float), high (float), low (float), close (float), volume (float)], ...]
                If None, the model will fetch latest data from the exchange.
            n_steps: int = 15 | Number of sampling steps
            n_scenarios: int = 32 | Number of scenarios to generate
            timestamp_unit: ("s" | "ms") = "ms" | Unit of the timestamps
            last_candle: ("auto" | "closed" | "ongoing") = "auto" | How to handle the last candle.
                Auto will assume that the informations sent are the most up to date and use the current timestamp to decide
                Note that if last_candle is set or resolved to "ongoing", the first forecasted candle will be the current one (at its closing time).
                otherwise, the first forecasted candle will be the next one.
            tag: Optional[str] = None | Optional user defined tag for telemetry
        """
        # Validate Inputs
        assert isinstance(pair, str), "pair must be a string"
        assert frequency in self.supported_frequencies, f"frequency must be one of {self.supported_frequencies}"
        if candles is not None:
            assert isinstance(candles, list), "candles must be a list"
            assert all(isinstance(candle, list) and len(candle) == 6 for candle in candles), "candles must be a list of lists of 6 elements: [timestamp (int), open (float), high (float), low (float), close (float), volume (float)]"
        assert isinstance(n_steps, int) and n_steps > 0, "n_steps must be a positive integer"
        assert isinstance(n_scenarios, int) and n_scenarios > 0, "n_scenarios must be a positive integer"
        assert timestamp_unit in {"s", "ms"}, "timestamp_unit must be 's' or 'ms'"
        assert last_candle in {"auto", "closed", "ongoing"}, "last_candle must be 'auto', 'closed' or 'ongoing'"
        # Fetch Latest Data
        if candles is None:
            try:
                import ccxt
            except ImportError:
                warnings.warn("ccxt module is not installed. Please install it to fetch latest data from the exchange.")
                return
            exchange = ccxt.binance()
            candles = exchange.fetch_ohlcv(pair, frequency, limit=120)
            if timestamp_unit == "s":
                candles = [[candle[0] // 1000] + candle[1:] for candle in candles]
            if last_candle == "closed":
                candles.pop()
        if last_candle == "auto":
            last_candle = "ongoing" if time.time() < candles[-1][0] * (1000 if timestamp_unit == "ms" else 1) + self.freq2sec[frequency] else "closed"
        # Prepare Request
        response = requests.post(
            self.base_url + "scenarios/generation",
            headers=self.headers,
            json={
                "inputs": {
                    "pair": pair,
                    "frequency": frequency,
                    "candles": candles,
                    "timestamp_unit": timestamp_unit,
                    "last_candle": last_candle,
                },
                "model": model,
                "n_steps": n_steps,
                "n_scenarios": n_scenarios,
                "tag": tag,
            },
            timeout=360,
        )
        response.raise_for_status()
        response = response.json()
        return Forecast(candles=candles, scenarios=response["scenarios"])
