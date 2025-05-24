"""
duonlabs.client submodule.

Copyright (c) 2025 Duon labs
"""
import os
import time
import warnings
import requests

from typing import Dict, List, Optional

from .utils import ListofListsofNumbers, freq2sec
from .forecast import Forecast


class DuonLabs:
    default_base_url: str = os.getenv("DUONLABS_API_URL", "https://api.duonlabs.com/v1/")
    headers: Dict[str, str] = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    supported_frequencies: List[str] = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "1d"]

    def __init__(self, token: str, base_url: Optional[str] = None):
        self.headers["Authorization"] = f"Token {token}"
        self.base_url = base_url or self.default_base_url

    def _scenario_generation(self, payload: Dict, passthrough: bool = False) -> Forecast:
        url = self.base_url + "scenarios/generation"
        if passthrough:
            url += "?passthrough=true"
        
        response = requests.post(
            url,
            headers=self.headers,
            json=payload,
            timeout=360,
        )
        response.raise_for_status()
        response = response.json()
        return Forecast(context=payload["inputs"]["candles"], scenarios=response["scenarios"])

    def forecast(
        self,
        pair: str,
        frequency: str,
        candles: Optional[ListofListsofNumbers] = None,
        model: str = "best",
        n_steps: int = 15,
        n_scenarios: int = 512,
        timestamp_unit: str = "s",
        last_candle: str = "auto",
        tag: Optional[str] = None,
        passthrough: bool = False,
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
            passthrough: bool = False | If True, inputs and outputs are not stored (requires special permissions)
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
            last_candle = "ongoing" if time.time() < candles[-1][0] / (1000 if timestamp_unit == "ms" else 1) + freq2sec[frequency] else "closed"
        # Prepare Request
        return self._scenario_generation({
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
        }, passthrough=passthrough)
