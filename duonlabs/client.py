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

    def _scenario_generation(self, payload: Dict) -> Forecast:
        response = requests.post(
            self.base_url + "scenarios/generation",
            headers=self.headers,
            json=payload,
            timeout=360,
        )
        response.raise_for_status()
        response = response.json()
        columns = payload["inputs"].get("columns", ["timestamp", "open", "high", "low", "close", "volume"])
        return Forecast(context=payload["inputs"]["steps"], scenarios=response["scenarios"], infos=response["infos"], channel_names=columns)

    def forecast(
        self,
        pair: str,
        frequency: str,
        candles: Optional[ListofListsofNumbers] = None,
        steps: Optional[ListofListsofNumbers] = None,
        model: str = "best",
        n_steps: int = 15,
        n_scenarios: int = 512,
        timestamp_unit: str = "s",
        last_candle: str = "auto",
        columns: Optional[List[str]] = None,
        tag: Optional[str] = None,
        **kwargs
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
            assert steps is None
        if steps is not None:
            assert isinstance(steps, list), "steps must be a list"
            assert columns is not None, "columns must be provided if steps is provided"
            assert all(isinstance(step, list) and len(step) == len(columns) for step in steps), "steps must be a list of lists with the same length as columns"
            assert candles is None, "candles must be None if steps is provided"
        assert isinstance(n_steps, int) and n_steps > 0, "n_steps must be a positive integer"
        assert isinstance(n_scenarios, int) and n_scenarios > 0, "n_scenarios must be a positive integer"
        assert timestamp_unit in {"s", "ms"}, "timestamp_unit must be 's' or 'ms'"
        assert last_candle in {"auto", "closed", "ongoing"}, "last_candle must be 'auto', 'closed' or 'ongoing'"
        # Fetch Latest Data
        if candles is None and steps is None:
            assert columns is None, "columns must be None if candles is None"
            response = requests.get(
                f"https://api.binance.com/api/v3/klines?interval={frequency}&limit=120&symbol={pair.replace('/', '')}",
                timeout=10,
            )
            response.raise_for_status()
            raw_candles = response.json()
            candles = []
            for candle in raw_candles:
                candles.append([
                    int(candle[0]) // (1000 if timestamp_unit == "s" else 1),  # timestamp
                    float(candle[1]),  # open
                    float(candle[2]),  # high
                    float(candle[3]),  # low
                    float(candle[4]),  # close
                    float(candle[5]),  # volume
                    int(candle[6]),  # close time
                    float(candle[7]),  # quote asset volume
                    int(candle[8]),  # number of trades
                    float(candle[9]),  # taker buy base asset volume
                    float(candle[10]),  # taker buy quote asset volume
                ])
            columns = ["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]
            if last_candle == "closed":
                candles.pop()
        else:
            if columns is None:
                columns = ["timestamp", "open", "high", "low", "close", "volume"]
        steps = candles or steps
        if last_candle == "auto":
            last_candle = "ongoing" if time.time() < steps[-1][0] / (1000 if timestamp_unit == "ms" else 1) + freq2sec[frequency] else "closed"
        # Prepare Request
        return self._scenario_generation({
            "inputs": {
                "pair": pair,
                "frequency": frequency,
                "columns": columns,
                "steps": steps,
                "timestamp_unit": timestamp_unit,
                "last_candle": last_candle,
            },
            "model": model,
            "n_steps": n_steps,
            "n_scenarios": n_scenarios,
            "tag": tag,
        }, **kwargs)
