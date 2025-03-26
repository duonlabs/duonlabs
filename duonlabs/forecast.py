"""
duonlabs.forecast submodule.

Copyright (c) 2025 Duon labs
"""

import json
import numpy as np

from typing import Any, Callable, Dict, List, TextIO, Union
from contextlib import nullcontext

from .candles import CandleListofLists, CANDLE_CHANNELS_NAME


class Forecast:
    def __init__(self, candles: CandleListofLists, scenarios: List[CandleListofLists]):
        """
        Args:
            candles: List[List[Union[int, float]]] (context_size, 6) | ccxt/binance format:
                [[timestamp (int), open (float), high (float), low (float), close (float), volume (float)], ...]
            scenarios: List[List[List[Union[int, float]]]] (n_scenarios, context_size, 6) | ccxt/binance format:
                [[[timestamp (int), open (float), high (float), low (float), close (float), volume (float)], ...], ...]
        """
        self.context = {"timestamps": [], "samples": []}
        for candle in candles:
            self.context["timestamps"].append(candle[0])
            self.context["samples"].append(candle[1:])
        self.scenarios = {"timestamps": [], "samples": []}
        for scenario in scenarios:
            self.scenarios["timestamps"].append([])
            self.scenarios["samples"].append([])
            for candle in scenario:
                self.scenarios["timestamps"][-1].append(candle[0])
                self.scenarios["samples"][-1].append(candle[1:])
        # Save as numpy arrays
        self.context = {"timestamps": np.array(self.context["timestamps"], dtype=np.int64), "samples": np.array(self.context["samples"], dtype=np.float32)}
        self.scenarios = {"timestamps": np.array(self.scenarios["timestamps"], dtype=np.int64), "samples": np.array(self.scenarios["samples"], dtype=np.float32)}
        # Save cutoff close
        self.cutoff_close = self.context["samples"][-1][3]

    def dump(self, f: Union[str, TextIO]):
        """
        Save the forecast to a json file.
        """
        with open(f, "w", encoding="utf-8") if isinstance(f, str) else nullcontext(f) as f:
            json.dump({
                "candles": [[t] + s for t, s in zip(self.context["timestamps"].tolist(), self.context["samples"].tolist())],
                "scenarios": [[[t] + s for t, s in zip(st, ss)] for st, ss in zip(self.scenarios["timestamps"].tolist(), self.scenarios["samples"].tolist())],
            }, f, indent=2)

    def map(self, f: Callable[[np.ndarray, Dict[str, np.ndarray]], Any]) -> List[Any]:
        """
        Apply a function to each scenario.
        Args:
            f: Callable[[Dict[str, np.ndarray]], Any] | Function to apply to each scenario.
                The argument is a dictionary with the keys "timestamp", "open", "high", "low", "close", "volume".
        Returns:
            List[Any] | List of results of the function applied to each scenario.
        """
        return list(map(lambda i: f(self[i]), range(len(self))))

    def probability(self, event: Callable[[Dict[str, np.ndarray]], bool]) -> float:
        """
        Compute the probability of an event.
        Args:
            event: Callable[[Dict[str, np.ndarray]], bool] | Function that takes a scenario and returns a boolean.
                The boolean indicates if the event happened in the scenario.
        Returns:
            float | Probability of the event happening.
        """
        return self.expectation(lambda s: float(event(s)))

    def expectation(self, f: Callable[[Dict[str, np.ndarray]], float]) -> float:
        """
        Compute the expectation of a quantity.
        Args:
            f: Callable[[Dict[str, np.ndarray]], float] | Function that computes the quantity of interest given a scenario.
        Returns:
            float | Expectation of the quantity of interest.
        """
        return np.nanmean(self.map(f)).item()

    def highest(self, f: Union[str, Callable[[Dict[str, np.ndarray]], float]]) -> Dict[str, np.ndarray]:
        """
        Return the scenario having the highest value of a quantity.
        Args:
            f: One of:
                str | Name of the channel to maximize
                Callable[[Dict[str, np.ndarray]], float] | Function that computes the quantity of interest given a scenario.
        Returns:
            Dict[str, np.ndarray] | Scenario with the highest value of the quantity.
        """
        return self[np.argmax(self.map((lambda s: s[f].max()) if isinstance(f, str) else f))]

    def lowest(self, f: Union[str, Callable[[np.ndarray, Dict[str, np.ndarray]], float]]) -> Dict[str, np.ndarray]:
        """
        Return the scenario having the lowest value of a quantity.
        Args:
            f: One of:
                str | Name of the channel to minimize
                Callable[[Dict[str, np.ndarray]], float] | Function that computes the quantity of interest given a scenario.
        Returns:
            Dict[str, np.ndarray] | Scenario with the lowest value of the quantity.
        """
        return self[np.argmin(self.map((lambda s: s[f].min()) if isinstance(f, str) else f))]

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """
        Return the scenario at the given index.
        Args:
            index: int | Index of the scenario to return.
        Returns:
            Dict[str, np.ndarray] | Scenario with the keys "timestamp", "open", "high", "low", "close", "volume".
        """
        return {"timestamp": self.scenarios["timestamps"][index]} | {k: self.scenarios["samples"][index, :, i] for i, k in enumerate(CANDLE_CHANNELS_NAME[1:])}

    def __len__(self) -> int:
        """
        Return the number of scenarios.
        """
        return len(self.scenarios["samples"])

    @classmethod
    def load_json(cls, f: Union[str, TextIO]) -> "Forecast":
        """
        Load a forecast from a json file.
        """
        with open(f, "r", encoding="utf-8") if isinstance(f, str) else nullcontext(f) as f:
            data = json.load(f)
        return cls(**data)
