"""
duonlabs.forecast submodule.

Copyright (c) 2025 Duon labs
"""

import json
import numpy as np

from typing import Any, Callable, Dict, List, TextIO, Union
from pathlib import Path
from contextlib import nullcontext

from .utils import ListofListsofNumbers


class Forecast:
    channel_names = ["timestamp", "open", "high", "low", "close", "volume"]
    channel_dtypes = [int, float, float, float, float, float]

    def __init__(self, context: ListofListsofNumbers, scenarios: List[ListofListsofNumbers], infos: Dict[str, Any] = None, channel_names: List[str] = None):
        """
        Args:
            context: List[List[Union[int, float]]] (context_size, 6) | ccxt/binance format:
                [[timestamp (int), open (float), high (float), low (float), close (float), volume (float)], ...]
            scenarios: List[List[List[Union[int, float]]]] (n_scenarios, context_size, 6) | ccxt/binance format:
                [[[timestamp (int), open (float), high (float), low (float), close (float), volume (float)], ...], ...]
        """
        self.infos = infos or {}
        self.n_scenarios = len(scenarios)
        channel_names = channel_names or self.channel_names
        self.context = {k: [] for k in channel_names}
        for row in context:
            for c, v in zip(channel_names, row):
                self.context[c].append(v)
        self.scenarios = []
        for scenario in scenarios:
            scenario_dict = {k: [] for k in channel_names}
            for row in scenario:
                for c, v in zip(channel_names, row):
                    scenario_dict[c].append(v)
            self.scenarios.append(scenario_dict)
        # Save as numpy arrays
        self.context = {k: np.array(v) for k, v in self.context.items()}
        self.scenarios = [{k: np.array(v) for k, v in scenario.items()} for scenario in self.scenarios] 
        # Save cutoff close
        self.cutoff_close = self.context["close"][-1]

    def dump(self, f: Union[str, Path, TextIO]):
        """
        Save the forecast to a json file.
        """
        with open(f, "w", encoding="utf-8") if isinstance(f, (str, Path)) else nullcontext(f) as f:
            json.dump({
                "context": list(map(list, zip(*[v.tolist() for v in self.context.values()]))),
                "scenarios": [list(map(list, zip(*[v.tolist() for v in scenario.values()]))) for scenario in self.scenarios],
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
    
    def quantile(self, f: Callable[[Dict[str, np.ndarray]], float], q: float) -> float:
        """
        Compute the quantile of a quantity.
        Args:
            f: Callable[[Dict[str, np.ndarray]], float] | Function that computes the quantity of interest given a scenario.
            q: float | Quantile to compute (0 <= q <= 1).
        Returns:
            float | Quantile of the quantity of interest.
        """
        return np.nanquantile(self.map(f), q).item()
    
    def expectation(self, f: Callable[[Dict[str, np.ndarray]], float]) -> float:
        """
        Compute the expectation of a quantity.
        Args:
            f: Callable[[Dict[str, np.ndarray]], float] | Function that computes the quantity of interest given a scenario.
        Returns:
            float | Expectation of the quantity of interest.
        """
        return np.nanmean(self.map(f)).item()

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
        return self.scenarios[index]

    def __len__(self) -> int:
        """
        Return the number of scenarios.
        """
        return self.n_scenarios

    @classmethod
    def load_json(cls, f: Union[str, Path, TextIO]) -> "Forecast":
        """
        Load a forecast from a json file.
        """
        with open(f, "r", encoding="utf-8") if isinstance(f, (str, Path)) else nullcontext(f) as f:
            data = json.load(f)
        return cls(**data)
