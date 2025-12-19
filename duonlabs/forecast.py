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
        self.cutoff_close = self.context["close"][-1].item()

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
    
    def min(self, f: Union[str, Callable[[Dict[str, np.ndarray]], float]]) -> float:
        """
        Return the minimum value of a quantity across all scenarios.
        Args:
            f: One of:
                str | Name of the channel to get the minimum value from
                Callable[[Dict[str, np.ndarray]], float] | Function that computes the quantity of interest given a scenario.
        Returns:
            float | Minimum value of the quantity across all scenarios.
        """
        return np.nanmin(self.map((lambda s: s[f].min()) if isinstance(f, str) else f)).item()
    
    def max(self, f: Union[str, Callable[[Dict[str, np.ndarray]], float]]) -> float:
        """
        Return the maximum value of a quantity across all scenarios.
        Args:
            f: One of:
                str | Name of the channel to get the maximum value from
                Callable[[Dict[str, np.ndarray]], float] | Function that computes the quantity of interest given a scenario.
        Returns:
            float | Maximum value of the quantity across all scenarios.
        """
        return np.nanmax(self.map((lambda s: s[f].max()) if isinstance(f, str) else f)).item()
    
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

    def compute_returns(self, tp_levels: np.ndarray, sl_levels: np.ndarray, is_in: bool = False, fees: float = 0.001) -> np.ndarray:
        """
        Compute the results of the trade idea based on the take profit and stop loss levels.
        Args:
            tp_levels: np.ndarray | Take profit levels, shape (...)
            sl_levels: np.ndarray | Stop loss levels, shape (...)
        Returns:
            np.ndarray | Results of the trade idea, shape (n_scenarios, ..., 1)
        """
        tp_levels, sl_levels = tp_levels[..., None], sl_levels[..., None]  # [..., 1]
        tp_trigger_mask = np.stack(self.map(lambda s: s["high"] >= tp_levels))  # [n_scenarios, ..., horizon]
        sl_trigger_mask = np.stack(self.map(lambda s: s["low"] <= sl_levels))  # [n_scenarios, ..., horizon]
        tp_trigger_id, sl_trigger_id = tp_trigger_mask.argmax(-1), sl_trigger_mask.argmax(-1)  # [n_scenarios, ...]
        sl_triggered, tp_triggered = sl_trigger_mask.any(-1), tp_trigger_mask.any(-1)  # [n_scenarios, ...]
        tp_triggered_first = tp_triggered & (~sl_triggered | (tp_trigger_id < sl_trigger_id))  # [n_scenarios, ...]
        sl_triggered_first = sl_triggered & (~tp_triggered | (sl_trigger_id <= tp_trigger_id))  # [n_scenarios, ...]
        returns = np.broadcast_to(np.stack(self.map(lambda s: s["close"][-1]))[(...,) + (None,)*(len(tp_levels.shape) - 1)], tp_triggered_first.shape).copy()  # [n_scenarios, ..., 1]
        returns[tp_triggered_first] = np.broadcast_to(tp_levels[None, ..., 0], tp_triggered_first.shape)[tp_triggered_first]
        returns[sl_triggered_first] = np.broadcast_to(sl_levels[None, ..., 0], sl_triggered_first.shape)[sl_triggered_first]
        return (returns * (1 - fees) - self.cutoff_close * (1.0 + (0.0 if is_in else fees))) / self.cutoff_close

    def evaluate_trade_idea(self, tp_levels: Union[float, np.ndarray], sl_levels: Union[float, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Evaluate a trade idea based on the take profit and stop loss levels.
        Args:
            tp_levels: Union[float, np.ndarray] | Take profit levels, can be a single value or an array of shape (n_steps, 1)
            sl_levels: Union[float, np.ndarray] | Stop loss levels, can be a single value or an array of shape (n_steps, 1)
        Returns:
            Dict[str, np.ndarray] | Dictionary with the keys:
                - expectation: np.ndarray | Expected return of the trade idea, shape (n_scenarios, n_steps, 1)
                - std: np.ndarray | Standard deviation of the return, shape (n_scenarios, n_steps, 1)
                - p_tp: np.ndarray | Probability of hitting the take profit, shape (n_scenarios, n_steps, 1)
                - p_sl: np.ndarray | Probability of hitting the stop loss, shape (n_scenarios, n_steps, 1)
        """
        if isinstance(tp_levels, (int, float)):
            tp_levels = np.array([tp_levels], dtype=np.float32)
        if isinstance(sl_levels, (int, float)):
            sl_levels = np.array([sl_levels], dtype=np.float32)
        return self.compute_returns(tp_levels, sl_levels)[..., 0]

    def __getitem__(self, index: int) -> Union["Forecast", Dict[str, np.ndarray]]:
        """
        Return the scenario at the given index.
        Args:
            index: int | Index of the scenario to return.
        Returns:
            Dict[str, np.ndarray] | Scenario with the keys "timestamp", "open", "high", "low", "close", "volume".
        """
        if isinstance(index, slice):
            shallow_copy = self.__class__.__new__(self.__class__)
            shallow_copy.__dict__.update(self.__dict__)
            shallow_copy.scenarios = self.scenarios[index]
            shallow_copy.n_scenarios = len(shallow_copy.scenarios)
            return shallow_copy
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
