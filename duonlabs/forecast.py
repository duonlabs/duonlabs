"""
duonlabs.forecast — v2 Forecast object.

Stores context + sampled scenarios as column-major numpy arrays keyed by
fully-qualified column names (e.g. "binance.spot.BTCUSDT.close"). Single-pair
and multi-pair share the same access pattern.

Copyright (c) 2025 Duon labs
"""

import json
import numpy as np

from typing import Any, Callable, Dict, List, Union
from pathlib import Path


def _derive_keys(columns: List[str]) -> List[str]:
    """Extract the ordered, deduplicated list of pair keys from a column list.

    A column name is `key.field` (e.g. `binance.spot.BTCUSDT.open`); the key is
    everything before the last dot. The `timestamp` column has no key.
    """
    keys: List[str] = []
    seen = set()
    for c in columns:
        if c == "timestamp":
            continue
        k = c.rsplit(".", 1)[0]
        if k not in seen:
            seen.add(k)
            keys.append(k)
    return keys


def _columns_to_arrays(columns: List[str], rows: List[List[Union[int, float]]]) -> Dict[str, np.ndarray]:
    """Transpose row-major steps into a column-name → 1-D numpy array dict."""
    if not rows:
        return {c: np.array([]) for c in columns}
    arr = np.asarray(rows, dtype=np.float64)
    return {c: arr[:, i] for i, c in enumerate(columns)}


class Forecast:
    """A sampled forecast over one or more pair keys.

    Attributes:
        columns: ["timestamp", "<key>.open", "<key>.high", ...] — wire-format column order.
        keys: Ordered list of pair keys derived from `columns`.
        context: column_name → np.ndarray (T_ctx,) — historical input.
        scenarios: column_name → np.ndarray (n_scenarios, n_steps) — sampled future.
        timestamps: np.ndarray (n_steps,) — generated step timestamps.
        context_timestamps: np.ndarray (T_ctx,) — historical timestamps.
        infos: dict — request metadata (model, seed, frequency, ...).
    """

    def __init__(
        self,
        columns: List[str],
        context_steps: List[List[Union[int, float]]],
        scenarios: List[List[List[Union[int, float]]]],
        infos: Dict[str, Any] = None,
    ):
        """
        Args:
            columns: Wire-format column names. `columns[0]` must be "timestamp".
            context_steps: (T_ctx, len(columns)) historical rows.
            scenarios: (n_scenarios, n_steps, len(columns)) sampled future rows.
            infos: Optional metadata dict.
        """
        if not columns or columns[0] != "timestamp":
            raise ValueError("columns[0] must be 'timestamp'")
        self.columns = list(columns)
        self.keys = _derive_keys(self.columns)
        self.infos = dict(infos) if infos else {}
        ## Context: column → (T_ctx,)
        ctx_arrays = _columns_to_arrays(self.columns, context_steps)
        self.context_timestamps = ctx_arrays["timestamp"].astype(np.int64)
        self.context = {c: ctx_arrays[c] for c in self.columns if c != "timestamp"}
        ## Scenarios: column → (n_scenarios, n_steps)
        self.n_scenarios = len(scenarios)
        if scenarios and scenarios[0]:
            self.n_steps = len(scenarios[0])
            ## Stack into (n_scenarios, n_steps, n_cols), then split column-wise
            arr = np.asarray(scenarios, dtype=np.float64)
            self.timestamps = arr[0, :, 0].astype(np.int64)
            self.scenarios: Dict[str, np.ndarray] = {
                c: arr[:, :, i] for i, c in enumerate(self.columns) if c != "timestamp"
            }
        else:
            self.n_steps = 0
            self.timestamps = np.array([], dtype=np.int64)
            self.scenarios = {c: np.zeros((self.n_scenarios, 0)) for c in self.columns if c != "timestamp"}

    def __getitem__(self, col_name: str) -> np.ndarray:
        """Return the scenarios array for `col_name`, shape (n_scenarios, n_steps)."""
        if col_name not in self.scenarios:
            raise KeyError(f"unknown column {col_name!r}; available: {list(self.scenarios.keys())}")
        return self.scenarios[col_name]

    def __len__(self) -> int:
        return self.n_scenarios

    def cutoff(self, col_name: str) -> float:
        """Last historical value for `col_name` (scalar)."""
        if col_name not in self.context:
            raise KeyError(f"unknown column {col_name!r}; available: {list(self.context.keys())}")
        return float(self.context[col_name][-1])

    def scenario(self, index: Union[int, slice]) -> Union["Forecast", Dict[str, np.ndarray]]:
        """Get a single scenario (int) as a {column: (n_steps,)} dict, or a sliced Forecast."""
        if isinstance(index, slice):
            shallow = self.__class__.__new__(self.__class__)
            shallow.__dict__.update(self.__dict__)
            shallow.scenarios = {c: v[index] for c, v in self.scenarios.items()}
            shallow.n_scenarios = next(iter(shallow.scenarios.values())).shape[0] if shallow.scenarios else 0
            return shallow
        return {c: v[index] for c, v in self.scenarios.items()}

    def map(self, f: Callable[[Dict[str, np.ndarray]], Any]) -> List[Any]:
        """Apply `f` to every scenario dict. Returns a list of length n_scenarios."""
        return [f(self.scenario(i)) for i in range(self.n_scenarios)]

    def _resolve(self, arg: Union[str, Callable]) -> Callable[[Dict[str, np.ndarray]], float]:
        """Turn a column name or callable into a per-scenario scalar function."""
        if isinstance(arg, str):
            if arg not in self.scenarios:
                raise KeyError(f"unknown column {arg!r}; available: {list(self.scenarios.keys())}")
            return lambda s, _arg=arg: s[_arg]
        return arg

    def probability(self, event: Callable[[Dict[str, np.ndarray]], bool]) -> float:
        """Empirical probability of `event` over scenarios."""
        return float(np.nanmean([float(event(s)) for s in (self.scenario(i) for i in range(self.n_scenarios))]))

    def expectation(self, quantity: Callable[[Dict[str, np.ndarray]], float]) -> float:
        """Empirical expectation of `quantity` over scenarios."""
        return float(np.nanmean(self.map(quantity)))

    def quantile(self, quantity: Callable[[Dict[str, np.ndarray]], float], q: float) -> float:
        """Empirical quantile `q` (0..1) of `quantity` over scenarios."""
        return float(np.nanquantile(self.map(quantity), q))

    def min(self, arg: Union[str, Callable]) -> float:
        """Minimum of `arg` across scenarios. `arg` can be a column name or a callable."""
        f = self._resolve(arg)
        return float(np.nanmin([np.nanmin(f(self.scenario(i))) for i in range(self.n_scenarios)]))

    def max(self, arg: Union[str, Callable]) -> float:
        """Maximum of `arg` across scenarios."""
        f = self._resolve(arg)
        return float(np.nanmax([np.nanmax(f(self.scenario(i))) for i in range(self.n_scenarios)]))

    def highest(self, arg: Union[str, Callable]) -> Dict[str, np.ndarray]:
        """Scenario achieving the highest value of `arg`."""
        f = self._resolve(arg)
        idx = int(np.argmax([np.nanmax(f(self.scenario(i))) for i in range(self.n_scenarios)]))
        return self.scenario(idx)

    def lowest(self, arg: Union[str, Callable]) -> Dict[str, np.ndarray]:
        """Scenario achieving the lowest value of `arg`."""
        f = self._resolve(arg)
        idx = int(np.argmin([np.nanmin(f(self.scenario(i))) for i in range(self.n_scenarios)]))
        return self.scenario(idx)

    def dump(self, path: Union[str, Path]) -> None:
        """Write the forecast to JSON at `path` (path-only — no file-handle support)."""
        ctx_rows = self._rows_from_arrays(self.context_timestamps, self.context)
        scen_rows = self._scenario_rows()
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "columns": self.columns,
                "context_steps": ctx_rows,
                "scenarios": scen_rows,
                "infos": self.infos,
            }, f, indent=2)

    def _rows_from_arrays(self, timestamps: np.ndarray, arrays: Dict[str, np.ndarray]) -> List[List[Union[int, float]]]:
        """Reassemble row-major steps from per-column arrays."""
        rows: List[List[Union[int, float]]] = []
        for i, t in enumerate(timestamps.tolist()):
            row: List[Union[int, float]] = [int(t)]
            for c in self.columns:
                if c == "timestamp":
                    continue
                row.append(float(arrays[c][i]))
            rows.append(row)
        return rows

    def _scenario_rows(self) -> List[List[List[Union[int, float]]]]:
        """Reassemble (n_scenarios, n_steps, n_cols) row-major scenarios for JSON dump."""
        out: List[List[List[Union[int, float]]]] = []
        ts_list = self.timestamps.tolist()
        for s in range(self.n_scenarios):
            scen: List[List[Union[int, float]]] = []
            for i, t in enumerate(ts_list):
                row: List[Union[int, float]] = [int(t)]
                for c in self.columns:
                    if c == "timestamp":
                        continue
                    row.append(float(self.scenarios[c][s, i]))
                scen.append(row)
            out.append(scen)
        return out

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "Forecast":
        """Load a forecast previously written with `dump`."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            columns=data["columns"],
            context_steps=data["context_steps"],
            scenarios=data["scenarios"],
            infos=data.get("infos", {}),
        )
