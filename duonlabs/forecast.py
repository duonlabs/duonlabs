"""
duonlabs.forecast submodule.

Copyright (c) 2025 Duon labs
"""

import json
import numpy as np

from typing import Any, Callable, Dict, List, TextIO, Tuple, Union
from contextlib import nullcontext


class Forecast:
    def __init__(self, timestamps: np.ndarray, samples: Dict[str, np.ndarray], scenarios: Dict[str, np.ndarray], densities: Dict[str, np.ndarray]):
        self.timestamps, self.samples, self.scenarios, self.densities = timestamps, samples, scenarios, densities

    def dump(self, f: Union[str, TextIO]):
        with open(f, "w", encoding="utf-8") if isinstance(f, str) else nullcontext(f) as f:
            json.dump({
                "timestamps": self.timestamps.tolist(),
                "samples": {k: v.tolist() for k, v in self.samples.items()},
                "scenarios": {k: v.tolist() for k, v in self.scenarios.items()},
            })

    def density_buckets(self, channel: str, timestep: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.densities["icdfs"][timestep, list(self.samples.keys()).index(channel)], np.diff(self.densities["qs"], prepend=0.0, append=1.0)

    def map(self, f: Callable[[np.ndarray, Dict[str, np.ndarray]], Any]) -> List[Any]:
        return list(map(lambda i: f(self[i]), range(len(self))))

    def probability(self, event: Callable[[np.ndarray, Dict[str, np.ndarray]], bool]) -> float:
        return np.nanmean(self.map(event)).item()

    def expectation(self, f: Callable[[np.ndarray, Dict[str, np.ndarray]], float]) -> float:
        return np.nanmean(self.map(f)).item()

    def highest(self, getter: Union[str, Callable[[np.ndarray, Dict[str, np.ndarray]], float]]) -> Dict[str, np.ndarray]:
        return self[np.argmax(self.map((lambda s: s[getter].max()) if isinstance(getter, str) else getter))]

    def lowest(self, getter: Union[str, Callable[[np.ndarray, Dict[str, np.ndarray]], float]]) -> Dict[str, np.ndarray]:
        return self[np.argmin(self.map((lambda s: s[getter].min()) if isinstance(getter, str) else getter))]

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        return {"timestamp": self.scenarios["timestamps"]} | {k: self.scenarios["samples"][:, index, i] for i, k in enumerate(self.samples.keys())}

    def __len__(self) -> int:
        return self.scenarios["samples"].shape[1]

    @classmethod
    def load_json(cls, f: Union[str, TextIO]) -> "Forecast":
        with open(f, "r", encoding="utf-8") if isinstance(f, str) else nullcontext(f) as f:
            data = json.load(f)
        return cls(
            timestamps=np.array(data["timestamps"]),
            samples={k: np.array(v) for k, v in data["samples"].items()},
            scenarios={k: np.array(v) for k, v in data["scenarios"].items()},
        )
