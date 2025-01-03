"""
duonlabs.api submodule.

Copyright (c) 2025 Duon labs
"""

import requests
import numpy as np

from typing import Dict, List, Optional, Union

from .forecast import Forecast


class DuonLabs:
    default_base_url: str = "https://api.duonlabs.com/v1/"
    headers: Dict[str, str] = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    def __init__(self, token: str, base_url: Optional[str] = None):
        self.headers["Authorization"] = f"Token {token}"
        self.base_url = base_url or self.default_base_url

    def forecast(
        self,
        timestamps: Union[List[float], np.ndarray],
        samples: Dict[str, Union[List[float], np.ndarray]],
        max_steps: int = 15,
        n_scenarios: int = 32,
    ) -> Forecast:
        response = requests.post(
            self.base_url + "scenarios/generation",
            headers=self.headers,
            json={
                "inputs": {
                    "timestamps": timestamps.astype(np.float64).tolist() if isinstance(timestamps, np.ndarray) else timestamps,
                    "samples": np.stack(tuple(samples.values()), -1).tolist(),
                },
                "max_steps": max_steps,
                "n_scenarios": n_scenarios,
            },
            timeout=180,
        )
        response.raise_for_status()
        response = response.json()

        return Forecast(
            timestamps=timestamps,
            samples=samples,
            scenarios={
                "timestamps": np.array(response["scenarii_timestamps"]),
                "samples": np.array(response["scenarii_samples"]),
                "log_probs": np.array(response["scenarii_log_probs"]),
            },
            densities={
                "qs": np.array(response["density"]["qs"]),
                "icdfs": np.array(response["density"]["icdfs"]),
            },
        )
