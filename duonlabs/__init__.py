"""
duonlabs — Python SDK for the Voyons forecasting API.

The top-level package exposes the v2 surface (multi-pair, fully-qualified column
names). The v1 surface is preserved verbatim under `duonlabs.legacy`.

Copyright (c) 2025 Duon labs
"""

from . import legacy, utils
from .client import DuonLabs
from .forecast import Forecast

__all__ = ["DuonLabs", "Forecast", "utils", "legacy"]
