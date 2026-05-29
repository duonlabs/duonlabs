"""
duonlabs.legacy — frozen v1 SDK.

The v1 surface is preserved verbatim here so users on legacy models can keep
using the old client. No code is shared with the v2 SDK at duonlabs/.

Copyright (c) 2025 Duon labs
"""

from .client import DuonLabs
from .forecast import Forecast
from .utils import ListofListsofNumbers

__all__ = ["DuonLabs", "Forecast", "ListofListsofNumbers"]
