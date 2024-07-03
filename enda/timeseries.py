"""
This module is deprecated and has been moved in tools.timeseries
"""

import warnings
from enda.tools.timeseries import TimeSeries

# Optionally, you can add a deprecation warning
warnings.warn("Module 'timeseries' is deprecated. Please use 'tools.timeseries' instead.",
              DeprecationWarning,
              stacklevel=2)
