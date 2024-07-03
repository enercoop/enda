"""
This module is deprecated and has been moved in tools.timezone_utils
"""

import warnings
from enda.tools.timezone_utils import TimezoneUtils

warnings.warn("Module 'timezone_utils' is deprecated. Please use 'tools.timezone_utils' instead.",
              DeprecationWarning,
              stacklevel=2)
