"""
.. include:: ../README.md
"""

__version__ = "1.0.0"

# import some subclasses here so users can use for instance :
# 'enda.Contracts' without knowing the internal structure.
# Do not import classes that need a specific packages like "H2OModel".

from enda.feature_engineering.calendar import Calendar, Holidays  # noqa
from enda.feature_engineering.datetime_features import DatetimeFeature  # noqa

from enda.backtesting import BackTesting  # noqa
from enda.contracts import Contracts  # noqa
from enda.power_predictor import PowerPredictor  # noqa
from enda.power_stations import PowerStations  # noqa
from enda.scoring import Scoring  # noqa

from enda.tools.portfolio_tools import PortfolioTools  # noqa
from enda.tools.resample import Resample  # noqa
from enda.tools.timeseries import TimeSeries  # noqa
from enda.tools.timezone_utils import TimezoneUtils  # noqa
