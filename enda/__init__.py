""" init file for enda package"""

__version__ = "1.0.0"

# Let users know if they're missing any of our hard dependencies
hard_dependencies = ("pandas","h2o","sklearn")
# note that we also need python-dateutil and pytz but pandas already depends on them, so importing pandas is enough
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError:
        missing_dependencies.append(dependency)

if len(missing_dependencies) > 0:
    raise ImportError(
        "Unable to import required dependencies:\n"
        + "\n".join(
            missing_dependencies,
        )
    )
del hard_dependencies, dependency, missing_dependencies

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
