__version__ = "0.1.0"

# Let users know if they're missing any of our hard dependencies
hard_dependencies = ("pandas",)
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

from enda.backtesting import BackTesting  # noqa

# import some subclasses here so users can use for instance :
# 'enda.Contracts' without knowing the internal structure.
# Do not import classes that need a specific packages like "H2OModel".
from enda.contracts import Contracts  # noqa
from enda.feature_engineering.datetime_features import DatetimeFeature  # noqa
from enda.power_predictor import PowerPredictor  # noqa
from enda.power_stations import PowerStations  # noqa
from enda.timeseries import TimeSeries  # noqa
