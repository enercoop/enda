# Let users know if they're missing any of our hard dependencies
hard_dependencies = ("pandas",)
# note that we also need python-dateutil and pytz but pandas already depends on them, so importing pandas is enough
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(dependency)

if len(missing_dependencies) > 0:
    raise ImportError(
        "Unable to import required dependencies:\n" + "\n".join(missing_dependencies)
    )
del hard_dependencies, dependency, missing_dependencies

# import some subclasses here so users can use for instance :
# 'enda.Contracts' without knowing the internal structure.
# Do not import classes that need a specific packages like "H2OModel".
from enda.contracts import (Contracts)
from enda.feature_engineering.datetime_features import (DatetimeFeature)
from enda.timeseries import (TimeSeries)
from enda.backtesting import (BackTesting)
