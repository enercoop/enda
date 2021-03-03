# Let users know if they're missing any of our hard dependencies
hard_dependencies = ("numpy", "pytz", "dateutil", "pandas", "h2o", "statsmodels")
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

from enda.contracts import (
    Contracts,
)

from enda.timeseries import (
    TimeSeries,
)
