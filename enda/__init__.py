"""
.. include:: ../README.md
"""

# import some subclasses here so users can use for instance :
# 'enda.Contracts' without knowing the internal structure.

from enda.feature_engineering.calendar import Calendar, Holidays  # noqa
from enda.feature_engineering.datetime_features import DatetimeFeature  # noqa

from enda.ml_backends.h2o_estimator import EndaH2OEstimator
from enda.ml_backends.sklearn_estimator import EndaSklearnEstimator

from enda.backtesting import BackTesting  # noqa
from enda.contracts import Contracts  # noqa
from enda.estimators import (
    EndaEstimator,
    EndaNormalizedEstimator,
    EndaEstimatorRecopy,
    EndaEstimatorWithFallback,
    EndaStackingEstimator)  # noqa
from enda.power_predictor import PowerPredictor  # noqa
from enda.power_stations import PowerStations  # noqa
from enda.scoring import Scoring  # noqa

from enda.tools.portfolio_tools import PortfolioTools  # noqa
from enda.tools.resample import Resample  # noqa
from enda.tools.timeseries import TimeSeries  # noqa
from enda.tools.timezone_utils import TimezoneUtils  # noqa
