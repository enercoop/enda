###
# This is an example of a program that can perform training and forecasts for electricity load.
#
# When the placeholder functions are implemented, you should be able to run the following commands
# (in the correct python virtualenv) :
#
# 1. Collect the full historic dataset, save it locally, and then test training with just 1000 data points:
# python -m {abc.xyz}.example_c --task learn --learn_data_limit 1000
# 2. Collect the forecast input data, save it locally, then test the "predict" mode :
# python -m {abc.xyz}.example_c --task predict
# 3. Train the algorithm with the full dataset saved locally at step 1:
# python -m {abc.xyz}.example_c --task learn --read_local_data
# 4. Predict load using the forecast input data saved at step 2, and send the prediction to CofyCloud :
# python -m {abc.xyz}.example_c --task predict --read_local_data --send_prediction_to_cofycloud


import os
import warnings
import pickle
import logging
import argparse
import time
import pandas as pd
from datetime import date, datetime
from dateutil.relativedelta import relativedelta

import h2o
from h2o.estimators import H2OGeneralizedLinearEstimator
from h2o.estimators import H2OXGBoostEstimator
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.estimators import H2ODeepLearningEstimator
from h2o.estimators import H2ORandomForestEstimator

import enda
from enda.estimators import (
    EndaStackingEstimator,
    EndaEstimatorWithFallback,
    EndaEstimator,
)
from enda.ml_backends.h2o_estimator import EndaH2OEstimator
from enda.timezone_utils import TimezoneUtils
from enda.feature_engineering.calendar import Calendar


class LoadForecast(EndaEstimator):
    """
    A class to define (strictly) the algorithm to use in the forecast.
    Here with just 1 group containing all customers
    """

    TSO_FEATURE = "rte_forecasts_mw"
    TARGET = "power_mw"
    TZINFO_STR = "Europe/Paris"
    INDEX_NAME = "time"
    FREQ_STR = "30min"  # could also be pd.TimeDelta('30min')

    def __init__(self):
        self.estimator = self.get_untrained_estimator()

    def train(self, df: pd.DataFrame, target_col: str):
        self.check_df(train_or_test="train", df=df)
        assert target_col == LoadForecast.TARGET
        self.estimator.train(df, target_col)

    def predict(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        self.check_df(train_or_test="test", df=df)
        assert target_col == LoadForecast.TARGET
        prediction = self.estimator.predict(df, target_col)
        prediction[LoadForecast.TARGET] = prediction[LoadForecast.TARGET].round(
            decimals=3
        )
        return prediction

    @classmethod
    def expected_columns_in_order(cls, train_or_test: str):
        columns = [
            # size of portfolio
            "contracts_count",  # total number of active contracts
            "kva",  # sum of the subscribed power of all customers
            # weather
            "t_weighted",
            "t_smooth",
            # calendar & holiday features
            "lockdown",
            "public_holiday",
            "nb_school_areas_off",
            "extra_long_weekend",
            # tso forecast
            LoadForecast.TSO_FEATURE,
            # time and "cyclic" time features
            "minuteofday",
            "dayofweek",
            "month",  # use month for the yearly period feature
            "minuteofday_cos",
            "minuteofday_sin",
            "dayofweek_cos",
            "dayofweek_sin",
            "dayofyear_cos",
            "dayofyear_sin",  # use dayofyear for the yearly "cyclic" feature
        ]

        if train_or_test == "train":
            columns = [LoadForecast.TARGET] + columns
        elif train_or_test == "test":
            pass
        else:
            raise ValueError(
                "unexpected argument train_or_test: {}".format(train_or_test)
            )

        return columns

    @classmethod
    def check_df(cls, train_or_test: str, df: pd.DataFrame):
        """train_or_test must be 'train' or 'test'"""

        # check the index first
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "index must be a pandas.DatetimeIndex, but found '{}'".format(
                    df.index.dtype
                )
            )
        if df.index.name != LoadForecast.INDEX_NAME:
            raise ValueError(
                "index name must be '{}', but found '{}'".format(
                    LoadForecast.INDEX_NAME, df.index.name
                )
            )
        if str(df.index.tz) != LoadForecast.TZINFO_STR:
            raise ValueError(
                "index must have timezone '{}', but found '{}'".format(
                    LoadForecast.TZINFO_STR, df.index.tz
                )
            )

        # check the columns, depends if the given df is a train or test dataframe (target present or not)
        expected_columns = cls.expected_columns_in_order(train_or_test)

        # check that df has the expected columns, in the expected order
        if list(df.columns) != expected_columns:
            raise ValueError(
                "Expected columns: {}, but given df with columns: {}".format(
                    expected_columns, df.columns
                )
            )

        # check that all features except rte_feature have no NaN value
        df_without_rte = df.drop(columns=[LoadForecast.TSO_FEATURE])
        count_na = df_without_rte.isnull().sum(axis=0).sum()
        # if missing data in any column, raise an Error
        if count_na > 0:
            df_nan = df_without_rte[df_without_rte.isna().any(axis=1)]
            raise ValueError(
                "Trying to predict on a dataframe with {} NaN values\n{}".format(
                    count_na, df_nan
                )
            )

    @classmethod
    def __get_untrained_base_estimators(cls):
        gbm = EndaH2OEstimator(
            H2OXGBoostEstimator(
                **{
                    "ntrees": 500,
                    "max_depth": 5,
                    "sample_rate": 0.8,
                    "min_rows": 10,
                    "seed": 1234,
                }
            )
        )

        xgboost = EndaH2OEstimator(
            H2OGradientBoostingEstimator(
                **{
                    "ntrees": 500,
                    "max_depth": 5,
                    "sample_rate": 0.5,
                    "min_rows": 5,
                    "seed": 1234,
                }
            )
        )

        random_forest = EndaH2OEstimator(
            H2ORandomForestEstimator(
                **{
                    "ntrees": 300,
                    "max_depth": 15,
                    "sample_rate": 0.8,
                    "min_rows": 10,
                    "nbins": 52,
                    "mtries": 3,
                    "seed": 1234,
                }
            )
        )

        nn = EndaH2OEstimator(
            H2ODeepLearningEstimator(
                **{
                    "activation": "Rectifier",
                    "hidden": [48, 48, 48],
                    "distribution": "gaussian",
                    "epochs": 40,
                    "seed": 1234,
                }
            )
        )

        return {
            "gbm": gbm,
            "xgboost": xgboost,
            "random_forest": random_forest,
            "nn": nn,
        }

    @classmethod
    def get_untrained_estimator(cls) -> EndaEstimator:
        """note: for GLM stacking(s), could try intercept=False, non_negative=True"""

        base_estimators_with_tso = cls.__get_untrained_base_estimators()
        base_estimators_without_tso = cls.__get_untrained_base_estimators()

        stacking_with_tso = EndaStackingEstimator(
            base_estimators=base_estimators_with_tso,
            final_estimator=EndaH2OEstimator(H2OGeneralizedLinearEstimator()),
            base_stack_split_pct=0.0167,  # 1 month out of 5 years
        )

        stacking_without_tso = EndaStackingEstimator(
            base_estimators=base_estimators_without_tso,
            final_estimator=EndaH2OEstimator(H2OGeneralizedLinearEstimator()),
            base_stack_split_pct=0.0167,  # 1 month out of 5 years
        )

        with_tso_fallback = EndaEstimatorWithFallback(
            resilient_column=LoadForecast.TSO_FEATURE,
            estimator_with=stacking_with_tso,
            estimator_without=stacking_without_tso,
        )

        return with_tso_fallback


class WeatherData:
    @classmethod
    def get_historic_weather_forecast(cls):
        """
        :returns: a time series with 2 columns : t_weigthed and t_smooth
        """

        raise NotImplementedError("")

    @classmethod
    def get_future_weather_forecast(cls, start_forecast_date, nb_days):
        """
        :returns: a time series with 2 columns : t_weigthed and t_smooth
        """

        raise NotImplementedError("")


class TsoData:
    @classmethod
    def get_historic_tso_forecast(cls):
        """
        :returns: a time series with 1 column : LoadForecast.TSO_FEATURE
        """

        raise NotImplementedError("")

    @classmethod
    def get_future_tso_forecast(cls, start_forecast_date, nb_days):
        """
        :returns: a time series with 1 column : LoadForecast.TSO_FEATURE
        """

        raise NotImplementedError("")


class PortfolioData:
    @classmethod
    def get_contracts_from_erp(cls):
        """A connector to your ERP to get the list of contracts"""

        return NotImplementedError()

    @classmethod
    def get_portfolio(cls):
        contracts = cls.get_contracts_from_erp()

        # maybe do some additional data manipulations, for instance if ERP end_date is inclusive :
        contracts["date_end_exclusive"] = contracts["date_end"] + pd.Timedelta("1 day")
        contracts.drop(columns=["date_end"])

        # add the column to count the number of active contracts
        contracts["contracts_count"] = 1

        portfolio_by_day = enda.Contracts.compute_portfolio_by_day(
            contracts,
            columns_to_sum=["contracts_count", "kva"],
            date_start_col="date_start",
            date_end_exclusive_col="date_end_exclusive",
        )

        portfolio = enda.TimeSeries.interpolate_daily_to_sub_daily_data(
            portfolio_by_day, freq=LoadForecast.FREQ_STR, tz=LoadForecast.TZINFO_STR
        )

        return portfolio

    @classmethod
    def get_forecast_portfolio(cls, portfolio, start_forecast_date, nb_days):
        """Example of portfolio forecast using a linear method over the last 90 days"""

        end_forecast_date_exclusive = TimezoneUtils.add_interval_to_day_dt(
            day_dt=start_forecast_date, interval=relativedelta(days=nb_days)
        )

        # We will forecast the portfolio using a linear method

        # only use recent portfolio trend to forecast the next few days
        min_time = TimezoneUtils.add_interval_to_day_dt(
            day_dt=start_forecast_date, interval=relativedelta(days=-90)
        )
        max_time = start_forecast_date  # don't use future events
        pf_last_90days = portfolio[
            (min_time <= portfolio.index) & (portfolio.index < max_time)
        ]

        forecast_portfolio = enda.Contracts.forecast_portfolio_linear(
            portfolio_df=pf_last_90days,
            start_forecast_date=start_forecast_date,
            end_forecast_date_exclusive=end_forecast_date_exclusive,
            freq=LoadForecast.FREQ_STR,
            tzinfo=LoadForecast.TZINFO_STR,
        )
        return forecast_portfolio


class HistoricLoad:
    @classmethod
    def get_historic_load(cls):
        """
        :returns: a time series with 1 column : LoadForecast.TARGET
        """

        raise NotImplementedError("")


class LoadForecastData:
    """
    A class to fetch data : either historic dataset or the forecast input data.

    Also provides helpers to test the algorithm more easily:
        - save data locally
        - return a subset of data
    """

    PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOAD_FORECAST_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "load_forecast")
    HISTORIC_PATH = os.path.join(LOAD_FORECAST_DATA_DIR, "historic.csv")
    FORECAST_INPUT_PATH = os.path.join(LOAD_FORECAST_DATA_DIR, "forecast_input.csv")
    MODEL_FILE_PATH = os.path.join(LOAD_FORECAST_DATA_DIR, "trained_model.pickle")

    # define logging somewhere
    LOGGER = logging.getLogger(PROJECT_ROOT_DIR)
    logger_handler = logging.StreamHandler()
    logger_handler.setFormatter(
        logging.Formatter(
            fmt="[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    LOGGER.addHandler(logger_handler)
    LOGGER.propagate = False

    logging.debug("invisible magic")  # some trick for logging to work properly
    LOGGER.setLevel(level=logging.INFO)

    @staticmethod
    def featurize_datetime_index(df):
        # put datetime features to capture the data frequencies: daily, weekly and yearly periods.
        df = enda.DatetimeFeature.split_datetime(
            df, split_list=["minuteofday", "dayofweek", "month"]
        )
        df = enda.DatetimeFeature.encode_cyclic_datetime_index(
            df, split_list=["minuteofday", "dayofweek", "dayofyear"]
        )

        return df

    @classmethod
    def log_dataframe_info(cls, df, title, logger):
        # check missing data in the time-series (based on the time index only)
        (
            freq,
            missing_periods,
            extra_points,
        ) = enda.TimeSeries.find_missing_and_extra_periods(
            dti=df.index, expected_freq=LoadForecast.FREQ_STR
        )
        logger.info(
            "{}:\n"
            "Shape: {}\n"
            "Columns: {}\n"
            "Freq: {}\n"
            "Date start : {}\n"
            "Date end : {} \n"
            "Missing periods:\n{}\n"
            "Extra points:\n{}".format(
                title,
                df.shape,
                df.columns,
                freq,
                df.index.min(),
                df.index.max(),
                missing_periods,
                extra_points,
            )
        )

    @classmethod
    def compute_historic_df(cls, logger):
        # retrieve all data
        portfolio = PortfolioData.get_portfolio()
        weather = WeatherData.get_historic_weather_forecast()
        load = HistoricLoad.get_historic_load()
        tso = TsoData.get_historic_tso_forecast()

        # features about national holidays and school holidays (French holidays here)
        special_days = Calendar().get_french_special_days()

        # concat data in one dataframe, 'inner' -> only keep rows where all time-series have a row (could have NaNs)
        historic = pd.concat([load, portfolio, weather, special_days, tso], 1, "inner")

        # check for NaN values
        count_na = historic.isnull().sum(axis=0).sum()
        if count_na > 0:
            raise ValueError("Historic has NaN values")

        historic = cls.featurize_datetime_index(historic)

        cls.log_dataframe_info(
            historic,
            title="historic dataframe to train short-term consumption forecast",
            logger=logger,
        )

        return historic

    @classmethod
    def compute_forecast_input_df(cls, logger, run_date: [pd.Timestamp, None] = None):
        """if run_date is None, it is today"""

        if run_date is None:
            today = date.today()
            today = datetime(year=today.year, month=today.month, day=today.day)
            today = pd.to_datetime(today).tz_localize(LoadForecast.TZINFO_STR)
            run_date = today

        if run_date.tzinfo is None or str(run_date.tzinfo) != LoadForecast.TZINFO_STR:
            raise ValueError(
                "run_date must be at timezone {} but given '{}'".format(
                    LoadForecast.TZINFO_STR, run_date.tzinfo
                )
            )

        tomorrow = TimezoneUtils.add_interval_to_day_dt(
            day_dt=run_date, interval=relativedelta(days=1)
        )

        historic_portfolio = PortfolioData.get_portfolio()
        forecast_portfolio = PortfolioData.get_forecast_portfolio(
            portfolio=historic_portfolio, start_forecast_date=tomorrow, nb_days=11
        )

        special_days = Calendar().get_french_special_days()
        forecast_tso = TsoData.get_future_tso_forecast(
            start_forecast_date=tomorrow, nb_days=7
        )
        forecast_weather = WeatherData.get_future_weather_forecast(
            start_forecast_date=tomorrow, nb_days=11
        )

        df = pd.concat([forecast_portfolio, forecast_weather, special_days], 1, "inner")
        # keep lines where TSO_FEATURE is not available (left join)
        df = df.merge(forecast_tso, how="left", left_index=True, right_index=True)

        logger.info(
            "Profiled future dataframe components (run_date = {}) :".format(run_date)
        )
        logger.info(
            "--- Portfolio : start = {}, end = {}".format(
                forecast_portfolio.index.min(), forecast_portfolio.index.max()
            )
        )
        logger.info(
            "--- Weather   : start = {}, end = {}".format(
                forecast_weather.index.min(), forecast_weather.index.max()
            )
        )
        logger.info(
            "--- Calendar  : start = {}, end = {}".format(
                special_days.index.min(), special_days.index.max()
            )
        )
        logger.info(
            "--- RTE       : start = {}, end = {}".format(
                forecast_tso.index.min(), forecast_tso.index.max()
            )
        )
        logger.info(
            "--- Result    : start = {}, end = {}".format(
                df.index.min(), df.index.max()
            )
        )

        df = cls.featurize_datetime_index(df)

        cls.log_dataframe_info(
            df, title="Input dataframe to predict short-term consumption", logger=logger
        )

        return df

    @staticmethod
    def read_local_dataframe(file_path):
        df = pd.read_csv(file_path, parse_dates=False, index_col=False)
        df[LoadForecast.INDEX_NAME] = pd.to_datetime(df[LoadForecast.INDEX_NAME])
        df[LoadForecast.INDEX_NAME] = enda.TimeSeries.align_timezone(
            df[LoadForecast.INDEX_NAME], tzinfo=LoadForecast.TZINFO_STR
        )
        df.set_index(LoadForecast.INDEX_NAME, drop=True, inplace=True)
        return df

    @classmethod
    def get_df(
        cls, dataset, limit_train_data, read_local_dataset, save_dataset_locally, logger
    ):
        """simply a wrapper around 'compute_historic_df' and 'compute_forecast_df' to manipule
        the train and test sets more easily

            - caching of the full set
            - limit_train_data > 0 -> to train/test on a small sample and check it works
        """

        if dataset == "historic":
            file_path = cls.HISTORIC_PATH
        elif dataset == "forecast_input":
            file_path = cls.FORECAST_INPUT_PATH
        else:
            raise ValueError(
                "Unexpected dataset '{}'. Expecting 'historic' or 'forecast_input'.".format(
                    dataset
                )
            )

        if not os.path.exists(cls.LOAD_FORECAST_DATA_DIR):
            os.makedirs(cls.LOAD_FORECAST_DATA_DIR)
            logger.info("Directories created : {}".format(cls.LOAD_FORECAST_DATA_DIR))

        if read_local_dataset:
            if not os.path.exists(file_path):
                raise FileNotFoundError(file_path)
            df = cls.read_local_dataframe(file_path)
            logger.info("{} dataset read locally from {}".format(dataset, file_path))
        else:
            # the main action of this function
            if dataset == "historic":
                df = cls.compute_historic_df()
            elif dataset == "forecast_input":
                df = cls.compute_forecast_input_df()
            else:
                raise ValueError(
                    "Unexpected dataset '{}'. Expecting 'historic' or 'forecast_input'.".format(
                        dataset
                    )
                )

        if not read_local_dataset and save_dataset_locally:
            # save training dataset for analysis or caching
            df.to_csv(file_path)  # (sep=',' and decimal='.') by default
            logger.info("{} dataset saved : {}".format(dataset, file_path))

        if limit_train_data > 0:
            # keep only a subset of the data for training
            limit = min(df.shape[0], limit_train_data)
            df = df[-limit:]

        logger.info("Returning {} set with shape : {}".format(dataset, df.shape))
        return df

    @classmethod
    def get_historic_df(
        cls, limit_train_data=0, read_local_dataset=False, save_dataset_locally=True
    ):
        return cls.get_df(
            "historic", limit_train_data, read_local_dataset, save_dataset_locally
        )

    @classmethod
    def get_forecast_input_df(
        cls, limit_train_data=0, read_local_dataset=False, save_dataset_locally=True
    ):
        return cls.get_df(
            "forecast_input", limit_train_data, read_local_dataset, save_dataset_locally
        )

    @classmethod
    def send_prediction_to_cofycloud(cls, predict_df):
        raise NotImplementedError("")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--task",
        action="store",
        required=True,
        choices=["learn", "predict"],
        help="task is either learn models or predict consumption",
    )

    # parser.add_argument("--verbose", action="store_true", default=False,
    #                    help="if set, will log out a lot of information on training and prediction.")

    parser.add_argument(
        "--read_local_data",
        action="store_true",
        default=False,
        help="use only for testing purposes: if set, will read the latest local file "
        "with historic or forecast_input data instead of fetching all the data from sources.",
    )

    parser.add_argument(
        "--learn_data_limit",
        action="store",
        type=int,
        default=0,
        help="use only for testing purposes: if set, will limit the train set for faster training"
        "(only applies to 'learn' task). '0' (default) will use the full training set. ",
    )

    parser.add_argument(
        "--send_prediction_to_cofycloud",
        action="store_true",
        default=False,
        help="if set, will send the prediction to CofyCloud (only applies to 'predict' task)",
    )

    args = parser.parse_args()

    # check if arguments are ok
    if args.send_prediction_to_cofycloud:
        if args.task != "predict":
            raise ValueError("Send predictions only in predict task.")

    if args.learn_data_limit > 0 and args.task != "learn":
        raise ValueError("use '--learn_data_limit' only in learn task.")

    # boot up local h2o server
    warnings.filterwarnings("ignore")
    h2o.init(nthreads=-1)
    h2o.no_progress()

    logger = LoadForecastData.LOGGER

    # now run the program
    if args.task == "learn":
        historic = LoadForecastData.get_historic_df(
            limit_train_data=args.learn_data_limit,
            read_local_dataset=args.read_local_data,
        )
        algo = LoadForecast()
        algo.train(historic, target_col=LoadForecast.TARGET)
        with open(LoadForecastData.MODEL_FILE_PATH, "wb") as f:
            pickle.dump(algo, f)
        logger.info(
            "Saved trained algo at : {}".format(LoadForecastData.MODEL_FILE_PATH)
        )

    if args.task == "predict":
        with open(LoadForecastData.MODEL_FILE_PATH, "rb") as f:
            algo = pickle.load(f)  # load algo model from disk

        forecast_input = LoadForecastData.get_forecast_input_df(
            read_local_dataset=args.read_local_data
        )
        predict_df = algo.predict(forecast_input, target_col=LoadForecast.TARGET)
        logger.info("Prediction:\n{}".format(predict_df))

        if args.send_prediction_to_cofycloud:
            LoadForecastData.send_prediction_to_cofycloud(predict_df)

    h2o.cluster().shutdown()
    logger.info("waiting for h2o to finish shutting down")
    time.sleep(8)


if __name__ == "__main__":
    main()
