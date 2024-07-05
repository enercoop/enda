"""This module contains methods useful to handle contracts data"""

import pandas as pd
from pandas.api.types import is_string_dtype, is_datetime64_dtype
from dateutil.relativedelta import relativedelta
import numpy as np
import pytz

from sklearn.linear_model import LinearRegression

from enda.ml_backends.sklearn_estimator import EndaSklearnEstimator
import enda.tools.decorators
from enda.tools.portfolio_tools import PortfolioTools
from enda.tools.resample import Resample
from enda.tools.timeseries import TimeSeries
from enda.tools.timezone_utils import TimezoneUtils


class Contracts:
    """
    A class to help handle contracts data.

    contracts : a dataframe with a contract or subcontract on each row.
        A contract row has fixed-values between two dates. In some systems, some values of a contract can change over
        time, for instance the "subscribed power" of an electricity consumption contract.
        Different ERPs can handle it in different ways, for instance with "subcontracts" inside a contract
        or with contract "amendments".
        Here each row is a period between two dates where all the values of the contract are fixed.

        Columns can be like in example_a.csv :
        [customer_id,
         contract_id,
         date_start,
         date_end_exclusive,  # contract ends at 00h00 that day, so that day is excluded from the contract period.
         sub_contract_end_reason,
         subscribed_power_kva,
         smart_metered,
         profile,
         customer_type,
         specific_price,
         estimated_annual_consumption_kwh,
         tension]

    Columns [date_start, date_end_exclusive] are required (names can differ), and with dtype=datetime64 (tz-naive).
    Other columns describe contract characteristics.


    """

    @classmethod
    def read_contracts_from_file(
        cls,
        file_path: str,
        date_start_col: str = "date_start",
        date_end_exclusive_col: str = "date_end_exclusive",
        date_format: str = "%Y-%m-%d",
    ) -> pd.DataFrame:
        """
        Reads contracts from a file. This will convert start and end date columns into dtype=datetime64 (tz-naive)
        :param file_path: where the source file is located.
        :param date_start_col: the name of your contract date start column.
        :param date_end_exclusive_col: the name of your contract date end column, end date is exclusive.
        :param date_format: the date format for pandas.to_datetime.
        :return: a pandas.DataFrame with a contract on each row.
        """

        df = pd.read_csv(file_path)
        for col in [date_start_col, date_end_exclusive_col]:
            if col not in df.columns:
                raise AttributeError(f"Column {col} is not present in the contracts file")
            if is_string_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], format=date_format)
        cls.check_contracts_dates(df, date_start_col, date_end_exclusive_col)
        return df

    @staticmethod
    def check_contracts_dates(
        df: pd.DataFrame,
        date_start_col: str,
        date_end_exclusive_col: str,
        is_naive: bool = True,
    ):
        """
        Checks that the two columns, date_start_col and date_end_exclusive_col, are present.
        Checks that date_start does not have any null value.
        A date_end_exclusive==NaT means that the contract has no limited duration.
        Checks that date_start < date_end_exclusive when date_end_exclusive is set
        If is_naive is set to True, check that timestamps in date_start_col and date_end_exclusive_col are naive
        If any of these checks fail, raise a ValueError

        :param df: the pandas.DataFrame containing the contracts
        :param date_start_col: the name of your contract date start column.
        :param date_end_exclusive_col: the name of your contract date end column, end date is exclusive.
        :param is_naive: whether the timestamps in the start and end dates columns are supposed to be naive
        """

        # check required columns and their types
        for col in [date_start_col, date_end_exclusive_col]:
            if col not in df.columns:
                raise ValueError(f"Required column not found : {col}")
            # data at a 'daily' precision should not have TZ information
            if not is_datetime64_dtype(df[col]) and is_naive:
                raise ValueError(
                    f"Expected tz-naive datetime dtype for column {col}, but found dtype: {df[col].dtype}"
                )
        # check NaN values for date_start
        if df[date_start_col].isnull().any():
            rows_with_nan_date_start = df[df[date_start_col].isnull()]
            raise ValueError(
                f"There are NaN values in these rows:\n{rows_with_nan_date_start}"
            )

        contracts_with_end = df.dropna(subset=[date_end_exclusive_col])
        not_ok = (
            contracts_with_end[date_start_col]
            >= contracts_with_end[date_end_exclusive_col]
        )
        if not_ok.sum() >= 1:
            raise ValueError(
                f"Some ending date happens before starting date:\n{contracts_with_end[not_ok]}"
            )

    @classmethod
    def compute_portfolio_by_day(
        cls,
        contracts: pd.DataFrame,
        columns_to_sum: list[str],
        date_start_col: str = "date_start",
        date_end_exclusive_col: str = "date_end_exclusive",
        max_date_exclusive: pd.Timestamp = None,
        ffill_until_max_date: bool = False,
    ) -> pd.DataFrame:
        """
        Given a list of contracts_with_group ,

        Returns the "portfolio" by day, which is, for each day from the first start of a contract
        to the last end of a contract, the quantities in "columns_to_sum" over time.
        See unittests or enda's guides for examples.

        If you want to compute the quantities for a group of customers, filter contracts before using this function.

        :param contracts: the dataframe containing the list of contracts to consider.
            Each contract has at least these columns :
                ["date_start", "date_end_exclusive", and all columns_to_sum]
                each column in columns_to_sum must be of a summable dtype
        :param columns_to_sum: the columns on which to compute a running-sum over time
        :param date_start_col: the name of your contract date start column.
        :param date_end_exclusive_col: the name of your contract date end column, end date is exclusive.
        :param max_date_exclusive: restricts the output to strictly before this date.
                                   Useful if you have end_dates far in the future.
        :param ffill_until_max_date: Whether to forward fill the last available value until max_date_exclusive.
            Default False
        :return: a 'portfolio' dataframe with one day per row,
                 and the following column hierarchy: (columns_to_sum, group_column)
                 Each day, we have the running sum of each columns_to_sum, for each group of contracts.
                 The first row is the earliest contract start date and the last row is the latest
                 contract start or contract end date.
        """

        for col in ["date"]:
            if col in contracts.columns:
                raise ValueError(
                    f"contracts has a column named {col}, but this name is reserved in this"
                    "function; rename your column."
                )

        cls.check_contracts_dates(contracts, date_start_col, date_end_exclusive_col)

        for col in columns_to_sum:
            if col not in contracts.columns:
                raise ValueError(f"missing column_to_sum: {col}")
            if contracts[col].isnull().any():
                rows_with_nan_c = contracts[contracts[col].isnull()]
                raise ValueError(
                    f"There are NaN values for column {col} in these rows:\n{rows_with_nan_c}"
                )

        # keep only useful columns
        df = contracts[[date_start_col, date_end_exclusive_col] + columns_to_sum]
        # create start and end events for each contract, sorted chronologically
        events = PortfolioTools.portfolio_to_events(
            df, date_start_col, date_end_exclusive_col
        )

        # remove events after max_date if they are not wanted
        if max_date_exclusive is not None:
            events = events[events["event_date"] <= max_date_exclusive]

        # for each column to sum, replace the value by their "increment" (+X if contract starts; -X if contract ends)
        for col in columns_to_sum:
            events[col] = events.apply(
                lambda row: row[col] if row["event_type"] == "start" else -row[col], axis=1
            )

        # group events by day and sum the individual contract increments of columns_to_sum to have daily increments
        df_by_day = (
            events
            .drop(columns={"event_type"})
            .groupby(["event_date"])
            .sum()
        )

        # add days that have no increment (with NA values), else the result can have gaps
        # new "NA" increments = no contract start or end event that day = increment is 0
        df_by_day = df_by_day.asfreq("D").fillna(0).astype(df_by_day.dtypes)

        # compute cumulative sums of daily increments to get daily totals
        portfolio = df_by_day.cumsum(axis=0)
        portfolio.index.name = "date"  # now values are not increments on an event_date but the total on this date

        # Forward fill the last available value until max_date_exclusive if specified
        if ffill_until_max_date:
            if max_date_exclusive is None:
                raise ValueError(
                    "ffill_until_max_date has been set to True, but no max_date_exclusive given"
                )
            portfolio = Resample.forward_fill_final_record(
                timeseries_df=portfolio, excl_end_time=max_date_exclusive
            )

        return portfolio

    @staticmethod
    @enda.tools.decorators.warning_deprecated_name(
        namespace_name="Contracts", new_namespace_name="PortfolioTools"
    )
    def get_portfolio_between_dates(
        portfolio: pd.DataFrame,
        start_datetime: pd.Timestamp,
        end_datetime_exclusive: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Keeps portfolio data between the specified dates.
        If the first date in portfolio is after start_datetime, we add missing dates with 0 as value.
        If the last date in portfolio is before end_datetime_exclusive, we forward fill the last present values
        until end_datetime_exclusive
        :param portfolio: The portfolio DataFrame. It must have a pd.DatetimeIndex with a frequency
        :param start_datetime: The start datetime from which to keep the portfolio
        :param end_datetime_exclusive: The exclusive end datetime until which to keep the portfolio
        :return: A portfolio DataFrame with values between specified dates
        """

        return PortfolioTools.get_portfolio_between_dates(
            portfolio_df=portfolio,
            start_datetime=start_datetime,
            end_datetime_exclusive=end_datetime_exclusive,
        )

    @classmethod
    def forecast_portfolio_linear(
        cls,
        portfolio_df: pd.DataFrame,
        start_forecast_date,
        end_forecast_date_exclusive,
        freq: [pd.Timedelta, str] = None,
        max_allowed_gap: pd.Timedelta = pd.Timedelta(days=1),
        tzinfo: [pytz.timezone, str, None] = None,
    ) -> pd.DataFrame:
        """
        Forecast portfolio using a linear extrapolation for the next nb_days
        The output has the frequency which is given as an input, and defaults to the one of input portfolio_df
         if nothing is provided.
        :param portfolio_df: The portfolio DataFrame to perform the forecast on
        :param start_forecast_date: The start datetime from which to extend the portfolio
        :param end_forecast_date_exclusive: The exclusive end datetime until which to extend the portfolio
        :param freq: The frequency of the resulting linearly-extrapolated portfolio
        :param max_allowed_gap: The max gap between the end of the portfolio and the start date of the extrapolation.
                                Returns an error if exceeded.
        :param tzinfo: The time-zone of the resulting dataframe, if we want to change it.
        :return: pd.DataFrame (the linearly-extrapolated forecast portfolio data)
        """
        if end_forecast_date_exclusive < start_forecast_date:
            raise ValueError("end_forecast_date must be after start_forecast_date")

        gap = portfolio_df.index.max() - start_forecast_date
        if gap > max_allowed_gap:
            raise ValueError(
                "The gap between the end of portfolio_df and start_forecast_date is too big:"
                f"{gap} (max_allowed_gap={max_allowed_gap}). Provide more recent data or change "
                "max_allowed_gap."
            )

        if not isinstance(portfolio_df.index, pd.DatetimeIndex):
            raise ValueError(
                f"portfolio_df should have a pandas.DatetimeIndex, but given {portfolio_df.index.dtype}"
            )

        if not freq:
            freq = TimeSeries.find_most_common_frequency(portfolio_df.index)

        result_index = pd.date_range(
            start=start_forecast_date,
            end=end_forecast_date_exclusive,
            freq=freq,
            name=portfolio_df.index.name,
            inclusive="left",
        )

        if tzinfo is not None:
            result_index = result_index.tz_convert(tzinfo)

        predictions = []

        for col in portfolio_df.columns:
            epoch_column = (
                "seconds_since_epoch_"
                if col != "seconds_since_epoch_"
                else "seconds_since_epoch__"
            )

            train_set = portfolio_df[[col]].copy(deep=True)
            train_set[epoch_column] = train_set.index.astype(np.int64) // 10**9

            test_set = pd.DataFrame(
                data={epoch_column: result_index.astype(np.int64) // 10**9},
                index=result_index,
            )

            lr = EndaSklearnEstimator(LinearRegression())
            lr.train(train_set, target_col=col)
            predictions.append(lr.predict(test_set, target_col=col))

        return pd.concat(predictions, axis=1, join="outer")

    @classmethod
    def forecast_portfolio_holt(
        cls,
        portfolio_df: pd.DataFrame,
        start_forecast_date,
        nb_days: int = 14,
        past_days: int = 100,
        holt_init_params=None,
        holt_fit_params=None,
    ) -> pd.DataFrame:
        """
        Forecast using exponential smoothing (Holt method) for the next nb_days
        The output has the same frequency as input portfolio_df.

        :param portfolio_df: The portfolio DataFrame to perform the forecast on
        :param start_forecast_date: when we stop the portfolio data and start forecasting
        :param nb_days: number of days after 'end_date' to forecast
        :param past_days: max number of days to use in the past used to make the forecast
                          (it is better to use only recent data)
        :param holt_init_params: the dict of parameters to give to the Holt __init__ method. If none, defaults to :
                                holt_fit_params={"initialization_method":"estimated"}
                                For more details see the statsmodels documentation :
                                https://www.statsmodels.org/stable/examples/notebooks/generated/exponential_smoothing.html
        :param holt_fit_params: the dict of params to give to the Holt.fit() method. If none, defaults to : {}
        :return: pd.DataFrame (the forecast data)
        """

        try:
            from statsmodels.tsa.api import Holt
        except ImportError as exc:
            raise ImportError(
                "statsmodels is required is you want to use this function. "
                "Try: pip install statsmodels>=0.12.0"
            ) from exc

        if holt_init_params is None:
            holt_init_params = {"initialization_method": "estimated"}

        if holt_fit_params is None:
            holt_fit_params = {}

        if nb_days < 1:
            raise ValueError(f"nb_days should be at least 1, given {nb_days}")

        if start_forecast_date > portfolio_df.index.max() + relativedelta(days=1):
            raise ValueError(
                f"Start forecast date ({start_forecast_date}) more than 1 day after the latest portfolio "
                f"information ({portfolio_df.index.max()}). Portfolio information given is too old."
            )

        if not isinstance(portfolio_df.index, pd.DatetimeIndex):
            raise ValueError(
                f"portfolio_df should have a pandas.DatetimeIndex, but given {portfolio_df.index.dtype}"
            )

        if portfolio_df.index.freq is None:
            raise ValueError(
                "Input portfolio_df must have a frequency. "
                "Maybe try to set it using pandas.index.inferred_freq"
            )

        # only keep portfolio data before the start_forecast_date
        freq = portfolio_df.index.freq
        tzinfo = portfolio_df.index.tzinfo  # can be None

        p_df = portfolio_df
        p_df = p_df[p_df.index <= start_forecast_date]

        end_forecast_date = TimezoneUtils.add_interval_to_day_dt(
            day_dt=start_forecast_date, interval=relativedelta(days=nb_days)
        )
        date_past_days_ago = TimezoneUtils.add_interval_to_day_dt(
            day_dt=start_forecast_date, interval=relativedelta(days=-past_days)
        )

        # only keep recent data to determine the trends
        p_df = p_df[p_df.index >= date_past_days_ago]

        future_index = pd.date_range(
            start_forecast_date,
            end_forecast_date,
            freq=freq,
            name=p_df.index.name,
            inclusive="left"
        )
        if tzinfo is not None:
            future_index = future_index.tz_convert(tzinfo)

        # holt needs a basic integer index
        p_df = p_df.reset_index(drop=True)
        # forecast each column (all columns are measures)
        result = p_df.apply(
            lambda x: Holt(x, **holt_init_params)
            .fit(**holt_fit_params)
            .forecast(len(future_index))
        )
        result = result.round(1)
        result.index = future_index

        return result
