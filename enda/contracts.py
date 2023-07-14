import pandas as pd
from pandas.api.types import is_string_dtype, is_datetime64_dtype
from dateutil.relativedelta import relativedelta
import pytz

from enda.timezone_utils import TimezoneUtils


class Contracts:
    """
    A class to help handle contracts data.

    contracts : a dataframe with a contract or sub-contract on each row.
        A contract row has fixed-values between two dates. In some systems, some values of a contract can change over
        time, for instance the "subscribed power" of an electricity consumption contract.
        Different ERPs can handle it in different ways, for instance with "sub-contracts" inside a contract
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
        file_path,
        date_start_col="date_start",
        date_end_exclusive_col="date_end_exclusive",
        date_format="%Y-%m-%d",
    ):
        """
        Reads contracts from a file. This will convert start and end date columns into dtype=datetime64 (tz-naive)
        :param file_path: where the source file is located.
        :param date_start_col: the name of your contract date start column.
        :param date_end_exclusive_col: the name of your contract date end column, end date is exclusive.
        :param date_format: the date format for pandas.to_datetime.
        :return: a pandas.DataFrame with a contract on each row.
        """

        df = pd.read_csv(file_path)
        for c in [date_start_col, date_end_exclusive_col]:
            if is_string_dtype(df[c]):
                df[c] = pd.to_datetime(df[c], format=date_format)
        cls.check_contracts_dates(df, date_start_col, date_end_exclusive_col)
        return df

    @staticmethod
    def check_contracts_dates(
        df, date_start_col, date_end_exclusive_col, is_naive=True
    ):
        """
        Checks that the two columns are present, with dtype=datetime64 (tz-naive)
        Checks that date_start is always present.
        A date_end_exclusive==NaT means that the contract has no limited duration.
        Checks that date_start < date_end_exclusive when date_end_exclusive is set

        :param df: the pandas.DataFrame containing the contracts
        :param date_start_col: the name of your contract date start column.
        :param date_end_exclusive_col: the name of your contract date end column, end date is exclusive.
        """

        # check required columns and their types
        for c in [date_start_col, date_end_exclusive_col]:
            if c not in df.columns:
                raise ValueError("Required column not found : {}".format(c))
            # data at a 'daily' precision should not have TZ information
            if not is_datetime64_dtype(df[c]) and is_naive:
                raise ValueError(
                    "Expected tz-naive datetime dtype for column {}, but found dtype: {}".format(
                        c, df[c].dtype
                    )
                )
        # check NaN values for date_start
        if df[date_start_col].isnull().any():
            rows_with_nan_date_start = df[df[date_start_col].isnull()]
            raise ValueError(
                "There are NaN values in these rows:\n{}".format(
                    rows_with_nan_date_start
                )
            )

        contracts_with_end = df.dropna(subset=[date_end_exclusive_col])
        not_ok = (
            contracts_with_end[date_start_col]
            >= contracts_with_end[date_end_exclusive_col]
        )
        if not_ok.sum() >= 1:
            raise ValueError(
                "Some ending date happens before starting date:\n{}".format(
                    contracts_with_end[not_ok]
                )
            )

    @staticmethod
    def __contract_to_events(contracts, date_start_col, date_end_exclusive_col):
        # check that no column is named "event_type" or "event_date"
        for c in ["event_type", "event_date"]:
            if c in contracts.columns:
                raise ValueError(
                    "contracts has a column named {}, but this name is reserved in this"
                    "function; rename your column.".format(c)
                )

        columns_to_keep = [
            c
            for c in contracts.columns
            if c not in [date_start_col, date_end_exclusive_col]
        ]
        events_columns = ["event_type", "event_date"] + columns_to_keep

        # compute "contract start" and "contract end" events
        start_contract_events = contracts.copy(
            deep=True
        )  # all contracts must have a start date
        start_contract_events["event_type"] = "start"
        start_contract_events["event_date"] = start_contract_events[date_start_col]
        start_contract_events = start_contract_events[events_columns]

        # for "contract end" events, only keep contracts with an end date (NaT = contract is not over)
        end_contract_events = contracts[contracts[date_end_exclusive_col].notna()].copy(
            deep=True
        )
        end_contract_events["event_type"] = "end"
        end_contract_events["event_date"] = end_contract_events[date_end_exclusive_col]
        end_contract_events = end_contract_events[events_columns]

        # concat all events together and sort them chronologically
        all_events = pd.concat([start_contract_events, end_contract_events])
        all_events.sort_values(by=["event_date", "event_type"], inplace=True)

        return all_events

    @classmethod
    def compute_portfolio_by_day(
        cls,
        contracts,
        columns_to_sum,
        date_start_col="date_start",
        date_end_exclusive_col="date_end_exclusive",
        max_date_exclusive=None,
    ):
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
        :param date_start_col:
        :param date_end_exclusive_col:
        :param max_date_exclusive: restricts the output to strictly before this date.
                                   Useful if you have end_dates far in the future.

        :return: a 'portfolio' dataframe with one day per row,
                 and the following column hierarchy: (columns_to_sum, group_column)
                 Each day, we have the running sum of each columns_to_sum, for each group of contracts.
                 The first row is the earliest contract start date and the last row is the latest
                 contract start or contract end date.
        """

        for c in ["date"]:
            if c in contracts.columns:
                raise ValueError(
                    "contracts has a column named {}, but this name is reserved in this"
                    "function; rename your column.".format(c)
                )

        cls.check_contracts_dates(contracts, date_start_col, date_end_exclusive_col)

        for c in columns_to_sum:
            if c not in contracts.columns:
                raise ValueError("missing column_to_sum: {}".format(c))
            if contracts[c].isnull().any():
                rows_with_nan_c = contracts[contracts[c].isnull()]
                raise ValueError(
                    "There are NaN values for column {} in these rows:\n{}".format(
                        c, rows_with_nan_c
                    )
                )

        # keep only useful columns
        df = contracts[[date_start_col, date_end_exclusive_col] + columns_to_sum]
        # create start and end events for each contract, sorted chronologically
        events = cls.__contract_to_events(df, date_start_col, date_end_exclusive_col)

        # remove events after max_date if they are not wanted
        if max_date_exclusive is not None:
            events = events[events["event_date"] <= max_date_exclusive]

        # for each column to sum, replace the value by their "increment" (+X if contract starts; -X if contract ends)
        for c in columns_to_sum:
            events[c] = events.apply(
                lambda row: row[c] if row["event_type"] == "start" else -row[c], axis=1
            )

        # group events by day and sum the individual contract increments of columns_to_sum to have daily increments
        df_by_day = events.groupby(["event_date"]).sum()

        # add days that have no increment (with NA values), else the result can have gaps
        # new "NA" increments = no contract start or end event that day = increment is 0
        df_by_day = df_by_day.asfreq("D").fillna(0)

        # compute cumulative sums of daily increments to get daily totals
        portfolio = df_by_day.cumsum(axis=0)
        portfolio.index.name = "date"  # now values are not increments on an event_date but the total on this date

        return portfolio

    @staticmethod
    def get_portfolio_between_dates(portfolio, start_datetime, end_datetime_exclusive):
        """
        Adds or removes dates if needed.

        If additional dates are needed at the beginning, add these dates with 0s
        If additional dates needed at the end, copy the portfolio of the last date into the additional dates

        :param portfolio: the dataframe with the portfolio, must have a pandas.DatetimeIndex with frequency
        :param start_datetime:
        :param end_datetime_exclusive:
        :return:
        """

        df = portfolio.copy(deep=True)

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(
                "The index of daily_portfolio should be a pd.DatetimeIndex, but given {}".format(
                    df.index.dtype
                )
            )

        if df.index.freq is None:
            raise ValueError(
                "portfolio.index needs to have a freq. "
                "Maybe try to set one using df.index.inferred_freq"
            )

        freq = df.index.freq

        # check that there is no missing value
        if not df.isnull().sum().sum() == 0:
            raise ValueError("daily_portfolio has NaN values.")

        if start_datetime is not None and df.index.min() > start_datetime:
            # add days with empty portfolio at the beginning
            df = df.append(pd.Series(name=start_datetime, dtype="object"))
            df.sort_index(inplace=True)  # put the new row first
            df = df.asfreq(freq).fillna(0)

        if (
            end_datetime_exclusive is not None
            and df.index.max() < end_datetime_exclusive
        ):
            # add days at the end, with the same portfolio as the last available day
            df = df.append(pd.Series(name=end_datetime_exclusive, dtype="object"))
            df.sort_index(inplace=True)  # make sure this new row is last
            df = df.asfreq(freq, method="ffill")

        # remove dates outside of desired range
        df = df[(df.index >= start_datetime) & (df.index < end_datetime_exclusive)]
        assert df.isnull().sum().sum() == 0  # check that there is no missing value

        return df

    @classmethod
    def forecast_portfolio_linear(
        cls,
        portfolio_df: pd.DataFrame,
        start_forecast_date,
        end_forecast_date_exclusive,
        freq: [pd.Timedelta, str],
        max_allowed_gap: pd.Timedelta = pd.Timedelta(days=1),
        tzinfo: [pytz.timezone, str, None] = None,
    ):
        if end_forecast_date_exclusive < start_forecast_date:
            raise ValueError("end_forecast_date must be after start_forecast_date")

        gap = portfolio_df.index.max() - start_forecast_date
        if gap > max_allowed_gap:
            raise ValueError(
                "The gap between the end of portfolio_df and start_forecast_date is too big:"
                "{} (max_allowed_gap={}). Provide more recent data or change max_allowed_gap.".format(
                    gap, max_allowed_gap
                )
            )

        if not isinstance(portfolio_df.index, pd.DatetimeIndex):
            raise ValueError(
                "portfolio_df should have a pandas.DatetimeIndex, but given {}".format(
                    portfolio_df.index.dtype
                )
            )

        try:
            from enda.ml_backends.sklearn_estimator import EndaSklearnEstimator
            from sklearn.linear_model import LinearRegression
            import numpy as np
        except ImportError:
            raise ImportError(
                "sklearn is required is you want to use this function. "
                "Try: pip install scikit-learn"
            )

        result_index = pd.date_range(
            start=start_forecast_date,
            end=end_forecast_date_exclusive,
            freq=freq,
            name=portfolio_df.index.name,
            closed="left",
        )

        if tzinfo is not None:
            result_index = result_index.tz_convert(tzinfo)

        predictions = []

        for c in portfolio_df.columns:
            epoch_column = (
                "seconds_since_epoch_"
                if c != "seconds_since_epoch_"
                else "seconds_since_epoch__"
            )

            train_set = portfolio_df[[c]].copy(deep=True)
            train_set[epoch_column] = train_set.index.astype(np.int64) // 10**9

            test_set = pd.DataFrame(
                data={epoch_column: result_index.astype(np.int64) // 10**9},
                index=result_index,
            )

            lr = EndaSklearnEstimator(LinearRegression())
            lr.train(train_set, target_col=c)
            predictions.append(lr.predict(test_set, target_col=c))

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
    ):
        """
        Forecast using exponential smoothing (Holt method) for the next nb_days
        The output has the same frequency as input portfolio_df.

        :param portfolio_df:
        :param start_forecast_date: when we stop the portfolio data and start forecasting
        :param nb_days: number of days after 'end_date' to forecast
        :param past_days: max number of days to use in the past used to make the forecast
                          (it is better to use only recent data)
        :param holt_init_params: the dict of parameters to give to the Holt __init__ method. If none, defaults to :
                                holt_fit_params={"initialization_method":"estimated"}
                                For more details see the statsmodels documentation :
                                https://www.statsmodels.org/stable/examples/notebooks/generated/exponential_smoothing.html
        :param holt_fit_params: the dict of params to give to the Holt.fi() method. If none, defaults to : {}
        :return: pd.DataFrame (the forecast data)
        """

        try:
            from statsmodels.tsa.api import Holt
        except ImportError:
            raise ImportError(
                "statsmodels is required is you want to use this function. "
                "Try: pip install statsmodels>=0.12.0"
            )

        if holt_init_params is None:
            holt_init_params = {"initialization_method": "estimated"}

        if holt_fit_params is None:
            holt_fit_params = {}

        if nb_days < 1:
            raise ValueError("nb_days should be at least 1, given {}".format(nb_days))

        if start_forecast_date > portfolio_df.index.max() + relativedelta(days=1):
            raise ValueError(
                "Start forecast date ({}) more than 1 day after the latest portfolio information ({}). "
                "Portfolio information given is too old.".format(
                    start_forecast_date, portfolio_df.index.max()
                )
            )

        if not isinstance(portfolio_df.index, pd.DatetimeIndex):
            raise ValueError(
                "portfolio_df should have a pandas.DatetimeIndex, but given {}".format(
                    portfolio_df.index.dtype
                )
            )

        if portfolio_df.index.freq is None:
            raise ValueError(
                "Input portfolio_df must have a frequency. "
                "Maybe try to set it using pandas.index.inferred_freq"
            )

        # only keep portfolio data before the start_forecast_date
        freq = portfolio_df.index.freq
        tzinfo = portfolio_df.index.tzinfo  # can be None

        pf = portfolio_df
        pf = pf[pf.index <= start_forecast_date]

        end_forecast_date = TimezoneUtils.add_interval_to_day_dt(
            day_dt=start_forecast_date, interval=relativedelta(days=nb_days)
        )
        date_past_days_ago = TimezoneUtils.add_interval_to_day_dt(
            day_dt=start_forecast_date, interval=relativedelta(days=-past_days)
        )

        # only keep recent data to determine the trends
        pf = pf[pf.index >= date_past_days_ago]

        future_index = pd.date_range(
            start_forecast_date,
            end_forecast_date,
            freq=freq,
            name=pf.index.name,
            closed="left",
        )
        if tzinfo is not None:
            future_index = future_index.tz_convert(tzinfo)

        # holt needs a basic integer index
        pf = pf.reset_index(drop=True)
        # forecast each column (all columns are measures)
        result = pf.apply(
            lambda x: Holt(x, **holt_init_params)
            .fit(**holt_fit_params)
            .forecast(len(future_index))
        )
        result = result.round(1)
        result.index = future_index

        return result
