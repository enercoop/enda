import pandas as pd
from pandas.api.types import is_string_dtype, is_datetime64_dtype
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.api import Holt
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
    def read_contracts_from_file(cls,
                                 file_path,
                                 date_start_col="date_start",
                                 date_end_exclusive_col="date_end_exclusive",
                                 date_format="%Y-%m-%d"):
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
    def check_contracts_dates(df, date_start_col, date_end_exclusive_col):
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
            if not is_datetime64_dtype(df[c]):
                raise ValueError("Expected tz-naive datetime dtype for column, but found dtype: {}"
                                 .format(c, df[c].dtype))
        # check NaN values for date_start
        if df[date_start_col].isnull().any():
            rows_with_nan_date_start = df[df[date_start_col].isnull()]
            raise ValueError("There are NaN values in these rows:\n{}".format(rows_with_nan_date_start))

        contracts_with_end = df.dropna(subset=[date_end_exclusive_col])
        not_ok = contracts_with_end[date_start_col] >= contracts_with_end[date_end_exclusive_col]
        if not_ok.sum() >= 1:
            raise ValueError("Some contracts end before they start:\n{}".format(contracts_with_end[not_ok]))

    @classmethod
    def compute_portfolio_by_day(
            cls,
            contracts_with_group,
            columns_to_sum,
            date_start_col="date_start",
            date_end_exclusive_col="date_end_exclusive",
            group_column="group",
            max_date_exclusive=None,
    ):
        """
        Given a list of contracts_with_group ,

        Returns the "portfolio" by day, which is, for each day and each group,
            the total number of active contracts: group_name_pdl
            and the sum of active kVA for this group: group_name_kva
        See unittests for examples.

        :param contracts_with_group: the dataframe containing the list of contract to consider.
            Each contract has at least these columns :
                ["date_start", "date_end_exclusive", "group", and all columns_to_sum]
                columns_to_sum must be of a summable dtype
        :param columns_to_sum: the columns on which to compute a running-sum over time, for each group
        :param date_start_col:
        :param date_end_exclusive_col:
        :param group_column:
        :param max_date_exclusive: restricts the output to strictly before this date.
                                   Useful if you have end_dates far in the future.

        :return: a 'portfolio' dataframe with one day per row,
                 and the following column hierarchy: (columns_to_sum, group_column)
                 Each day, we have the running sum of each columns_to_sum, for each group of contracts.
                 The first row is the earliest contract start date and the last row is the latest
                 contract start or contract end date.
        """

        cls.check_contracts_dates(contracts_with_group, date_start_col, date_end_exclusive_col)
        if group_column not in contracts_with_group.columns:
            raise ValueError("missing group_column: {}".format(group_column))

        for c in columns_to_sum:
            if c not in contracts_with_group.columns:
                raise ValueError("missing column_to_sum: {}".format(c))
            if contracts_with_group[c].isnull().any():
                rows_with_nan_c = contracts_with_group[contracts_with_group[c].isnull()]
                raise ValueError("There are NaN values for column {} in these rows:\n{}".format(c, rows_with_nan_c))

        # check that no column is named "event_type" or "event_date"
        for c in ["event_type", "event_date", "date"]:
            if c in contracts_with_group.columns:
                raise ValueError("contracts_with_group has a column named {}, but this name is reserved in this"
                                 "function; rename your column.".format(c))

        # keep only useful columns
        df = contracts_with_group[[date_start_col, date_end_exclusive_col, group_column] + columns_to_sum]
        events_columns = ["event_type", "event_date", group_column] + columns_to_sum

        # compute "contract start" and "contract end" events
        start_contract_events = df.copy(deep=True)  # all contracts must have a start date
        start_contract_events["event_type"] = "start"
        start_contract_events["event_date"] = start_contract_events[date_start_col]
        start_contract_events = start_contract_events[events_columns]

        end_contract_events = df[df[date_end_exclusive_col].notna()].copy(deep=True)  # keep contracts with an end date
        end_contract_events["event_type"] = "end"
        end_contract_events["event_date"] = end_contract_events[date_end_exclusive_col]
        end_contract_events = end_contract_events[events_columns]

        # concat all events together and sort them chronologically
        all_events = pd.concat([start_contract_events, end_contract_events])
        all_events.sort_values(by=["event_date", "event_type"], inplace=True)

        # remove events after max_date if they are not wanted
        if max_date_exclusive is not None:
            all_events = all_events[all_events["event_date"] <= max_date_exclusive]

        # for each columns_to_sum, replace the value by their "increment" (+X if contract starts; -X if contract ends)
        for c in columns_to_sum:
            all_events[c] = all_events.apply(lambda row: row[c] if row["event_type"] == "start" else -row[c], axis=1)

        # group events by day and group, and sum the increments of columns_to_sum
        df_by_day_and_group = all_events.groupby(["event_date", group_column]).sum()
        df_by_day_and_group = df_by_day_and_group.reset_index(drop=False)

        # separate groups in different columns (pivot "long" to "wide" format)
        df_by_day = df_by_day_and_group.pivot_table(
            index="event_date",
            columns=[group_column],
            values=columns_to_sum
        )
        # Give a name to level 0 of column MultiIndex
        df_by_day.columns.names = ["measure", group_column]

        # add days that have no increment (with NA values), else the result can have gaps
        # new "NA" increments = no contract start or end event that day = increment is 0
        df_by_day = df_by_day.asfreq('D').fillna(0)

        # compute cumulative sums of daily increments to get daily totals
        portfolio = df_by_day.cumsum(axis=0)
        portfolio.index.name = "date"  # now values are not increments on an event_date but the total on this date

        # careful, portfolio's columns are a hierarchy: (group, columns_to_sum)
        assert len(portfolio.columns.names) == 2

        return portfolio

    @staticmethod
    def get_daily_portfolio_between_dates(daily_portfolio, start_date, end_date_exclusive):
        """
        Adds or removes dates if needed.

        If additional dates are needed at the beginning, add these dates with 0s
        If additional dates needed at the end, copy the portfolio of the last date into the additional dates

        TODO : add unittests and doc
        :param daily_portfolio:
        :param start_date:
        :param end_date_exclusive:
        :return:
        """

        df = daily_portfolio.copy(deep=True)

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("The index of daily_portfolio should be a pd.DatetimeIndex, but given {}"
                            .format(df.index.dtype))

        if not df.index.freqstr == 'D':
            raise NotImplementedError("This function works only for freq='D' but given {}".format(df.index.freqstr))

        # TODO check types of: start_date, end_date_exclusive

        # check that there is no missing value
        if not df.isnull().sum().sum() == 0:
            raise ValueError("daily_portfolio has NaN values.")

        if start_date is not None and df.index.min() > start_date:
            # add days with empty portfolio at the beginning
            df = df.append(pd.Series(name=start_date, dtype='object'))
            df.sort_index(inplace=True)  # put the new row first
            df = df.asfreq('D').fillna(0)

        if end_date_exclusive is not None and df.index.max() < end_date_exclusive:
            # add days at the end, with the same portfolio as the last available day
            df = df.append(pd.Series(name=end_date_exclusive, dtype='object'))
            df.sort_index(inplace=True)  # make sure this new row is last
            df = df.asfreq('D', method='ffill')

        # remove dates outside of desired range
        df = df[(df.index >= start_date) & (df.index < end_date_exclusive)]
        assert df.isnull().sum().sum() == 0  # check that there is no missing value

        return df

    @classmethod
    def forecast_using_trend(cls, portfolio_df, start_forecast_date, freq, nb_days=14, past_days=100):
        """
        Forecast using exponential smoothing (Holt method) for the next nb_days
        :param portfolio_df:
        :param start_forecast_date: when we stop the portfolio data and start forecasting
        :param freq: the frequency of the output, it must be the same frequency as the input portfolio_df
        :param nb_days: number of days after 'end_date' to forecast
        :param past_days: max number of days to use in the past used to make the forecast
                          (it is better to use only recent data)
        :return: pd.DataFrame (the forecast data)
        """

        if nb_days < 1:
            raise ValueError("nb_days should be at least 1, given {}".format(nb_days))

        if not isinstance(start_forecast_date, pd.Timestamp):
            raise ValueError("Expected a pandas.Timestamp for start_forecast_date, but got {} of type {}. "
                             "You can use pandas.to_datetime(...) before passing this argument."
                             .format(start_forecast_date, type(start_forecast_date)))

        if start_forecast_date > portfolio_df.index.max() + relativedelta(days=1):
            raise ValueError("Start forecast date ({}) more than 1 day after the latest portfolio information ({}). "
                             "Portfolio information given is too old."
                             .format(start_forecast_date, portfolio_df.index.max()))

        if not isinstance(portfolio_df.index, pd.DatetimeIndex):
            raise ValueError("portfolio_df should have a pandas.DatetimeIndex, but given {}"
                             .format(portfolio_df.index.dtype))

        # only keep portfolio data before the start_forecast_date
        pf = portfolio_df
        pf = pf[pf.index <= start_forecast_date]

        end_forecast_date = TimezoneUtils.add_interval_to_day_dt(
            day_dt=start_forecast_date,
            interval=relativedelta(days=nb_days)
        )
        date_past_days_ago = TimezoneUtils.add_interval_to_day_dt(
            day_dt=start_forecast_date,
            interval=relativedelta(days=-past_days)
        )

        # only keep recent data to determine the trends
        pf = pf[pf.index >= date_past_days_ago]

        future_index = pd.date_range(
            start_forecast_date,
            end_forecast_date,
            freq=freq,
            name=pf.index.name,
            closed='left'
        ).tz_convert(pf.index.tzinfo)

        # holt needs a basic integer index
        pf = pf.reset_index(drop=True)
        # forecast each column (all columns are measures)
        result = pf.apply(lambda x: Holt(x, initialization_method="estimated").fit().forecast(len(future_index)))
        result = result.round(1)
        result.index = future_index

        return result

