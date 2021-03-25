import os
import pathlib
import pandas as pd
from enda.contracts import Contracts
from enda.timeseries import TimeSeries


class TestUtils:
    """ Some functions useful for the tests. """

    EXAMPLE_A_DIR = os.path.join(pathlib.Path(__file__).parent.absolute(), "example_a")

    @staticmethod
    def read_example_a_train_test_sets():
        """ Read the example a data, put all contracts in just one group and compute train/test sets """

        contracts = Contracts.read_contracts_from_file(os.path.join(TestUtils.EXAMPLE_A_DIR, 'contracts.csv'))
        contracts["contracts_count"] = 1  # add a variable to count the number of contracts for each row

        # count the running total, each day, of some columns
        portfolio_by_day = Contracts.compute_portfolio_by_day(
            contracts,
            columns_to_sum=["contracts_count", "subscribed_power_kva", "estimated_annual_consumption_kwh"],
            date_start_col="date_start",
            date_end_exclusive_col="date_end_exclusive"
        )

        # restrict/extend the portfolio_by_day to desired dates
        portfolio_by_day = Contracts.get_portfolio_between_dates(
            portfolio_by_day,
            start_datetime=pd.to_datetime('2020-09-16'),
            end_datetime_exclusive=pd.to_datetime('2020-09-24')
        )

        # turn the portfolio_by_day into a portfolio timeseries with our desired freq and timezone
        portfolio = TimeSeries.interpolate_daily_to_sub_daily_data(
            portfolio_by_day,
            freq='35min',
            tz='Europe/Paris'
        )

        # read historical load, weather and TSO forecast data
        historic_load_measured = pd.read_csv(os.path.join(TestUtils.EXAMPLE_A_DIR, "historic_load_measured.csv"))
        weather_and_tso_forecasts = pd.read_csv(os.path.join(TestUtils.EXAMPLE_A_DIR, "weather_and_tso_forecasts.csv"))

        for df in [historic_load_measured, weather_and_tso_forecasts]:
            df['time'] = pd.to_datetime(df['time'])
            # for now df['time'] can be of dtype "object" because there are 2 french timezones: +60min and +120min.
            # it is important to align time-zone to 'Europe/Paris' to make sure the df has a pandas.DatetimeIndex
            df['time'] = TimeSeries.align_timezone(df['time'], tzinfo='Europe/Paris')
            df.set_index('time', inplace=True)

        # prepare datasets
        target_name = "load_kw"
        train_set = portfolio[portfolio.index <= historic_load_measured.index.max()]
        train_set = pd.merge(
            train_set,
            (historic_load_measured['slp_kw']+historic_load_measured['smart_metered_kw']).to_frame(target_name),
            how='inner', left_index=True, right_index=True
        )
        train_set = pd.merge(
            train_set,
            weather_and_tso_forecasts,
            how='inner', left_index=True, right_index=True
        )

        # lets create the input data for our forecast
        test_set = portfolio[portfolio.index >= pd.to_datetime('2020-09-20 00:00:00+02:00')]
        test_set = pd.merge(
            test_set,
            weather_and_tso_forecasts,
            how='inner', left_index=True, right_index=True
        )

        assert isinstance(test_set.index, pd.DatetimeIndex)

        return train_set, test_set, target_name

    @staticmethod
    def read_example_b_train_test_sets():
        pass
