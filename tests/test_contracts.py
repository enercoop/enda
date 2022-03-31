import logging
import os
import pandas as pd
import pathlib
import unittest

from enda.contracts import Contracts
from enda.timeseries import TimeSeries


class TestContracts(unittest.TestCase):

    def setUp(self):
        logging.captureWarnings(True)
        logging.disable(logging.ERROR)

    def tearDown(self):
        logging.captureWarnings(False)
        logging.disable(logging.NOTSET)

    EXAMPLE_A_DIR = os.path.join(pathlib.Path(__file__).parent.absolute(), "example_a")
    CONTRACTS_PATH = os.path.join(EXAMPLE_A_DIR, "contracts.csv")

    def test_read_contracts_from_file(self):
        contracts = Contracts.read_contracts_from_file(TestContracts.CONTRACTS_PATH)
        self.assertEqual((7, 12), contracts.shape)

    def test_check_contracts_dates(self):
        contracts = Contracts.read_contracts_from_file(
            TestContracts.CONTRACTS_PATH,
            date_start_col="date_start",
            date_end_exclusive_col="date_end_exclusive",
            date_format="%Y-%m-%d"
        )

        # check that it fails if the given date_start_col is not there
        with self.assertRaises(ValueError):
            Contracts.check_contracts_dates(
                contracts,
                date_start_col="dummy",
                date_end_exclusive_col="date_end_exclusive"
            )

        # check that it fails if one contract ends before it starts
        c = contracts.copy(deep=True)
        # set a wrong date_end_exclusive for the first contract
        c.loc[0, "date_end_exclusive"] = pd.to_datetime("2020-09-16")
        with self.assertRaises(ValueError):
            Contracts.check_contracts_dates(
                c,
                date_start_col="dummy",
                date_end_exclusive_col="date_end_exclusive"
            )

    @staticmethod
    def get_simple_portfolio_by_day():
        contracts = Contracts.read_contracts_from_file(TestContracts.CONTRACTS_PATH)
        contracts["contracts_count"] = 1  # add a variable to count the number of contracts for each row

        # count the running total, each day, of some columns
        portfolio_by_day = Contracts.compute_portfolio_by_day(
            contracts,
            columns_to_sum=["contracts_count", "subscribed_power_kva", "estimated_annual_consumption_kwh"],
            date_start_col="date_start",
            date_end_exclusive_col="date_end_exclusive"
        )

        return portfolio_by_day

    def test_compute_portfolio_by_day_1(self):
        """" test with a single group """
        portfolio_by_day = TestContracts.get_simple_portfolio_by_day()

        # print(portfolio_by_day)
        self.assertEqual((11, 3), portfolio_by_day.shape)
        self.assertEqual(4, portfolio_by_day.loc["2020-09-26", "contracts_count"])
        self.assertEqual(30, portfolio_by_day.loc["2020-09-26", "subscribed_power_kva"])
        self.assertEqual(5, portfolio_by_day["contracts_count"].max())
        self.assertEqual(48, portfolio_by_day["subscribed_power_kva"].max())

    def test_compute_portfolio_by_day_2(self):
        """" test with 2 groups , and a single measure to sum"""

        contracts = Contracts.read_contracts_from_file(TestContracts.CONTRACTS_PATH)

        contracts_sm = contracts[contracts["smart_metered"]]
        pf_sm = Contracts.compute_portfolio_by_day(contracts_sm, columns_to_sum=["subscribed_power_kva"])

        contracts_slp = contracts[~contracts["smart_metered"]]
        pf_slp = Contracts.compute_portfolio_by_day(contracts_slp, columns_to_sum=["subscribed_power_kva"])

        # print(pf_sm, pf_slp)
        self.assertEqual(pf_sm.shape, (5, 1))
        self.assertEqual(pf_slp.shape, (11, 1))

        self.assertEqual(18, pf_sm.loc["2020-09-20", "subscribed_power_kva"])
        self.assertEqual(27, pf_slp.loc["2020-09-26", "subscribed_power_kva"])

    def test_get_portfolio_between_dates_1(self):
        """ test with a portfolio by day """
        portfolio_by_day = TestContracts.get_simple_portfolio_by_day()

        self.assertEqual(pd.to_datetime("2020-09-16"), portfolio_by_day.index.min())
        self.assertEqual(pd.to_datetime("2020-09-26"), portfolio_by_day.index.max())

        pf = Contracts.get_portfolio_between_dates(
            portfolio_by_day,
            start_datetime=pd.to_datetime("2020-09-10"),
            end_datetime_exclusive=pd.to_datetime("2020-09-30")
        )
        # print(pf["contracts_count"])
        self.assertEqual(pd.to_datetime("2020-09-10"), pf.index.min())
        self.assertEqual(pd.to_datetime("2020-09-29"), pf.index.max())
        self.assertEqual(0, pf.loc["2020-09-12", "contracts_count"])
        self.assertEqual(4, pf.loc["2020-09-28", "contracts_count"])

    def test_get_portfolio_between_dates_2(self):
        """ test with a portfolio by 15min step """
        portfolio_by_day = TestContracts.get_simple_portfolio_by_day()
        tz_str = 'Europe/Berlin'

        pf = TimeSeries.interpolate_daily_to_sub_daily_data(
            portfolio_by_day,
            freq='15min',
            tz=tz_str
        )
        # print(pf)
        self.assertEqual(pd.to_datetime("2020-09-16 00:00:00+02:00"), pf.index.min())
        self.assertEqual(pd.to_datetime("2020-09-26 23:45:00+02:00"), pf.index.max())
        self.assertIsInstance(pf.index, pd.DatetimeIndex)
        self.assertEqual(tz_str, str(pf.index[0].tzinfo))

        pf2 = Contracts.get_portfolio_between_dates(
            pf,
            start_datetime=pd.to_datetime("2020-09-10 00:00:00+02:00").tz_convert(tz_str),
            end_datetime_exclusive=pd.to_datetime("2020-09-30 00:00:00+02:00").tz_convert(tz_str)
        )
        # print(pf2)
        self.assertEqual(pd.to_datetime("2020-09-10 00:00:00+02:00"), pf2.index.min())
        self.assertEqual(pd.to_datetime("2020-09-29 23:45:00+02:00"), pf2.index.max())
        self.assertEqual(0, pf2.loc["2020-09-12 10:30:00+02:00", "contracts_count"])
        self.assertEqual(4, pf2.loc["2020-09-27 05:15:00+02:00", "contracts_count"])

    def test_forecast_portfolio_linear_1(self):
        """ Test on a portfolio_by_day"""

        portfolio_by_day = TestContracts.get_simple_portfolio_by_day()

        # print(portfolio_by_day['subscribed_power_kva'])

        # linear forecast using the 11 days gives an increasing trend
        forecast_by_day_a = Contracts.forecast_portfolio_linear(
            portfolio_by_day,
            start_forecast_date=pd.to_datetime("2020-09-27"),
            end_forecast_date_exclusive=pd.to_datetime("2020-09-30"),
            freq='D'
        )

        # print(forecast_by_day_a['subscribed_power_kva'])

        self.assertEqual((3, 3), forecast_by_day_a.shape)
        self.assertEqual(pd.to_datetime("2020-09-29"), forecast_by_day_a.index.max())
        self.assertGreaterEqual(forecast_by_day_a.loc["2020-09-27", "subscribed_power_kva"], 40)
        self.assertGreaterEqual(forecast_by_day_a.loc["2020-09-29", "subscribed_power_kva"], 40)

        # linear forecast using only the last 7 days gives a decreasing trend
        forecast_by_day_b = Contracts.forecast_portfolio_linear(
            portfolio_by_day[portfolio_by_day.index >= '2020-09-20'],
            start_forecast_date=pd.to_datetime("2020-09-27"),
            end_forecast_date_exclusive=pd.to_datetime("2020-10-02"),
            freq='D'
        )

        # print(forecast_by_day_b['subscribed_power_kva'])

        self.assertEqual((5, 3), forecast_by_day_b.shape)
        self.assertEqual(pd.to_datetime("2020-10-01"), forecast_by_day_b.index.max())
        self.assertLessEqual(forecast_by_day_b.loc["2020-09-27", "subscribed_power_kva"], 40)
        self.assertLessEqual(forecast_by_day_b.loc["2020-09-29", "subscribed_power_kva"], 40)

    def test_forecast_portfolio_linear_2(self):
        """Test on a portfolio at freq=7min"""

        portfolio_by_day = TestContracts.get_simple_portfolio_by_day()
        portfolio_by_20min = TimeSeries.interpolate_daily_to_sub_daily_data(portfolio_by_day,
                                                                            freq='20min', tz='Europe/Paris')

        # print(portfolio_by_20min)

        # linear forecast_by_10min, give it a portfolio_by_20min to train
        forecast_by_10min = Contracts.forecast_portfolio_linear(
            portfolio_by_20min,
            start_forecast_date=pd.to_datetime("2020-09-27 00:00:00+02:00").tz_convert("Europe/Paris"),
            end_forecast_date_exclusive=pd.to_datetime("2020-09-30 00:00:00+02:00").tz_convert("Europe/Paris"),
            freq='10min',
            tzinfo='Europe/Paris'
        )

        # print(forecast_by_10min)
        self.assertEqual((432, 3), forecast_by_10min.shape)
        self.assertEqual("Europe/Paris", str(forecast_by_10min.index.tzinfo))
        self.assertGreaterEqual(forecast_by_10min.loc["2020-09-27 00:00:00+02:00", "subscribed_power_kva"], 40)
        self.assertGreaterEqual(forecast_by_10min.loc["2020-09-29 00:00:00+02:00", "subscribed_power_kva"], 40)

    def test_forecast_portfolio_holt_1(self):
        """ Test on a portfolio_by_day"""

        portfolio_by_day = TestContracts.get_simple_portfolio_by_day()

        # print(portfolio_by_day)

        forecast_by_day = Contracts.forecast_portfolio_holt(
            portfolio_by_day,
            start_forecast_date=pd.to_datetime("2020-09-27"),
            nb_days=3,
            past_days=10  # looking at 10 days : expect increasing trend
        )

        # print(forecast_by_day)
        self.assertEqual((3, 3), forecast_by_day.shape)
        self.assertEqual(pd.to_datetime("2020-09-29"), forecast_by_day.index.max())
        self.assertLessEqual(38, forecast_by_day.loc["2020-09-27", "subscribed_power_kva"])
        self.assertGreaterEqual(40, forecast_by_day.loc["2020-09-29", "subscribed_power_kva"])

    def test_forecast_portfolio_holt_2(self):
        """ Test on a portfolio at freq=5min """

        portfolio_by_day = TestContracts.get_simple_portfolio_by_day()
        portfolio_5min = TimeSeries.interpolate_daily_to_sub_daily_data(portfolio_by_day,
                                                                        freq='5min', tz='Europe/Paris')
        # print(portfolio_5min[['subscribed_power_kva']][portfolio_5min.index >= '2020-09-19 00:00:00+02:00'])

        forecast_5_min = Contracts.forecast_portfolio_holt(
            portfolio_5min,
            start_forecast_date=pd.to_datetime("2020-09-27 00:00:00+02:00").tz_convert("Europe/Paris"),
            nb_days=5,
            past_days=7,  # if we look only at the last 7 days, the trend is decreasing (10 days would be increasing),
            holt_init_params={"exponential": True, "damped_trend": True, "initialization_method": "estimated"},
            holt_fit_params={"damping_trend": 0.98}
        )
        # print(forecast_5_min[['subscribed_power_kva']])

        self.assertEqual((12*24*5, 3), forecast_5_min.shape)
        self.assertEqual(
            pd.to_datetime("2020-10-01 23:55:00+02:00"),
            forecast_5_min.index.max()
        )
        self.assertGreaterEqual(30, forecast_5_min.loc["2020-09-27 00:00:00+02:00", "subscribed_power_kva"])
        self.assertGreaterEqual(30, forecast_5_min.loc["2020-10-01 23:55:00+02:00", "subscribed_power_kva"])
