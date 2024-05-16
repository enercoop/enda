"""A module for testing the Contracts class in enda/contracts.py"""

import logging
import os
import pathlib
import unittest
import numpy as np
import pandas as pd

from enda.contracts import Contracts
from enda.tools.timeseries import TimeSeries


class TestContracts(unittest.TestCase):
    """
    This class aims at testing the functions of the Contracts class in enda/contracts.py
    """

    def setUp(self):
        logging.captureWarnings(True)
        logging.disable(logging.ERROR)

    def tearDown(self):
        logging.captureWarnings(False)
        logging.disable(logging.NOTSET)

    EXAMPLE_A_DIR = os.path.join(pathlib.Path(__file__).parent.absolute(), "example_a")
    CONTRACTS_PATH = os.path.join(EXAMPLE_A_DIR, "contracts.csv")

    def test_read_contracts_from_file(self):
        """
        Test the read_contracts_from_file function
        """

        test_contracts_dir = os.path.join(
            pathlib.Path(__file__).parent.absolute(), "unittests_data/contracts"
        )
        no_date_contracts_path = os.path.join(
            test_contracts_dir, "contracts_no_date.csv"
        )
        normal_contracts_path = os.path.join(test_contracts_dir, "contracts.csv")

        # Check what happens if date columns are not dates in the file
        # (it should raise a ValueError in check_contracts_dates)

        with self.assertRaises(ValueError):
            Contracts.read_contracts_from_file(
                file_path=no_date_contracts_path,
                date_start_col="date_start",
                date_end_exclusive_col="date_end_excl",
            )

        # Check that providing a wrong date format raises a ValueError
        with self.assertRaises(ValueError):
            Contracts.read_contracts_from_file(
                file_path=normal_contracts_path,
                date_start_col="date_start",
                date_end_exclusive_col="date_end_excl",
                date_format="%d-%m-%Y",
            )

        # Check the returned DataFrame
        expected_output_df = pd.DataFrame(
            data=[
                {
                    "station": "station1",
                    "date_start": pd.Timestamp(2023, 1, 1),
                    "date_end_excl": pd.Timestamp(2024, 1, 1),
                },
                {
                    "station": "station2",
                    "date_start": pd.Timestamp(2023, 1, 1),
                    "date_end_excl": pd.NaT,
                },
            ],
            index=[0, 1],
        )

        output_df = Contracts.read_contracts_from_file(
            file_path=normal_contracts_path,
            date_start_col="date_start",
            date_end_exclusive_col="date_end_excl",
        )

        pd.testing.assert_frame_equal(output_df, expected_output_df)

    def test_check_contracts_dates(self):
        """
        Test the check_contracts_dates function
        """
        dummy_contracts_df = pd.DataFrame(
            data=[
                {
                    "station": "station1",
                    "date_start": pd.Timestamp(year=2023, month=1, day=1),
                    "date_end_excl": pd.Timestamp(year=2024, month=1, day=1),
                },
                {
                    "station": "station2",
                    "date_start": pd.Timestamp(year=2023, month=1, day=1),
                    "date_end_excl": np.nan,
                },
            ]
        )

        # Check that specifying a wrong column name raises an error
        with self.assertRaises(ValueError):
            Contracts.check_contracts_dates(
                dummy_contracts_df,
                date_start_col="dummy",
                date_end_exclusive_col="date_end_excl",
            )

        # Check that having localized timestamps with is_naive=True raises an error

        error_localized_df = dummy_contracts_df.copy()
        error_localized_df.loc[1, "date_end_excl"] = pd.Timestamp(
            year=2024, month=1, day=1, tz="Europe/Paris"
        )

        with self.assertRaises(ValueError):
            Contracts.check_contracts_dates(
                error_localized_df,
                date_start_col="date_start",
                date_end_exclusive_col="date_end_excl",
                is_naive=True,
            )

        # Check that having non-timestamps values in date cols raises an error

        str_df = dummy_contracts_df.copy()
        str_df.loc[0, "date_start"] = "I forgot to put a date"
        with self.assertRaises(ValueError):
            Contracts.check_contracts_dates(
                str_df,
                date_start_col="date_start",
                date_end_exclusive_col="date_end_excl",
            )

        # Check that having nan values in date_start column raises an error

        nan_df = dummy_contracts_df.copy()
        nan_df.loc[1, "date_start"] = np.nan

        with self.assertRaises(ValueError):
            Contracts.check_contracts_dates(
                nan_df,
                date_start_col="date_start",
                date_end_exclusive_col="date_end_excl",
            )

        # Check that having a contract end date earlier than the start date raises an error

        end_before_start_df = dummy_contracts_df.copy()
        end_before_start_df.loc[0, "date_end_excl"] = pd.Timestamp(
            year=2022, month=12, day=31
        )

        with self.assertRaises(ValueError):
            Contracts.check_contracts_dates(
                end_before_start_df,
                date_start_col="date_start",
                date_end_exclusive_col="date_end_excl",
            )

        # Check that the function works with naïve timestamps
        Contracts.check_contracts_dates(
            dummy_contracts_df,
            date_start_col="date_start",
            date_end_exclusive_col="date_end_excl",
        )

        # Check that the function works with localized timestamps and is_naive=False

        localized_df = dummy_contracts_df.copy()
        localized_df["date_start"] = localized_df["date_start"].dt.tz_localize(
            "Europe/Paris"
        )
        localized_df["date_end_excl"] = localized_df["date_end_excl"].dt.tz_localize(
            "Europe/Paris"
        )

        Contracts.check_contracts_dates(
            localized_df,
            date_start_col="date_start",
            date_end_exclusive_col="date_end_excl",
            is_naive=False,
        )

    def test_compute_portfolio_by_day(self):
        """
        Test the compute_portfolio_by_day function
        """

        input_contracts_df = pd.DataFrame(
            data=[
                {
                    "contract_id": "contract_1",
                    "start_date": pd.Timestamp(2023, 1, 1),
                    "excl_end_date": pd.Timestamp(2023, 2, 1),
                    "subscribed_kva": 6,
                },
                {
                    "contract_id": "contract_2",
                    "start_date": pd.Timestamp(2023, 1, 2),
                    "excl_end_date": pd.Timestamp(2023, 1, 4),
                    "subscribed_kva": 4,
                },
                {
                    "contract_id": "contract_3",
                    "start_date": pd.Timestamp(2023, 1, 1),
                    "excl_end_date": pd.Timestamp(2023, 1, 3),
                    "subscribed_kva": 10,
                },
                {
                    "contract_id": "contract_3",
                    "start_date": pd.Timestamp(2023, 1, 3),
                    "excl_end_date": pd.Timestamp(2023, 1, 5),
                    "subscribed_kva": 15,
                },
            ]
        )

        input_contracts_df["contracts_count"] = 1

        # Check when "date" column is already present in DataFrame

        wrong_colname_df = input_contracts_df.copy()
        wrong_colname_df.rename(columns={"start_date": "date"}, inplace=True)

        with self.assertRaises(ValueError):
            Contracts.compute_portfolio_by_day(
                contracts=wrong_colname_df,
                columns_to_sum=["contracts_count", "subscribed_kva"],
                date_start_col="start_date",
                date_end_exclusive_col="excl_end_date",
            )

        # Check with a missing column to sum

        with self.assertRaises(ValueError):
            Contracts.compute_portfolio_by_day(
                contracts=input_contracts_df,
                columns_to_sum=["number_contracts", "subscribed_kva"],
                date_start_col="start_date",
                date_end_exclusive_col="excl_end_date",
            )

        # Check with nan values

        nan_df = input_contracts_df.copy()
        nan_df.loc[1, "contracts_count"] = np.nan

        with self.assertRaises(ValueError):
            Contracts.compute_portfolio_by_day(
                contracts=nan_df,
                columns_to_sum=["contracts_count", "subscribed_kva"],
                date_start_col="start_date",
                date_end_exclusive_col="excl_end_date",
            )

        # Check when ffill_until_max_date is True but no max_date_exclusive is given

        with self.assertRaises(ValueError):
            Contracts.compute_portfolio_by_day(
                contracts=input_contracts_df,
                columns_to_sum=["contracts_count", "subscribed_kva"],
                date_start_col="start_date",
                date_end_exclusive_col="excl_end_date",
                ffill_until_max_date=True,
            )

        # Check the result with no max_date_exclusive

        expected_output_df = pd.DataFrame(
            data=[
                {"contracts_count": 2.0, "subscribed_kva": 16.0},
                {"contracts_count": 3, "subscribed_kva": 20},
                {"contracts_count": 3, "subscribed_kva": 25},
                {"contracts_count": 2, "subscribed_kva": 21},
            ]
            + [{"contracts_count": 1, "subscribed_kva": 6} for _ in range(27)]
            + [{"contracts_count": 0, "subscribed_kva": 0}],
            index=pd.date_range(
                start=pd.Timestamp(2023, 1, 1),
                end=pd.Timestamp(2023, 2, 1),
                freq="D",
            ),
        )

        expected_output_df.index.name = "date"

        output_df = Contracts.compute_portfolio_by_day(
            contracts=input_contracts_df,
            columns_to_sum=["contracts_count", "subscribed_kva"],
            date_start_col="start_date",
            date_end_exclusive_col="excl_end_date",
        )

        pd.testing.assert_frame_equal(output_df, expected_output_df)

        # Check the result with max_date_exclusive

        expected_output_maxdate_df = expected_output_df.loc[
            expected_output_df.index <= pd.Timestamp(2023, 1, 5)
        ].astype(int)
        max_date_exclusive = pd.Timestamp(2023, 1, 10)

        output_maxdate_df = Contracts.compute_portfolio_by_day(
            contracts=input_contracts_df,
            columns_to_sum=["contracts_count", "subscribed_kva"],
            date_start_col="start_date",
            date_end_exclusive_col="excl_end_date",
            max_date_exclusive=max_date_exclusive,
        )

        pd.testing.assert_frame_equal(output_maxdate_df, expected_output_maxdate_df)

        # Check the result with ffill_until_max_date

        expected_output_ffill_df = expected_output_df.loc[
            expected_output_df.index <= pd.Timestamp(2023, 1, 9)
        ]

        output_ffill_df = Contracts.compute_portfolio_by_day(
            contracts=input_contracts_df,
            columns_to_sum=["contracts_count", "subscribed_kva"],
            date_start_col="start_date",
            date_end_exclusive_col="excl_end_date",
            max_date_exclusive=max_date_exclusive,
            ffill_until_max_date=True,
        )

        pd.testing.assert_frame_equal(output_ffill_df, expected_output_ffill_df)

    def test_get_portfolio_between_dates(self):
        """
        Test the get_portfolio_between_dates function
        """

        input_portfolio_df = pd.DataFrame(
            data=[
                {"total_power_kw": 1500, "stations_count": 35},
                {"total_power_kw": 1480, "stations_count": 33},
                {"total_power_kw": 1470, "stations_count": 32},
            ],
            index=[
                pd.Timestamp(2023, 1, 1),
                pd.Timestamp(2023, 1, 2),
                pd.Timestamp(2023, 1, 3),
            ],
        )
        input_portfolio_df.index.name = "day"
        input_portfolio_df.index.freq = "D"

        # Check that providing a DataFrame without a DatetimeIndex raises an error

        no_dti_df = input_portfolio_df.copy()
        no_dti_df = no_dti_df.reset_index()

        with self.assertRaises(TypeError):
            Contracts.get_portfolio_between_dates(
                portfolio=no_dti_df,
                start_datetime=pd.Timestamp(2023, 1, 1),
                end_datetime_exclusive=pd.Timestamp(2023, 1, 5),
            )

        # Check with a start datetime before the first date in the DataFrame and an exclusive end datetime after the
        # final date in the DataFrame

        expected_output_df = pd.DataFrame(
            data=[
                {"total_power_kw": 0, "stations_count": 0},
                {"total_power_kw": 0, "stations_count": 0},
                {"total_power_kw": 1500, "stations_count": 35},
                {"total_power_kw": 1480, "stations_count": 33},
                {"total_power_kw": 1470, "stations_count": 32},
                {"total_power_kw": 1470, "stations_count": 32},
            ],
            index=pd.date_range(
                start=pd.Timestamp(2022, 12, 30),
                end=pd.Timestamp(2023, 1, 5),
                freq="D",
                inclusive="left",
            ),
        )
        expected_output_df.index.name = "day"

        output_df = Contracts.get_portfolio_between_dates(
            portfolio=input_portfolio_df,
            start_datetime=pd.Timestamp(2022, 12, 30),
            end_datetime_exclusive=pd.Timestamp(2023, 1, 5),
        )

        pd.testing.assert_frame_equal(output_df, expected_output_df)

    @staticmethod
    def get_simple_portfolio_by_day():
        """Reads a simple portfolio file and converts it to daily data. This is used by other testing functions """
        contracts = Contracts.read_contracts_from_file(TestContracts.CONTRACTS_PATH)
        contracts[
            "contracts_count"
        ] = 1  # add a variable to count the number of contracts for each row

        # count the running total, each day, of some columns
        portfolio_by_day = Contracts.compute_portfolio_by_day(
            contracts,
            columns_to_sum=[
                "contracts_count",
                "subscribed_power_kva",
                "estimated_annual_consumption_kwh",
            ],
            date_start_col="date_start",
            date_end_exclusive_col="date_end_exclusive",
        )

        return portfolio_by_day

    def test_forecast_portfolio_linear_1(self):
        """Test on a portfolio_by_day"""

        portfolio_by_day = TestContracts.get_simple_portfolio_by_day()

        # print(portfolio_by_day['subscribed_power_kva'])

        # linear forecast using the 11 days gives an increasing trend
        forecast_by_day_a = Contracts.forecast_portfolio_linear(
            portfolio_by_day,
            start_forecast_date=pd.to_datetime("2020-09-27"),
            end_forecast_date_exclusive=pd.to_datetime("2020-09-30"),
            freq="D",
        )

        # print(forecast_by_day_a['subscribed_power_kva'])

        self.assertEqual((3, 3), forecast_by_day_a.shape)
        self.assertEqual(pd.to_datetime("2020-09-29"), forecast_by_day_a.index.max())
        self.assertGreaterEqual(
            forecast_by_day_a.loc["2020-09-27", "subscribed_power_kva"], 40
        )
        self.assertGreaterEqual(
            forecast_by_day_a.loc["2020-09-29", "subscribed_power_kva"], 40
        )

        # linear forecast using only the last 7 days gives a decreasing trend
        forecast_by_day_b = Contracts.forecast_portfolio_linear(
            portfolio_by_day[portfolio_by_day.index >= "2020-09-20"],
            start_forecast_date=pd.to_datetime("2020-09-27"),
            end_forecast_date_exclusive=pd.to_datetime("2020-10-02"),
            freq="D",
        )

        # print(forecast_by_day_b['subscribed_power_kva'])

        self.assertEqual((5, 3), forecast_by_day_b.shape)
        self.assertEqual(pd.to_datetime("2020-10-01"), forecast_by_day_b.index.max())
        self.assertLessEqual(
            forecast_by_day_b.loc["2020-09-27", "subscribed_power_kva"], 40
        )
        self.assertLessEqual(
            forecast_by_day_b.loc["2020-09-29", "subscribed_power_kva"], 40
        )

    def test_forecast_portfolio_linear_2(self):
        """Test on a portfolio at freq=7min"""

        portfolio_by_day = TestContracts.get_simple_portfolio_by_day()
        portfolio_by_20min = TimeSeries.interpolate_daily_to_sub_daily_data(
            portfolio_by_day, freq="20min", tz="Europe/Paris"
        )

        # print(portfolio_by_20min)

        # linear forecast_by_10min, give it a portfolio_by_20min to train
        forecast_by_10min = Contracts.forecast_portfolio_linear(
            portfolio_by_20min,
            start_forecast_date=pd.to_datetime("2020-09-27 00:00:00+02:00").tz_convert(
                "Europe/Paris"
            ),
            end_forecast_date_exclusive=pd.to_datetime(
                "2020-09-30 00:00:00+02:00"
            ).tz_convert("Europe/Paris"),
            freq="10min",
            tzinfo="Europe/Paris",
        )

        # print(forecast_by_10min)
        self.assertEqual((432, 3), forecast_by_10min.shape)
        self.assertEqual("Europe/Paris", str(forecast_by_10min.index.tzinfo))
        self.assertGreaterEqual(
            forecast_by_10min.loc["2020-09-27 00:00:00+02:00", "subscribed_power_kva"],
            40,
        )
        self.assertGreaterEqual(
            forecast_by_10min.loc["2020-09-29 00:00:00+02:00", "subscribed_power_kva"],
            40,
        )

    def test_forecast_portfolio_holt_1(self):
        """Test on a portfolio_by_day"""

        portfolio_by_day = TestContracts.get_simple_portfolio_by_day()

        # print(portfolio_by_day)

        forecast_by_day = Contracts.forecast_portfolio_holt(
            portfolio_by_day,
            start_forecast_date=pd.to_datetime("2020-09-27"),
            nb_days=3,
            past_days=10,  # looking at 10 days : expect increasing trend
        )

        # print(forecast_by_day)
        self.assertEqual((3, 3), forecast_by_day.shape)
        self.assertEqual(pd.to_datetime("2020-09-29"), forecast_by_day.index.max())
        self.assertLessEqual(
            38, forecast_by_day.loc["2020-09-27", "subscribed_power_kva"]
        )
        self.assertGreaterEqual(
            40, forecast_by_day.loc["2020-09-29", "subscribed_power_kva"]
        )

    def test_forecast_portfolio_holt_2(self):
        """Test on a portfolio at freq=5min"""

        portfolio_by_day = TestContracts.get_simple_portfolio_by_day()
        portfolio_5min = TimeSeries.interpolate_daily_to_sub_daily_data(
            portfolio_by_day, freq="5min", tz="Europe/Paris"
        )
        # print(portfolio_5min[['subscribed_power_kva']][portfolio_5min.index >= '2020-09-19 00:00:00+02:00'])

        forecast_5_min = Contracts.forecast_portfolio_holt(
            portfolio_5min,
            start_forecast_date=pd.to_datetime("2020-09-27 00:00:00+02:00").tz_convert(
                "Europe/Paris"
            ),
            nb_days=5,
            past_days=7,  # if we look only at the last 7 days, the trend is decreasing (10 days would be increasing),
            holt_init_params={
                "exponential": True,
                "damped_trend": True,
                "initialization_method": "estimated",
            },
            holt_fit_params={"damping_trend": 0.98},
        )
        # print(forecast_5_min[['subscribed_power_kva']])

        self.assertEqual((12 * 24 * 5, 3), forecast_5_min.shape)
        self.assertEqual(
            pd.to_datetime("2020-10-01 23:55:00+02:00"), forecast_5_min.index.max()
        )
        self.assertGreaterEqual(
            30, forecast_5_min.loc["2020-09-27 00:00:00+02:00", "subscribed_power_kva"]
        )
        self.assertGreaterEqual(
            30, forecast_5_min.loc["2020-10-01 23:55:00+02:00", "subscribed_power_kva"]
        )
