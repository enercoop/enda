"""A module for testing the PortfolioTools class in enda/tools/portfolio_tools.py"""

import logging
import unittest
import numpy as np
import pandas as pd

from enda.tools.portfolio_tools import PortfolioTools


class TestPortfolioTools(unittest.TestCase):
    """
    This class aims at testing the functions of the PortfolioTools class in enda/tools/portfolio_tools.py
    """

    def setUp(self):
        logging.captureWarnings(True)
        logging.disable(logging.ERROR)

    def tearDown(self):
        logging.captureWarnings(False)
        logging.disable(logging.NOTSET)

    def test_portfolio_to_events(self):
        """Test the _portfolio_to_events function"""

        # Check that having 'event_type' or 'event_date' as a column in the original DataFrame raises an error

        wrong_input_df = pd.DataFrame(
            data=[
                {
                    "station": "station1",
                    "start_date": pd.Timestamp(2023, 1, 1),
                    "excl_end_date": pd.Timestamp(2024, 1, 1),
                    "power_installed_kw": 100,
                    "event_type": "event1",
                },
                {
                    "station": "station1",
                    "start_date": pd.Timestamp(2024, 1, 1),
                    "excl_end_date": pd.Timestamp(2024, 6, 1),
                    "power_installed_kw": 100,
                    "event_type": "event2",
                },
            ]
        )

        with self.assertRaises(ValueError):
            PortfolioTools.portfolio_to_events(
                portfolio_df=wrong_input_df,
                date_start_col="start_date",
                date_end_exclusive_col="excl_end_date",
            )

        # Check with normal case

        input_df = pd.DataFrame(
            data=[
                {
                    "station": "station1",
                    "start_date": pd.Timestamp(2023, 1, 1),
                    "excl_end_date": pd.Timestamp(2024, 1, 1),
                    "power_installed_kw": 100,
                },
                {
                    "station": "station1",
                    "start_date": pd.Timestamp(2024, 1, 1),
                    "excl_end_date": pd.Timestamp(2024, 6, 1),
                    "power_installed_kw": 100,
                },
                {
                    "station": "station1",
                    "start_date": pd.Timestamp(2024, 6, 1),
                    "excl_end_date": pd.NaT,
                    "power_installed_kw": 100,
                },
                {
                    "station": "station2",
                    "start_date": pd.Timestamp(2023, 6, 1),
                    "excl_end_date": pd.Timestamp(2023, 7, 1),
                    "power_installed_kw": 80,
                },
                {
                    "station": "station2",
                    "start_date": pd.Timestamp(2023, 8, 1),
                    "excl_end_date": pd.Timestamp(2023, 9, 1),
                    "power_installed_kw": 50,
                },
            ]
        )

        expected_output_df = pd.DataFrame(
            data=[
                {
                    "event_type": "start",
                    "event_date": pd.Timestamp(2023, 1, 1),
                    "station": "station1",
                    "power_installed_kw": 100,
                },
                {
                    "event_type": "start",
                    "event_date": pd.Timestamp(2023, 6, 1),
                    "station": "station2",
                    "power_installed_kw": 80,
                },
                {
                    "event_type": "end",
                    "event_date": pd.Timestamp(2023, 7, 1),
                    "station": "station2",
                    "power_installed_kw": 80,
                },
                {
                    "event_type": "start",
                    "event_date": pd.Timestamp(2023, 8, 1),
                    "station": "station2",
                    "power_installed_kw": 50,
                },
                {
                    "event_type": "end",
                    "event_date": pd.Timestamp(2023, 9, 1),
                    "station": "station2",
                    "power_installed_kw": 50,
                },
                {
                    "event_type": "end",
                    "event_date": pd.Timestamp(2024, 1, 1),
                    "station": "station1",
                    "power_installed_kw": 100,
                },
                {
                    "event_type": "start",
                    "event_date": pd.Timestamp(2024, 1, 1),
                    "station": "station1",
                    "power_installed_kw": 100,
                },
                {
                    "event_type": "end",
                    "event_date": pd.Timestamp(2024, 6, 1),
                    "station": "station1",
                    "power_installed_kw": 100,
                },
                {
                    "event_type": "start",
                    "event_date": pd.Timestamp(2024, 6, 1),
                    "station": "station1",
                    "power_installed_kw": 100,
                },
            ]
        )

        output_df = PortfolioTools.portfolio_to_events(
            portfolio_df=input_df,
            date_start_col="start_date",
            date_end_exclusive_col="excl_end_date",
        )

        pd.testing.assert_frame_equal(
            output_df.reset_index(drop=True), expected_output_df.reset_index(drop=True)
        )

    def test_get_portfolio_between_dates_single_index(self):
        """
        Test the get_portfolio_between_dates function with single-indexed DataFrames
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

        # Check that providing a DataFrame without a DatetimeIndex raises an error

        no_dti_df = input_portfolio_df.copy()
        no_dti_df = no_dti_df.reset_index()

        with self.assertRaises(TypeError):
            PortfolioTools.get_portfolio_between_dates(
                portfolio_df=no_dti_df,
                start_datetime=pd.Timestamp(2023, 1, 1),
                end_datetime_exclusive=pd.Timestamp(2023, 1, 5),
            )

        # Check that providing a DataFrame with no inferrable frequency raises an error

        wrong_index_df = input_portfolio_df.copy()
        wrong_index_df.index = [
            pd.Timestamp(2023, 1, 1),
            pd.Timestamp(2023, 1, 2),
            pd.Timestamp(2023, 1, 4),
        ]

        with self.assertRaises(ValueError):
            PortfolioTools.get_portfolio_between_dates(
                portfolio_df=wrong_index_df,
                start_datetime=pd.Timestamp(2023, 1, 1),
                end_datetime_exclusive=pd.Timestamp(2023, 1, 5),
            )

        # Check that having nan values raises an error

        nan_df = input_portfolio_df.copy()
        nan_df.loc[pd.Timestamp(2023, 1, 2), "stations_count"] = np.nan

        with self.assertRaises(ValueError):
            PortfolioTools.get_portfolio_between_dates(
                portfolio_df=nan_df,
                start_datetime=pd.Timestamp(2023, 1, 1),
                end_datetime_exclusive=pd.Timestamp(2023, 1, 5),
            )

        # Check with a start datetime before the first date in the DataFrame and an exclusive end datetime after the
        # final date in the DataFrame

        expected_output_df = pd.DataFrame(
            data=[
                {"total_power_kw": 0.0, "stations_count": 0.0},
                {"total_power_kw": 0.0, "stations_count": 0.0},
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

        output_df = PortfolioTools.get_portfolio_between_dates(
            portfolio_df=input_portfolio_df,
            start_datetime=pd.Timestamp(2022, 12, 30),
            end_datetime_exclusive=pd.Timestamp(2023, 1, 5),
        )

        pd.testing.assert_frame_equal(output_df, expected_output_df)

    def test_get_portfolio_between_dates_multi_index(self):
        """
        Test the get_portfolio_between_dates function with multi-indexed DataFrames
        """

        dt1 = pd.date_range(
            start=pd.Timestamp(2023, 1, 1),
            end=pd.Timestamp(2023, 1, 2),
            freq="1H",
            inclusive="left",
        )

        dt2 = pd.date_range(
            start=pd.Timestamp(2023, 1, 2),
            end=pd.Timestamp(2023, 1, 3),
            freq="1H",
            inclusive="left",
        )

        dt3 = pd.date_range(
            start=pd.Timestamp(2023, 1, 2),
            end=pd.Timestamp(2023, 1, 5),
            freq="1H",
            inclusive="left",
        )

        input_portfolio_df = pd.DataFrame(
            data=[
                {"station": "station1", "time": x, "power_installed_kw": 30}
                for x in dt2
            ]
            + [
                {"station": "station2", "time": x, "power_installed_kw": 100}
                for x in dt2
            ]
        )

        input_portfolio_df = input_portfolio_df.set_index(["station", "time"])

        # Check that providing a DataFrame without a DatetimeIndex as second index raises an error

        no_dti_df = input_portfolio_df.copy()
        no_dti_df = no_dti_df.reset_index(level="time").set_index(
            "power_installed_kw", append=True
        )

        with self.assertRaises(TypeError):
            PortfolioTools.get_portfolio_between_dates(
                portfolio_df=no_dti_df,
                start_datetime=pd.Timestamp(2023, 1, 1),
                end_datetime_exclusive=pd.Timestamp(2023, 1, 5),
            )

        wrong_index_order_df = input_portfolio_df.copy()
        wrong_index_order_df = wrong_index_order_df.swaplevel()

        with self.assertRaises(TypeError):
            PortfolioTools.get_portfolio_between_dates(
                portfolio_df=wrong_index_order_df,
                start_datetime=pd.Timestamp(2023, 1, 1),
                end_datetime_exclusive=pd.Timestamp(2023, 1, 5),
            )

        # Check with a start datetime before the first date in the DataFrame and an exclusive end datetime after the
        # final date in the DataFrame

        expected_output_df = pd.DataFrame(
            data=[
                {"station": "station1", "time": x, "power_installed_kw": 0.0}
                for x in dt1
            ]
            + [
                {"station": "station1", "time": x, "power_installed_kw": 30}
                for x in dt3
            ]
            + [{"station": "station2", "time": x, "power_installed_kw": 0} for x in dt1]
            + [
                {"station": "station2", "time": x, "power_installed_kw": 100}
                for x in dt3
            ]
        )

        expected_output_df = expected_output_df.set_index(["station", "time"])

        output_df = PortfolioTools.get_portfolio_between_dates(
            portfolio_df=input_portfolio_df,
            start_datetime=pd.Timestamp(2023, 1, 1),
            end_datetime_exclusive=pd.Timestamp(2023, 1, 5),
        )

        pd.testing.assert_frame_equal(output_df, expected_output_df)
