"""A module for testing the DatetimeFeatures class in enda/feature_engineering/datetime_features.py"""

import logging
import unittest
import pandas as pd

from enda.feature_engineering.datetime_features import DatetimeFeature


class TestDatetimeFeatures(unittest.TestCase):
    """
    This class aims at testing the functions of the DatetimeFeatures class in
        enda/feature_engineering/datetime_features.py
    """

    def setUp(self):
        logging.disable(logging.INFO)

        self.timeseries_df = pd.DataFrame(
            data=[
                {"col1": 1},
                {"col1": 2},
                {"col1": 3},
                {"col1": 4},
                {"col1": 5},
                {"col1": 6},
            ],
            index=[
                pd.Timestamp(year=2023, month=1, day=1, tz="Europe/Paris"),
                pd.Timestamp(year=2023, month=2, day=1, tz="Europe/Paris"),
                pd.Timestamp(year=2023, month=2, day=1, hour=12, tz="Europe/Paris"),
                pd.Timestamp(
                    year=2023, month=2, day=15, hour=15, minute=20, tz="Europe/Paris"
                ),
                pd.Timestamp(
                    year=2023, month=10, day=13, hour=20, minute=44, tz="Europe/Paris"
                ),
                pd.Timestamp(year=2024, month=4, day=1, tz="Europe/Paris"),
            ],
        )

    def tearDown(self):
        logging.captureWarnings(False)
        logging.disable(logging.NOTSET)

    def test_get_nb_hours_in_day(self):
        """
        Test the get_nb_hours_in_day function
        """

        # Test that we raise an error for naive Timestamps
        naive_dt = pd.Timestamp(2023)
        with self.assertRaises(AttributeError):
            DatetimeFeature.get_nb_hours_in_day(naive_dt)

        # Check for a normal day
        normal_dt = pd.Timestamp(year=2023, month=12, day=23, tz="Europe/Paris")
        nb_hours = DatetimeFeature.get_nb_hours_in_day(normal_dt)
        self.assertEqual(nb_hours, 24)

        # Check for daylight saving day
        dst_dt = pd.Timestamp(year=2014, month=3, day=30, tz="Europe/Paris")
        nb_hours = DatetimeFeature.get_nb_hours_in_day(dst_dt)
        self.assertEqual(nb_hours, 23)

        # Check for DST on other time zone
        us_dst_dt = pd.Timestamp(year=2023, month=11, day=5, tz="America/New_York")
        nb_hours = DatetimeFeature.get_nb_hours_in_day(us_dst_dt)
        self.assertEqual(nb_hours, 25)

    def test_split_datetime(self):
        """
        Test the split_datetime function
        """
        split_list = [
            "minute",
            "minuteofday",
            "hour",
            "day",
            "month",
            "year",
            "dayofweek",
            "weekofyear",
            "dayofyear",
        ]

        # Check that no column or index specified raises a ValueError

        with self.assertRaises(ValueError):
            DatetimeFeature.split_datetime(
                df=self.timeseries_df, split_list=split_list, index=False
            )

        # Check that specifying index=True AND a colname raises a ValueError

        with self.assertRaises(ValueError):
            DatetimeFeature.split_datetime(
                df=self.timeseries_df, split_list=split_list, index=True, colname="col1"
            )

        # Check that specifying an unexpected split raises a NotImplementedError

        with self.assertRaises(NotImplementedError):
            DatetimeFeature.split_datetime(
                df=self.timeseries_df, split_list=["minute", "minuteofyear"], index=True
            )

        # Check that specifying a column not present in the DataFrames raises an AttributeError

        with self.assertRaises(AttributeError):
            DatetimeFeature.split_datetime(
                df=self.timeseries_df,
                split_list=split_list,
                index=False,
                colname="wrong_colname",
            )

        # Check that specifying a column with wrong type (not datetimes) raises a TypeError

        with self.assertRaises(TypeError):
            DatetimeFeature.split_datetime(
                df=self.timeseries_df,
                split_list=split_list,
                index=False,
                colname="col1",
            )

        # Check that passing an index which isn't a DatetimeIndex raises a TypeError

        dummy_df = pd.DataFrame(index=[1, 2, 3])

        with self.assertRaises(TypeError):
            DatetimeFeature.split_datetime(
                df=dummy_df, split_list=split_list, index=True
            )

        # Check that the result is correct when splitting an index

        expected_output_df = pd.DataFrame(
            data=[
                {
                    "col1": 1,
                    "minute": 0,
                    "minuteofday": 0,
                    "hour": 0,
                    "day": 1,
                    "month": 1,
                    "year": 2023,
                    "dayofweek": 6,
                    "weekofyear": 52,
                    "dayofyear": 1,
                },
                {
                    "col1": 2,
                    "minute": 0,
                    "minuteofday": 0,
                    "hour": 0,
                    "day": 1,
                    "month": 2,
                    "year": 2023,
                    "dayofweek": 2,
                    "weekofyear": 5,
                    "dayofyear": 32,
                },
                {
                    "col1": 3,
                    "minute": 0,
                    "minuteofday": 720,
                    "hour": 12,
                    "day": 1,
                    "month": 2,
                    "year": 2023,
                    "dayofweek": 2,
                    "weekofyear": 5,
                    "dayofyear": 32,
                },
                {
                    "col1": 4,
                    "minute": 20,
                    "minuteofday": 920,
                    "hour": 15,
                    "day": 15,
                    "month": 2,
                    "year": 2023,
                    "dayofweek": 2,
                    "weekofyear": 7,
                    "dayofyear": 46,
                },
                {
                    "col1": 5,
                    "minute": 44,
                    "minuteofday": 1244,
                    "hour": 20,
                    "day": 13,
                    "month": 10,
                    "year": 2023,
                    "dayofweek": 4,
                    "weekofyear": 41,
                    "dayofyear": 286,
                },
                {
                    "col1": 6,
                    "minute": 0,
                    "minuteofday": 0,
                    "hour": 0,
                    "day": 1,
                    "month": 4,
                    "year": 2024,
                    "dayofweek": 0,
                    "weekofyear": 14,
                    "dayofyear": 92,
                },
            ],
            index=self.timeseries_df.index,
        )

        output_df = DatetimeFeature.split_datetime(
            df=self.timeseries_df, split_list=split_list, index=True
        )

        output_df = output_df.astype("int64")

        pd.testing.assert_frame_equal(output_df, expected_output_df)

        # Same check when splitting a column

        input_df = self.timeseries_df.copy()
        input_df = (
            input_df.reset_index().set_index("col1").rename(columns={"index": "time"})
        )

        expected_output_df = (
            expected_output_df.reset_index()
            .set_index("col1")
            .rename(columns={"index": "time"})
        )

        output_df = DatetimeFeature.split_datetime(
            df=input_df, split_list=split_list, index=False, colname="time"
        )

        output_df[split_list] = output_df[split_list].astype("int64")

        pd.testing.assert_frame_equal(output_df, expected_output_df)

    def test_encode_cyclic_datetime_index(self):
        """
        Test the encode_cyclic_datetime_index function
        """
        split_list = [
            "minute",
            "minuteofday",
            "hour",
            "day",
            "month",
            "dayofweek",
            "dayofyear",
        ]

        # Check that specifying an unexpected split raises a NotImplementedError

        with self.assertRaises(NotImplementedError):
            DatetimeFeature.encode_cyclic_datetime_index(
                df=self.timeseries_df, split_list=["minute", "minuteofyear"]
            )

        # Check that passing an index which isn't a DatetimeIndex raises a TypeError

        dummy_df = pd.DataFrame(index=[1, 2, 3])

        with self.assertRaises(TypeError):
            DatetimeFeature.encode_cyclic_datetime_index(
                df=dummy_df, split_list=split_list
            )

        # Check that returned result is correct

        expected_output_df = pd.DataFrame(
            data=[
                {
                    "col1": 1,
                    "minute_cos": 1,
                    "minute_sin": 0,
                    "minuteofday_cos": 1,
                    "minuteofday_sin": 0,
                    "hour_cos": 1,
                    "hour_sin": 0,
                    "day_cos": 1,
                    "day_sin": 0,
                    "month_cos": 1,
                    "month_sin": 0,
                    "dayofweek_cos": 0.623,
                    "dayofweek_sin": -0.782,
                    "dayofyear_cos": 1,
                    "dayofyear_sin": 0,
                },
                {
                    "col1": 2,
                    "minute_cos": 1,
                    "minute_sin": 0,
                    "minuteofday_cos": 1,
                    "minuteofday_sin": 0,
                    "hour_cos": 1,
                    "hour_sin": 0,
                    "day_cos": 1,
                    "day_sin": 0,
                    "month_cos": 0.866,
                    "month_sin": 0.5,
                    "dayofweek_cos": -0.223,
                    "dayofweek_sin": 0.975,
                    "dayofyear_cos": 0.861,
                    "dayofyear_sin": 0.509,
                },
                {
                    "col1": 3,
                    "minute_cos": 1,
                    "minute_sin": 0,
                    "minuteofday_cos": -1,
                    "minuteofday_sin": 0,
                    "hour_cos": -1,
                    "hour_sin": 0,
                    "day_cos": 1,
                    "day_sin": 0,
                    "month_cos": 0.866,
                    "month_sin": 0.5,
                    "dayofweek_cos": -0.223,
                    "dayofweek_sin": 0.975,
                    "dayofyear_cos": 0.861,
                    "dayofyear_sin": 0.509,
                },
                {
                    "col1": 4,
                    "minute_cos": -0.5,
                    "minute_sin": 0.866,
                    "minuteofday_cos": -0.643,
                    "minuteofday_sin": -0.766,
                    "hour_cos": -0.707,
                    "hour_sin": -0.707,
                    "day_cos": -1,
                    "day_sin": 0,
                    "month_cos": 0.866,
                    "month_sin": 0.5,
                    "dayofweek_cos": -0.223,
                    "dayofweek_sin": 0.975,
                    "dayofyear_cos": 0.715,
                    "dayofyear_sin": 0.699,
                },
                {
                    "col1": 5,
                    "minute_cos": -0.105,
                    "minute_sin": -0.995,
                    "minuteofday_cos": 0.656,
                    "minuteofday_sin": -0.755,
                    "hour_cos": 0.5,
                    "hour_sin": -0.866,
                    "day_cos": -0.759,
                    "day_sin": 0.651,
                    "month_cos": 0,
                    "month_sin": -1,
                    "dayofweek_cos": -0.901,
                    "dayofweek_sin": -0.434,
                    "dayofyear_cos": 0.192,
                    "dayofyear_sin": -0.981,
                },
                {
                    "col1": 6,
                    "minute_cos": 1,
                    "minute_sin": 0,
                    "minuteofday_cos": 1,
                    "minuteofday_sin": 0,
                    "hour_cos": 1,
                    "hour_sin": 0,
                    "day_cos": 1,
                    "day_sin": 0,
                    "month_cos": 0,
                    "month_sin": 1,
                    "dayofweek_cos": 1,
                    "dayofweek_sin": 0,
                    "dayofyear_cos": 0.009,
                    "dayofyear_sin": 1,
                },
            ],
            index=self.timeseries_df.index,
        )

        output_df = DatetimeFeature.encode_cyclic_datetime_index(
            df=self.timeseries_df, split_list=split_list
        )

        pd.testing.assert_frame_equal(
            output_df, expected_output_df, check_exact=False, atol=1e-3
        )

    def test_encode_cyclic_datetime(self):
        """
        Test the encode_cyclic_datetime function
        """

        # Test with naïve Timestamp, on regular year
        ts1 = pd.Timestamp(year=2023, month=2, day=15, hour=15, minute=20)

        expected_output_ts1_df = pd.DataFrame(
            data=[
                {
                    "minute_cos": -0.5,
                    "minute_sin": 0.866,
                    "minuteofday_cos": -0.643,
                    "minuteofday_sin": -0.766,
                    "hour_cos": -0.707,
                    "hour_sin": -0.707,
                    "day_cos": -1.0,
                    "day_sin": 0.0,
                    "month_cos": 0.866,
                    "month_sin": 0.5,
                    "dayofweek_cos": -0.223,
                    "dayofweek_sin": 0.975,
                    "dayofyear_cos": 0.715,
                    "dayofyear_sin": 0.699,
                },
            ],
            index=[ts1],
        )

        output_df = DatetimeFeature.encode_cyclic_datetime(ts1)

        pd.testing.assert_frame_equal(
            output_df, expected_output_ts1_df, check_exact=False, atol=1e-3
        )

        # Test with localized Timestamp on leap year
        ts2 = pd.Timestamp(year=2024, month=4, day=1, tz="Europe/Paris")

        expected_output_ts2_df = pd.DataFrame(
            data=[
                {
                    "minute_cos": 1.0,
                    "minute_sin": 0.0,
                    "minuteofday_cos": 1.0,
                    "minuteofday_sin": 0.0,
                    "hour_cos": 1.0,
                    "hour_sin": 0.0,
                    "day_cos": 1.0,
                    "day_sin": 0.0,
                    "month_cos": 0.0,
                    "month_sin": 1.0,
                    "dayofweek_cos": 1.0,
                    "dayofweek_sin": 0.0,
                    "dayofyear_cos": 0.009,
                    "dayofyear_sin": 1.0,
                },
            ],
            index=[ts2],
        )

        output_df = DatetimeFeature.encode_cyclic_datetime(ts2)

        pd.testing.assert_frame_equal(
            output_df, expected_output_ts2_df, check_exact=False, atol=1e-3
        )

    def test_get_datetime_features(self):
        """
        Test the get_datetime_features function. This is more an integration test as this function is just two
        successive calls to encode_cyclic_datetime_index and split_datetime
        """

        split_list = ["minute", "hour"]

        input_df = pd.DataFrame(index=[pd.Timestamp(2023, 1, 1)])

        expected_output_df = pd.DataFrame(
            data=[
                {
                    "minute_cos": 1.0,
                    "minute_sin": 0.0,
                    "hour_cos": 1.0,
                    "hour_sin": 0.0,
                    "minute": 0,
                    "hour": 0,
                }
            ],
            index=[pd.Timestamp(2023, 1, 1)],
        )

        output_df = DatetimeFeature.get_datetime_features(input_df, split_list)

        pd.testing.assert_frame_equal(output_df, expected_output_df)

    def test_daylight_saving_time_dates(self):
        """
        Test the daylight_saving_time_dates function
        """

        # Test on European timezone

        europe_expected_output_df = pd.DataFrame(
            data=[23, 25, 23, 25, 23, 25],
            columns=["nb_hour"],
            index=[
                pd.Timestamp(year=2027, month=3, day=28, tz="Europe/Paris"),
                pd.Timestamp(year=2027, month=10, day=31, tz="Europe/Paris"),
                pd.Timestamp(year=2028, month=3, day=26, tz="Europe/Paris"),
                pd.Timestamp(year=2028, month=10, day=29, tz="Europe/Paris"),
                pd.Timestamp(year=2029, month=3, day=25, tz="Europe/Paris"),
                pd.Timestamp(year=2029, month=10, day=28, tz="Europe/Paris"),
            ],
        )

        europe_output_df = DatetimeFeature.daylight_saving_time_dates(tz="Europe/Paris")

        # DataFrame should go from 1995 to 2029 (included), with 2 days per year
        self.assertEqual(len(europe_output_df), 70)
        pd.testing.assert_frame_equal(europe_output_df[-6:], europe_expected_output_df)

        # Test on US timezone

        us_expected_output_df = pd.DataFrame(
            data=[23, 25, 23, 25, 23, 25],
            columns=["nb_hour"],
            index=[
                pd.Timestamp(year=2027, month=3, day=14, tz="America/New_York"),
                pd.Timestamp(year=2027, month=11, day=7, tz="America/New_York"),
                pd.Timestamp(year=2028, month=3, day=12, tz="America/New_York"),
                pd.Timestamp(year=2028, month=11, day=5, tz="America/New_York"),
                pd.Timestamp(year=2029, month=3, day=11, tz="America/New_York"),
                pd.Timestamp(year=2029, month=11, day=4, tz="America/New_York"),
            ],
        )

        us_output_df = DatetimeFeature.daylight_saving_time_dates(tz="America/New_York")

        # DataFrame should go from 1995 to 2029 (included), with 2 days per year
        self.assertEqual(len(us_output_df), 70)
        pd.testing.assert_frame_equal(us_output_df[-6:], us_expected_output_df)
