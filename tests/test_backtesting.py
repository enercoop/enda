"""A module for testing the Backtesting class in enda/backtesting.py"""

import logging
import unittest
import pandas as pd
import numpy as np

from enda.backtesting import BackTesting


class TestBackTesting(unittest.TestCase):
    """
    This class aims at testing the functions of the Backtesting class in enda/backtesting.py
    """

    def setUp(self):
        logging.disable(logging.INFO)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_error_raises_yield_train_test(self):
        """
        Checks all cases where yield_train_test should raise an error
        """

        dummy_single_index_df = pd.DataFrame(
            data=[1, 2],
            columns=["col1"],
            index=[
                pd.Timestamp(year=2023, month=1, day=1, tz="Europe/Paris"),
                pd.Timestamp(year=2023, month=1, day=2, tz="Europe/Paris"),
            ],
        )

        # Check that datetime is not more precise than days

        dummy_dt = pd.Timestamp(year=2023, month=1, day=1, hour=3, tz="Europe/Paris")

        with self.assertRaises(ValueError):
            for _, _ in BackTesting.yield_train_test(
                df=dummy_single_index_df,
                start_eval_datetime=dummy_dt,
                days_between_trains=5,
            ):
                pass

        # -- Single indexed DataFrame

        # Check that start_eval_datetime and df have the same time zone

        naive_dt = pd.Timestamp(2023, 1, 2)

        with self.assertRaises(ValueError):
            for _, _ in BackTesting.yield_train_test(
                df=dummy_single_index_df,
                start_eval_datetime=naive_dt,
                days_between_trains=5,
            ):
                pass

        # Check that start_eval_datetime is not before the first timestamp in the DataFrame

        early_dt = pd.Timestamp(year=2022, month=12, day=31, tz="Europe/Paris")

        with self.assertRaises(ValueError):
            for _, _ in BackTesting.yield_train_test(
                df=dummy_single_index_df,
                start_eval_datetime=early_dt,
                days_between_trains=5,
            ):
                pass

        # -- Multi-indexed DataFrame

        # Test that 3 level multi-indexed DataFrame raises an error

        three_level_df = pd.DataFrame(
            data=[1, 2],
            columns=["col1"],
            index=[
                np.array([0, 1]),
                np.array(
                    [
                        pd.Timestamp(year=2023, month=1, day=1, tz="Europe/Paris"),
                        pd.Timestamp(year=2023, month=1, day=2, tz="Europe/Paris"),
                    ]
                ),
                np.array(["station1", "station1"]),
            ],
        )

        start_dt = pd.Timestamp(year=2023, month=1, day=2, tz="Europe/Paris")

        with self.assertRaises(TypeError):
            for _, _ in BackTesting.yield_train_test(
                df=three_level_df, start_eval_datetime=start_dt, days_between_trains=5
            ):
                pass

        # Test that an error is raised if the second level index is not a DatetimeIndex

        wrong_order_df = pd.DataFrame(
            data=[1, 2],
            columns=["col1"],
            index=[
                np.array(
                    [
                        pd.Timestamp(year=2023, month=1, day=1, tz="Europe/Paris"),
                        pd.Timestamp(year=2023, month=1, day=2, tz="Europe/Paris"),
                    ]
                ),
                np.array(["station1", "station1"]),
            ],
        )

        with self.assertRaises(TypeError):
            for _, _ in BackTesting.yield_train_test(
                df=wrong_order_df, start_eval_datetime=start_dt, days_between_trains=5
            ):
                pass

        # Check that start_eval_datetime and df have the same time zone

        dummy_multiindex_df = pd.DataFrame(
            data=[1, 2],
            columns=["col1"],
            index=[
                np.array(["station1", "station1"]),
                np.array(
                    [
                        pd.Timestamp(year=2023, month=1, day=1, tz="Europe/Paris"),
                        pd.Timestamp(year=2023, month=1, day=2, tz="Europe/Paris"),
                    ]
                ),
            ],
        )

        dummy_multiindex_df.index.names = ["index1", "index2"]

        wrong_tz_dt = pd.Timestamp(year=2023, month=1, day=2, tz="UTC")

        with self.assertRaises(ValueError):
            for _, _ in BackTesting.yield_train_test(
                df=dummy_multiindex_df,
                start_eval_datetime=wrong_tz_dt,
                days_between_trains=5,
            ):
                pass

        # Check that start_eval_datetime is not before the first timestamp in the DataFrame

        early_dt = pd.Timestamp(year=2022, month=12, day=31, tz="Europe/Paris")

        with self.assertRaises(ValueError):
            for _, _ in BackTesting.yield_train_test(
                df=dummy_multiindex_df,
                start_eval_datetime=early_dt,
                days_between_trains=5,
            ):
                pass

        # Check that the DataFrame has a DatetimeIndex

        no_dt_df = pd.DataFrame(data=[1, 2], columns=["col1"], index=[0, 1])

        start_dt = pd.Timestamp(2023, 1, 2)

        with self.assertRaises(ValueError):
            for _, _ in BackTesting.yield_train_test(
                df=no_dt_df,
                start_eval_datetime=start_dt,
                days_between_trains=5,
            ):
                pass

    def test_yield_train_test_single_index(self):
        """
        Test the yield_train_test function on a single indexed DataFrame, with gap_days_between_train_and_eval
        parameter equal to 0 and daily frequency
        """
        input_df = pd.date_range(
            start=pd.Timestamp(year=2015, month=1, day=1, tz="Europe/Paris"),
            end=pd.Timestamp(year=2021, month=1, day=1, tz="Europe/Paris"),
            freq="D",
            tz="Europe/Paris",
            name="time",
        ).to_frame()
        input_df = input_df.set_index("time")
        input_df["value"] = 1

        start_eval_datetime = pd.Timestamp(year=2019, month=1, day=1, tz="Europe/Paris")

        expected_output_train_df = pd.date_range(
            start=pd.Timestamp(year=2015, month=1, day=1, tz="Europe/Paris"),
            end=pd.Timestamp(year=2019, month=2, day=25, tz="Europe/Paris"),
            freq="D",
            tz="Europe/Paris",
            name="time",
        ).to_frame()
        expected_output_train_df = expected_output_train_df.set_index("time")
        expected_output_train_df["value"] = 1

        expected_output_test_df = pd.date_range(
            start=pd.Timestamp(year=2019, month=2, day=26, tz="Europe/Paris"),
            end=pd.Timestamp(year=2019, month=3, day=4, tz="Europe/Paris"),
            freq="D",
            tz="Europe/Paris",
            name="time",
        ).to_frame()
        expected_output_test_df = expected_output_test_df.set_index("time")
        expected_output_test_df["value"] = 1

        train_df_list, test_df_list = [], []

        for train_set, test_set in BackTesting.yield_train_test(
            input_df, start_eval_datetime=start_eval_datetime, days_between_trains=7
        ):
            train_df_list.append(train_set)
            test_df_list.append(test_set)

        self.assertEqual(len(train_df_list), 105)
        self.assertEqual(len(test_df_list), 105)

        output_train_df = train_df_list[8]
        output_test_df = test_df_list[8]

        pd.testing.assert_frame_equal(output_train_df, expected_output_train_df)
        pd.testing.assert_frame_equal(output_test_df, expected_output_test_df)

    def test_yield_train_test_multi_index(self):
        """
        Test the yield_train_test function on a multi-indexed DataFrame, with gap_days_between_train_and_eval
        parameter different from 0 and hourly frequency
        """
        input_range = pd.date_range(
            start=pd.Timestamp(year=2019, month=1, day=1, tz="Europe/Paris"),
            end=pd.Timestamp(year=2026, month=1, day=1, tz="Europe/Paris"),
            freq="H",
            tz="Europe/Paris",
            inclusive="left",
        )

        input_df = pd.DataFrame(
            index=[np.array(range(len(input_range))), np.array(input_range)]
        )
        input_df["value"] = 1
        input_df.index.names = ["col1", "time"]

        start_eval_datetime = pd.Timestamp(
            year=2023, month=11, day=1, tz="Europe/Paris"
        )

        train_output_range = pd.date_range(
            start=pd.Timestamp(year=2019, month=1, day=1, tz="Europe/Paris"),
            end=pd.Timestamp(year=2024, month=6, day=14, tz="Europe/Paris"),
            freq="H",
            tz="Europe/Paris",
            inclusive="left",
        )

        expected_output_train_df = pd.DataFrame(
            index=[np.array(range(47783)), np.array(train_output_range)]
        )
        expected_output_train_df["value"] = 1
        expected_output_train_df.index.names = ["col1", "time"]

        test_output_range = pd.date_range(
            start=pd.Timestamp(year=2024, month=6, day=28, tz="Europe/Paris"),
            end=pd.Timestamp(year=2024, month=7, day=8, tz="Europe/Paris"),
            freq="H",
            tz="Europe/Paris",
            inclusive="left",
        )

        expected_output_test_df = pd.DataFrame(
            index=[np.array(range(48119, 48359)), np.array(test_output_range)]
        )
        expected_output_test_df["value"] = 1
        expected_output_test_df.index.names = ["col1", "time"]

        train_df_list, test_df_list = [], []

        for train_set, test_set in BackTesting.yield_train_test(
            input_df,
            start_eval_datetime=start_eval_datetime,
            days_between_trains=10,
            gap_days_between_train_and_eval=14,
        ):
            train_df_list.append(train_set)
            test_df_list.append(test_set)

        self.assertEqual(len(train_df_list), 80)
        self.assertEqual(len(test_df_list), 80)

        output_train_df = train_df_list[24]
        output_test_df = test_df_list[24]

        pd.testing.assert_frame_equal(output_train_df, expected_output_train_df)
        pd.testing.assert_frame_equal(output_test_df, expected_output_test_df)
