"""A module for testing the Backtesting class in enda/backtesting.py"""

import logging
from random import randint
import unittest

import pandas as pd
import numpy as np

from enda.backtesting import BackTesting


# --- Helpers

def _compare_frames_helper(train, test, input_df, indices, i_split):
    """Helper function to call the testing frame comparing function"""
    pd.testing.assert_frame_equal(
        train,
        input_df.iloc[indices[i_split]['train'][0]: indices[i_split]['train'][1]]
    )
    pd.testing.assert_frame_equal(
        test,
        input_df.iloc[indices[i_split]['test'][0]: indices[i_split]['test'][1]]
    )


class TestBackTesting(unittest.TestCase):
    """
    This class aims at testing the functions of the Backtesting class in enda/backtesting.py
    """

    def setUp(self):
        logging.disable(logging.INFO)

        # a timeseries obtained from financial information of some companies
        # cf: https://github.com/INRIA/scikit-learn-mooc/tree/main/datasets
        self.input_df = pd.DataFrame([
            (pd.Timestamp('2007-06-01'), 81, 78, 75),
            (pd.Timestamp('2007-06-04'), 82, 78, 75),
            (pd.Timestamp('2007-06-05'), 82, 79, 76),
            (pd.Timestamp('2007-06-06'), 82, 78, 75),
            (pd.Timestamp('2007-06-07'), 81, 78, 75),
            (pd.Timestamp('2007-06-08'), 80, 76, 72),
            (pd.Timestamp('2007-06-11'), 80, 77, 73),
            (pd.Timestamp('2007-06-12'), 81, 77, 73),
            (pd.Timestamp('2007-06-13'), 80, 77, 73),
            (pd.Timestamp('2007-06-14'), 81, 78, 75),
            (pd.Timestamp('2007-06-15'), 82, 79, 76),
            (pd.Timestamp('2007-06-18'), 83, 80, 77),
            (pd.Timestamp('2007-06-19'), 82, 80, 77),
            (pd.Timestamp('2007-06-20'), 83, 80, 76),
            (pd.Timestamp('2007-06-21'), 81, 78, 75),
            (pd.Timestamp('2007-06-22'), 82, 79, 76),
            (pd.Timestamp('2007-06-25'), 81, 78, 75),
            (pd.Timestamp('2007-06-26'), 83, 78, 76),
            (pd.Timestamp('2007-06-27'), 81, 74, 72),
            (pd.Timestamp('2007-06-28'), 84, 77, 74),
            (pd.Timestamp('2007-06-29'), 84, 77, 74)],
            columns=['date', 'chevron', 'phillips', 'valero']
        ).set_index('date')

        # and a multiindex
        self.input_multi_df = (
            self.input_df
            .copy()
            .assign(rand_class=[randint(0, 1) for _ in range(len(self.input_df))])
            .reset_index()
            .set_index(['rand_class', 'date'])
            .sort_index()
        )

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

        with self.assertRaises(TypeError):
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

        self.assertEqual(len(train_df_list), 104)
        self.assertEqual(len(test_df_list), 104)

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
            end=pd.Timestamp(year=2025, month=5, day=30, tz="Europe/Paris"),
            freq="H",
            tz="Europe/Paris",
            inclusive="left",
        )

        expected_output_train_df = pd.DataFrame(
            index=[np.array(range(56183)), np.array(train_output_range)]
        )
        expected_output_train_df["value"] = 1
        expected_output_train_df.index.names = ["col1", "time"]

        test_output_range = pd.date_range(
            start=pd.Timestamp(year=2025, month=6, day=13, tz="Europe/Paris"),
            end=pd.Timestamp(year=2025, month=6, day=23, tz="Europe/Paris"),
            freq="H",
            tz="Europe/Paris",
            inclusive="left",
        )

        expected_output_test_df = pd.DataFrame(
            index=[np.array(range(56519, 56759)), np.array(test_output_range)]
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

        self.assertEqual(len(train_df_list), 33)
        self.assertEqual(len(test_df_list), 33)

        output_train_df = train_df_list[24]
        output_test_df = test_df_list[24]

        pd.testing.assert_frame_equal(output_train_df, expected_output_train_df)
        pd.testing.assert_frame_equal(output_test_df, expected_output_test_df)

    def test_yield_train_test_regular_split(self):
        """
        Test yield_train_test_split with a simple datetime index, using n_splits parameters
        """

        # we will split the input dataset in 3 sets
        # define indexes of the successive expected train/test df
        indices = [
            {'train': [0, 6], 'test': [6, 11]},
            {'train': [0, 11], 'test': [11, 16]},
            {'train': [0, 16], 'test': [16, 21]}
        ]

        i_split = 0
        for train, test in BackTesting.yield_train_test_regular_split(
            df=self.input_df,
            n_splits=3
        ):
            _compare_frames_helper(train, test, self.input_df, indices, i_split)
            i_split += 1

        # with a multiindex now: note it should not have an impact on the splitting
        i_split = 0
        for train, test in BackTesting.yield_train_test_regular_split(
            df=self.input_multi_df,
            n_splits=3
        ):
            _compare_frames_helper(train, test,
                                   self.input_multi_df.sort_index(level=-1),  # we built self.multi_df to be sorted on
                                                                              # the first index
                                   indices, i_split)
            i_split += 1

        # we will split the input dataset in 2 sets with a gap_test_freq of '3D'
        # define indexes of the successive expected train/test df
        indices = [
            {'train': [0, 4], 'test': [7, 14]},
            {'train': [0, 11], 'test': [14, None]},
        ]

        i_split = 0
        for train, test in BackTesting.yield_train_test_regular_split(
            df=self.input_df.sort_index(ascending=False),  # for testing purposes
            n_splits=2,
            gap_size='3D'
        ):
            _compare_frames_helper(train, test, self.input_df, indices, i_split)
            i_split += 1

        # with a multiindex now: note it should not have an impact on the splitting
        i_split = 0
        for train, test in BackTesting.yield_train_test_regular_split(
            df=self.input_multi_df,
            n_splits=2,
            gap_size='3D'
        ):
            _compare_frames_helper(train, test,
                                   self.input_multi_df.sort_index(level=-1),  # we built self.multi_df to be sorted on
                                                                              # the first index
                                   indices, i_split)
            i_split += 1

        # we will split the input dataset in 2 sets:
        # - with a gap_test_freq of '2D'
        # - with an initial train_frame of at least 10D
        indices = [
            {'train': [0, 9], 'test': [11, 16]},
            {'train': [0, 14], 'test': [16, None]},
        ]

        i_split = 0
        for train, test in BackTesting.yield_train_test_regular_split(
                df=self.input_df,
                n_splits=2,
                gap_size='2D',
                min_train_size='10D'
        ):
            _compare_frames_helper(train, test, self.input_df, indices, i_split)
            i_split += 1

        # with a multiindex now: note it should not have an impact on the splitting
        i_split = 0
        for train, test in BackTesting.yield_train_test_regular_split(
                df=self.input_multi_df,
                n_splits=2,
                gap_size='2D',
                min_train_size='10D'
        ):
            _compare_frames_helper(train, test,
                                   self.input_multi_df.sort_index(level=-1),  # we built self.multi_df to be sorted on
                                   # the first index
                                   indices, i_split)
            i_split += 1

        # sometimes, there cannot be a split, because there's not enough data
        with self.assertRaises(ValueError):
            list(BackTesting.yield_train_test_regular_split(self.input_df, n_splits=2, gap_size='1W'))

    def test_yield_train_test_periodic_split(self):
        """
        Test yield_train_test_split with a simple datetime index, using the test_size_freq parameter
        """

        # we will split on a 1W scale
        # define indexes of the successive expected train_test_df
        indices = [
            {'train': [0, 5], 'test': [5, 10]},
            {'train': [0, 10], 'test': [10, 15]},
            {'train': [0, 15], 'test': [15, 20]}
         ]

        i_split = 0
        for train, test in BackTesting.yield_train_test_periodic_split(
                df=self.input_df.sort_index(ascending=False),
                test_size='1W'
        ):
            _compare_frames_helper(train, test, self.input_df, indices, i_split)
            i_split += 1

        # with a multiindex
        i_split = 0
        for train, test in BackTesting.yield_train_test_periodic_split(
                df=self.input_multi_df.sort_index(ascending=False),  # testing purposes
                test_size='1W'
        ):
            _compare_frames_helper(train, test,
                                   self.input_multi_df.sort_index(level=-1),  # we built self.multi_df to be sorted on
                                                                              # the first index
                                   indices, i_split)
            i_split += 1

        # split on a 1W scale, with train_size initial of 2W, and gap of 2D
        indices = [
            {'train': [0, 10], 'test': [11, 16]},
         ]

        i_split = 0
        for train, test in BackTesting.yield_train_test_periodic_split(
                df=self.input_df,
                test_size='1W',
                min_train_size='2W',
                gap_size='2D'
        ):
            _compare_frames_helper(train, test, self.input_df, indices, i_split)
            i_split += 1

        # with a multiindex
        i_split = 0
        for train, test in BackTesting.yield_train_test_periodic_split(
                df=self.input_multi_df.sort_index(ascending=False),  # testing purposes
                test_size='1W',
                min_train_size='2W',
                gap_size='2D'
        ):
            _compare_frames_helper(train, test,
                                   self.input_multi_df.sort_index(level=-1),  # we built self.multi_df to be sorted on
                                                                              # the first index
                                   indices, i_split)
            i_split += 1