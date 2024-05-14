"""A module for testing the Backtesting class in enda/backtesting.py"""

import logging
from random import randint
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from enda.ml_backends.sklearn_estimator import EndaSklearnEstimator
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

        # a timeseries_data obtained from financial information of some companies
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
            end=pd.Timestamp(year=2024, month=6, day=28, tz="Europe/Paris"),
            freq="H",
            tz="Europe/Paris",
            inclusive="left",
        )

        expected_output_train_df = pd.DataFrame(
            index=[np.array(range(48119)), np.array(train_output_range)]
        )
        expected_output_train_df["value"] = 1
        expected_output_train_df.index.names = ["col1", "time"]

        test_output_range = pd.date_range(
            start=pd.Timestamp(year=2024, month=7, day=12, tz="Europe/Paris"),
            end=pd.Timestamp(year=2024, month=7, day=22, tz="Europe/Paris"),
            freq="H",
            tz="Europe/Paris",
            inclusive="left",
        )

        expected_output_test_df = pd.DataFrame(
            index=[np.array(range(48455, 48695)), np.array(test_output_range)]
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

        output_train_df = train_df_list[24]
        output_test_df = test_df_list[24]

        pd.testing.assert_frame_equal(output_train_df, expected_output_train_df)
        pd.testing.assert_frame_equal(output_test_df, expected_output_test_df)

    def test_yield_train_test_regular_split(self):
        """
        Test yield_train_test_regular_split
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
        Test yield_train_test_periodic_split
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
            {'train': [0, 15], 'test': [16, None]}
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

        # test error if min_last_test_size_pct is greater than 1
        with self.assertRaises(ValueError):
            list(BackTesting.yield_train_test_periodic_split(
                self.input_df, test_size='1OD', min_last_test_size_pct=1.01)
            )

    def test_backtest(self):
        """
        We will test backtest using the real timeseries_data from the INRIA dataset instead of the simplified model
        used until now.
        """

        # working dataframe
        symbols = {"TOT": "Total", "XOM": "Exxon", "CVX": "Chevron",
                   "COP": "ConocoPhillips", "VLO": "Valero Energy"}
        template_name = "timeseries_data/{}.csv"
        quotes = {}
        for symbol in symbols:
            data = pd.read_csv(
                template_name.format(symbol), index_col=0, parse_dates=True
            )
            quotes[symbols[symbol]] = data["open"]
        df = pd.DataFrame(quotes)
        target_col = 'Chevron'

        # working multiindex dataframe
        multi_df = (
            df
            .copy()
            .assign(rand_class=[randint(0, 1) for _ in range(len(df))])
            .reset_index()
            .set_index(['rand_class', 'date'])
            .sort_index()
        )

        score_list = ["rmse", "mape", "r2", "max_error"]

        # test the backtesting
        estimator = EndaSklearnEstimator(LinearRegression())

        # default testing
        result_backtesting = BackTesting.backtest(
            estimator=estimator,
            df=df,
            target_col=target_col,
            score_list=score_list,
        )

        expected_output = pd.DataFrame(
            data=[
                [1.15720606, 0.01399608, 0.91744179, 2.91517062, 3.77332323, 0.03647801, 0.62810329, 8.82897026],
                [1.22062137, 0.01287377, 0.98723791, 3.46643535, 52.95126112, 0.95276092, -59.56571253, 63.1466695],
                [8.24960648, 0.08378842, 0.67934394, 38.47040522, 21.79879097, 0.29710267, -85.8451338, 45.9405591],
                [9.42273962, 0.1157566, 0.5260241, 27.50317796, 102.1765989, 1.44211716, -447.46275825, 138.38196339],
                [10.63660877, 0.12647602, 0.27043878, 29.98877238, 25.97474301, 0.27556018, -13.35245056, 39.23644086]
            ],
            columns=[
                'train_rmse', 'train_mape', 'train_r2', 'train_max_error', 'test_rmse',
                'test_mape', 'test_r2', 'test_max_error'
            ]
        )

        pd.testing.assert_frame_equal(expected_output, result_backtesting)

        # with a multiindex
        result_backtesting = BackTesting.backtest(
            estimator=estimator,
            df=multi_df,
            target_col=target_col,
            score_list=score_list,
        )

        pd.testing.assert_frame_equal(expected_output, result_backtesting)

        # test with a periodic backtesting
        result_backtesting = BackTesting.backtest(
            estimator=estimator,
            df=df,
            target_col=target_col,
            score_list=score_list,
            split_method='periodic',
            test_size='6M',
            gap_size="2W",
            min_train_size='1Y'
        )

        expected_output = pd.DataFrame(
            data=[
                [1.12012364, 0.01332617, 0.95065545, 3.09069785, 2.28543189, 0.02230775, 2.51794177e-01, 4.2457436],
                [1.14292136, 0.01255906, 0.98635499, 3.27970403, 37.97724098, 0.59882608, -2.65982372e+00, 49.01153002],
                [8.18732385, 0.07290751, 0.64485879, 39.0304432, 26.69567957, 0.35471052, -8.44856070e+01, 61.52276424],
                [8.24789383, 0.08363784, 0.6795263, 38.48157249, 27.68732255, 0.41571257, -1.31376739e+02, 46.28708362],
                [9.2608702, 0.11244811, 0.56334664, 30.52625499, 39.10286031, 0.54313922, -2.66806549e+02, 80.71559812],
                [10.23554988, 0.12730966, 0.42421692, 28.35328834, 34.26207391, 0.48453803, -8.79942003e+01,
                 52.25896329],
                [10.67757512, 0.1266696, 0.29131941, 29.76228873, 17.73775561, 0.20433729, -8.31820536e+00,
                 38.45528669],
                [10.75172972, 0.12958557, 0.23976944, 29.7410903, 24.69344502, 0.27023469, -4.47675014e+01,
                 31.91738036],
            ],
            columns=[
                'train_rmse', 'train_mape', 'train_r2', 'train_max_error', 'test_rmse',
                'test_mape', 'test_r2', 'test_max_error'

            ]
        )

        pd.testing.assert_frame_equal(expected_output, result_backtesting)

        result_backtesting = BackTesting.backtest(
            estimator=estimator,
            df=multi_df,
            target_col=target_col,
            score_list=score_list,
            split_method='periodic',
            test_size='6M',
            gap_size="2W",
            min_train_size='1Y'
        )

        pd.testing.assert_frame_equal(expected_output, result_backtesting)
