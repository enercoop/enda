"""A module containing functions used for the backtesting of models"""
import datetime
from collections.abc import Generator
from typing import Union, Optional

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from enda.estimators import EndaEstimator
from enda.scoring import Scoring
from enda.tools.decorators import warning_deprecated_name
from enda.tools.timeseries import TimeSeries


class BackTesting:
    """
    A class to help with back-testing on algorithms.
    """

    # @staticmethod
    # def yield_train_test(
    #     df: pd.DataFrame,
    #     start_eval_datetime: pd.Timestamp,
    #     days_between_trains: int,
    #     gap_days_between_train_and_eval: int = 0,
    # ) -> Generator[(pd.DataFrame, pd.DataFrame)]:
    #     """
    #     Returns pairs of (train set, test set) to perform back-testing on the data
    #
    #     :param df: the dataset, a pandas.DataFrame with a pandas.DatetimeIndex
    #     :param start_eval_datetime: the beginning of the first eval, with same timezone as the dataset DataFrame index
    #     :param days_between_trains: number of days between two train sets, it is also the duration of each test set
    #     :param gap_days_between_train_and_eval: optional, represents the time gap between the moment when
    #                                             things occurred and when we have the data available for training.
    #                                             (typically a few days or weeks)
    #     :return: a generator of (train set, test set) pairs
    #     """
    #
    #     if not (
    #         start_eval_datetime.hour
    #         == start_eval_datetime.minute
    #         == start_eval_datetime.second
    #         == start_eval_datetime.microsecond
    #         == 0
    #     ):
    #         raise ValueError(
    #             "start_eval_datetime must be datetime with only years, months or days (not more precise),"
    #             f" but given: {type(start_eval_datetime)}, {start_eval_datetime}"
    #         )
    #
    #     # initialize date-times: end_train, start_test and end_test
    #     if gap_days_between_train_and_eval > 0:
    #         end_train = TimezoneUtils.add_interval_to_day_dt(
    #             start_eval_datetime,
    #             relativedelta(days=-gap_days_between_train_and_eval),
    #         )
    #     else:
    #         end_train = start_eval_datetime
    #     train_interval = relativedelta(days=days_between_trains)
    #     start_test = start_eval_datetime
    #     end_test = TimezoneUtils.add_interval_to_day_dt(start_test, train_interval)
    #
    #     if isinstance(df.index, pd.DatetimeIndex):
    #         if str(df.index.tz) != str(start_eval_datetime.tzinfo):
    #             raise ValueError(
    #                 f"df.index (tzinfo={df.index.tz}) and start_eval_datetime "
    #                 f"(tzinfo={start_eval_datetime.tzinfo}) must have the same tzinfo."
    #             )
    #
    #         if start_eval_datetime <= df.index.min():
    #             raise ValueError(
    #                 f"start_eval_datetime ({start_eval_datetime,}) must be after the beginning of df "
    #                 f"({df.index.min()})"
    #             )
    #
    #         # go through the dataset and yield pairs of (train set, test set)
    #         while start_test < df.index.max():
    #             yield df[df.index < end_train], df[
    #                 (df.index >= start_test) & (df.index < end_test)
    #             ]
    #
    #             end_train = TimezoneUtils.add_interval_to_day_dt(
    #                 end_train, train_interval
    #             )
    #             start_test = TimezoneUtils.add_interval_to_day_dt(
    #                 start_test, train_interval
    #             )
    #             end_test = TimezoneUtils.add_interval_to_day_dt(
    #                 end_test, train_interval
    #             )
    #
    #     elif isinstance(df.index, pd.MultiIndex):
    #
    #         if not isinstance(df.index.levels[-1], pd.DatetimeIndex):
    #             raise TypeError(
    #                 f"The last index of the dataframe should be a pd.DatetimeIndex, but given"
    #                 f" {df.index.levels[-1].dtype}"
    #             )
    #
    #         if str(df.index.levels[-1].tz) != str(
    #             start_eval_datetime.tzinfo
    #         ):
    #             raise ValueError(
    #                 f"df.index (tzinfo={df.index.levels[-1].tz}) and start_eval_datetime "
    #                 f"(tzinfo={start_eval_datetime.tzinfo}) must have the same tzinfo."
    #             )
    #
    #         if start_eval_datetime <= df.index.levels[-1].min():
    #             raise ValueError(
    #                 f"start_eval_datetime ({start_eval_datetime}) must be after the beginning of df "
    #                 f"({df.index.get_level_values(-1).min()})"
    #             )
    #
    #         # go through the dataset and yield pairs of (train set, test set)
    #         while start_test < df.index.levels[-1].max():
    #             yield df[df.index.get_level_values(-1) < end_train], df[
    #                 (df.index.get_level_values(-1) >= start_test)
    #                 & (df.index.get_level_values(-1) < end_test)
    #             ]
    #
    #             end_train = TimezoneUtils.add_interval_to_day_dt(
    #                 end_train, train_interval
    #             )
    #             start_test = TimezoneUtils.add_interval_to_day_dt(
    #                 start_test, train_interval
    #             )
    #             end_test = TimezoneUtils.add_interval_to_day_dt(
    #                 end_test, train_interval
    #             )
    #
    #     else:
    #         raise ValueError(
    #             "df must have a pandas.DatetimeIndex index, even if it's a MultiIndex"
    #         )

    @staticmethod
    @warning_deprecated_name(namespace_name='Backtesting', new_function_name='yield_train_test_periodic_split')
    def yield_train_test(
        df: pd.DataFrame,
        start_eval_datetime: pd.Timestamp,
        days_between_trains: int,
        gap_days_between_train_and_eval: int = 0,
    ) -> Generator[(pd.DataFrame, pd.DataFrame)]:

        # for consistency with previous implementation
        if not (start_eval_datetime.hour
                == start_eval_datetime.minute
                == start_eval_datetime.second
                == start_eval_datetime.microsecond
                == 0
        ):
            raise ValueError(
                "start_eval_datetime must be datetime with only years, months or days (not more precise),"
                f" but given: {type(start_eval_datetime)}, {start_eval_datetime}"
            )

        # start time
        start_time = df.index.get_level_values(-1).min()

        if not isinstance(start_time, pd.Timestamp):
            raise TypeError(f"Last index should be a DatetimeIndex, found{type(start_time)}")

        # for consistency with previous implementation
        if start_time.tzinfo != start_eval_datetime.tzinfo:
            raise ValueError(
                f"df.index (tzinfo={start_time.tzinfo}) and start_eval_datetime "
                f"(tzinfo={start_eval_datetime.tzinfo}) must have the same tzinfo."
            )
        if start_eval_datetime <= start_time:
            raise ValueError(
                             f"start_eval_datetime ({start_eval_datetime,}) must be after the beginning of df "
                             f"({start_time})"
                         )

        yield from BackTesting.yield_train_test_periodic_split(
            df=df,
            test_size=str(days_between_trains) + 'D',
            gap_size=str(gap_days_between_train_and_eval) + 'D',
            min_train_size=start_eval_datetime - start_time
        )

    @staticmethod
    def yield_train_test_regular_split(
            df: pd.DataFrame,
            n_splits: Optional[int] = 5,
            gap_size: Optional[Union[str, pd.Timedelta]] = '0D',
            min_train_size : Optional[Union[str, pd.Timedelta]] = None,
    ) -> Generator[(pd.DataFrame, pd.DataFrame)]:
        """
        Returns pairs of (train set, test set) to perform back-testing on the data
        :param df: the dataset, a pd.DataFrame with a pd.DatetimeIndex or MultiIndex with last index being
            the DatetimeIndex.
        :param n_splits: number of splits train-test sets. If none, defaults to 5
        :param gap_size: eg '1D', '1W'... Size of the data between the train and test samples, expressed as a freqstr
            or a pd.Timedelta.
        :param min_train_size: the minimal initial size of the train set, out of the splitting routine.
            not this does not correspond to the exact size of the first train set, which is min_train_size + test_size
        :return: a generator of (train set, test set) pairs
        """

        sorted_df = df.copy().sort_index(level=-1)

        # we need to convert gap_freq to a number of samples to be used by the builtin functions
        # of scikit, according to the dataframe datetimeindex frequency.
        # note this supposes the dataframe has a regular frequency and a few missing data
        initial_freq = TimeSeries.find_most_common_frequency(
            sorted_df.index.get_level_values(-1), skip_duplicate_timestamps=True
        )
        gap = int(TimeSeries.freq_as_approximate_nb_days(gap_size) /
                  TimeSeries.freq_as_approximate_nb_days(initial_freq)
                  )

        # if min_train_size is defined, we find the location of the index that begins after it
        # to get that out of the splitting algorithm
        freq_before_splitting = min_train_size
        if min_train_size is None:
            freq_before_splitting = '0D'

        end_init_train_set_time = TimeSeries.add_timedelta(sorted_df.index.get_level_values(-1).min(),
                                                           freq_before_splitting)
        shift_index = len(sorted_df.loc[sorted_df.index.get_level_values(-1) < end_init_train_set_time])

        # define a time_series_split object from the shifted object
        time_series_split = TimeSeriesSplit(n_splits=n_splits, gap=gap)

        # yield the train-test split from the sorted_df that begins after
        initial_train_index = [_ for _ in range(shift_index)]
        for train_index, test_index in time_series_split.split(sorted_df.iloc[shift_index:]):
            train_index = initial_train_index + [_ + shift_index for _ in train_index]
            test_index = [_ + shift_index for _ in test_index]
            yield sorted_df.iloc[train_index], sorted_df.iloc[test_index]

    @staticmethod
    def yield_train_test_periodic_split(
            df: pd.DataFrame,
            test_size: Union[str, pd.Timedelta],
            gap_size: Optional[Union[str, pd.Timedelta]] = '0D',
            min_train_size: Optional[Union[str, pd.Timedelta]] = None
    ) -> Generator[(pd.DataFrame, pd.DataFrame)]:
        """
        Returns pairs of (train set, test set) to perform back-testing on the data
        :param df: the dataset, a pd.DataFrame with a pd.DatetimeIndex or MultiIndex with last index being
            the DatetimeIndex.
        :param test_size: eg. '28D', '2M'... Size of the test sample, which is the same
            as the initial train sample, and then, size of the increase of the train sample for each iteration.
        :param gap_size: eg '1D', '1W'... Size of the data between the train and test samples.
        :param min_train_size: the size of the first initial train test. I None, it defaults to test_size.
        :return: a generator of (train set, test set) pairs
        """
        sorted_df = df.copy().sort_index(level=-1)

        # hardcode a minimal size of the test set to avoid side effects in the backtesting
        min_test_set_size = str(TimeSeries.freq_as_approximate_nb_days(test_size) / 2) + 'D'
        min_index = df.index.get_level_values(-1).min()
        max_index = df.index.get_level_values(-1).max()

        # initial train size
        if min_train_size is None:
            min_train_size = test_size

        # yield train-test; each iteration adds test_size (for the new test_set) and gap_size
        start_train = min_index
        excl_end_train = TimeSeries.add_timedelta(start_train, min_train_size)
        start_test = TimeSeries.add_timedelta(excl_end_train, gap_size)
        excl_end_test = TimeSeries.add_timedelta(start_test, test_size)

        while start_test < max_index:

            # check test set is big enough: it's been set to half of the test_size_freq
            # if not, it's the last run, and test set is too small to be retrieved
            if (max_index - start_test) < pd.to_timedelta(min_test_set_size):
                break

            yield (
                sorted_df.loc[sorted_df.index.get_level_values(-1) < excl_end_train],
                sorted_df.loc[
                    (sorted_df.index.get_level_values(-1) >= start_test) &
                    (sorted_df.index.get_level_values(-1) < excl_end_test)
                    ]
            )

            # prepare next iteration
            excl_end_train = excl_end_test
            start_test = TimeSeries.add_timedelta(excl_end_train, gap_size)
            excl_end_test = TimeSeries.add_timedelta(start_test, test_size)

    @staticmethod
    def backtest(
            estimator: EndaEstimator,
            df: pd.DataFrame,
            target_col: str,
            score_list: list[str] = None,
            split_method: str = 'regular',
            **kwargs
    ) -> pd.DataFrame:
        """
        Backtest an estimator over a dataset.
        :param estimator: the EndaEstimator to backtest
        :param df: the data dataframe on which the estimator is back-tested.
        :param target_col: the target column
        :param score_list: the list of loss functions to estimate.
        :param split_method: the split method to use. Either regular or periodic. If periodic is selected,
            the argument 'test_size_freq' must be given to the method in kwargs.
        :param kwargs: extra argument to pass to the chosen split method yield_train_test_split(),
            such as n_splits, test_size_freq, gap_size_freq, start_eval_time.
        :return: a dataframe with the train and test results for each statistics and each split.
        """
        if score_list is None:
            score_list = ['rmse']

        if split_method == 'regular':
            split_generator = BackTesting.yield_train_test_regular_split(df=df, **kwargs)
        elif split_method == 'periodic':
            split_generator = BackTesting.yield_train_test_periodic_split(df=df, **kwargs)
        else:
            raise ValueError(f"split_method must be regular or periodic. Found {split_method}")

        scoring_result_list = []
        for train_set, test_set in split_generator:

            # train estimator
            estimator.train(df=train_set, target_col=target_col)

            # get training score
            training_score = estimator.get_loss_training(score_list=score_list)

            # get test score
            predict_df = estimator.predict(df=test_set.drop(columns=target_col), target_col=target_col)
            test_score = Scoring.compute_loss(predicted_df=predict_df,
                                              actual_df=test_set[target_col],
                                              score_list=score_list
                                              )

            # rename indexes and create a dataframe with scores for that iteration
            training_score.index = ['train_' + _ for _ in training_score.index]
            test_score.index = ['test_' + _ for _ in test_score.index]
            scoring_result_list.append(pd.concat([training_score, test_score]).to_frame().T)

        return pd.concat(scoring_result_list).reset_index(drop=True)
