"""A module containing functions used for the backtesting of models"""

from collections.abc import Generator
from typing import Union, Optional, Callable, Tuple

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from enda.estimators import EndaEstimator
from enda.scoring import Scoring
from enda.tools.decorators import warning_deprecated_name
from enda.tools.timeseries import TimeSeries
from enda.config import get_logger


class BackTesting:
    """
    A class to help with back-testing on algorithms.
    """

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
        Returns pairs of (train set, test set) to perform back-testing on the data.
        The splitting relies on the Scikit object 'TimeSeriesSplit'. The size of the test dataset is determined
        by scikit, so that the dataset can be split into n_splits.
        :param df: the dataset, a pd.DataFrame with a pd.DatetimeIndex or MultiIndex with last index being
            the DatetimeIndex.
        :param n_splits: number of splits train-test sets. If none, defaults to 5
        :param gap_size: eg '1D', '1W'... Size of the data between the train and test samples, expressed as a freqstr
            or a pd.Timedelta.
        :param min_train_size: the minimal initial size of the train set, out of the splitting routine.
            not this does not correspond to the exact size of the first train set, which is min_train_size + test_size
        :return: a generator of (train set, test set) pairs
        """

        sorted_df = df.sort_index(level=-1)

        # we need to convert gap_freq to a number of samples to be used by the builtin functions
        # of scikit, according to the dataframe datetimeindex frequency.
        # note this supposes the dataframe has a regular frequency and a few missing data
        initial_freq = TimeSeries.find_most_common_frequency(
            sorted_df.index.get_level_values(-1), skip_duplicate_timestamps=True
        )
        gap = int(TimeSeries.freq_as_approximate_nb_seconds(gap_size) /
                  TimeSeries.freq_as_approximate_nb_seconds(initial_freq)
                  )

        # if min_train_size is defined, we find the location of the index that begins after it
        # to get that out of the splitting algorithm
        freq_before_splitting = min_train_size
        if min_train_size is None:
            freq_before_splitting = '0D'

        end_init_train_set_time = TimeSeries.add_timedelta(sorted_df.index.get_level_values(-1).min(),
                                                           freq_before_splitting)
        shift_index_int = len(sorted_df.loc[sorted_df.index.get_level_values(-1) < end_init_train_set_time])

        # define a time_series_split object from the shifted object
        time_series_split = TimeSeriesSplit(n_splits=n_splits, gap=gap)

        # yield the train-test split from the sorted_df that begins after
        initial_train_index = [_ for _ in range(shift_index_int)]
        for train_index, test_index in time_series_split.split(sorted_df.iloc[shift_index_int:]):
            train_index = initial_train_index + [_ + shift_index_int for _ in train_index]
            test_index = [_ + shift_index_int for _ in test_index]
            yield sorted_df.iloc[train_index], sorted_df.iloc[test_index]

    @staticmethod
    def yield_train_test_periodic_split(
            df: pd.DataFrame,
            test_size: Union[str, pd.Timedelta],
            gap_size: Optional[Union[str, pd.Timedelta]] = '0D',
            min_train_size: Optional[Union[str, pd.Timedelta]] = None,
            min_last_test_size_pct: Optional[float] = 0.5
    ) -> Generator[(pd.DataFrame, pd.DataFrame)]:
        """
        Returns pairs of (train set, test set) to perform back-testing on the data.
        There, we do not indicate the number of splits to perform, but the size of the test dataset, for instance
            '1M', or '10D'. If min_train_size is None, test_size has the same size as the initial train set.
            Then, for each iteration, the train set is increased, but the size of the test set remains constant.
            If the last test sample is smaller than the desired test size, it might not be returned to avoid
            side effects.
        :param df: the dataset, a pd.DataFrame with a pd.DatetimeIndex or MultiIndex with last index being
            the DatetimeIndex.
        :param test_size: eg. '28D', '2M'... Fixed size of the test sample.
        :param gap_size: eg '1D', '1W'... Size of the data between the train and test samples.
        :param min_train_size: the size of the first initial train test. If None, it defaults to test_size.
        :param min_last_test_size_pct: The last test set can be smaller than the desired test_size. This parameter
            is used to control the minimal size of the last test set, based on a percentage of the 'test_set' variable.
            If the last test set is smaller than min_last_test_size_pct * test_size, it is not returned.
            It defaults to 0.5 (i.e. the last test set must be greater than half of test_size).
        :return: a generator of (train set, test set) pairs
        """
        sorted_df = df.sort_index(level=-1)

        # hardcode a minimal size of the test set to avoid side effects in the backtesting
        if min_last_test_size_pct > 1 or min_last_test_size_pct < 0:
            raise ValueError(f"last_min_test_size_pct must be a float between 0 and 1, found {min_last_test_size_pct}.")
        min_test_set_size = str(TimeSeries.freq_as_approximate_nb_seconds(test_size) * min_last_test_size_pct) + 'S'
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
            excl_end_train = TimeSeries.add_timedelta(excl_end_train, test_size)
            start_test = TimeSeries.add_timedelta(excl_end_train, gap_size)
            excl_end_test = TimeSeries.add_timedelta(start_test, test_size)

    @staticmethod
    def backtest(
            estimator: EndaEstimator,
            df: pd.DataFrame,
            target_col: str,
            scores: Optional[Union[list[str], dict[str, str]]] = None,
            process_forecast_specs: Optional[Tuple[Callable, dict]] = None,
            retrain_estimator: bool = True,
            verbose: bool = False,
            **kwargs
    ) -> dict[str, pd.DataFrame]:
        """
        Backtest an estimator over a dataset. That means performing successive training and prediction on growing
            timeseries datasets, and compute on each set some scores to estimate the quality of the estimator over the
            dataset. The backtesting scheme (train/test splits) is defined using either the function
            yield_train_test_periodic_split() if test_size is given in the arguments of this function, or
            yield_train_test_regular_split() otherwise.
            This function returns a dict with two keys, 'score' and 'forecast':
            - 'score' contains a dataframe with the result of the scoring statistic on each train and test set.
            - 'forecast' contains a dataframe with the forecast on each test set.
        :param estimator: the EndaEstimator to backtest.
        :param df: the input dataframe on which the estimator is back-tested.
        :param target_col: the target column.
        :param scores: Optional. Define the score function to use for the backtesting.
            It can be a list of loss functions to estimate as defined in Scoring().
            It can be a dict of name-methods, and the end-user can provide any function in that case.
            If nothing is given, it defaults to RMSE.
        :param process_forecast_specs: Optional. If given, it defines a function to apply to the result of
            each prediction before calculating the scoring (it is also applied to the training loss). A typical example
            is the PowerStation.clip_column() function, which is used to clamp the forecast load factor between
            0 and 1.
            process_forecast_specs must be a tuple, with the function to be applied, and a dict with all
            the function keywords arguments names and values.
        :param retrain_estimator: boolean, if True (default), perform a real backtesting during which an
            estimator is retrained before being used to perform a forecast. If False, the estimator must be
            already trained, and this function only serves to perform successive forecasts on test sets.
        :param kwargs: extra argument to pass to the chosen split method yield_train_test_regular_split() or
            yield_train_test_periodic_split(), such as n_splits, test_size, gap_size, min_train_size,
            min_last_test_size_pct...
            If nothing is given, yield_train_test_regular_split(n_splits=5) is called.
        :param verbose: boolean, defaults False. Print or no information.
        :return: a dict which contains:
            - for the 'score' key: a dataframe with the train and test results for each statistics and each split.
            - for the 'forecast' key: a dataframe with the successive forecasts on the test sets.
        """

        if 'test_size' in kwargs:
            split_generator = BackTesting.yield_train_test_periodic_split(df=df, **kwargs)
        else:
            split_generator = BackTesting.yield_train_test_regular_split(df=df, **kwargs)

        logger = get_logger()
        scoring_result_list = []
        all_forecasts_list = []
        backtest_iter = 0

        for train_set, test_set in split_generator:

            backtest_iter += 1
            if verbose:
                logger.info(f"Train index: {(train_set.index.min(), train_set.index.max())}")

            # train estimator
            if retrain_estimator:
                estimator.train(df=train_set, target_col=target_col)

            if verbose:
                logger.info(f"Test index: {(test_set.index.min(), test_set.index.max())}")

            # predict
            predict_df = estimator.predict(df=test_set.drop(columns=target_col), target_col=target_col)

            # process the result of the forecast in case a specification has been provided
            if process_forecast_specs is not None:
                process_forecast_function, process_forecast_kwargs = process_forecast_specs
                predict_df = process_forecast_function(predict_df, **process_forecast_kwargs)

            # store predict
            all_forecasts_list.append(predict_df.assign(backtest_iter=backtest_iter))

            # -- compute scores

            # get training score
            training_score = estimator.get_loss_training(scores=scores,
                                                         process_forecast_specs=process_forecast_specs
                                                         )

            # get test score,
            test_score = Scoring.compute_loss(predicted_df=predict_df,
                                              actual_df=test_set[target_col],
                                              scores=scores
                                              )

            # rename indexes for scoring and create a dataframe with scores for that iteration
            training_score.index = ['train_' + _ for _ in training_score.index]
            test_score.index = ['test_' + _ for _ in test_score.index]
            scores_df = pd.concat([training_score, test_score]).to_frame().T
            start_end_times_df = pd.DataFrame(
                {
                    "train_start": [train_set.index.get_level_values(-1).min()],
                    "train_end": [train_set.index.get_level_values(-1).max()],
                    "test_start": [test_set.index.get_level_values(-1).min()],
                    "test_end": [test_set.index.get_level_values(-1).max()],
                }
            )
            scores_df = pd.concat([scores_df, start_end_times_df], axis=1)

            if verbose:
                logger.info(f"Partial scores: \n{scores_df.to_string()}")

            scoring_result_list.append(scores_df)

        result_dict = {"score": pd.concat(scoring_result_list).reset_index(drop=True),
                       "forecast": pd.concat(all_forecasts_list)}

        return result_dict
