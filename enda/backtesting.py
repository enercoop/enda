"""A module containing functions used for the backtesting of models"""

from collections.abc import Generator

from dateutil.relativedelta import relativedelta
import pandas as pd

from enda.timezone_utils import TimezoneUtils


class BackTesting:
    """
    A class to help with back-testing on algorithms.
    """

    @staticmethod
    def yield_train_test(
        df: pd.DataFrame,
        start_eval_datetime: pd.Timestamp,
        days_between_trains: int,
        gap_days_between_train_and_eval: int = 0,
    ) -> Generator[(pd.DataFrame, pd.DataFrame)]:
        """
        Returns pairs of (train set, test set) to perform back-testing on the data

        :param df: the dataset, a pandas.DataFrame with a pandas.DatetimeIndex
        :param start_eval_datetime: the beginning of the first eval, with same timezone as the dataset DataFrame index
        :param days_between_trains: number of days between two train sets, it is also the duration of each test set
        :param gap_days_between_train_and_eval: optional, represents the time gap between the moment when
                                                things occurred and when we have the data available for training.
                                                (typically a few days or weeks)
        :return: a generator of (train set, test set) pairs
        """

        if not (
            start_eval_datetime.hour
            == start_eval_datetime.minute
            == start_eval_datetime.second
            == start_eval_datetime.microsecond
            == 0
        ):
            raise ValueError(
                "start_eval_datetime must be datetime with only years, months or days (not more precise),"
                f" but given: {type(start_eval_datetime)}, {start_eval_datetime}"
            )

        # initialize date-times: end_train, start_test and end_test
        if gap_days_between_train_and_eval > 0:
            end_train = TimezoneUtils.add_interval_to_day_dt(
                start_eval_datetime,
                relativedelta(days=-gap_days_between_train_and_eval),
            )
        else:
            end_train = start_eval_datetime
        train_interval = relativedelta(days=days_between_trains)
        start_test = start_eval_datetime
        end_test = TimezoneUtils.add_interval_to_day_dt(start_test, train_interval)

        if isinstance(df.index, pd.DatetimeIndex):
            if str(df.index.tz) != str(start_eval_datetime.tzinfo):
                raise ValueError(
                    f"df.index (tzinfo={df.index.tz}) and start_eval_datetime "
                    f"(tzinfo={start_eval_datetime.tzinfo}) must have the same tzinfo."
                )

            if start_eval_datetime <= df.index.min():
                raise ValueError(
                    f"start_eval_datetime ({start_eval_datetime,}) must be after the beginning of df "
                    f"({df.index.min()})"
                )

            # go through the dataset and yield pairs of (train set, test set)
            while start_test < df.index.max():
                yield df[df.index < end_train], df[
                    (df.index >= start_test) & (df.index < end_test)
                ]

                end_train = TimezoneUtils.add_interval_to_day_dt(
                    end_train, train_interval
                )
                start_test = TimezoneUtils.add_interval_to_day_dt(
                    start_test, train_interval
                )
                end_test = TimezoneUtils.add_interval_to_day_dt(
                    end_test, train_interval
                )

        elif isinstance(df.index, pd.MultiIndex):
            if len(df.index.levels) != 2:
                raise TypeError(
                    "The provided multi-indexed dataframe must be a two-levels one, the "
                    "second one being the date index."
                )

            if not isinstance(df.index.levels[1], pd.DatetimeIndex):
                raise TypeError(
                    f"The second index of the dataframe should be a pd.DatetimeIndex, but given"
                    f" {df.index.levels[1].dtype}"
                )

            time_col = df.index.levels[1].name

            # TODO : this line fails if the DatetimeIndex col has no name
            if str(df.index.get_level_values(time_col).tz) != str(
                start_eval_datetime.tzinfo
            ):
                raise ValueError(
                    f"df.index (tzinfo={df.index.get_level_values(time_col).tz}) and start_eval_datetime "
                    f"(tzinfo={start_eval_datetime.tzinfo}) must have the same tzinfo."
                )

            if start_eval_datetime <= df.index.get_level_values(time_col).min():
                raise ValueError(
                    f"start_eval_datetime ({start_eval_datetime}) must be after the beginning of df "
                    f"({df.index.get_level_values(time_col).min()})"
                )

            # go through the dataset and yield pairs of (train set, test set)
            while start_test < df.index.get_level_values(time_col).max():
                yield df[df.index.get_level_values(time_col) < end_train], df[
                    (df.index.get_level_values(time_col) >= start_test)
                    & (df.index.get_level_values(time_col) < end_test)
                ]

                end_train = TimezoneUtils.add_interval_to_day_dt(
                    end_train, train_interval
                )
                start_test = TimezoneUtils.add_interval_to_day_dt(
                    start_test, train_interval
                )
                end_test = TimezoneUtils.add_interval_to_day_dt(
                    end_test, train_interval
                )

        else:
            raise ValueError(
                "df must have a pandas.DatetimeIndex index, even if it's a MultiIndex"
            )
