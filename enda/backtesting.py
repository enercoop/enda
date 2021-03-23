import pandas as pd
from enda.timezone_utils import TimezoneUtils
from dateutil.relativedelta import relativedelta


class BackTesting:
    """
    A class to help with back-testing on algorithms.
    """

    @staticmethod
    def yield_train_test(
            df,
            start_eval_datetime,
            days_between_trains,
            gap_days_between_train_and_eval=0
    ):
        """
        Returns pairs of (train set, test set) to perform back-testing on the data

        :param df: the dataset, a pandas.DataFrame with a pandas.DatetimeIndex
        :param start_eval_datetime: the beginning of the first eval, with same timezone as df.index
        :param days_between_trains: number of days between two train sets, it is also the duration of each test set
        :param gap_days_between_train_and_eval: optional, represents the time gap between the moment when
                                                things occurred and when we have the data available for training.
                                                (typically a few days or weeks)
        :return: a generator of (train set, test set) pairs
        """

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("df must have a pandas.DatetimeIndex index")

        if str(df.index.tz) != str(start_eval_datetime.tzinfo):
            raise ValueError("df.index (tzinfo={}) and start_eval_datetime (tzinfo={}) must have the same "
                             "tzinfo.".format(df.index.tz, start_eval_datetime.tzinfo))

        if start_eval_datetime <= df.index.min():
            raise ValueError("start_eval_datetime ({}) must be after the beginning of df ({})"
                             .format(start_eval_datetime, df.index.min()))

        if not (start_eval_datetime.hour == start_eval_datetime.minute
                == start_eval_datetime.second == start_eval_datetime.microsecond == 0):
            raise ValueError("start_eval_datetime must be datetime with only years, months or days (not more precise),"
                             " but given: {}, {}".format(type(start_eval_datetime), start_eval_datetime))

        # initialize date-times: end_train, start_test and end_test
        if gap_days_between_train_and_eval > 0:
            end_train = TimezoneUtils.add_interval_to_day_dt(
                start_eval_datetime,
                relativedelta(days=-gap_days_between_train_and_eval)
            )
        else:
            end_train = start_eval_datetime
        train_interval = relativedelta(days=days_between_trains)
        start_test = start_eval_datetime
        end_test = TimezoneUtils.add_interval_to_day_dt(start_test, train_interval)

        # go through the dataset and yield pairs of (train set, test set)
        while start_test < df.index.max():
            yield df[df.index < end_train], df[(df.index >= start_test) & (df.index < end_test)]

            end_train = TimezoneUtils.add_interval_to_day_dt(end_train, train_interval)
            start_test = TimezoneUtils.add_interval_to_day_dt(start_test, train_interval)
            end_test = TimezoneUtils.add_interval_to_day_dt(end_test, train_interval)
