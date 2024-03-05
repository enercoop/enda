"""This module contains various functions for dealing with timezones in temporal data in Python"""

import datetime
from typing import Union
from dateutil.relativedelta import relativedelta
import pandas as pd
import pytz

import enda.tools.decorators


class TimezoneUtils:
    """
    This class contains various methods for helping deal with timezones
    """

    @staticmethod
    def is_timezone_aware(dt: Union[datetime.datetime, pd.Timestamp]) -> bool:
        """
        Check if a datetime /timestamp is timezone-aware or not
        """
        return dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None

    @classmethod
    def add_interval_to_day_dt(cls, day_dt, interval):
        """Adds an interval (not more precise than a day) to a day,
         correctly dealing with timezone-aware (and naive) day_dt;
         works around daylight savings time changes.

        Normally, to add an interval to a day which is not timezone aware, simply use:
            day_dt + interval.
        This does not work properly for timezone-aware days, so we added this function.

        :param day_dt: a timezone_aware datetime which is a day (hour=minute=seconds=microsecond=0)
        :param interval: an interval of type relativedelta not more precise than a day
        """

        if not (
            day_dt.hour == day_dt.minute == day_dt.second == day_dt.microsecond == 0
        ):
            raise ValueError(
                "day_dt must be datetime with only years, months or days (not more precise),"
                f" but given: {type(day_dt)}, {day_dt}"
            )
        if not (
            isinstance(interval, relativedelta)
            and interval.hours
            == interval.minutes
            == interval.seconds
            == interval.microseconds
            == 0
        ):
            raise (
                ValueError(
                    "Interval must be a relativedelta with only years, months or days "
                    f"(not more precise), but given: {type(interval)}, {interval}"
                )
            )

        tz = day_dt.tzinfo
        day_naive = day_dt.replace(tzinfo=None)
        day_naive = day_naive + interval

        if tz is not None:
            day_aware = tz.localize(day_naive, is_dst=None)
            return day_aware

        return day_naive

    @staticmethod
    @enda.tools.decorators.handle_series_as_datetimeindex(
        arg_name="time_series", return_input_type=True
    )
    def _set_timezone_dti(
        time_series: Union[pd.DatetimeIndex, pd.Series],
        tz_info: Union[str, datetime.tzinfo],
        tz_base: Union[str, datetime.tzinfo] = None,
    ) -> Union[pd.DatetimeIndex, pd.Series]:
        """
        Make a time series timezone-aware or convert it to a target timezone.
        :param time_series: a time series, tz-naive or tz-aware.
        :param tz_info: the target time zone
        :param tz_base: optional, the base time zone if we know it while the time series is time zone naive.
        :return: the time series with the new time zone
        """

        # check tz_info
        if isinstance(tz_info, str):
            tz_info = pytz.timezone(tz_info)
        if not isinstance(tz_info, datetime.tzinfo):
            raise TypeError(
                "parameter 'tzinfo' should be of type str or datetime.tzinfo"
            )

        # store freq to reset it afterward
        # cf https://github.com/pandas-dev/pandas/issues/33677
        freq = time_series.freq

        # do we have a tz-aware time series
        if time_series.tzinfo is not None:
            new_time_series = time_series.tz_convert(tz_info)

        else:
            # else it's a naive time series
            if tz_base is not None:
                if isinstance(tz_base, str):
                    tz_base = pytz.timezone(tz_base)
                if not isinstance(tz_base, datetime.tzinfo):
                    raise TypeError(
                        "parameter 'tz_base' should be of type str or datetime.tzinfo"
                    )

                new_time_series = time_series.tz_localize(tz_base).tz_convert(tz_info)

            else:
                new_time_series = time_series.tz_localize(tz_info)

        # reset freq (dropped during localizing)
        new_time_series.freq = freq

        return new_time_series

    @staticmethod
    @enda.tools.decorators.handle_multiindex(arg_name="df")
    def _set_timezone_frame(
        df: pd.DataFrame,
        tz_info: Union[str, datetime.tzinfo],
        tz_base: Union[str, datetime.tzinfo] = None,
    ) -> pd.DataFrame:
        """
        Make a single-datetime-indexed dataframe's index timezone-aware or convert it to a target timezone.
        :param df: a dataframe with a datetime index
        :param tz_info: the target time zone
        :param tz_base: optional, the base time zone if we know it while the time series is time zone naive.
        :return: the dataframe with the index in the new time zone
        """
        df.index = TimezoneUtils._set_timezone_dti(df.index, tz_info, tz_base)

        return df

    @staticmethod
    def set_timezone(
        time_series: Union[pd.DataFrame, pd.DatetimeIndex, pd.Series],
        tz_info: Union[str, datetime.tzinfo],
        tz_base: Union[str, datetime.tzinfo] = None,
    ) -> Union[pd.DataFrame, pd.DatetimeIndex, pd.Series]:
        """
        Make:
         - a single-datetime-indexed dataframe's index timezone-aware or convert it to a target timezone.
         - a time series timezone-aware or convert it to a target timezone.
        If the time series is time zone naive:
            > if tz_base is None, the function localize the time series to tz_info.
            > if tz_base is given, the function localize the time series to  tz_base and convert it to tz_info
        If the time series is time zone aware, the function converts it to the provided tz_info,
        whatever the value of tz_base
        :param time_series: a datetimeindex, a series, or a dataframe with a datetime index
        :param tz_info: the target time zone
        :param tz_base: optional, the base time zone if we know it while the time series is time zone naive.
        :return: the dataframe with the index in the new time zone
        """

        if isinstance(time_series, pd.DataFrame):
            return TimezoneUtils._set_timezone_frame(time_series, tz_info, tz_base)
        if isinstance(time_series, (pd.DatetimeIndex, pd.Series)):
            return TimezoneUtils._set_timezone_dti(time_series, tz_info, tz_base)

        raise TypeError(
            "time_series should be either a DataFrame, a DatetimeIndex or a Series, "
            f"got {type(time_series)}"
        )

    @staticmethod
    def convert_dtype_from_object_to_tz_aware(
        time_series: [pd.Series, pd.DatetimeIndex],
        tz_info: Union[str, pytz.timezone],
    ) -> Union[pd.Series, pd.DatetimeIndex]:
        """
        Sometimes a time series is of pandas type "object" just because the time-zone information
        is not well-read initially. Such a series can't be translated to a pd.DatetimeIndex.
        This function makes sure the time zone information of the input series is set to the input
        tz_info for each row and also for the series.

        Example :
        time_series = a time_series with some times at timezone +01:00 (French winter time)
                      and others at timezone +02:00 (French summer)
                      So its pandas dtype is "object"
        tz = pytz.timezone("Europe/Paris")

        :param time_series: a series with tz-aware date-times. If a datetime-index is passed, the function
                            process it
        :param tz_info: a str or a datetime.tzinfo
        :return: a DatetimeIndex of dtype: datetime[ns, tzinfo]
        """

        if isinstance(tz_info, str):
            tz_info = pytz.timezone(tz_info)
        if not isinstance(tz_info, datetime.tzinfo):
            raise TypeError(
                "parameter 'tzinfo' should be of type str or datetime.tzinfo"
            )

        # map pandas.Timestamp with the good tzinfo
        result = time_series.map(lambda x: x.astimezone(tz_info))

        return result
