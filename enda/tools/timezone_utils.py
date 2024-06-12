"""This module contains various functions for dealing with timezones in temporal data in Python"""

import datetime
from typing import Union

import pandas as pd
import pytz
from dateutil.relativedelta import relativedelta
from pandas.api.types import is_string_dtype

import enda.tools.decorators


class TimezoneUtils:
    """
    This class contains various methods for helping deal with timezones
    """

    # ---- Handle DST

    @staticmethod
    def is_timezone_aware(dt: Union[datetime.datetime, pd.Timestamp]) -> bool:
        """
        Check if a datetime /timestamp is timezone-aware or not
        """
        return dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None

    @classmethod
    def add_interval_to_day_dt(
            cls,
            day_dt:Union[datetime.datetime, pd.Timestamp],
            interval: relativedelta
    ) -> Union[datetime.datetime, pd.Timestamp]:
        """
        Adds an interval (not more precise than a day) to a day,
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

    @classmethod
    def add_interval_to_date_object(
            cls,
            date_obj: Union[datetime.datetime, pd.Timestamp, datetime.date],
            interval: relativedelta
    ) -> Union[datetime.datetime, pd.Timestamp, datetime.date]:

        """
        If day is a date or not timezone aware, we simply add a correct relativedelta object.
        Else, we have to call add_interval_to_day_dt() to handle the DST change time
        :param date_obj: a date, a datetime, or a pd.Timestamp
        :param interval: an interval of type relativedelta not more precise than a day.
            Note that the more general Timeseries.add_timedelta() wraps this function and should be preferred.
        """

        # we do not enforce interval to really be a day if we do not need to handle timezones.
        # in that case, that check is delegated to add_interval_to_day_dt()

        if isinstance(date_obj, (datetime.datetime, pd.Timestamp)):

            if TimezoneUtils.is_timezone_aware(date_obj):

                # build a tz-aware datetime object from the timestamp or datetime object
                day_obj = date_obj.tzinfo.localize(
                    datetime.datetime(date_obj.year, date_obj.month, date_obj.day, 0, 0, 0, 0),
                )

                # this is required to manage DST changes
                new_day_obj = TimezoneUtils.add_interval_to_day_dt(day_obj, interval)

                # before resetting the intraday part of the date_object
                new_day_obj = new_day_obj.replace(hour=date_obj.hour,
                                                  minute=date_obj.minute,
                                                  second=date_obj.second,
                                                  microsecond=date_obj.microsecond
                                                  )

                if isinstance(date_obj, pd.Timestamp):
                    return pd.Timestamp(new_day_obj)

                return new_day_obj

        return date_obj + interval

    # --- Manage series

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

    @staticmethod
    def read_csv_and_set_tz_aware_columns(
            file_path: str,
            time_cols_list: list[str],
            tz_info: Union[str, datetime.tzinfo],
            **kwargs
    ) -> pd.DataFrame:
        """
        Given a file path, read it and set datetime columns to tz-aware columns
        :param file_path: path to a csv file
        :param time_cols_list: list of columns that contain time information to set to the correct time zone.
        :param tz_info: a str or a datetime.tzinfo
        :return: a Dataframe with date columns translated into the correct target timezone.
        """

        # parse the file as a dataframe
        result_df = pd.read_csv(file_path, parse_dates=time_cols_list, **kwargs)

        for time_col in time_cols_list:
            if is_string_dtype(result_df[time_col]):
                # time columns have not been turned to a datetime-like column by read_csv() function
                # if it's because of a mixture of timezone, next time correct it.
                # else, it's because the column contains information that cannot be turned to datetime, and next
                # snippet fail.
                result_df[time_col] = TimezoneUtils.convert_dtype_from_object_to_tz_aware(
                    result_df[time_col],
                    tz_info=tz_info
                )
            else:
                # in that case it's been correctly read by read_csv().
                # next line set it to the correct timezone
                result_df[time_col] = TimezoneUtils.set_timezone(result_df[time_col], tz_info=tz_info)

        return result_df
