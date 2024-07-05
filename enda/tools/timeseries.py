"""This module contains functions to help manipulating timeseries"""

import datetime
import re
from typing import Union

import pandas as pd
import pytz
from dateutil.relativedelta import relativedelta

import enda.tools.decorators
import enda.tools.resample
import enda.tools.timezone_utils


class TimeSeries:
    """
    This class contains methods for manipulating timeseries
    """

    # ------------------------
    # Frequencies / Timedelta
    # ------------------------

    # mapping of common frequencies codes to approximate number of days
    # checkout https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    # to get more information
    FREQ_UNIT_TO_DAYS = {
        "ms": 1 / (24 * 3600 * 1000),  # millisecond
        "s": 1 / (24 * 3600),  # second
        "S": 1 / (24 * 3600),  # second
        "min": 1 / 1440,  # minute
        "MIN": 1 / 1440,  # minute
        "T": 1 / 1440,  # minute
        "H": 1 / 24,  # hour
        "h": 1 / 24,  # hour
        "D": 1,  # day
        "d": 1,  # day
        "B": 1,  # business day
        "b": 1,  # business day
        "W": 7,  # week
        "w": 7,  # week
        "W-SUN": 7,  # week
        "W-MON": 7,  # week
        "M": 30.4,  # average number of days in a month
        "MS": 30.4,  # average number of days in a month
        "Q": 91,  # quarter (approximation)
        "A": 365,  # assuming 365 days in a year (approximation)
        "Y": 365,  # assuming 365 days in a year (approximation)
    }

    # mapping of common frequencies codes to approximate number of seconds
    FREQ_UNIT_TO_SECONDS = {
        "ms": 1 / 1000,  # millisecond
        "s": 1,  # second
        "S": 1,  # second
        "min": 60,  # minute
        "MIN": 60,  # minute
        "T": 60,  # minute
        "H": 3600,  # hour
        "h": 3600,  # hour
        "D": 86400,  # number of seconds in a day (out of DST)
        "d": 86400,  # day (out of DST)
        "B": 86400,  # business day (out of DST)
        "b": 86400,  # business day (out of DST)
        "W": 604800,  # week (out of DST)
        "w": 604800,  # week (out of DST)
        "W-SUN": 604800,  # week (out of DST)
        "W-MON": 604800,  # week (out of DST)
        "M": 2626560,  # average number of seconds in a month
        "MS": 2626560,  # average number of seconds in a month
        "Q": 7862400,  # quarter (approximation)
        "A": 31536000,  # assuming 365 days in a year (approximation)
        "Y": 31536000,  # assuming 365 days in a year (approximation)
    }

    @staticmethod
    def split_amount_and_unit_from_freq(
        freq: Union[str, pd.Timedelta]
    ) -> tuple[int, str]:
        """
        Given a frequency as a string, such as '1D', '10min', '-3MS'
        extract the amount (e.g. 1, 10, -3) and the unit part (e.g. 'D', 'min', 'MS'), capitalized.
        :param freq: the frequency
        :return: the unit part of that frequency, and the number of units
        """

        # if freq is a pd.Timedelta, it should be convertible to a string
        if isinstance(freq, pd.Timedelta):
            freq = pd.tseries.frequencies.to_offset(freq).freqstr

        # separate the freq string in numeric and alphabetic parts
        freq_string_parts = re.findall(r"(\d+|\D+)", freq)

        # several cases are possible to take into account the +/- sign
        if len(freq_string_parts) == 3:
            # in that case, first character must be a + or - sign, eg. '-1D'
            sign = 1
            if freq_string_parts[0] == "-":
                sign = -1
            elif freq_string_parts[0] == "+":
                pass
            else:
                raise ValueError("First character is not a plus or minus sign")

            # numeric part - error if type is wrong
            numeric_part_int = sign * int(freq_string_parts[1])

            # freq part  - error if type is wrong
            freq_unit_str = freq_string_parts[-1]

        elif len(freq_string_parts) == 2:
            # in that case, first group must be a number, second group must be a unit e.g. '2MS'
            numeric_part_int = int(freq_string_parts[0])
            freq_unit_str = freq_string_parts[-1]

        elif len(freq_string_parts) == 1:
            # in that case, there is no number but first character might be a sign. e.g.  or 'MS' or '-D'
            if freq_string_parts[0][0] == "-":
                numeric_part_int = -1
                freq_unit_str = freq_string_parts[0][1:]
            elif freq_string_parts[0][0] == "+":
                numeric_part_int = 1
                freq_unit_str = freq_string_parts[0][1:]
            else:
                numeric_part_int = 1
                freq_unit_str = freq_string_parts[0]

        else:
            raise ValueError(f"freq {freq} is not valid")

        # simple check
        if freq_unit_str not in TimeSeries.FREQ_UNIT_TO_SECONDS:
            raise ValueError(
                f"Unknown frequency unit {freq_unit_str} obtained from frequency {freq}"
            )

        return numeric_part_int, freq_unit_str

    @staticmethod
    def is_regular_freq(freq: str) -> bool:
        """
        Return a boolean that indicates the frequency is something regular or not
        eg 'D' 'min' are regular, 'MS', 'Y', 'Q' are not.
        For instance, irregular frequencies cannot be turned to pd.Timedelta elements.
        Note that if the frequency is a pd.Timedelta, it is always regular
        :param freq: the frequency given as a string or a pd.Timedelta
        :return: a boolean that indicates whether the considered frequency is 'even'
        """
        _, freq_unit_str = TimeSeries.split_amount_and_unit_from_freq(freq)
        if freq_unit_str in ["M", "MS", "Q", "A", "Y"]:
            return False
        return True

    @staticmethod
    def freq_as_approximate_nb_days(freq: Union[str, pd.Timedelta]) -> int:
        """
        Map pandas freq string to an approximate length of days.
        This serves to compare freq strings, such as '1D', '3MS', and so on.
        It replaces the calls to total_seconds() possibly used only with pd.Timedelta().
        Note that the computation is exact if a pd.Timedelta is given.
        :param freq: the frequency as a string, such as '10min', '3MS', or a pd.Timedelta.
        :return: a sometimes approximate length of the provided frequency in days
        """

        # separate the freq string in numeric and alphabetic parts
        numeric_part_int, freq_unit_str = TimeSeries.split_amount_and_unit_from_freq(
            freq
        )

        # error if is not in FREQ_UNIT_TO_DAYS
        return numeric_part_int * TimeSeries.FREQ_UNIT_TO_DAYS[freq_unit_str]

    @staticmethod
    def freq_as_approximate_nb_seconds(freq: Union[str, pd.Timedelta]) -> int:
        """
        Map pandas freq string to an approximate number of seconds.
        This serves to compare freq strings, such as '1D', '3MS', and so on.
        It replaces the calls to total_seconds() possibly used only with pd.Timedelta().
        Note that the computation is exact if a pd.Timedelta is given.
        :param freq: the frequency as a string, such as '10min', '3MS', or a pd.Timedelta.
        :return: a sometimes approximate length of the provided frequency in days
        """

        # separate the freq string in numeric and alphabetic parts
        numeric_part_int, freq_unit_str = TimeSeries.split_amount_and_unit_from_freq(
            freq
        )

        # error if is not in FREQ_UNIT_TO_SECONDS
        return numeric_part_int * TimeSeries.FREQ_UNIT_TO_SECONDS[freq_unit_str]

    @staticmethod
    def add_timedelta(
        date: Union[datetime.date, datetime.datetime, pd.Timestamp],
        timedelta: Union[str, pd.Timedelta],
    ) -> Union[datetime.date, datetime.datetime, pd.Timestamp]:
        """
        Define how to add a timedelta according to the way it's provided (string, timedelta), regular or irregular
        absolute length.
        :param date: a date, provided as a pd.Timestamp (naive or tz-aware), a date, a datetime
        :param timedelta:  a timedelta, given as a freq string (e.g. '2MS') or a pd.Timedelta object.
        """
        if not isinstance(timedelta, (pd.Timedelta, str)):
            raise TypeError("timedelta must be of type Timedelta or string ")

        timedelta_str = timedelta
        if isinstance(timedelta, pd.Timedelta):
            timedelta_str = pd.tseries.frequencies.to_offset(timedelta).freqstr

        # if it's a string, it is not necessarily convertible to timedelta, as
        # it might be an irregular length (month, year, quarter...)
        # in that case, we handle these cases before defaulting to pd.to_timedelta()
        (
            numeric_part_int,
            unit_freq_str,
        ) = TimeSeries.split_amount_and_unit_from_freq(timedelta_str)

        if unit_freq_str in ["D", 'd', "B", "b"]:
            return enda.tools.timezone_utils.TimezoneUtils.add_interval_to_date_object(
                    date, relativedelta(days=numeric_part_int)
                )
        if unit_freq_str in ["W", 'w']:
            return enda.tools.timezone_utils.TimezoneUtils.add_interval_to_date_object(
                date, relativedelta(weeks=numeric_part_int)
            )
        if unit_freq_str in ["M", "MS"]:
            return enda.tools.timezone_utils.TimezoneUtils.add_interval_to_date_object(
                date, relativedelta(months=numeric_part_int)
            )
        if unit_freq_str in ["Y", "A"]:
            return enda.tools.timezone_utils.TimezoneUtils.add_interval_to_date_object(
                date, relativedelta(years=numeric_part_int)
            )
        if unit_freq_str in ["Q"]:
            raise ValueError(
                "Cannot simply add a quarter, it does not mean anything in general"
            )

        # there, it should be convertible to pd.Timedelta with no ambiguous interpretation
        # because we're under the day scale (no DST issue)
        timedelta = str(numeric_part_int) + unit_freq_str
        return date + pd.to_timedelta(timedelta)

    @staticmethod
    def subtract_timedelta(
        date: Union[datetime.date, datetime.datetime, pd.Timestamp],
        timedelta: Union[str, pd.Timedelta],
    ) -> Union[datetime.date, datetime.datetime, pd.Timestamp]:
        """
        Define how to subtract a timedelta according to the way it's provided (string, timedelta), regular or irregular
        absolute length.
        :param date: a date, provided as a pd.Timestamp (naive or tz-aware), a date, a datetime
        :param timedelta: a timedelta, given as a freq string (e.g. '2MS') or a pd.Timedelta object.
        """
        if not isinstance(timedelta, (pd.Timedelta, str)):
            raise TypeError("timedelta must be of type Timedelta or string ")

        timedelta_str = timedelta
        if isinstance(timedelta, pd.Timedelta):
            timedelta_str = pd.tseries.frequencies.to_offset(timedelta).freqstr

        # if it's a string, it is not necessarily convertible to timedelta, as
        # it might be an irregular length (month, year, quarter...)
        # in that case, we handle these cases before defaulting to pd.to_timedelta()
        (
            numeric_part_int,
            unit_freq_str,
        ) = TimeSeries.split_amount_and_unit_from_freq(timedelta_str)

        if unit_freq_str in ["D", 'd', "B", "b"]:
            return enda.tools.timezone_utils.TimezoneUtils.add_interval_to_date_object(
                    date, relativedelta(days=-numeric_part_int)
                )
        if unit_freq_str in ["W", 'w']:
            return enda.tools.timezone_utils.TimezoneUtils.add_interval_to_date_object(
                date, relativedelta(weeks=-numeric_part_int)
            )
        if unit_freq_str in ["M", "MS"]:
            return enda.tools.timezone_utils.TimezoneUtils.add_interval_to_date_object(
                date, relativedelta(months=-numeric_part_int)
            )
        if unit_freq_str in ["Y", "A"]:
            return enda.tools.timezone_utils.TimezoneUtils.add_interval_to_date_object(
                date, relativedelta(years=-numeric_part_int)
            )
        if unit_freq_str in ["Q"]:
            raise ValueError(
                "Cannot simply subtract a quarter, it does not mean anything in general"
            )

        # there, it should be convertible to pd.Timedelta with no ambiguous interpretation
        # because we're under the day scale (no DST issue)
        timedelta = str(numeric_part_int) + unit_freq_str
        return date - pd.to_timedelta(timedelta)

    # ------------
    # Time-series
    # ------------

    @staticmethod
    @enda.tools.decorators.handle_series_as_datetimeindex(
        arg_name="dti", return_input_type=False
    )
    def has_nan_or_empty(dti: pd.DatetimeIndex) -> bool:
        """
        Check whether a datetime index has NaN or is empty
        :param dti: pd.DatetimeIndex to investigate
        """

        if not isinstance(dti, pd.DatetimeIndex):
            raise TypeError("parameter 'dti' should be a pandas.DatetimeIndex")

        if dti.isna().sum() != 0:
            return True

        if len(dti) == 0:
            return True

        return False

    @staticmethod
    @enda.tools.decorators.handle_series_as_datetimeindex(
        arg_name="dti", return_input_type=False
    )
    def find_nb_records(dti: pd.DatetimeIndex, skip_duplicate_timestamps=False) -> int:
        """
        Compute the number of records
         :param dti: pd.DatetimeIndex to investigate
        :param skip_duplicate_timestamps: bool that indicates if duplicates must be considered
               or not when computing the number of records
        :return: the number of records in the time series
        """

        if skip_duplicate_timestamps:
            dti = dti.drop_duplicates()

        return len(dti)

    @staticmethod
    @enda.tools.decorators.handle_series_as_datetimeindex(
        arg_name="dti", return_input_type=False
    )
    def find_gap_distribution(
        dti: pd.DatetimeIndex, skip_duplicate_timestamps=False
    ) -> pd.Series:
        """
        Find frequencies in a pd.DatetimeIndex. The function computes
        all timedelta between successive indexes, and count them. It returns an ordered
        pd.Series, with the most commonly found in index 0.


        :param dti: pd.DatetimeIndex to investigate
        :param skip_duplicate_timestamps: bool that indicates if duplicates must be considered
               or not when the gap distribution is considered.
        :return: a pd.Series with all frequencies found, and their cardinality.
                eg. >>> find_frequencies(dti)
                #    01:00:00    1181
                #    02:00:00     499
                #    03:00:00     180
                #    ....
        """

        if TimeSeries.has_nan_or_empty(dti):
            raise ValueError("Nan or empty dti")

        dti = dti.sort_values()

        if skip_duplicate_timestamps:
            dti = dti.drop_duplicates()

        # compute the gap distribution
        gap_dist = (pd.Series(dti[1:]) - pd.Series(dti[:-1])).value_counts()

        return gap_dist

    @staticmethod
    @enda.tools.decorators.handle_series_as_datetimeindex(
        arg_name="dti", return_input_type=False
    )
    def find_most_common_frequency(
        dti: pd.DatetimeIndex, skip_duplicate_timestamps=False
    ) -> str:
        """
        Find most common frequency in pd.DatetimeIndex. If several frequencies are found, it returns the most common.
        The function checks the index is not null,
        :param dti: pd.DatetimeIndex to investigate
        :param skip_duplicate_timestamps: if True, the function will not consider duplicate timestamps
                                          in the calculation.
        :return : the most common frequency found in the datetime index as a string
        """

        if TimeSeries.has_nan_or_empty(dti):
            raise ValueError("Nan or empty datetime index")

        dti = dti.sort_values()

        if skip_duplicate_timestamps:
            dti = dti.drop_duplicates()

        freq = None
        nb_records = TimeSeries.find_nb_records(dti)
        if nb_records == 1:
            # we cannot get a frequency out of the series, as it's a single element series !
            return ""

        if nb_records >= 3:
            # try getting single frequency from pandas if the series is big enough
            freq = pd.infer_freq(dti)

        if freq is None:
            # it means there is some kind of irregularity in the datetimeindex, as pandas could not find it or
            # e.g. there might be more than one frequency in the dataframe.
            # note that there must be at least two records in the datetimeindex at that point

            # get all gap distributions in the dataframe
            gap_distribution = TimeSeries.find_gap_distribution(dti)

            # return the more common
            freq = gap_distribution.index[0]

            # turn it to a string
            freq = pd.tseries.frequencies.to_offset(freq).freqstr

        # if the freq is 'one' of a particular unit, pandas just returns the unit,
        # This is not practical for other functions, so we add 1 to the string freq.
        # e.g. the freq is one day, pandas may return 'D'
        # we transform it to '1D' in that case
        freq = freq if freq[0].isdecimal() else "1" + freq

        return freq

    @staticmethod
    @enda.tools.decorators.handle_series_as_datetimeindex(
        arg_name="dti", return_input_type=True
    )
    def find_duplicates(dti: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """
        Check for duplicates in the timeseries
        :param dti: a datetime index, without NaN
        :return: duplicates
        """
        return dti[dti.duplicated()]

    @staticmethod
    @enda.tools.decorators.handle_series_as_datetimeindex(
        arg_name="dti", return_input_type=True
    )
    def find_extra_points(
        dti: pd.DatetimeIndex, expected_freq: Union[str, pd.Timedelta] = None
    ) -> pd.DatetimeIndex:
        """
        Check for extra data points in the timeseries, i.e. data points that are
        outside the expected frequency (or the most common frequency if not known) of data points.
        This is based on the min and max of the index, which means no extra point might be found
        if min or max are the extra points (in that case, there might be missing periods!)
        :param dti: a datetime index, without NaN
        :param expected_freq: the expected freq of the datetime index if known, default None.
        :return: extra data points
        """

        # get freq if None
        if not expected_freq:
            expected_freq = TimeSeries.find_most_common_frequency(
                dti, skip_duplicate_timestamps=True
            )

        # inspect extra periods: we must work on a sorted dti without index
        data_index = dti.drop_duplicates().sort_values()
        expected_index = pd.date_range(
            data_index[0], data_index[-1], freq=expected_freq
        )
        extra_points = data_index.difference(expected_index)

        return extra_points

    @staticmethod
    def find_duplicates_and_extra_points(
        dti: Union[pd.DatetimeIndex, pd.Series],
        expected_freq: Union[str, pd.Timedelta] = None,
    ) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
        """
        Check for extra data points in the timeseries, i.e. data points that are duplicated and/or data points that are
        outside the expected frequency (or the most common frequency if not known) of data points.
        :param dti: a time_series
        :param expected_freq: the expected freq of the datetime index if known, default None.
        :return: extra data points
        """
        return TimeSeries.find_duplicates(dti), TimeSeries.find_extra_points(
            dti, expected_freq
        )

    @staticmethod
    @enda.tools.decorators.handle_series_as_datetimeindex(
        arg_name="dti", return_input_type=True
    )
    def find_missing_points(
        dti: pd.DatetimeIndex,
        expected_freq: Union[str, pd.Timedelta] = None,
        expected_start_datetime=None,
        expected_excl_end_datetime=None,
    ) -> pd.DatetimeIndex:
        """
        Check for missing periods in the time series with an expected frequency.
        :param dti: a datetime index, without NaN
        :param expected_freq: the expected freq of the datetime index if known, default None.
        :param expected_start_datetime: the expected start time of the datetime index, default None
        :param expected_excl_end_datetime: the expected exclusive end time of the datetime index, default None
        :return: missing points as a datetime index
        """

        if TimeSeries.has_nan_or_empty(dti):
            raise ValueError("Nan or empty datetime index")

        # process freq
        if not expected_freq:
            expected_freq = TimeSeries.find_most_common_frequency(dti)

        # drop duplicates and sort
        dti = dti.drop_duplicates().sort_values()

        # start date
        if expected_start_datetime is not None:
            if not isinstance(expected_start_datetime, pd.Timestamp):
                raise TypeError("expected_start_datetime must be a pandas.Datetime")
            start_datetime = expected_start_datetime
        else:
            start_datetime = dti[0]

        # end date
        if expected_excl_end_datetime is not None:
            if not isinstance(expected_excl_end_datetime, pd.Timestamp):
                raise TypeError("expected_end_datetime must be a pandas.Datetime")
            end_datetime = TimeSeries.subtract_timedelta(
                expected_excl_end_datetime, expected_freq
            )
        else:
            end_datetime = dti[-1]

        # compute the expected index
        expected_index = pd.date_range(start_datetime, end_datetime, freq=expected_freq)
        missing_points = expected_index.difference(dti)

        return missing_points

    @staticmethod
    @enda.tools.decorators.handle_series_as_datetimeindex(
        arg_name="dti", return_input_type=False
    )
    def collapse_to_periods(
        dti: pd.DatetimeIndex, freq: [str, pd.Timedelta]
    ) -> list[
        tuple[
            Union[pd.Timestamp, datetime.date, datetime.datetime],
            Union[pd.Timestamp, datetime.date, datetime.datetime],
        ]
    ]:
        """
        Given a datetime index and a frequency, it gives the list of regular periods found in the datetime index, as
        a collection of tuples that contain the start time and exclusive time of the periods found. More precisely,
        if a timestamp is missing (ie if t is present but t + freq is missing), a new period is defined (t
        becomes the end time of the current period, and the next timestamp found after t defines the beginning of a new
        period).
        Duplicates are dropped in the operation.
        :param dti: a datetime index, without NaN
        :param freq: the freq of the datetime index used to find missing periods.
        :return: a list of tuples that contain the regular periods in the datetime index (defined as the start and
        inclusive end times of the period).
        """

        if dti.shape[0] == 0:
            return []

        # drop duplicates and sort the index
        dti = dti.drop_duplicates().sort_values()

        current_period_start = dti[0]
        periods_list = []
        for i in range(1, dti.shape[0]):
            if TimeSeries.add_timedelta(dti[i - 1], freq) != dti[i]:
                # end the current period and start a new one
                periods_list.append((current_period_start, dti[i - 1]))
                current_period_start = dti[i]

        periods_list.append((current_period_start, dti[-1]))  # Don't forget last period

        return periods_list

    @staticmethod
    @enda.tools.decorators.handle_series_as_datetimeindex(
        arg_name="dti", return_input_type=False
    )
    def find_missing_periods(
        dti: pd.DatetimeIndex,
        expected_freq=None,
        expected_start_datetime=None,
        expected_excl_end_datetime=None,
    ) -> list:
        """
        Find missing periods in a datetimeIndex. It finds all missing points, and collapse them
        in periods
        :param dti: a datetime index, without NaN
        :param expected_freq: the expected freq of the datetime index if known, default None.
        :param expected_start_datetime: the expected start time of the datetime index, default None
        :param expected_excl_end_datetime: the expected exclusive end time of the datetime index, default None
        :return: missing periods as a list of tuples of start and end times of the missing periods.
        """

        freq = expected_freq
        if expected_freq is None:
            freq = TimeSeries.find_most_common_frequency(dti)

        missing_points = TimeSeries.find_missing_points(
            dti,
            expected_freq=freq,
            expected_start_datetime=expected_start_datetime,
            expected_excl_end_datetime=expected_excl_end_datetime,
        )

        return TimeSeries.collapse_to_periods(missing_points, freq=freq)

    @staticmethod
    @enda.tools.decorators.handle_series_as_datetimeindex(
        arg_name="dti", return_input_type=False
    )
    def has_single_frequency(
        dti: pd.DatetimeIndex,
        variable_duration_freq_included: bool = True,
        skip_duplicate_timestamps=False,
    ) -> bool:
        """
        Return True if the provided datetime index has a single frequency, i.e.
        does not have missing periods, extra points, nor change of frequency
        :param dti: pd.DatetimeIndex to investigate
        :param variable_duration_freq_included: 'Frequency' may have a double meaning, whether fixed-time frequencies
                                           are considered or not. For instance, a frequency of one month leads to
                                           intervals with different absolute length (in terms of days, and seconds).
                                           Setting True means the datetimeindex is inspected with absolute duration
                                           in mind. It is useful when we expect:
                                           * an x-min difference between timestamps
                                           * an x-day difference between dates
        :param skip_duplicate_timestamps: If True, the function will not consider duplicate timestamps
        :return: True if the data has a clean single frequency defined.
        """

        if skip_duplicate_timestamps:
            dti = dti.drop_duplicates()

        # easy case: check the exact differences in duration between timestamps / dates
        if variable_duration_freq_included is False:
            return len(TimeSeries.find_gap_distribution(dti)) == 1

        # else, it might be more tricky, because the frequency may be irregular in terms of total_seconds
        # (e.g. when freq is months, years, or even days because of change of hour). We need to rely on pd.infer_freq().
        # Safer way is to reconstruct a time-series index from scratch using the most common frequency found in the
        # input index, and compare it to the index.

        # most common freq
        most_common_freq = TimeSeries.find_most_common_frequency(dti)

        # we rely on find extra points and missing points to detect it.
        extra_points = TimeSeries.find_extra_points(dti, expected_freq=most_common_freq)
        if len(extra_points) > 0:
            return False

        missing_points = TimeSeries.find_missing_points(
            dti, expected_freq=most_common_freq
        )
        if len(missing_points) > 0:
            return False

        return True

    # ----------------------------------
    # DEPRECATED FUNCTIONS AND CALL
    # ----------------------------------

    @classmethod
    @enda.tools.decorators.warning_deprecated_name(
        namespace_name="TimeSeries",
        new_namespace_name="TimezoneUtils",
        new_function_name="convert_dtype_from_object_to_tz_aware",
    )
    def align_timezone(
        cls,
        time_series: pd.Series,
        tzinfo: Union[
            str,
            pytz.timezone,
        ],
    ):
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
        :param tzinfo: a str or a datetime.tzinfo
        :return: a DatetimeIndex of dtype: datetime[ns, tzinfo]
        """
        return pd.DatetimeIndex(
            enda.tools.timezone_utils.TimezoneUtils.convert_dtype_from_object_to_tz_aware(
                time_series=time_series, tz_info=tzinfo
            )
        )

    @classmethod
    @enda.tools.decorators.warning_deprecated_name(
        namespace_name="TimeSeries",
        new_function_name="find_missing_periods and find_extra_points",
    )
    def find_missing_and_extra_periods(
        cls,
        dti,
        expected_freq=None,
        expected_start_datetime=None,
        expected_end_datetime=None,
    ):
        """
        Check for missing and extra data points
        :param dti: a series of type DatetimeIndex, in ascending order and without duplicates or NaNs
        :param expected_freq: pandas formatted frequency. If None is given, will infer the frequency, taking the
            most common gap between 2 consecutive points.
        :param expected_start_datetime: a pandas.Datetime, if None is given, will take dti[0]
        :param expected_end_datetime: a pandas.Datetime, if None is given, will take dti[-1]
        :return: Missing and extra data points collapsed in periods
        """

        # for the sake of compatibility
        if not dti.equals(dti.sort_values(ascending=True)):
            raise ValueError("given dti is not in ascending order")

        if dti.duplicated().sum() != 0:
            raise ValueError("given dti has duplicates.")

        freq = TimeSeries.find_most_common_frequency(dti)
        missing_periods = TimeSeries.find_missing_periods(
            dti,
            expected_freq=expected_freq,
            expected_start_datetime=expected_start_datetime,
            expected_excl_end_datetime=expected_end_datetime,
        )
        extra_points = TimeSeries.find_extra_points(dti)
        return pd.to_timedelta(freq), missing_periods, extra_points

    @classmethod
    @enda.tools.decorators.warning_deprecated_name(
        namespace_name="TimeSeries", new_function_name="collapse_to_periods"
    )
    def collapse_dt_series_into_periods(
        cls, dti: pd.DatetimeIndex, freq: [str, pd.Timedelta]
    ) -> list[
        tuple[
            Union[pd.Timestamp, datetime.date, datetime.datetime],
            Union[pd.Timestamp, datetime.date, datetime.datetime],
        ]
    ]:
        """
        Given a datetime index and a frequency, it gives the list of regular periods found in the datetime index, as
        a collection of tuples that contain the start time and exclusive time of the periods found. More precisely,
        if a timestamp is missing (ie if t is present but t + freq is missing), a new period is defined (t
        becomes the end time of the current period, and the next timestamp found after t defines the beginning of a new
        period).
        :param dti: a datetime index, without NaN
        :param freq: the freq of the datetime index used to find missing periods.
        :return: a list of tuples that contain the regular periods in the datetime index (defined as the start and
        inclusive end times of the period).
        """

        # legacy: check duplicates and multiple of period
        for i in range(1, dti.shape[0]):
            if dti[i] <= dti[i - 1]:
                raise ValueError("dti must be sorted and without duplicates!")

            if (dti[i] - dti[i - 1]).total_seconds() % pd.to_timedelta(
                freq
            ).total_seconds() != 0:
                raise ValueError(
                    f"Timedelta between {dti[i - 1]} and {dti[i]} is not a multiple of freq ({freq})."
                )

        return TimeSeries.collapse_to_periods(dti=dti, freq=freq)

    @staticmethod
    @enda.tools.decorators.warning_deprecated_name(
        namespace_name="TimeSeries", new_function_name="find_most_common_frequency"
    )
    def get_timeseries_frequency(index: pd.DatetimeIndex):
        """
        Retrieve the frequency of a pandas dataframe's index.
        """
        if not TimeSeries.has_single_frequency(index):
            raise ValueError("Found several frequencies in the datetimeIndex")

        return TimeSeries.find_most_common_frequency(dti=index)

    @staticmethod
    @enda.tools.decorators.warning_deprecated_name(
        namespace_name="TimeSeries",
        new_namespace_name="Resample",
        new_function_name="upsample_and_interpolate",
    )
    def interpolate_freq_to_sub_freq_data(
        df: pd.DataFrame,
        freq: [str, pd.Timedelta],
        tz: [str, datetime.tzinfo],
        index_name: str = None,
        method: str = "linear",
        enforce_single_freq=True,
    ):
        """
        Interpolate dataframe data on a smaller frequency than the one initially defined
        in the dataframe
        The original index of the data must have a well-defined frequency, i.e. it must be
        able to retrieve its frequency with inferred_freq
        :param df: pd.DataFrame
        :param freq: a frequency e.g. 'H', '30min', '15min', etc.
        :param tz: the target time zone.
        :param index_name: name to give to the new index. Usually going from 'date' to 'time'.
        :param method: how are data interpolated between two consecutive dates (e.g. 'ffill', 'linear', etc.)
        :param enforce_single_freq: is there a single frequency in the original dataframe
        :return: pd.DataFrame
        """

        df = enda.tools.resample.Resample.upsample_and_interpolate(
            timeseries_df=df,
            freq=freq,
            method=method,
            forward_fill=False,
            is_original_frequency_unique=enforce_single_freq,
            index_name=index_name,
        )

        df = enda.tools.timezone_utils.TimezoneUtils.set_timezone(df, tz_info=tz)

        return df

    @staticmethod
    @enda.tools.decorators.handle_multiindex(arg_name="df")
    @enda.tools.decorators.warning_deprecated_name(
        namespace_name="TimeSeries", new_namespace_name="Resample"
    )
    def forward_fill_final_record(
        df: pd.DataFrame,
        gap_frequency: [str, pd.Timedelta],
        cut_off_frequency: [str, pd.Timedelta] = None,
    ):
        """
        Forward-fill last record
        """

        freq = TimeSeries.get_timeseries_frequency(df.index)
        if freq is None:
            raise ValueError("The dataframe has no frequency and cannot be extended")

        if (
            pd.to_timedelta(freq).total_seconds()
            > pd.to_timedelta(gap_frequency).total_seconds()
        ):
            raise ValueError(
                "Cannot extend the dataframe on a smaller frequency than itself"
            )

        df = enda.tools.resample.Resample.forward_fill_final_record(
            timeseries_df=df, gap_timedelta=gap_frequency, cut_off=cut_off_frequency
        )
        df.index.freq = freq
        return df

    @staticmethod
    @enda.tools.decorators.handle_multiindex(arg_name="df")
    @enda.tools.decorators.warning_deprecated_name(
        namespace_name="TimeSeries",
        new_namespace_name="Resample",
        new_function_name="upsample_and_interpolate",
    )
    def interpolate_daily_to_sub_daily_data(
        df: pd.DataFrame,
        freq: [str, pd.Timedelta],
        tz: [str, datetime.tzinfo],
        index_name: str = "time",
        method: str = "ffill",
    ):
        """
        Interpolate daily data in a dataframe (with a DatetimeIndex) to sub-daily data using a given method.
        The last daily record is resampled using a forward-fill in any case.
        :param df: pd.DataFrame
        :param freq: a frequency < 'D' (e.g. 'H', '30min', '15min', etc)
        :param tz: the time zone (None not accepted because important)
        :param index_name: name to give to the new index. Usually going from 'date' to 'time'.
        :param method: how are data interpolated between two consecutive dates (e.g. 'ffill', 'linear', etc)
        :return: pd.DataFrame
        """

        return enda.tools.resample.Resample.upsample_and_interpolate(
            timeseries_df=df,
            freq=freq,
            method=method,
            forward_fill=True,
            index_name=index_name,
            tz_info=tz,
        )

    @staticmethod
    @enda.tools.decorators.warning_deprecated_name(
        namespace_name="TimeSeries",
        new_namespace_name="Resample",
        new_function_name="downsample",
    )
    def average_to_upper_freq(
        df: pd.DataFrame,
        freq: [str, pd.Timedelta],
        tz: [str, datetime.tzinfo],
        index_name: str = None,
        enforce_single_freq=True,
    ):
        """
        Downsample data provided in a given dataframe with a DatetimeIndex, or a two-levels
        compatible Multiindex.
        The provided frequency serves as a basis to group the data and average.
        If the initial dataframe has no frequency, we raise an error.

        Example:

        1. Given df:
        time_index                value
        2021-01-01 00:00:00+01:00 1
        2021-01-01 00:12:00+01:00 2
        2021-01-02 00:00:00+01:00 3

        average_to_upper_freq(df, freq='1D'):
        2021-01-01 00:00:00+01:00 1.5
        2021-01-02 00:00:00+01:00 3
        """

        df = enda.tools.resample.Resample.downsample(
            df,
            freq=freq,
            agg_functions="mean",
            is_original_frequency_unique=enforce_single_freq,
            index_name=index_name,
        )

        df = enda.tools.timezone_utils.TimezoneUtils.set_timezone(df, tz_info=tz)

        return df
