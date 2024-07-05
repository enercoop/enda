"""This module contains functions for resampling timeseries"""

import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from typing import Optional, Union
import warnings

import enda.tools.timeseries
import enda.tools.timezone_utils
import enda.tools.decorators


class Resample:
    """This class contains methods to resample (up or down) timeseries DataFrames"""

    @staticmethod
    @enda.tools.decorators.handle_multiindex(arg_name="timeseries_df")
    def downsample(
            timeseries_df: pd.DataFrame,
            freq: Union[str, pd.Timedelta],
            groupby: list[str] = None,
            agg_functions: Union[dict, str] = "mean",
            origin: str = "start_day",
            is_original_frequency_unique: bool = False,
            index_name: str = None,
    ) -> pd.DataFrame:
        """
        Downsample a datetime-indexed pd.DataFrame to a provided frequency using one or several aggregation functions.
        The provided frequency must be greater than the original one in the dataframe index.
        :param timeseries_df: the dataframe with a datetime-like index
        :param freq: the aimed frequency. If given as a pd.Timedelta, it must be convertible to freq string
        :param groupby: If a list of columns is provided, the DataFrame will be grouped by specified columns
                        before resampling
        :param agg_functions: aggregate functions used for resampling passed to aggregate().
                              If a dict is provided, it must indicate an agg function for each column of the dataframe
                              If a string is provided, it will be used for all columns of the dataframe.
                              Default is 'mean' for all columns
        :param origin: the timestamp on which to adjust the grouping.
                       Default is 'start_day', i.e. origin is the first day at midnight of the timeseries
                       Other options are 'epoch', 'start', 'end', 'end_day'. For more details, see
                       https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html
        :param is_original_frequency_unique: boolean, if True, ensure there's no missing value, no extra-point,
                                             and a single well-defined frequency in the original dataframe
        :param index_name: a name to give to the new index. For instance going from 'date' to 'time'.
        """

        if groupby is None:
            groupby = []

        if not isinstance(timeseries_df.index, pd.DatetimeIndex):
            raise TypeError("The dataframe index must be a DatetimeIndex")

        if isinstance(agg_functions, dict):
            cols_with_missing_function = set(timeseries_df.columns).difference(
                set(agg_functions.keys()).union(set(groupby))
            )
            if len(cols_with_missing_function) > 0:
                raise ValueError(
                    "Aggregation function to use has not been set for next columns "
                    f" of dataframe: {cols_with_missing_function}"
                )
        elif isinstance(agg_functions, str):
            agg_functions = {
                _: agg_functions
                for _ in list(set(timeseries_df.columns).difference(set(groupby)))
            }
        else:
            raise TypeError(
                f"agg_functions must be of type str or dict, found {type(agg_functions)} instead"
            )

        # If we need to group by other columns than the index, it means that there might be duplicates in time index.
        # That could lead 'find_most_common_frequency' to return null frequency as the most frequent.
        # Note this does NOT imply the duplicates are reset when downsampling !
        skip_duplicates_for_unique_freq_check = len(groupby) <= 0

        # get frequency from the initial dataframe
        # it return None only if the dataframe has a single record, else it returns the most common frequency
        original_freq = enda.tools.timeseries.TimeSeries.find_most_common_frequency(
            timeseries_df.index,
            skip_duplicate_timestamps=skip_duplicates_for_unique_freq_check,
        )

        if (
                original_freq
                and is_original_frequency_unique
                and (
                not enda.tools.timeseries.TimeSeries.has_single_frequency(
                    timeseries_df.index,
                    variable_duration_freq_included=True,
                    skip_duplicate_timestamps=skip_duplicates_for_unique_freq_check,
                )
        )
        ):
            raise RuntimeError("Frequency is not unique in the dataframe")

        # make sure we downsample
        if original_freq and (
                enda.tools.timeseries.TimeSeries.freq_as_approximate_nb_seconds(freq)
                < enda.tools.timeseries.TimeSeries.freq_as_approximate_nb_seconds(original_freq)
        ):
            raise RuntimeError(
                f"The required frequency {freq}"
                f" is smaller than the original one {original_freq}"
            )

        # resample using the aggregation function
        if len(groupby) == 0:
            resampled_df = timeseries_df.resample(freq, origin=origin).aggregate(
                agg_functions
            )
        else:
            columns_in_order_list = timeseries_df.columns
            resampled_df = (
                timeseries_df.groupby(by=groupby)
                .resample(freq, origin=origin)
                .aggregate(agg_functions)
                .reset_index(level=groupby)
                .sort_index()  # reorder index
                .filter(columns_in_order_list)  # reorder columns
            )

        # change the index_name if required
        if index_name is not None:
            resampled_df.index.name = index_name

        return resampled_df

    @staticmethod
    @enda.tools.decorators.handle_multiindex(arg_name="timeseries_df")
    def upsample_and_divide_evenly(
            timeseries_df: pd.DataFrame,
            freq: Union[str, pd.Timedelta],
            index_name: str = None,
            tz_info: Union[str, datetime.tzinfo] = None,
    ) -> pd.DataFrame:
        """
        Upsample a datetime-indexed pd.DataFrame to a provided frequency and divide the values so that the sum
        of columns remains the same :
        {2023-01-01 00:00:00 : 50, 2023-01-01 00:30:00: 40} with 15 minutes resampling should give
        {2023-01-01 00:00:00 : 25, 2023-01-01 00:15:00 : 25,
         2023-01-01 00:30:00 : 20, 2023-01-01 00:45:00 : 20}
        This is intended for columns with numerical values that vary when changing frequency (such as energy and cost,
        but not power)
        Note there is a forward-filling of the last record before dividing, because it's the only behaviour that makes
        sense.
        The provided frequency must be lower than the original one in the dataframe index. Does not work if frequency
        in the original DataFrame isn't unique (for example : monthly values, as number of days in a month varies)
        :param timeseries_df: the dataframe with a datetime-like index
        :param freq: the aimed frequency. If given as string, it must be convertible to pd.Timedelta, eg '1D' or '1H'
        :param index_name: a name to give to the new index. For instance going from 'date' to 'time'.
        :param tz_info: the target time zone in case the index is resampled in another timezone / or if we want to
                        go from a non-localized timestamp (e.g. dates) to a localized one (e.g. date-times)
        :return: the upsampled timeseries dataframe
        """

        if not isinstance(timeseries_df.index, pd.DatetimeIndex):
            raise TypeError("The dataframe index must be a DatetimeIndex")

        # check it's not a one-row dataframe, because it cannot be upsampled.
        # Consider using forward_fill and give a gap_timedelta to achieve the objective.
        if enda.tools.timeseries.TimeSeries.find_nb_records(timeseries_df.index) <= 1:
            raise ValueError(
                "Cannot upsample an empty or single-record time series; because its initial"
                " frequency cannot be determined. Consider using 'forward_fill_last_record prior."
            )

        # get to copy
        timeseries_df = timeseries_df.copy()

        # set timezone in case it's been indicated
        if tz_info is not None:
            timeseries_df = enda.tools.timezone_utils.TimezoneUtils.set_timezone(
                timeseries_df, tz_info
            )

        # get frequency from the initial dataframe
        original_freq = enda.tools.timeseries.TimeSeries.find_most_common_frequency(
            timeseries_df.index
        )

        # if it's not a regular frequency, this function cannot work
        if (not enda.tools.timeseries.TimeSeries.is_regular_freq(original_freq)) or (
                not enda.tools.timeseries.TimeSeries.is_regular_freq(freq)
        ):
            raise ValueError(
                f"Cannot upsample and divide for an input dataframe with an irregular frequency, "
                f"(in terms of length). Here, we have original_freq={original_freq}, "
                f" and target freq = {freq}"
            )

        if not enda.tools.timeseries.TimeSeries.has_single_frequency(
                timeseries_df.index,
                variable_duration_freq_included=False,
                skip_duplicate_timestamps=False,
        ):
            raise ValueError("Frequency is not single-defined in the dataframe")

        # compute the frequency ration to know how to divide. It should be convertible to pd.Timedelta
        freq_ratio = (
                pd.to_timedelta(original_freq).total_seconds()
                / pd.to_timedelta(freq).total_seconds()
        )

        # make sure we upsample
        if freq_ratio < 1:
            raise ValueError(
                f"The required frequency {freq}"
                f" is higher than the original one {original_freq}"
            )

        # we must artificially add one timestep to the original dataframe to make sure we don't miss time steps in the
        # new frequency. With the example of the docstring, if we don't do this the 2023-01-01 00:45:00 won't be
        # added when resampling which will make the sum of values in the numerical columns false
        # Note this implies the last record of the resulting index is greater than the initial one.
        timeseries_df.loc[
            enda.tools.timeseries.TimeSeries.add_timedelta(
                timeseries_df.index.max(), original_freq
            )
        ] = 0
        timeseries_df = timeseries_df.resample(freq).ffill().div(freq_ratio).iloc[:-1]

        # change the index_name if required
        if index_name is not None:
            timeseries_df.index.name = index_name

        return timeseries_df

    @staticmethod
    @enda.tools.decorators.handle_multiindex(arg_name="timeseries_df")
    def upsample_monthly_data_and_divide_evenly(
            timeseries_df: pd.DataFrame,
            freq: Union[str, pd.Timedelta],
            tz_info: Union[str, datetime.tzinfo] = None,
    ) -> pd.DataFrame:
        """
        # @ TODO integrate to the previous function ?
        Upsample a monthly datetime-indexed pd.DataFrame to a provided frequency and averages the values so that the sum
        of columns remains the same :
        {2023-01-01 00:00:00 : 50, 2023-01-01 00:30:00: 40} with 15 minutes resampling should give
        {2023-01-01 00:00:00 : 25, 2023-01-01 00:15:00 : 25,
         2023-01-01 00:30:00 : 20, 2023-01-01 00:45:00 : 20}
        This is intended for columns with numerical values that vary when changing frequency (such as energy and cost,
        but not power)
        The provided frequency must be lower than the original one in the dataframe index.
        :param timeseries_df: the dataframe with a datetime-like index
        :param freq: the aimed frequency. If given as string, it must be convertible to pd.Timedelta, eg '1D' or '1H'
        :param tz_info: the target time zone in case the index is resampled in another timezone / or if we want to
                        go from a non-localized timestamp (eg. dates) to a localized one (e.g. date-times)
        :return: the upsampled timeseries dataframe
        """

        if not isinstance(timeseries_df.index, pd.DatetimeIndex):
            raise TypeError("The dataframe index must be a DatetimeIndex")

        # get frequency from the initial dataframe
        original_freq = enda.tools.timeseries.TimeSeries.find_most_common_frequency(
            timeseries_df.index
        )

        # check it's monthly
        (
            amount_months,
            month_freq,
        ) = enda.tools.timeseries.TimeSeries.split_amount_and_unit_from_freq(original_freq)
        if month_freq not in ["MS", "M"] or amount_months != 1:
            raise ValueError(f"Frequency should be monthly but found {original_freq}")

        resampled_df = timeseries_df.copy()

        # set timezone in case it's been indicated
        if tz_info is not None:
            resampled_df = enda.tools.timezone_utils.TimezoneUtils.set_timezone(
                resampled_df, tz_info
            )

        # resample using the aggregation function
        # We must artificially add one timestep to the original dataframe to make sure we don't miss time steps in the
        # new frequency. With the example of the docstring, if we don't do this the 2023-01-01 00:45:00 won't be
        # added when resampling which will make the sum of values in the numerical columns false
        resampled_df.loc[resampled_df.index.max() + relativedelta(months=1)] = 0
        resampled_df = (
            resampled_df.resample(freq)
            .ffill()
            .iloc[:-1]
            .groupby(lambda x: datetime.datetime(x.year, x.month, 1))
            .transform(lambda x: (x / x.count()))
        )

        return resampled_df

    @staticmethod
    @enda.tools.decorators.handle_multiindex(arg_name="timeseries_df")
    def upsample_and_interpolate(
            timeseries_df: pd.DataFrame,
            freq: Union[str, pd.Timedelta],
            method: str = "linear",
            forward_fill: bool = False,
            is_original_frequency_unique: bool = False,
            index_name: str = None,
            tz_info: Union[str, datetime.tzinfo] = None,
            expected_initial_freq: Optional[Union[str, pd.Timedelta]] = None,
            **kwargs
    ):
        """
        Upsample a datetime-indexed dataframe, and interpolate the columns data according to an interpolating method
        :param timeseries_df: the dataframe to resample
        :param freq: the target frequency (timedelta) e.g. 'H', '30min', '15min', etc.
        :param method: how are data interpolated between two consecutive dates (e.g. 'ffill', 'linear', etc.)
        :param forward_fill: If True, the upsampling is performed for the last element of the datetimeindex
        :param is_original_frequency_unique: check whether the frequency is unique in the initial dataframe
        :param index_name: a name to give to the new index. For instance going from 'date' to 'time'.
        :param tz_info: the target time zone in case the index is resampled in another timezone / or if we want to
                        go from a non-localized timestamp (e.g. dates) to a localized one (e.g. date-times)
        :param expected_initial_freq: the expected initial frequency of the dataframe. This serves to check the
            resampling, and for the particular case of one-row dataframe which must be forward-filled.
        :param **kwargs: arguments to pass to pandas.interpolate()
        :return: the upsampled timeseries dataframe
        """

        if not isinstance(timeseries_df.index, pd.DatetimeIndex):
            raise TypeError("The dataframe index must be a DatetimeIndex")

        initial_dtypes = timeseries_df.dtypes
        timeseries_df = timeseries_df.copy()

        # set timezone in case it's been indicated
        if tz_info is not None:
            timeseries_df = enda.tools.timezone_utils.TimezoneUtils.set_timezone(
                timeseries_df, tz_info
            )

        # special case of empty dataframe
        if timeseries_df.empty:
            timeseries_df.index.freq = freq
            return timeseries_df

        # special case of a one-row dataframe
        if enda.tools.timeseries.TimeSeries.find_nb_records(timeseries_df.index) == 1:
            # this is required to set the frequency
            timeseries_df = timeseries_df.resample(freq).last()
            if forward_fill:
                if expected_initial_freq is not None:
                    timeseries_df = Resample.forward_fill_final_record(
                        timeseries_df, gap_timedelta=expected_initial_freq, freq=freq
                    )
                else:
                    raise ValueError("One-row dataframe with no expected_initial_freq provided")

            return timeseries_df

        # get frequency from the initial dataframe
        original_freq = enda.tools.timeseries.TimeSeries.find_most_common_frequency(
            timeseries_df.index
        )

        if is_original_frequency_unique and (
                not enda.tools.timeseries.TimeSeries.has_single_frequency(
                    timeseries_df.index,
                    variable_duration_freq_included=True,
                    skip_duplicate_timestamps=False,
                )
        ):
            raise ValueError("Frequency is not unique in the dataframe")

        original_freq_seconds = enda.tools.timeseries.TimeSeries.freq_as_approximate_nb_seconds(original_freq)
        freq_seconds = enda.tools.timeseries.TimeSeries.freq_as_approximate_nb_seconds(freq)
        if (expected_initial_freq is not None) and (
                enda.tools.timeseries.TimeSeries.freq_as_approximate_nb_seconds(expected_initial_freq)
                != original_freq_seconds):
            raise ValueError(f"The initial expected freq of the dataframe index {expected_initial_freq} is not the "
                             f"same as the one actually found in the dataframe index {original_freq}")

        # make sure we upsample
        if freq_seconds >= original_freq_seconds:
            raise ValueError(f"The required frequency {freq} is higher than the original one {original_freq}")

        # resample and interpolate
        timeseries_df = timeseries_df.resample(freq).interpolate(method=method, **kwargs)

        # if method is 'ffill' or "bfill" or 'pad'
        if method in ["ffill", "bfill", "pad"]:
            timeseries_df = timeseries_df.astype(initial_dtypes)

        if forward_fill:
            # this serves to extend the dataframe, and resample using 'ffill' the last element
            # of the dataframe (not impacted by the previous resampling operation).
            # note that in that case, the timedelta must correspond to the initial frequency
            timeseries_df = Resample.forward_fill_final_record(
                timeseries_df, gap_timedelta=original_freq
            )

        # change the index_name if required
        if index_name is not None:
            timeseries_df.index.name = index_name

        return timeseries_df

    @staticmethod
    @enda.tools.decorators.handle_multiindex(arg_name="timeseries_df")
    def equal_sample_fillna(
            timeseries_df: pd.DataFrame,
            fill_value: any = None,
            method_filling: str = None,
            safe_run: bool = True,
            expected_freq: Optional[Union[str, pd.Timedelta]] = None,
            start_time: Optional[Union[datetime.date, datetime.datetime, pd.Timestamp]] = None,
            excl_end_time: Optional[Union[datetime.date, datetime.datetime, pd.Timestamp]] = None,
    ) -> pd.DataFrame:
        """
        Fill missing values in a datetime-indexed pd.DataFrame with a frequency.
        Two options are possible to fill the missing values:
            - 'fill_value' is used to fill with the same value (can be NaN, default behavior if set to None)
              all missing timestamp, calling Resampler.asfreq()
            - 'method_filling' fills missing values calling Resampler.fillna() if method_filling
              is in {‘pad’, ‘backfill’, ‘ffill’, ‘bfill’, ‘nearest’}

        :param timeseries_df: the dataframe with a datetime-like index
        :param fill_value: value to fill all missing values.
        :param method_filling: the method used to fill nan values.
        :param safe_run: boolean, if True, check there is no duplicates nor extra_periods
                         that the resampling could delete
        :param expected_freq: if known, the expected freq of the dataframe (and resulting)
        :param start_time: the index beginning. If None, it's the min of the dataframe index
        :param excl_end_time: the index excl end date. If None, it's the min of the dataframe index
        """

        # check both options (fill_with and method_filling) are not active simultaneously
        if fill_value is not None and method_filling is not None:
            raise ValueError(
                f"fill_value = {fill_value} which contradicts method_filling = {method_filling}. "
                "Only one can be not None."
            )

        if expected_freq is not None and safe_run is False:
            warnings.warn("expected_freq is given, but is ignored, as safe_run is set to False.")

        # get frequency from the initial dataframe.
        freq = enda.tools.timeseries.TimeSeries.find_most_common_frequency(
            timeseries_df.index, skip_duplicate_timestamps=True
        )

        if safe_run:

            # check expected_freq if given
            if expected_freq is not None:
                if enda.tools.timeseries.TimeSeries.freq_as_approximate_nb_days(freq) != \
                        enda.tools.timeseries.TimeSeries.freq_as_approximate_nb_days(expected_freq):
                    raise RuntimeError(
                        f"Equal-resampling with safe_run active, but expected_freq {expected_freq} is different "
                        f"from the most common freq found in the index {freq}"
                    )

            # check duplicates
            duplicates = enda.tools.timeseries.TimeSeries.find_duplicates(timeseries_df.index)
            if len(duplicates) > 0:
                raise RuntimeError(
                    f"Duplicates found in the index of the dataframe. The function would delete them, "
                    f"so it's unsafe to run. duplicates = {duplicates}"
                )

            # check extra-periods
            extra_periods = enda.tools.timeseries.TimeSeries.find_extra_points(
                timeseries_df.index
            )
            if len(extra_periods) > 0:
                raise ValueError(
                    f"Extra periods found in the index of the dataframe. The function would delete them,"
                    f" so it's unsafe to run. extra_periods = {extra_periods}"
                )

        # manage start_time
        if start_time is not None:
            start_time = pd.to_datetime(start_time)

            if start_time < timeseries_df.index.min():
                # add an extra row which is the same as the first one, but date is changed
                extra_row = timeseries_df.loc[[timeseries_df.index.min()], :].iloc[0].to_frame().T
                extra_row.index = [start_time]
                extra_row.index.name = timeseries_df.index.name
                timeseries_df = pd.concat([extra_row, timeseries_df], ignore_index=False)

            # anyway, call that
            timeseries_df = timeseries_df.loc[timeseries_df.index >= start_time]

        # resample
        if method_filling is not None:
            # resample using the method of Resampler.fillna() if provided
            # throw an error if method_filling is not a valid option
            result_df = timeseries_df.resample(freq).fillna(method=method_filling)
        else:
            # if method_filling is None, we use Resampler.asfreq() with fill_value
            result_df = timeseries_df.resample(freq).asfreq(fill_value=fill_value)

        # manage excl_end_time
        if excl_end_time is not None:
            excl_end_time = pd.to_datetime(excl_end_time)

            if excl_end_time > result_df.index.max():
                # forward-fill, function is already coded
                result_df = Resample.forward_fill_final_record(result_df, excl_end_time=excl_end_time)

            # anyway, call that
            result_df = result_df.loc[result_df.index < excl_end_time]

        return result_df

    @staticmethod
    @enda.tools.decorators.handle_multiindex(arg_name="timeseries_df")
    def forward_fill_final_record(
            timeseries_df: pd.DataFrame,
            gap_timedelta: Optional[Union[str, pd.Timedelta]] = None,
            cut_off: Optional[Union[str, pd.Timedelta]] = None,
            excl_end_time: Optional[Union[str, datetime.date, datetime.datetime, pd.Timestamp]] = None,
            freq: Optional[Union[str, pd.Timedelta]] = None,
    ) -> pd.DataFrame:
        """
        Forward-fill the final record of a regular datetime-indexed dataframe, keeping the frequency of
        the initial time series. This function not only add a timestamp, but resample with a forward-fill
        the timeseries until the desired timestamp.
        Naming 'final_ts' the max of the index of the initial dataframe timeseries_df, the resulting index of the
        final dataframe can be determined from two manners:
        - using the argument 'gap_timedelta' so that the new index goes until final_ts + gap_timedelta (excluded)
        - using the argument 'excl_end_final_date' so that the new index goes until excl_end_time (excluded)
        The resampling frequency is determined from the frequency of the initial dataframe.
        The extra parameter 'cut_off' can be used to set up a limit not to overpass. It means the resulting index is
        truncated to take into account the given cut-off.
        This function is typically used in junction with upsample_*() to forward-fill the last record.

        Here are some examples:

        1. Given timeseries_df:
        time_index                value
        2021-01-01 00:00:00+01:00 1
        2021-01-01 12:00:00+01:00 2
        2021-01-02 00:00:00+01:00 3

        forward_fill_final_record(timeseries_df, gap_timedelta='1D'):
        2021-01-01 00:00:00+01:00 1
        2021-01-01 12:00:00+01:00 2
        2021-01-02 00:00:00+01:00 3
        2021-01-02 12:00:00+01:00 3

        2. Given df:
        time_index                value
        2021-01-01 19:00:00+01:00 1
        2021-01-01 20:00:00+01:00 2
        2021-01-01 21:00:00+01:00 3
        2021-01-01 22:00:00+01:00 4

        forward_fill_final_record(timeseries_df, gap_timedelta='3H', cut_off=None):
        2021-01-01 19:00:00+01:00 1
        2021-01-01 20:00:00+01:00 2
        2021-01-01 21:00:00+01:00 3
        2021-01-01 22:00:00+01:00 4
        2021-01-01 23:00:00+01:00 4
        2021-01-02 00:00:00+01:00 4

        3. Given df:
        time_index                value
        2021-01-01 19:00:00+01:00 1
        2021-01-01 20:00:00+01:00 2
        2021-01-01 21:00:00+01:00 3
        2021-01-01 22:00:00+01:00 4

        forward_fill_final_record(timeseries_df, gap_timedelta='3H', cut_off='1D'):
        2021-01-01 19:00:00+01:00 1
        2021-01-01 20:00:00+01:00 2
        2021-01-01 21:00:00+01:00 3
        2021-01-01 22:00:00+01:00 4
        2021-01-01 23:00:00+01:00 4

        :param timeseries_df: input datetime-indexed dataframe to be forward-filled.
        :param gap_timedelta: The forward-filling is performed using this argument if provided.
                              Basically, the last index is extended until last_index + gap_timedelta
                              This option is incompatible with the setting of the excl_end_time argument
        :param cut_off: a timedelta that serves as a cut-off beyond which the final record is not extended
        :param excl_end_time: The forward-filling is performed using this argument if provided.
                              Basically, the last index is extended until excl_end_time - timeseries_df.index.freq
                              This option is incompatible with the setting of the gap_timedelta argument
        :param freq: The frequency to use to forward-fill. Usually, it's not given, and determined from the dataframe
            index directly, except if the dataframe is a one-row dataframe.
        :return: datetime-indexed dataframe similar to the initial one, with the last record being forward filled
                 using the initial frequency of the dataframe index.
        """

        if timeseries_df.empty:
            return timeseries_df

        if freq is None:
            freq = enda.tools.timeseries.TimeSeries.find_most_common_frequency(
                timeseries_df.index
            )

        # the last record is understood as the max time (not the last record of the index)
        incl_end_time = timeseries_df.index.max()

        if gap_timedelta and excl_end_time:
            raise ValueError(
                "Cannot provide both arguments excl_end_time and gap_timedelta at once"
            )
        if gap_timedelta is not None:
            extra_excl_end_time = enda.tools.timeseries.TimeSeries.add_timedelta(
                incl_end_time, gap_timedelta
            )
        elif excl_end_time is not None:
            extra_excl_end_time = pd.to_datetime(excl_end_time)
            if extra_excl_end_time <= incl_end_time:
                raise ValueError(
                    f"Provided extra_excl_end_time={extra_excl_end_time} <= "
                    f"max(timeseries_df.index) = {incl_end_time}. Cannot forward-fill."
                )
        else:
            raise ValueError(
                "One argument among excl_end_final_date and gap_timedelta must be given"
            )

        # the dataframe is extended using the row of index.max()
        end_row = timeseries_df.loc[[incl_end_time], :]
        if len(end_row) > 1:
            raise ValueError("Duplicate last record, cannot find a way to extrapolate")
        extra_row = timeseries_df.loc[[incl_end_time], :].copy()
        extra_row.index = [extra_excl_end_time]
        extra_row.index.name = timeseries_df.index.name

        # Add the extra row, and extrapolate using resample
        result = pd.concat([end_row, extra_row], ignore_index=False)
        result = result.resample(freq).interpolate(method="ffill")
        result = result.drop([incl_end_time, extra_excl_end_time])

        if cut_off is not None:
            cut_off_end = enda.tools.timeseries.TimeSeries.add_timedelta(
                incl_end_time.floor(cut_off), cut_off
            )
            result = result[result.index < cut_off_end]

        result = pd.concat([timeseries_df, result], axis=0)
        result = result.astype(timeseries_df.dtypes)
        result.index.freq = timeseries_df.index.freq

        return result
