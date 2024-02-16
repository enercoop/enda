from typing import Union
import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

import enda.timeseries
import enda.timezone_utils
import enda.decorators


class Resample:

    @staticmethod
    @enda.decorators.handle_multiindex(arg_name='timeseries_df')
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
            agg_functions = {_: agg_functions for _ in list(set(timeseries_df.columns).difference(set(groupby)))}
        else:
            raise TypeError(f"agg_functions must be of type str or dict, found {type(agg_functions)} instead")

        # If we need to group by other columns than the index, it means that there might be duplicates in time index.
        # That could lead 'find_most_common_frequency' to return null frequency as the most frequent.
        # Note this does NOT imply the duplicates are reset when downsampling !
        skip_duplicates_for_unique_freq_check = not len(groupby) > 0

        # get frequency from the initial dataframe
        # it return None only if the dataframe has a single record, else it returns the most common frequency
        original_freq = enda.timeseries.TimeSeries.find_most_common_frequency(
            timeseries_df.index,
            skip_duplicate_timestamps=skip_duplicates_for_unique_freq_check,
        )

        if (original_freq is not None) and is_original_frequency_unique and \
                (not enda.timeseries.TimeSeries.has_single_frequency(
                    timeseries_df.index,
                    variable_duration_freq_included=True,
                    skip_duplicate_timestamps=skip_duplicates_for_unique_freq_check)
                 ):
            raise RuntimeError("Frequency is not unique in the dataframe")

        # make sure we downsample
        if (original_freq is not None) and (enda.timeseries.TimeSeries.freq_as_approximate_nb_days(freq) <
                                            enda.timeseries.TimeSeries.freq_as_approximate_nb_days(original_freq)):
            raise RuntimeError(
                f"The required frequency {freq}" f" is smaller than the original one {original_freq}"
            )

        # resample using the aggregation function
        if len(groupby) == 0:
            resampled_df = timeseries_df.resample(freq, origin=origin).aggregate(agg_functions)
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
    @enda.decorators.handle_multiindex(arg_name='timeseries_df')
    def upsample_and_divide_evenly(
            timeseries_df: pd.DataFrame,
            freq: Union[str, pd.Timedelta],
            index_name: str = None,
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
        """

        if not isinstance(timeseries_df.index, pd.DatetimeIndex):
            raise TypeError("The dataframe index must be a DatetimeIndex")

        # check it's not a one-row dataframe, because it cannot be up-sampled.
        # Consider using forward_fill and give a gap_timedelta to achieve the objective.
        if enda.timeseries.TimeSeries.find_nb_records(timeseries_df.index) <= 1:
            raise ValueError("Cannot upsample an empty or single-record time series; because it's initial"
                             " frequency cannot be determined. Consider using 'forward_fill_last_record prior.")

        # get frequency from the initial dataframe
        original_freq = enda.timeseries.TimeSeries.find_most_common_frequency(timeseries_df.index)

        # if it's not a regular frequency, this function cannot work
        if (not enda.timeseries.TimeSeries.is_regular_freq(original_freq)) or (
                not enda.timeseries.TimeSeries.is_regular_freq(freq)):
            raise ValueError(f"Cannot upsample and divide for an input dataframe with an irregular frequency, "
                             f"(in terms of length). Here, we have original_freq={original_freq}, "
                             f" and target freq = {freq}")

        if not enda.timeseries.TimeSeries.has_single_frequency(
                timeseries_df.index,
                variable_duration_freq_included=False,
                skip_duplicate_timestamps=False):
            raise ValueError("Frequency is not single-defined in the dataframe")

        # compute the frequency ration to know how to divide. It should be convertible to pd.Timedelta
        freq_ratio = pd.to_timedelta(original_freq).total_seconds() / pd.to_timedelta(freq).total_seconds()

        # make sure we upsample
        if freq_ratio < 1:
            raise ValueError(f"The required frequency {freq}" f" is higher than the original one {original_freq}")

        # we must artificially add one timestep to the original dataframe to make sure we don't miss time steps in the
        # new frequency. With the example of the docstring, if we don't do this the 2023-01-01 00:45:00 won't be
        # added when resampling which will make the sum of values in the numerical columns false
        # Note this implies the last record of the resulting index is greater than the initial one.
        timeseries_df.loc[enda.timeseries.TimeSeries.add_timedelta(timeseries_df.index.max(), original_freq)] = 0
        timeseries_df = timeseries_df.resample(freq).ffill().div(freq_ratio).iloc[:-1]

        # change the index_name if required
        if index_name is not None:
            timeseries_df.index.name = index_name

        return timeseries_df

    @staticmethod
    @enda.decorators.handle_multiindex(arg_name='timeseries_df')
    def upsample_monthly_data_and_divide_evenly(
            timeseries_df: pd.DataFrame, freq: Union[str, pd.Timedelta]
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
        """

        if not isinstance(timeseries_df.index, pd.DatetimeIndex):
            raise TypeError("The dataframe index must be a DatetimeIndex")

        # get frequency from the initial dataframe
        original_freq = enda.timeseries.TimeSeries.find_most_common_frequency(timeseries_df.index)

        # check it's monthly
        amount_months, month_freq = enda.timeseries.TimeSeries.split_amount_and_unit_from_freq(original_freq)
        if month_freq not in ['MS', 'M'] or amount_months != 1:
            raise ValueError(f"Frequency should be monthly but found {original_freq}")

        # resample using the aggregation function
        # We must artificially add one timestep to the original dataframe to make sure we don't miss time steps in the
        # new frequency. With the example of the docstring, if we don't do this the 2023-01-01 00:45:00 won't be
        # added when resampling which will make the sum of values in the numerical columns false
        resampled_df = timeseries_df.copy()
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
    @enda.decorators.handle_multiindex(arg_name='timeseries_df')
    def upsample_and_interpolate(
            timeseries_df: pd.DataFrame,
            freq: [str, pd.Timedelta],
            method: str = "linear",
            forward_fill: bool = False,
            is_original_frequency_unique: bool = False,
            index_name: str = None,
    ):
        """
        Upsample a datetime-indexed dataframe, and interpolate the columns data according to an interpolating method
        :param timeseries_df: the dataframe to resample
        :param freq: the target frequency (timedelta) e.g. 'H', '30min', '15min', etc.
        :param method: how are data interpolated between two consecutive dates (e.g. 'ffill', 'linear', etc.)
        :param forward_fill: If True, the up-sampling is performed for the last element of the datetimeindex
        :param is_original_frequency_unique: check whether the frequency is unique in the initial dataframe
        :param index_name: a name to give to the new index. For instance going from 'date' to 'time'.
        :return: a pd.DataFrame with a resampled index
        """

        if type(timeseries_df.index) != pd.DatetimeIndex:
            raise TypeError("The dataframe index must be a DatetimeIndex")

        # check it's not a one-row dataframe, because it cannot be up-sampled.
        if enda.timeseries.TimeSeries.find_nb_records(timeseries_df.index) <= 1:
            raise ValueError("Cannot upsample an empty or single-record time series; because it's initial"
                             " frequency cannot be determined. Consider using 'forward_fill_last_record prior.")
        # get frequency from the initial dataframe
        original_freq = enda.timeseries.TimeSeries.find_most_common_frequency(
            timeseries_df.index
        )

        if is_original_frequency_unique and \
                (not enda.timeseries.TimeSeries.has_single_frequency(
                    timeseries_df.index,
                    variable_duration_freq_included=True,
                    skip_duplicate_timestamps=False)
                 ):
            raise ValueError("Frequency is not unique in the dataframe")

        # make sure we downsample
        if (enda.timeseries.TimeSeries.freq_as_approximate_nb_days(freq) >=
                enda.timeseries.TimeSeries.freq_as_approximate_nb_days(original_freq)):
            raise ValueError(
                f"The required frequency {freq}" f" is smaller than the original one {original_freq}"
            )
        # resample and interpolate
        timeseries_df = timeseries_df.resample(freq).interpolate(method=method)

        if forward_fill:
            # this serves to extend the dataframe, and resample using 'ffill' the last element
            # of the dataframe (not impacted by the previous resampling operation).
            # note that in that case, the timedelta must correspond to the initial frequency
            timeseries_df = Resample.forward_fill_final_record(timeseries_df, gap_timedelta=original_freq)

        # change the index_name if required
        if index_name is not None:
            timeseries_df.index.name = index_name

        return timeseries_df

    @staticmethod
    @enda.decorators.handle_multiindex(arg_name='timeseries_df')
    def equal_sample_fillna(
            timeseries_df: pd.DataFrame,
            fill_value: any = None,
            method_filling: str = None,
            safe_run: bool = True
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
        """

        # check both options (fill_with and method_filling) are not active simultaneously
        if fill_value is not None and method_filling is not None:
            raise ValueError(
                f"fill_value = {fill_value} which contradicts method_filling = {method_filling}. "
                "Only one can be not None."
            )

        # get frequency from the initial dataframe.
        freq = enda.timeseries.TimeSeries.find_most_common_frequency(timeseries_df.index,
                                                                     skip_duplicate_timestamps=True)

        if safe_run:
            # check duplicates
            duplicates = enda.timeseries.TimeSeries.find_duplicates(timeseries_df.index)
            if len(duplicates) > 0:
                raise RuntimeError(
                    f"Duplicates found in the index of the dataframe. The function would delete them, "
                    f"so it's unsafe to run. duplicates = {duplicates}"
                )

            # check extra-periods
            extra_periods = enda.timeseries.TimeSeries.find_extra_points(timeseries_df.index)
            if len(extra_periods) > 0:
                raise ValueError(
                    f"Extra periods found in the index of the dataframe. The function would delete them,"
                    f" so it's unsafe to run. extra_periods = {extra_periods}"
                )

        if method_filling is not None:
            # resample using the method of Resampler.fillna() if provided
            # throw an error if method_filling is not a valid option
            result_df = timeseries_df.resample(freq).fillna(method=method_filling)
        else:
            # if method_filling is None, we use Resampler.asfreq() with fill_value
            result_df = timeseries_df.resample(freq).asfreq(fill_value=fill_value)

        return result_df

    @staticmethod
    @enda.decorators.handle_multiindex(arg_name='timeseries_df')
    def forward_fill_final_record(
            timeseries_df: pd.DataFrame,
            gap_timedelta: [str, pd.Timedelta],
            cut_off: [str, pd.Timedelta] = None,
    ):
        """
        Forward-fill the final record of a regular datetime-indexed dataframe, keeping the sampling of
        the initial time series.
        The index of the resulting dataframe is determined using the parameter 'gap_timedelta' so that
        final_index = index + gap_timedelta.
        The resampling frequency is determined from the frequency of the initial dataframe.
        The extra parameter 'cut_off_frequency' can be used to set up a limit not to overpass
        This function is typically used in junction with interpolate_freq_to_sub_freq() to forward-fill the last record
        (gap_timedelta in that case is the initial frequency of the dataframe before the interpolation).

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
        :param gap_timedelta: the timedelta used to extend the final dataframe record.
        :param cut_off: a timedelta that serves as a cut-off beyond which the final record is not extended
        :return: datetime-indexed dataframe similar to the initial one, with the last record being forward filled
                 using the initial frequency of the dataframe index
        """

        freq = enda.timeseries.TimeSeries.find_most_common_frequency(timeseries_df.index)

        # the last record is understood as the max time (not the last index)
        incl_end_time = timeseries_df.index.max()
        extra_end_time = enda.timeseries.TimeSeries.add_timedelta(incl_end_time, gap_timedelta)

        # the dataframe is extended using the row of index.max()
        end_row = timeseries_df.loc[[incl_end_time], :]
        if len(end_row) > 1:
            raise RuntimeError("Duplicate last record, cannot find a way to extrapolate")
        extra_row = pd.DataFrame(
            timeseries_df.loc[[incl_end_time], :].values,
            index=[extra_end_time],
            columns=timeseries_df.columns
        )
        extra_row.index.name = timeseries_df.index.name

        # Add the extra row, and extrapolate using resample
        result = pd.concat([end_row, extra_row], ignore_index=False)
        result = result.resample(freq).interpolate(method="ffill")
        result = result.drop([incl_end_time, extra_end_time])

        if cut_off is not None:
            cut_off_end = enda.timeseries.TimeSeries.add_timedelta(incl_end_time.floor(cut_off), cut_off)
            result = result[result.index < cut_off_end]

        result = pd.concat([timeseries_df, result], axis=0)
        result.index.freq = timeseries_df.index.freq
        return result
