import datetime
import numpy as np
import pandas as pd
import pytz

from enda.decorators import handle_multiindex


class TimeSeries:

    @classmethod
    def align_timezone(cls, time_series: pd.Series, tzinfo: [str, pytz.timezone]):
        """
        Sometimes a time_series is of pandas type "object" just because the time-zone information
        is not well read initially.

        Example :
        time_series = a time_series with some times at timezone +01:00 (French winter time)
                      and others at timezone +02:00 (French summer time)
                      So its pandas dtype is "object"
        tz = pytz.timezone("Europe/Paris")
        We want to make sure the timezone information is tzinfo for each row and also for the series.

        :param time_series: a series with tz-aware date-times
        :param tzinfo: a str or a datetime.tzinfo
        :return: a DatetimeIndex of dtype: datetime[ns, tzinfo]
        """

        if not isinstance(time_series, pd.Series) and not isinstance(time_series, pd.DatetimeIndex):
            raise TypeError("parameter 'series' should be a pandas.Series")

        if isinstance(tzinfo, str):
            tzinfo = pytz.timezone(tzinfo)

        if not isinstance(tzinfo, datetime.tzinfo):
            raise TypeError("parameter 'tzinfo' should be of type str or datetime.tzinfo")

        result = time_series.map(lambda x: x.astimezone(tzinfo))
        # now all values in 'result' have the same type: pandas.Timestamp with the same tzinfo
        result = pd.DatetimeIndex(result)
        return result

    @classmethod
    def find_missing_and_extra_periods(
            cls,
            dti,
            expected_freq=None,
            expected_start_datetime=None,
            expected_end_datetime=None):
        """
        Check for missing and extra data points
        :param dti: a series of type DatetimeIndex, in ascending order and without duplicates or NaNs
        :param expected_freq: pandas formatted frequency. If None is given, will infer the frequency, taking the
            most common gap between 2 consecutive points.
        :param expected_start_datetime: a pandas.Datetime, if None is given, will take dti[0]
        :param expected_end_datetime: a pandas.Datetime, if None is given, will take dti[-1]
        :return: Missing and extra data points collapsed in periods
        """

        if not isinstance(dti, pd.DatetimeIndex):
            raise TypeError("parameter 'dti' should be a pandas.DatetimeIndex")

        if dti.isna().sum() != 0:
            raise ValueError("given dti has NaN values.")

        if dti.duplicated().sum() != 0:
            raise ValueError("given dti has duplicates.")

        if not dti.equals(dti.sort_values(ascending=True)):
            raise ValueError("given dti is not in ascending order")

        if len(dti) == 0:
            raise ValueError("given dti is empty")

        if expected_start_datetime is not None:
            if not isinstance(expected_start_datetime, pd.Timestamp):
                raise TypeError("expected_start_datetime must be a pandas.Datetime")
            start = expected_start_datetime
        else:
            start = dti[0]

        if expected_end_datetime is not None:
            if not isinstance(expected_end_datetime, pd.Timestamp):
                raise TypeError("expected_end_datetime must be a pandas.Datetime")
            end = expected_end_datetime
        else:
            end = dti[-1]

        if expected_freq is not None:
            if not isinstance(expected_freq, str) and not isinstance(expected_freq, pd.Timedelta):
                raise TypeError("expected_freq should be str or pd.Timedelta")
            freq = expected_freq

        else:
            # Infer frequency

            # compute the gap distribution
            gap_dist = (pd.Series(dti[1:]) - pd.Series(dti[:-1])).value_counts()
            # gap_dist =
            #    01:00:00    1181
            #    02:00:00     499
            #    03:00:00     180
            #    ....

            # take the most common gap and use it as expected_freq
            freq = gap_dist.index[0]  # this is a pd.Timedelta object
            assert isinstance(freq, pd.Timedelta)

        expected_index = pd.date_range(start, end, freq=freq)
        missing_points = expected_index.difference(dti)
        # group missing points together as "missing periods"
        missing_periods = cls.collapse_dt_series_into_periods(missing_points, freq=freq)
        extra_points = dti.difference(expected_index)

        return freq, missing_periods, extra_points

    @classmethod
    def collapse_dt_series_into_periods(
            cls,
            dti: pd.DatetimeIndex,
            freq: [str, pd.Timedelta]
    ):
        """
        This function does not work if freq < 1s
        :param freq:
        :param dti: DatetimeIndex, must be sorted
        :return:
        """

        assert isinstance(dti, pd.DatetimeIndex)
        assert isinstance(freq, str) or isinstance(freq, pd.Timedelta)
        if pd.to_timedelta(freq).total_seconds() < 1.0:
            raise ValueError("freq must be more than 1 second, but given {}".format(freq))

        if dti.shape[0] == 0:
            return []

        current_period_start = dti[0]
        periods_list = []
        for i in range(1, dti.shape[0]):
            if dti[i] <= dti[i-1]:
                raise ValueError("dti must be sorted and without duplicates!")

            if (dti[i] - dti[i-1]).total_seconds() % pd.to_timedelta(freq).total_seconds() != 0:
                raise ValueError("Timedelta between {} and {} is not a multiple of freq ({})."
                                 .format(dti[i-1], dti[i], freq))

            if pd.to_timedelta(freq) != dti[i] - dti[i-1]:  # End the current period and start a new one
                periods_list.append((current_period_start, dti[i-1]))
                current_period_start = dti[i]
        periods_list.append((current_period_start, dti[-1]))  # Don't forget last period
        return periods_list

    @staticmethod 
    def get_timeseries_frequency(
            index: pd.DatetimeIndex
            ):
        '''
        Retrieve the frequency of a pandas index. This calls pd.infer_freq() under the hood,
        but strictly forces the frequency to be defined.
        '''
        if(type(index) != pd.DatetimeIndex): 
            raise ValueError("get_timeseries_frequency(args) expects a pandas DatetimeIndex")
        differences = index.to_series().sort_values().diff()
        if differences.nunique() != 1:
            str_error = ("Frequency could not be inferred from the original data. "
                         "There are at least two different time intervals found: "
                         f"{differences.unique()[1] / np.timedelta64(1, 'm')} min, and "
                         f"{differences.unique()[2] / np.timedelta64(1, 'm')} min. "
                         "The function cannot be used.")
            raise ValueError(str_error)
        else: 
            return pd.infer_freq(index)

    @staticmethod
    @handle_multiindex
    def interpolate_freq_to_sub_freq_data(
            df: pd.DataFrame,
            *,
            freq: [str, pd.Timedelta],
            tz: [str, datetime.tzinfo],
            index_name: str = None,
            method: str = 'ffill',
            expand_initial=False, 
            expand_final=False, 
            cut_off_frequency=None
            ):
        '''
        Interpolate dataframe data on a smaller frequency than the one initially defined 
        in the dataframe
        The original index of the data must have a well-defined frequency, ie. it must be 
        able to retrieve its frequency with inferred_freq
        :param df: pd.DataFrame
        :param freq: a frequency < 'D' (e.g. 'H', '30min', '15min', etc)
        :param tz: the time zone (None not accepted because important)
        :param index_name: name to give to the new index. Usually going from 'date' to 'time'.
        :param method: how are data interpolated between two consecutive dates (e.g. 'ffill', 'linear', etc)
        :return: pd.DataFrame
        '''

        assert type(df.index) == pd.DatetimeIndex
        freq_original = TimeSeries.get_timeseries_frequency(df.index)
        assert pd.to_timedelta(freq).total_seconds() <= pd.to_timedelta(freq_original).total_seconds()

        if df.index.tzinfo is None:
            df.index = pd.to_datetime(df.index).tz_localize(tz)
        else:
            df.index = pd.to_datetime(df.index).tz_convert(tz)
        
        start_date = df.index.min() 
        extra_start_date = start_date - pd.to_timedelta(freq_original)
        end_date = df.index.max()
        extra_end_date = end_date + pd.to_timedelta(freq_original)

        result = df.copy()       

        if expand_final:
            extra_row = pd.DataFrame(result.iloc[[-1]].values, index=[extra_end_date], columns=result.columns)
            result = pd.concat([result, extra_row], axis=0, ignore_index=False)

        if expand_initial: 
            extra_row = pd.DataFrame(result.iloc[[0]].values, index=[extra_start_date], columns=result.columns)
            result = pd.concat([extra_row, result], axis=0, ignore_index=False)

        result = result.resample(freq).interpolate(method=method)

        if expand_final or expand_initial:
            if extra_end_date in df.index:
                result = result.drop(extra_end_date)
            if cut_off_frequency is not None:
                cut_off_start = start_date.floor(cut_off_frequency)
                cut_off_end = (end_date.floor(cut_off_frequency) + pd.to_timedelta(cut_off_frequency))
                result = result[(result.index >= cut_off_start) & (result.index < cut_off_end)]
       
        result.index.name = index_name

        return result

    @handle_multiindex
    def interpolate_daily_to_sub_daily_data(
            df: pd.DataFrame,
            *,
            freq: [str, pd.Timedelta],
            tz: [str, datetime.tzinfo],
            index_name: str = None,
            method: str = 'ffill'):
        """
        Interpolate daily data in a dataframe (with a DatetimeIndex) to sub-daily data using a given method.
        :param df: pd.DataFrame
        :param freq: a frequency < 'D' (e.g. 'H', '30min', '15min', etc)
        :param tz: the time zone (None not accepted because important)
        :param index_name: name to give to the new index. Usually going from 'date' to 'time'.
        :param method: how are data interpolated between two consecutive dates (e.g. 'ffill', 'linear', etc)
        :return: pd.DataFrame
        """

        assert type(df.index) == pd.DatetimeIndex
        assert pd.to_timedelta(freq).total_seconds() <= 24*60*60

        if df.index.tzinfo is None:
            df.index = pd.to_datetime(df.index).tz_localize(tz)
        else:
            df.index = pd.to_datetime(df.index).tz_convert(tz)

        new_end_date = df.index.max() + datetime.timedelta(days=1)
        extra_row = df.iloc[[-1]].reindex([new_end_date])

        # result =  df.append(extra_row, ignore_index=False)
        result = pd.concat([df, extra_row], axis=0, ignore_index=False)
        result = result.resample(freq).interpolate(method=method)
        if new_end_date in result.index:
            result = result.drop(new_end_date)
        result.index.name = index_name

        return result

    @handle_multiindex
    def average_to_upper_freq(
            df: pd.DataFrame,
            *,
            freq: [str, pd.Timedelta],
            tz: [str, datetime.tzinfo],
            index_name: str):
        """
        Upsample data provided in a given dataframe with a DatetimeIndex, or a two-levels 
        compatible Multiindex. 

        """

        assert type(df.index) == pd.DatetimeIndex
        freq_original = TimeSeries.get_timeseries_frequency(df.index)
        assert pd.to_timedelta(freq).total_seconds() >= pd.to_timedelta(freq_original).total_seconds()

        if df.index.tzinfo is None:
            df.index = pd.to_datetime(df.index).tz_localize(tz)
        else:
            df.index = pd.to_datetime(df.index).tz_convert(tz)
        
        result = df.copy()        
        result = result.resample(freq).mean()

        result.index.name = index_name

        return result
