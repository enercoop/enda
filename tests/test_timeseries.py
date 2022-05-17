import unittest
import pandas as pd
import pandas.testing as pd_testing
import pytz

from enda.timeseries import TimeSeries


class TestTimeSeries(unittest.TestCase):

    def setUp(self):
        # add the possibility to unitetst dataframe equality
        def assertDataframeEqual(a, b, msg):
            try:
                pd_testing.assert_frame_equal(a, b)
            except AssertionError as e:
                raise self.failureException(msg) from e
        self.addTypeEqualityFunc(pd.DataFrame, assertDataframeEqual)

        # define a single index df 
        df = pd.date_range(
            start=pd.to_datetime('2021-01-01 00:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-04 00:00:00+01:00').tz_convert('Europe/Paris'),
            freq='D',
            tz='Europe/Paris',
            name='date'
        ).to_frame().set_index('date')
        df["value"] = [0] * 2 + [1] * 2
        self.single_index_test = df

        # define a multiindex_df
        df1 = df.copy()
        df1['key'] = 'key1'
        df2 = df.copy()
        df2["value"] = [2] * 2 + [3] * 2  # change values
        df2 = df2.iloc[:-1, ]  # suppress last day
        df2['key'] = 'key2'
        self.multi_index_test = pd.concat([df1, df2], axis=0).reset_index().set_index(['key', 'date'])

    def test_collapse_dt_series_into_periods(self):

        # periods is a list of (start, end) pairs.
        periods = [
            (pd.to_datetime('2018-01-01 00:15:00+01:00'), pd.to_datetime('2018-01-01 00:45:00+01:00')),
            (pd.to_datetime('2018-01-01 10:15:00+01:00'), pd.to_datetime('2018-01-01 15:45:00+01:00')),
            (pd.to_datetime('2018-01-01 20:15:00+01:00'), pd.to_datetime('2018-01-01 21:45:00+01:00')),
        ]

        # expand periods to build a time-series with gaps
        dti = pd.DatetimeIndex([])
        for s, e in periods:
            dti = dti.append(pd.date_range(s, e, freq="30min"))
        self.assertEqual(2+12+4, dti.shape[0])

        # now find periods in the time-series
        # should work with 2 types of freq arguments
        for freq in ["30min", pd.to_timedelta("30min")]:
            computed_periods = TimeSeries.collapse_dt_series_into_periods(dti, freq)
            self.assertEqual(len(computed_periods), len(periods))

            for i in range(len(periods)):
                self.assertEqual(computed_periods[i][0], periods[i][0])
                self.assertEqual(computed_periods[i][1], periods[i][1])

    def test_collapse_dt_series_into_periods_2(self):
        dti = pd.DatetimeIndex([
            pd.to_datetime('2018-01-01 00:15:00+01:00'),
            pd.to_datetime('2018-01-01 00:45:00+01:00'),
            pd.to_datetime('2018-01-01 00:30:00+01:00'),
            pd.to_datetime('2018-01-01 01:00:00+01:00')
        ])

        with self.assertRaises(ValueError):
            # should raise an error because 15min gaps are not multiples of freq=30min
            TimeSeries.collapse_dt_series_into_periods(dti, freq="30min")

    def test_collapse_dt_series_into_periods_3(self):
        dti = pd.DatetimeIndex([
            pd.to_datetime('2018-01-01 00:00:00+01:00'),
            pd.to_datetime('2018-01-01 00:15:00+01:00'),
            pd.to_datetime('2018-01-01 00:30:00+01:00'),
            pd.to_datetime('2018-01-01 00:45:00+01:00')
        ])

        with self.assertRaises(ValueError):
            # should raise an error because 15min gaps are not multiples of freq=30min
            TimeSeries.collapse_dt_series_into_periods(dti, "30min")

    def test_find_missing_and_extra_periods_1(self):
        dti = pd.DatetimeIndex([
            pd.to_datetime('2018-01-01 00:00:00+01:00'),
            pd.to_datetime('2018-01-01 00:15:00+01:00'),
            pd.to_datetime('2018-01-01 00:30:00+01:00'),
            pd.to_datetime('2018-01-01 00:45:00+01:00'),
            pd.to_datetime('2018-01-01 00:50:00+01:00'),
            pd.to_datetime('2018-01-01 01:00:00+01:00'),
            pd.to_datetime('2018-01-01 02:00:00+01:00'),
            pd.to_datetime('2018-01-01 02:20:00+01:00')
        ])
        freq, missing_periods, extra_points = TimeSeries.find_missing_and_extra_periods(dti, expected_freq="15min")
        self.assertEqual(len(missing_periods), 2)  # (01:15:00 -> 01:45:00), (02:15:00 -> 02:15:00)
        self.assertEqual(len(extra_points), 2)  # [00:50:00, 02:20:00]

    def test_find_missing_and_extra_periods_2(self):
        dti = pd.DatetimeIndex([
            pd.to_datetime('2018-01-01 00:00:00+01:00'),
            pd.to_datetime('2018-01-01 00:15:00+01:00'),
            pd.to_datetime('2018-01-01 00:30:00+01:00'),
            pd.to_datetime('2018-01-01 00:45:00+01:00'),
            pd.to_datetime('2018-01-01 00:50:00+01:00'),
            pd.to_datetime('2018-01-01 01:00:00+01:00'),
            pd.to_datetime('2018-01-01 02:00:00+01:00'),
            pd.to_datetime('2018-01-01 02:20:00+01:00')
        ])

        # should work when we infer "expected_freq"
        freq, missing_periods, extra_points = TimeSeries.find_missing_and_extra_periods(dti, expected_freq=None)
        self.assertEqual(freq, pd.Timedelta("15min"))  # inferred a 15min freq
        self.assertEqual(len(missing_periods), 2)  # (01:15:00 -> 01:45:00), (02:15:00 -> 02:15:00)
        self.assertEqual(len(extra_points), 2)  # [00:50:00, 02:20:00]

    def test_find_missing_and_extra_periods_3(self):
        dti = pd.DatetimeIndex([
            pd.to_datetime('2018-01-01'),
            pd.to_datetime('2018-01-02'),
            pd.to_datetime('2018-01-03'),
            pd.to_datetime('2018-01-03 12:00:00'),
            pd.to_datetime('2018-01-04'),
            pd.to_datetime('2018-01-05'),
            pd.to_datetime('2018-01-06')
        ])
        freq, missing_periods, extra_points = TimeSeries.find_missing_and_extra_periods(dti, '1D')
        self.assertEqual(len(missing_periods), 0)
        self.assertEqual(len(extra_points), 1)

    def test_find_missing_and_extra_periods_4(self):

        dti_with_duplicates = pd.DatetimeIndex([
            pd.to_datetime('2018-01-01'),
            pd.to_datetime('2018-01-01')])

        dti_empty = pd.DatetimeIndex([])

        dti_not_sorted = pd.DatetimeIndex([
            pd.to_datetime('2018-01-02'),
            pd.to_datetime('2018-01-01')])

        for dti in [dti_with_duplicates, dti_empty, dti_not_sorted]:
            with self.assertRaises(ValueError):
                TimeSeries.find_missing_and_extra_periods(dti, '1D')

    def test_align_timezone_1(self):

        # useful typically when daylight savings changes
        s = pd.Series([
            pd.to_datetime('2021-12-31 02:00:00+02:00'),
            pd.to_datetime('2021-12-31 03:00:00+02:00'),
            pd.to_datetime('2021-12-31 03:00:00+01:00'),
            pd.to_datetime('2021-12-31 04:00:00+01:00')
        ])

        # s is of dtype "object" because of varying timezone
        assert not pd.api.types.is_datetime64_any_dtype(s)
        self.assertEqual(s.dtype, "object")

        # cannot convert because of varying timezone
        with self.assertRaises(ValueError):
            pd.DatetimeIndex(s)

        # should work for 2 types of timezone object
        for tz in ["Europe/Paris", pytz.timezone("Europe/Paris")]:
            dti = TimeSeries.align_timezone(s, tzinfo=tz)
            self.assertIsInstance(dti, pd.DatetimeIndex)
            self.assertEqual(len(dti), len(s))

    def test_align_timezone_2(self):

        # should also work when given already a DatetimeIndex, and we convert it to the desired timezone
        dti = pd.DatetimeIndex([
            pd.to_datetime('2021-12-31 04:00:00+03:00'),
            pd.to_datetime('2021-12-31 05:00:00+03:00'),
            pd.to_datetime('2021-12-31 06:00:00+03:00')
        ])
        self.assertEqual(dti.dtype, "datetime64[ns, pytz.FixedOffset(180)]")

        dti2 = TimeSeries.align_timezone(dti, tzinfo=pytz.timezone("Europe/Berlin"))
        self.assertIsInstance(dti2, pd.DatetimeIndex)
        self.assertEqual(len(dti), len(dti2))
        self.assertEqual(dti2.dtype, "datetime64[ns, Europe/Berlin]")

    def test_align_timezone_3(self):
        
        # should not work when given time-series is time zone naive or has naive elements,
        # or element other than a tz-aware datetime
           
        s1 = pd.Series([
            pd.to_datetime('2021-12-31 04:00:00')
        ])

        s2 = pd.Series([
            pd.to_datetime('2021-12-31 04:00:00'),
            pd.to_datetime('2021-12-31 05:00:00+01:00'),
        ])

        s3 = pd.DatetimeIndex([
            pd.to_datetime('2021-12-31 04:00:00'),
            pd.to_datetime('2021-12-31 05:00:00'),
        ])

        s4 = pd.Series([0, 1])

        for s in [s1, s2, s3]:
            with self.assertRaises(TypeError):
                TimeSeries.align_timezone(s, tzinfo="UTC")

        for s in [s4]:
            with self.assertRaises(AttributeError):
                TimeSeries.align_timezone(s, tzinfo="UTC")

    def test_interpolate_daily_to_sub_daily_data_single_index(self): 
        
        df_expected = pd.date_range(
            start=pd.to_datetime('2021-01-01 00:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-04 18:00:00+01:00').tz_convert('Europe/Paris'),
            freq='6H', 
            tz='Europe/Paris',
            name='time'
        ).to_frame().set_index('time')
        df_expected["value"] = [0.] * 8 + [1.] * 8
        df_expected.index.freq = '6H'
        
        # test with full kwargs
        sub_df = TimeSeries.interpolate_daily_to_sub_daily_data(
                     df=self.single_index_test, 
                     freq='6H',
                     tz='Europe/Paris'
                 )

        self.assertEqual(df_expected, sub_df)

        # test with a combination of args and kwargs
        sub_df = TimeSeries.interpolate_daily_to_sub_daily_data(
                     self.single_index_test, 
                     freq='6H',
                     tz='Europe/Paris'
                 )
        self.assertEqual(df_expected, sub_df)

    def test_interpolate_daily_to_sub_daily_data_multi_index(self): 
                
        df1_expected = pd.date_range(
            start=pd.to_datetime('2021-01-01 00:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-04 18:00:00+01:00').tz_convert('Europe/Paris'),
            freq='6H', 
            tz='Europe/Paris',
            name='time'
        ).to_frame()
        df1_expected["value"] = [0.] * 8 + [1.] * 8
        df1_expected["key"] = 'key1'
        df1_expected.index.freq = '6H'
        
        df2_expected = pd.date_range(
            start=pd.to_datetime('2021-01-01 00:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-03 18:00:00+01:00').tz_convert('Europe/Paris'),
            freq='6H', 
            tz='Europe/Paris',
            name='time'
        ).to_frame()
        df2_expected["value"] = [2.] * 8 + [3.] * 4
        df2_expected["key"] = 'key2'
        df2_expected.index.freq = '6H'

        df_expected = pd.concat([df1_expected, df2_expected], axis=0).set_index(['key', 'time'])
         
        # test with full kwargs
        sub_df = TimeSeries.interpolate_daily_to_sub_daily_data(
                     df=self.multi_index_test, 
                     freq='6H',
                     tz='Europe/Paris'
                 )
        self.assertEqual(df_expected, sub_df)

        # test with a combination of args and kwargs
        sub_df = TimeSeries.interpolate_daily_to_sub_daily_data(
                     df=self.multi_index_test,
                     freq='6H',
                     tz='Europe/Paris'
                 )
        self.assertEqual(df_expected, sub_df)

    def test_interpolate_freq_to_sub_freq_data_single_index(self): 

        # test interpolate freq to subfreq on a 6H basis
        df_expected = pd.date_range(
              start=pd.to_datetime('2021-01-01 00:00:00+01:00').tz_convert('Europe/Paris'),
              end=pd.to_datetime('2021-01-04 00:00:00+01:00').tz_convert('Europe/Paris'),
              freq='6H', 
              tz='Europe/Paris',
              name='time'
        ).to_frame().set_index('time')
        df_expected.index.freq = '6H'
        df_expected['value'] = [0.] * 5 + [0.25, 0.5, 0.75] + [1] * 5

        sub_df = TimeSeries.interpolate_freq_to_sub_freq_data(
            df=self.single_index_test, freq='6H', tz='Europe/Paris', index_name='time'
        )
        self.assertEqual(df_expected, sub_df)

    def test_interpolate_freq_to_sub_freq_data_single_index_2(self): 

        # test interpolate freq to subfreq on a 6H basis
        df_expected = pd.date_range(
              start=pd.to_datetime('2021-01-01 00:00:00+01:00').tz_convert('Europe/Paris'),
              end=pd.to_datetime('2021-01-04 00:00:00+01:00').tz_convert('Europe/Paris'),
              freq='6H', 
              tz='Europe/Paris',
              name='date'
        ).to_frame().set_index('date')
        df_expected.index.freq = '6H'
        df_expected['value'] = [0.] * 8 + [1] * 5

        sub_df = TimeSeries.interpolate_freq_to_sub_freq_data(
            self.single_index_test, freq='6H', tz='Europe/Paris', method='ffill'
            )
        self.assertEqual(df_expected, sub_df)

    def test_interpolate_freq_to_sub_freq_data_multi_index(self): 
        
        # test interpolation with a multiindex

        df1_expected = pd.date_range(
            start=pd.to_datetime('2021-01-01 00:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-04 00:00:00+01:00').tz_convert('Europe/Paris'),
            freq='6H', 
            tz='Europe/Paris',
            name='time'
        ).to_frame()
        df1_expected["value"] = [0.] * 8 + [1.] * 5
        df1_expected["key"] = 'key1'
        df1_expected.index.freq = '6H'
        
        df2_expected = pd.date_range(
            start=pd.to_datetime('2021-01-01 00:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-03 00:00:00+01:00').tz_convert('Europe/Paris'),
            freq='6H', 
            tz='Europe/Paris',
            name='time'
        ).to_frame()
        df2_expected["value"] = [2.] * 8 + [3.] * 1
        df2_expected["key"] = 'key2'
        df2_expected.index.freq = '6H'

        df_expected = pd.concat([df1_expected, df2_expected], axis=0).set_index(['key', 'time'])
        
        sub_df = TimeSeries.interpolate_freq_to_sub_freq_data(
            self.multi_index_test, freq='6H', tz='Europe/Paris', method='ffill', index_name='time'
            )

        self.assertEqual(df_expected, sub_df)

    def test_forward_fill_final_record_1(self): 

        # test extend_final_data: this is a ffill of the last record 
        # on the provided frequency 
        df_test = pd.date_range(
            start=pd.to_datetime('2021-01-01 00:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-02 00:00:00+01:00').tz_convert('Europe/Paris'),
            freq='12H', tz='Europe/Paris', name='time'
        ).to_frame().set_index('time')
        df_test["value"] = [0., 1., 2.]

        df_expected = pd.date_range(
            start=pd.to_datetime('2021-01-01 00:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-02 12:00:00+01:00').tz_convert('Europe/Paris'),
            freq='12H', tz='Europe/Paris', name='time'
        ).to_frame().set_index('time')
        df_expected["value"] = [0., 1., 2., 2.]
        df_expected.index.freq = '12H'

        sub_df = TimeSeries.forward_fill_final_record(
            df=df_test, 
            gap_frequency='1D'
        )
        self.assertEqual(df_expected, sub_df)

    def test_forward_fill_final_record_2(self): 

        # forward_fill_final_record with a None cutoff frequency
        # and 3H as the original frequency 

        df_test = pd.date_range(
            start=pd.to_datetime('2021-01-01 19:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-01 22:00:00+01:00').tz_convert('Europe/Paris'),
            freq='1H', tz='Europe/Paris', name='time'
        ).to_frame().set_index('time')
        df_test["value"] = [0., 1., 2., 3.]

        df_expected = pd.date_range(
            start=pd.to_datetime('2021-01-01 19:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-02 00:00:00+01:00').tz_convert('Europe/Paris'),
            freq='1H', tz='Europe/Paris', name='time'
        ).to_frame().set_index('time')
        df_expected["value"] = [0., 1., 2., 3., 3., 3.]
        df_expected.index.freq = '1H'

        sub_df = TimeSeries.forward_fill_final_record(
            df=df_test, 
            gap_frequency='3H',
            cut_off_frequency=None
        )
        self.assertEqual(df_expected, sub_df)

    def test_forward_fill_final_record_3(self): 

        # forward_fill_final_record with a cutoff frequency

        df_test = pd.date_range(
            start=pd.to_datetime('2021-01-01 19:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-01 22:00:00+01:00').tz_convert('Europe/Paris'),
            freq='1H', tz='Europe/Paris', name='time'
        ).to_frame().set_index('time')
        df_test["value"] = [0., 1., 2., 3.]

        df_expected = pd.date_range(
            start=pd.to_datetime('2021-01-01 19:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-01 23:00:00+01:00').tz_convert('Europe/Paris'),
            freq='1H', tz='Europe/Paris', name='time'
        ).to_frame().set_index('time')
        df_expected["value"] = [0., 1., 2., 3., 3.]
        df_expected.index.freq = '1H'

        sub_df = TimeSeries.forward_fill_final_record(
            df=df_test, 
            gap_frequency='3H',
            cut_off_frequency='1D'
        )
        self.assertEqual(df_expected, sub_df)

    def test_average_to_upper_freq_1(self): 

        # average to upper freq

        df_test = pd.date_range(
            start=pd.to_datetime('2021-01-01 00:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-02 00:00:00+01:00').tz_convert('Europe/Paris'),
            freq='12H', tz='Europe/Paris', name='time'
        ).to_frame().set_index('time')
        df_test["value"] = [0., 1., 2.]

        df_expected = pd.date_range(
            start=pd.to_datetime('2021-01-01 00:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-02 00:00:00+01:00').tz_convert('Europe/Paris'),
            freq='1D', tz='Europe/Paris', name='time'
        ).to_frame().set_index('time')
        df_expected["value"] = [0.5, 2.]
        df_expected.index.freq = '1D'

        sub_df = TimeSeries.average_to_upper_freq(
            df=df_test, 
            freq='1D',
            tz="Europe/Paris"
        )
        self.assertEqual(df_expected, sub_df)

    def test_average_to_upper_freq_2(self): 

        # average to upper freq

        df_test = pd.date_range(
            start=pd.to_datetime('2021-01-01 00:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-02 00:00:00+01:00').tz_convert('Europe/Paris'),
            freq='12H', tz='Europe/Paris', name='time'
        ).to_frame().set_index('time')
        df_test["value"] = [0., 1., 2.]

        df_expected = pd.date_range(
            start=pd.to_datetime('2021-01-01 00:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-02 00:00:00+01:00').tz_convert('Europe/Paris'),
            freq='1D', tz='Europe/Paris', name='time'
        ).to_frame().set_index('time')
        df_expected["value"] = [0.5, 2.]
        df_expected.index.freq = '1D'

        sub_df = TimeSeries.average_to_upper_freq(
            df=df_test, 
            freq='1D',
            tz="Europe/Paris"
        )
        self.assertEqual(df_expected, sub_df)