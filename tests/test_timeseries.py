import unittest
from enda.timeseries import TimeSeries
import pandas as pd
import pytz


class TestTimeSeries(unittest.TestCase):

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
