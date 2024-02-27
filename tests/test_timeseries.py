import datetime
import logging
import pandas as pd
import pytz
import unittest

from enda.timeseries import TimeSeries


class TestTimeSeries(unittest.TestCase):

    def setUp(self):

        logging.captureWarnings(True)
        logging.disable(logging.ERROR)

        # define several frequencies
        self.ok_freq_list = ['2d', '1MS', '3H', 'D', '10MS', '2Y', '-3D', '2min', '+1S',
                             '2S', '-d', pd.Timedelta('10min'), pd.Timedelta('3D'), '1Q']
        self.not_ok_freq_list = ['D2', '1MS2', 'DD', '', 5]

        # define  datetime index and series of type datetime
        self.dti = pd.DatetimeIndex(['2024-01-01 01:00:00', '2024-01-01 02:00:00',
                                     '2024-01-01 03:00:00', '2024-01-01 04:00:00'])  # 1H regular
        self.almost_inperfect_dti = pd.DatetimeIndex(
            ['2024-01-01 01:00:00+01', '2024-01-01 02:00:00+01', '2024-01-01 02:00:00+01',
             '2024-01-01 04:00:00+01', '2024-01-01 03:00:00+01', '2024-01-01 05:00:00+01'])  # 1H duplicates unordered
        self.inperfect_dti = pd.DatetimeIndex(
            ['2024-01-01 01:00:00+01', '2024-01-01 02:00:00+01', '2024-01-01 03:00:00+01',
             '2024-01-01 06:00:00+01', '2024-01-01 08:00:00+01', '2024-01-01 09:00:00+01',
             '2024-01-01 09:30:00+01'])  # 1H with holes and an extra period (last point)
        self.monthly_dti = pd.DatetimeIndex(['2024-01-01', '2024-02-01', '2024-03-01'])  # monthly
        self.almost_inperfect_monthly_dti = pd.DatetimeIndex(
            ['2024-01-01', '2024-03-01', '2024-03-01', '2024-02-01']  # monthly duplicates unordered
        )
        self.series = self.dti.to_series().reset_index(drop=True)

        # define a single index df 
        self.single_index_df = (
            pd.date_range(
                start=pd.to_datetime('2021-01-01').tz_localize('Europe/Paris'),
                end=pd.to_datetime('2021-01-04').tz_localize('Europe/Paris'),
                freq='D',
                tz='Europe/Paris',
                name='date'
            )
            .to_frame()
            .set_index('date')
            .assign(value=[0] * 2 + [1] * 2)
        )

        # # define a multiindex_df
        df1 = self.single_index_df.copy()
        df1['key'] = 'key1'
        df2 = self.single_index_df.copy()
        df2["value"] = [2] * 2 + [3] * 2  # change values
        df2 = df2.iloc[:-1, ]  # suppress last day
        df2['key'] = 'key2'
        self.multi_index_test = pd.concat([df1, df2], axis=0).reset_index().set_index(['key', 'date'])

    def tearDown(self) -> None:
        logging.disable(logging.NOTSET)

    # ------------------------
    # Frequencies / Timedelta
    # ------------------------

    def test_split_amount_and_unit_from_freq(self):

        # test with correct frequencies
        parsed_freq_list = [TimeSeries.split_amount_and_unit_from_freq(_) for _ in self.ok_freq_list]

        expected_result_list = [(2, 'D'), (1, 'MS'), (3, 'H'), (1, 'D'), (10, 'MS'), (2, 'Y'), (-3, 'D'),
                                (2, 'MIN'), (1, 'S'), (2, 'S'), (-1, 'D'), (10, 'T'), (3, 'D'), (1, 'Q')]

        for result, expected_result in zip(parsed_freq_list, expected_result_list):
            self.assertEqual(result[0], expected_result[0])
            self.assertEqual(result[1], expected_result[1])

        # test with in fail frequencies
        for freq in self.not_ok_freq_list[:-1]:
            with self.assertRaises(ValueError):
                TimeSeries.split_amount_and_unit_from_freq(freq)

        with self.assertRaises(TypeError):
            TimeSeries.split_amount_and_unit_from_freq(self.not_ok_freq_list[-1])

    def test_is_regular_freq(self):

        # test with correct frequencies
        parsed_freq_list = [TimeSeries.is_regular_freq(_) for _ in self.ok_freq_list]

        expected_result_list = [True, False, True, True, False, False, True, True,
                                True, True, True, True, True, False]

        for result, expected_result in zip(parsed_freq_list, expected_result_list):
            self.assertEqual(result, expected_result)

    def test_freq_as_approximate_nb_days(self):

        # test with correct frequencies
        result_list = [TimeSeries.freq_as_approximate_nb_days(_) for _ in self.ok_freq_list]

        expected_result_list = [2, 30.4, 0.125, 1, 304, 730, -3,  2. / 1440, 1 / (24 * 3600),
                                2 / (24 * 3600), -1, 10 / 1440, 3, 91]

        for result, expected_result in zip(result_list, expected_result_list):
            self.assertAlmostEqual(result, expected_result)

    def test_add_timedelta(self):

        # test with naive timestamp
        test_timestamp = pd.to_datetime("2021-01-01 00:00:00")
        result_list = [TimeSeries.add_timedelta(test_timestamp, freq) for freq in self.ok_freq_list[:-1]]

        expected_result_list = [pd.to_datetime("2021-01-03 00:00:00"),
                                pd.to_datetime("2021-02-01 00:00:00"),
                                pd.to_datetime("2021-01-01 03:00:00"),
                                pd.to_datetime("2021-01-02 00:00:00"),
                                pd.to_datetime("2021-11-01 00:00:00"),
                                pd.to_datetime("2023-01-01 00:00:00"),
                                pd.to_datetime("2020-12-29 00:00:00"),
                                pd.to_datetime("2021-01-01 00:02:00"),
                                pd.to_datetime("2021-01-01 00:00:01"),
                                pd.to_datetime("2021-01-01 00:00:02"),
                                pd.to_datetime("2020-12-31 00:00:00"),
                                pd.to_datetime("2021-01-01 00:10:00"),
                                pd.to_datetime("2021-01-04 00:00:00")]

        for result, expected_result in zip(result_list, expected_result_list):
            self.assertEqual(result, expected_result)

        # test with tz-aware timestamp
        test_timestamp_tz = pd.to_datetime("2021-01-01").tz_localize("Europe/Paris")
        result_list = [TimeSeries.add_timedelta(test_timestamp_tz, freq) for freq in self.ok_freq_list[:-1]]

        expected_result_list = [pd.Timestamp("2021-01-03 00:00:00+01"),
                                pd.Timestamp("2021-02-01 00:00:00+01"),
                                pd.Timestamp("2021-01-01 03:00:00+01"),
                                pd.Timestamp("2021-01-02 00:00:00+01"),
                                pd.Timestamp("2021-11-01 00:00:00+01"),
                                pd.Timestamp("2023-01-01 00:00:00+01"),
                                pd.Timestamp("2020-12-29 00:00:00+01"),
                                pd.Timestamp("2021-01-01 00:02:00+01"),
                                pd.Timestamp("2021-01-01 00:00:01+01"),
                                pd.Timestamp("2021-01-01 00:00:02+01"),
                                pd.Timestamp("2020-12-31 00:00:00+01"),
                                pd.Timestamp("2021-01-01 00:10:00+01"),
                                pd.Timestamp("2021-01-04 00:00:00+01")]

        for result, expected_result in zip(result_list, expected_result_list):
            self.assertEqual(result, expected_result)

        # test with datetime.date
        test_date = datetime.date(year=2021, month=1, day=1)
        result_list = [TimeSeries.add_timedelta(test_date, freq) for freq in self.ok_freq_list[:-1]]

        expected_result_list = [datetime.date(year=2021, month=1, day=3),
                                datetime.date(year=2021, month=2, day=1),
                                datetime.date(year=2021, month=1, day=1),
                                datetime.date(year=2021, month=1, day=2),
                                datetime.date(year=2021, month=11, day=1),
                                datetime.date(year=2023, month=1, day=1),
                                datetime.date(year=2020, month=12, day=29),
                                datetime.date(year=2021, month=1, day=1),
                                datetime.date(year=2021, month=1, day=1),
                                datetime.date(year=2021, month=1, day=1),
                                datetime.date(year=2020, month=12, day=31),
                                datetime.date(year=2021, month=1, day=1),
                                datetime.date(year=2021, month=1, day=4)]

        for result, expected_result in zip(result_list, expected_result_list):
            self.assertEqual(result, expected_result)

        # test with datetime.datetime
        test_datetime = datetime.datetime(year=2021, month=1, day=1, hour=2)
        result_list = [TimeSeries.add_timedelta(test_datetime, freq) for freq in self.ok_freq_list[:-1]]

        expected_result_list = [datetime.datetime(year=2021, month=1, day=3, hour=2),
                                datetime.datetime(year=2021, month=2, day=1, hour=2),
                                datetime.datetime(year=2021, month=1, day=1, hour=5),
                                datetime.datetime(year=2021, month=1, day=2, hour=2),
                                datetime.datetime(year=2021, month=11, day=1, hour=2),
                                datetime.datetime(year=2023, month=1, day=1, hour=2),
                                datetime.datetime(year=2020, month=12, day=29, hour=2),
                                datetime.datetime(year=2021, month=1, day=1, hour=2, minute=2),
                                datetime.datetime(year=2021, month=1, day=1, hour=2, minute=0, second=1),
                                datetime.datetime(year=2021, month=1, day=1, hour=2, minute=0, second=2),
                                datetime.datetime(year=2020, month=12, day=31, hour=2),
                                datetime.datetime(year=2021, month=1, day=1, hour=2, minute=10),
                                datetime.datetime(year=2021, month=1, day=4, hour=2)]

        for result, expected_result in zip(result_list, expected_result_list):
            self.assertEqual(result, expected_result)

        # test fails frequencies

        # 1Q should fail, as it really means nothing and cannot be achieved
        with self.assertRaises(ValueError):
            TimeSeries.add_timedelta(test_datetime, self.ok_freq_list[-1])

        # test other fails
        for freq in self.not_ok_freq_list[:-1]:
            with self.assertRaises(ValueError):
                TimeSeries.add_timedelta(test_datetime, freq)

        with self.assertRaises(TypeError):
            TimeSeries.add_timedelta(test_datetime, self.not_ok_freq_list[-1])

    def test_subtract_timedelta(self):

        # test with naive timestamp
        test_timestamp = pd.to_datetime("2021-01-01 00:00:00")
        result_list = [TimeSeries.subtract_timedelta(test_timestamp, freq) for freq in self.ok_freq_list[:-1]]

        expected_result_list = [pd.to_datetime("2020-12-30 00:00:00"),
                                pd.to_datetime("2020-12-01 00:00:00"),
                                pd.to_datetime("2020-12-31 21:00:00"),
                                pd.to_datetime("2020-12-31 00:00:00"),
                                pd.to_datetime("2020-03-01 00:00:00"),
                                pd.to_datetime("2019-01-01 00:00:00"),
                                pd.to_datetime("2021-01-04 00:00:00"),
                                pd.to_datetime("2020-12-31 23:58:00"),
                                pd.to_datetime("2020-12-31 23:59:59"),
                                pd.to_datetime("2020-12-31 23:59:58"),
                                pd.to_datetime("2021-01-02 00:00:00"),
                                pd.to_datetime("2020-12-31 23:50:00"),
                                pd.to_datetime("2020-12-29 00:00:00")]

        for result, expected_result in zip(result_list, expected_result_list):
            self.assertEqual(result, expected_result)

        # 1Q should fail, as it really means nothing and cannot be achieved
        with self.assertRaises(ValueError):
            TimeSeries.add_timedelta(test_timestamp, self.ok_freq_list[-1])

        # test other fails
        for freq in self.not_ok_freq_list[:-2]:
            with self.assertRaises(ValueError):
                TimeSeries.add_timedelta(test_timestamp, freq)

        with self.assertRaises(TypeError):
            TimeSeries.add_timedelta(test_timestamp, self.not_ok_freq_list[-1])

    # ------------
    # Time series
    # ------------

    def test_has_nan_or_empty(self):

        # test with a correct dti
        result = TimeSeries.has_nan_or_empty(self.dti)
        self.assertEqual(result, False)

        # test with a correct series
        result = TimeSeries.has_nan_or_empty(self.series)
        self.assertEqual(result, False)

        # test with nan
        result = TimeSeries.has_nan_or_empty(pd.DatetimeIndex(['2024-01-01', None]))
        self.assertEqual(result, True)

        # test with empty
        result = TimeSeries.has_nan_or_empty(pd.DatetimeIndex(data=[]))
        self.assertEqual(result, True)

        # test weird type
        with self.assertRaises(ValueError):
            TimeSeries.has_nan_or_empty(5)

    def test_find_nb_records(self):
        result = TimeSeries.find_nb_records(self.almost_inperfect_dti)
        self.assertEqual(result, 6)

        result = TimeSeries.find_nb_records(self.almost_inperfect_dti, True)
        self.assertEqual(result, 5)

    def test_find_gap_distribution(self):

        # test dti
        pd.testing.assert_series_equal(TimeSeries.find_gap_distribution(self.dti),
                                       pd.Series(data=[3], index=[pd.Timedelta("1H")]))

        pd.testing.assert_series_equal(TimeSeries.find_gap_distribution(self.dti, True),
                                       pd.Series(data=[3], index=[pd.Timedelta("1H")]))

        # test almost_inperfect_dti
        pd.testing.assert_series_equal(TimeSeries.find_gap_distribution(self.almost_inperfect_dti),
                                       pd.Series(data=[4, 1],
                                                 index=[pd.Timedelta("1H"), pd.Timedelta("0H")]))

        pd.testing.assert_series_equal(TimeSeries.find_gap_distribution(dti=self.almost_inperfect_dti,
                                                                        skip_duplicate_timestamps=True),
                                       pd.Series(data=[4],
                                                 index=[pd.Timedelta("1H")]))

        # test dti_imperfect
        pd.testing.assert_series_equal(TimeSeries.find_gap_distribution(self.inperfect_dti),
                                       pd.Series(data=[3, 1, 1, 1],
                                                 index=[pd.Timedelta("1H"), pd.Timedelta("3H"),
                                                        pd.Timedelta("2H"), pd.Timedelta("30T")]))

        # test monthly data
        pd.testing.assert_series_equal(TimeSeries.find_gap_distribution(self.monthly_dti),
                                       pd.Series(data=[1, 1], index=[pd.Timedelta("31D"), pd.Timedelta("29D")]))

        # test inperfect monthly data
        pd.testing.assert_series_equal(TimeSeries.find_gap_distribution(dti=self.almost_inperfect_monthly_dti),
                                       pd.Series(data=[1, 1, 1],
                                                 index=[pd.Timedelta("31D"), pd.Timedelta("29D"), pd.Timedelta("0D")]))

        # test series
        pd.testing.assert_series_equal(TimeSeries.find_gap_distribution(self.series),
                                       pd.Series(data=[3], index=[pd.Timedelta("1H")]))

        pd.testing.assert_series_equal(
            TimeSeries.find_gap_distribution(self.series, skip_duplicate_timestamps=True),
            pd.Series(data=[3], index=[pd.Timedelta("1H")]))

    def test_find_most_common_frequency(self):

        # test dti
        self.assertEqual(TimeSeries.find_most_common_frequency(self.dti), "1H")
        self.assertEqual(TimeSeries.find_most_common_frequency(self.dti, True), "1H")

        # test almost_inperfect_dti
        self.assertEqual(TimeSeries.find_most_common_frequency(self.almost_inperfect_dti), "1H")
        self.assertEqual(TimeSeries.find_most_common_frequency(dti=self.almost_inperfect_dti,
                                                               skip_duplicate_timestamps=True),
                         "1H")

        # test dti_imperfect
        self.assertEqual(TimeSeries.find_most_common_frequency(self.inperfect_dti), "1H")

        # test monthly data
        # that's the interesting part !
        self.assertEqual(TimeSeries.find_most_common_frequency(self.monthly_dti), "1MS")
        self.assertEqual(TimeSeries.find_most_common_frequency(self.almost_inperfect_monthly_dti), "31D")
        self.assertEqual(TimeSeries.find_most_common_frequency(self.almost_inperfect_monthly_dti,
                                                               skip_duplicate_timestamps=True),
                         "1MS")

        # test series
        self.assertEqual(TimeSeries.find_most_common_frequency(self.series), "1H")
        self.assertEqual(TimeSeries.find_most_common_frequency(self.series,
                                                               skip_duplicate_timestamps=True),
                         "1H")

        self.assertEqual(TimeSeries.find_most_common_frequency(self.series,
                                                               skip_duplicate_timestamps=True),
                         "1H")

    def test_find_duplicates(self):

        empty_index = pd.DatetimeIndex(data=[])
        empty_series = pd.Series(data=[], dtype='datetime64[ns]')

        pd.testing.assert_index_equal(TimeSeries.find_duplicates(self.dti), empty_index)
        pd.testing.assert_series_equal(TimeSeries.find_duplicates(self.series), empty_series)

        pd.testing.assert_index_equal(TimeSeries.find_duplicates(self.almost_inperfect_dti),
                                      pd.DatetimeIndex(['2024-01-01 02:00:00+01']))
        pd.testing.assert_index_equal(TimeSeries.find_duplicates(self.almost_inperfect_monthly_dti),
                                      pd.DatetimeIndex(['2024-03-01']))

    def test_find_extra_points(self):

        result_index = pd.DatetimeIndex(data=[])
        pd.testing.assert_index_equal(TimeSeries.find_extra_points(self.dti), result_index)

        result_series = pd.Series(data=[], dtype='datetime64[ns]')
        pd.testing.assert_series_equal(TimeSeries.find_extra_points(self.series), result_series)

        # extra periods are found from the start
        result_index = pd.DatetimeIndex(["2024-01-01 02:00:00", "2024-01-01 04:00:00"])
        pd.testing.assert_index_equal(TimeSeries.find_extra_points(self.dti, expected_freq='2H'), result_index)

        result_series = pd.Series(data=["2024-01-01 02:00:00", "2024-01-01 04:00:00"], dtype='datetime64[ns]')
        pd.testing.assert_series_equal(TimeSeries.find_extra_points(self.series, expected_freq='2H'),
                                       result_series)

        # > result_index = pd.DatetimeIndex(data=[]).tz_localize("Europe/Paris")
        # > pd.testing.assert_index_equal(TimeSeries.find_extra_periods(self.almost_inperfect_dti), result_index)
        #  Fails because it's hard to set an empty index with type tz-aware datetime.
        #  In that case, just test the length of the index
        self.assertEqual(len(TimeSeries.find_extra_points(self.almost_inperfect_dti)), 0)

        result_index = pd.DatetimeIndex(["2024-01-01 09:30:00+01"])
        pd.testing.assert_index_equal(TimeSeries.find_extra_points(self.inperfect_dti), result_index)

        self.assertEqual(len(TimeSeries.find_extra_points(self.monthly_dti)), 0)
        self.assertEqual(len(TimeSeries.find_extra_points(self.almost_inperfect_monthly_dti)), 0)

    def test_find_missing_points(self):

        pd.testing.assert_index_equal(
            TimeSeries.find_missing_points(self.dti),
            pd.DatetimeIndex([], dtype='datetime64[ns]')
        )

        pd.testing.assert_index_equal(
            TimeSeries.find_missing_points(self.dti, expected_freq="20min"),
            pd.DatetimeIndex(['2024-01-01 01:20:00', '2024-01-01 01:40:00',
                              '2024-01-01 02:20:00', '2024-01-01 02:40:00',
                              '2024-01-01 03:20:00', '2024-01-01 03:40:00',
                              ])
        )

        pd.testing.assert_index_equal(
            TimeSeries.find_missing_points(self.dti, expected_freq="55min"),
            pd.DatetimeIndex(['2024-01-01 01:55:00', '2024-01-01 02:50:00', '2024-01-01 03:45:00'])
        )

        pd.testing.assert_series_equal(
            TimeSeries.find_missing_points(self.series),
            pd.Series([], dtype='datetime64[ns]')
        )

    def test_has_single_frequency(self):

        self.assertEqual(TimeSeries.has_single_frequency(self.dti), True)
        self.assertEqual(TimeSeries.has_single_frequency(self.dti, False), True)
        self.assertEqual(TimeSeries.has_single_frequency(self.dti, True, True), True)

        self.assertEqual(TimeSeries.has_single_frequency(self.series), True)

        self.assertEqual(TimeSeries.has_single_frequency(self.inperfect_dti), False)
        self.assertEqual(TimeSeries.has_single_frequency(self.inperfect_dti,
                                                         variable_duration_freq_included=False,
                                                         skip_duplicate_timestamps=True),
                         False)

        self.assertEqual(TimeSeries.has_single_frequency(self.inperfect_dti,
                                                         skip_duplicate_timestamps=True),
                         False)

        # interesting part
        self.assertEqual(TimeSeries.has_single_frequency(self.monthly_dti), True)
        self.assertEqual(TimeSeries.has_single_frequency(self.monthly_dti,
                                                         variable_duration_freq_included=False),
                         False)

    def test_collapse_to_periods(self):

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

        # now find periods in the time-series
        # should work with 2 types of freq arguments
        for freq in ["30min", pd.to_timedelta("30min")]:
            computed_periods = TimeSeries.collapse_to_periods(dti, freq)
            self.assertEqual(len(computed_periods), len(periods))

            for i in range(len(periods)):
                self.assertEqual(computed_periods[i][0], periods[i][0])
                self.assertEqual(computed_periods[i][1], periods[i][1])

    def test_find_missing_and_extra_periods(self):
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
        self.assertEqual(pd.Timedelta(freq), pd.Timedelta("15min"))  # inferred a 15min freq
        self.assertEqual(len(missing_periods), 2)  # (01:15:00 -> 01:45:00), (02:15:00 -> 02:15:00)
        self.assertEqual(len(extra_points), 2)  # [00:50:00, 02:20:00]

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

        with self.assertRaises(ValueError):
            TimeSeries.find_missing_and_extra_periods(pd.DatetimeIndex([]), '1D')

    # ----------------------------------------
    # deprecated functions (moved elsewhere)
    # ----------------------------------------

    def test_align_timezone(self):

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

        # test with DatetimeIndex
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

    def test_interpolate_daily_to_sub_daily_data(self):

        expected_df = pd.date_range(
            start=pd.to_datetime('2021-01-01 00:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-04 18:00:00+01:00').tz_convert('Europe/Paris'),
            freq='6H',
            tz='Europe/Paris',
            name='time'
        ).to_frame().set_index('time')
        expected_df["value"] = [0.] * 8 + [1.] * 8
        expected_df.index.freq = '6H'

        # test with full kwargs
        sub_df = TimeSeries.interpolate_daily_to_sub_daily_data(
            df=self.single_index_df,
            freq='6H',
            tz="Europe/Paris"
        )

        pd.testing.assert_frame_equal(expected_df, sub_df)

        expected_df1 = pd.date_range(
            start=pd.to_datetime('2021-01-01 00:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-04 18:00:00+01:00').tz_convert('Europe/Paris'),
            freq='6H',
            tz='Europe/Paris',
            name='time'
        ).to_frame()
        expected_df1["value"] = [0.] * 8 + [1.] * 8
        expected_df1["key"] = 'key1'
        expected_df1.index.freq = '6H'

        expected_df2 = pd.date_range(
            start=pd.to_datetime('2021-01-01 00:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-03 18:00:00+01:00').tz_convert('Europe/Paris'),
            freq='6H',
            tz='Europe/Paris',
            name='time'
        ).to_frame()
        expected_df2["value"] = [2.] * 8 + [3.] * 4
        expected_df2["key"] = 'key2'
        expected_df2.index.freq = '6H'

        expected_df = pd.concat([expected_df1, expected_df2], axis=0).set_index(['key', 'time'])

        # test with full kwargs
        sub_df = TimeSeries.interpolate_daily_to_sub_daily_data(
            df=self.multi_index_test,
            freq='6H',
            tz='Europe/Paris'
        )
        pd.testing.assert_frame_equal(expected_df, sub_df)

    def test_interpolate_freq_to_sub_freq_data(self):

        # test interpolate freq to sub-freq on a 6H basis
        expected_df = pd.date_range(
            start=pd.to_datetime('2021-01-01 00:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-04 00:00:00+01:00').tz_convert('Europe/Paris'),
            freq='6H',
            tz='Europe/Paris',
            name='time'
        ).to_frame().set_index('time')
        expected_df.index.freq = '6H'
        expected_df['value'] = [0.] * 5 + [0.25, 0.5, 0.75] + [1] * 5

        sub_df = TimeSeries.interpolate_freq_to_sub_freq_data(
            df=self.single_index_df, freq='6H', tz='Europe/Paris', index_name='time'
        )
        pd.testing.assert_frame_equal(expected_df, sub_df)

        # test interpolate freq to sub-freq using ffill
        expected_df['value'] = [0.] * 8 + [1] * 5
        expected_df.index.name = 'date'
        sub_df = TimeSeries.interpolate_freq_to_sub_freq_data(
            self.single_index_df, freq='6H', tz='Europe/Paris', method='ffill'
        )
        pd.testing.assert_frame_equal(expected_df, sub_df)

        # test interpolation with a multiindex

        expected_df1 = pd.date_range(
            start=pd.to_datetime('2021-01-01 00:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-04 00:00:00+01:00').tz_convert('Europe/Paris'),
            freq='6H',
            tz='Europe/Paris',
            name='time'
        ).to_frame()
        expected_df1["value"] = [0.] * 8 + [1.] * 5
        expected_df1["key"] = 'key1'
        expected_df1.index.freq = '6H'

        expected_df2 = pd.date_range(
            start=pd.to_datetime('2021-01-01 00:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-03 00:00:00+01:00').tz_convert('Europe/Paris'),
            freq='6H',
            tz='Europe/Paris',
            name='time'
        ).to_frame()
        expected_df2["value"] = [2.] * 8 + [3.] * 1
        expected_df2["key"] = 'key2'
        expected_df2.index.freq = '6H'

        expected_df = pd.concat([expected_df1, expected_df2], axis=0).set_index(['key', 'time'])

        sub_df = TimeSeries.interpolate_freq_to_sub_freq_data(
            self.multi_index_test, freq='6H', tz='Europe/Paris', method='ffill', index_name='time'
        )

        pd.testing.assert_frame_equal(expected_df, sub_df)

    def test_forward_fill_final_record(self):
        """
        Test extend_final_data: this is a ffill of the last record
        on the provided frequency
        """

        test_df = pd.date_range(
            start=pd.to_datetime('2021-01-01 00:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-02 00:00:00+01:00').tz_convert('Europe/Paris'),
            freq='12H', tz='Europe/Paris', name='time'
        ).to_frame().set_index('time')
        test_df["value"] = [0., 1., 2.]
        test_df.index.freq = '12H'

        expected_df = pd.date_range(
            start=pd.to_datetime('2021-01-01 00:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-02 12:00:00+01:00').tz_convert('Europe/Paris'),
            freq='12H', tz='Europe/Paris', name='time'
        ).to_frame().set_index('time')
        expected_df["value"] = [0., 1., 2., 2.]
        expected_df.index.freq = '12H'

        sub_df = TimeSeries.forward_fill_final_record(
            df=test_df,
            gap_frequency='1D'
        )
        pd.testing.assert_frame_equal(expected_df, sub_df)

        # forward_fill_final_record with a None cutoff frequency
        # and 3H as the original frequency
        test_df = pd.date_range(
            start=pd.to_datetime('2021-01-01 19:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-01 22:00:00+01:00').tz_convert('Europe/Paris'),
            freq='1H', tz='Europe/Paris', name='time'
        ).to_frame().set_index('time')
        test_df["value"] = [0., 1., 2., 3.]

        expected_df = pd.date_range(
            start=pd.to_datetime('2021-01-01 19:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-02 00:00:00+01:00').tz_convert('Europe/Paris'),
            freq='1H', tz='Europe/Paris', name='time'
        ).to_frame().set_index('time')
        expected_df["value"] = [0., 1., 2., 3., 3., 3.]
        expected_df.index.freq = '1H'

        sub_df = TimeSeries.forward_fill_final_record(
            df=test_df,
            gap_frequency='3H',
            cut_off_frequency=None
        )
        pd.testing.assert_frame_equal(expected_df, sub_df)

        # forward_fill_final_record with a cutoff frequency
        expected_df = pd.date_range(
            start=pd.to_datetime('2021-01-01 19:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-01 23:00:00+01:00').tz_convert('Europe/Paris'),
            freq='1H', tz='Europe/Paris', name='time'
        ).to_frame().set_index('time')
        expected_df["value"] = [0., 1., 2., 3., 3.]
        expected_df.index.freq = '1H'

        sub_df = TimeSeries.forward_fill_final_record(
            df=test_df,
            gap_frequency='3H',
            cut_off_frequency='1D'
        )
        pd.testing.assert_frame_equal(expected_df, sub_df)

    def test_average_to_upper_freq(self):

        # average to upper freq

        test_df = pd.date_range(
            start=pd.to_datetime('2021-01-01 00:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-02 00:00:00+01:00').tz_convert('Europe/Paris'),
            freq='12H', tz='Europe/Paris', name='time'
        ).to_frame().set_index('time')
        test_df["value"] = [0., 1., 2.]

        expected_df = pd.date_range(
            start=pd.to_datetime('2021-01-01 00:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-02 00:00:00+01:00').tz_convert('Europe/Paris'),
            freq='1D', tz='Europe/Paris', name='time'
        ).to_frame().set_index('time')
        expected_df["value"] = [0.5, 2.]
        expected_df.index.freq = '1D'

        sub_df = TimeSeries.average_to_upper_freq(
            df=test_df,
            freq='1D',
            tz="Europe/Paris"
        )
        pd.testing.assert_frame_equal(expected_df, sub_df)
