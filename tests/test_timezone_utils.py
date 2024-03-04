"""A module for testing the TimezoneUtils class in enda/timezone_utils.py"""

import unittest
import pandas as pd
import pytz
from dateutil.relativedelta import relativedelta
from dateutil import parser as date_parser
from enda.timezone_utils import TimezoneUtils


class TestTimezoneUtils(unittest.TestCase):
    """
    This class aims at testing the functions of the TimezoneUtils class in enda/timezone_utils.py
    """

    TZ_PARIS = pytz.timezone("Europe/Paris")

    def test_is_timezone_aware(self):
        """
        Test the is_timezone_aware function
        """
        for dt, is_aware in [
            (date_parser.isoparse("2018-10-28T00:00:00+02:00"), True),
            (date_parser.isoparse("2018-10-28T00:00:00Z"), True),
            (date_parser.isoparse("2018-10-28T00:00:00"), False),
            (pd.to_datetime("2018-10-28T00:00:00"), False),
            (pd.to_datetime("2018-10-28 00:00:00+02:00"), True),
            (pd.Timestamp(year=2023, month=1, day=1), False),
            (pd.Timestamp(year=2023, month=1, day=1, tz="UTC"), True),
        ]:
            self.assertEqual(TimezoneUtils.is_timezone_aware(dt), is_aware)

    def test_add_interval_day_dt(self):
        """Test the add_interval_day_dt function"""

        # expect error when day_dt is not an exact day
        self.assertRaises(
            ValueError,
            TimezoneUtils.add_interval_to_day_dt,
            day_dt=pd.to_datetime("2018-03-24T00:01:00+01"),
            interval=relativedelta(days=1),
        )

        # expect error when interval is too precise
        self.assertRaises(
            ValueError,
            TimezoneUtils.add_interval_to_day_dt,
            day_dt=pd.to_datetime("2018-03-24T00:00:00+01"),
            interval=relativedelta(minutes=5),
        )

        for a, b, interval in [
            (
                "2018-03-24T00:00:00+01:00",
                "2018-03-25T00:00:00+01:00",
                relativedelta(days=1),
            ),
            (
                "2018-03-25T00:00:00+01:00",
                "2018-03-26T00:00:00+02:00",
                relativedelta(days=1),
            ),
            (
                "2018-03-24T00:00:00+01:00",
                "2018-03-28T00:00:00+02:00",
                relativedelta(days=4),
            ),
            (
                "2017-03-24T00:00:00+01:00",
                "2018-04-25T00:00:00+02:00",
                relativedelta(years=1, months=1, days=1),
            ),
            (
                "2020-01-31T00:00:00+01:00",
                "2020-02-29T00:00:00+01:00",
                relativedelta(months=1),
            ),
            (
                "2018-01-31T00:00:00+01:00",
                "2018-02-28T00:00:00+01:00",
                relativedelta(months=1),
            ),
            (
                "2018-10-27T00:00:00+02:00",
                "2018-10-28T00:00:00+02:00",
                relativedelta(days=1),
            ),
            (
                "2018-10-28T00:00:00+02:00",
                "2018-10-29T00:00:00+01:00",
                relativedelta(days=1),
            ),
            (
                "2018-10-30T00:00:00+01:00",
                "2018-10-27T00:00:00+02:00",
                relativedelta(days=-3),
            ),
            (
                "2017-10-28T00:00:00+02:00",
                "2017-11-02T00:00:00+01:00",
                relativedelta(days=5),
            ),
            # should also work on tz-naive input
            ("2018-03-24T00:00:00", "2018-03-28T00:00:00", relativedelta(days=4)),
        ]:
            at = pd.to_datetime(a)
            # read like that, timezone is pytz.FixedOffset(60) or pytz.FixedOffset(120)
            # we convert it to a real life one
            if at.tzinfo is not None:
                at = at.tz_convert(TestTimezoneUtils.TZ_PARIS)

            bt = pd.to_datetime(b)
            if bt.tzinfo is not None:
                bt = bt.tz_convert(TestTimezoneUtils.TZ_PARIS)

            self.assertEqual(at.isoformat(), a)
            self.assertEqual(bt.isoformat(), b)

            ct = TimezoneUtils.add_interval_to_day_dt(day_dt=at, interval=interval)

            self.assertEqual(bt, ct)

    def test_convert_dtype_from_object_to_tz_aware(self):
        """
        test convert_dtype_from_object_to_tz_aware, which is useful typically when daylight savings changes
        """

        object_series = pd.Series(
            [
                pd.to_datetime("2021-12-31 02:00:00+02:00"),
                pd.to_datetime("2021-12-31 03:00:00+02:00"),
                pd.to_datetime("2021-12-31 03:00:00+01:00"),
                pd.to_datetime("2021-12-31 04:00:00+01:00"),
            ]
        )

        # s is of dtype "object" because of varying timezone
        # cannot convert because of varying timezone
        self.assertEqual(object_series.dtype, "object")
        with self.assertRaises(ValueError):
            pd.DatetimeIndex(object_series)

        # test the simple conversion for a series with elements of dtype object
        expected_series = pd.Series(
            [
                pd.to_datetime("2021-12-31 01:00:00+01:00"),
                pd.to_datetime("2021-12-31 02:00:00+01:00"),
                pd.to_datetime("2021-12-31 03:00:00+01:00"),
                pd.to_datetime("2021-12-31 04:00:00+01:00"),
            ],
            dtype="datetime64[ns, Europe/Paris]",
        )

        # should work for 2 types of timezone object
        for tz in ["Europe/Paris", pytz.timezone("Europe/Paris")]:
            result_dt_series = TimezoneUtils.convert_dtype_from_object_to_tz_aware(
                object_series, tz_info=tz
            )
            pd.testing.assert_series_equal(expected_series, result_dt_series)

        # test that given a tz-aware pd.DatetimeIndex, it returns the same if the same time zone is provided
        expected_dti = pd.DatetimeIndex(
            [
                pd.to_datetime("2021-12-31 01:00:00+01:00"),
                pd.to_datetime("2021-12-31 02:00:00+01:00"),
                pd.to_datetime("2021-12-31 03:00:00+01:00"),
                pd.to_datetime("2021-12-31 04:00:00+01:00"),
            ],
            dtype="datetime64[ns, Europe/Paris]",
        )

        for tz in ["Europe/Paris", pytz.timezone("Europe/Paris")]:
            result_dti = TimezoneUtils.convert_dtype_from_object_to_tz_aware(
                expected_dti, tz_info=tz
            )
            pd.testing.assert_index_equal(expected_dti, result_dti)

        # test that given a tz-aware pd.Series, it behaves like tz_convert if another
        # time zone is required
        tz_aware_series = expected_series.copy()  # Paris time zone

        expected_series = pd.Series(
            [
                pd.to_datetime("2021-12-31 09:00:00+09:00"),
                pd.to_datetime("2021-12-31 10:00:00+09:00"),
                pd.to_datetime("2021-12-31 11:00:00+09:00"),
                pd.to_datetime("2021-12-31 12:00:00+09:00"),
            ],
            dtype="datetime64[ns, Asia/Tokyo]",
        )

        for tz in ["Asia/Tokyo", pytz.timezone("Asia/Tokyo")]:
            result_series = TimezoneUtils.convert_dtype_from_object_to_tz_aware(
                tz_aware_series, tz
            )
            pd.testing.assert_series_equal(expected_series, result_series)

        # test that given a tz-aware pd.DatetimeIndex, it behaves like tz_convert if another
        # time zone is required
        tz_aware_dti = expected_dti.copy()  # Paris time zone

        expected_dti = pd.DatetimeIndex(
            [
                pd.to_datetime("2021-12-31 09:00:00+09:00"),
                pd.to_datetime("2021-12-31 10:00:00+09:00"),
                pd.to_datetime("2021-12-31 11:00:00+09:00"),
                pd.to_datetime("2021-12-31 12:00:00+09:00"),
            ],
            dtype="datetime64[ns, Asia/Tokyo]",
        )

        for tz in ["Asia/Tokyo", pytz.timezone("Asia/Tokyo")]:
            result_dti = TimezoneUtils.convert_dtype_from_object_to_tz_aware(
                tz_aware_dti, tz
            )
            pd.testing.assert_index_equal(expected_dti, result_dti)

        # test it does not work with a tz-naive DatetimeIndex
        # we should use tz_localize
        naive_dti = pd.DatetimeIndex(
            [
                pd.to_datetime("2021-12-31 02:00:00"),
                pd.to_datetime("2021-12-31 03:00:00"),
                pd.to_datetime("2021-12-31 04:00:00"),
            ]
        )

        with self.assertRaises(TypeError):
            TimezoneUtils.convert_dtype_from_object_to_tz_aware(
                naive_dti, "Europe/Paris"
            )

        # with a naive dt series
        naive_series = pd.Series(
            [
                pd.to_datetime("2021-12-31 02:00:00"),
                pd.to_datetime("2021-12-31 03:00:00"),
                pd.to_datetime("2021-12-31 04:00:00"),
            ],
            dtype="datetime64[ns]",
        )

        with self.assertRaises(TypeError):
            TimezoneUtils.convert_dtype_from_object_to_tz_aware(
                naive_series, "Europe/Paris"
            )

        # dumb tz_info
        with self.assertRaises(TypeError):
            TimezoneUtils.convert_dtype_from_object_to_tz_aware(object_series, 4)

    def test_set_timezone(self):
        """
        Test set_timezone with multiple series
        """

        # with a naive dt series
        naive_series = pd.Series(
            [
                pd.to_datetime("2021-12-31 02:00:00"),
                pd.to_datetime("2021-12-31 03:00:00"),
                pd.to_datetime("2021-12-31 04:00:00"),
            ],
            dtype="datetime64[ns]",
        )

        # test case tz_base is not given
        expected_series = pd.Series(
            [
                pd.to_datetime("2021-12-31 02:00:00+01"),
                pd.to_datetime("2021-12-31 03:00:00+01"),
                pd.to_datetime("2021-12-31 04:00:00+01"),
            ],
            dtype="datetime64[ns, Europe/Paris]",
        )

        result_series = TimezoneUtils.set_timezone(naive_series, tz_info="Europe/Paris")

        pd.testing.assert_series_equal(expected_series, result_series)

        # test case tz_base is given, e.G UTC -> it assumes naive_series is given in UTC time
        # it changes in case we set it to Europe/Paris
        expected_series = pd.Series(
            [
                pd.to_datetime("2021-12-31 03:00:00+01"),
                pd.to_datetime("2021-12-31 04:00:00+01"),
                pd.to_datetime("2021-12-31 05:00:00+01"),
            ],
            dtype="datetime64[ns, Europe/Paris]",
        )

        result_series = TimezoneUtils.set_timezone(
            naive_series, tz_info="Europe/Paris", tz_base="UTC"
        )

        pd.testing.assert_series_equal(expected_series, result_series)

        # with a tz-aware series
        aware_series = pd.Series(
            [
                pd.to_datetime("2021-12-31 09:00:00+09"),
                pd.to_datetime("2021-12-31 10:00:00+09"),
                pd.to_datetime("2021-12-31 11:00:00+09"),
            ],
            dtype="datetime64[ns, Asia/Tokyo]",
        )

        # test case tz_base is not given
        expected_series = pd.Series(
            [
                pd.to_datetime("2021-12-31 01:00:00+01"),
                pd.to_datetime("2021-12-31 02:00:00+01"),
                pd.to_datetime("2021-12-31 03:00:00+01"),
            ],
            dtype="datetime64[ns, Europe/Paris]",
        )

        result_series = TimezoneUtils.set_timezone(aware_series, tz_info="Europe/Paris")

        pd.testing.assert_series_equal(expected_series, result_series)

        # same but with a datetimeindex

        # with a naive datetimeindex
        naive_dti = pd.DatetimeIndex(
            [
                pd.to_datetime("2021-12-31 02:00:00"),
                pd.to_datetime("2021-12-31 03:00:00"),
                pd.to_datetime("2021-12-31 04:00:00"),
            ],
            dtype="datetime64[ns]",
        )

        # test case tz_base is not given
        expected_dti = pd.DatetimeIndex(
            [
                pd.to_datetime("2021-12-31 02:00:00+01"),
                pd.to_datetime("2021-12-31 03:00:00+01"),
                pd.to_datetime("2021-12-31 04:00:00+01"),
            ],
            dtype="datetime64[ns, Europe/Paris]",
        )

        result_dti = TimezoneUtils.set_timezone(naive_dti, tz_info="Europe/Paris")

        pd.testing.assert_index_equal(expected_dti, result_dti)

        # test case tz_base is given, e.G UTC -> it assumes naive_dti is given in UTC time
        # it changes in case we set it to Europe/Paris
        expected_dti = pd.DatetimeIndex(
            [
                pd.to_datetime("2021-12-31 03:00:00+01"),
                pd.to_datetime("2021-12-31 04:00:00+01"),
                pd.to_datetime("2021-12-31 05:00:00+01"),
            ],
            dtype="datetime64[ns, Europe/Paris]",
        )

        result_dti = TimezoneUtils.set_timezone(
            naive_dti, tz_info="Europe/Paris", tz_base="UTC"
        )

        pd.testing.assert_index_equal(expected_dti, result_dti)

        # with a tz-aware dti
        aware_dti = pd.DatetimeIndex(
            [
                pd.to_datetime("2021-12-31 09:00:00+09"),
                pd.to_datetime("2021-12-31 10:00:00+09"),
                pd.to_datetime("2021-12-31 11:00:00+09"),
            ],
            dtype="datetime64[ns, Asia/Tokyo]",
        )

        # test case tz_base is not given
        expected_dti = pd.DatetimeIndex(
            [
                pd.to_datetime("2021-12-31 01:00:00+01"),
                pd.to_datetime("2021-12-31 02:00:00+01"),
                pd.to_datetime("2021-12-31 03:00:00+01"),
            ],
            dtype="datetime64[ns, Europe/Paris]",
        )

        result_dti = TimezoneUtils.set_timezone(aware_dti, tz_info="Europe/Paris")

        pd.testing.assert_index_equal(expected_dti, result_dti)

        # check raise if tz_info is wrong type
        with self.assertRaises(TypeError):
            TimezoneUtils.set_timezone(naive_dti, tz_info=5)

        with self.assertRaises(TypeError):
            TimezoneUtils.set_timezone(naive_dti, tz_info="Europe/Paris", tz_base=5)
