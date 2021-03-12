import unittest
import pandas as pd
import pytz
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dateutil import parser as date_parser
from enda.timezone_utils import TimezoneUtils


class TestTimezoneUtils(unittest.TestCase):

    TZ_PARIS = pytz.timezone("Europe/Paris")

    def test_pandas_timestamp_bug(self):

        dt = datetime(year=2019, month=10, day=27, hour=2)
        ts = pd.Timestamp(year=2019, month=10, day=27, hour=2)
        dt_bis = ts.to_pydatetime()

        self.assertFalse(TimezoneUtils.is_timezone_aware(dt))
        self.assertFalse(TimezoneUtils.is_timezone_aware(ts))
        self.assertFalse(TimezoneUtils.is_timezone_aware(dt_bis))

        dt1 = TestTimezoneUtils.TZ_PARIS.localize(dt, is_dst=True)
        dt2 = TestTimezoneUtils.TZ_PARIS.localize(dt, is_dst=False)

        ts1 = TestTimezoneUtils.TZ_PARIS.localize(ts, is_dst=True)
        ts2 = TestTimezoneUtils.TZ_PARIS.localize(ts, is_dst=False)

        dt_bis1 = TestTimezoneUtils.TZ_PARIS.localize(dt_bis, is_dst=True)
        dt_bis2 = TestTimezoneUtils.TZ_PARIS.localize(dt_bis, is_dst=False)

        self.assertNotEqual(dt1, dt2)

        self.assertEqual(dt1, ts1)
        self.assertNotEqual(dt2, ts2)  # -> if no bug, these should be equal, but bug

        self.assertEqual(dt1, dt_bis1)  # -> works fine if we convert pandas Timestamp to python datetime
        self.assertEqual(dt2, dt_bis2)

    def test_is_timezone_aware(self):

        for dt, is_aware in [
            (date_parser.isoparse("2018-10-28T00:00:00+02:00"), True),
            (date_parser.isoparse("2018-10-28T00:00:00Z"), True),
            (date_parser.isoparse("2018-10-28T00:00:00"), False),
            (pd.to_datetime("2018-10-28T00:00:00"), False),
            (pd.to_datetime("2018-10-28 00:00:00+02:00"), True),
        ]:
            self.assertEqual(TimezoneUtils.is_timezone_aware(dt), is_aware)

    def test_add_interval_day_dt(self):

        # expect error when day_dt is not an exact day
        self.assertRaises(
            ValueError,
            TimezoneUtils.add_interval_to_day_dt,
            day_dt=pd.to_datetime("2018-03-24T00:01:00+01"),
            interval=relativedelta(days=1)
        )

        # expect error when interval is too precise
        self.assertRaises(
            ValueError,
            TimezoneUtils.add_interval_to_day_dt,
            day_dt=pd.to_datetime("2018-03-24T00:01:00+01"),
            interval=relativedelta(minutes=5)
        )

        for a, b, interval in [
            ("2018-03-24T00:00:00+01:00", "2018-03-25T00:00:00+01:00", relativedelta(days=1)),
            ("2018-03-25T00:00:00+01:00", "2018-03-26T00:00:00+02:00", relativedelta(days=1)),
            ("2018-03-24T00:00:00+01:00", "2018-03-28T00:00:00+02:00", relativedelta(days=4)),
            ("2017-03-24T00:00:00+01:00", "2018-04-25T00:00:00+02:00", relativedelta(years=1, months=1, days=1)),

            ("2020-01-31T00:00:00+01:00", "2020-02-29T00:00:00+01:00", relativedelta(months=1)),
            ("2018-01-31T00:00:00+01:00", "2018-02-28T00:00:00+01:00", relativedelta(months=1)),

            ("2018-10-27T00:00:00+02:00", "2018-10-28T00:00:00+02:00", relativedelta(days=1)),
            ("2018-10-28T00:00:00+02:00", "2018-10-29T00:00:00+01:00", relativedelta(days=1)),
            ("2018-10-30T00:00:00+01:00", "2018-10-27T00:00:00+02:00", relativedelta(days=-3)),
            ("2017-10-28T00:00:00+02:00", "2017-11-02T00:00:00+01:00", relativedelta(days=5)),

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
