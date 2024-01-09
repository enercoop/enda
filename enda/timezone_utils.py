from dateutil.relativedelta import relativedelta


class TimezoneUtils:
    @staticmethod
    def is_timezone_aware(dt):
        return dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None

    @classmethod
    def add_interval_to_day_dt(cls, day_dt, interval):
        """Adds an interval (not more precise than a day) to a day,
         correctly dealing with timezone-aware (and naive) day_dt;
         works around daylight savings time changes.

        Normally, to add an interval to a day which is not timezone aware, simply use:
            day_dt + interval.
        This does not work properly for timezone-aware days so we added this function.

        :param day_dt: a timezone_aware datetime which is a day (hour=minute=seconds=microsecond=0)
        :param interval: a interval of type relativedelta not more precise than a day
        """

        if not (
            day_dt.hour == day_dt.minute == day_dt.second == day_dt.microsecond == 0
        ):
            raise ValueError(
                "day_dt must be datetime with only years, months or days (not more precise),"
                " but given: {}, {}".format(type(day_dt), day_dt)
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
                    "(not more precise), but given: {}, {}".format(
                        type(interval), interval
                    )
                )
            )

        tz = day_dt.tzinfo
        day_naive = day_dt.replace(tzinfo=None)
        day_naive = day_naive + interval

        if tz is not None:
            day_aware = tz.localize(day_naive, is_dst=None)
            return day_aware
        else:
            return day_naive
