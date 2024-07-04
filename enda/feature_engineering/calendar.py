"""A module useful for knowing special days such as public/school holidays or lockdown periods"""

import abc
import datetime
from typing import Union
import warnings
import unidecode

import pandas as pd

from jours_feries_france import JoursFeries
from vacances_scolaires_france import SchoolHolidayDates

from enda.feature_engineering.datetime_features import DatetimeFeature
from enda.tools.resample import Resample
from enda.tools.timeseries import TimeSeries
from enda.tools.decorators import warning_deprecated_name

TZ_PARIS = "Europe/Paris"


# --------- Holidays ------------

class Holidays:
    """
    Factory class to gather holidays for different countries
    """

    @staticmethod
    def get_public_holidays(country: str, years_list: list[int] = None, handling_missing_year: str = 'warning'):
        """
        Return public holidays for a year, and a country
        :param country: the country for which holidays must be returned
        :param years_list: list of years for which holidays must be returned
        :param handling_missing_year: either 'warning' (default)  or 'error' or 'silent'
        :return: a dataframe with the school holidays for the concerned country
        """
        if country == 'FR':
            public_holidays = FrenchHolidays.get_public_holidays(years_list=years_list,
                                                                 handling_missing_year=handling_missing_year)
        else:
            raise NotImplementedError(f"Country '{country}' not supported.")

        return public_holidays

    @staticmethod
    def get_school_holidays(country: str, years_list: list[int] = None, handling_missing_year: str = 'warning'):
        """
        Return school holidays for a list of years, and a country
        :param country: the country for which holidays must be returned
        :param years_list: list of years for which holidays must be returned
        :param handling_missing_year: either 'warning' (default)  or 'error' or 'silent'
        :return: a dataframe with the school holidays for the concerned country
        """
        if country == 'FR':
            school_holidays = FrenchHolidays.get_school_holidays(years_list=years_list,
                                                                 handling_missing_year=handling_missing_year)
        else:
            raise NotImplementedError(f"Country '{country}' not supported.")

        return school_holidays


class BaseHolidays:
    """
    Base class for holidays (public and schools)
    """

    @staticmethod
    @abc.abstractmethod
    def get_public_holidays(years_list: list[int] = None,
                            orientation: str = "rows",
                            handling_missing_year: str = 'warning'
                            ) -> pd.DataFrame:
        """
        Return public holidays for a list of year, and a country
        :param years_list: list of years for which holidays must be returned
        :param orientation: rows or columns
        :param handling_missing_year: either 'warning' (default)  or 'error' or 'silent'
        :return: a list of datetime which are the school holidays for the concerned country
        """
        raise NotImplementedError("Abstract method")

    @staticmethod
    @abc.abstractmethod
    def get_school_holidays(years_list: list[int],
                            handling_missing_year: str = 'warning') -> pd.DataFrame:
        """
        Return school holidays for a list of year
        :param years_list: list of years for which holidays must be returned
        :param handling_missing_year: either 'warning' (default)  or 'error' or 'silent'
        :return: a list of datetime which are the school holidays for the concerned country
        """
        raise NotImplementedError("Abstract method")


class FrenchHolidays(BaseHolidays):
    """
    Child class for French holidays (public and schools)
    """

    @staticmethod
    def get_public_holidays(years_list: list[int] = None,
                            orientation: str = "rows",
                            handling_missing_year: str = 'warning') -> pd.DataFrame:
        """
        Return public French holidays for a list of years
        :param years_list: list of years for which holidays must be returned
        :param orientation: 'rows' (default) or 'columns'. The way to orient the dataframe
        :param handling_missing_year: either 'warning' (default)  or 'error' or 'silent'
        :return: a list of datetime which are the school holidays for the concerned country
        """

        if years_list is None:
            years_list = range(2000, 2051)

        all_years_holiday_df = pd.DataFrame()
        for year in years_list:
            try:
                year_holiday_df = pd.DataFrame.from_dict(JoursFeries.for_year(year), orient="index")
                year_holiday_df.index = year_holiday_df.index.map(unidecode.unidecode)
                all_years_holiday_df = pd.concat([all_years_holiday_df, year_holiday_df.T], ignore_index=True)
            except Exception as exception:
                if handling_missing_year == 'warning':
                    warnings.warn(f"Missing french public holidays : {exception}")
                elif handling_missing_year == 'error':
                    raise exception

        if orientation != "columns":
            all_years_holiday_df = (
                all_years_holiday_df
                .stack()
                .reset_index(level=0, drop=True)
                .rename_axis("public_holiday_name")
                .to_frame("date")
                .reset_index(drop=False)
                .filter(["date", "public_holiday_name"])
            )

            all_years_holiday_df["date"] = pd.to_datetime(all_years_holiday_df["date"])

        return all_years_holiday_df

    @staticmethod
    def get_school_holidays(years_list: list[int],
                            handling_missing_year: str = 'warning') -> pd.DataFrame:
        """
        Return school holidays for a list of year
        :param years_list: list of years for which holidays must be returned
        :param handling_missing_year: either 'warning' (default)  or 'error' or 'silent'
        :return: a list of datetime which are the school holidays for the concerned country
        """

        if years_list is None:
            years_list = range(2000, 2051)

        school_holidays_dates = SchoolHolidayDates()
        all_years_holiday_df = pd.DataFrame()
        for year in years_list:
            try:
                year_holiday_df = pd.DataFrame.from_dict(school_holidays_dates.holidays_for_year(year), orient="index")
                year_holiday_df = year_holiday_df.rename(columns={"nom_vacances": "school_holiday_name"})
                year_holiday_df.loc[:, "school_holiday_name"] = year_holiday_df.loc[:, "school_holiday_name"].map(
                    unidecode.unidecode
                )
                year_holiday_df = year_holiday_df.reset_index(drop=True)
                all_years_holiday_df = pd.concat([all_years_holiday_df, year_holiday_df], ignore_index=True)
            except Exception as exception:
                if handling_missing_year == 'warning':
                    warnings.warn(f"Missing french school holidays : {exception}")
                elif handling_missing_year == 'error':
                    raise exception
        return all_years_holiday_df


# --------- Calendar ------------


class Calendar:
    """
    Factory class allowing to gather special days (lockdowns, long weekends) for a given country
    """

    @staticmethod
    def get_lockdown(country: str, years_list: list[int] = None):
        """
        Return a dataframe indicating for each day if a national lockdown was ongoing.
        :param country: country of interest
        :param years_list: the list of target years
        :return: a dataframe with a daily DatetimeIndex and a float 'lockdown' column containing
                 1 if the day is within a lockdown period and 0 otherwise
        """
        if country == 'FR':
            return FrenchCalendar.get_lockdown(years_list=years_list)
        raise NotImplementedError(f"Country '{country}' not supported.")

    @staticmethod
    def get_public_holidays(country: str, years_list: list[int] = None, handling_missing_year: str = 'warning'):
        """
        Return a dataframe (at max from 2000-01-01 to 2050-12-31) indicating for each day
        whether it is a public holiday (denoted by a 1) or not (denoted by a 0)
        :param country: country of interest
        :param years_list: list of years for which holidays must be returned
        :param handling_missing_year: either 'warning' (default)  or 'error' or 'silent'
        :return: a dataFrame with a daily DatetimeIndex and a float 'public holiday' column containing
                 1 if the day is a public holiday and 0 otherwise
        """
        return BaseCalendar.get_public_holidays(country, years_list=years_list,
                                                handling_missing_year=handling_missing_year)

    @staticmethod
    def get_extra_long_weekend(country: str, years_list: list[int] = None, handling_missing_year: str = 'warning'):
        """
        Return a dataframe (at max from 2000-01-01 to 2050-12-31) indicating for each day
        - if the previous (resp. the next day) is a public holiday
        AND
        - if the current day is a friday (resp. a monday)
        If both conditions are fulfilled then the day is denoted by a 1 (0 otherwise)
        :param country: country of interest
        :param years_list: the list of target years
        :param handling_missing_year: either 'warning' (default)  or 'error' or 'silent'
        :return: a DataFrame with a daily DatetimeIndex and an int 'extra_long_weekend' column containing 1 if
                 the day meets the criteria described above and 0 otherwise
        """
        return BaseCalendar.get_extra_long_weekend(country,
                                                   years_list=years_list,
                                                   handling_missing_year=handling_missing_year)

    @staticmethod
    def feature_special_days(country: str,
                             years_list: list[int] = None,
                             freq: Union[str, pd.Timedelta] = "30min",
                             handling_missing_year: str = 'warning'):
        """
        Return a DataFrame featuring all special days
        :param country: country of interest
        :param years_list: list of years for which special days must be returned
        :param handling_missing_year: either 'warning' (default)  or 'error' or 'silent'
        :param freq: A string indicating the frequency at which to interpolate the DataFrame
        """
        if country == 'FR':
            return FrenchCalendar.feature_special_days(years_list=years_list,
                                                       freq=freq,
                                                       handling_missing_year=handling_missing_year)
        raise NotImplementedError(f"Country '{country}' not supported.")

    # ----- deprecated functions -----

    @staticmethod
    @warning_deprecated_name(
        namespace_name="Calendar", new_namespace_name="Resample", new_function_name="upsample_and_interpolate"
    )
    def interpolate_daily_to_subdaily_data(
            df: pd.DataFrame, freq: str, method: str = "ffill", tz: str = TZ_PARIS
    ) -> pd.DataFrame:
        """
        Interpolate daily data in a dataframe (with a DatetimeIndex) to sub-daily data using a given method.
        :param df: pd.DataFrame
        :param freq: a frequency < 'D' (e.g. 'H', '30min', '15min', etc.)
        :param method: how data is interpolated between two consecutive dates (e.g. 'ffill', 'linear', etc.)
        :param tz: timezone (TZ_Paris)
        :return: pd.DataFrame
        """
        return TimeSeries.interpolate_daily_to_sub_daily_data(
            df, freq=freq, method=method, tz=tz
        )

    @warning_deprecated_name(
        namespace_name="Calendar", new_namespace_name="FrenchCalendar", new_function_name="get_number_school_areas_off"
    )
    def get_school_holidays(self) -> pd.DataFrame:
        """
        Return number of school areas off in France
        """
        return FrenchCalendar.get_number_school_areas_off()

    @warning_deprecated_name(namespace_name="Calendar", new_function_name="get_lockdown")
    def get_french_lockdown(self) -> pd.DataFrame:
        """
        Return lockdown days for France
        """
        return Calendar.get_lockdown(country='FR')


class BaseCalendar:
    """
    Base class allowing to define functions meant to gather special days
    (lockdowns, long weekends) for a given country
    """

    @staticmethod
    @abc.abstractmethod
    def get_lockdown(years_list: list[int] = None) -> pd.DataFrame:
        """
        Return a dataframe indicating for each day if a national lockdown was ongoing.
        :param years_list: the list of target years
        :return: a dataframe with a daily DatetimeIndex and a float 'lockdown' column containing
                 1 if the day is within a lockdown period and 0 otherwise
        """
        raise NotImplementedError()

    @staticmethod
    def get_public_holidays(country: str = 'FR', years_list: list[int] = None,
                            handling_missing_year: str = 'warning') -> pd.DataFrame:
        """
        Return a dataframe (at max from 2000-01-01 to 2050-12-31) indicating for each day
        whether it is a public holiday (denoted by a 1) or not (denoted by a 0)
        :param country: country of interest
        :param years_list: list of years for which holidays must be returned
        :param handling_missing_year: either 'warning' (default)  or 'error' or 'silent'
        :return: a dataFrame with a daily DatetimeIndex and a float 'public holiday' column containing
                 1 if the day is a public holiday and 0 otherwise
        """

        if years_list is None:
            years_list = range(2000, 2051)

        # get public holidays
        public_holidays = Holidays.get_public_holidays(country=country,
                                                       years_list=years_list,
                                                       handling_missing_year=handling_missing_year)

        # work it out
        public_holidays = public_holidays.set_index("date")
        public_holidays.index = pd.to_datetime(public_holidays.index)
        public_holidays = public_holidays[
            ~public_holidays.index.duplicated(keep="first")
        ]
        public_holidays["public_holiday"] = 1

        # reindex and fillna
        start_date = datetime.date(min(years_list), 1, 1)
        incl_end_date = datetime.date(max(years_list), 12, 31)
        public_holidays = (
            public_holidays
            .reindex(pd.date_range(start_date, incl_end_date))
            .fillna(0)
        )

        return public_holidays[["public_holiday"]]

    @staticmethod
    def get_extra_long_weekend(country: str = 'FR', years_list: list[int] = None,
                               handling_missing_year: str = "warning") -> pd.DataFrame:
        """
        Return a dataframe (at max from 2000-01-01 to 2050-12-31) indicating for each day
        - if the previous (resp. the next day) is a public holiday
        AND
        - if the current day is a friday (resp. a monday)
        If both conditions are fulfilled then the day is denoted by a 1 (0 otherwise)
        :param country: country of interest
        :param years_list: the list of target years
        :param handling_missing_year: either 'warning' (default)  or 'error' or 'silent'
        :return: a DataFrame with a daily DatetimeIndex and an int 'extra_long_weekend' column containing 1 if
                 the day meets the criteria described above and 0 otherwise
        """

        public_holidays = Calendar.get_public_holidays(country, years_list=years_list,
                                                       handling_missing_year=handling_missing_year)
        public_holidays = DatetimeFeature.split_datetime(
            public_holidays, split_list=["dayofweek"], index=True
        )

        public_holidays["is_yesterday_day_off"] = public_holidays[
            "public_holiday"
        ].shift()
        public_holidays["is_tomorrow_day_off"] = public_holidays[
            "public_holiday"
        ].shift(-1)
        public_holidays["extra_long_weekend"] = 0

        mondays = public_holidays[public_holidays["dayofweek"] == 0]
        mondays_off = mondays[mondays["is_tomorrow_day_off"] == 1].index

        fridays = public_holidays[public_holidays["dayofweek"] == 4]
        fridays_off = fridays[fridays["is_yesterday_day_off"] == 1].index

        extra_long_weekend_index = mondays_off.append(fridays_off)
        extra_long_weekend_index = sorted(extra_long_weekend_index)

        public_holidays.loc[extra_long_weekend_index, "extra_long_weekend"] = 1

        return public_holidays[["extra_long_weekend"]]

    @staticmethod
    @abc.abstractmethod
    def feature_special_days(years_list: list[int] = None, freq: Union[str, pd.Timedelta] = "30min") -> pd.DataFrame:
        """
        Return a DataFrame featuring all special days
        :param years_list: list of years for which special days must be returned
        :param freq: A string indicating the frequency at which to interpolate the DataFrame
        """
        raise NotImplementedError()


class FrenchCalendar(BaseCalendar):
    """
    Child class for French calendar
    """

    @staticmethod
    def get_lockdown(years_list: list[int] = None) -> pd.DataFrame:
        """
        Return a dataframe from indicating for each day if national lockdown was ongoing.
        So far, the main lockdown period goes from 2020-03-17 to 2020-05-11.
        :param years_list: the list of target years
        :return: a dataframe with a daily DatetimeIndex and a float 'lockdown' column containing
                 1 if the day is within a lockdown period and 0 otherwise
        """

        # define years list
        if years_list is None:
            years_list = range(2000, 2051)

        # build french lockdown
        lockdown_period = pd.date_range(pd.to_datetime("2020-03-17"), pd.to_datetime("2020-05-11"))
        lockdown_df = pd.DataFrame(index=lockdown_period, columns=["lockdown"], data=1)
        lockdown_df.index.name = "date"

        # reindex over the required years
        start_date = datetime.date(years_list[0], 1, 1)
        incl_end_date = datetime.date(years_list[-1], 12, 31)
        lockdown_df = (
            lockdown_df
            .reindex(pd.date_range(start_date, incl_end_date))
            .fillna(0)
        )

        lockdown_df = lockdown_df.loc[lockdown_df.index.year.isin(years_list)]

        return lockdown_df

    @staticmethod
    def get_number_school_areas_off(years_list=None, handling_missing_year: str = "warning") -> pd.DataFrame:
        """
        Return a dataframe from 2000-01-01 to as far as possible indicating for each day
        the number of school areas (zone A, B et C) in vacation (either 0, 1, 2 or 3)
        :param years_list: the list of target years
        :param handling_missing_year: either 'warning' (default)  or 'error' or 'silent'
        :return: a DataFrame with a daily DatetimeIndex and a float 'nb_schools_area_off'
                 column indicating the number of school areas in vacation
        """

        school_holidays = FrenchHolidays.get_school_holidays(years_list=years_list,
                                                             handling_missing_year=handling_missing_year)

        school_holidays = school_holidays.set_index("date")
        school_holidays.index = pd.to_datetime(school_holidays.index)
        school_holidays = school_holidays.drop("school_holiday_name", axis=1)

        school_holidays["nb_school_areas_off"] = school_holidays.sum(axis=1)
        school_holidays = school_holidays.asfreq("D")
        school_holidays = school_holidays.fillna(0)

        return school_holidays[["nb_school_areas_off"]]

    @staticmethod
    def feature_special_days(years_list: list[int] = None,
                             freq: Union[str, pd.Timedelta] = "30min",
                             index_name: str = 'time',
                             handling_missing_year: str = "warning"
                             ) -> pd.DataFrame:
        """
        Return a DataFrame containing all special french days: public and school holidays, lockdowns, and extra long
            weekends
        :param years_list: the list of years
        :param freq: A string indicating the frequency at which to interpolate the DataFrame
        :param index_name:A string indicating the name of the resulting index
        :param handling_missing_year: either 'warning' (default)  or 'error' or 'silent'
        :return:
            A DataFrame with a DatetimeIndex at the specified frequency, and 4 float columns :
            - The 'lockdown' column contains 1 if the timestamp is within a lockdown period and 0 otherwise
            - The 'public_holiday' column contains 1 if the day is a public holiday and 0 otherwise
            - The 'nb_school_areas_off' indicates the number of French zones (3 zones : A, B and C) that are in
            school holidays
            - The 'extra_long_weekend' contains 1 if the day is part of a long weekend (a Monday with a public holiday
            on Tuesday, or a Friday with a public holiday in Thursday) and 0 otherwise
        """

        # get all constitutive elements of special days
        lockdown_df = FrenchCalendar.get_lockdown(years_list=years_list)
        public_holidays_df = FrenchCalendar.get_public_holidays(years_list=years_list,
                                                                handling_missing_year=handling_missing_year)
        school_areas_off_df = FrenchCalendar.get_number_school_areas_off(years_list=years_list,
                                                                         handling_missing_year=handling_missing_year)
        extra_long_weekend_df = FrenchCalendar.get_extra_long_weekend(years_list=years_list,
                                                                      handling_missing_year=handling_missing_year)

        # resample everything to the desired freq
        lockdown_df = Resample.upsample_and_interpolate(
            lockdown_df, freq=freq, method="ffill", forward_fill=True, index_name=index_name, tz_info=TZ_PARIS
        )
        public_holidays_df = Resample.upsample_and_interpolate(
            public_holidays_df, freq=freq, method="ffill", forward_fill=True, index_name=index_name, tz_info=TZ_PARIS
        )
        school_areas_off_df = Resample.upsample_and_interpolate(
            school_areas_off_df, freq=freq, method="ffill", forward_fill=True, index_name=index_name, tz_info=TZ_PARIS
        )
        extra_long_weekend_df = Resample.upsample_and_interpolate(
            extra_long_weekend_df, freq=freq, method="ffill", forward_fill=True, index_name=index_name, tz_info=TZ_PARIS
        )

        result = pd.concat(
            [
                lockdown_df,
                public_holidays_df,
                school_areas_off_df,
                extra_long_weekend_df,
            ],
            axis=1,
            join="outer",
        )

        return result
