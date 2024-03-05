"""A module useful for knowing special days such as public/school holidays or lockdown periods"""

import datetime
import warnings
import pandas as pd
from enda.feature_engineering.datetime_features import DatetimeFeature
from enda.tools.timeseries import TimeSeries

try:
    import unidecode
    from jours_feries_france import JoursFeries
    from vacances_scolaires_france import SchoolHolidayDates, UnsupportedYearException
except ImportError as exc:
    raise ImportError(
        "unidecode, jours_feries_france, vacances_scolaires_france are required if you want to use "
        "enda's FrenchHolidays and FrenchCalendar classes. "
        "Try: pip install jours-feries-france vacances-scolaires-france Unidecode"
    ) from exc


class FrenchHolidays:
    """
    Gather methods that allow to get special days such as public and school holidays in France
    """

    @staticmethod
    def get_public_holidays(
        year_list: list[int] = None, orientation: str = "rows"
    ) -> pd.DataFrame:
        """
        Compute a DataFrame indicating all public holidays in France
        :param year_list: A list of integers representing the years for which we want public holidays information
        :param orientation: Either 'rows' or 'columns', define the format of the DataFrame returned.
            If 'rows', then the DataFrame will have 2 columns 'date' and 'nom_jour_ferie' indicating public holidays
            name and associated date.
            If 'columns', then the DataFrame will have the different public holidays types as columns, and each row
            will contain associated dates for a year
        :return: A DataFrame of public holidays.
        """

        if year_list is None:
            year_list = range(2000, 2051)

        result = pd.DataFrame()
        for year in year_list:
            try:
                res = JoursFeries.for_year(year)
                df_res = pd.DataFrame.from_dict(res, orient="index")
                df_res.index = df_res.index.map(unidecode.unidecode)
                result = pd.concat([result, df_res.T], ignore_index=True)
            except UnsupportedYearException as e:
                warnings.warn(f"Missing french public holidays : {e}")

        if orientation != "columns":
            result = result.stack().reset_index(level=0, drop=True)
            result.index.name = "nom_jour_ferie"
            result = result.to_frame("date")
            result = result.reset_index(drop=False)
            result = result[["date", "nom_jour_ferie"]]

        return result

    @staticmethod
    def get_school_holidays(year_list: list[int] = None) -> pd.DataFrame:
        """
        Get a DataFrame indicating school holidays periods in France (these include weekends at the beginning/end of
            each period)
        :param year_list: A list of integers representing the years for which we want school holidays information
        :return:
            A DataFrame containing days for which at least one of the three French zones (A, B and C) is in holidays.
            The DataFrame has a 'date' column which indicates the day, one column for each zone with booleans
            indicating if the corresponding zone is in school holidays, and a 'nom_vacances' column indicating the
            name of the related holiday period
        """

        if year_list is None:
            current_year = datetime.datetime.utcnow().year
            year_list = range(2000, current_year + 2)

        d = SchoolHolidayDates()
        result = pd.DataFrame()
        for year in year_list:
            try:
                res = d.holidays_for_year(year)
                df_res = pd.DataFrame.from_dict(res, orient="index")
                df_res["nom_vacances"] = df_res["nom_vacances"].map(unidecode.unidecode)
                df_res = df_res.reset_index(drop=True)
                result = pd.concat([result, df_res], ignore_index=True)
            except UnsupportedYearException as e:
                warnings.warn(f"Missing french school holidays : {e}")

        return result


class Calendar:
    """
    A class allowing to gather special days (school/public holidays, lockdowns, long weekends) for a given country
    """

    def __init__(self, country: str = "FR"):
        self.country = country

    def get_french_lockdown(self) -> pd.DataFrame:
        """
        Return a dataframe from 2000-01-01 to 2050-12-25 indicating for each day if national lockdown was ongoing.
        So far, the main lockdown period goes from 2020-03-17 to 2020-05-11.
        :return:
            A DataFrame with a daily DatetimeIndex and a float 'lockdown' column containing 1 if the day is within a
            lockdown period and 0 otherwise
        """

        if self.country != "FR":
            raise NotImplementedError(f"Public holidays in {self.country} unknown")

        start_lockdown_date = pd.to_datetime("2020-03-17")
        end_lockdown_date = pd.to_datetime("2020-05-11")
        lockdown_period = pd.date_range(start_lockdown_date, end_lockdown_date)

        df_lockdown = pd.DataFrame(index=lockdown_period, columns=["lockdown"], data=1)
        df_lockdown.index.name = "date"

        result = df_lockdown.reindex(pd.date_range("2000-01-01", "2050-12-25"))
        result = result.fillna(0)

        return result

    def get_public_holidays(self) -> pd.DataFrame:
        """
        Return a dataframe from 2000-01-01 to 2050-12-25 indicating for each day
        whether it is a public holiday (denoted by a 1) or not (denoted by a 0)
        :return:
            A DataFrame with a daily DatetimeIndex and a float 'public holiday' column containing 1 if the day is a
            public holiday and 0 otherwise
        """

        if self.country == "FR":
            public_holidays = FrenchHolidays.get_public_holidays()
        else:
            raise NotImplementedError(f"Public holidays in {self.country} unknown")

        public_holidays = public_holidays.set_index("date")
        public_holidays.index = pd.to_datetime(public_holidays.index)
        public_holidays = public_holidays[
            ~public_holidays.index.duplicated(keep="first")
        ]  # 2008-05-01

        public_holidays["public_holiday"] = 1
        public_holidays = public_holidays.asfreq("D")
        public_holidays = public_holidays.fillna(0)

        return public_holidays[["public_holiday"]]

    def get_school_holidays(self) -> pd.DataFrame:
        """
        Return a dataframe from 2000-01-01 to as far as possible (2021-08-29) indicating for each day
        the number of school areas (zone A, B et C) in vacation (either 0, 1, 2 or 3)
        :return:
            A DataFrame with a daily DatetimeIndex and a float 'nb_schools_area_off' column indicating the number
            of school areas in vacation
        """

        if self.country == "FR":
            school_holidays = FrenchHolidays.get_school_holidays()
        else:
            raise NotImplementedError(f"School holidays in {self.country} unknown")

        school_holidays = school_holidays.set_index("date")
        school_holidays.index = pd.to_datetime(school_holidays.index)
        school_holidays = school_holidays.drop("nom_vacances", axis=1)

        school_holidays["nb_school_areas_off"] = school_holidays.sum(axis=1)
        school_holidays = school_holidays.asfreq("D")
        school_holidays = school_holidays.fillna(0)

        return school_holidays[["nb_school_areas_off"]]

    def get_extra_long_weekend(self) -> pd.DataFrame:
        """
        Return a dataframe from 2000-01-01 to 2050-12-25 indicating for each day
        - if the previous (resp. the next day) is a public holiday
        AND
        - if the current day is a friday (resp. a monday)
        If both conditions are fulfilled then the day is denoted by a 1 (0 otherwise)
        :return:
            A DataFrame with a daily DatetimeIndex and an int 'extra_long_weekend' column containing 1 if the day meets
            the criteria described above and 0 otherwise
        """

        public_holidays = self.get_public_holidays()
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
    def interpolate_daily_to_subdaily_data(
        df: pd.DataFrame, freq: str, method: str = "ffill", tz: str = "Europe/Paris"
    ) -> pd.DataFrame:
        """
        Interpolate daily data in a dataframe (with a DatetimeIndex) to subdaily data using a given method.
        :param df: pd.DataFrame
        :param freq: a frequency < 'D' (e.g. 'H', '30min', '15min', etc)
        :param method: how are data interpolated between two consecutive dates (e.g. 'ffill', 'linear', etc)
        :param tz: timezone ('Europe/Paris')
        :return: pd.DataFrame
        """

        warnings.warn(
            "The Calendar.interpolate_daily_to_sub_daily_data method is deprecated "
            "and will be removed from enda in a future version. "
            "Use TimeSeries.interpolate_daily_to_sub_daily_data instead.",
            FutureWarning,
        )

        return TimeSeries.interpolate_daily_to_sub_daily_data(
            df, freq=freq, method=method, tz=tz
        )

    def get_french_special_days(self, freq: str = "30min") -> pd.DataFrame:
        """
        Return a DataFrame containing all special french days: public and school holidays, lockdowns, and extra long
            weekends
        :param freq: A string indicating the frequency at which to interpolate the DataFrame
        :return:
            A DataFrame with a DatetimeIndex at the specified frequency, and 4 float columns :
            - The 'lockdown' column contains 1 if the timestamp is within a lockdown period and 0 otherwise
            - The 'public_holiday' column contains 1 if the day is a public holiday and 0 otherwise
            - The 'nb_school_areas_off' indicates the number of French zones (3 zones : A, B and C) that are in
            school holidays
            - The 'extra_long_weekend' contains 1 if the day is part of a long weekend (a Monday with a public holiday
            on Tuesday, or a Friday with a public holiday in Thursday) and 0 otherwise
        """

        lockdown = self.get_french_lockdown()
        public_holidays = self.get_public_holidays()
        school_holidays = self.get_school_holidays()
        extra_long_weekend = self.get_extra_long_weekend()

        lockdown_new_freq = TimeSeries.interpolate_daily_to_sub_daily_data(
            lockdown, freq=freq, method="ffill", tz="Europe/Paris"
        )
        public_holidays_new_freq = TimeSeries.interpolate_daily_to_sub_daily_data(
            public_holidays, freq=freq, method="ffill", tz="Europe/Paris"
        )
        school_holidays_new_freq = TimeSeries.interpolate_daily_to_sub_daily_data(
            school_holidays, freq=freq, method="ffill", tz="Europe/Paris"
        )
        extra_long_weekend_new_freq = TimeSeries.interpolate_daily_to_sub_daily_data(
            extra_long_weekend, freq=freq, method="ffill", tz="Europe/Paris"
        )

        result = pd.concat(
            [
                lockdown_new_freq,
                public_holidays_new_freq,
                school_holidays_new_freq,
                extra_long_weekend_new_freq,
            ],
            axis=1,
            join="outer",
        )

        return result
