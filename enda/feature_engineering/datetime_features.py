"""This module helps for preprocessing timeseries, by splitting them by attributes such as minutes and hours, and
 also by encoding features based on these attributes"""

import datetime

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime

from enda.tools.decorators import handle_multiindex


class DatetimeFeature:
    """A class containing methods to split datetimes into attributes, and encode these attributes with trigonometric
    functions (mainly to be usable by machine learning algorithms)"""

    @staticmethod
    @handle_multiindex(arg_name="df")
    def split_datetime(
        df: pd.DataFrame,
        split_list: list[str] = None,
        index: bool = True,
        colname: str = None,
    ) -> pd.DataFrame:
        """
        Split a specific datetime column or datetime index into different date and time attributes (given by split list)
        Return the dataframe df with the new columns.
        :param df: pd.DataFrame
        :param split_list: attributes in ['minute', 'minuteofday', 'hour', 'day', 'month', 'year', 'dayofweek',
         'weekofyear', 'dayofyear']
        :param index: Bool (True if working on a DatetimeIndex, False if working on a column)
        :param colname: str (only if index=False)
        :return: pd.DataFrame with new columns (split_list)
        """

        if index is False and colname is None:
            raise ValueError(
                "To split a datetime column, please specify the name of the column (argument colname)\n"
                "To split a datetime index, please specify the argument index=True"
            )

        if index is True and colname is not None:
            raise ValueError(
                "You can split either your index or a given column but not both at the same time"
            )

        split_list_implemented = [
            "minute",
            "minuteofday",
            "hour",
            "day",
            "month",
            "year",
            "dayofweek",
            "weekofyear",
            "dayofyear",
        ]
        if split_list is None:
            split_list = split_list_implemented
        else:
            if not all(split in split_list_implemented for split in split_list):
                extra_split = list(set(split_list) - set(split_list_implemented))
                raise NotImplementedError(
                    f"Split {extra_split} not in {split_list_implemented}"
                )

        df_split = pd.DataFrame(index=df.index)
        if index:
            if not isinstance(df.index, pd.DatetimeIndex):
                raise TypeError(
                    "To split a datetime index, please provide a DatetimeIndex in your dataframe"
                )

            for split in split_list:
                if split == "weekofyear":  # To avoid a FutureWarning
                    df_split[split] = df.index.isocalendar().week
                elif split == "minuteofday":
                    df_split[split] = df.index.hour * 60 + df.index.minute
                else:
                    df_split[split] = getattr(df.index, split)

        else:
            if colname not in df.columns:
                raise AttributeError(
                    f"Specified colname {colname} is not a column of the input DataFrame"
                )
            if not is_datetime(df[colname]):
                raise TypeError(
                    f"{colname} is not a datetime column : {df[[colname]].dtypes}"
                )

            for split in split_list:
                if split == "weekofyear":  # To avoid a FutureWarning
                    df_split[split] = df[colname].dt.isocalendar().week
                elif split == "minuteofday":
                    df_split[split] = df[colname].dt.hour * 60 + df[colname].dt.minute
                else:
                    df_split[split] = getattr(df[colname].dt, split)

        result = pd.concat([df, df_split], axis=1, join="inner")

        return result

    @staticmethod
    def get_nb_hours_in_day(d: pd.Timestamp) -> int:
        """
        Return the number of hours in the day containing the specified timestamp
        :param d: A pandas Timestamp object
        :return: An integer
        """
        if d.tzinfo is None:
            raise AttributeError(
                "Please provide a timezone information to your datetime"
            )

        return len(
            pd.date_range(
                d.date(),
                d.date() + datetime.timedelta(days=1),
                tz=d.tzinfo,
                freq="H",
                inclusive="left",
            )
        )

    @staticmethod
    def daylight_saving_time_dates(tz: str = "Europe/Paris") -> pd.DataFrame:
        """
        Return a pd.Dataframe with
        - as index : the dates when daylight saving time starts or ends for the specified timezone
        - as column : the number of hour in this particular day (23 or 25)
        Example :
                                   nb_hour
        1995-03-26 00:00:00+01:00       23
        1995-09-24 00:00:00+02:00       25
        1996-03-31 00:00:00+01:00       23
                                       ...
        2027-10-31 00:00:00+02:00       25
        2028-03-26 00:00:00+01:00       23
        """

        df = pd.DataFrame(
            index=pd.date_range(
                "1995-01-01", "2030-01-01", freq="H", tz=tz, inclusive="left"
            )
        )
        df["nb_hour"] = df.index.hour
        df_by_day = df.resample("D").count()
        df_daylight_saving_time_dates = df_by_day[df_by_day["nb_hour"] != 24]
        return df_daylight_saving_time_dates

    @staticmethod
    def encode_cyclic_datetime(d: pd.Timestamp) -> pd.DataFrame:
        """
        Get the cyclic properties of a datetime, represented as points on the unit circle.
        :param d: A pd.Timestamp object
        :return: A DataFrame containing the sine and cosine for following attributes : minute, minuteofday, hour,
            day, month and dayofweek. Each value is contained in a column (so we have columns such as minute_cos,
            hour_sin ...) and the input DataFrame is the index
        """

        days_in_year = 365 if not d.is_leap_year else 366
        days_in_month = d.days_in_month

        month = d.month - 1
        dayofmonth = d.day - 1
        dayofyear = d.dayofyear - 1

        result = {
            "minute_cos": [np.cos(2 * np.pi * d.minute / 60)],
            "minute_sin": [np.sin(2 * np.pi * d.minute / 60)],
            "minuteofday_cos": [
                np.cos(2 * np.pi * (d.hour * 60 + d.minute) / (24 * 60))
            ],
            "minuteofday_sin": [
                np.sin(2 * np.pi * (d.hour * 60 + d.minute) / (24 * 60))
            ],
            "hour_cos": [np.cos(2 * np.pi * d.hour / 24)],
            "hour_sin": [np.sin(2 * np.pi * d.hour / 24)],
            "day_cos": [np.cos(2 * np.pi * dayofmonth / days_in_month)],
            "day_sin": [np.sin(2 * np.pi * dayofmonth / days_in_month)],
            "month_cos": [np.cos(2 * np.pi * month / 12)],
            "month_sin": [np.sin(2 * np.pi * month / 12)],
            "dayofweek_cos": [np.cos(2 * np.pi * d.dayofweek / 7)],
            "dayofweek_sin": [np.sin(2 * np.pi * d.dayofweek / 7)],
            "dayofyear_cos": [np.cos(2 * np.pi * dayofyear / days_in_year)],
            "dayofyear_sin": [np.sin(2 * np.pi * dayofyear / days_in_year)],
        }

        result = pd.DataFrame.from_dict(data=result)
        result.index = [d]

        return result

    @staticmethod
    @handle_multiindex(arg_name="df")
    def encode_cyclic_datetime_index(df: pd.DataFrame, split_list: list[str] = None):
        """
        Split and encode a datetime index into different date and time attributes (given by split list).
        Encoding method : for each attribute, cosinus and sinus are provided.
        Return the DataFrame df with the new columns.
        :param df: The input DataFrame with a DatetimeIndex
        :param split_list: attributes in ['minute', 'minuteofday', 'hour', 'day', 'month', 'dayofweek', 'dayofyear']
        :return: pd.DataFrame with new columns ['hour_cos', 'hour_sin', 'day_cos', ...]
        """

        max_dict = {
            "minute": 60,
            "hour": 24,
            "minuteofday": 1440,
            "month": 12,
            "dayofweek": 7,
        }
        split_list_implemented = [
            "minute",
            "minuteofday",
            "hour",
            "day",
            "month",
            "dayofweek",
            "dayofyear",
        ]
        if split_list is None:
            split_list = split_list_implemented
        else:
            if not all(split in split_list_implemented for split in split_list):
                extra_split = list(set(split_list) - set(split_list_implemented))
                raise NotImplementedError(
                    f"Split {extra_split} not in {split_list_implemented}"
                )

        df_split = pd.DataFrame(index=df.index)
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(
                "To split a datetime index, please provide a DatetimeIndex in your dataframe"
            )

        for split in split_list:
            if split == "minuteofday":
                df_split[split] = df.index.hour * 60 + df.index.minute
            else:
                df_split[split] = getattr(df.index, split)

            if split in max_dict:
                df_split[f"{split}_max"] = max_dict[split]

            if split == "dayofyear":
                df_split[f"{split}_max"] = df_split.index.is_leap_year
                df_split[f"{split}_max"] = df_split[f"{split}_max"].replace(
                    {True: 366, False: 365}
                )
                df_split[f"{split}_max"] = df_split[f"{split}_max"].astype(int)

            if split == "day":
                df_split[f"{split}_max"] = df_split.index.days_in_month

            if split in ["month", "day", "dayofyear"]:
                df_split[split] = df_split[split] - 1

            df_split[f"{split}_cos"] = np.cos(
                (2 * np.pi * df_split[split]) / df_split[f"{split}_max"]
            )
            df_split[f"{split}_sin"] = np.sin(
                (2 * np.pi * df_split[split]) / df_split[f"{split}_max"]
            )

            df_split = df_split.drop(columns=split)
            df_split = df_split.drop(columns=f"{split}_max")

        result = pd.concat([df, df_split], axis=1, join="inner", verify_integrity=True)

        return result

    @staticmethod
    def get_datetime_features(df: pd.DataFrame, split_list: list[str] = None):
        """
        This function makes successive calls to encode_cyclic_datetime_index() and to split_datetime()
        :param df: The input DataFrame with a DatetimeIndex
        :param split_list: attributes in ['minute', 'minuteofday', 'hour', 'day', 'month', 'dayofweek', 'dayofyear']
        :return: The DataFrame with new columns resulting from encoding/splitting
        """
        result = DatetimeFeature.encode_cyclic_datetime_index(df, split_list)
        result = DatetimeFeature.split_datetime(result, split_list)
        return result
