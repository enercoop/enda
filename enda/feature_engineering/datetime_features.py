import datetime

import numpy as np
import pandas as pd

from enda.decorators import handle_multiindex


class DatetimeFeature:
    @staticmethod
    @handle_multiindex
    def split_datetime(df, split_list=None, index=True, colname=None):
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
                    "Split {} not in {}".format(extra_split, split_list_implemented)
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
                    df_split[split] = df.index.__getattribute__(split)

        else:
            if colname not in list(df.select_dtypes(include=[np.datetime64])):
                raise TypeError(
                    "{} is not a datetime column : {}".format(
                        colname, df[[colname]].dtypes
                    )
                )

            for split in split_list:
                if split == "weekofyear":  # To avoid a FutureWarning
                    df_split[split] = df[colname].dt.isocalendar().week
                else:
                    df_split[split] = df[colname].dt.__getattribute__(split)

        result = pd.concat([df, df_split], axis=1, join="inner")

        return result

    @staticmethod
    def get_nb_hours_in_day(d):
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
                closed="left",
            )
        )

    @staticmethod
    def daylight_saving_time_dates():
        """
        Return a pd.Dataframe with
        - as index : the dates when daylight saving time starts or ends
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
                "1995-01-01", "2030-01-01", freq="H", tz="Europe/Paris", closed="left"
            )
        )
        df["nb_hour"] = df.index.hour
        df_by_day = df.resample("D").count()
        df_daylight_saving_time_dates = df_by_day[df_by_day["nb_hour"] != 24]
        return df_daylight_saving_time_dates

    @staticmethod
    def encode_cyclic_datetime(d):
        """
        Get the cyclic properties of a datetime, represented as points on the unit circle.
        :param d: pandas datetime object
        :return: pd.DataFrame of sine and cosine
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
    @handle_multiindex
    def encode_cyclic_datetime_index(df, split_list=None):
        """
        Split and encode a datetime index into different date and time attributes (given by split list).
        Encoding method : for each attribute, cosinus and sinus are provided.
        Return the dataframe df with the new columns.
        :param df: pd.DataFrame
        :param split_list: attributes in ['hour', 'day', 'month', 'dayofweek', 'dayofyear']
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
                    "Split {} not in {}".format(extra_split, split_list_implemented)
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
                df_split[split] = df.index.__getattribute__(split)

            if split in max_dict.keys():
                df_split["{}_max".format(split)] = max_dict[split]

            if split == "dayofyear":
                df_split["{}_max".format(split)] = df_split.index.is_leap_year
                df_split["{}_max".format(split)] = df_split[
                    "{}_max".format(split)
                ].replace({True: 366, False: 365})
                df_split["{}_max".format(split)] = df_split[
                    "{}_max".format(split)
                ].astype(int)

            if split == "day":
                df_split["{}_max".format(split)] = df_split.index.days_in_month

            if split in ["month", "day", "dayofyear"]:
                df_split[split] = df_split[split] - 1

            df_split["{}_{}".format(split, "cos")] = np.cos(
                (2 * np.pi * df_split[split]) / df_split["{}_max".format(split)]
            )
            df_split["{}_{}".format(split, "sin")] = np.sin(
                (2 * np.pi * df_split[split]) / df_split["{}_max".format(split)]
            )

            df_split = df_split.drop(columns=split)
            df_split = df_split.drop(columns="{}_max".format(split))

        result = pd.concat([df, df_split], axis=1, join="inner", verify_integrity=True)

        return result

    @staticmethod
    def get_datetime_features(df, split_list=None, index=True, colname=None):
        result = DatetimeFeature.encode_cyclic_datetime_index(
            df, split_list, index, colname
        )
        result = DatetimeFeature.split_datetime(result, split_list)
        return result
