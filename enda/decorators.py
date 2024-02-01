"""This script contains a decorator used to compute functions on multi-indexed DataFrames"""

import functools
from typing import Callable

import pandas as pd


def handle_multiindex(func: Callable) -> Callable:
    """
    This function is a wrapper around functions defined for a single dataframe with a datetime index so that
    they also work for multi-indexed dataframes. More specifically, functions designed for a
    dataframe with a DatetimeIndex also work for a two-levels dataframe defined with a first
    index that defines a group, and a second index which is a DatetimeIndex. Both index levels must have a name.
    This function is meant to be used as a decorator.
    :param func: the function to decorate
    """

    @functools.wraps(func)
    def wrapper_handle_multiindex(*args, **kwargs) -> pd.DataFrame:

        if "df" in kwargs:
            df = kwargs["df"]
        else:
            df = args[0]

        # if it is a single indexed dataframe, call the function directly
        if not isinstance(df.index, pd.MultiIndex):
            return func(*args, **kwargs)

        # the multiindex must be a two-level
        if len(df.index.levels) != 2:
            raise TypeError(
                "The provided multi-indexed dataframe must be a two-levels one, the "
                "second one being the date index."
            )

        if not isinstance(df.index.levels[1], pd.DatetimeIndex):
            raise TypeError(
                f"The second index of the dataframe should be a pd.DatetimeIndex, but given {df.index.levels[1].dtype}"
            )

        # and for now, we cannot accept no-key arguments except df for a multiindex
        if ("df" in kwargs and len(args) > 0) or (
            "df" not in kwargs and len(args) != 1
        ):
            raise NotImplementedError(
                "The function with multi-index dataframes as input only works "
                "using keyword-only arguments (except for 'df' argument)"
            )

        key_col = df.index.levels[0].name
        date_col = df.index.levels[1].name

        if not key_col or not date_col:
            raise ValueError(
                "Both index levels of the input DataFrame must be named"
            )

        df_new = pd.DataFrame()
        for key, data in df.groupby(level=0, sort=False):
            data = data.reset_index().set_index(date_col).drop(columns=[key_col])
            args_decorator = (data,)
            kwargs_decorator = {x: y for x, y in kwargs.items() if x != "df"}
            data = func(*args_decorator, **kwargs_decorator)
            if not isinstance(data, pd.DataFrame):
                raise TypeError(
                    "@handle_multiindex decorator cannot be used with "
                    "a function which does not return a dataframe"
                )
            new_date_col = data.index.name
            data[key_col] = key
            data = data.reset_index().set_index([key_col, new_date_col])
            df_new = pd.concat([df_new, data], axis=0)

        return df_new

    return wrapper_handle_multiindex
