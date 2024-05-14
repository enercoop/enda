"""This module contains functions to handle power stations data"""

import numpy as np
import pandas as pd

from enda.contracts import Contracts
from enda.tools.portfolio_tools import PortfolioTools
from enda.tools.timezone_utils import TimezoneUtils
import enda.tools.decorators


class PowerStations:
    """
    A class to help handle power_stations data.

    Columns [time, id] are required (names can differ), and with dtype=datetime64 (tz-naive).
    Other columns describe power_station characteristics (used as features).

    All functions are meant to handle a set of station datapoints, that are considered as
    records (samples) for the power plant algorithms.
    """

    # ------ Check

    @classmethod
    def check_stations(
        cls,
        df: pd.DataFrame,
        station_col: str,
        date_start_col: str,
        date_end_exclusive_col: str,
        is_naive: bool = True,
    ):
        """
        - Checks that station_col is in the DataFrame and has no NaN
        - Checks that there are no duplicate timestamps for a power plant.
        - Checks that date_start is not null and that it is before date_end when present
        :param df: The DataFrame to check
        :param station_col: The name of the plant identifier column
        :param date_start_col: The name of the column containing contracts start dates
        :param date_end_exclusive_col: The name of the column containing contracts exclusive end dates
        :param is_naive: Whether date_start_col and date_end_exclusive_col are supposed to contain naïve Timestamps
            (raises an error if it doesn't match)
        """

        if station_col not in df.columns:
            raise ValueError(f"Required column not found : {station_col}")
        if df[station_col].isna().any():
            rows_with_nan_time = df[df[station_col].isna()]
            raise ValueError(
                f"There are NaN values for {station_col} in these rows:\n"
                f"{rows_with_nan_time}"
            )

        rows_with_duplicates = df.duplicated([station_col, date_start_col])
        if rows_with_duplicates.any():
            raise ValueError(
                f"Duplicated station_col date_col for these rows:\n"
                f"{df[rows_with_duplicates]}"
            )

        # Check date start and end using Contracts
        Contracts.check_contracts_dates(
            df, date_start_col, date_end_exclusive_col, is_naive
        )

    # ------ Build daily dataframes

    @classmethod
    def get_stations_daily(
        cls,
        stations: pd.DataFrame,
        station_col: str = "station",
        date_start_col: str = "date_start",
        date_end_exclusive_col: str = "date_end_exclusive",
        max_date_exclusive: pd.Timestamp = None,
        drop_gaps=False,
    ) -> pd.DataFrame:
        """
        This function creates a daily dataframe from a power station contracts dataframe.
        It checks the provided dataframe is consistent (dates)
        :param stations: The DataFrame containing station information
        :param station_col: The column containing station name
        :param date_start_col: The column containing start date information
        :param date_end_exclusive_col: The column containing exclusive end date information
        :param max_date_exclusive: A Timestamp indicating the maximum date until which to keep data. This is
            mainly useful if you have contracts with no specified end date, as they will not be taken into account by
            default.
        :param drop_gaps: if True, will drop daily values that are gaps within the range of input contracts.
            By default, these days are kept in the index with 0 as values
            (For example, if we have a station with a first contract from 2023-01-01 to 2023-02-01 and a second
            contract from 2023-03-01 to 2023-06-01, the period from 2023-02-01 to 2023-03-01 will be kept by default
            with 0s for all values)
        :return: A DataFrame with daily station information
        """

        for c in ["date", "event_date"]:
            if c in stations.columns:
                raise ValueError(
                    f"stations has a column named {c}, but this name is reserved in this"
                    "function; rename your column."
                )

        cls.check_stations(
            stations, station_col, date_start_col, date_end_exclusive_col
        )

        # get an event-like dataframe
        events = PortfolioTools.portfolio_to_events(
            stations, date_start_col, date_end_exclusive_col
        )

        # remove events after max_date if they are not wanted
        if max_date_exclusive is not None:
            events = events[events["event_date"] <= max_date_exclusive]

            # For stations where the last event before max_date_exclusive is a start, it means that a contract is still
            # active at the time of max_date_exclusive. We add a 'fake' end for these contracts so that daily data is
            # computed until max_date_exclusive (otherwise there would be no row to ffill to)
            last_event_df = events.copy()
            last_event_df = last_event_df.groupby(station_col).last().reset_index()
            last_event_df = last_event_df.loc[last_event_df.event_type == "start"]
            last_event_df["event_type"] = 'end'
            last_event_df["event_date"] = max_date_exclusive

            events = pd.concat([events, last_event_df])

        other_columns = set(stations.columns) - {
            station_col,
            date_start_col,
            date_end_exclusive_col,
        }
        for c in other_columns:
            events[c] = events.apply(
                lambda row: row[c] if row["event_type"] == "start" else 0, axis=1
            )

        if drop_gaps:
            if "control_column" in events.columns:
                raise ValueError(
                    "control_column should not be one of the columns in the events DataFrame, it is used"
                    "in this function"
                )

            # We create a control column which is used to keep only days included in the events DataFrame period
            events["control_column"] = "0"
            events.loc[events.event_type == "start", "control_column"] = "control"

        events = events.groupby([station_col, "event_date"]).last().reset_index()
        events = events.drop(columns=["event_type"])

        # iterate over a partition of the dataframe (each station) to resample
        # at a daily frequency, using a backfill of the dates.
        df = pd.DataFrame()
        for _, station_contracts in events.groupby(station_col):
            station_contracts = station_contracts.set_index("event_date").asfreq(
                "D", method="ffill"
            )
            station_contracts = station_contracts.iloc[:-1]
            df = pd.concat([df, station_contracts], axis=0)

        df.index.name = "date"

        if drop_gaps:
            # The way we compute a daily DataFrame, there is a problem if there is a gap between an end event and the
            # next start event : for example, if a contract line ends on 2022 December 31st and next one begins on
            # 2024 January 1st, the days in 2023 will still be present in df_daily_pf.
            # However, the columns other than plant identification column will have '0' as a value,
            # so using this control column we can keep only relevant dates.
            df = df.loc[df.control_column == "control"]
            df.drop("control_column", axis=1, inplace=True)

        return df.reset_index().set_index([station_col, "date"])

    @staticmethod
    @enda.tools.decorators.warning_deprecated_name(
        namespace_name="PowerStations",
        new_namespace_name="PortfolioTools",
        new_function_name="get_portfolio_between_dates",
    )
    def get_stations_between_dates(
        stations: pd.DataFrame,
        start_datetime: pd.Timestamp,
        end_datetime_exclusive: pd.Timestamp,
        freq: str = None,
    ) -> pd.DataFrame:
        """
        Adds or removes dates if needed.

        If additional dates needed at the end, copy the data of the last date into the additional dates

        :param stations: the dataframe with the stations. It must be a MultiIndex DataFrame,
                         whose second order index is a pandas.DatetimeIndex with frequency.
        :param start_datetime: the start date column, it is the same for all stations
        :param end_datetime_exclusive: the end date (exclusive)
        :param freq: The frequency of the DataFrame (ex: '30min'). If not specified, the function will try to infer it
        :return: a station portfolio with characteristics between date_start and date_end.
        """

        return PortfolioTools.get_portfolio_between_dates(
            portfolio_df=stations,
            start_datetime=start_datetime,
            end_datetime_exclusive=end_datetime_exclusive,
            freq=freq,
        )

    # ------ Outages

    @staticmethod
    def get_outages_from_file(
        file_path: str,
        time_start_col: str = "time_start",
        time_end_exclusive_col: str = "time_end_exclusive",
        tzinfo: str = "Europe/Paris",
        pct_outages_col: str = None,
    ):
        """
        Reads outages from a file. This will convert start and end date columns into dtype=datetime64 (tz-naive) and
        check that pct_outages_col is present in the DataFrame if specified, with values between 0 and 100 when not None
        :param file_path: where the source file is located.
        :param time_start_col: the name of the outage time start column.
        :param time_end_exclusive_col: the name of the outage time end column, end date is exclusive.
        :param tzinfo: The time zone of the data we read
        :param pct_outages_col: The percentage of unavailability of the power plant (100 means complete shutdown).
        :return: a pandas.DataFrame with an outage on each row.
        """

        outages_df = TimezoneUtils.read_csv_and_set_tz_aware_columns(
            file_path=file_path,
            time_cols_list=[time_start_col, time_end_exclusive_col],
            tz_info=tzinfo
        )

        # check pct_outage_col
        if pct_outages_col is not None:
            if pct_outages_col not in outages_df.columns:
                raise ValueError(
                    f"Provided column {pct_outages_col} is not present in dataframe"
                )
            if not (outages_df[pct_outages_col].dropna().between(0, 100)).all():
                raise ValueError(
                    f"Some values in {pct_outages_col} are not percentages between 0 and 100"
                )

        return outages_df

    @classmethod
    def read_outages_from_file(
        cls,
        file_path: str,
        station_col: str = "station",
        time_start_col: str = "time_start",
        time_end_exclusive_col: str = "time_end_exclusive",
        tzinfo: str = "Europe/Paris",
        pct_outages_col: str = None,
    ) -> pd.DataFrame:
        """
        Reads outages from a file and checks that the resulting DataFrame is coherent. This will convert start and end
        date columns into dtype=datetime64 (tz-naive)
        :param file_path: where the source file is located.
        :param station_col: the name of the column containing stations names
        :param time_start_col: the name of the outage time start column.
        :param time_end_exclusive_col: the name of the outage time end column, end date is exclusive.
        :param tzinfo: The time zone of the data we read
        :param pct_outages_col: The percentage of unavailability of the power plant (100 means complete shutdown).
        :return: a pandas.DataFrame with an outage on each row.
        """

        # Read the data and convert time columns to the correct type
        df = cls.get_outages_from_file(
            file_path, time_start_col, time_end_exclusive_col, tzinfo, pct_outages_col
        )

        # Check that the resulting DataFrame is coherent
        cls.check_stations(
            df, station_col, time_start_col, time_end_exclusive_col, is_naive=False
        )

        return df

    @staticmethod
    def integrate_availability_from_outages(
        df_stations: pd.DataFrame,
        df_outages: pd.DataFrame,
        station_col: str,
        time_start_col: str,
        time_end_exclusive_col: str,
        pct_outages_col: str = None,
        availability_col: str = None,
    ) -> pd.DataFrame:
        """
        This function starts from a multi-indexed dataframe with stations-timeseries.
        It takes another non-indexed dataframe containing the detail of the shutdowns
        and outages for the stations.
        It integrates to the station-timeseries data an extra column that describe the
        availability of the stations.

        Conditions:
        - The stations ID column must be present in both dataframes.
        - df_stations must have a frequency. This condition is not strictly necessary
          to integrate outages a priori, but in our applications, it's almost always the case.
        - df_outages must have time_start/time_end fields (the length of the shutdown).

        :param df_stations: The DataFrame which contains the stations features as timeseries
        :param df_outages: The non-indexed outages DataFrame
        :param station_col: The column containing the station identifier
        :param time_start_col: The column containing the start time of the outage
        :param time_end_exclusive_col: The column containing the exclusive end time of the outage
        :param pct_outages_col: The column containing the information of outage impact on capacity.
            If a null value is given, it is assumed the power plant is simply shutdown.
        :param availability_col: The name of the availability column in the new DataFrame
        """

        # check df_stations
        if not isinstance(df_stations.index, pd.MultiIndex):
            raise TypeError("stations must be multi-indexed dataframe")

        if len(df_stations.index.levels) != 2:
            raise TypeError(
                "The provided multi-indexed dataframe must be a two-levels one"
            )

        if not isinstance(df_stations.index.levels[1], pd.DatetimeIndex):
            raise TypeError(
                "The second index of the dataframe should be a pd.DatetimeIndex, "
                f"but {df_stations.index.levels[1].dtype} is found"
            )

        # check df_outages
        if station_col not in df_outages.columns:
            raise ValueError(f"Station column {station_col} is not present in outages")

        if time_start_col not in df_outages.columns:
            raise ValueError(
                f"Time start column {time_start_col} is not present in outages"
            )

        if time_end_exclusive_col not in df_outages.columns:
            raise ValueError(
                f"Time end column {time_end_exclusive_col} is not present in outages"
            )

        # reset the availability column
        if availability_col is None:
            if "availability" in df_stations.columns:
                raise ValueError(
                    "'availability' is a reserved keyword for this function"
                )
            availability_col = "availability"

        df_stations = df_stations.copy()
        df_stations[availability_col] = 1

        # loop over outages to set the availability
        for _, outage in df_outages.iterrows():
            mask = (
                (df_stations.index.get_level_values(0) == outage[station_col])
                & (df_stations.index.get_level_values(1) >= outage[time_start_col])
                & (
                    df_stations.index.get_level_values(1)
                    < outage[time_end_exclusive_col]
                )
            )

            availability = (
                0
                if pd.isna(outage[pct_outages_col])
                else 1 - (outage[pct_outages_col] / 100.0)
            )
            if abs(availability) < 1e-6:
                availability = 0
            df_stations.loc[mask, availability_col] = availability

        return df_stations

    @staticmethod
    def reset_installed_capacity(
        df: pd.DataFrame,
        installed_capacity_kw: str,
        stations_availability: str,
        drop_availability: bool = True,
    ) -> pd.DataFrame:
        """
        This function is meant to reset the installed capacity of a station using
        a helper column. The helper column stores number between 0 and 1 detailing
        the availability of the station.
        :param df: The dataframe to be changed.
        :param installed_capacity_kw: The column of df that contains the installed_capacity in kW
        :param stations_availability: The name of the helper column used to compute installed capacity.
            Values should be between 0 and 1, 0 meaning a shutdown and 1 meaning full installed capacity available
        :param drop_availability: boolean flag which indicates whether the availability
                                   column shall be dropped.
        :return: The DataFrame with corrected installed capacity
        """

        for c in [installed_capacity_kw, stations_availability]:
            if c not in df.columns:
                raise ValueError(f"Required column not found: {c}")

        if df[stations_availability].isna().any():
            raise ValueError(f"{stations_availability} column should not contain NaN")

        max_value = df[stations_availability].max()
        if df[stations_availability].max() > 1:
            raise ValueError(
                f"{stations_availability} column should contain values"
                f" between 0 and 1, found: {max_value}"
            )

        min_value = df[stations_availability].min()
        if df[stations_availability].min() < 0:
            raise ValueError(
                f"{stations_availability} column should contain values"
                f" between 0 and 1, found: {min_value}"
            )

        df = df.copy()
        df[installed_capacity_kw] = (
            df[installed_capacity_kw] * df[stations_availability]
        )

        if drop_availability:
            df = df.drop(columns=stations_availability)

        return df

    @staticmethod
    def integrate_outages(
        df_stations: pd.DataFrame,
        df_outages: pd.DataFrame,
        station_col: str,
        time_start_col: str,
        time_end_exclusive_col: str,
        installed_capacity_col: str,
        pct_outages_col: str = None,
    ) -> pd.DataFrame:
        """
        This function makes successive calls to integrate_availability_from_outages()
        and to reset_installed_capacity().

        :param df_stations: The Dataframe which contains the stations features as timeseries
        :param df_outages: The DataFrame containing outages information
        :param station_col: The column containing the station identifier
        :param time_start_col: The column containing the start time of the outage
        :param time_end_exclusive_col: The column containing the exclusive end time of the outage
        :param installed_capacity_col: The column containing the installed capacity in kW
        :param pct_outages_col: The column containing the information of outage impact on capacity
        :return: The DataFrame with corrected installed capacity
        """

        # check station availability_col
        if "availability_col" in df_stations.columns:
            raise ValueError(
                "'availability_col' is a reserved keyword for this function"
            )

        df = PowerStations.integrate_availability_from_outages(
            df_stations=df_stations,
            df_outages=df_outages,
            station_col=station_col,
            time_start_col=time_start_col,
            time_end_exclusive_col=time_end_exclusive_col,
            pct_outages_col=pct_outages_col,
            availability_col="availability_col",
        )

        df = PowerStations.reset_installed_capacity(
            df=df,
            installed_capacity_kw=installed_capacity_col,
            stations_availability="availability_col",
            drop_availability=True,
        )

        return df

    # ------ Load factor

    @staticmethod
    def compute_load_factor(
        df: pd.DataFrame,
        installed_capacity_kw: str,
        power_kw: str,
        load_factor_col: str = "load_factor",
        drop_power_kw: bool = True,
    ) -> pd.DataFrame:
        """
        This function computes the load_factor, which is the target column of most
        methods implemented for the power stations production prediction
        :param df: The input DataFrame for computing load factor
        :param installed_capacity_kw: The column that contains the installed_capacity in kW
        :param power_kw: The column that contains the power (in kW)
        :param load_factor_col: The name of the computed load factor column
        :param drop_power_kw: A boolean flag which indicates whether the power
                              column shall be dropped.
        """
        df = df.copy()

        for c in [installed_capacity_kw, power_kw]:
            if c not in df.columns:
                raise ValueError(f"Required column not found : {c}")

        df[load_factor_col] = np.where(
            df[installed_capacity_kw] < 1e-5,
            0,
            df[power_kw] / df[installed_capacity_kw],
        )

        if drop_power_kw:
            df = df.drop(columns=power_kw)

        return df

    @staticmethod
    def compute_power_kw_from_load_factor(
        df: pd.DataFrame,
        installed_capacity_kw: str,
        load_factor: str,
        power_kw_col: str = "power_kw",
        drop_load_factor: bool = True,
    ) -> pd.DataFrame:
        """
        This function computes the power (in kW) from the computed load_factor.
        It is the inverse transform of compute_load_factor()
        :param df: The input DataFrame for computing power
        :param installed_capacity_kw: The column that contains the installed_capacity in kW
        :param load_factor: The column which contains the load factor
        :param power_kw_col: The name of the computed power column
        :param drop_load_factor: A boolean flag which indicates whether the load factor
                                 column shall be dropped.
        """

        df = df.copy()
        for c in [installed_capacity_kw, load_factor]:
            if c not in df.columns:
                raise ValueError(f"Required column not found : {c}")

        df[power_kw_col] = df[installed_capacity_kw] * df[load_factor]

        if drop_load_factor:
            df = df.drop(columns=load_factor)

        return df

    @staticmethod
    def clip_column(
            df: pd.DataFrame,
            column_name: str,
            lower_bound: float = None,
            upper_bound: float = None
    ) -> pd.DataFrame:
        """Checks that values in the indicated colname are between the two specified bounds (bounds included).
        If not, replaces the excessive values with the associated bounds
        :param df: The DataFrame to use the function on
        :param column_name: The column to check and clip
        :param lower_bound: If not None, the lower bound that we can't go below of.
        :param upper_bound: If not None, the upper bound that we can't go above of. Defaults to None.
        :return: A dataframe where all the values in the specified columns are between the two bounds. If we find
            excessive values, we set them equal to the closer bound"""

        result_df = df.copy()

        if lower_bound is not None:
            result_df.loc[(result_df[column_name] < lower_bound), column_name] = lower_bound

        if upper_bound is not None:
            result_df.loc[(result_df[column_name] > upper_bound), column_name] = upper_bound

        return result_df
