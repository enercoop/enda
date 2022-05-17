import pandas as pd
from pandas.api.types import is_string_dtype
from enda.timeseries import TimeSeries
from enda.contracts import Contracts


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
    def check_stations(cls, df, station_col, date_start_col, date_end_exclusive_col, is_naive=True):
        '''
        Check there is no NaN for date_start and power station id. 
        Check the same timestamp is not used twice for the same power plant. 
        Check the date starts and date end
        '''
        
        if station_col not in df.columns:
            raise ValueError(f"Required column not found : {station_col}")
        if df[station_col].isna().any():
            rows_with_nan_time = df[df[station_col].isna()]
            raise ValueError(f"There are NaN values for {station_col} in these rows:\n"
                             f"{rows_with_nan_time}")
        
        rows_with_duplicates = df.duplicated([station_col, date_start_col])   
        if rows_with_duplicates.any():
            raise ValueError(f"Duplicated station_col date_col for these rows:\n"
                             f"{df[rows_with_duplicates]}")
        
        # Check date start and end using Contracts
        Contracts.check_contracts_dates(df, date_start_col, date_end_exclusive_col, is_naive)

    # ------ Build daily dataframes

    @staticmethod
    def __station_to_events(stations, date_start_col, date_end_exclusive_col):
        '''
        This function is basically the same as Contracts.__contract_to_event()
        '''

        # check that no column is named "event_type" or "event_date"
        for c in ["event_type", "event_date"]:
            if c in stations.columns:
                raise ValueError("stations has a column named {}, but this name is reserved in this"
                                 "function; rename your column.".format(c))

        columns_to_keep = [c for c in stations.columns if c not in [date_start_col, date_end_exclusive_col]]
        events_columns = ["event_type", "event_date"] + columns_to_keep

        # compute "station start" and "station end" events
        start_station_events = stations.copy(deep=True)  # all stations must have a start date
        start_station_events["event_type"] = "start"
        start_station_events["event_date"] = start_station_events[date_start_col]
        start_station_events = start_station_events[events_columns]

        # for "station end" events, only keep stations with an end date (NaT = station is not over)
        end_station_events = stations[stations[date_end_exclusive_col].notna()].copy(deep=True)
        end_station_events["event_type"] = "end"
        end_station_events["event_date"] = end_station_events[date_end_exclusive_col]
        end_station_events = end_station_events[events_columns]

        # concat all events together and sort them chronologically
        all_events = pd.concat([start_station_events, end_station_events])
        all_events.sort_values(by=["event_date", "event_type"], inplace=True)

        return all_events

    @classmethod
    def get_stations_daily(
            cls, 
            stations, 
            station_col="station",
            date_start_col="date_start",
            date_end_exclusive_col="date_end_exclusive",
            max_date_exclusive=None
    ): 
        '''
        This function creates a daily dataframe from a power station contracts dataframe.
        It checks the provided dataframe is consistent (dates)
        '''

        for c in ["date", "event_date"]:
            if c in stations.columns:
                raise ValueError("stations has a column named {}, but this name is reserved in this"
                                 "function; rename your column.".format(c))

        cls.check_stations(stations, station_col, date_start_col, date_end_exclusive_col)
        
        # get an event-like dataframe, using the end date of 
        events = cls.__station_to_events(stations, date_start_col, date_end_exclusive_col)

        # remove events after max_date if they are not wanted
        if max_date_exclusive is not None:
            events = events[events["event_date"] <= max_date_exclusive]

        # other_columns = set(stations.columns) - set(not_summable_columns) - set([station_col, date_start_col, date_end_exclusive_col])
        other_columns = set(stations.columns) - set([station_col, date_start_col, date_end_exclusive_col])
        for c in other_columns:
            events[c] = events.apply(lambda row: row[c] if row["event_type"] == "start" else 0, axis=1)
        
        events = events.groupby([station_col, "event_date"]).last().reset_index()
        events = events.drop(columns=["event_type"])

        # iterate over a partition of the dataframe (each station) to resample 
        # at a daily frequency, using a backfill of the dates. 
        df = pd.DataFrame()
        for station, station_contracts in events.groupby(station_col):
            station_contracts = station_contracts.set_index("event_date").asfreq('D', method='ffill')
            station_contracts = station_contracts.iloc[:-1]
            df = pd.concat([df, station_contracts], axis=0)
        
        df.index.name = "date"
        return df.reset_index().set_index([station_col, "date"])

    @staticmethod
    def get_stations_between_dates(
            stations,
            start_datetime, 
            end_datetime_exclusive, 
            freq=None
            ):
        """
        Adds or removes dates if needed.

        If additional dates needed at the end, copy the data of the last date into the additional dates

        :param stations: the dataframe with the stations. It must be a MultiIndex datframe,
                         whose second order index is a pandas.DatetimeIndex with frequency.
        :param start_datetime: the start date column, it is the same for all stations
        :param end_datetime_exclusive: the end date (exclusive)
        :return: a station portfolio with characteristics between date_start and date_end. 
        """

        df = stations.copy(deep=True)

        if not isinstance(df.index, pd.MultiIndex):
            raise TypeError("daily_stations must be a MultiIndex")

        if len(df.index.levels) != 2:
            raise TypeError("daily_stations must be a MultiIndex with two levels (stations and date)")

        if not isinstance(df.index.levels[1], pd.DatetimeIndex):
            raise TypeError("The second index of daily_stations should be a pd.DatetimeIndex, but given {}"
                            .format(df.index.levels[1].dtype))
      
        if freq is None:
            try:
                freq = df.index.levels[1].inferred_freq
            except Exception:
                raise ValueError("No freq has been provided, and it could not be inferred"
                                 "from the index itself. Please set it or check the data.")

        # check that there is no missing value
        if not df.isnull().sum().sum() == 0:
            raise ValueError("daily_stations has NaN values.")

        key_col = df.index.levels[0].name
        date_col = df.index.levels[1].name 

        df_new = pd.DataFrame()
        for station, data in df.groupby(level=0):
            if start_datetime is not None and data.index.levels[1].min() > start_datetime:
                # add days with empty portfolio at the beginning
                data.loc[(station, start_datetime), :] = tuple([0 for x in range(len(df.columns))])
                data.sort_index(inplace=True)  # put the new row first
                data = data.reset_index().set_index(date_col).asfreq(freq, method='ffill')
                data = data.reset_index().set_index([key_col, date_col])
        
            if end_datetime_exclusive is not None and data.index.levels[1].max() < end_datetime_exclusive:
                # add days at the end, with the same portfolio as the last available day
                data.loc[(station, end_datetime_exclusive), :] = tuple([0 for x in range(len(df.columns))])
                data.sort_index(inplace=True) 
                data = data.reset_index().set_index(date_col).asfreq(freq, method='ffill')
                data = data.reset_index().set_index([key_col, date_col])
            
            # remove dates outside of desired range
            data = data[(data.index.get_level_values(date_col) >= start_datetime) & 
                        (data.index.get_level_values(date_col) < end_datetime_exclusive)]
            assert data.isnull().sum().sum() == 0  # check that there is no missing value
            
            df_new = pd.concat([df_new, data], axis=0)

        return df_new

    # ------ Outages

    @classmethod
    def read_outages_from_file(
            cls,
            file_path,
            station_col="station",
            time_start_col="time_start",
            time_end_exclusive_col="time_end_exclusive",
            tzinfo="Europe/Paris",          
            pct_outages_col=None
            ):
        """
        Reads outages from a file. This will convert start and end date columns into dtype=datetime64 (tz-naive)
        :param file_path: where the source file is located.
        :param time_start_col: the name of the outage time start column.
        :param time_end_exclusive_col: the name of the outage time end column, end date is exclusive.
        :param time_format: the time format for pandas.to_datetime
        :param pct_outages_col: the percentage of availability of the powerplant. 
               If none is given, it is assumled the power plant is simply shutdown. 
        :return: a pandas.DataFrame with an outage on each row.
        """

        df = pd.read_csv(file_path)
        for c in [time_start_col, time_end_exclusive_col]:
            if is_string_dtype(df[c]):
                df[c] = pd.to_datetime(df[c])
                df[c] = TimeSeries.align_timezone(df[c], tzinfo=tzinfo)

        # check stations
        cls.check_stations(df, station_col, time_start_col, time_end_exclusive_col, is_naive=False)
        
        # check pct_outage_col
        if pct_outages_col is not None:
            if pct_outages_col not in df.columns:
                raise ValueError(f"Provided column {pct_outages_col} is not present in dataframe")
            if not (df[pct_outages_col].dropna().between(0, 100)).all(): 
                raise ValueError(f"Some values in {pct_outages_col} are not percentage")    
        
        return df

    @staticmethod
    def integrate_outages(
                stations, 
                outages,
                station_col, 
                time_start_col, 
                time_end_exclusive_col, 
                installed_capacity_col,
                pct_outages_col=None):
        '''
        This function integrates shutdown and outages in the stations dataframe.
        The column that describes the stations must be present in both dataframes.
        stations must be a multiindexed time-series. 
        events must have time_start/time_end fields (the length of the shutdown).

        If a feature column is present in both dataframes, we integrate the event change into 
        the 

        :param stations: the dataframe which contains the stations definition over a period of time 
        :param events: the events dataframe
        '''
        # check stations
        if not isinstance(stations.index, pd.MultiIndex):
            raise TypeError("stations must be multiindexed dataframe")
    
        if len(stations.index.levels) != 2:
            raise TypeError("The provided multi-indexed dataframe must be a two-levels one")
    
        if not isinstance(stations.index.levels[1], pd.DatetimeIndex):
            raise TypeError("The second index of the dataframe should be a pd.DatetimeIndex, "
                            f"but {stations.index.levels[1].dtype} is found")
        
        if installed_capacity_col not in stations.columns:
            raise ValueError(f"Installed capacity {installed_capacity_col} is not present in stations")
            
        # check outages
        if station_col not in outages.columns:
            raise ValueError(f"Station column {station_col} is not present in outages")
            
        if time_start_col not in outages.columns:
            raise ValueError(f"Time start column {time_start_col} is not present in outages")
            
        if time_end_exclusive_col not in outages.columns:
            raise ValueError(f"Time end column {time_end_exclusive_col} is not present in outages")
            
        if (pct_outages_col is not None) and (pct_outages_col not in outages.columns):
            raise ValueError(f"Pct outage {pct_outages_col} is not present in outages")
        
        # process
        key_col_stations = stations.index.levels[0].name
        time_col_stations = stations.index.levels[1].name
        
        df_stations = stations.reset_index()

        # loop over outages
        for index, outage in outages.iterrows():
            mask = (df_stations[key_col_stations] == outage[station_col])\
                   & (df_stations[time_col_stations] >= outage[time_start_col])\
                   & (df_stations[time_col_stations] < outage[time_end_exclusive_col])
            
            outage_pct = 100 if outage[pct_outages_col] is None else outage[pct_outages_col]
            df_stations.loc[mask, installed_capacity_col] = (100 - outage_pct) *\
                df_stations.loc[mask, installed_capacity_col] 
    
        return df_stations.set_index([key_col_stations, time_col_stations])

    # ------ Load factor

    @staticmethod
    def compute_load_factor(
            df,
            installed_capacity_kw, 
            power_kw, 
            load_factor_col="load_factor",
            drop_power_kw=True):
        '''
        This function computes the load_factor, which is the target column of most 
        methods implemented for the power stations production prediction
        :param installed_capacity_kw: the column of df that contains the installed_capacity in kw
        :param power_kw: the column which contains the power (in kW)
        :param load_factor_col: name of the computed load factor column
        :param drop_power_kw: boolean flag which indicates whether the power  
                              column shall be dropped. 
        '''

        df = df.copy()
        for c in [installed_capacity_kw, power_kw]:
            if c not in df.columns:
                raise ValueError("Required column not found : {}".format(c))
        
        def load_factor_manage_no_capacity(row):
            if abs(row[installed_capacity_kw]) < 1e-3:
                return 0 
            else:
                return row[power_kw] / row[installed_capacity_kw]

        df[load_factor_col] = df.apply(load_factor_manage_no_capacity, axis=1)

        if drop_power_kw:
            df = df.drop(columns=power_kw)

        return df 

    @staticmethod
    def compute_power_kw_from_load_factor(
            df,
            installed_capacity_kw, 
            load_factor, 
            power_kw_col="power_kw",
            drop_load_factor=True):
        '''
        This function computes the power (in kW) from the computed load_factor. 
        It is the inverse transform of compute_load_factor()
        :param installed_capacity_kw: the column of df that contains the installed_capacity in kw
        :param load_factor: the column which contains the load factor
        :param power_kw_col: name of the computed power column 
        :param drop_load_factor: boolean flag which indicates whether the load factor 
                                 column shall be dropped. 
        '''

        df = df.copy()
        for c in [installed_capacity_kw, load_factor]:
            if c not in df.columns:
                raise ValueError(f"Required column not found : {c}")
        
        df[power_kw_col] = df[installed_capacity_kw] * df[load_factor]

        if drop_load_factor:
            df = df.drop(columns=load_factor)

        return df 
