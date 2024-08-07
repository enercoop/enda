"""A module for testing the functions in enda/power_stations.py"""

import logging
import os
import pathlib
import unittest

import numpy as np
import pandas as pd

from enda.contracts import Contracts
from enda.power_stations import PowerStations


class TestPowerStations(unittest.TestCase):
    """This class aims at testing the functions of the PowerStations class in enda/power_stations.py"""

    def setUp(self):
        logging.captureWarnings(True)
        logging.disable(logging.ERROR)

        self.outages_path = os.path.join(
            pathlib.Path(__file__).parent.absolute(), "unittests_data/outages"
        )
        self.correct_outages_filepath = os.path.join(self.outages_path, "outages.csv")
        self.correct_outages_timestamp_filepath = os.path.join(self.outages_path, "outages_timestamp.csv")

        self.expected_outages_df = pd.DataFrame(
            data=[
                {
                    "station": "station1",
                    "start_date": pd.Timestamp(
                        year=2023, month=1, day=1, tz="Europe/Paris"
                    ),
                    "excl_end_date": pd.Timestamp(
                        year=2023, month=6, day=1, tz="Europe/Paris"
                    ),
                    "pct_outages": 100,
                },
                {
                    "station": "station1",
                    "start_date": pd.Timestamp(
                        year=2023, month=6, day=1, tz="Europe/Paris"
                    ),
                    "excl_end_date": pd.Timestamp(
                        year=2023, month=6, day=2, tz="Europe/Paris"
                    ),
                    "pct_outages": 30,
                },
                {
                    "station": "station1",
                    "start_date": pd.Timestamp(
                        year=2023, month=6, day=2, tz="Europe/Paris"
                    ),
                    "excl_end_date": pd.Timestamp(
                        year=2023, month=7, day=1, tz="Europe/Paris"
                    ),
                    "pct_outages": 100,
                },
                {
                    "station": "station2",
                    "start_date": pd.Timestamp(
                        year=2023, month=1, day=1, tz="Europe/Paris"
                    ),
                    "excl_end_date": pd.Timestamp(
                        year=2023, month=6, day=1, tz="Europe/Paris"
                    ),
                    "pct_outages": 100,
                },
            ]
        )

        self.expected_outages_timestamp_df = pd.DataFrame(
            data=[
                {
                    "station": "station1",
                    "start_date": pd.Timestamp(
                        year=2023, month=1, day=1, tz="Europe/Paris"
                    ),
                    "excl_end_date": pd.Timestamp(
                        year=2023, month=6, day=1, tz="Europe/Paris"
                    ),
                    "pct_outages": 100,
                },
                {
                    "station": "station1",
                    "start_date": pd.Timestamp(
                        year=2023, month=6, day=1, tz="Europe/Paris"
                    ),
                    "excl_end_date": pd.Timestamp(
                        year=2023, month=6, day=2, tz="Europe/Paris"
                    ),
                    "pct_outages": 30,
                },
                {
                    "station": "station1",
                    "start_date": pd.Timestamp(
                        year=2023, month=6, day=2, tz="Europe/Paris"
                    ),
                    "excl_end_date": pd.Timestamp(
                        year=2023, month=7, day=1, tz="Europe/Paris"
                    ),
                    "pct_outages": 100,
                },
                {
                    "station": "station2",
                    "start_date": pd.Timestamp(
                        year=2023, month=1, day=1, tz="Europe/Paris"
                    ),
                    "excl_end_date": pd.Timestamp(
                        year=2023, month=6, day=1, tz="Europe/Paris"
                    ),
                    "pct_outages": 100,
                },
            ]
        )

        # We create reference stations and outages DataFrame to test the outages integration functions

        dt1 = pd.date_range(
            start=pd.Timestamp(2023, 1, 1),
            end=pd.Timestamp(2024, 1, 1),
            inclusive="left",
            tz="Europe/Paris",
            freq="30min",
        )

        dt2 = pd.date_range(
            start=pd.Timestamp(2023, 1, 1),
            end=pd.Timestamp(2023, 6, 15),
            inclusive="left",
            tz="Europe/Paris",
            freq="30min",
        )
        dt3 = pd.date_range(
            start=pd.Timestamp(2023, 6, 15),
            end=pd.Timestamp(2023, 6, 19),
            inclusive="left",
            tz="Europe/Paris",
            freq="30min",
        )
        dt4 = pd.date_range(
            start=pd.Timestamp(2023, 6, 19),
            end=pd.Timestamp(2024, 1, 1),
            inclusive="left",
            tz="Europe/Paris",
            freq="30min",
        )

        ref_stations_df = pd.DataFrame(
            data=[
                {
                    "station": "station1",
                    "time": x,
                    "power_installed_kw": 100,
                    "type_de_centrale": "PV",
                }
                for x in dt1
            ]
            + [
                {
                    "station": "station2",
                    "time": x,
                    "power_installed_kw": 2000,
                    "type_de_centrale": "EO",
                }
                for x in dt1
            ]
        )

        self.ref_stations_df = ref_stations_df.set_index(["station", "time"])

        # We create outages with a hole (June 17th-18th) for station1 to check that the plant is considered
        # unavailable for these dates
        self.ref_outages_df = pd.DataFrame(
            data=[
                {
                    "station": "station1",
                    "start_date": pd.Timestamp(
                        year=2023, month=6, day=15, tz="Europe/Paris"
                    ),
                    "excl_end_date": pd.Timestamp(
                        year=2023, month=6, day=17, tz="Europe/Paris"
                    ),
                    "pct_outages": 100,
                },
                {
                    "station": "station1",
                    "start_date": pd.Timestamp(
                        year=2023, month=6, day=17, tz="Europe/Paris"
                    ),
                    "excl_end_date": pd.Timestamp(
                        year=2023, month=6, day=19, tz="Europe/Paris"
                    ),
                    "pct_outages": None,
                },
                {
                    "station": "station2",
                    "start_date": pd.Timestamp(
                        year=2023, month=1, day=1, tz="Europe/Paris"
                    ),
                    "excl_end_date": pd.Timestamp(
                        year=2024, month=1, day=1, tz="Europe/Paris"
                    ),
                    "pct_outages": 60,
                },
            ]
        )

        ref_availability_df = pd.DataFrame(
            data=[
                {
                    "station": "station1",
                    "time": x,
                    "power_installed_kw": 100,
                    "type_de_centrale": "PV",
                    "availability": 1.0,
                }
                for x in dt2
            ]
            + [
                {
                    "station": "station1",
                    "time": x,
                    "power_installed_kw": 100,
                    "type_de_centrale": "PV",
                    "availability": 0.0,
                }
                for x in dt3
            ]
            + [
                {
                    "station": "station1",
                    "time": x,
                    "power_installed_kw": 100,
                    "type_de_centrale": "PV",
                    "availability": 1.0,
                }
                for x in dt4
            ]
            + [
                {
                    "station": "station2",
                    "time": x,
                    "power_installed_kw": 2000,
                    "type_de_centrale": "EO",
                    "availability": 0.4,
                }
                for x in dt1
            ]
        )

        self.ref_availability_df = ref_availability_df.set_index(["station", "time"])

        ref_integrated_outages_df = pd.DataFrame(
            data=[
                {
                    "station": "station1",
                    "time": x,
                    "power_installed_kw": 100.0,
                    "type_de_centrale": "PV",
                }
                for x in dt2
            ]
            + [
                {
                    "station": "station1",
                    "time": x,
                    "power_installed_kw": 0,
                    "type_de_centrale": "PV",
                }
                for x in dt3
            ]
            + [
                {
                    "station": "station1",
                    "time": x,
                    "power_installed_kw": 100,
                    "type_de_centrale": "PV",
                }
                for x in dt4
            ]
            + [
                {
                    "station": "station2",
                    "time": x,
                    "power_installed_kw": 800,
                    "type_de_centrale": "EO",
                }
                for x in dt1
            ]
        )

        self.ref_integrated_outages_df = ref_integrated_outages_df.set_index(
            ["station", "time"]
        )

        # DataFrame containing a timeseries of power data

        power_df = pd.DataFrame(
            data=[
                {
                    "station": "station1",
                    "time": pd.Timestamp(2023, 1, 1),
                    "power_installed_kw": 100,
                    "power_kw": 10.0,
                },
                {
                    "station": "station1",
                    "time": pd.Timestamp(2023, 1, 1, 1),
                    "power_installed_kw": 100,
                    "power_kw": 20,
                },
                {
                    "station": "station1",
                    "time": pd.Timestamp(2023, 1, 1, 2),
                    "power_installed_kw": 100,
                    "power_kw": 30,
                },
                {
                    "station": "station2",
                    "time": pd.Timestamp(2023, 1, 1),
                    "power_installed_kw": 0,
                    "power_kw": 0,
                },
                {
                    "station": "station2",
                    "time": pd.Timestamp(2023, 1, 1, 1),
                    "power_installed_kw": 0,
                    "power_kw": 0,
                },
                {
                    "station": "station2",
                    "time": pd.Timestamp(2023, 1, 1, 2),
                    "power_installed_kw": 0,
                    "power_kw": 0,
                },
            ]
        )

        self.power_df = power_df.set_index(["station", "time"])

        # DataFrame containing a timeseries of load factor data

        load_factor_df = pd.DataFrame(
            data=[
                {
                    "station": "station1",
                    "time": pd.Timestamp(2023, 1, 1),
                    "power_installed_kw": 100,
                    "load_factor": 0.1,
                },
                {
                    "station": "station1",
                    "time": pd.Timestamp(2023, 1, 1, 1),
                    "power_installed_kw": 100,
                    "load_factor": 0.2,
                },
                {
                    "station": "station1",
                    "time": pd.Timestamp(2023, 1, 1, 2),
                    "power_installed_kw": 100,
                    "load_factor": 0.3,
                },
                {
                    "station": "station2",
                    "time": pd.Timestamp(2023, 1, 1),
                    "power_installed_kw": 0,
                    "load_factor": 0,
                },
                {
                    "station": "station2",
                    "time": pd.Timestamp(2023, 1, 1, 1),
                    "power_installed_kw": 0,
                    "load_factor": 0,
                },
                {
                    "station": "station2",
                    "time": pd.Timestamp(2023, 1, 1, 2),
                    "power_installed_kw": 0,
                    "load_factor": 0,
                },
            ]
        )

        self.load_factor_df = load_factor_df.set_index(["station", "time"])

    def tearDown(self):
        logging.captureWarnings(False)
        logging.disable(logging.NOTSET)

    EXAMPLE_D_DIR = os.path.join(pathlib.Path(__file__).parent.absolute(), "example_d")
    STATIONS_PATH = os.path.join(EXAMPLE_D_DIR, "wind", "stations_wind.csv")

    def test_check_stations(self):
        stations = Contracts.read_contracts_from_file(
            TestPowerStations.STATIONS_PATH,
            date_start_col="date_start",
            date_end_exclusive_col="date_end_exclusive",
            date_format="%Y-%m-%d",
        )

        # check it fails if the station col is not found
        with self.assertRaises(ValueError):
            PowerStations.check_stations(
                stations,
                station_col="dummy",
                date_start_col="date_start",
                date_end_exclusive_col="date_end_exclusive",
            )

        # check it fails if the date_start_end is not found
        with self.assertRaises(ValueError):
            PowerStations.check_stations(
                stations,
                station_col="station",
                date_start_col="date_start",
                date_end_exclusive_col="dummy",
            )

        # check it fails if a NaN is found in the station col
        c = stations.copy(deep=True)
        c.loc[0, "station"] = None
        with self.assertRaises(ValueError):
            PowerStations.check_stations(
                c,
                station_col="station",
                date_start_col="date_start",
                date_end_exclusive_col="date_end_exclusive",
            )

        # check it fails if a duplicate station-date is found
        c.loc[0, "station"] = "eo_1"
        c = pd.concat([c, c.loc[0, :].to_frame().T], axis=0, ignore_index=True)
        with self.assertRaises(ValueError):
            PowerStations.check_stations(
                c,
                station_col="station",
                date_start_col="date_start",
                date_end_exclusive_col="date_end_exclusive",
            )

    def test_get_outages_from_file(self):
        """Test the get_outages_from_file function, with a working example and examples where outages percentage are
        either absent or with aberrant values"""

        # Check with a file that works

        output_df = PowerStations.get_outages_from_file(
            file_path=self.correct_outages_filepath,
            time_start_col="start_date",
            time_end_exclusive_col="excl_end_date",
            tzinfo="Europe/Paris",
            pct_outages_col="pct_outages",
        )

        pd.testing.assert_frame_equal(output_df, self.expected_outages_df)

        # Check when specifying a wrong outages column name

        with self.assertRaises(ValueError):
            PowerStations.get_outages_from_file(
                file_path=self.correct_outages_filepath,
                time_start_col="start_date",
                time_end_exclusive_col="excl_end_date",
                tzinfo="Europe/Paris",
                pct_outages_col="outages_pct",
            )

        # Check when the file has outages values outside the [0, 100] range

        filepath_wrong_df = os.path.join(self.outages_path, "outages_wrong_values.csv")

        with self.assertRaises(ValueError):
            PowerStations.get_outages_from_file(
                file_path=filepath_wrong_df,
                time_start_col="start_date",
                time_end_exclusive_col="excl_end_date",
                tzinfo="Europe/Paris",
                pct_outages_col="pct_outages",
            )

    def test_read_outages_from_file(self):
        """Test the read_outages_from_file function, once with a file that should work and one where the check_stations
        function raises an Error"""

        # Check with a file that should work

        output_df = PowerStations.read_outages_from_file(
            file_path=self.correct_outages_filepath,
            station_col="station",
            time_start_col="start_date",
            time_end_exclusive_col="excl_end_date",
            tzinfo="Europe/Paris",
            pct_outages_col="pct_outages",
        )

        pd.testing.assert_frame_equal(output_df, self.expected_outages_df)

        # Check with a file where an end date is inferior to a start date

        filepath_wrong_dates = os.path.join(
            self.outages_path, "outages_wrong_dates.csv"
        )

        with self.assertRaises(ValueError):
            PowerStations.read_outages_from_file(
                file_path=filepath_wrong_dates,
                station_col="station",
                time_start_col="start_date",
                time_end_exclusive_col="excl_end_date",
                tzinfo="Europe/Paris",
                pct_outages_col="pct_outages",
            )

    def test_get_daily_stations(self):
        """Test the get_daily_stations function"""

        input_stations_df = pd.DataFrame(
            data=[
                {
                    "station": "station1",
                    "date_start": pd.Timestamp(2023, 1, 1),
                    "date_end_excl": pd.Timestamp(2023, 1, 4),
                    "power_installed_kw": 30,
                },
                {
                    "station": "station1",
                    "date_start": pd.Timestamp(2023, 1, 7),
                    "date_end_excl": pd.Timestamp(2023, 1, 10),
                    "power_installed_kw": 60,
                },
                {
                    "station": "station2",
                    "date_start": pd.Timestamp(2023, 1, 1),
                    "date_end_excl": None,
                    "power_installed_kw": 100,
                },
            ]
        )

        max_date_exclusive = pd.Timestamp(2023, 1, 8)
        dti = pd.date_range(
            start=pd.Timestamp(2023, 1, 1), end=pd.Timestamp(2023, 1, 9), freq="D"
        )

        # Check when input DataFrame has a "date" or "event_date" column

        wrong_colname_df = input_stations_df.copy()
        wrong_colname_df["date"] = "oups"
        wrong_colname_df["control_column"] = "I did it again"

        with self.assertRaises(ValueError):
            PowerStations.get_stations_daily(
                stations=wrong_colname_df,
                station_col="station",
                date_start_col="date_start",
                date_end_exclusive_col="date_end_excl",
            )

        # Check when input DataFrame has a "control_column" and drop_gaps = True

        wrong_colname_df.drop("date", axis=1, inplace=True)

        with self.assertRaises(ValueError):
            PowerStations.get_stations_daily(
                stations=wrong_colname_df,
                station_col="station",
                date_start_col="date_start",
                date_end_exclusive_col="date_end_excl",
                drop_gaps=True,
            )

        # Check the result with no specified max_date_exclusive and drop_gaps = False

        expected_output_df = pd.DataFrame(
            data=[
                {"date": x, "station": "station1", "power_installed_kw": 30}
                for x in dti[:3]
            ]
            + [
                {"date": x, "station": "station1", "power_installed_kw": 0}
                for x in dti[3:6]
            ]
            + [
                {"date": x, "station": "station1", "power_installed_kw": 60}
                for x in dti[6:]
            ]
        )

        expected_output_df.set_index(["station", "date"], inplace=True)

        output_df = PowerStations.get_stations_daily(
            stations=input_stations_df,
            station_col="station",
            date_start_col="date_start",
            date_end_exclusive_col="date_end_excl",
        )

        pd.testing.assert_frame_equal(output_df, expected_output_df)

        # Check the result with drop_gaps = True

        expected_drop_gaps_df = expected_output_df.loc[
            expected_output_df.power_installed_kw > 0
        ]

        output_drop_gaps_df = PowerStations.get_stations_daily(
            stations=input_stations_df,
            station_col="station",
            date_start_col="date_start",
            date_end_exclusive_col="date_end_excl",
            drop_gaps=True,
        )

        pd.testing.assert_frame_equal(output_drop_gaps_df, expected_drop_gaps_df)

        # Check the result with a max_date_exclusive specified

        expected_max_date_df = pd.DataFrame(
            data=[
                {"date": x, "station": "station1", "power_installed_kw": 30}
                for x in dti[:3]
            ]
            + [
                {"date": x, "station": "station1", "power_installed_kw": 0}
                for x in dti[3:6]
            ]
            + [{"date": dti[6], "station": "station1", "power_installed_kw": 60}]
            + [
                {"date": x, "station": "station2", "power_installed_kw": 100}
                for x in dti[:7]
            ]
        )

        expected_max_date_df.set_index(["station", "date"], inplace=True)

        output_maxdate_df = PowerStations.get_stations_daily(
            stations=input_stations_df,
            station_col="station",
            date_start_col="date_start",
            date_end_exclusive_col="date_end_excl",
            max_date_exclusive=max_date_exclusive,
        )

        pd.testing.assert_frame_equal(output_maxdate_df, expected_max_date_df)

        # Check the result with a max_date_exclusive specified and drop_gaps = True

        expected_drop_gaps_max_date_df = expected_max_date_df.loc[
            expected_max_date_df.power_installed_kw > 0
        ]

        output_drop_gaps_max_date_df = PowerStations.get_stations_daily(
            stations=input_stations_df,
            station_col="station",
            date_start_col="date_start",
            date_end_exclusive_col="date_end_excl",
            max_date_exclusive=max_date_exclusive,
            drop_gaps=True,
        )

        pd.testing.assert_frame_equal(
            output_drop_gaps_max_date_df, expected_drop_gaps_max_date_df
        )

    def test_integrate_availability_from_outages(self):
        """Test the integrate_availability_from_outages function"""

        # Check when df_stations doesn't have a MultiIndex

        single_indexed_df = self.ref_stations_df.copy()
        single_indexed_df.reset_index(level="station", inplace=True)

        with self.assertRaises(TypeError):
            PowerStations.integrate_availability_from_outages(
                df_stations=single_indexed_df,
                df_outages=self.ref_outages_df,
                station_col="station",
                time_start_col="start_date",
                time_end_exclusive_col="excl_end_date",
                pct_outages_col="pct_outages",
            )

        # Check when df_stations has too many index levels

        three_levels_df = self.ref_stations_df.copy()
        three_levels_df.set_index("type_de_centrale", append=True, inplace=True)

        with self.assertRaises(TypeError):
            PowerStations.integrate_availability_from_outages(
                df_stations=three_levels_df,
                df_outages=self.ref_outages_df,
                station_col="station",
                time_start_col="start_date",
                time_end_exclusive_col="excl_end_date",
                pct_outages_col="pct_outages",
            )

        # Check when second index level of df_stations is not a DatetimeIndex

        swapped_index_df = self.ref_stations_df.copy()
        swapped_index_df = swapped_index_df.swaplevel()

        with self.assertRaises(TypeError):
            PowerStations.integrate_availability_from_outages(
                df_stations=swapped_index_df,
                df_outages=self.ref_outages_df,
                station_col="station",
                time_start_col="start_date",
                time_end_exclusive_col="excl_end_date",
                pct_outages_col="pct_outages",
            )

        # Check when specified station_col is not in outages_df

        with self.assertRaises(ValueError):
            PowerStations.integrate_availability_from_outages(
                df_stations=self.ref_stations_df.copy(),
                df_outages=self.ref_outages_df,
                station_col="identifiant_centrale",
                time_start_col="start_date",
                time_end_exclusive_col="excl_end_date",
                pct_outages_col="pct_outages",
            )

        # Check when specified time_start_col is not in outages_df

        with self.assertRaises(ValueError):
            PowerStations.integrate_availability_from_outages(
                df_stations=self.ref_stations_df.copy(),
                df_outages=self.ref_outages_df,
                station_col="station",
                time_start_col="time_start",
                time_end_exclusive_col="excl_end_date",
                pct_outages_col="pct_outages",
            )

        # Check when specified time_end_exclusive_col is not in outages_df

        with self.assertRaises(ValueError):
            PowerStations.integrate_availability_from_outages(
                df_stations=self.ref_stations_df.copy(),
                df_outages=self.ref_outages_df,
                station_col="station",
                time_start_col="start_date",
                time_end_exclusive_col="time_end",
                pct_outages_col="pct_outages",
            )

        # Check when "availability" column name is already in df_stations and no new name is specified

        wrong_col_df = self.ref_stations_df.copy().copy()
        wrong_col_df["availability"] = "oups"

        with self.assertRaises(ValueError):
            PowerStations.integrate_availability_from_outages(
                df_stations=wrong_col_df,
                df_outages=self.ref_outages_df,
                station_col="station",
                time_start_col="start_date",
                time_end_exclusive_col="excl_end_date",
                pct_outages_col="pct_outages",
            )

        # Check the result when inputs are correct

        output_df = PowerStations.integrate_availability_from_outages(
            df_stations=self.ref_stations_df.copy(),
            df_outages=self.ref_outages_df,
            station_col="station",
            time_start_col="start_date",
            time_end_exclusive_col="excl_end_date",
            pct_outages_col="pct_outages",
        )

        pd.testing.assert_frame_equal(output_df, self.ref_availability_df)

    def test_reset_installed_capacity(self):
        """Test the reset_installed_capacity function"""

        # Check with wrong column names

        with self.assertRaises(ValueError):
            PowerStations.reset_installed_capacity(
                df=self.ref_availability_df,
                installed_capacity_kw="puissance_installee_kw",
                stations_availability="availability",
            )

        # Check with some null values for availability

        stations_with_nan_df = self.ref_availability_df.copy()
        stations_with_nan_df.loc[
            stations_with_nan_df.availability == 0, "availability"
        ] = np.nan

        with self.assertRaises(ValueError):
            PowerStations.reset_installed_capacity(
                df=stations_with_nan_df,
                installed_capacity_kw="power_installed_kw",
                stations_availability="availability",
            )

        # Check with availability values superior to 1 (for example if they are still a percentage)

        big_values_df = self.ref_availability_df.copy()
        big_values_df.availability *= 100

        with self.assertRaises(ValueError):
            PowerStations.reset_installed_capacity(
                df=big_values_df,
                installed_capacity_kw="power_installed_kw",
                stations_availability="availability",
            )

        # Check with negative availability values

        neg_values_df = self.ref_availability_df.copy()
        neg_values_df.availability *= -1

        with self.assertRaises(ValueError):
            PowerStations.reset_installed_capacity(
                df=neg_values_df,
                installed_capacity_kw="power_installed_kw",
                stations_availability="availability",
            )

        # Check the result when inputs are correct

        output_df = PowerStations.reset_installed_capacity(
            df=self.ref_availability_df,
            installed_capacity_kw="power_installed_kw",
            stations_availability="availability",
        )

        pd.testing.assert_frame_equal(output_df, self.ref_integrated_outages_df)

    def test_integrate_outages(self):
        """Test the integrate_outages function"""

        # Check with 'availability_col' already in the stations_df columns

        wrong_columns_df = self.ref_stations_df.copy()
        wrong_columns_df["availability_col"] = "oups"

        with self.assertRaises(ValueError):
            PowerStations.integrate_outages(
                df_stations=wrong_columns_df,
                df_outages=self.ref_outages_df,
                station_col="station",
                time_start_col="start_date",
                time_end_exclusive_col="excl_end_date",
                installed_capacity_col="power_installed_kw",
                pct_outages_col="pct_outages",
            )

        # Check the result when inputs are correct

        output_df = PowerStations.integrate_outages(
            df_stations=self.ref_stations_df,
            df_outages=self.ref_outages_df,
            station_col="station",
            time_start_col="start_date",
            time_end_exclusive_col="excl_end_date",
            installed_capacity_col="power_installed_kw",
            pct_outages_col="pct_outages",
        )

        pd.testing.assert_frame_equal(output_df, self.ref_integrated_outages_df)

    def test_compute_load_factor(self):
        """Test the compute_load_factor function"""

        # Check with wrong column names

        with self.assertRaises(ValueError):
            PowerStations.compute_load_factor(
                df=self.power_df,
                installed_capacity_kw="puissance_installee_kw",
                power_kw="power_kw",
            )
        # Check the result when inputs are correct

        output_df = PowerStations.compute_load_factor(
            df=self.power_df,
            installed_capacity_kw="power_installed_kw",
            power_kw="power_kw",
        )

        pd.testing.assert_frame_equal(output_df, self.load_factor_df)

    def test_compute_power_kw_from_load_factor(self):
        """Test the compute_power_kw_from_load_factor function"""

        # Check with wrong column names

        with self.assertRaises(ValueError):
            PowerStations.compute_power_kw_from_load_factor(
                df=self.load_factor_df,
                installed_capacity_kw="power_installed_kw",
                load_factor="facteur_de_charge",
            )
        # Check the result when inputs are correct

        output_df = PowerStations.compute_power_kw_from_load_factor(
            df=self.load_factor_df,
            installed_capacity_kw="power_installed_kw",
            load_factor="load_factor",
        )

        pd.testing.assert_frame_equal(output_df, self.power_df)

    def test_clip_column(self):
        """
        Test clip column
        """

        input_load_factor_df = pd.DataFrame(data=[0.1, 0, 0.7, 1, -0.2, 3], columns=['test_column'])

        result_df = PowerStations.clip_column(
            df=input_load_factor_df,
            column_name='test_column',
            upper_bound=4,
            lower_bound=-1
        )
        pd.testing.assert_frame_equal(result_df, input_load_factor_df)

        result_df = PowerStations.clip_column(
            df=input_load_factor_df,
            column_name='test_column',
        )
        pd.testing.assert_frame_equal(result_df, input_load_factor_df)

        result_df = PowerStations.clip_column(
            df=input_load_factor_df,
            column_name='test_column',
            upper_bound=1
        )
        expected_result = pd.DataFrame(data=[0.1, 0, 0.7, 1, -0.2, 1], columns = ['test_column'])
        pd.testing.assert_frame_equal(result_df, expected_result)

        result_df = PowerStations.clip_column(
            df=input_load_factor_df,
            column_name='test_column',
            lower_bound=0
        )
        expected_result = pd.DataFrame(data=[0.1, 0, 0.7, 1, 0, 3], columns = ['test_column'])
        pd.testing.assert_frame_equal(result_df, expected_result)

        result_df = PowerStations.clip_column(
            df=input_load_factor_df,
            column_name='test_column',
            lower_bound=0,
            upper_bound=1
        )
        expected_result = pd.DataFrame(data=[0.1, 0, 0.7, 1, 0, 1], columns = ['test_column'])
        pd.testing.assert_frame_equal(result_df, expected_result)

        # with a multiindex
        input_load_factor_multi_df = input_load_factor_df.set_index([["A"] * 6, ["B"] * 6])
        result_df = PowerStations.clip_column(
            df=input_load_factor_multi_df,
            column_name='test_column',
            lower_bound=0,
            upper_bound=1
        )
        expected_result = pd.DataFrame(
            data=[0.1, 0, 0.7, 1, 0, 1], columns = ['test_column']
        ).set_index([["A"] * 6, ["B"] * 6])
        pd.testing.assert_frame_equal(result_df, expected_result)
