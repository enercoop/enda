import logging
import os
import pandas as pd
import pathlib
import unittest

from enda.contracts import Contracts
from enda.power_stations import PowerStations
from enda.timeseries import TimeSeries


class TestPowerStations(unittest.TestCase):
    def setUp(self):
        logging.captureWarnings(True)
        logging.disable(logging.ERROR)

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
        c = pd.concat([c, c.loc[0, :]], axis=0, ignore_index=True)
        with self.assertRaises(ValueError):
            PowerStations.check_stations(
                c,
                station_col="station",
                date_start_col="date_start",
                date_end_exclusive_col="date_end_exclusive",
            )

    @staticmethod
    def get_simple_stations_by_day():
        stations = Contracts.read_contracts_from_file(TestPowerStations.STATIONS_PATH)

        # count the running total, each day, of some columns
        stations_by_day = PowerStations.get_stations_daily(
            stations,
            station_col="station",
            date_start_col="date_start",
            date_end_exclusive_col="date_end_exclusive",
        )

        return stations_by_day

    def test_get_stations_daily(self):
        stations = TestPowerStations.get_simple_stations_by_day()

        self.assertEqual(stations.shape, (4612, 1))
        self.assertEqual(stations.iloc[[1], 0].item(), 1200.0)
        self.assertEqual(stations.loc[["eo_4"], "installed_capacity_kw"].max(), 3750.0)
        self.assertEqual(stations.index.get_level_values(0).nunique(), 4)

    def test_get_stations_between_dates(self):
        # test with a daily frequency

        stations = TestPowerStations.get_simple_stations_by_day()

        stations = PowerStations.get_stations_between_dates(
            stations=stations,
            start_datetime=pd.to_datetime("2020-02-18"),
            end_datetime_exclusive=pd.to_datetime("2020-02-20"),
        )

        # print(stations.loc[(['eo_4'], [start_datetime]),
        #                               "installed_capacity_kw"])
        # stations.loc[(['eo_4'], [start_datetime]), "installed_capacity_kw"] = 2

        # print(stations.loc[(['eo_4'], [start_datetime]),
        #                               "installed_capacity_kw"])

        self.assertEqual(stations.shape, (8, 1))
        self.assertEqual(stations.index.get_level_values(0).nunique(), 4)
        self.assertEqual(stations.iloc[[-1], 0].item(), 3000.0)
        self.assertEqual(stations.iloc[[-2], 0].item(), 3750.0)

    def test_get_stations_between_dates_2(self):
        # test with a different frequency

        stations = TestPowerStations.get_simple_stations_by_day()
        tz_str = "Europe/Paris"

        stations = TimeSeries.interpolate_daily_to_sub_daily_data(
            stations, freq="30min", tz=tz_str
        )

        stations = PowerStations.get_stations_between_dates(
            stations=stations,
            start_datetime=pd.to_datetime("2020-02-18 00:00:00+01:00").tz_convert(
                tz_str
            ),
            end_datetime_exclusive=pd.to_datetime(
                "2020-02-20 00:00:00+01:00"
            ).tz_convert(tz_str),
        )

        self.assertEqual(stations.shape, (384, 1))
        self.assertEqual(stations.index.get_level_values(0).nunique(), 4)
        self.assertEqual(stations.iloc[[0], 0].item(), 1200.0)
        self.assertEqual(stations.iloc[[300], 0].item(), 3750.0)
        self.assertEqual(stations.iloc[[-1], 0].item(), 3000.0)

    def test_compute_load_factor(self):
        # test compute_load_factor with default arguments

        stations = TestPowerStations.get_simple_stations_by_day()
        stations = PowerStations.get_stations_between_dates(
            stations=stations,
            start_datetime=pd.to_datetime("2020-02-18"),
            end_datetime_exclusive=pd.to_datetime("2020-02-20"),
        )

        # add a dummy 'power_kw' column to the dataframe
        # stations produce constant power over the 2 days
        stations["power_kw"] = sum([[i] * 2 for i in range(4)], [])

        final_stations = PowerStations.compute_load_factor(
            stations, installed_capacity_kw="installed_capacity_kw", power_kw="power_kw"
        )

        self.assertEqual(final_stations.shape, (8, 2))
        self.assertEqual(final_stations.index.get_level_values(0).nunique(), 4)
        self.assertEqual(final_stations.columns.nunique(), 2)
        self.assertEqual(final_stations.columns.unique()[0], "installed_capacity_kw")
        self.assertEqual(final_stations.columns.unique()[1], "load_factor")
        self.assertEqual(final_stations.iloc[[0], 1].item(), 0)
        self.assertEqual(final_stations.iloc[[-2], 1].item(), 0.0008)
        self.assertEqual(final_stations.iloc[[-1], 1].item(), 0.001)

        final_stations = PowerStations.compute_load_factor(
            stations,
            installed_capacity_kw="installed_capacity_kw",
            power_kw="power_kw",
            load_factor_col="load_kw",
            drop_power_kw=False,
        )

        self.assertEqual(final_stations.shape, (8, 3))
        self.assertEqual(final_stations.index.get_level_values(0).nunique(), 4)
        self.assertEqual(final_stations.columns.nunique(), 3)
        self.assertEqual(final_stations.columns.unique()[0], "installed_capacity_kw")
        self.assertEqual(final_stations.columns.unique()[1], "power_kw")
        self.assertEqual(final_stations.columns.unique()[2], "load_kw")
        self.assertEqual(final_stations.iloc[[0], 2].item(), 0)
        self.assertEqual(final_stations.iloc[[-2], 2].item(), 0.0008)
        self.assertEqual(final_stations.iloc[[-1], 2].item(), 0.001)

    def test_compute_power_kw_from_load_factor(self):
        # test compute_power_kw with default arguments

        stations = TestPowerStations.get_simple_stations_by_day()
        stations = PowerStations.get_stations_between_dates(
            stations=stations,
            start_datetime=pd.to_datetime("2020-02-18"),
            end_datetime_exclusive=pd.to_datetime("2020-02-20"),
        )

        # add a dummy 'load_kw' column to the dataframe
        # stations produce constant power over the 2 days
        stations["load_kw"] = [
            0,
            0,
            0.000556,
            0.000556,
            0.000351,
            0.000351,
            0.0008,
            0.001,
        ]

        final_stations = PowerStations.compute_power_kw_from_load_factor(
            stations,
            installed_capacity_kw="installed_capacity_kw",
            load_factor="load_kw",
        )

        self.assertEqual(final_stations.shape, (8, 2))
        self.assertEqual(final_stations.index.get_level_values(0).nunique(), 4)
        self.assertEqual(final_stations.columns.nunique(), 2)
        self.assertEqual(final_stations.columns.unique()[0], "installed_capacity_kw")
        self.assertEqual(final_stations.columns.unique()[1], "power_kw")
        self.assertEqual(final_stations.iloc[[0], 1].item(), 0)
        self.assertEqual(final_stations.iloc[[-2], 1].item(), 3)
        self.assertEqual(final_stations.iloc[[-1], 1].item(), 3)

        final_stations = PowerStations.compute_power_kw_from_load_factor(
            stations,
            installed_capacity_kw="installed_capacity_kw",
            load_factor="load_kw",
            power_kw_col="power_new",
            drop_load_factor=False,
        )

        self.assertEqual(final_stations.shape, (8, 3))
        self.assertEqual(final_stations.index.get_level_values(0).nunique(), 4)
        self.assertEqual(final_stations.columns.nunique(), 3)
        self.assertEqual(final_stations.columns.unique()[0], "installed_capacity_kw")
        self.assertEqual(final_stations.columns.unique()[1], "load_kw")
        self.assertEqual(final_stations.columns.unique()[2], "power_new")
        self.assertEqual(final_stations.iloc[[0], 2].item(), 0)
        self.assertEqual(final_stations.iloc[[-2], 2].item(), 3)
        self.assertEqual(final_stations.iloc[[-1], 2].item(), 3)
