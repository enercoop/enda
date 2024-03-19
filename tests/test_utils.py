import os
import pathlib
import pandas as pd
from enda.contracts import Contracts
from enda.feature_engineering.datetime_features import DatetimeFeature
from enda.power_stations import PowerStations
from enda.tools.portfolio_tools import PortfolioTools
from enda.tools.resample import Resample
from enda.tools.timeseries import TimeSeries
from enda.tools.timezone_utils import TimezoneUtils


class TestUtils:
    """Some functions useful for the tests."""

    EXAMPLE_A_DIR = os.path.join(pathlib.Path(__file__).parent.absolute(), "example_a")
    EXAMPLE_D_DIR = os.path.join(pathlib.Path(__file__).parent.absolute(), "example_d")

    @staticmethod
    def read_example_a_train_test_sets():
        """Read the example a data, put all contracts in just one group and compute train/test sets"""

        contracts = Contracts.read_contracts_from_file(
            os.path.join(TestUtils.EXAMPLE_A_DIR, "contracts.csv")
        )
        contracts[
            "contracts_count"
        ] = 1  # add a variable to count the number of contracts for each row

        # count the running total, each day, of some columns
        portfolio_by_day = Contracts.compute_portfolio_by_day(
            contracts,
            columns_to_sum=[
                "contracts_count",
                "subscribed_power_kva",
                "estimated_annual_consumption_kwh",
            ],
            date_start_col="date_start",
            date_end_exclusive_col="date_end_exclusive",
        )

        # restrict/extend the portfolio_by_day to desired dates
        portfolio_by_day = PortfolioTools.get_portfolio_between_dates(
            portfolio_by_day,
            start_datetime=pd.to_datetime("2020-09-16"),
            end_datetime_exclusive=pd.to_datetime("2020-09-24"),
        )

        # turn the portfolio_by_day into a portfolio timeseries with our desired freq and timezone
        portfolio = Resample.upsample_and_interpolate(
            portfolio_by_day, freq="15min", tz_info="Europe/Berlin", forward_fill=True, index_name='time'
        )

        # read historical load, weather and TSO forecast data
        historic_load_measured = pd.read_csv(
            os.path.join(TestUtils.EXAMPLE_A_DIR, "historic_load_measured.csv")
        )
        weather_and_tso_forecasts = pd.read_csv(
            os.path.join(TestUtils.EXAMPLE_A_DIR, "weather_and_tso_forecasts.csv")
        )

        for df in [historic_load_measured, weather_and_tso_forecasts]:
            df["time"] = pd.to_datetime(df["time"])
            # for now df['time'] can be of dtype "object" because there are 2 french timezones: +60min and +120min.
            # it is important to align time-zone to 'Europe/Paris' to make sure the df has a pandas.DatetimeIndex
            df["time"] = TimezoneUtils.convert_dtype_from_object_to_tz_aware(df["time"], tz_info="Europe/Berlin")
            df.set_index("time", inplace=True)

        # prepare datasets
        target_name = "load_kw"
        train_set = portfolio[portfolio.index <= historic_load_measured.index.max()]
        train_set = pd.merge(
            train_set,
            (
                historic_load_measured["slp_kw"]
                + historic_load_measured["smart_metered_kw"]
            ).to_frame(target_name),
            how="inner",
            left_index=True,
            right_index=True,
        )
        train_set = pd.merge(
            train_set,
            weather_and_tso_forecasts,
            how="inner",
            left_index=True,
            right_index=True,
        )

        # lets create the input data for our forecast
        test_set = portfolio[
            portfolio.index >= pd.to_datetime("2020-09-20 00:00:00+02:00")
        ]
        test_set = pd.merge(
            test_set,
            weather_and_tso_forecasts,
            how="inner",
            left_index=True,
            right_index=True,
        )

        assert isinstance(train_set.index, pd.DatetimeIndex)
        assert str(train_set.index.tz) == "Europe/Berlin"
        assert isinstance(test_set.index, pd.DatetimeIndex)
        assert str(test_set.index.tz) == "Europe/Berlin"

        assert train_set.shape == (384, 6)
        assert test_set.shape == (384, 5)

        return train_set, test_set, target_name

    @staticmethod
    def read_example_b_train_test_sets():
        raise NotImplementedError()

    @staticmethod
    def read_example_d_dataset(source):
        """
        Read the example d data for a techno in particular,
        and compute train/test sets
        """
        if source not in ["wind", "solar", "river"]:
            raise NotImplementedError("unknown source argument")

        # get station portfolio
        stations = Contracts.read_contracts_from_file(
            os.path.join(TestUtils.EXAMPLE_D_DIR, source, "stations_" + source + ".csv")
        )

        # display it as a multiindex with day as second index
        stations = PowerStations.get_stations_daily(
            stations,
            station_col="station",
            date_start_col="date_start",
            date_end_exclusive_col="date_end_exclusive",
        )

        # between dates of interest
        stations = PortfolioTools.get_portfolio_between_dates(
            stations,
            start_datetime=pd.to_datetime("2020-01-01"),
            end_datetime_exclusive=pd.to_datetime("2021-01-11"),
        )

        # on a 30-minutes scale
        stations = Resample.upsample_and_interpolate(
            stations, freq="30min", tz_info="Europe/Paris", forward_fill=True, index_name="time"
        )

        # get events like outages and shutdowns
        filepath = os.path.join(TestUtils.EXAMPLE_D_DIR, "events.csv")
        outages = PowerStations.read_outages_from_file(
            filepath,
            station_col="station",
            time_start_col="time_start",
            time_end_exclusive_col="time_end",
            pct_outages_col="impact_production_pct_kw",
            tzinfo="Europe/Paris",
        )

        stations = PowerStations.integrate_outages(
            df_stations=stations,
            df_outages=outages,
            station_col="station",
            time_start_col="time_start",
            time_end_exclusive_col="time_end",
            installed_capacity_col="installed_capacity_kw",
            pct_outages_col="impact_production_pct_kw",
        )

        # get production
        production = pd.read_csv(
            os.path.join(
                TestUtils.EXAMPLE_D_DIR, source, "production_" + source + ".csv"
            )
        )
        production["time"] = pd.to_datetime(production["time"])
        production["time"] = TimezoneUtils.convert_dtype_from_object_to_tz_aware(
            production["time"], tz_info="Europe/Paris"
        )
        production.set_index(["station", "time"], inplace=True)

        production = Resample.downsample(
            production,
            freq="30min",
            agg_functions="mean",
            is_original_frequency_unique=True,
            index_name="time"
        )

        production = TimezoneUtils.set_timezone(production, tz_info="Europe/Paris")

        dataset = pd.merge(
            stations, production, how="inner", left_index=True, right_index=True
        )

        # get weather for wind and solar
        if source in ["wind", "solar"]:
            weather = pd.read_csv(
                os.path.join(
                    TestUtils.EXAMPLE_D_DIR,
                    source,
                    "weather_forecast_" + source + ".csv",
                )
            )
            weather["time"] = pd.to_datetime(weather["time"])
            weather["time"] = TimezoneUtils.convert_dtype_from_object_to_tz_aware(
                weather["time"], tz_info="Europe/Paris"
            )
            weather.set_index(["station", "time"], inplace=True)

            weather = TimeSeries.interpolate_freq_to_sub_freq_data(
                weather,
                freq="30min",
                tz="Europe/Paris",
                index_name="time",
                method="linear",
            )

            dataset = pd.merge(
                dataset, weather, how="inner", left_index=True, right_index=True
            )

        # featurize for solar
        if source == "solar":
            dataset = DatetimeFeature.split_datetime(
                dataset, split_list=["minuteofday", "dayofyear"]
            )

            dataset = DatetimeFeature.encode_cyclic_datetime_index(
                dataset, split_list=["minuteofday", "dayofyear"]
            )

        # compute load factor
        dataset = PowerStations.compute_load_factor(
            dataset,
            installed_capacity_kw="installed_capacity_kw",
            power_kw="power_kw",
            drop_power_kw=True,
        )

        return dataset

    @staticmethod
    def get_example_d_train_test_sets(source):
        # read the full dataset
        dataset = TestUtils.read_example_d_dataset(source)
        target_col = "load_factor"

        # lets create the input data for our forecast
        test_set = dataset[
            dataset.index.get_level_values(1)
            >= pd.to_datetime("2021-01-02 00:00:00+01:00")
        ]
        test_set = test_set.drop(columns=target_col)

        # let's create the input train dataset
        train_set = dataset[
            dataset.index.get_level_values(1)
            < pd.to_datetime("2021-01-01 00:00:00+01:00")
        ]

        return train_set, test_set, target_col
