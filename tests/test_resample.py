""""A module for testing the Resample class in enda/resample.py"""

import logging
import unittest

import numpy as np
import pandas as pd
import enda.tools.resample


class TestResample(unittest.TestCase):
    """
    This class aims at testing the functions of the Resample class in
        enda/resample.py
    """

    def setUp(self):
        logging.captureWarnings(True)
        logging.disable(logging.ERROR)

        # a typical perfect time series with tz-aware timestamps
        self.perfect_df = pd.DataFrame(
            [
                [pd.to_datetime("2021-01-01 00:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-01 12:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-02 00:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-02 12:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-03 00:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-03 12:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-04 00:00:00+01:00"), 2],
                [pd.to_datetime("2021-01-04 12:00:00+01:00"), 2],
                [pd.to_datetime("2021-01-05 00:00:00+01:00"), 2],
                [pd.to_datetime("2021-01-05 12:00:00+01:00"), 2],
                [pd.to_datetime("2021-01-06 00:00:00+01:00"), 2],
                [pd.to_datetime("2021-01-06 12:00:00+01:00"), 2],
            ],
            columns=["time", "value"],
        ).set_index("time")

        # the same time series with missing and extra points
        self.imperfect_df = pd.DataFrame(
            [
                [pd.to_datetime("2021-01-01 00:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-01 12:00:00+01:00"), 1],
                # 2021-01-02 00:00:00 missing
                [pd.to_datetime("2021-01-02 12:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-03 00:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-03 12:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-03 20:00:00+01:00"), 1],  # 20H in extra value
                [pd.to_datetime("2021-01-04 00:00:00+01:00"), 2],
                [pd.to_datetime("2021-01-04 12:00:00+01:00"), 2],
                [pd.to_datetime("2021-01-05 00:00:00+01:00"), 2],
                [pd.to_datetime("2021-01-05 12:00:00+01:00"), 2],
                [
                    pd.to_datetime("2021-01-05 00:00:00+01:00"),
                    3,
                ],  # midnight repeated, poorly placed
                [pd.to_datetime("2021-01-06 00:00:00+01:00"), 2],
                [pd.to_datetime("2021-01-06 12:00:00+01:00"), 2],
            ],
            columns=["time", "value"],
        ).set_index("time")

        # define equivalent dataframes with an extra column
        self.perfect_two_columns_df = (
            self.perfect_df.copy()
            .assign(day=lambda _: _.index.day)
            .assign(is_start_week=lambda _: _.day < 3)
            .drop(columns="day")
        )

        self.imperfect_two_columns_df = (
            self.imperfect_df.copy()
            .assign(day=lambda _: _.index.day)
            .assign(is_start_week=lambda _: _.day < 3)
            .drop(columns="day")
        )

        # define 3-levels multi-index dataframes based on them
        perfect_multi_df = self.perfect_two_columns_df.reset_index().copy()
        perfect_multi_df["name"] = perfect_multi_df["time"].apply(
            lambda _: "name1" if (_.day <= 4) else "name2"
        )
        self.perfect_multi_df = perfect_multi_df.set_index(
            ["name", "is_start_week", "time"]
        ).sort_index()

        imperfect_multi_df = self.imperfect_two_columns_df.reset_index().copy()
        imperfect_multi_df["name"] = imperfect_multi_df["time"].apply(
            lambda _: "name1" if (_.day <= 4) or (_.hour == 12) else "name2"
        )
        self.imperfect_multi_df = imperfect_multi_df.set_index(
            ["name", "is_start_week", "time"]
        ).sort_index()

        # monthly df
        self.monthly_df = pd.DataFrame(
            [
                [pd.Timestamp(2021, 1, 1), 100],
                [pd.Timestamp(2021, 2, 1), 200],
                [pd.Timestamp(2021, 3, 1), 300],
            ],
            columns=["time", "value"],
        ).set_index("time")

        # 30 seconds-timestamp df
        self.thirty_seconds_df = pd.DataFrame(
            [
                [pd.Timestamp("2023-01-01 00:00:00"), 1, "val"],
                [pd.Timestamp("2023-01-01 00:00:30"), 2, "val"],
            ],
            columns=["time", "value", "value_str"],
        ).set_index("time")

        # single-row dataframe
        self.single_row_df = pd.DataFrame(
            {"time": [pd.Timestamp("2023-01-01").tz_localize(("Europe/Paris"))],
             "value": [3]
             }).set_index("time")

        # empty dataframe
        schema = {'time': np.datetime64, "value": int}
        self.empty_df = pd.DataFrame(columns=schema.keys()).astype(schema).set_index("time")

    def test_downsample(self):
        """
        Test downsample
        """

        # test that we can't downsample to a smaller period
        with self.assertRaises(RuntimeError):
            enda.tools.resample.Resample.downsample(self.perfect_df, freq="6H")

        # test down-sampling to a 1D (24H) frequency, 'sum as agg_function, and change of index name
        # with perfect df (test check frequency unique)
        expected_df = pd.DataFrame(
            [
                [pd.to_datetime("2021-01-01 00:00:00+01:00"), 2],
                [pd.to_datetime("2021-01-02 00:00:00+01:00"), 2],
                [pd.to_datetime("2021-01-03 00:00:00+01:00"), 2],
                [pd.to_datetime("2021-01-04 00:00:00+01:00"), 4],
                [pd.to_datetime("2021-01-05 00:00:00+01:00"), 4],
                [pd.to_datetime("2021-01-06 00:00:00+01:00"), 4],
            ],
            columns=["date", "value"],
        ).set_index("date")
        expected_df.index.freq = "1D"

        result_df = enda.tools.resample.Resample.downsample(
            self.perfect_df,
            freq="1D",
            agg_functions="sum",
            is_original_frequency_unique=True,
            index_name="date",
        )
        pd.testing.assert_frame_equal(result_df, expected_df)

        # test down-sampling to a 1D (24H) frequency, 'sum' as agg_function,
        # with imperfect df
        expected_df = pd.DataFrame(
            [
                [pd.to_datetime("2021-01-01 00:00:00+01:00"), 2],
                [pd.to_datetime("2021-01-02 00:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-03 00:00:00+01:00"), 3],
                [pd.to_datetime("2021-01-04 00:00:00+01:00"), 4],
                [pd.to_datetime("2021-01-05 00:00:00+01:00"), 7],
                [pd.to_datetime("2021-01-06 00:00:00+01:00"), 4],
            ],
            columns=["time", "value"],
        ).set_index("time")
        expected_df.index.freq = "1D"

        result_df = enda.tools.resample.Resample.downsample(
            self.imperfect_df, freq="24H", agg_functions="sum"
        )
        pd.testing.assert_frame_equal(result_df, expected_df)

        # test down-sampling to a 1D (24H) frequency, 'mean' as agg_function with group by option
        expected_df = pd.DataFrame(
            [
                [pd.to_datetime("2021-01-01 00:00:00+01:00"), 3, True],
                [pd.to_datetime("2021-01-03 00:00:00+01:00"), 14, False],
                [pd.to_datetime("2021-01-06 00:00:00+01:00"), 4, False],
            ],
            columns=["date", "value", "is_start_week"],
        ).set_index("date")
        expected_df.index.freq = None

        result_df = enda.tools.resample.Resample.downsample(
            self.imperfect_two_columns_df,
            freq="3D",
            agg_functions="sum",
            groupby=["is_start_week"],
            index_name="date",
        )
        pd.testing.assert_frame_equal(result_df, expected_df)

        # test changing the origin, and the agg function to dict
        # use the implicit conversion bool -> int for is_start_week
        result_df = enda.tools.resample.Resample.downsample(
            self.imperfect_two_columns_df,
            freq="2D",
            agg_functions={"value": "sum", "is_start_week": "mean"},
            origin=pd.to_datetime("2021-01-02 00:00:00+01:00"),
        )

        expected_df = pd.DataFrame(
            [
                [pd.to_datetime("2020-12-31 00:00:00+01:00"), 2, 1],
                [pd.to_datetime("2021-01-02 00:00:00+01:00"), 4, 0.25],
                [pd.to_datetime("2021-01-04 00:00:00+01:00"), 11, 0],
                [pd.to_datetime("2021-01-06 00:00:00+01:00"), 4, 0],
            ],
            columns=["time", "value", "is_start_week"],
        ).set_index("time")
        expected_df.index.freq = "48H"

        pd.testing.assert_frame_equal(result_df, expected_df)

        # test changing the origin with group by
        result_df = enda.tools.resample.Resample.downsample(
            self.imperfect_two_columns_df,
            freq="2D",
            agg_functions="sum",
            origin=pd.to_datetime("2021-01-02 00:00:00+01:00"),
            groupby=["is_start_week"],
        )
        expected_df = pd.DataFrame(
            [
                [pd.to_datetime("2020-12-31 00:00:00+01:00"), 2, True],
                [pd.to_datetime("2021-01-02 00:00:00+01:00"), 3, False],
                [pd.to_datetime("2021-01-02 00:00:00+01:00"), 1, True],
                [pd.to_datetime("2021-01-04 00:00:00+01:00"), 11, False],
                [pd.to_datetime("2021-01-06 00:00:00+01:00"), 4, False],
            ],
            columns=["time", "value", "is_start_week"],
        ).set_index("time")
        expected_df.index.freq = None

        pd.testing.assert_frame_equal(result_df, expected_df)

        # test with a multi-index
        result_df = enda.tools.resample.Resample.downsample(
            self.imperfect_multi_df, freq="1D", agg_functions="sum"
        )
        expected_df = pd.DataFrame(
            [
                [pd.to_datetime("2021-01-03 00:00:00+01:00"), "name1", False, 3],
                [pd.to_datetime("2021-01-04 00:00:00+01:00"), "name1", False, 4],
                [pd.to_datetime("2021-01-05 00:00:00+01:00"), "name1", False, 2],
                [pd.to_datetime("2021-01-06 00:00:00+01:00"), "name1", False, 2],
                [pd.to_datetime("2021-01-01 00:00:00+01:00"), "name1", True, 2],
                [pd.to_datetime("2021-01-02 00:00:00+01:00"), "name1", True, 1],
                [pd.to_datetime("2021-01-05 00:00:00+01:00"), "name2", False, 5],
                [pd.to_datetime("2021-01-06 00:00:00+01:00"), "name2", False, 2],
            ],
            columns=["time", "name", "is_start_week", "value"],
        ).set_index(["name", "is_start_week", "time"])
        expected_df.index.freq = None

        pd.testing.assert_frame_equal(result_df, expected_df)

        # test with monthly dataframe
        result_df = enda.tools.resample.Resample.downsample(
            self.monthly_df, freq="2MS", agg_functions="mean"
        )
        expected_df = pd.DataFrame(
            [
                [pd.to_datetime("2021-01-01"), 150.0],
                [pd.to_datetime("2021-03-01"), 300],
            ],
            columns=["time", "value"],
        ).set_index(["time"])
        expected_df.index.freq = "2MS"

        pd.testing.assert_frame_equal(result_df, expected_df)

        # test with single row dataframe
        result_df = enda.tools.resample.Resample.downsample(
            self.single_row_df, freq="Y", agg_functions="mean"
        )
        expected_df = pd.DataFrame(
            [[pd.to_datetime("2023-12-31").tz_localize("Europe/Paris"), 3.]],
            columns=["time", "value"],
        ).set_index(["time"])
        expected_df.index.freq = "Y"

        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_equal_sample_fillna(self):
        """
        Test equal_sample_fillna
        """

        # check with self.df_12h_imperfect: error because duplicates and extra periods
        with self.assertRaises(RuntimeError):
            enda.tools.resample.Resample.equal_sample_fillna(self.imperfect_df)

        # in case both options are active
        with self.assertRaises(ValueError):
            enda.tools.resample.Resample.equal_sample_fillna(
                self.perfect_df, fill_value=2, method_filling="bfill"
            )

        # in case non-existent
        # with self.assertRaises(RuntimeError):
        #    enda.resample.Resample.equal_sample_fillna(self.df_12h_perfect, method_filling="dumb")

        # check with imperfect dataframe with no extra value, and different input for testing
        df_12h_imperfect_no_extra_value = pd.DataFrame(
            [
                [pd.to_datetime("2021-01-01 00:00:00+01:00"), 1.0],
                [
                    pd.to_datetime("2021-01-01 12:00:00+01:00"),
                    3.0,
                ],  # 2021-01-02 00:00:00 missing
                [pd.to_datetime("2021-01-02 12:00:00+01:00"), 2],
                [pd.to_datetime("2021-01-03 00:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-03 12:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-04 00:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-04 12:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-05 00:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-05 12:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-06 00:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-06 12:00:00+01:00"), 1],
            ],
            columns=["time", "value"],
        ).set_index("time")

        outcome_df = enda.tools.resample.Resample.equal_sample_fillna(
            df_12h_imperfect_no_extra_value
        )

        expected_df = pd.DataFrame(
            [
                [pd.to_datetime("2021-01-01 00:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-01 12:00:00+01:00"), 3],
                [pd.to_datetime("2021-01-02 00:00:00+01:00"), None],
                [pd.to_datetime("2021-01-02 12:00:00+01:00"), 2],
                [pd.to_datetime("2021-01-03 00:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-03 12:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-04 00:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-04 12:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-05 00:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-05 12:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-06 00:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-06 12:00:00+01:00"), 1],
            ],
            columns=["time", "value"],
        ).set_index("time")
        expected_df.index.freq = pd.to_timedelta("12H")

        pd.testing.assert_frame_equal(outcome_df, expected_df)

        # test with other input parameters
        outcome_df = enda.tools.resample.Resample.equal_sample_fillna(
            df_12h_imperfect_no_extra_value, fill_value=4
        )
        expected_df.loc[pd.to_datetime("2021-01-02 00:00:00+01:00"), "value"] = 4
        pd.testing.assert_frame_equal(outcome_df, expected_df)

        outcome_df = enda.tools.resample.Resample.equal_sample_fillna(
            df_12h_imperfect_no_extra_value, method_filling="ffill"
        )
        expected_df.loc[pd.to_datetime("2021-01-02 00:00:00+01:00"), "value"] = 3
        pd.testing.assert_frame_equal(outcome_df, expected_df)

        outcome_df = enda.tools.resample.Resample.equal_sample_fillna(
            df_12h_imperfect_no_extra_value, method_filling="bfill"
        )
        expected_df.loc[pd.to_datetime("2021-01-02 00:00:00+01:00"), "value"] = 2
        pd.testing.assert_frame_equal(outcome_df, expected_df)

    def test_upsample_and_interpolate_regular(self):
        """
        Test upsample_and_interpolate
        """

        # test up sampling to a 6H frequency, linear interpolation, and change of index name
        # with perfect df (test check frequency unique)
        expected_df = pd.DataFrame(
            [
                [pd.to_datetime("2021-01-01 00:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-01 06:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-01 12:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-01 18:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-02 00:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-02 06:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-02 12:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-02 18:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-03 00:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-03 06:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-03 12:00:00+01:00"), 1],
                [pd.to_datetime("2021-01-03 18:00:00+01:00"), 1.5],
                [pd.to_datetime("2021-01-04 00:00:00+01:00"), 2],
                [pd.to_datetime("2021-01-04 06:00:00+01:00"), 2],
                [pd.to_datetime("2021-01-04 12:00:00+01:00"), 2],
                [pd.to_datetime("2021-01-04 18:00:00+01:00"), 2],
                [pd.to_datetime("2021-01-05 00:00:00+01:00"), 2],
                [pd.to_datetime("2021-01-05 06:00:00+01:00"), 2],
                [pd.to_datetime("2021-01-05 12:00:00+01:00"), 2],
                [pd.to_datetime("2021-01-05 18:00:00+01:00"), 2],
                [pd.to_datetime("2021-01-06 00:00:00+01:00"), 2],
                [pd.to_datetime("2021-01-06 06:00:00+01:00"), 2],
                [
                    pd.to_datetime("2021-01-06 12:00:00+01:00"),
                    2,
                ],  # note 18h is not present
            ],
            columns=["six_hours", "value"],
        ).set_index("six_hours")
        expected_df.index.freq = "6H"

        result_df = enda.tools.resample.Resample.upsample_and_interpolate(
            self.perfect_df,
            freq="6H",
            is_original_frequency_unique=True,
            index_name="six_hours",
        )
        pd.testing.assert_frame_equal(result_df, expected_df)

        # test the same call, but with forward-fill at the end
        expected_df = pd.concat(
            [
                expected_df,
                pd.DataFrame(
                    [[pd.to_datetime("2021-01-06 18:00:00+01:00"), 2]],
                    columns=["six_hours", "value"],
                ).set_index("six_hours"),
            ]
        )
        expected_df.index.freq = "6H"

        result_df = enda.tools.resample.Resample.upsample_and_interpolate(
            self.perfect_df, freq="6H", forward_fill=True, index_name="six_hours"
        )
        pd.testing.assert_frame_equal(result_df, expected_df)

        # test with a multi-index
        expected_df = pd.DataFrame(
            [
                ["name1", False, pd.to_datetime("2021-01-03 00:00:00+01:00"), 1.0],
                ["name1", False, pd.to_datetime("2021-01-03 06:00:00+01:00"), 1.0],
                ["name1", False, pd.to_datetime("2021-01-03 12:00:00+01:00"), 1.0],
                ["name1", False, pd.to_datetime("2021-01-03 18:00:00+01:00"), 1.5],
                ["name1", False, pd.to_datetime("2021-01-04 00:00:00+01:00"), 2.0],
                ["name1", False, pd.to_datetime("2021-01-04 06:00:00+01:00"), 2.0],
                ["name1", False, pd.to_datetime("2021-01-04 12:00:00+01:00"), 2.0],
                ["name1", False, pd.to_datetime("2021-01-04 18:00:00+01:00"), 2.0],
                ["name1", True, pd.to_datetime("2021-01-01 00:00:00+01:00"), 1.0],
                ["name1", True, pd.to_datetime("2021-01-01 06:00:00+01:00"), 1.0],
                ["name1", True, pd.to_datetime("2021-01-01 12:00:00+01:00"), 1.0],
                ["name1", True, pd.to_datetime("2021-01-01 18:00:00+01:00"), 1.0],
                ["name1", True, pd.to_datetime("2021-01-02 00:00:00+01:00"), 1.0],
                ["name1", True, pd.to_datetime("2021-01-02 06:00:00+01:00"), 1.0],
                ["name1", True, pd.to_datetime("2021-01-02 12:00:00+01:00"), 1.0],
                ["name1", True, pd.to_datetime("2021-01-02 18:00:00+01:00"), 1.0],
                ["name2", False, pd.to_datetime("2021-01-05 00:00:00+01:00"), 2.0],
                ["name2", False, pd.to_datetime("2021-01-05 06:00:00+01:00"), 2.0],
                ["name2", False, pd.to_datetime("2021-01-05 12:00:00+01:00"), 2.0],
                ["name2", False, pd.to_datetime("2021-01-05 18:00:00+01:00"), 2.0],
                ["name2", False, pd.to_datetime("2021-01-06 00:00:00+01:00"), 2.0],
                ["name2", False, pd.to_datetime("2021-01-06 06:00:00+01:00"), 2.0],
                ["name2", False, pd.to_datetime("2021-01-06 12:00:00+01:00"), 2.0],
                ["name2", False, pd.to_datetime("2021-01-06 18:00:00+01:00"), 2.0],
            ],
            columns=["name", "is_start_week", "time", "value"],
        ).set_index(["name", "is_start_week", "time"])

        result_df = enda.tools.resample.Resample.upsample_and_interpolate(
            self.perfect_multi_df, freq="6H", forward_fill=True
        )
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_upsample_and_interpolate_types(self):
        """
        Test upsample_and_interpolate with a 30 minutes timestamp and strings as column and forward fill
        """

        # test with a 30 minutes timestamp and strings as column and forward fill
        result_df = enda.tools.resample.Resample.upsample_and_interpolate(self.thirty_seconds_df,
                                                                          freq="15s",
                                                                          tz_info="Europe/Paris",
                                                                          forward_fill=True,
                                                                          method="ffill")
        expected_df = (
            pd.date_range(
                start=pd.to_datetime("2023-01-01 00:00:00").tz_localize("Europe/Paris"),
                end=pd.to_datetime("2023-01-01 00:01:00").tz_localize("Europe/Paris"),
                freq="15S",
                name="time",
                inclusive="left"
            )
            .to_frame()
            .assign(value=[1, 1, 2, 2])
            .assign(value_str=["val", "val", "val", "val"])
            .set_index("time")
        )
        expected_df.index.freq = "15S"

        pd.testing.assert_frame_equal(result_df, expected_df)

        # test with a linear interpolation: string values are not linearly interpolated, it makes a np.nan
        result_df = enda.tools.resample.Resample.upsample_and_interpolate(self.thirty_seconds_df,
                                                                          freq="15s",
                                                                          tz_info="Europe/Paris",
                                                                          forward_fill=True,
                                                                          method="index")
        expected_df = (
            pd.date_range(
                start=pd.to_datetime("2023-01-01 00:00:00").tz_localize("Europe/Paris"),
                end=pd.to_datetime("2023-01-01 00:01:00").tz_localize("Europe/Paris"),
                freq="15S",
                name="time",
                inclusive="left"
            )
            .to_frame()
            .assign(value=[1., 1.5, 2, 2])
            .assign(value_str=["val", np.nan, "val", "val"])
            .set_index("time")
        )
        expected_df.index.freq = "15S"
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_upsample_and_interpolate_empty_dataframe(self):
        """
        Test upsample_and_interpolate with an empty dataframe
        """

        # test with an empty dataframe, and another time zone
        result_df = enda.tools.resample.Resample.upsample_and_interpolate(self.empty_df,
                                                                          freq="10min",
                                                                          tz_info="Europe/London")
        expected_df = self.empty_df.copy()
        expected_df.index = expected_df.index.tz_localize("Europe/London")
        expected_df.index.freq = "10min"

        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_upsample_and_interpolate_single_row(self):
        """
        Test upsample_and_interpolate with a single row dataframe
        """

        # test with a one-rowed dataframe
        result_df = enda.tools.resample.Resample.upsample_and_interpolate(self.single_row_df,
                                                                          expected_initial_freq="30min",
                                                                          forward_fill=True,
                                                                          freq="10min")
        expected_df = (
            pd.date_range(
                start=pd.to_datetime("2023-01-01 00:00:00").tz_localize("Europe/Paris"),
                end=pd.to_datetime("2023-01-01 00:30:00").tz_localize("Europe/Paris"),
                freq="10min",
                name="time",
                inclusive="left"
            )
            .to_frame()
            .assign(value=[3, 3, 3])
            .set_index("time")
        )
        expected_df.index.freq = '10min'
        pd.testing.assert_frame_equal(result_df, expected_df)

        # test with a single row and not forward-fill, and another time zone
        # it just sets the time zone and the index freq
        result_df = enda.tools.resample.Resample.upsample_and_interpolate(
            self.single_row_df, freq="W", tz_info="Europe/London"
        )
        expected_df = self.single_row_df.copy()
        expected_df.index = expected_df.index.tz_convert("Europe/London")
        expected_df = expected_df.resample("W").last()  # only way to set a frequency for a one-row dataframe in pandas
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_upsample_and_interpolate_errors(self):
        """
        Test with errors
        """

        # test that we can't upsample to a higher period
        with self.assertRaises(ValueError):
            enda.tools.resample.Resample.upsample_and_interpolate(
                self.perfect_df, freq="1D"
            )

        # test error with duplicates
        with self.assertRaises(ValueError):
            enda.tools.resample.Resample.upsample_and_interpolate(
                self.imperfect_df, freq="6H"
            )

        # test error when expected initial freq is not the same as the one expected
        with self.assertRaises(ValueError):
            enda.tools.resample.Resample.upsample_and_interpolate(
                self.imperfect_df, freq="6H"
            )

        # test if expected initial freq is given and not equal to the inferred freq
        with self.assertRaises(ValueError):
            enda.tools.resample.Resample.upsample_and_interpolate(
                self.perfect_df, freq="6H", expected_initial_freq='2D'
            )

    def test_upsample_and_divide_evenly(self):
        """
        Test upsample_and_divide_evenly
        """

        # test that we can't upsample because initial period (12 hours) is lower than the aimed one (24 hours)
        with self.assertRaises(ValueError):
            enda.tools.resample.Resample.upsample_and_divide_evenly(
                self.perfect_df, freq="24H"
            )

        # test upsample and divide
        # test up sampling to a 6H frequency, and change of index name with perfect df
        expected_df = pd.DataFrame(
            [
                [pd.to_datetime("2021-01-01 00:00:00+01:00"), 1 / 2],
                [pd.to_datetime("2021-01-01 06:00:00+01:00"), 1 / 2],
                [pd.to_datetime("2021-01-01 12:00:00+01:00"), 1 / 2],
                [pd.to_datetime("2021-01-01 18:00:00+01:00"), 1 / 2],
                [pd.to_datetime("2021-01-02 00:00:00+01:00"), 1 / 2],
                [pd.to_datetime("2021-01-02 06:00:00+01:00"), 1 / 2],
                [pd.to_datetime("2021-01-02 12:00:00+01:00"), 1 / 2],
                [pd.to_datetime("2021-01-02 18:00:00+01:00"), 1 / 2],
                [pd.to_datetime("2021-01-03 00:00:00+01:00"), 1 / 2],
                [pd.to_datetime("2021-01-03 06:00:00+01:00"), 1 / 2],
                [pd.to_datetime("2021-01-03 12:00:00+01:00"), 1 / 2],
                [pd.to_datetime("2021-01-03 18:00:00+01:00"), 1 / 2],
                [pd.to_datetime("2021-01-04 00:00:00+01:00"), 2 / 2],
                [pd.to_datetime("2021-01-04 06:00:00+01:00"), 2 / 2],
                [pd.to_datetime("2021-01-04 12:00:00+01:00"), 2 / 2],
                [pd.to_datetime("2021-01-04 18:00:00+01:00"), 2 / 2],
                [pd.to_datetime("2021-01-05 00:00:00+01:00"), 2 / 2],
                [pd.to_datetime("2021-01-05 06:00:00+01:00"), 2 / 2],
                [pd.to_datetime("2021-01-05 12:00:00+01:00"), 2 / 2],
                [pd.to_datetime("2021-01-05 18:00:00+01:00"), 2 / 2],
                [pd.to_datetime("2021-01-06 00:00:00+01:00"), 2 / 2],
                [pd.to_datetime("2021-01-06 06:00:00+01:00"), 2 / 2],
                [pd.to_datetime("2021-01-06 12:00:00+01:00"), 2 / 2],
                [
                    pd.to_datetime("2021-01-06 18:00:00+01:00"),
                    2 / 2,
                ],  # note 18h IS present
            ],
            columns=["six_hours", "value"],
        ).set_index("six_hours")
        expected_df.index.freq = "6H"

        result_df = enda.tools.resample.Resample.upsample_and_divide_evenly(
            self.perfect_df, freq="6H", index_name="six_hours"
        )
        pd.testing.assert_frame_equal(result_df, expected_df)

        # test with a multi-index
        expected_df = pd.DataFrame(
            [
                ["name1", False, pd.to_datetime("2021-01-03 00:00:00+01:00"), 1 / 2],
                ["name1", False, pd.to_datetime("2021-01-03 06:00:00+01:00"), 1 / 2],
                ["name1", False, pd.to_datetime("2021-01-03 12:00:00+01:00"), 1 / 2],
                ["name1", False, pd.to_datetime("2021-01-03 18:00:00+01:00"), 1 / 2],
                ["name1", False, pd.to_datetime("2021-01-04 00:00:00+01:00"), 2 / 2],
                ["name1", False, pd.to_datetime("2021-01-04 06:00:00+01:00"), 2 / 2],
                ["name1", False, pd.to_datetime("2021-01-04 12:00:00+01:00"), 2 / 2],
                ["name1", False, pd.to_datetime("2021-01-04 18:00:00+01:00"), 2 / 2],
                ["name1", True, pd.to_datetime("2021-01-01 00:00:00+01:00"), 1 / 2],
                ["name1", True, pd.to_datetime("2021-01-01 06:00:00+01:00"), 1 / 2],
                ["name1", True, pd.to_datetime("2021-01-01 12:00:00+01:00"), 1 / 2],
                ["name1", True, pd.to_datetime("2021-01-01 18:00:00+01:00"), 1 / 2],
                ["name1", True, pd.to_datetime("2021-01-02 00:00:00+01:00"), 1 / 2],
                ["name1", True, pd.to_datetime("2021-01-02 06:00:00+01:00"), 1 / 2],
                ["name1", True, pd.to_datetime("2021-01-02 12:00:00+01:00"), 1 / 2],
                ["name1", True, pd.to_datetime("2021-01-02 18:00:00+01:00"), 1 / 2],
                ["name2", False, pd.to_datetime("2021-01-05 00:00:00+01:00"), 2 / 2],
                ["name2", False, pd.to_datetime("2021-01-05 06:00:00+01:00"), 2 / 2],
                ["name2", False, pd.to_datetime("2021-01-05 12:00:00+01:00"), 2 / 2],
                ["name2", False, pd.to_datetime("2021-01-05 18:00:00+01:00"), 2 / 2],
                ["name2", False, pd.to_datetime("2021-01-06 00:00:00+01:00"), 2 / 2],
                ["name2", False, pd.to_datetime("2021-01-06 06:00:00+01:00"), 2 / 2],
                ["name2", False, pd.to_datetime("2021-01-06 12:00:00+01:00"), 2 / 2],
                ["name2", False, pd.to_datetime("2021-01-06 18:00:00+01:00"), 2 / 2],
            ],
            columns=["name", "is_start_week", "time", "value"],
        ).set_index(["name", "is_start_week", "time"])

        result_df = enda.tools.resample.Resample.upsample_and_divide_evenly(
            self.perfect_multi_df, freq="6H"
        )
        pd.testing.assert_frame_equal(result_df, expected_df)

        # test error with duplicates
        with self.assertRaises(ValueError):
            enda.tools.resample.Resample.upsample_and_divide_evenly(
                self.imperfect_df, freq="6H"
            )

        # test with a single row df
        with self.assertRaises(ValueError):
            enda.tools.resample.Resample.upsample_and_divide_evenly(
                self.single_row_df, freq="1D"
            )

        # error with monthly data
        with self.assertRaises(ValueError):
            enda.tools.resample.Resample.upsample_and_divide_evenly(
                self.monthly_df, freq="1D"
            )

    def test_upsample_monthly_data_and_divide_evenly(self):
        """
        Test upsample_monthly_data_and_divide_evenly
        """

        # basic test
        input_df = pd.DataFrame(
            [
                [pd.Timestamp(2023, 1, 1), 100],
                [pd.Timestamp(2023, 2, 1), 200],
                [pd.Timestamp(2023, 3, 1), 300],
            ],
            columns=["time", "value"],
        ).set_index("time")

        index_1 = pd.date_range(
            pd.Timestamp(2023, 1, 1),
            pd.Timestamp(2023, 2, 1),
            freq="30min",
            inclusive="left",
        )
        index_2 = pd.date_range(
            pd.Timestamp(2023, 2, 1),
            pd.Timestamp(2023, 3, 1),
            freq="30min",
            inclusive="left",
        )
        index_3 = pd.date_range(
            pd.Timestamp(2023, 3, 1),
            pd.Timestamp(2023, 4, 1),
            freq="30min",
            inclusive="left",
        )
        value1 = 100 / 1488  # len(index_1)
        value2 = 200 / 1344  # len(index_2)
        value3 = 300 / 1488  # len(index_3)

        expected_output_df = pd.DataFrame(
            data=[value1 for _ in range(len(index_1))]
                 + [value2 for _ in range(len(index_2))]
                 + [value3 for _ in range(len(index_3))],
            columns=["value"],
            index=list(index_1) + list(index_2) + list(index_3),
        ).asfreq("30min")
        expected_output_df.index.name = "time"

        output_df = enda.tools.resample.Resample.upsample_monthly_data_and_divide_evenly(
            timeseries_df=input_df, freq="30min"
        )

        pd.testing.assert_frame_equal(output_df, expected_output_df)

    def test_forward_fill_last_record(self):
        """
        Test forward_fill_last_record
        """

        # test with dataframe, and gap_timedelta
        input_df = (
            pd.date_range(
                start=pd.to_datetime("2021-01-01 00:00:00+01:00"),
                end=pd.to_datetime("2021-01-02 00:00:00+01:00"),
                freq="12H",
                name="time",
            )
            .to_frame()
            .assign(value=[0.0, 1.0, 2.0])
            .set_index("time")
        )

        expected_df = pd.concat(
            [
                input_df,
                pd.DataFrame(
                    [[pd.to_datetime("2021-01-02 12:00:00+01:00"), 2]],
                    columns=["time", "value"],
                ).set_index("time"),
            ]
        )

        result_df = enda.tools.resample.Resample.forward_fill_final_record(
            timeseries_df=input_df, gap_timedelta="1D"
        )
        pd.testing.assert_frame_equal(expected_df, result_df)

        # test with excl_end_time as timestamp
        result_df = enda.tools.resample.Resample.forward_fill_final_record(
            timeseries_df=input_df,
            excl_end_time=pd.Timestamp("2021-01-03 00:00:00+01:00"),
        )
        pd.testing.assert_frame_equal(expected_df, result_df)

        # test with excl_end_time as string
        result_df = enda.tools.resample.Resample.forward_fill_final_record(
            timeseries_df=input_df, excl_end_time="2021-01-03 00:00:00+01:00"
        )
        pd.testing.assert_frame_equal(expected_df, result_df)

        # test error excl_end_time
        with self.assertRaises(ValueError):
            enda.tools.resample.Resample.forward_fill_final_record(
                timeseries_df=input_df, excl_end_time="2021-01-02 00:00:00+01:00"
            )

        # forward_fill_final_record with a cut-off frequency
        input_df = (
            pd.date_range(
                start=pd.to_datetime("2021-01-01 19:00:00+01:00"),
                end=pd.to_datetime("2021-01-01 22:00:00+01:00"),
                freq="1H",
                name="time",
            )
            .to_frame()
            .assign(value=[0.0, 1.0, 2.0, 3])
            .set_index("time")
        )

        expected_df = (
            pd.date_range(
                start=pd.to_datetime("2021-01-01 19:00:00+01:00"),
                end=pd.to_datetime("2021-01-01 23:00:00+01:00"),
                freq="1H",
                name="time",
            )
            .to_frame()
            .assign(value=[0.0, 1.0, 2.0, 3, 3])
            .set_index("time")
        )

        result_df = enda.tools.resample.Resample.forward_fill_final_record(
            timeseries_df=input_df, gap_timedelta="3H", cut_off="1D"
        )
        pd.testing.assert_frame_equal(expected_df, result_df)

        # test with an empty dataframe (and right dtypes)
        result_df = enda.tools.resample.Resample.forward_fill_final_record(self.empty_df, gap_timedelta="2H")
        pd.testing.assert_frame_equal(self.empty_df, result_df)

        # test with a one-rowed dataframe
        result_df = enda.tools.resample.Resample.forward_fill_final_record(self.single_row_df,
                                                                           gap_timedelta="30min",
                                                                           freq="10min")

        expected_df = (
            pd.date_range(
                start=pd.to_datetime("2023-01-01 00:00:00").tz_localize("Europe/Paris"),
                end=pd.to_datetime("2023-01-01 00:30:00").tz_localize("Europe/Paris"),
                freq="10min",
                name="time",
                inclusive="left"
            )
            .to_frame()
            .assign(value=[3, 3, 3])
            .set_index("time")
        )

        pd.testing.assert_frame_equal(expected_df, result_df)
