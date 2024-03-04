"""A module for testing the functions of enda/decorators.py"""

import logging
import unittest
import warnings
import pandas as pd

import enda.decorators


class TestDecorator(unittest.TestCase):
    """This class aims at testing the decorator functions in enda/decorators.py"""

    def setUp(self):
        logging.captureWarnings(True)
        logging.disable(logging.ERROR)

        # datetime index / series ; unordered and duplicates
        self.test_dti = pd.DatetimeIndex(
            ["2024-01-01", "2023-01-01", "2023-01-01", "2025-01-01"]
        )
        self.test_str_series = pd.Series(
            ["2024-01-01", "2023-01-01", "2023-01-01", "2025-01-01"]
        )
        self.test_series = self.test_dti.to_series().reset_index(drop=True)

        # single-indexed dataframe
        self.single_df = (
            pd.date_range(
                start=pd.to_datetime("2021-01-01"),
                end=pd.to_datetime("2021-01-06"),
                freq="D",
            )
            .to_frame(name="date")
            .set_index("date")
            .assign(value=[0] * 3 + [1] * 3)
        )

        # multi-indexed dataframe
        other_df = (
            self.single_df.copy()
            .iloc[:-1, :]  # to make it more complex, we will drop the last date
            .assign(value=[-1] * 3 + [-2] * 2)
        )
        self.multi_df = (
            pd.concat(
                [
                    self.single_df.copy().assign(id="first"),
                    other_df.assign(id="second"),
                ],
                axis=0,
            )
            .reset_index()
            .set_index(["id", "date"])
        )

        # 3-levels multi-indexed dataframe
        self.multi_levels_df = (
            self.multi_df.copy()
            .assign(is_before_3=self.multi_df.index.get_level_values("date").day < 3)
            .reset_index()
            .set_index(["id", "is_before_3", "date"])
        )

    def tearDown(self):
        logging.captureWarnings(False)
        logging.disable(logging.NOTSET)

    def test_handle_series_as_datetimeindex_return_input_type_is_true(self):
        """
        Test with a function working only on datetimeindex, performing an action on itself
        We set return_input_type to True
        """

        @enda.decorators.handle_series_as_datetimeindex(
            arg_name="time_series", return_input_type=True
        )
        def tz_localize_time_series(time_series: pd.DatetimeIndex) -> pd.DatetimeIndex:
            return time_series.tz_localize(
                "Europe/Paris"
            )  # would require dt. to work with Series, usually

        # test with dti
        result_dti = tz_localize_time_series(time_series=self.test_dti)
        expected_dti = pd.DatetimeIndex(
            [
                "2024-01-01 00:00:00+01:00",
                "2023-01-01 00:00:00+01:00",
                "2023-01-01 00:00:00+01:00",
                "2025-01-01 00:00:00+01:00",
            ],
            dtype="datetime64[ns, Europe/Paris]",
        )
        pd.testing.assert_index_equal(result_dti, expected_dti)

        # test with series
        result_series = tz_localize_time_series(time_series=self.test_series)
        expected_series = pd.Series(
            [
                "2024-01-01 00:00:00+01:00",
                "2023-01-01 00:00:00+01:00",
                "2023-01-01 00:00:00+01:00",
                "2025-01-01 00:00:00+01:00",
            ],
            dtype="datetime64[ns, Europe/Paris]",
        )
        pd.testing.assert_series_equal(result_series, expected_series)

        # test with dti and args instead of kwargs
        result_dti = tz_localize_time_series(self.test_dti)
        pd.testing.assert_index_equal(result_dti, expected_dti)

        # test with series and args instead of kwargs
        result_series = tz_localize_time_series(self.test_series)
        pd.testing.assert_series_equal(result_series, expected_series)

    def test_handle_series_as_datetimeindex_return_input_type_is_false(self):
        """
        Test with a function working only on datetimeindex, performing an action on itself
        We set return_input_type to False
        """

        # compared to previous example, return_input_type is set to False
        @enda.decorators.handle_series_as_datetimeindex(
            arg_name="dti", return_input_type=False
        )
        def tz_localize_time_series(dti: pd.DatetimeIndex) -> pd.DatetimeIndex:
            return dti.tz_localize("Europe/Paris")

        # test with dti
        result_dti = tz_localize_time_series(dti=self.test_dti)
        expected_dti = pd.DatetimeIndex(
            [
                "2024-01-01 00:00:00+01:00",
                "2023-01-01 00:00:00+01:00",
                "2023-01-01 00:00:00+01:00",
                "2025-01-01 00:00:00+01:00",
            ],
            dtype="datetime64[ns, Europe/Paris]",
        )
        pd.testing.assert_index_equal(result_dti, expected_dti)

        # test with series
        # shall return a dti
        result_dti = tz_localize_time_series(dti=self.test_series)
        pd.testing.assert_index_equal(result_dti, expected_dti)

        # test with dti and args instead of kwargs
        result_dti = tz_localize_time_series(self.test_dti)
        pd.testing.assert_index_equal(result_dti, expected_dti)

        # test with series and args instead of kwargs
        # shall return a dti
        result_dti = tz_localize_time_series(self.test_series)
        pd.testing.assert_index_equal(result_dti, expected_dti)

    def test_handle_series_as_datetimeindex_errors(self):
        """
        Test errors in handle_series_as_datetimeindex
        """

        @enda.decorators.handle_series_as_datetimeindex(
            arg_name="dti", return_input_type=False
        )
        def tz_localize_time_series(dti: pd.DatetimeIndex) -> pd.DatetimeIndex:
            return dti.tz_localize("Europe/Paris")

        # test if dti is not a series convertible to datetimeindex
        with self.assertRaises(ValueError):
            tz_localize_time_series(2)

    def test_handle_multiindex_return_float(self):
        """
        Test handle_multiindex in the case it returns a float (convertible to series)
        """

        @enda.decorators.handle_multiindex(arg_name="df")
        def compute_mean_as_float(df: pd.DataFrame):
            return df.mean().squeeze()

        # test over single_df
        result = compute_mean_as_float(self.single_df)
        self.assertEqual(result, 0.5)

        # test over multi_df
        result_df = compute_mean_as_float(self.multi_df)
        expected_df = pd.DataFrame(
            data=[["first", 0.5], ["second", -1.4]], columns=["id", 0]
        ).set_index("id")
        pd.testing.assert_frame_equal(expected_df, result_df)

        # test over multi_levels_df
        result_df = compute_mean_as_float(self.multi_levels_df)
        expected_df = pd.DataFrame(
            data=[
                ["first", True, 0.0],
                ["first", False, 0.75],
                ["second", True, -1.0],
                ["second", False, -5 / 3.0],
            ],
            columns=["id", "is_before_3", 0],
        ).set_index(["id", "is_before_3"])
        pd.testing.assert_frame_equal(expected_df, result_df)

    def test_handle_multiindex_return_series(self):
        """
        Test handle_multiindex in the case it returns a series (single-valued)
        """

        @enda.decorators.handle_multiindex(arg_name="df")
        def compute_mean_as_series(df: pd.DataFrame):
            return df.mean()

        # test over single_df
        result_df = compute_mean_as_series(self.single_df)
        expected_df = pd.Series(data=[0.5], index=["value"])
        pd.testing.assert_series_equal(expected_df, result_df)

        # test over multi_df
        result_df = compute_mean_as_series(self.multi_df)
        expected_df = pd.DataFrame(
            data=[["first", 0.5], ["second", -1.4]],
            columns=["id", "value"],  # note a slight difference with previous test
        ).set_index("id")
        pd.testing.assert_frame_equal(expected_df, result_df)

        # test over multi_levels_df
        result_df = compute_mean_as_series(self.multi_levels_df)
        expected_df = pd.DataFrame(
            data=[
                ["first", True, 0.0],
                ["first", False, 0.75],
                ["second", True, -1.0],
                ["second", False, -5 / 3.0],
            ],
            columns=["id", "is_before_3", "value"],
        ).set_index(["id", "is_before_3"])
        pd.testing.assert_frame_equal(expected_df, result_df)

    def test_handle_multiindex_return_same_index(self):
        """
        Test handle_multiindex in the case it returns a dataframe with the same index
        """

        @enda.decorators.handle_multiindex(arg_name="test_df")
        def add_one(test_df: pd.DataFrame, col_name="value"):
            test_df = test_df.copy()
            test_df[col_name] += 1
            return test_df

        # test over single_df
        result_df = add_one(self.single_df, col_name="value")
        expected_df = (
            pd.date_range(
                start=pd.to_datetime("2021-01-01"),
                end=pd.to_datetime("2021-01-06"),
                freq="D",
            )
            .to_frame(name="date")
            .set_index("date")
            .assign(value=[1] * 3 + [2] * 3)
        )
        pd.testing.assert_frame_equal(result_df, expected_df)

        # test over multi_df
        result_df = add_one(col_name="value", test_df=self.multi_df)  # also test kwargs
        expected_df = self.multi_df.copy()
        expected_df["value"] = expected_df["value"] + 1
        pd.testing.assert_frame_equal(result_df, expected_df)

        # test over multi_levels_df
        result_df = add_one(self.multi_levels_df)
        expected_df = self.multi_levels_df.copy()
        expected_df["value"] = expected_df["value"] + 1
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_handle_multiindex_return_new_index(self):
        """
        Test handle_multiindex with a function that returns a dataframe with a new index
        """

        @enda.decorators.handle_multiindex(arg_name="timeseries_df")
        def twelve_hours_resampler(timeseries_df: pd.DataFrame):
            return timeseries_df.resample("12H").ffill()

        # test over single_df
        result_df = twelve_hours_resampler(self.single_df)
        expected_df = (
            pd.date_range(
                start=pd.to_datetime("2021-01-01"),
                end=pd.to_datetime("2021-01-06"),
                freq="12H",
            )
            .to_frame(name="date")
            .set_index("date")
            .assign(value=[0] * 6 + [1] * 5)
        )
        expected_df.index.freq = "12H"
        pd.testing.assert_frame_equal(result_df, expected_df)

        # test over multi_df
        result_df = twelve_hours_resampler(timeseries_df=self.multi_df)
        expected_df = pd.concat(
            [
                (
                    pd.date_range(
                        start=pd.to_datetime("2021-01-01"),
                        end=pd.to_datetime("2021-01-06"),
                        freq="12H",
                    )
                    .to_frame(name="date")
                    .assign(value=[0] * 6 + [1] * 5)
                    .assign(id="first")
                ),
                (
                    pd.date_range(
                        start=pd.to_datetime("2021-01-01"),
                        end=pd.to_datetime("2021-01-05"),
                        freq="12H",
                    )
                    .to_frame(name="date")
                    .assign(value=[-1] * 6 + [-2] * 3)
                    .assign(id="second")
                ),
            ]
        ).set_index(["id", "date"])

        pd.testing.assert_frame_equal(result_df, expected_df)

        # test over multi_levels_df
        result_df = twelve_hours_resampler(self.multi_levels_df)
        expected_df = pd.concat(
            [
                (
                    pd.date_range(
                        start=pd.to_datetime("2021-01-01"),
                        end=pd.to_datetime("2021-01-02"),
                        freq="12H",
                    )
                    .to_frame(name="date")
                    .assign(value=[0] * 3)
                    .assign(id="first")
                    .assign(is_before_3=True)
                ),
                (
                    pd.date_range(
                        start=pd.to_datetime("2021-01-03"),
                        end=pd.to_datetime("2021-01-06"),
                        freq="12H",
                    )
                    .to_frame(name="date")
                    .assign(value=[0] * 2 + [1] * 5)
                    .assign(id="first")
                    .assign(is_before_3=False)
                ),
                (
                    pd.date_range(
                        start=pd.to_datetime("2021-01-01"),
                        end=pd.to_datetime("2021-01-02"),
                        freq="12H",
                    )
                    .to_frame(name="date")
                    .assign(value=[-1] * 3)
                    .assign(id="second")
                    .assign(is_before_3=True)
                ),
                (
                    pd.date_range(
                        start=pd.to_datetime("2021-01-03"),
                        end=pd.to_datetime("2021-01-05"),
                        freq="12H",
                    )
                    .to_frame(name="date")
                    .assign(value=[-1] * 2 + [-2] * 3)
                    .assign(id="second")
                    .assign(is_before_3=False)
                ),
            ]
        ).set_index(["id", "is_before_3", "date"])
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_handle_multiindex_args_kwargs_combination(self):
        """
        Test the same decorated function called with different combination
        of args and kwargs.
        """

        # simple function with three arguments that returns the length of axis in df
        @enda.decorators.handle_multiindex(arg_name="df")
        def return_len_df(
            df: pd.DataFrame, axis: int, reverse_sign: bool = False
        ) -> int:
            if reverse_sign:
                return -1 * df.shape[axis]
            return df.shape[axis]

        # expected result
        expected_df = pd.DataFrame(
            [["first", 6], ["second", 5]], columns=["id", 0]
        ).set_index("id")

        # ok, well-defined function if all kwargs
        result_df = return_len_df(df=self.multi_df, axis=0, reverse_sign=False)
        pd.testing.assert_frame_equal(result_df, expected_df)

        # ok, well-defined function if two kwargs and default (even in bad order)
        result_df = return_len_df(axis=0, df=self.multi_df)
        pd.testing.assert_frame_equal(result_df, expected_df)

        # ok, well-defined function if single args
        result_df = return_len_df(self.multi_df, axis=0)
        pd.testing.assert_frame_equal(result_df, expected_df)

        # ok, well-defined function if two args and default
        result_df = return_len_df(self.multi_df, 0)
        pd.testing.assert_frame_equal(result_df, expected_df)

        # ok, well-defined function if three args
        result_df = return_len_df(self.multi_df, 0, False)
        pd.testing.assert_frame_equal(result_df, expected_df)

        # same function, but invert df and axis arguments (BAD PRACTICE)
        @enda.decorators.handle_multiindex(arg_name="df")
        def alternative_return_len_df(
            axis: int, df: pd.DataFrame, reverse_sign: bool = False
        ) -> int:
            if reverse_sign:
                return -1 * df.shape[axis]
            return df.shape[axis]

        # ok, well-defined function if all kwargs
        result_df = alternative_return_len_df(axis=0, df=self.multi_df)
        pd.testing.assert_frame_equal(result_df, expected_df)

        # ok, well-defined function if kwargs at least for arg_name
        result_df = alternative_return_len_df(0, df=self.multi_df)
        pd.testing.assert_frame_equal(result_df, expected_df)

        # error if no kwargs and dataframe not first argument
        with self.assertRaises(ValueError):
            alternative_return_len_df(0, self.multi_df)

    def test_handle_multiindex_errors(self):
        """
        Test several errors
        """

        @enda.decorators.handle_multiindex(arg_name="df")
        def return_len_df(
            df: pd.DataFrame, axis: int, reverse_sign: bool = False
        ) -> int:
            if reverse_sign:
                return -1 * df.shape[axis]
            return df.shape[axis]

        # not a dataframe is passed
        with self.assertRaises(ValueError):
            return_len_df(0, 1)

        # time index not named
        test_no_time_index_name_df = self.multi_df.copy()
        test_no_time_index_name_df.index.names = ["id", None]
        with self.assertRaises(ValueError):
            return_len_df(test_no_time_index_name_df, 1)

        # first index not named
        test_no_time_index_name_df = self.multi_df.copy()
        test_no_time_index_name_df.index.names = [None, "time"]
        with self.assertRaises(ValueError):
            return_len_df(test_no_time_index_name_df, 1)

    def test_warning_deprecated_name(self):
        """
        Test warning_deprecated_name
        """

        # reactivate warnings
        logging.captureWarnings(False)

        # decorate a function
        @enda.decorators.warning_deprecated_name(
            namespace_name="Old",
            new_namespace_name="New",
            new_function_name="alternative_return_len_df",
        )
        def return_len_df(df: pd.DataFrame, axis: int) -> int:
            return df.shape[axis]

        with warnings.catch_warnings(record=True) as caught_warning:
            warnings.simplefilter("always")
            return_len_df(self.single_df, 0)  # trigger a warning
            self.assertEqual(len(caught_warning), 1)
            self.assertIs(caught_warning[-1].category, DeprecationWarning)
            self.assertEqual(
                str(caught_warning[-1].message),
                "return_len_df in Old is deprecated, use alternative_return_len_df from New instead.",
            )
        # inactivate warnings
        logging.captureWarnings(True)
