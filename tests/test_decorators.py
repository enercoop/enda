"""A module for testing the enda/decorators.py script"""

import logging
import unittest
import numpy as np
import pandas as pd

import enda.decorators


class TestDecorator(unittest.TestCase):
    def setUp(self):
        logging.captureWarnings(True)
        logging.disable(logging.ERROR)

        np.random.seed(10)

        # Create a dummy DataFrame with a basic DatetimeIndex: it spans
        # 10 days, with a dummy column where half values are 0, and other are 1
        dummy_single_df = pd.date_range(
            start=pd.Timestamp(year=2021, month=1, day=1, tz="Europe/Paris"),
            end=pd.Timestamp(year=2021, month=1, day=10, tz="Europe/Paris"),
            freq="D",
            tz="Europe/Paris",
            name="date",
        ).to_frame()
        dummy_single_df = dummy_single_df.set_index("date")
        dummy_single_df["value"] = [0] * 5 + [1] * 5
        self.dummy_single_df = dummy_single_df.copy(deep=True)

        # Create another dummy DataFrame with random fill of values
        # To make it more complex, we will drop the last date
        dummy_other_df = dummy_single_df.copy(deep=True)
        dummy_other_df = dummy_other_df.iloc[:-1, :]
        dummy_other_df["value"] = np.random.uniform(0, 1, dummy_other_df.shape[0])

        # Use both dataframes to create a MultiIndex DataFrame
        dummy_single_df["id"] = "static"
        dummy_other_df["id"] = "rand"
        dummy_multi_df = pd.concat([dummy_single_df, dummy_other_df], axis=0)
        self.dummy_multi_df = dummy_multi_df.reset_index().set_index(["id", "date"])

    def tearDown(self):
        logging.captureWarnings(False)
        logging.disable(logging.NOTSET)

    def test_handle_multiindex(self):
        """
        Test the handle_multiindex decorator on a basic function
        """

        @enda.decorators.handle_multiindex
        def mock_max_function_change_index(df, target_col):
            """
            Dummy function meant to test the decorator.
            It sets 'value' to the max, and change the index name
            """
            df[target_col] = df[target_col].max()
            df.index.name = "time"
            return df

        # Check that everything works properly with a single indexed DataFrame

        single_index_expected_output_df = pd.DataFrame(
            index=self.dummy_single_df.index.copy()
        )

        single_index_expected_output_df["value"] = 1
        single_index_expected_output_df.index.name = "time"

        single_index_output_df = mock_max_function_change_index(
            self.dummy_single_df, target_col="value"
        )

        pd.testing.assert_frame_equal(
            single_index_output_df, single_index_expected_output_df
        )

        # Check that it works with a MultiIndex DataFrame

        max_val = self.dummy_multi_df.loc[
            self.dummy_multi_df.index.get_level_values(0) == "rand", "value"
        ].max()

        multi_index_expected_output_df = pd.DataFrame(
            index=self.dummy_multi_df.index.copy()
        )

        multi_index_expected_output_df["value"] = 1
        multi_index_expected_output_df.index.names = ["id", "time"]
        multi_index_expected_output_df.loc[
            multi_index_expected_output_df.index.get_level_values(0) == "rand", "value"
        ] = max_val

        # test the multi index
        multi_index_output_df = mock_max_function_change_index(
            self.dummy_multi_df, target_col="value"
        )

        pd.testing.assert_frame_equal(
            multi_index_output_df, multi_index_expected_output_df
        )

    def test_error_raises_handle_multiindex(self):
        """
        Check all cases where handle_multiindex should raise an error
        """

        @enda.decorators.handle_multiindex
        def dummy_function(df):
            """
            Dummy function meant to test the decorator. It returns the input
            """
            return df

        # Test with a 3-level indexed DataFrame

        three_level_df = pd.DataFrame(
            data=[1, 2],
            columns=["col1"],
            index=[
                np.array([1, 2]),
                np.array(
                    [
                        pd.Timestamp(year=2023, month=1, day=1),
                        pd.Timestamp(year=2024, month=1, day=1),
                    ]
                ),
                np.array(["a", "b"]),
            ],
        )

        three_level_df.index.names = ["index1", "time", "index2"]

        with self.assertRaises(TypeError):
            dummy_function(three_level_df)

        # Test with a MultiIndex where second level is not a DatetimeIndex

        input_df = self.dummy_multi_df.swaplevel()

        with self.assertRaises(TypeError):
            dummy_function(input_df)

        # Test with a DataFrame with no index name

        no_index_name_df = pd.DataFrame(
            data=[1, 2],
            columns=["col1"],
            index=[
                np.array([1, 2]),
                np.array(
                    [
                        pd.Timestamp(year=2023, month=1, day=1),
                        pd.Timestamp(year=2024, month=1, day=1),
                    ]
                ),
            ],
        )

        with self.assertRaises(ValueError):
            dummy_function(no_index_name_df)

        # Test with a function that does not return a dataframe.
        # It must not work with a multiindex, and a TypeError is raised.

        @enda.decorators.handle_multiindex
        def mock_max_function_error(df, target_col):
            return df[target_col].max()

        with self.assertRaises(TypeError):
            mock_max_function_error(self.dummy_multi_df, target_col="value")

        # Test with a base function that takes non keyword arguments other than df

        @enda.decorators.handle_multiindex
        def test_func(df, *integers, colname="col"):
            df[colname] = 0
            for x in integers:
                df[colname] += x

            return df

        with self.assertRaises(NotImplementedError):
            test_func(self.dummy_multi_df, 1, 2)
