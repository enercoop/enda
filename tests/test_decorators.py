import logging
import numpy as np
import pandas as pd
import unittest

import enda.decorators


class TestDecorator(unittest.TestCase):
    def setUp(self):
        logging.captureWarnings(True)
        logging.disable(logging.ERROR)

        np.random.seed(10)

        # create a dummy dataframe with a basic DatetimeIndex: it spans
        # 10 days, with a dummy column where half values are 0, and other are 1
        dummy_single_df = pd.date_range(
            start=pd.to_datetime("2021-01-01 00:00:00+01:00").tz_convert(
                "Europe/Paris"
            ),
            end=pd.to_datetime("2021-01-10 00:00:00+01:00").tz_convert("Europe/Paris"),
            freq="D",
            tz="Europe/Paris",
            name="date",
        ).to_frame()
        dummy_single_df = dummy_single_df.set_index("date")
        dummy_single_df["value"] = [0] * 5 + [1] * 5
        self.dummy_single_df = dummy_single_df.copy(deep=True)

        # create a another dummy dataframe with random fill of values
        # to make it more complex, we will drop the last date
        dummy_other_df = dummy_single_df.copy(deep=True)
        dummy_other_df = dummy_other_df.iloc[:-1, :]
        dummy_other_df["value"] = np.random.uniform(0, 1, dummy_other_df.shape[0])

        # use both dataframes to create a multiindex dataframe
        dummy_single_df["id"] = "static"
        dummy_other_df["id"] = "rand"
        dummy_multi_df = pd.concat([dummy_single_df, dummy_other_df], axis=0)
        self.dummy_multi_df = dummy_multi_df.reset_index().set_index(["id", "date"])

    def tearDown(self):
        logging.captureWarnings(False)
        logging.disable(logging.NOTSET)

    def test_1(self):
        # a basic test with the input properly given as a

        @enda.decorators.handle_multiindex
        def mock_max_function_change_index(df, target_col):
            """
            Dummy function meant to test the decorator.
            It sets 'value' to the max, and change the index name
            """
            df[target_col] = df[target_col].max()
            df.index.name = "time"
            return df

        # test the single index
        result_df = mock_max_function_change_index(
            self.dummy_single_df, target_col="value"
        )

        self.assertTrue((result_df.index == self.dummy_single_df.index).all())
        self.assertEqual(0, result_df["value"].isna().sum())
        self.assertEqual(result_df["value"].nunique(), 1)
        self.assertAlmostEqual(result_df["value"].max(), 1, places=3)

        # test the multi index
        result_df = mock_max_function_change_index(
            self.dummy_multi_df, target_col="value"
        )

        self.assertTrue((result_df.index == self.dummy_multi_df.index).all())
        self.assertEqual(0, result_df["value"].isna().sum())
        self.assertEqual(result_df.loc[["static"], "value"].nunique(), 1)
        self.assertEqual(result_df.loc[["static"], "value"].max(), 1)
        self.assertEqual(result_df.loc[["rand"], "value"].nunique(), 1)
        self.assertAlmostEqual(
            result_df.loc[["rand"], "value"].max(), 0.77132064, places=5
        )

    def test_2(self):
        # test with a function that does not return a dataframe.
        # It must not work with a multiindex, and a TypeError is raised.

        @enda.decorators.handle_multiindex
        def mock_max_function_error(df, target_col):
            return df[target_col].max()

        with self.assertRaises(TypeError):
            mock_max_function_error(self.dummy_multi_df, target_col="value")
