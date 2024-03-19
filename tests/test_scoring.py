"""A module for testing the functions in enda/scoring.py"""

import unittest
import logging
import pandas as pd
from enda.scoring import Scoring


class TestScoring(unittest.TestCase):
    """This class aims at testing the functions of the Scoring class in enda/scoring.py"""

    def setUp(self):
        logging.disable(logging.ERROR)

        self.index = [
            pd.Timestamp(2023, 1, 1),
            pd.Timestamp(2023, 2, 1),
            pd.Timestamp(2023, 3, 1),
            pd.Timestamp(2023, 4, 1),
            pd.Timestamp(2024, 1, 1),
        ]

        self.prediction_df = pd.DataFrame(
            data=[
                {"target": 10, "algo_1": 8, "algo_2": 9.4},
                {"target": 11.5, "algo_1": 10, "algo_2": 11.2},
                {"target": 12, "algo_1": 12.5, "algo_2": 12.3},
                {"target": 14, "algo_1": 13.8, "algo_2": 14.2},
                {"target": 8.7, "algo_1": 8.7, "algo_2": 15.6},
            ],
            index=self.index,
        )

        self.scoring = Scoring(predictions_df=self.prediction_df, target="target")

    def tearDown(self):
        logging.captureWarnings(False)
        logging.disable(logging.NOTSET)

    def test_init_scoring(self):
        """Test the initialisation method of the Scoring class"""

        # Check when target column is missing

        with self.assertRaises(ValueError):
            Scoring(predictions_df=self.prediction_df, target="missed_target")

        # Check when there isn't enough columns

        smol_df = self.prediction_df.copy()
        smol_df.drop(["algo_1", "algo_2"], axis=1, inplace=True)

        with self.assertRaises(ValueError):
            Scoring(predictions_df=smol_df, target="target")

    def test_error(self):
        """Test the error function"""

        expected_error_df = pd.DataFrame(
            data=[
                {"algo_1": -2, "algo_2": -0.6},
                {"algo_1": -1.5, "algo_2": -0.3},
                {"algo_1": 0.5, "algo_2": 0.3},
                {"algo_1": -0.2, "algo_2": 0.2},
                {"algo_1": 0, "algo_2": 6.9},
            ],
            index=self.index,
        )

        pd.testing.assert_frame_equal(self.scoring.error(), expected_error_df)

    def test_mean_error(self):
        """Test the mean_error function"""

        expected_mean_error_series = pd.Series({"algo_1": -0.64, "algo_2": 1.3})

        pd.testing.assert_series_equal(
            self.scoring.mean_error(), expected_mean_error_series
        )

    def test_absolute_error(self):
        """Test the error function"""

        expected_abs_error_df = pd.DataFrame(
            data=[
                {"algo_1": 2, "algo_2": 0.6},
                {"algo_1": 1.5, "algo_2": 0.3},
                {"algo_1": 0.5, "algo_2": 0.3},
                {"algo_1": 0.2, "algo_2": 0.2},
                {"algo_1": 0, "algo_2": 6.9},
            ],
            index=self.index,
        )

        pd.testing.assert_frame_equal(
            self.scoring.absolute_error(), expected_abs_error_df
        )

    def test_absolute_error_statistics(self):
        """Test the absolute_error_statistics function"""

        expected_abs_stat_df = pd.DataFrame(
            data=[
                {"algo_1": 5, "algo_2": 5},
                {"algo_1": 0.84, "algo_2": 1.66},
                {"algo_1": 0.867756, "algo_2": 2.933087},
                {"algo_1": 0, "algo_2": 0.2},
                {"algo_1": 0.5, "algo_2": 0.3},
                {"algo_1": 1.5, "algo_2": 0.6},
                {"algo_1": 1.8, "algo_2": 4.38},
                {"algo_1": 1.9, "algo_2": 5.64},
                {"algo_1": 1.98, "algo_2": 6.648},
                {"algo_1": 2, "algo_2": 6.9},
            ],
            index=[
                "count",
                "mean",
                "std",
                "min",
                "50%",
                "75%",
                "90%",
                "95%",
                "99%",
                "max",
            ],
        )

        pd.testing.assert_frame_equal(
            self.scoring.absolute_error_statistics(), expected_abs_stat_df
        )

    def test_mean_absolute_error(self):
        """Test the mean_absolute_error function"""

        expected_mean_abs_error_series = pd.Series({"algo_1": 0.84, "algo_2": 1.66})

        pd.testing.assert_series_equal(
            self.scoring.mean_absolute_error(), expected_mean_abs_error_series
        )

    def test_mean_absolute_error_by_month(self):
        """Test the mean_absolute_error_by_month function"""

        expected_mean_abs_err_by_month_df = pd.DataFrame(
            data=[
                {"algo_1": 1, "algo_2": 3.75},
                {"algo_1": 1.5, "algo_2": 0.3},
                {"algo_1": 0.5, "algo_2": 0.3},
                {"algo_1": 0.2, "algo_2": 0.2},
            ],
            index=range(1, 5),
        )

        pd.testing.assert_frame_equal(
            self.scoring.mean_absolute_error_by_month(),
            expected_mean_abs_err_by_month_df,
        )

    def test_percentage_error(self):
        """Test the percentage_error function"""

        expected_pct_error_df = pd.DataFrame(
            data=[
                {"algo_1": -20, "algo_2": -6},
                {"algo_1": -13.04, "algo_2": -2.61},
                {"algo_1": 4.17, "algo_2": 2.5},
                {"algo_1": -1.43, "algo_2": 1.43},
                {"algo_1": 0, "algo_2": 79.31},
            ],
            index=self.index,
        )

        pd.testing.assert_frame_equal(
            self.scoring.percentage_error(), expected_pct_error_df, atol=0.01
        )

    def test_absolute_percentage_error(self):
        """Test the absolute_percentage_error function"""

        expected_abs_pct_error_df = pd.DataFrame(
            data=[
                {"algo_1": 20, "algo_2": 6},
                {"algo_1": 13.04, "algo_2": 2.61},
                {"algo_1": 4.17, "algo_2": 2.5},
                {"algo_1": 1.43, "algo_2": 1.43},
                {"algo_1": 0, "algo_2": 79.31},
            ],
            index=self.index,
        )

        pd.testing.assert_frame_equal(
            self.scoring.absolute_percentage_error(),
            expected_abs_pct_error_df,
            atol=0.01,
        )

    def test_absolute_percentage_error_statistics(self):
        """Test the absolute_percentage_error_statistics function"""

        expected_abs_pct_stat_df = pd.DataFrame(
            data=[
                {"algo_1": 5, "algo_2": 5},
                {"algo_1": 7.73, "algo_2": 18.37},
                {"algo_1": 8.53, "algo_2": 34.11},
                {"algo_1": 0, "algo_2": 1.43},
                {"algo_1": 4.17, "algo_2": 2.61},
                {"algo_1": 13.04, "algo_2": 6},
                {"algo_1": 17.22, "algo_2": 49.99},
                {"algo_1": 18.61, "algo_2": 64.65},
                {"algo_1": 19.72, "algo_2": 76.38},
                {"algo_1": 20, "algo_2": 79.31},
            ],
            index=[
                "count",
                "mean",
                "std",
                "min",
                "50%",
                "75%",
                "90%",
                "95%",
                "99%",
                "max",
            ],
        )

        pd.testing.assert_frame_equal(
            self.scoring.absolute_percentage_error_statistics(),
            expected_abs_pct_stat_df,
            atol=0.01,
        )

    def test_mean_absolute_percentage_error(self):
        """Test the mean_absolute_error function"""

        expected_mean_abs_pct_error_series = pd.Series(
            {"algo_1": 7.73, "algo_2": 18.37}
        )

        pd.testing.assert_series_equal(
            self.scoring.mean_absolute_percentage_error(),
            expected_mean_abs_pct_error_series,
            atol=0.01,
        )

    def test_mean_absolute_percentage_error_by_month(self):
        """Test the mean_absolute_percentage_error_by_month function"""

        expected_mean_abs_pct_err_by_month_df = pd.DataFrame(
            data=[
                {"algo_1": 10, "algo_2": 42.66},
                {"algo_1": 13.04, "algo_2": 2.61},
                {"algo_1": 4.17, "algo_2": 2.5},
                {"algo_1": 1.43, "algo_2": 1.43},
            ],
            index=range(1, 5),
        )

        pd.testing.assert_frame_equal(
            self.scoring.mean_absolute_percentage_error_by_month(),
            expected_mean_abs_pct_err_by_month_df,
            atol=0.01,
        )

    def test_normalized_absolute_error(self):
        """Test the normalized_absolute_error_function"""

        # Check when no normalization_col has been specified

        with self.assertRaises(ValueError):
            self.scoring.normalized_absolute_error()

        # Check the result when normalization_col is there

        norm_df = self.prediction_df.copy()
        norm_df["norm"] = range(1, 6)

        scoring_norm = Scoring(
            predictions_df=norm_df, target="target", normalizing_col="norm"
        )

        expected_norm_err_df = pd.DataFrame(
            data=[
                {"algo_1": 2, "algo_2": 0.6},
                {"algo_1": 0.75, "algo_2": 0.15},
                {"algo_1": 0.16666666666, "algo_2": 0.1},
                {"algo_1": 0.05, "algo_2": 0.05},
                {"algo_1": 0, "algo_2": 1.38},
            ],
            index=self.index,
        )

        pd.testing.assert_frame_equal(
            scoring_norm.normalized_absolute_error(), expected_norm_err_df
        )
