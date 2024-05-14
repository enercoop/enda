"""This module contains methods to evaluate the performance of predictions"""

import numpy as np
import pandas as pd
from typing import Union
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def _root_mean_squared_error(x1, x2):
    return np.sqrt(mean_squared_error(x1, x2))


METRICS_FUNCTION_DICT = {
    "max_error": max_error,
    "mae": mean_absolute_error,
    "mape": mean_absolute_percentage_error,
    "mse": mean_squared_error,
    "rmse": _root_mean_squared_error,
    "r2": r2_score
}


class Scoring:
    """
    A class to help scoring algorithms
    predictions_df must include the 'target' column and the predictions in all other columns
    """

    def __init__(
        self, predictions_df: pd.DataFrame, target: str, normalizing_col: str = None
    ):
        """
        Initialize the scoring object
        :param predictions_df: A DataFrame containing the predictions to be scored in columns, as well as a column with the target. The index must be a datetime-index.
        :param target: The name of the column which contains the target values against which predictions are scored.
        :param normalizing_col: Optional, a normalizing column for computing normalized absolute error
        """

        self.predictions_df = predictions_df
        self.target = target
        self.normalizing_col = normalizing_col
        if self.target not in self.predictions_df.columns:
            raise ValueError(
                f"target={self.target} must be in predictions_df columns : {self.predictions_df}"
            )
        if len(self.predictions_df.columns) < 2:
            raise ValueError(
                "predictions_df must have at least 2 columns (1 target and 1 prediction)"
            )

        algo_names = list(
            c
            for c in self.predictions_df.columns
            if c not in [self.target, self.normalizing_col]
        )

        error_df = self.predictions_df.copy(deep=True)
        for x in algo_names:
            error_df[x] = error_df[x] - error_df[self.target]
        error_df = error_df[algo_names]
        self.error_df = error_df

        self.pct_error_df = (
            self.error_df.div(self.predictions_df[self.target], axis=0) * 100
        )

    def error(self) -> pd.DataFrame:
        """
        :return: A DataFrame of the error between the prediction and the target (prediction - target)
        """
        return self.error_df

    def mean_error(self) -> pd.Series:
        """
        :return: A Series of the mean error between each algorithm and the target, with algorithm names as the index
        """
        return self.error().mean()

    def absolute_error(self) -> pd.DataFrame:
        """
        :return: A DataFrame of the absolute error between the prediction and the target
        """
        return self.error().abs()

    def absolute_error_statistics(self) -> pd.DataFrame:
        """
        :return: A DataFrame describing the statistics of the absolute error between target and predictions,
            such as mean, std and some quantiles
        """
        return self.absolute_error().describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])

    def mean_absolute_error(self) -> pd.Series:
        """
        :return: A Series of the mean absolute error between each algorithm and the target,
            with algorithm names as the index
        """
        return self.absolute_error().mean()

    def mean_absolute_error_by_month(self) -> pd.DataFrame:
        """
        :return: A DataFrame of the mean absolute error grouped by month
        """
        abs_error = self.absolute_error()
        return abs_error.groupby(abs_error.index.month).mean()

    def percentage_error(self) -> pd.DataFrame:
        """
        :return: A DataFrame of the relative error in percentage between prediction and target
        """
        return self.pct_error_df

    def absolute_percentage_error(self) -> pd.DataFrame:
        """
        :return: A DataFrame of the absolute relative error in percentage between prediction and target
        """
        return self.percentage_error().abs()

    def absolute_percentage_error_statistics(self):
        """
        :return: A DataFrame describing the statistics of the absolute relative error in percentage between target
            and predictions, such as mean, std and some quantiles
        """
        return self.absolute_percentage_error().describe(
            percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]
        )

    def mean_absolute_percentage_error(self) -> pd.Series:
        """
        :return: A Series of the mean absolute percentage error between each algorithm and the target,
            with algorithm names as the index
        """
        return self.absolute_percentage_error().mean()

    def mean_absolute_percentage_error_by_month(self) -> pd.DataFrame:
        """
        :return: A DataFrame of the mean absolute percentage error grouped by month
        """
        abs_percentage_error = self.absolute_percentage_error()
        return abs_percentage_error.groupby(abs_percentage_error.index.month).mean()

    def normalized_absolute_error(self) -> pd.DataFrame:
        """
        :return: A DataFrame of the absolute error between the predictions and the target, normalized by the
            normalized_col (if it isn't defined, will raise en error)
        """
        if self.normalizing_col is None:
            raise ValueError(
                "Cannot use this function without defining normalizing_col in Scoring"
            )
        return self.error_df.abs().div(
            self.predictions_df[self.normalizing_col], axis=0
        )

    @staticmethod
    def compute_loss(predicted_df: pd.DataFrame,
                     actual_df: Union[pd.DataFrame, pd.Series],
                     score_list: list[str] = None) -> pd.Series:
        """
        Compute the loss (i.e. the score) between a model prediction and the actual data
        :param predicted_df: the result of the prediction
        :param actual_df: the actual target data
        :param score_list: the statistics to consider. Either 'max_error', 'mae', 'rmse', 'r2', 'mape', 'mse'.
            Defaults to 'rmse'.
        :return: a series that contains for each statistics the score of the model on the training set
        """

        # default is rmse
        if score_list is None:
            score_list = ['rmse']

        # check that scores are allowed
        for score in score_list:
            if score not in METRICS_FUNCTION_DICT:
                raise ValueError(f"Score must be one of {METRICS_FUNCTION_DICT.keys()} but got {score}")

        # we have to compute on the training set the score for each of the method chosen
        scoring_result_list = []
        for score in score_list:
            method = METRICS_FUNCTION_DICT[score]
            result_score = method(actual_df, predicted_df)
            scoring_result_list.append(result_score)

        return pd.Series(data=scoring_result_list, index=score_list)
