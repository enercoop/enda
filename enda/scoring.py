"""This module contains methods to evaluate the performance of predictions"""

import pandas as pd


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
        :param predictions_df: A DataFrame containing the predictions that we want a score on as well as the target
        :param target: The column containing the target values to compare predictions to
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
        :return: A DataFrame describing the statistics of the error between target and predictions,
            such as mean, std and some quantiles
        """
        return self.error().describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])

    def mean_absolute_error(self) -> pd.DataFrame:
        """
        :return: A Series of the mean absolute error between each algorithm and the target,
            with algorithm names as the index
        """
        return self.absolute_error().mean()

    def mean_absolute_error_by_month(self) -> pd.DataFrame:
        """
        :return: A DataFrame of the mean absolute error grouped by month
        """
        # TODO : checker que grouper par numéro de mois est ok (1 au lieu de 2023-01)
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
