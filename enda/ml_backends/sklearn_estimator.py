"""This script contains a wrapper for scikit-learn estimators"""

import pandas as pd

from enda.estimators import EndaEstimator


class EndaSklearnEstimator(EndaEstimator):
    """
    This is a simple wrapper around any Scikit-learn estimator.
    It makes it easier to deal pandas time-series dataframes as input and output.
    """

    def __init__(self, sklearn_estimator):
        """
        Like in scikit-learn we use duck typing here, so we don't check the type of argument 'sklearn_estimator'
        """

        self.model = sklearn_estimator

    def train(self, df: pd.DataFrame, target_col: str):
        """
        Train a scikit-learn-based model from an input dataframe with features and a target column
        :param df: the input dataframe
        :param target_col: the target column name
        """
        x = df.drop(columns=[target_col])
        y = df[target_col]
        self.model.fit(x, y)

    def predict(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Predict from a scikit-learn-based trained model using an input dataframe with features
        :param df: the input dataframe
        :param target_col: the target column name
        :return: a single-column dataframe with the predicted target
        """
        a = self.model.predict(df)  # numpy array
        s = pd.Series(
            a, name=target_col, index=df.index
        )  # pandas series with correct name and index
        return s.to_frame()

    def get_model_name(self) -> str:
        """
        Return the scikit-learn model name instead of EndaSklearnEstimator
        """
        return self.model.__class__.__name__

    def get_model_params(self) -> dict:
        """
        Return a dict with the model name and the model hyperparameters
        """
        return {self.get_model_name(): self.model.get_params()}
