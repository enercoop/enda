"""This script contains a wrapper for scikit-learn estimators"""

import numpy as np
import pandas as pd

from sklearn.linear_model import __all__ as linear_model_sklearn_list
from sklearn.ensemble import __all__ as ensemble_model_sklearn_list
from sklearn.tree import __all__ as tree_model_sklearn_list

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
        super().__init__()
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

        # store the training
        self._training_df = df
        self._target = target_col

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

    def get_feature_importance(self) -> pd.Series:
        """
        Return the feature's importance once a model has been trained.
        This function only work if the wrapped scikit model is a linear model or a tree model.
        In the case it's a linear model, it returns the coefficients of the fit, standardized using the standard
        deviation of the input features (it's coherent with H2O).
        If it's a tree, the feature importance is directly calculated by the algorithm.
        :return: a series that contain the percentage of importance for each variable
        """

        # if _training_df or _target is None, that means the model has not been trained
        if self._training_df is None or self._target is None:
            raise ValueError("The model must be trained before calling this method.")

        if self.get_model_name() in linear_model_sklearn_list:
            # first case, we're faced to a linear model.
            # in that case, the variable importance can be assimilated to the coefficients of the regression
            # once they have been standardized (that's what H2O does)
            feature_importance_series = np.abs(self.model.coef_) * \
                                        (
                                            self._training_df
                                            .drop(columns=[self._target])
                                            .std()
                                        )

            # it must be turned to percentages
            feature_importance_series /= feature_importance_series.sum()

        elif self.get_model_name() in ensemble_model_sklearn_list + tree_model_sklearn_list:
            # in that case, the tree itself is able to compute the variable importance
            feature_importance_series = pd.Series(
                data=self.model.feature_importances_,
                index=self._training_df.drop(columns=[self._target]).columns,
                dtype=float
            )

        else:
            # neural networks for instance
            raise NotImplementedError()

        # set name
        feature_importance_series.name = 'variable_importance_pct'

        # return sorted values to fit h2o behaviour
        return feature_importance_series.sort_values(ascending=False)
