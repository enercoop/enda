import pandas as pd
import numpy as np
from enda.models import ModelInterface


class LinearRegression(ModelInterface):

    def __init__(self):
        self.coefficients = []

    @staticmethod
    def _concatenate_ones(x):
        """ adds a column of '1's at the beginning of x, for the intercept """
        ones = np.ones(shape=x.shape[0]).reshape(-1, 1)
        return np.concatenate((ones, x), 1)

    def train(self, df: pd.DataFrame, target_col: str):
        x = df.drop(columns=[target_col])
        y = df[target_col]

        if len(x.shape) == 1:  # reshape scalar as vector if needed
            x = x.reshape(-1, 1)

        x = LinearRegression._concatenate_ones(x)

        try:
            self.coefficients = np.linalg.inv(x.transpose().dot(x)).dot(x.transpose()).dot(y)
        except np.linalg.LinAlgError:
            # x is not cannot be inverted, keep rows
            x = np.matrix(x)
            lambdas, v = np.linalg.eig(x.T)
            x = x[lambdas == 0, :]
            self.coefficients = np.linalg.inv(x.transpose().dot(x)).dot(x.transpose()).dot(y)

    def predict(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        x = df if len(df.shape) > 1 else df.reshape(-1, 1)  # reshape scalar as vector if needed
        x = LinearRegression._concatenate_ones(x)

        a = (x * self.coefficients).sum(axis=1)
        s = pd.Series(a, name=target_col, index=df.index)  # pandas series with correct name and index
        return s.to_frame()

