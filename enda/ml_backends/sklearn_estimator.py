import pandas
from enda.models import ModelInterface


class SklearnEstimator(ModelInterface):
    """
    This is a simple wrapper around any Scikit-learn estimator (or anything with the same fit/predict methods)
    It makes it easier to deal pandas time-series dataframes as input and output.

    SklearnEstimator implements enda's ModelInterface.
    """

    def __init__(self, sklearn_estimator):
        # will error out if the object passed is not a correct sklearn estimator
        # check_estimator(sklearn_estimator)
        self.model = sklearn_estimator

    def train(self, df: pandas.DataFrame, target_col: str):
        x = df.drop(columns=[target_col])
        y = df[target_col]
        self.model.fit(x, y)

    def predict(self, df: pandas.DataFrame, target_col: str):
        a = self.model.predict(df)  # numpy array
        s = pandas.Series(a, name=target_col, index=df.index)  # pandas series with correct name and index
        return s.to_frame()
