import pandas

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

    def train(self, df: pandas.DataFrame, target_col: str):
        x = df.drop(columns=[target_col])
        y = df[target_col]
        self.model.fit(x, y)

    def predict(self, df: pandas.DataFrame, target_col: str):
        a = self.model.predict(df)  # numpy array
        s = pandas.Series(
            a, name=target_col, index=df.index
        )  # pandas series with correct name and index
        return s.to_frame()
