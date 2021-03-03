from sklearn.linear_model import LinearRegression
import pandas
from enda.models import ModelInterface


class SKLearnLinearRegression(ModelInterface):
    """ The simplest model, serving as an example of an implementation of ModelInterface """

    def __init__(self):
        self.lin_reg_model = None

    def train(self, df: pandas.DataFrame, target_col: str):
        x = df.drop(columns=[target_col])
        y = df[target_col]
        self.lin_reg_model = LinearRegression().fit(x, y)

    def predict(self, df: pandas.DataFrame, target_col: str):
        a = self.lin_reg_model.predict(df)  # numpy array
        s = pandas.Series(a, name=target_col, index=df.index)  # pandas series with correct name and index
        return s.to_frame()
