import pandas
from enda.models import ModelInterface

try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    raise ImportError("h2o is required is you want to use this enda's H2OModel. Try: pip install scikit-learn>=0.24.1")


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
