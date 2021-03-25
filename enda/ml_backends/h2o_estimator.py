import pandas
from enda.models import ModelInterface

try:
    import h2o
except ImportError:
    raise ImportError("h2o is required is you want to use this enda's H2OEstimator. "
                      "Try: pip install h2o>=3.32.0.3")


class H2OEstimator(ModelInterface):
    """
    This is a wrapper around any H2O estimator (or anything with the same train/predict methods).
    H2OEstimator implements enda's ModelInterface.

    If you have a large dataset and need it on the h2o cluster only,
    using a H2OFrame exclusively and not in a pandas.Dataframe,
    just use your H2O model directly to train and predict and copy some lines found here.
    """

    def __init__(self, h2o_estimator):
        self.model = h2o_estimator

    def train(self, df: pandas.DataFrame, target_col: str):
        x = [c for c in df.columns if c != target_col]  # for H20, x is the list of features
        y = target_col  # for H20, y is the name of the target
        training_frame = h2o.H2OFrame(df)  # H20 training frame containing both features and target
        self.model.train(x, y, training_frame)

    def predict(self, df: pandas.DataFrame, target_col: str):
        test_data = h2o.H2OFrame(df)  # convert to H20 Frame
        forecast = self.model.predict(test_data)  # returns an H20Frame named 'predict'
        forecast = forecast.as_data_frame().rename(columns={'predict': target_col})  # convert to pandas and rename
        forecast.index = df.index  # put back the correct index (typically a pandas.DatetimeIndex)
        return forecast
