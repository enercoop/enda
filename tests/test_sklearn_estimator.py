import unittest
import pandas as pd
from tests.test_utils import TestUtils
from enda.ml_backends.sklearn_estimator import EndaSklearnEstimator

try:
    from sklearn.linear_model import LinearRegression, SGDRegressor
    from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
    from sklearn.svm import SVR, LinearSVR
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.feature_selection import SelectFromModel
    from sklearn.neural_network import MLPRegressor
except ImportError as e:
    raise ImportError("scikit-learn is required is you want to test enda's SklearnEstimator. "
                      "Try: pip install scikit-learn>=0.24.1", e)


class TestSklearnEstimator(unittest.TestCase):

    def test_estimators(self):
        train_set, test_set, target_name = TestUtils.read_example_a_train_test_sets()

        for estimator in [
            LinearRegression(),
            AdaBoostRegressor(),
            SVR(),
            Pipeline([('poly', PolynomialFeatures(degree=3)),
                      ('linear', LinearRegression(fit_intercept=False))]),
            Pipeline([('standard_scaler', StandardScaler()),
                      ('sgd_regressor', SGDRegressor())]),
            KNeighborsRegressor(n_neighbors=10),
            GaussianProcessRegressor(),
            Pipeline([('feature_selection', SelectFromModel(LinearSVR())),
                      ('classification', RandomForestRegressor())]),
            Pipeline([
                ('standard_scaler', StandardScaler()),
                ('mlp_regressor', MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 5), random_state=1))]
            ),

        ]:
            m = EndaSklearnEstimator(estimator)
            m.train(train_set, target_name)
            prediction = m.predict(test_set, target_name)

            # prediction must preserve the pandas.DatetimeIndex
            self.assertIsInstance(prediction.index, pd.DatetimeIndex)
            self.assertTrue((test_set.index == prediction.index).all())
