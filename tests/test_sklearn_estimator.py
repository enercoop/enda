import logging
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
    raise ImportError(
        "scikit-learn is required is you want to test enda's EndaSklearnEstimator. "
        "Try: pip install scikit-learn>=0.24.1",
        e,
    )


class TestEndaSklearnEstimator(unittest.TestCase):
    def setUp(self):
        logging.captureWarnings(True)
        logging.disable(logging.ERROR)

        # set up a simple dataset; it comes from the Palmer penguins dataset
        self.training_df = pd.DataFrame.from_records(
            [(180, 3700), (182, 3200), (191, 3800), (198, 4400),
             (185, 3700), (195, 3450), (197, 4500), (184, 3325),
             (194, 4200)],
            columns=['flipper_length', 'body_mass']
        )
        self.target_col = 'body_mass'

    def tearDown(self):
        logging.captureWarnings(False)
        logging.disable(logging.NOTSET)

    def test_estimators(self):
        train_set, test_set, target_name = TestUtils.read_example_a_train_test_sets()

        for estimator in [
            LinearRegression(),
            AdaBoostRegressor(),
            SVR(),
            Pipeline(
                [
                    ("poly", PolynomialFeatures(degree=3)),
                    ("linear", LinearRegression(fit_intercept=False)),
                ]
            ),
            Pipeline(
                [
                    ("standard_scaler", StandardScaler()),
                    ("sgd_regressor", SGDRegressor()),
                ]
            ),
            KNeighborsRegressor(n_neighbors=10),
            GaussianProcessRegressor(),
            Pipeline(
                [
                    ("feature_selection", SelectFromModel(LinearSVR())),
                    ("classification", RandomForestRegressor()),
                ]
            ),
            Pipeline(
                [
                    ("standard_scaler", StandardScaler()),
                    (
                        "mlp_regressor",
                        MLPRegressor(
                            solver="lbfgs",
                            alpha=1e-5,
                            hidden_layer_sizes=(5, 5),
                            random_state=1,
                        ),
                    ),
                ]
            ),
        ]:
            m = EndaSklearnEstimator(estimator)
            m.train(train_set, target_name)
            prediction = m.predict(test_set, target_name)

            # prediction must preserve the pandas.DatetimeIndex
            self.assertIsInstance(prediction.index, pd.DatetimeIndex)
            self.assertTrue((test_set.index == prediction.index).all())

    def test_get_loss_training(self):
        """
        Test get_loss_training
        """

        # define an enda linear estimator
        enda_lin = EndaSklearnEstimator(LinearRegression())

        with self.assertRaises(ValueError):
            # not yet trained estimator
            enda_lin.get_loss_training(score_list=['rmse', 'mae', 'r2', 'mape'])

        # train the estimator
        enda_lin.train(self.training_df, target_col=self.target_col)

        # get the loss training
        loss_training = enda_lin.get_loss_training(score_list = ['rmse', 'mae', 'r2', 'mape'])

        # expected output
        expected_output = pd.Series(
            [299.933078, 255.588197, 0.534021, 0.069019],
            index=['rmse', 'mae', 'r2', 'mape']
        )

        pd.testing.assert_series_equal(loss_training, expected_output)

