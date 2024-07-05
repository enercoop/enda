import logging
import unittest
import pandas as pd
from tests.test_utils import TestUtils
from enda.ml_backends.sklearn_estimator import EndaSklearnEstimator

try:
    from sklearn.linear_model import LinearRegression, SGDRegressor
    from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
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
        # cf: https://github.com/INRIA/scikit-learn-mooc/tree/main/datasets
        self.training_df = pd.DataFrame.from_records(
            [(180, 3700), (182, 3200), (191, 3800), (198, 4400),
             (185, 3700), (195, 3450), (197, 4500), (184, 3325),
             (194, 4200)],
            columns=['flipper_length', 'body_mass']
        )

        # same dataset with three features
        self.several_features_training_df = pd.DataFrame.from_records(
            [(180, 3700, 37.8, 17.3), (182, 3200, 41.1, 17.6),
             (191, 3800, 38.6, 21.2), (198, 4400, 34.6, 21.1),
             (185, 3700, 36.6, 17.8), (195, 3450, 38.7, 19.),
             (197, 4500, 42.5, 20.7), (184, 3325, 34.4, 18.4),
             (194, 4200, 46., 21.5)],
            columns=['flipper_length', 'body_mass', 'culmen_length', 'culmen_depth']
        )

        # target col is body mass
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
            enda_lin.get_loss_training(scores=['rmse', 'mae', 'r2', 'mape'])

        # train the estimator
        enda_lin.train(self.training_df, target_col=self.target_col)

        # get the loss training
        loss_training = enda_lin.get_loss_training(scores = ['rmse', 'mae', 'r2', 'mape'])

        # expected output
        expected_output = pd.Series(
            [299.933078, 255.588197, 0.534021, 0.069019],
            index=['rmse', 'mae', 'r2', 'mape']
        )

        pd.testing.assert_series_equal(loss_training, expected_output)

    def test_get_feature_importance(self):
        """
        Test get_feature_importance
        """

        # define and train an enda linear estimator
        enda_lin = EndaSklearnEstimator(LinearRegression())
        enda_lin.train(self.several_features_training_df, target_col=self.target_col)

        # expected output
        expected_output = pd.Series(
            [0.55191, 0.39653, 0.05156],
            index=['culmen_depth', 'flipper_length', 'culmen_length'],
            name='variable_importance_pct'
        )

        # output and check it
        feature_importance_series = enda_lin.get_feature_importance()
        pd.testing.assert_series_equal(feature_importance_series, expected_output)

        # define and train an enda tree estimator
        enda_gb = EndaSklearnEstimator(GradientBoostingRegressor(n_estimators=20, max_depth=5, random_state=1234))
        enda_gb.train(self.several_features_training_df, target_col=self.target_col)

        # expected output
        expected_output = pd.Series(
            [0.784954, 0.155499, 0.059548],
            index=['culmen_depth', 'flipper_length', 'culmen_length'],
            name='variable_importance_pct'
        )

        # output and check it
        feature_importance_series = enda_gb.get_feature_importance()
        pd.testing.assert_series_equal(feature_importance_series, expected_output)

        # check errors
        enda_mlp = EndaSklearnEstimator(MLPRegressor())
        with self.assertRaises(ValueError):
            # untrained estimator
            enda_mlp.get_feature_importance()

        enda_mlp.train(self.several_features_training_df, target_col=self.target_col)
        with self.assertRaises(NotImplementedError):
            # not implemented model for feature importance
            enda_mlp.get_feature_importance()
