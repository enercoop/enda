import logging
import numpy as np
import pandas as pd
import unittest

try:
    from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
except ImportError:
    raise ImportError("scikit-learn required")

from enda.estimators import EndaEstimatorWithFallback, EndaStackingEstimator
from enda.estimators import EndaNormalizedEstimator, EndaEstimatorRecopy
from tests.test_utils import TestUtils
from enda.ml_backends.sklearn_estimator import EndaSklearnEstimator


class TestEndaEstimatorWithFallback(unittest.TestCase):
    def setUp(self):
        logging.captureWarnings(True)
        logging.disable(logging.ERROR)

    def tearDown(self):
        logging.captureWarnings(False)
        logging.disable(logging.NOTSET)

    def test_1(self):
        # read sample dataset from unittest files
        train_set, test_set, target_name = TestUtils.read_example_a_train_test_sets()

        # remove some values of 'tso_forecast_load_mw' the test_set on purpose
        test_set.loc[
            test_set.index == "2020-09-20 00:30:00+02:00", "tso_forecast_load_mw"
        ] = np.NaN
        test_set.loc[
            test_set.index >= "2020-09-23 23:15:00+02:00", "tso_forecast_load_mw"
        ] = np.NaN

        # the dtype of the column should still be 'float64'
        self.assertEqual("float64", str(test_set["tso_forecast_load_mw"].dtype))

        m = EndaEstimatorWithFallback(
            resilient_column="tso_forecast_load_mw",
            estimator_with=EndaSklearnEstimator(AdaBoostRegressor()),
            estimator_without=EndaSklearnEstimator(RandomForestRegressor()),
        )

        m.train(train_set, target_name)
        # print(m.predict_both(test_set, target_name))
        prediction = m.predict(test_set, target_name)
        # print(prediction)

        # EndaEstimatorWithFallback performs a prediction even when 'tso_forecast_load_mw' was NaN
        # check there is no NaN value in the prediction :
        self.assertEqual(0, prediction[target_name].isna().sum())


class TestEndaStackingEstimator(unittest.TestCase):
    def setUp(self):
        logging.captureWarnings(True)
        logging.disable(logging.ERROR)

    def tearDown(self):
        logging.captureWarnings(False)
        logging.disable(logging.NOTSET)

    def test_1(self):
        train_set, test_set, target_name = TestUtils.read_example_a_train_test_sets()

        m = EndaStackingEstimator(
            base_estimators={
                "ada": EndaSklearnEstimator(RandomForestRegressor()),
                "rf": EndaSklearnEstimator(LinearRegression()),
            },
            final_estimator=EndaSklearnEstimator(AdaBoostRegressor()),
            base_stack_split_pct=0.10,
        )

        m.train(train_set, target_name)
        prediction = m.predict(test_set, target_name)
        # print(prediction)

        self.assertIsInstance(prediction.index, pd.DatetimeIndex)
        self.assertTrue((test_set.index == prediction.index).all())
        self.assertEqual(0, prediction[target_name].isna().sum())

        # also check the underlying base predictions
        base_predictions = m.predict_base_estimators(test_set, target_name)
        # print(base_predictions)
        self.assertListEqual(["ada", "rf"], list(base_predictions.columns))


class TestEndaNormalizedEstimator(unittest.TestCase):
    def setUp(self):
        logging.captureWarnings(True)
        logging.disable(logging.ERROR)

    def tearDown(self):
        logging.captureWarnings(False)
        logging.disable(logging.NOTSET)

    def test_1(self):
        train_set, test_set, target_name = TestUtils.read_example_a_train_test_sets()

        m = EndaNormalizedEstimator(
            inner_estimator=EndaSklearnEstimator(RandomForestRegressor()),
            target_col="load_kw",
            normalization_col="subscribed_power_kva",
            columns_to_normalize=[
                "contracts_count",
                "estimated_annual_consumption_kwh",
            ],
        )

        m.train(train_set, target_name)
        prediction = m.predict(test_set, target_name)
        # print(prediction)

        self.assertIsInstance(prediction.index, pd.DatetimeIndex)
        self.assertTrue((test_set.index == prediction.index).all())
        self.assertEqual(0, prediction[target_name].isna().sum())


class TestEndaEstimatorRecopy(unittest.TestCase):
    def setUp(self):
        logging.captureWarnings(True)
        logging.disable(logging.ERROR)

    def tearDown(self):
        logging.captureWarnings(False)
        logging.disable(logging.NOTSET)

    def test_test_train_predict_1(self):
        train_set, test_set, target_name = TestUtils.read_example_a_train_test_sets()

        train_set_test = train_set.copy(deep=True)
        test_set_test = test_set.copy(deep=True)

        m = EndaEstimatorRecopy(period="1D")

        m.train(train_set, target_name)

        # access train set
        self.assertAlmostEqual(m.training_data["load_kw"], 14.7191, places=3)

        prediction = m.predict(test_set, target_name)

        self.assertIsInstance(prediction.index, pd.DatetimeIndex)
        self.assertTrue((test_set.index == prediction.index).all())
        self.assertEqual(0, prediction[target_name].isna().sum())
        self.assertEqual(prediction[target_name].nunique(), 1)
        self.assertAlmostEqual(prediction[target_name].min(), 14.7191, places=3)

        # check test_set and train_test have not been modified
        self.assertTrue(train_set_test.equals(train_set))
        self.assertTrue(test_set_test.equals(test_set))

    def test_train_predict_2(self):
        train_set, test_set, target_name = TestUtils.read_example_a_train_test_sets()

        # period is None
        m = EndaEstimatorRecopy()

        m.train(train_set, target_name)

        # access train set
        self.assertAlmostEqual(m.training_data["load_kw"], 12.0085, places=3)

        prediction = m.predict(test_set, target_name)

        self.assertIsInstance(prediction.index, pd.DatetimeIndex)
        self.assertTrue((test_set.index == prediction.index).all())
        self.assertEqual(0, prediction[target_name].isna().sum())
        self.assertEqual(prediction[target_name].nunique(), 1)
        self.assertAlmostEqual(prediction[target_name].min(), 12.0085, places=3)
