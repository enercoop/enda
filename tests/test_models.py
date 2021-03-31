import unittest
import numpy as np
import pandas as pd
from enda.estimators import EndaEstimatorWithFallback, EndaStackingEstimator, EndaNormalizedEstimator
from tests.test_utils import TestUtils
from enda.ml_backends.sklearn_estimator import EndaSklearnEstimator
try:
    from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
except ImportError as e:
    raise ImportError("scikit-learn required")


class TestModelWithFallback(unittest.TestCase):

    def test_1(self):
        # read sample dataset from unittest files
        train_set, test_set, target_name = TestUtils.read_example_a_train_test_sets()

        # remove some values of 'tso_forecast_load_mw' the test_set on purpose
        test_set.loc[test_set.index == '2020-09-20 00:30:00+02:00', 'tso_forecast_load_mw'] = np.NaN
        test_set.loc[test_set.index >= '2020-09-23 23:15:00+02:00', 'tso_forecast_load_mw'] = np.NaN

        # the dtype of the column should still be 'float64'
        self.assertEqual('float64', str(test_set['tso_forecast_load_mw'].dtype))

        m = EndaEstimatorWithFallback(
            resilient_column='tso_forecast_load_mw',
            estimator_with=EndaSklearnEstimator(AdaBoostRegressor()),
            estimator_without=EndaSklearnEstimator(RandomForestRegressor())
        )

        m.train(train_set, target_name)
        # print(m.predict_both(test_set, target_name))
        prediction = m.predict(test_set, target_name)
        # print(prediction)

        # ModelWithFallback performs a prediction even when 'tso_forecast_load_mw' was NaN
        # check there is no NaN value in the prediction :
        self.assertEqual(0, prediction[target_name].isna().sum())


class TestStackingModel(unittest.TestCase):

    def test_1(self):
        train_set, test_set, target_name = TestUtils.read_example_a_train_test_sets()

        m = EndaStackingEstimator(
            base_estimators={
                "ada": EndaSklearnEstimator(RandomForestRegressor()),
                "rf": EndaSklearnEstimator(LinearRegression())
            },
            final_estimator=EndaSklearnEstimator(AdaBoostRegressor()),
            base_stack_split_pct=0.10
        )

        m.train(train_set, target_name)
        prediction = m.predict(test_set, target_name)
        # print(prediction)

        self.assertIsInstance(prediction.index, pd.DatetimeIndex)
        self.assertTrue((test_set.index == prediction.index).all())
        self.assertEqual(0, prediction[target_name].isna().sum())

        # also check the underlying base model predictions
        base_model_predictions = m.predict_base_estimators(test_set, target_name)
        # print(base_model_predictions)
        self.assertListEqual(["ada", "rf"], list(base_model_predictions.columns))


class TestNormalizedModel(unittest.TestCase):

    def test_1(self):
        train_set, test_set, target_name = TestUtils.read_example_a_train_test_sets()

        m = EndaNormalizedEstimator(
            inner_estimator=EndaSklearnEstimator(RandomForestRegressor()),
            target_col="load_kw",
            normalization_col="subscribed_power_kva",
            columns_to_normalize=["contracts_count", "estimated_annual_consumption_kwh"]
        )

        m.train(train_set, target_name)
        prediction = m.predict(test_set, target_name)
        # print(prediction)

        self.assertIsInstance(prediction.index, pd.DatetimeIndex)
        self.assertTrue((test_set.index == prediction.index).all())
        self.assertEqual(0, prediction[target_name].isna().sum())
