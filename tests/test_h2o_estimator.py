import unittest
import pandas as pd
import time
from tests.test_utils import TestUtils
from enda.ml_backends.h2o_estimator import H2OEstimator

try:
    import h2o
    from h2o.estimators import H2OGeneralizedLinearEstimator
    from h2o.estimators import H2OXGBoostEstimator
    from h2o.estimators import H2OGradientBoostingEstimator
    from h2o.estimators import H2ORandomForestEstimator
    from h2o.estimators import H2ODeepLearningEstimator

except ImportError as e:
    raise ImportError("h2o is required is you want to test enda's H2OEstimator. "
                      "Try: pip install h2o>=3.32.0.3", e)


class TestH2OEstimator(unittest.TestCase):

    def test_estimators(self):
        train_set, test_set, target_name = TestUtils.read_example_a_train_test_sets()

        h2o.init(nthreads=-1)  # starts an h2o local server
        h2o.no_progress()  # don't print out progress bars

        for estimator in [
            H2OGeneralizedLinearEstimator(),
            H2OXGBoostEstimator(),
            H2OGradientBoostingEstimator(),
            H2ORandomForestEstimator(),
            H2ODeepLearningEstimator()
        ]:
            print(type(estimator))
            m = H2OEstimator(estimator)
            m.train(train_set, target_name)
            prediction = m.predict(test_set, target_name)

            # prediction must preserve the pandas.DatetimeIndex
            self.assertIsInstance(prediction.index, pd.DatetimeIndex)
            self.assertTrue((test_set.index == prediction.index).all())

        h2o.cluster().shutdown()  # shutdown h2o local server
        time.sleep(8)  # wait for h2o to actually finish shutting down
