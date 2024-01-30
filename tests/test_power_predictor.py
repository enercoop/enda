import logging
import os
import time
import unittest

try:
    import h2o
    from h2o.estimators import H2OGeneralizedLinearEstimator

except ImportError as e:
    raise ImportError(
        "h2o is required is you want to test enda's H2OEstimator. "
        "Try: pip install h2o>=3.32.0.3",
        e,
    )

try:
    import joblib
except ImportError as e:
    raise ImportError("joblib is required is you want to test enda's H2OEstimator.", e)

from tests.test_utils import TestUtils
from enda.ml_backends.h2o_estimator import EndaH2OEstimator

# what is being tested
from enda.power_predictor import PowerPredictor


class TestPowerPredictor(unittest.TestCase):
    def setUp(self):
        logging.captureWarnings(True)
        logging.disable(logging.ERROR)
        h2o.init(nthreads=-1)  # starts an h2o local server
        h2o.no_progress()  # don't print out progress bars

    def tearDown(self):
        logging.captureWarnings(False)
        logging.disable(logging.NOTSET)
        h2o.cluster().shutdown()  # shutdown h2o local server
        # print("H2O cluster shutdown...")
        time.sleep(3)  # wait for h2o to actually finish shutting down

    def test_joblib(self):
        train_set, test_set, target_col = TestUtils.get_example_d_train_test_sets(
            source="river"
        )

        m = EndaH2OEstimator(H2OGeneralizedLinearEstimator())
        p = PowerPredictor(standard_plant=True)
        p.train(train_set, estimator=m, target_col=target_col)

        file_path_joblib = os.path.join(
            TestUtils.EXAMPLE_D_DIR, "tmp_test_joblib_h2o_estimator.joblib"
        )
        joblib.dump(p, file_path_joblib)

        # shutdown h2o local server and start another to tests persistence
        h2o.cluster().shutdown()
        time.sleep(3)
        h2o.init(nthreads=-1)
        h2o.no_progress()

        # load joblib object
        m_joblib = joblib.load(file_path_joblib)

        # try to use the H2OEstimator loaded using joblib
        m_joblib.predict(train_set.drop(columns=[target_col]), target_col)
        m_joblib.predict(test_set, target_col)

        # clean the file of this test
        os.remove(file_path_joblib)
