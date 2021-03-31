import unittest
import pandas as pd
import time
from tests.test_utils import TestUtils
from enda.ml_backends.h2o_estimator import EndaH2OEstimator
import pickle
import os
import shutil
import copy

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

try:
    import joblib
except ImportError as e:
    raise ImportError("joblib is required is you want to test enda's H2OEstimator.", e)


class TestH2OEstimator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """ initialize a local h2o server to tests h2o backend"""
        h2o.init(nthreads=-1)  # starts an h2o local server
        h2o.no_progress()  # don't print out progress bars

    @classmethod
    def tearDownClass(cls):
        """ shutdown the local h2o server """
        h2o.cluster().shutdown()  # shutdown h2o local server
        print("H2O cluster shutdown...")
        time.sleep(3)  # wait for h2o to actually finish shutting down

    def test_estimators(self):
        train_set, test_set, target_name = TestUtils.read_example_a_train_test_sets()

        for estimator in [
            H2OGeneralizedLinearEstimator(),
            H2OXGBoostEstimator(),
            H2OGradientBoostingEstimator(),
            H2ORandomForestEstimator(),
            H2ODeepLearningEstimator()
        ]:
            print(type(estimator))
            m = EndaH2OEstimator(estimator)
            m.train(train_set, target_name)
            prediction = m.predict(test_set, target_name)

            # prediction must preserve the pandas.DatetimeIndex
            self.assertIsInstance(prediction.index, pd.DatetimeIndex)
            self.assertTrue((test_set.index == prediction.index).all())

    def test_joblib(self):
        train_set, test_set, target_name = TestUtils.read_example_a_train_test_sets()
        m = EndaH2OEstimator(H2OGeneralizedLinearEstimator())
        m.train(train_set, target_name)

        file_path_joblib = os.path.join(TestUtils.EXAMPLE_A_DIR, "tmp_test_joblib_h2o_estimator.joblib")
        joblib.dump(m, file_path_joblib)

        # shutdown h2o local server and start another to tests persistence
        h2o.cluster().shutdown()
        time.sleep(3)
        h2o.init(nthreads=-1)
        h2o.no_progress()

        # load joblib object
        m_joblib = joblib.load(file_path_joblib)

        # try to use the H2OEstimator loaded using joblib
        m_joblib.predict(train_set.drop(columns=[target_name]), target_name)
        m_joblib.predict(test_set, target_name)

        # clean the file of this test
        os.remove(file_path_joblib)

    def test_pickle(self):
        """
        a normal use of the pickle library on a H2OEstimator object
        h2o server is shutdown and another is started in the middle of this test to test persistence
        """

        train_set, test_set, target_name = TestUtils.read_example_a_train_test_sets()
        m = EndaH2OEstimator(H2OGeneralizedLinearEstimator())
        m.train(train_set, target_name)

        file_path_pickle = os.path.join(TestUtils.EXAMPLE_A_DIR, "tmp_test_pickle_h2o_estimator.pickle")
        with open(file_path_pickle, "wb") as file:
            pickle.dump(m, file)

        # shutdown h2o local server and start another to tests persistence
        h2o.cluster().shutdown()
        time.sleep(3)
        h2o.init(nthreads=-1)
        h2o.no_progress()

        # load pickled object
        with open(file_path_pickle, "rb") as file:
            m_pickle = pickle.load(file)

        # try to use the H2OEstimator loaded using pickle
        m_pickle.predict(train_set.drop(columns=[target_name]), target_name)
        m_pickle.predict(test_set, target_name)

        # clean the file of this test
        os.remove(file_path_pickle)

    def test_deepcopy(self):
        train_set, test_set, target_name = TestUtils.read_example_a_train_test_sets()

        for estimator in [
            H2OXGBoostEstimator(),
            H2OGradientBoostingEstimator(ntrees=10, max_depth=5, sample_rate=0.5, min_rows=5)
        ]:

            m = EndaH2OEstimator(estimator)
            m.train(train_set, target_name)

            m_deepcopy = copy.deepcopy(m)
            m_deepcopy.predict(train_set.drop(columns=[target_name]), target_name)
            m_deepcopy.predict(test_set, target_name)

    def test_deepcopy_model_not_trained(self):
        m_not_trained = EndaH2OEstimator(H2OGeneralizedLinearEstimator())
        with self.assertRaises(ValueError):
            copy.deepcopy(m_not_trained)

    def test_regular_h2o_save_load(self):
        """ An example to show how to work with h2o models even with just the h2o functions """
        train_set, test_set, target_name = TestUtils.read_example_a_train_test_sets()
        m = EndaH2OEstimator(H2OGeneralizedLinearEstimator())
        m.train(train_set, target_name)

        file_path_1 = os.path.join(TestUtils.EXAMPLE_A_DIR, "test_regular_h2o_save_load_1")
        file_path_2 = os.path.join(TestUtils.EXAMPLE_A_DIR, "test_regular_h2o_save_load_2")

        # "regular" save
        model_path_from_h2o = h2o.save_model(model=m.model, path=file_path_1, force=True)
        shutil.move(src=model_path_from_h2o, dst=file_path_2)
        shutil.rmtree(file_path_1)

        # shutdown h2o local server and start another to tests persistence
        h2o.cluster().shutdown()
        time.sleep(3)
        h2o.init(nthreads=-1)
        h2o.no_progress()

        # "regular" load
        h2o.upload_model(file_path_2)

        # this is just cleaning this test
        os.remove(file_path_2)
