import copy
import logging
import os
import pickle
import shutil
import time
import unittest


import pandas as pd

try:
    import h2o
    from h2o.estimators import H2OGeneralizedLinearEstimator
    from h2o.estimators import H2OXGBoostEstimator
    from h2o.estimators import H2OGradientBoostingEstimator
    from h2o.estimators import H2ORandomForestEstimator
    from h2o.estimators import H2ODeepLearningEstimator

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


class TestH2OEstimator(unittest.TestCase):

    def setUp(self):
        """initialize a local h2o server to tests h2o backend"""
        logging.captureWarnings(True)
        logging.disable(logging.ERROR)

        h2o.init(nthreads=-1)  # starts an h2o local server
        h2o.no_progress()  # don't print out progress bars

        # set up a simple dataset; it comes from the Palmer penguins dataset
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
        """shutdown the local h2o server"""
        logging.captureWarnings(False)
        logging.disable(logging.NOTSET)
        h2o.cluster().shutdown()  # shutdown h2o local server
        # print("H2O cluster shutdown...")
        time.sleep(3)  # wait for h2o to actually finish shutting down

    def test_estimators(self):
        train_set, test_set, target_name = TestUtils.read_example_a_train_test_sets()

        for estimator in [
            H2OGeneralizedLinearEstimator(),
            H2OXGBoostEstimator(),
            H2OGradientBoostingEstimator(),
            H2ORandomForestEstimator(),
            H2ODeepLearningEstimator(),
        ]:
            # print(type(estimator))
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

        file_path_joblib = os.path.join(
            TestUtils.EXAMPLE_A_DIR, "tmp_test_joblib_h2o_estimator.joblib"
        )
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

        file_path_pickle = os.path.join(
            TestUtils.EXAMPLE_A_DIR, "tmp_test_pickle_h2o_estimator.pickle"
        )
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
            H2OGradientBoostingEstimator(
                ntrees=10, max_depth=5, sample_rate=0.5, min_rows=5
            ),
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
        """An example to show how to work with h2o models even with just the h2o functions"""
        train_set, test_set, target_name = TestUtils.read_example_a_train_test_sets()
        m = EndaH2OEstimator(H2OGeneralizedLinearEstimator())
        m.train(train_set, target_name)

        file_path_1 = os.path.join(
            TestUtils.EXAMPLE_A_DIR, "test_regular_h2o_save_load_1"
        )
        file_path_2 = os.path.join(
            TestUtils.EXAMPLE_A_DIR, "test_regular_h2o_save_load_2"
        )

        # "regular" save
        model_path_from_h2o = h2o.save_model(
            model=m.model, path=file_path_1, force=True
        )
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

    def test_get_loss_training(self):
        """
        Test get_loss_training
        """

        # define an h2o linear estimator (with no regularization)
        h2o_lin = EndaH2OEstimator(H2OGeneralizedLinearEstimator(lambda_=0))

        with self.assertRaises(ValueError):
            # not yet trained estimator
            h2o_lin.get_loss_training(score_list=['rmse', 'mae', 'r2'])

        # train the estimator
        h2o_lin.train(self.training_df, target_col=self.target_col)

        # get the loss training
        loss_training = h2o_lin.get_loss_training(score_list=['rmse', 'mae', 'r2'])

        # expected output
        expected_output = pd.Series(
            [299.933078, 255.588197, 0.534021],
            index=['rmse', 'mae', 'r2']
        )

        pd.testing.assert_series_equal(loss_training, expected_output)

    def test_get_feature_importance(self):
        """
        Test get_feature_importance
        """

        # define and train an enda linear estimator
        enda_h2o_lin = EndaH2OEstimator(H2OGeneralizedLinearEstimator(lambda_=0))
        enda_h2o_lin.train(self.several_features_training_df, target_col=self.target_col)

        # expected output -> same as for scikit-learn estimator
        expected_output = pd.Series(
            [0.55191, 0.39653, 0.05156],
            index=['culmen_depth', 'flipper_length', 'culmen_length'],
            name='variable_importance_pct'
        )

        # output and check it
        feature_importance_series = enda_h2o_lin.get_feature_importance()
        pd.testing.assert_series_equal(feature_importance_series, expected_output)

        # define and train an enda H2O tree estimator
        enda_h2o_gb = EndaH2OEstimator(H2OGradientBoostingEstimator(ntrees=20, max_depth=5, min_rows=4, seed=1234))
        enda_h2o_gb.train(self.several_features_training_df, target_col=self.target_col)

        # expected output
        expected_output = pd.Series(
            [0.829228, 0.151105, 0.019667],
            index=['culmen_depth', 'flipper_length', 'culmen_length'],
            name='variable_importance_pct'
        )

        # output and check it
        feature_importance_series = enda_h2o_gb.get_feature_importance()
        pd.testing.assert_series_equal(feature_importance_series, expected_output, atol=1.e-5)

        # check errors
        enda_h2o_lin = EndaH2OEstimator(H2OGeneralizedLinearEstimator(lambda_=0))
        with self.assertRaises(TypeError):
            # untrained estimator
            enda_h2o_lin.get_feature_importance()
