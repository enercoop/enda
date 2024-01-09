import os
import shutil
import time
import warnings

import h2o
import pandas as pd

from enda.estimators import EndaEstimator

try:
    from h2o.automl import H2OAutoML
    from h2o.estimators import (
        H2ODeepLearningEstimator,
        H2OGeneralizedLinearEstimator,
        H2OGradientBoostingEstimator,
        H2ORandomForestEstimator,
        H2OXGBoostEstimator,
    )
    from h2o.grid.grid_search import H2OGridSearch
except ImportError:
    raise ImportError(
        "h2o is required is you want to use this enda's H2OModel. Try: pip install h2o>=3.32.0.3"
    )


class EndaH2OModel(EndaEstimator):
    def __init__(
        self,
        algo_name,
        model_id,
        algo_param_dict,
        target,
        model_path=None,
        verbose=False,
        logger=None,
        seed=1234,
    ):
        warnings.warn(
            "H2OModel will be obsolete soon, use H2OEstimator",
            PendingDeprecationWarning,
        )

        self.algo_name = algo_name
        self.model_id = model_id
        self.algo_param_dict = algo_param_dict.copy()
        self.target = target
        self.model_path = model_path
        self.algo = None
        self.best_model = None
        self.verbose = verbose
        if self.verbose:
            if not logger:
                raise ValueError("When verbose=True, a logger must be provided")
        self.logger = logger

        algo_implemented_list = [
            "glm",
            "gbm",
            "xgboost",
            "randomforest",
            "deeplearning",
            "automl",
        ]
        if self.algo_name not in algo_implemented_list:
            raise NotImplementedError(
                "Algo {} not in {}".format(self.algo_name, algo_implemented_list)
            )

        if "glm" in self.algo_name.lower():
            if "intercept" in self.algo_param_dict:
                intercept = self.algo_param_dict["intercept"]
                del self.algo_param_dict["intercept"]
                self.algo = H2OGeneralizedLinearEstimator(
                    seed=seed, standardize=False, intercept=intercept
                )
            else:
                self.algo = H2OGeneralizedLinearEstimator(seed=seed, standardize=False)

        if "gbm" in self.algo_name.lower():
            self.algo = H2OGradientBoostingEstimator(seed=seed)

        if "xgboost" in self.algo_name.lower():
            self.algo = H2OXGBoostEstimator(seed=seed)

        if "randomforest" in self.algo_name.lower():
            self.algo = H2ORandomForestEstimator(seed=seed)

        if "deeplearning" in self.algo_name.lower():
            self.algo = H2ODeepLearningEstimator(seed=seed)

        if "automl" in self.algo_name.lower():
            self.algo = H2OAutoML(seed=seed)

    def train(
        self,
        df: pd.DataFrame,
        target_col: str,
        validation_frame=None,
        search_criteria=None,
        nfolds=0,
        score_performance="rmse",
    ):
        if target_col != self.target:
            raise ValueError

        self.grid_search(
            df, validation_frame, search_criteria, nfolds, score_performance
        )

    def grid_search(
        self,
        training_frame,
        validation_frame=None,
        search_criteria=None,
        nfolds=0,
        score_performance="rmse",
    ):
        """
        Allow grid search using self.algo_param_dict as hyperparameter space.
        By default the research is done by sampling uniformly 10 sets from the set of all possible hyperparameter value
        combinations.
        The best model is then the one showing the best 'score_performance' (RMSE by default) on the validation_frame.
        """

        h2o.no_progress()

        # clean h2o from any previous grid-search on this model,
        # else re-training using grid-search will error out
        h2o.remove(self.model_id)
        self.best_model = None

        if search_criteria is None:
            search_criteria = {
                "strategy": "RandomDiscrete",
                "max_models": 20,
                "seed": 1234,
            }

        train = h2o.H2OFrame(training_frame)
        valid = h2o.H2OFrame(validation_frame) if validation_frame is not None else None
        test = (
            validation_frame.copy()
            if validation_frame is not None
            else training_frame.copy()
        )

        perf_on_valid = True if validation_frame is not None else False

        x = list(training_frame.drop(self.target, 1))
        y = self.target

        grid = H2OGridSearch(
            model=self.algo,
            grid_id=self.model_id,
            hyper_params=self.algo_param_dict,
            search_criteria=search_criteria,
        )

        if self.verbose:
            self.logger.info(
                "Start learning -- {} -- from {} to {}".format(
                    self.model_id,
                    training_frame.index.min(),
                    training_frame.index.max(),
                )
            )

        start = time.time()
        # h2o.show_progress()

        grid.train(
            x=x, y=y, training_frame=train, validation_frame=valid, nfolds=nfolds
        )

        end = time.time()
        time_training = round((end - start) / 60.0, 2)  # noqa

        grid_perf = grid.get_grid(sort_by=score_performance, decreasing=False)
        best_model = grid_perf.models[0]

        self.best_model = best_model

        if self.verbose:
            eval_df = self.predict(test)
            eval_df = pd.concat([eval_df, test[self.target]], 1, "inner")
            # scoring = Scoring(eval_df, self.target, [self.model_id])
            # mape = scoring.mean_absolute_percentage_error().values[0]
            # bias = scoring.mean_error().values[0]
            print(grid_perf)
            print(best_model.varimp(use_pandas=True))
            print(best_model.model_performance(valid=perf_on_valid))
            # print("MAPE : {}".format(mape))
            # print("Bias : {}".format(bias))

        # LOGGER.info("End learning -- {} -- it took {} minutes".format(self.model_id, time_training))

        return best_model

    def predict(self, df: pd.DataFrame, target_col: str):
        h2o.no_progress()

        if self.best_model is None:
            raise ValueError(
                "Prediction impossible, no model in memory, please use H2OModel.grid_search() "
                "before trying to predict"
            )

        testing_frame = df
        if self.target in df.columns:
            testing_frame = testing_frame.drop(self.target, 1)

        # LOGGER.info("Predicting -- {} -- from {} to {}"
        #    .format(self.model_id, testing_frame.index.min(), testing_frame.index.max()))

        test = h2o.H2OFrame(df)
        forecast = self.best_model.predict(test)
        forecast = forecast.as_data_frame()
        forecast = forecast.rename(columns={"predict": self.model_id})
        forecast.index = testing_frame.index

        return forecast

    @staticmethod
    def free_memory():
        h2o.remove_all()

    def save(self, tmp_dir=None):
        if self.best_model is None:
            raise ValueError(
                "Saving impossible, no model in memory, please use H2OModel.grid_search() "
                "before trying to save a model"
            )

        model_path_tmp = os.path.join(tmp_dir, "tmp")
        if not os.path.exists(model_path_tmp):
            os.makedirs(model_path_tmp)

        model_path_from_h2o = h2o.save_model(
            model=self.best_model, path=model_path_tmp, force=True
        )
        # file_existed = os.path.exists(self.model_path)
        # message_log = 'overwritten' if file_existed else 'first time'
        shutil.move(model_path_from_h2o, self.model_path)
        # LOGGER.info("Model saved ({}) : {}".format(message_log, self.model_path))

    def load(self):
        # LOGGER.info("Loading model : {}".format(self.model_path))
        uploaded_model = h2o.upload_model(self.model_path)
        self.best_model = uploaded_model
