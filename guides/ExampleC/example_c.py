from enda.estimators import EndaNormalizedEstimator, EndaStackingEstimator, EndaEstimatorWithFallback

from enda.ml_backends.h2o_estimator import EndaH2OEstimator
from h2o.estimators import H2OGeneralizedLinearEstimator
from h2o.estimators import H2OXGBoostEstimator
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.estimators import H2ORandomForestEstimator


class ExampleC:
    """
    If you haven't already, read Example A and Example B first.
    In example_c.py, we show an example of actual python program that can be called
    from the command line in order to train the algorithm or make predictions.
    However it does not contain data-retrieval functions :
    they are specific to each project & environment.
    """

    @staticmethod
    def get_untrained_estimator():
        random_forest = EndaH2OEstimator(H2ORandomForestEstimator(
            **{
                "ntrees": 300,
                "max_depth": 15,
                "sample_rate": 0.8,
                "min_rows": 10,
                "nbins": 52,
                "mtries": 3,
                "seed": 17
            }
        ))

        gbm = EndaH2OEstimator(H2OXGBoostEstimator(
            **{
                "ntrees": 500,
                "max_depth": 5,
                "sample_rate": 0.8,
                "min_rows": 10,
                "seed": 17
            }
        ))

        xgboost = EndaH2OEstimator(H2OGradientBoostingEstimator(
            **{
                "ntrees": 500,
                "max_depth": 5,
                "sample_rate": 0.5,
                "min_rows": 5,
                "seed": 17
            }
        ))

        stacking = EndaStackingEstimator(
            base_estimators={
                "random_forest": random_forest,
                "gbm": gbm,
                "xgboost": xgboost
            },
            final_estimator=EndaH2OEstimator(H2OGeneralizedLinearEstimator()),
            base_stack_split_pct=0.15
        )

        fallback_gbm = EndaH2OEstimator(H2OXGBoostEstimator(
            **{
                "ntrees": 500,
                "max_depth": 5,
                "sample_rate": 0.8,
                "min_rows": 10,
                "seed": 17
            }
        ))

        fallback_normalized = EndaNormalizedEstimator(
            inner_estimator=fallback_gbm,
            target_col="load_kw",
            normalization_col="kva",
            columns_to_normalize=["contracts_count"]
        )

        with_fallback = EndaEstimatorWithFallback(
            resilient_column="tso_forecast_load_mw",
            estimator_with=stacking,
            estimator_without=fallback_normalized
        )

        return with_fallback
