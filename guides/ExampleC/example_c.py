# If you haven't already, read Example A and example B first.
# In example_c.py, we show an example of actual python program that can be called
# from the command line in order to train the algorithm or make predictions. 
# However it does not contain data-retrieval functions : 
# they are specific to each project & environment.


# TODO



# below is out of scope:

# an algorithm using Enda : "Normlized xgboost using kva"
enda_n = enda.models.NormalizedModel(
    normalized_model = H2OModel(
        algo_name="xgboost",
        model_id="enda_n_xgboost",
        target="load_kw",
        algo_param_dict= {
            "ntrees": [500],
            "max_depth": [5],
            "sample_rate": [0.8],
            "min_rows": [10]
        },
    ),
    target_col = "load_kw",
    normalization_col = "kva",
    columns_to_normalize = ["contracts_count"]
)
# all_models['enda_n'] = enda_n



# another algorithm using Enda : "glm-stacking of [randomforest, gbm, xgboost]"
_m_randomforest = H2OModel(
    algo_name="randomforest",
    model_id="enda_s_randomforest",
    target="load_kw",
    algo_param_dict= {
        "ntrees": [300],
        "max_depth": [15],
        "sample_rate": [0.8],
        "min_rows": [10],
        "nbins": [52],
        "mtries": [3]
    }
)

_m_gbm = H2OModel(
    algo_name="gbm",
    model_id="enda_s_gbm",
    target="load_kw",
    algo_param_dict= {
        "ntrees": [500],
        "max_depth": [5],
        "sample_rate": [0.5],
        "min_rows": [5]
    }
)

_m_xgboost = H2OModel(
    algo_name="xgboost",
    model_id="enda_s_xgboost",
    target="load_kw",
    algo_param_dict= {
        "ntrees": [500],
        "max_depth": [5],
        "sample_rate": [0.8],
        "min_rows": [10]
    }
)

enda_s = enda.models.StackingModel(
    base_models = {
        "randomforest": _m_randomforest,
        "gbm": _m_gbm,
        "xgboost": _m_xgboost
    },
    final_model = H2OModel(
        algo_name="xgboost",
        model_id="enda_s_xgboost",
        target="load_kw",
        algo_param_dict= {
            "ntrees": [500],
            "max_depth": [5],
            "sample_rate": [0.8],
            "min_rows": [10]
        }
    )
)
# all_models['enda_s'] = enda_s


# another Enda algorithm : "normalized glm-stacking of [randomforest, gbm, xgboost]"

_m_randomforest = H2OModel(
    algo_name="randomforest",
    model_id="enda_ns_randomforest",
    target="load_kw",
    algo_param_dict= {
        "ntrees": [300],
        "max_depth": [15],
        "sample_rate": [0.8],
        "min_rows": [10],
        "nbins": [52],
        "mtries": [3]
    }
)

_m_gbm = H2OModel(
    algo_name="gbm",
    model_id="enda_ns_gbm",
    target="load_kw",
    algo_param_dict= {
        "ntrees": [500],
        "max_depth": [5],
        "sample_rate": [0.5],
        "min_rows": [5]
    }
)

_m_xgboost = H2OModel(
    algo_name="xgboost",
    model_id="enda_ns_xgboost",
    target="load_kw",
    algo_param_dict= {
        "ntrees": [500],
        "max_depth": [5],
        "sample_rate": [0.8],
        "min_rows": [10]
    }
)

_m_stacking = enda.models.StackingModel(
    base_models = {
        "randomforest": _m_randomforest,
        "gbm": _m_gbm,
        "xgboost": _m_xgboost
    },
    final_model = H2OModel(algo_name="glm", model_id="enda_ns_stacking_glm", target="load_kw", algo_param_dict={})
)

m_enda_ns = enda.models.NormalizedModel(
    normalized_model = _m_stacking,
    target_col = "load_kw",
    normalization_col = "kva",
    columns_to_normalize = ["contracts_count"]
)

# all_models["enda_ns"] = m_enda_ns

