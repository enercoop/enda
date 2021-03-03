import h2o
import h2o.estimators
import pandas as pd


if __name__ == "__main__":
    x = h2o.H2OFrame(pd.DataFrame(data=[]))
    x.asfactor()

    h2o.estimators.H2OXGBoostEstimator()
