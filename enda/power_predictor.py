"""A class that handles day-ahead power prediction"""

import copy

import pandas as pd

from enda.estimators import EndaEstimator


class PowerPredictor:
    """
    This class handles the day-ahead power prediction.
    It accesses the train() and predict() methods of an EndaEstimator object.
    We have the possibility to apply a standard power plant method
    considering all observations to be occurrences of the same theoretical plant
    under different conditions (e.g. meteo, or installed kw).
    This is applied for solar and wind plants notably.
    We also have the possibility to treat each plant independently. Even if
    this is less interesting when applying a usual AI algorithm, this option
    is typically used with a naive estimator for plants along the run of rivers.
    For these plants, we simply recopy the most recent information available.
    """

    def __init__(self, standard_plant: bool = False):
        """
        :param standard_plant: boolean that indicates if we want to use a
               standard plant approach, merging all observations over the plant
               portfolio.
        """

        self.standard_plant = standard_plant
        self.prod_estimators = None

    def train(self, df: pd.DataFrame, estimator: EndaEstimator, target_col: str):
        """
        We provide to this function an EndaEstimator, a training dataframe (two-levels
        multi-indexed), and the target column.

        To train the estimator, we have two options. In the first case, we merge all the
        observations for all the plants, and consider them to be simple realizations
        of the same single power plant with different characteristics ; that's the standard
        power plant model, used notably for wind and solar stations.
        The second option is used for power plants along river, for which no IA is required.
        It is a naive recopy estimator which is used.

        The training sets self.prod_estimators as a dictionary of stations ID - estimator
        In case of a standard plant approach, the returned dictionary has a single entry,
        called "standard_plant", which becomes a reserved ID.

        :param df: the training two-levels multi-indexed dataframe
        :param estimator: an EndaEstimator that will serve as a canvas to create other
               estimators of the same type that will be trained over each plant or over
               all of them in case of non-standard plant.
        :param target_col: the target column
        """

        if not isinstance(estimator, EndaEstimator):
            raise ValueError("Couldn't generate an estimator")

        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError(
                "Prediction for power stations must be performed using "
                "a two-levels multi-indexed dataframe"
            )

        key_col = df.index.levels[0].name
        date_col = df.index.levels[1].name

        df_train = df.copy()

        if self.standard_plant:
            # we don't consider the plants individually.
            # create a dictionary with a single entry called 'standard_plant'
            # if already present, throw an error
            if "standard_plant" in df_train.index.get_level_values(0):
                raise ValueError(
                    "Found a station named 'standard_plant', which is "
                    "a reserved name in this situation."
                )

            df_train = (
                df_train.reset_index().set_index(date_col).drop(columns=[key_col])
            )
            estimator.train(df_train, target_col)
            prod_estimator = copy.deepcopy(estimator)
            self.prod_estimators = {"standard_plant": prod_estimator}

        else:
            # we consider the individual plants
            # create a dictionary with the id of the plant and a dedicated estimator
            self.prod_estimators = {}
            for station_id, data in df.groupby(level=0):
                data = data.reset_index().set_index(date_col).drop(columns=[key_col])
                estimator.train(data, target_col)
                prod_estimator = copy.deepcopy(estimator)
                self.prod_estimators[station_id] = prod_estimator

    def predict(
        self,
        df: pd.DataFrame,
        target_col: str,
        is_positive: bool = False,
        is_normally_clamped: bool = False,
    ):
        """
        Predict target_column values once train() has been called.
        :param df: the forecast two-levels multi-indexed dataframe
        :param target_col: the target column
        :param is_positive: If True, will set negative predicted values to 0
        :param is_normally_clamped: If True, will set negative predicted values to 0, and values higher than 1 to 1
        :return: the two-levels dataframe with the predicted target only.
        """

        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError(
                "Prediction for power generation must be performed using "
                "a two-levels multi-indexed dataframe"
            )

        if self.prod_estimators is None:
            raise ValueError("The estimators have to be trained before being used.")

        key_col = df.index.levels[0].name
        time_col = df.index.levels[1].name

        df_predict = df.copy(deep=True)

        df_new = pd.DataFrame()
        for station_id, data in df_predict.groupby(level=0):
            data = data.reset_index().set_index(time_col).drop(columns=[key_col])

            if self.standard_plant:
                data = self.prod_estimators["standard_plant"].predict(data, target_col)
            else:
                if station_id in self.prod_estimators:
                    data = self.prod_estimators[station_id].predict(data, target_col)
                else:
                    data[target_col] = 0
                    data = data.loc[:, [target_col]]

            if is_positive or is_normally_clamped:
                # reset to 0 negative values
                data.loc[(data[target_col] < 0), target_col] = 0

            if is_normally_clamped:
                # reset to 1 values greater than 1
                data.loc[(data[target_col] > 1), target_col] = 1

            data[key_col] = station_id
            data = data.reset_index().set_index([key_col, time_col])
            df_new = pd.concat([df_new, data], axis=0)

        return df_new
