import abc
import typing
from collections import OrderedDict
from typing import Iterable, Optional

import pandas


class EndaEstimator(metaclass=abc.ABCMeta):
    """
    This interface represents a simple machine learning estimator with some universal functions.
    We require these functions :
        train : train the estimator
        predict : predict using the estimator

    To save and load instances of a class, use tools like pickle or joblib
    (see information for instance here: https://scikit-learn.org/stable/modules/model_persistence.html).

    This interface is useful to create more advanced estimators based on these building blocks :
        EndaNormalizedEstimator : uses one of the inputs as a 'normalization variable' instead of a 'training feature'.
        EndaStackingEstimator : combines several estimators to create a more robust estimator
                                (cross-algorithm 'ensemble' method).
        EndaEstimatorWithFallback : an estimator that can cope with missing input in a robust way.

    See tutorials about Python interfaces for instance here https://realpython.com/python-interface/ .
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "train")
            and callable(subclass.train)
            and hasattr(subclass, "predict")
            and callable(subclass.predict)
        )

    @abc.abstractmethod
    def train(self, df: pandas.DataFrame, target_col: str):
        """Trains the estimator using the given data."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, df: pandas.DataFrame, target_col: str) -> pandas.DataFrame:
        """Predicts and returns a dataframe with just 1 column: target_col_name"""
        raise NotImplementedError


class EndaNormalizedEstimator(EndaEstimator):
    def __init__(
        self,
        inner_estimator: EndaEstimator,
        target_col: str,
        normalization_col: str,
        columns_to_normalize: Optional[Iterable[str]] = None,
    ):
        """
        An estimator that uses one of the inputs as a linear 'normalization variable' instead of a 'training feature'.
        The prediction is:
        predict(X) = X[normalization_col] * predict(X without normalization_col)

        The predict(X without normalization_col) is made by another underlying estimator: inner_estimator.

        :param inner_estimator: the estimator that will perform predict(X without normalization_col)
        :param normalization_col: name of the column used as a multiplier and not as a feature.
        :param columns_to_normalize: (optional) columns in 'X without normalization_col' that must be divided
        by normalization_col for the the underlying estimator to train/predict correctly.
        """

        if not issubclass(type(inner_estimator), EndaEstimator):
            raise TypeError(
                "inner_estimator's type is '{}' : it must implement EndaEstimator.".format(
                    type(inner_estimator)
                )
            )

        if columns_to_normalize and normalization_col in columns_to_normalize:
            raise ValueError(
                "normalisation_col '{}'should not be in columns_to_normalize {}".format(
                    normalization_col, columns_to_normalize
                )
            )

        self.inner_estimator = inner_estimator
        self.target_col = target_col
        self.normalisation_col = normalization_col
        self.columns_to_normalize = columns_to_normalize

    def check_normalization_col(self, df: pandas.DataFrame):
        zeros_df = df[df[self.normalisation_col] <= 0]
        if not zeros_df.empty:
            raise ValueError(
                "Normalisation_col '{}' : zeros found\n{}".format(
                    self.normalisation_col, zeros_df
                )
            )

    def normalize(self, df: pandas.DataFrame):
        self.check_normalization_col(df)
        df_norm = df.copy(deep=True)

        if self.columns_to_normalize:
            for c in df.columns:
                if c in self.columns_to_normalize:
                    df_norm[c] = df_norm[c] / df[self.normalisation_col]

        # always normalize the target if it is in the df (present in train mode, not in predict mode)
        if self.target_col in df.columns:
            df_norm[self.target_col] = (
                df_norm[self.target_col] / df[self.normalisation_col]
            )

        df_norm.drop(columns=self.normalisation_col, inplace=True)
        return df_norm

    def train(
        self,
        df: pandas.DataFrame,
        target_col: str = None,
        drop_where_normalization_under_zero: bool = False,
    ):
        if target_col and self.target_col != target_col:
            raise ValueError(
                "target should be None or {}, but given: {}".format(
                    self.target_col, target_col
                )
            )

        if drop_where_normalization_under_zero:
            df = df.loc[df[self.normalisation_col] > 0, :]
        df_norm = self.normalize(df)
        self.inner_estimator.train(df_norm, self.target_col)

    def predict(self, df: pandas.DataFrame, target_col: str = None):
        if target_col and self.target_col != target_col:
            raise ValueError(
                "target should be None or '{}', but given: '{}'".format(
                    self.target_col, target_col
                )
            )

        df_norm = self.normalize(
            df
        )  # error out if any value of normalization_col is <= 0
        predict_norm = self.inner_estimator.predict(df_norm, self.target_col)

        if (predict_norm.index != df.index).any():
            raise ValueError(
                "prediction must have the same index as given df. "
                "Check that for self.inner_estimator, the method 'predict' conserves index."
            )

        predict = predict_norm.multiply(df[self.normalisation_col], axis="index")
        return predict


class EndaStackingEstimator(EndaEstimator):
    def __init__(
        self,
        base_estimators: typing.Mapping[str, EndaEstimator],
        final_estimator: EndaEstimator,
        base_stack_split_pct: float = 0.20,
    ):
        """
        This class serves the same purpose as the Scikit-Learn "Stacking Regressor". However since we work on
        time-series, we need fine control on which data is passed to train the base_estimators before training
        the final_estimator (to keep it all chronologically consistent).

        Training is made this way :
            temporarily train base_estimators on train_set[:x]
            use base_estimators to predict on train_set[x:] -> base_predictions
            train final_estimator on the base_predictions
            re-train base_estimators on the full train-set

        :param base_estimators: a dict of {estimator_id -> estimator}, each estimator must be an EndaEstimator.
        :param final_estimator: the estimator used for stacking, must also implement be an EndaEstimator.
        :param base_stack_split_pct: the % of data used to train the base_estimators for
                                     before training final_estimator
                                     (this can be overwritten in the train function).
        """

        if len(base_estimators) <= 1:
            raise ValueError(
                "At least 2 base_estimators are required, but given: {}".format(
                    len(base_estimators)
                )
            )

        # We store estimators in an ordered dict to make sure we always iterate over them in the same order
        self.base_estimators = OrderedDict()
        for estimator_id in sorted(base_estimators.keys()):
            self.base_estimators[estimator_id] = base_estimators[estimator_id]

        self.final_estimator = final_estimator
        self.base_stack_split_pct = base_stack_split_pct

    def train(
        self,
        df: pandas.DataFrame,
        target_col: str,
        base_stack_split_pct: Optional[float] = None,
    ):
        split_pct = (
            base_stack_split_pct if base_stack_split_pct else self.base_stack_split_pct
        )

        # training final_estimator will temporarily train single estimators on part of the data,
        # so it must be done before training the actual single estimators
        self.train_final_estimator(df, target_col, split_pct)

        # re-train base estimators with the full dataset
        self.train_base_estimators(df, target_col)

    def train_final_estimator(self, df, target_col, split_pct):
        """
        Trains the final estimator used for stacking.

        (Temporarily) train the single estimators with a subset of the data,
        then apply them on the rest of the data. Use this to train the stacking estimator.
        """

        # split the training frame : ,
        split_int = int(df.shape[0] * (1 - split_pct))
        split_idx = df.index[split_int]

        df_base_estimators = df[
            df.index < split_idx
        ]  # one part to train the base estimators
        df_stacking = df[
            df.index >= split_idx
        ]  # the other to train the stacking estimator

        if df_base_estimators.shape[0] == 0 or df_stacking.shape[0] == 0:
            raise ValueError(
                "The split gave an empty train set for the base estimators or the final estimator. "
                "Change parameter 'split_pct' (given {}) or provide a larger training set.".format(
                    split_pct
                )
            )

        self.train_base_estimators(df_base_estimators, target_col)

        # make predictions with these temporary base_estimators, without the target column
        base_predictions = self.predict_base_estimators(
            df_stacking.drop(columns=[target_col]), target_col
        )

        # add the target back, to train the final estimator
        base_predictions[target_col] = df_stacking[target_col]

        self.final_estimator.train(base_predictions, target_col)

    def predict_base_estimators(self, df, target_col):
        """
        :return: a Dataframe with the prediction of each base estimator in each column (on several rows).
        Each column's name is the estimator_id given on initialisation.
        """

        estimator_dfs = []
        for estimator_id, estimator in self.base_estimators.items():
            estimator_predict = estimator.predict(df, target_col)

            if (estimator_predict.index != df.index).any():
                raise ValueError(
                    "prediction must have the same index as given df. "
                    "Check that for estimator with id '{}', the method 'predict' conserves index.".format(
                        estimator_id
                    )
                )

            estimator_predict.rename(columns={target_col: estimator_id}, inplace=True)
            estimator_dfs.append(estimator_predict)

        predict_df = pandas.concat(estimator_dfs, axis=1, join="outer")

        if predict_df.shape[0] != df.shape[0]:
            raise ValueError(
                "Given {} values to predict, but predicted {}".format(
                    df.shape[0], predict_df.shape[0]
                )
            )

        return predict_df

    def train_base_estimators(self, df, target_col):
        for estimator_id, estimator in self.base_estimators.items():
            estimator.train(df, target_col)

    def predict(self, df: pandas.DataFrame, target_col: str):
        base_predictions = self.predict_base_estimators(df, target_col)
        prediction = self.final_estimator.predict(base_predictions, target_col)

        if (prediction.index != df.index).any():
            raise ValueError(
                "prediction must have the same index as given df. "
                "Check that self.final_estimator.predict conserves index."
            )

        return prediction


class EndaEstimatorWithFallback(EndaEstimator):
    """
    This estimator allows to make a prediction even when some important input is missing.

    In order to deal with missing values, it is common practice to replace None/NA with some meaningful value like
    the mean or the median of the values found in the train set for this feature.
    However this is problematic when the missing variable has a significant impact on the prediction.

    Instead, this estimator trains 2 underlying estimators:
    an 'estimator_with' and an 'estimator_without' the column that can be missing.
    When predicting, it will use the 'estimator_with' for inputs with the column present
    and the 'estimator_without' for the others.
    """

    def __init__(
        self,
        resilient_column: str,
        estimator_with: EndaEstimator,
        estimator_without: EndaEstimator,
    ):
        """Provide 2 different raw estimators ready to be trained."""

        if estimator_with is estimator_without:  # check identity
            raise AttributeError(
                "estimator_with and estimator_without must be different objects. "
                "If you want the same base estimator, you can use copying tools "
                " (like copy.deepcopy()) to duplicate the raw estimator."
            )

        self.resilient_column = resilient_column
        self.estimator_with = estimator_with
        self.estimator_without = estimator_without

    def train(self, df: pandas.DataFrame, target_col: str):
        """
        Trains the two estimators : estimator_with and estimator_without the 'column_name'
        """

        # only train the "estimator_with" where resilient_column is present (not NA)
        self.estimator_with.train(df.dropna(subset=[self.resilient_column]), target_col)
        # train the "estimator_without" without resilient_column
        self.estimator_without.train(
            df.drop(columns=[self.resilient_column]), target_col
        )

    def predict_both(self, df: pandas.DataFrame, target_col: str):
        df_with = df[
            df[self.resilient_column].notna()
        ]  # only keeps rows where resilient_column is not NaN
        prediction_with = self.estimator_with.predict(df_with, target_col)
        if prediction_with.shape[0] != df.shape[0]:
            # make estimator_with predict NaN where resilient_column is Nan :
            prediction_with = prediction_with.reindex(
                df.index
            )  # adds rows with NaN values

        if (prediction_with.index != df.index).any():
            raise ValueError(
                "prediction_with must have the same index as given df. "
                "Check that self.estimator_with.predict conserves index."
            )

        df_without = df.drop(columns=[self.resilient_column])
        prediction_without = self.estimator_without.predict(df_without, target_col)

        if (prediction_without.index != df.index).any():
            raise ValueError(
                "prediction_without must have the same index as given df. "
                "Check that self.estimator_without.predict conserves index."
            )

        return prediction_with, prediction_without

    def predict(self, df: pandas.DataFrame, target_col: str) -> pandas.DataFrame:
        predict_with, predict_without = self.predict_both(df, target_col)

        # keep prediction with column_name when available, else take the prediction of the estimator without it
        result = predict_with[target_col].fillna(
            predict_without[target_col]
        )  # pandas series
        result = result.to_frame(target_col)
        return result


class EndaEstimatorRecopy(EndaEstimator):
    """
    This estimator is used to recopy the information
    It is notably used to predict the production of river power plants, for which no
    artificial intelligence is relevant.
    It simply recopies the most recent data on a daily basis.
    """

    def __init__(self, period: [str, pandas.Timedelta] = None, key_col: str = None):
        """
        Set up the attribute data that will store the dataframe
        :param period: The period on which past data should be averaged to be used as future value.
                       It must be convertible to a pandas.Timedelta object, eg '1D', '2H', etc...
                       If nothing is provided, the last past value is used in the future
        """
        self.period = pandas.to_timedelta(period) if period is not None else None
        self.training_data = None

    def train(self, df: pandas.DataFrame, target_col: str):
        """
        This function keeps the more recent data of the input dataframe,
        and stores it in the attribute training_data. If a period has been
        given to the estimator constructor, it is used to define a period on
        which to average the data.

        :param df: The input dataframe, with a single DatetimeIndex
        :param target_col: the target column
        """

        if type(df.index) != pandas.DatetimeIndex:
            raise ValueError("Index should be of type DatetimeIndex")

        if target_col not in df.columns:
            raise ValueError(
                f"Target column {target_col} not found in the training dataframe"
            )

        if self.period is None:
            self.training_data = df.sort_index().iloc[
                -1, df.columns.get_indexer([target_col])
            ]

        else:
            # the training dataframe must have a well-defined frequency
            self.training_data = df.iloc[
                df.index > df.index.max() - self.period,
                df.columns.get_indexer([target_col]),
            ].mean()

    def predict(self, df: pandas.DataFrame, target_col: str):
        """
        Make a prediction just copying the retained information.
        :param df: The input forecast dataframe, with a single DatetimeIndex
        :param target_col: the target column
        """

        if self.training_data is None:
            raise ValueError(
                "There is no training dataset defined. Must call 'train' before 'predict'"
            )

        if not isinstance(df.index, type(self.training_data.index)):
            raise ValueError(
                "Forecast dataset index should be of same type of input dataset"
            )

        df_predict = df.copy(deep=True)

        df_predict[target_col] = self.training_data[target_col]

        return df_predict.loc[:, [target_col]]
