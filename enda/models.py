import abc
import pandas
from collections import OrderedDict
import typing


class ModelInterface(metaclass=abc.ABCMeta):
    """
    This interface represents a simple machine learning model with some universal functions.
    We require these functions :
        train : train the model
        predict : predict using the model

    To save and load instances of a class, use tools like pickle or joblib
    (see information for instance here: https://scikit-learn.org/stable/modules/model_persistence.html).

    This interface is useful to create more advanced models based on these building blocks :
        NormalizedModel : uses one of the inputs as a 'normalization variable' instead of a 'training feature'.
        StackingModel : combines several models to create a more robust model (cross-algorithm 'ensemble' method).
        ModelWithFallback : a model that can cope with missing input in a robust way.

    See tutorials about Python interfaces for instance here https://realpython.com/python-interface/ .
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, 'train') and callable(subclass.train) and
            hasattr(subclass, 'predict') and callable(subclass.predict))

    @abc.abstractmethod
    def train(self, df: pandas.DataFrame, target_col: str):
        """Trains the model using the given data."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, df: pandas.DataFrame, target_col: str) -> pandas.DataFrame:
        """ Predicts and returns a dataframe with just 1 column: target_col_name """
        raise NotImplementedError


class NormalizedModel(ModelInterface):

    def __init__(self,
                 normalized_model: ModelInterface,
                 target_col: str,
                 normalization_col: str,
                 columns_to_normalize: [typing.Iterable[str], None] = None):
        """
        A model that uses one of the inputs as a linear 'normalization variable' instead of a 'training feature'.
        The prediction is:
        predict(X) = X[normalization_col] * predict(X without normalization_col)

        The predict(X without normalization_col) is made by another underlying model.

        :param normalized_model: the model that will perform predict(X without normalization_col)
        :param normalization_col: name of the column used as a multiplier and not as a feature.
        :param columns_to_normalize: (optional) columns in 'X without normalization_col' that must be divided
        by normalization_col for the the underlying model to train/predict correctly.
        """

        if not issubclass(type(normalized_model), ModelInterface):
            raise TypeError("model's type '{}' should implement enda.ModelInterface".format(type(normalized_model)))

        if columns_to_normalize and normalization_col in columns_to_normalize:
            raise ValueError("normalisation_col '{}'should not be in columns_to_normalize {}"
                             .format(normalization_col, columns_to_normalize))

        self.normalized_model = normalized_model
        self.target_col = target_col
        self.normalisation_col = normalization_col
        self.columns_to_normalize = columns_to_normalize

    def check_normalization_col(self, df: pandas.DataFrame):
        zeros_df = df[df[self.normalisation_col] <= 0]
        if not zeros_df.empty:
            raise ValueError("Normalisation_col '{}' : zeros found\n{}".format(self.normalisation_col, zeros_df))

    def normalize(self, df: pandas.DataFrame):
        self.check_normalization_col(df)
        df_norm = df.copy(deep=True)

        if self.columns_to_normalize:
            for c in df.columns:
                if c in self.columns_to_normalize:
                    df_norm[c] = df_norm[c]/df[self.normalisation_col]

        # always normalize the target if it is in the df (present in train mode, not in predict mode)
        if self.target_col in df.columns:
            df_norm[self.target_col] = df_norm[self.target_col]/df[self.normalisation_col]

        df_norm.drop(columns=self.normalisation_col, inplace=True)
        return df_norm

    def train(self, df: pandas.DataFrame, target_col: str = None, drop_where_normalization_under_zero: bool = False):
        if target_col and self.target_col != target_col:
            raise ValueError("target should be None or {}, but given: {}".format(self.target_col, target_col))

        if drop_where_normalization_under_zero:
            df = df.loc[df[self.normalisation_col] > 0, :]
        df_norm = self.normalize(df)
        self.normalized_model.train(df_norm, self.target_col)

    def predict(self, df: pandas.DataFrame, target_col: str = None):
        if target_col and self.target_col != target_col:
            raise ValueError("target should be None or '{}', but given: '{}'".format(self.target_col, target_col))

        df_norm = self.normalize(df)  # error out if any value of normalization_col is <= 0
        predict_norm = self.normalized_model.predict(df_norm, self.target_col)

        if (predict_norm.index != df.index).any():
            raise ValueError("prediction must have the same index as given df. "
                             "Check that for self.normalized_model, the method 'predict' conserves index.")

        predict = predict_norm.multiply(df[self.normalisation_col], axis='index')
        return predict


class StackingModel(ModelInterface):

    def __init__(self,
                 base_models: typing.Mapping[str, ModelInterface],
                 final_model: ModelInterface):
        """
        This class serves the same purpose as the Scikit-Learn "Stacking Regressor". However since we work on
        time-series, we need fine control on which data is passed to train the base_models before training
        the final_model.


        :param base_models: a dict of {model_id -> model}, each model must implement enda.ModelInterface.
        :param final_model: the model used for stacking, must also implement enda.ModelInterface
        """

        if len(base_models) <= 1:
            raise ValueError("At least 2 base_models are required, but given: {}".format(len(base_models)))

        # We store models in an ordered dict to make sure we always iterate over them in the same order
        self.base_models = OrderedDict()
        for model_id in sorted(base_models.keys()):
            self.base_models[model_id] = base_models[model_id]

        self.final_model = final_model

    def train(self, df: pandas.DataFrame, target_col: str, base_stack_split_pct: float = 0.05):

        # training stacking will temporarily train single models on part of the data,
        # so it must be done before training the actual single models
        self._train_final_model(df, target_col, base_stack_split_pct)

        # re-train base models with the full dataset
        self._train_base_models(df, target_col)

    def _train_final_model(self, df, target_col, base_stack_split_pct):
        """
        Trains the final model used for stacking.

        (Temporarily) train the single models with a subset of the data,
        then apply them on the rest of the data. Use this to train the stacking model.
        """

        # split the training frame : ,
        split_int = int(df.shape[0] * (1-base_stack_split_pct))
        split_idx = df.index[split_int]

        df_base_models = df[df.index < split_idx]  # one part to train the base models
        df_stacking = df[df.index >= split_idx]  # the other to train the stacking model

        if df_base_models.shape[0] == 0 or df_stacking.shape[0] == 0:
            raise ValueError("The split gave an empty train set for the base models or the stacking model. Change"
                             "parameter 'base_stack_split_pct' (given {}) or provide a larger training set."
                             .format(base_stack_split_pct))

        self._train_base_models(df_base_models, target_col)

        # make predictions with these temporary base_models, without the target column
        base_model_predictions = self._predict_base_models(df_stacking.drop(columns=[target_col]), target_col)

        # add the target back, to train the stacking model
        base_model_predictions[target_col] = df_stacking[target_col]

        self.final_model.train(base_model_predictions, target_col)

    def _predict_base_models(self, df, target_col):
        """
        :return: a Dataframe with the prediction of each base model in each column.
        """

        model_dfs = []
        for model_id, model in self.base_models.items():
            model_predict = model.predict(df, target_col)

            if (model_predict.index != df.index).any():
                raise ValueError("prediction must have the same index as given df. "
                                 "Check that for model with id '{}', the method 'predict' conserves index."
                                 .format(model_id))

            model_predict.rename(columns={target_col: model_id}, inplace=True)
            model_dfs.append(model_predict)

        predict_df = pandas.concat(model_dfs, axis=1, join='outer')

        if predict_df.shape[0] != df.shape[0]:
            raise ValueError("Given {} values to predict, but predicted {}"
                             .format(df.shape[0], predict_df.shape[0]))

        return predict_df

    def _train_base_models(self, df, target_col):
        # train base models
        for model_id, model in self.base_models.items():
            model.train(df, target_col)

    def predict(self, df: pandas.DataFrame, target_col: str):
        base_model_predictions = self._predict_base_models(df, target_col)
        prediction = self.final_model.predict(base_model_predictions, target_col)

        if (prediction.index != df.index).any():
            raise ValueError("prediction must have the same index as given df. "
                             "Check that self.final_model.predict conserves index.")

        return prediction


class ModelWithFallback(ModelInterface):
    """
    This models allows to make a prediction even when some important input is missing.

    In order to deal with missing values, it is common practice to replace None/NA with some meaningful value like
    the mean or the median of the values found in the train set for this feature.
    However this is problematic when the missing variable has a significant impact on the prediction.

    Instead, this model trains 2 underlying models: a 'model_with' and a 'model_without' the column that can be missing.
    When predicting, it will use the 'model_with' for inputs with the column present and the 'model_without' for the
    others.
    """

    def __init__(self,
                 resilient_column: str,
                 model_with: ModelInterface,
                 model_without: ModelInterface):
        """ Provide 2 different raw models ready to be trained. """

        if model_with is model_without:  # check identity
            raise AttributeError("model_with and model_without must be different objects. If you want the same base "
                                 "model, you can use copying tools (like copy.deepcopy()) to duplicate the raw model.")

        self.resilient_column = resilient_column
        self.model_with = model_with
        self.model_without = model_without

    def train(self, df: pandas.DataFrame, target_col: str):
        """
        Trains the two models : model_with and model_without the 'column_name'
        """

        # only train the "model_with" where column_name is present (not NA)
        self.model_with.train(df.dropna(subset=[self.resilient_column]), target_col)
        # train the "model_without" column_name
        self.model_without.train(df.drop(columns=[self.resilient_column]), target_col)

    def predict_both(self, df: pandas.DataFrame, target_col: str):
        df_with = df[df[self.resilient_column].notna()]
        predict_with = self.model_with.predict(df_with, target_col)
        if predict_with.shape[0] != df.shape[0]:
            # Add missing rows with NaN prediction
            predict_with = predict_with.reindex(df.index)

        df_without = df.drop(columns=[self.resilient_column])
        predict_without = self.model_without.predict(df_without, target_col)

        if (predict_with.index != df.index).any():
            raise ValueError("predict_with must have the same index as given df. "
                             "Check that self.model_with.predict conserves index.")

        if (predict_without.index != df.index).any():
            raise ValueError("predict_without must have the same index as given df. "
                             "Check that self.predict_without.predict conserves index.")

        return predict_with, predict_without

    def predict(self, df: pandas.DataFrame, target_col: str) -> pandas.DataFrame:
        predict_with, predict_without = self.predict_both(df, target_col)

        # keep prediction with column_name when available, else take the prediction of the model without it
        result = predict_with[target_col].fillna(predict_without[target_col])  # pandas series
        result = result.to_frame(target_col)
        return result
