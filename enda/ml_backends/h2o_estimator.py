import os
import shutil
import tempfile
import warnings

import pandas

from enda.estimators import EndaEstimator

try:
    import h2o
    import h2o.exceptions
except ImportError:
    raise ImportError(
        "h2o is required is you want to use this enda's H2OEstimator. "
        "Try: pip install h2o>=3.32.0.3"
    )


class EndaH2OEstimator(EndaEstimator):
    """
    This is a wrapper around any H2O estimator (or anything with the same train/predict methods).
    H2OEstimator implements enda's ModelInterface.

    If you have a large dataset and need it on the h2o cluster only,
    using a H2OFrame exclusively and not in a pandas.Dataframe,
    just use your H2O model directly to train and predict and copy some lines found here.

    H2O saves model data on the h2o server, so we cannot simply use pickle/joblib out of the box for these
    objects. So we have methods for that.

    """

    def __init__(self, h2o_estimator):
        self.model = h2o_estimator
        self.__model_binary = None

    def train(self, df: pandas.DataFrame, target_col: str):
        x = [
            c for c in df.columns if c != target_col
        ]  # for H20, x is the list of features
        y = target_col  # for H20, y is the name of the target
        training_frame = h2o.H2OFrame(
            df
        )  # H20 training frame containing both features and target
        self.model.train(x, y, training_frame)

    def predict(self, df: pandas.DataFrame, target_col: str):
        test_data = h2o.H2OFrame(df)  # convert to H20 Frame
        forecast = self.model.predict(test_data)  # returns an H20Frame named 'predict'
        forecast = forecast.as_data_frame().rename(
            columns={"predict": target_col}
        )  # convert to pandas and rename
        forecast.index = (
            df.index
        )  # put back the correct index (typically a pandas.DatetimeIndex)
        return forecast

    # All below is just for model persistence : to comply with pickle and deepcopy.

    __tmp_file_path_1 = os.path.join(
        tempfile.gettempdir(), "__h2o_estimator_tmp_file_1_136987"
    )
    __tmp_file_path_2 = os.path.join(
        tempfile.gettempdir(), "__h2o_estimator_tmp_file_2_136987"
    )

    @staticmethod
    def __h2o_model_to_binary(h2o_model):
        # save h2o model to tmp file
        try:
            model_path_from_h2o = h2o.save_model(
                h2o_model, path=EndaH2OEstimator.__tmp_file_path_1, force=True
            )
        except (h2o.exceptions.H2OResponseError, TypeError) as e:
            raise ValueError(
                "Problem getting the model from h2o server. Train the model first. "
                "Cannot access model binary before training (for pickle or deepcopy or other uses).",
                e,
            )

        # model can be saved in __tmp_file_path_1, or in private/__tmp_file_path_1
        potential_startswith = [
            EndaH2OEstimator.__tmp_file_path_1,
            "/private" + EndaH2OEstimator.__tmp_file_path_1,
        ]
        if not model_path_from_h2o.startswith(tuple(potential_startswith)):
            warnings.warn(
                "Expected model_path_from_h2o={} to start with {}".format(
                    model_path_from_h2o, EndaH2OEstimator.__tmp_file_path_1
                )
            )

        # last step actually made a folder and a file inside (at path 'model_path_from_h2o')
        # but we must keep the file inside to read it later,
        # else reading the model later based on the folder does not work despite what H2O docs say
        shutil.move(model_path_from_h2o, EndaH2OEstimator.__tmp_file_path_2)
        shutil.rmtree(EndaH2OEstimator.__tmp_file_path_1)

        # read the tmp file as binary
        with open(EndaH2OEstimator.__tmp_file_path_2, mode="rb") as file:
            binary = file.read()

        # cleanup tmp file
        os.remove(EndaH2OEstimator.__tmp_file_path_2)

        return binary

    @staticmethod
    def __h2o_model_from_binary(binary):
        # save binary to tmp file
        with open(EndaH2OEstimator.__tmp_file_path_2, mode="wb") as file:
            file.write(binary)

        # load H2O model from the file
        h2o_model = h2o.upload_model(path=EndaH2OEstimator.__tmp_file_path_2)

        # cleanup tmp file
        os.remove(EndaH2OEstimator.__tmp_file_path_2)

        return h2o_model

    def __getstate__(self):
        # for pickle, see https://docs.python.org/3/library/pickle.html#pickle-state

        self.__model_binary = EndaH2OEstimator.__h2o_model_to_binary(self.model)
        state = self.__dict__.copy()
        # Remove the un-picklable entry : 'model', but keep the picklable entry: __model_binary
        del state["model"]
        return state

    def __setstate__(self, state):
        # for pickle, see https://docs.python.org/3/library/pickle.html#pickle-state

        self.__dict__.update(state)  # loads only __model_binary
        self.model = EndaH2OEstimator.__h2o_model_from_binary(self.__model_binary)
        # we don't need to keep the __model_binary
        self.__model_binary = None
