import pandas


class Scoring:
    """
    A class to help scoring algorithms
    predictions_df must include the 'target' column and the predictions in all other columns
    """

    def __init__(
        self, predictions_df: pandas.DataFrame, target: str, normalizing_col: str = None
    ):
        self.predictions_df = predictions_df
        self.target = target
        self.normalizing_col = normalizing_col
        if self.target not in self.predictions_df.columns:
            raise ValueError(
                "target={} must be in predictions_df columns : {}".format(
                    self.target, self.predictions_df
                )
            )
        if len(self.predictions_df.columns) < 2:
            raise ValueError(
                "predictions_df must have at least 2 columns (1 target and 1 prediction)"
            )

        algo_names = list(
            [
                c
                for c in self.predictions_df.columns
                if c not in [self.target, self.normalizing_col]
            ]
        )

        error_df = self.predictions_df.copy(deep=True)
        for x in algo_names:
            error_df[x] = error_df[x] - error_df[self.target]
        error_df = error_df[algo_names]
        self.error_df = error_df

        self.pct_error_df = (
            self.error_df.div(self.predictions_df[self.target], axis=0) * 100
        )

    def error(self):
        return self.error_df

    def mean_error(self):
        return self.error().mean()

    def absolute_error(self):
        return self.error().abs()

    def absolute_error_statistics(self):
        return self.error().describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])

    def mean_absolute_error(self):
        return self.absolute_error().mean()

    def mean_absolute_error_by_month(self):
        abs_error = self.absolute_error()
        return abs_error.groupby(abs_error.index.month).mean()

    def percentage_error(self):
        return self.pct_error_df

    def absolute_percentage_error(self):
        return self.percentage_error().abs()

    def absolute_percentage_error_statistics(self):
        return self.absolute_percentage_error().describe(
            percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]
        )

    def mean_absolute_percentage_error(self):
        return self.absolute_percentage_error().mean()

    def mean_absolute_percentage_error_by_month(self):
        abs_percentage_error = self.absolute_percentage_error()
        return abs_percentage_error.groupby(abs_percentage_error.index.month).mean()

    def normalized_absolute_error(self):
        if self.normalizing_col is None:
            raise ValueError(
                "Cannot use this function without defining normalizing_col in Scoring"
            )
        return self.error_df.abs().div(
            self.predictions_df[self.normalizing_col], axis=0
        )
