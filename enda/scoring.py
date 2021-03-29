class Scoring:
    """ A class to help scoring algorithms """

    def __init__(self, predictions_df, target, algo_names):

        self.predictions_df = predictions_df
        self.target = target
        self.algo_names = algo_names

        error_df = self.predictions_df.copy(deep=True)
        for x in self.algo_names:
            error_df[x] = error_df[x] - error_df[self.target]
        self.error_df = error_df[[self.algo_names]]

        self.pct_error_df = self.error_df.div(self.predictions_df[self.target], axis=0)*100

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
        return self.absolute_percentage_error().describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])

    def mean_absolute_percentage_error(self):
        return self.absolute_percentage_error().mean()

    def mean_absolute_percentage_error_by_month(self):
        abs_percentage_error = self.absolute_percentage_error()
        return abs_percentage_error.groupby(abs_percentage_error.index.month).mean()
