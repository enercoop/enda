import logging
import pandas as pd
import unittest

import enda.decorators


class TestDecorator(unittest.TestCase):
    
    def setUp(self):
        logging.captureWarnings(True)
        logging.disable(logging.ERROR)

        # datetime index / series ; unordered and duplicates
        self.test_dti = pd.DatetimeIndex(['2024-01-01', '2023-01-01', '2023-01-01', '2025-01-01'])
        self.test_str_series = pd.Series(['2024-01-01', '2023-01-01', '2023-01-01', '2025-01-01'])
        self.test_series = self.test_dti.to_series().reset_index(drop=True)

        # single-indexed dataframe
        self.single_df = (
            pd.date_range(start=pd.to_datetime('2021-01-01'),
                          end=pd.to_datetime('2021-01-06'),
                          freq='D',
                          )
            .to_frame(name='date')
            .set_index('date')
            .assign(value=[0] * 3 + [1] * 3)
        )

        # multi-indexed dataframe
        other_df = (
            self.single_df.copy()
            .iloc[:-1, :]  # to make it more complex, we will drop the last date
            .assign(value=[-1] * 3 + [-2] * 2)
        )
        self.multi_df = (
            pd.concat([self.single_df.copy().assign(id='first'), other_df.assign(id='second')], axis=0)
            .reset_index()
            .set_index(['id', 'date'])
        )

        # 3-levels multi-indexed dataframe
        self.multi_levels_df = (
            self.multi_df.copy()
            .assign(is_before_3=self.multi_df.index.get_level_values("date").day < 3)
            .reset_index().
            set_index(["id", "is_before_3", "date"])
        )

    def tearDown(self):
        logging.captureWarnings(False)
        logging.disable(logging.NOTSET)

    def test_handle_series_as_datetimeindex_default_behaviour(self):

        # test with a function working only on datetimeindex, performing an action on itself
        @enda.decorators.handle_series_as_datetimeindex()
        def tz_localize_time_series(time_series: pd.DatetimeIndex) -> pd.DatetimeIndex:
            return time_series.tz_localize('Europe/Paris')  # would require dt. to work with Series, usually

        # test with dti
        result_dti = tz_localize_time_series(time_series=self.test_dti)

        # test with series
        result_series = tz_localize_time_series(time_series=self.test_series)

        # test with dti and args instead of kwargs
        result_dti = tz_localize_time_series(self.test_dti)

        # test with series and args instead of kwargs
        result_series = tz_localize_time_series(self.test_series)

    def test_handle_series_as_datetimeindex_overload_decorator_parameters(self):

        # test with a function working only on datetimeindex, performing an action on itself
        # we change the arg name, and the return type
        @enda.decorators.handle_series_as_datetimeindex(arg_name='dti', return_input_type=False)
        def tz_localize_time_series(dti: pd.DatetimeIndex) -> pd.DatetimeIndex:
            return dti.tz_localize('Europe/Paris')

        # test with dti
        result_dti = tz_localize_time_series(dti=self.test_dti)

        # test with series
        # shall return a dti
        result_dti = tz_localize_time_series(dti=self.test_series)

        # test with dti and args instead of kwargs
        result_dti = tz_localize_time_series(self.test_dti)

        # test with series and args instead of kwargs
        # shall return a dti
        result_dti = tz_localize_time_series(self.test_series)

    def test_handle_multiindex_return_float(self):

        # Test handle_multiindex in the case it returns a single value
        @enda.decorators.handle_multiindex()
        def compute_mean_as_float(df: pd.DataFrame):
            return df.mean().squeeze()
       
        # test over single_df
        result_df = compute_mean_as_float(self.single_df)

        # test over multi_df
        result_df = compute_mean_as_float(self.multi_df)

        # test over multi_levels_df
        result_df = compute_mean_as_float(self.multi_levels_df)

    def test_handle_multiindex_return_series(self):

        # Test handle_multiindex in the case it returns a single value
        @enda.decorators.handle_multiindex()
        def compute_mean_as_series(df: pd.DataFrame):
            return df.mean()

        # test over single_df
        result_df = compute_mean_as_series(self.single_df)

        # test over multi_df
        result_df = compute_mean_as_series(self.multi_df)

        # test over multi_levels_df
        result_df = compute_mean_as_series(self.multi_levels_df)

    def test_handle_multiindex_return_same_index(self):

        # Test handle_multiindex in the case it returns a dataframe with the same index
        @enda.decorators.handle_multiindex(arg_name='test_df')
        def add_one(test_df: pd.DataFrame, col_name='value'):
            test_df = test_df.copy()
            test_df[col_name] += 1
            return test_df

        # test over single_df
        result_df = add_one(self.single_df, col_name='value')
        # self.assertEqual(result_df)

        # test over multi_df
        # also test kwargs
        result_df = add_one(col_name='value', test_df=self.multi_df)

        # test over multi_levels_df
        result_df = add_one(self.multi_levels_df)

    def test_handle_multiindex_return_new_index(self):

        # Test handle_multiindex in the case it returns a dataframe with a new index
        @enda.decorators.handle_multiindex(arg_name='timeseries_df')
        def twelve_hours_resampler(timeseries_df: pd.DataFrame):
            return timeseries_df.resample("12H").ffill()

        # test over single_df
        result_df = twelve_hours_resampler(self.single_df)
        # self.assertEqual(result_df)

        # test over multi_df
        result_df = twelve_hours_resampler(self.multi_df)

        # test over multi_levels_df
        result_df = twelve_hours_resampler(self.multi_levels_df)
