import logging
import pandas as pd
import unittest

from enda.backtesting import BackTesting


class TestBackTesting(unittest.TestCase):

    def setUp(self):
        logging.disable(logging.ERROR)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_yield_train_test_1(self):

        df = pd.date_range(
            start=pd.to_datetime('2015-01-01 00:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2021-01-01 00:00:00+01:00').tz_convert('Europe/Paris'),
            freq='D',
            tz='Europe/Paris',
            name='time'
        ).to_frame()
        df = df.set_index('time')
        df["value"] = 1

        count_iterations = 0
        for train_set, test_set in BackTesting.yield_train_test(
                df,
                start_eval_datetime=pd.to_datetime('2019-01-01 00:00:00+01:00').tz_convert('Europe/Paris'),
                days_between_trains=7,
                gap_days_between_train_and_eval=14
        ):
            # if count_iterations % 20 == 0:
            #    print("Iteration {}/105, train set {}->{}, test set {}->{}"
            #          .format(count_iterations, train_set.index.min(), train_set.index.max(),
            #                  test_set.index.min(), test_set.index.max()))
            count_iterations += 1

        self.assertEqual(count_iterations, 105)

    def test_yield_train_test_2(self):

        df = pd.date_range(
            start=pd.to_datetime('2015-01-01 00:00:00'),
            end=pd.to_datetime('2021-01-01 00:00:00'),
            freq='D',
            name='time'
        ).to_frame()
        df = df.set_index('time')
        df["value"] = 1

        count_iterations = 0
        for train_set, test_set in BackTesting.yield_train_test(
                df,
                start_eval_datetime=pd.to_datetime('2018-01-01 00:00:00'),
                days_between_trains=10,
                gap_days_between_train_and_eval=5
        ):
            # if count_iterations % 20 == 0:
            #    print("Iteration {}/110, train set {}->{}, test set {}->{}"
            #          .format(count_iterations, train_set.index.min(), train_set.index.max(),
            #                  test_set.index.min(), test_set.index.max()))
            count_iterations += 1

        self.assertEqual(count_iterations, 110)



