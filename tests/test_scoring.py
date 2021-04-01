import unittest
import pandas as pd
from enda.scoring import Scoring


class TestScoring(unittest.TestCase):

    def test_scoring_1(self):

        index = pd.date_range(
            start=pd.to_datetime('2020-01-01 00:00:00+01:00').tz_convert('Europe/Paris'),
            end=pd.to_datetime('2020-01-01 02:00:00+01:00').tz_convert('Europe/Paris'),
            freq='H',
            tz='Europe/Paris',
            name='time'
        )
        df = pd.DataFrame(
            data=[[10.0, 9.0, 11.0], [1.0, 0.85, 1.2], [5, 5.1, 4.9]],
            columns=['actual', 'algo1', 'algo2'],
            index=index,
        )

        scoring = Scoring(predictions_df=df, target='actual')

        # print(df, scoring.error(), scoring.percentage_error())
        # print(scoring.mean_absolute_percentage_error())

        self.assertAlmostEqual(-1, scoring.error().loc['2020-01-01 00:00:00+01:00', 'algo1'])
        self.assertAlmostEqual(-10, scoring.percentage_error().loc['2020-01-01 00:00:00+01:00', 'algo1'])
        self.assertAlmostEqual(10, scoring.absolute_percentage_error().loc['2020-01-01 00:00:00+01:00', 'algo1'])
        self.assertAlmostEqual(9, scoring.mean_absolute_percentage_error()['algo1'])

        self.assertAlmostEqual(-0.1, scoring.error().loc['2020-01-01 02:00:00+01:00', 'algo2'])
        self.assertAlmostEqual(-2.0, scoring.percentage_error().loc['2020-01-01 02:00:00+01:00', 'algo2'])
        self.assertAlmostEqual(2.0, scoring.absolute_percentage_error().loc['2020-01-01 02:00:00+01:00', 'algo2'])
        self.assertAlmostEqual(32/3, scoring.mean_absolute_percentage_error()['algo2'])
