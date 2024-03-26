"""A module for testing the Calendar and Holidays classes in enda/calendar.py"""

import logging
import unittest
import pandas as pd

from enda.feature_engineering.calendar import Calendar, FrenchCalendar, Holidays


class TestHolidays(unittest.TestCase):
    """
    This class aims at testing the functions of the Holidays class
    """

    def setUp(self):
        logging.disable(logging.INFO)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_get_public_holidays(self):
        """
        Test get_public_holidays
        """

        with self.assertRaises(NotImplementedError):
            Holidays.get_public_holidays(country='ES')

        result_df = Holidays.get_public_holidays('FR', years_list=[2021, 2030])
        expected_df = pd.DataFrame(
            [(pd.Timestamp(2021, 1, 1), '1er janvier'),
             (pd.Timestamp(2021, 4, 5), 'Lundi de Paques'),
             (pd.Timestamp(2021, 5, 1), '1er mai'),
             (pd.Timestamp(2021, 5, 8), '8 mai'),
             (pd.Timestamp(2021, 5, 13), 'Ascension'),
             (pd.Timestamp(2021, 5, 24), 'Lundi de Pentecote'),
             (pd.Timestamp(2021, 7, 14), '14 juillet'),
             (pd.Timestamp(2021, 8, 15), 'Assomption'),
             (pd.Timestamp(2021, 11, 1), 'Toussaint'),
             (pd.Timestamp(2021, 11, 11), '11 novembre'),
             (pd.Timestamp(2021, 12, 25), 'Jour de Noel'),
             (pd.Timestamp(2030, 1, 1), '1er janvier'),
             (pd.Timestamp(2030, 4, 22), 'Lundi de Paques'),
             (pd.Timestamp(2030, 5, 1), '1er mai'),
             (pd.Timestamp(2030, 5, 8), '8 mai'),
             (pd.Timestamp(2030, 5, 30), 'Ascension'),
             (pd.Timestamp(2030, 6, 10), 'Lundi de Pentecote'),
             (pd.Timestamp(2030, 7, 14), '14 juillet'),
             (pd.Timestamp(2030, 8, 15), 'Assomption'),
             (pd.Timestamp(2030, 11, 1), 'Toussaint'),
             (pd.Timestamp(2030, 11, 11), '11 novembre'),
             (pd.Timestamp(2030, 12, 25), 'Jour de Noel')],
            columns=['date', 'public_holiday_name']
        )

        pd.testing.assert_frame_equal(result_df, expected_df)

        result_df = Holidays.get_public_holidays('FR')
        self.assertEqual(result_df["date"].min(), pd.Timestamp(2000, 1, 1))
        self.assertEqual(result_df["date"].max(), pd.Timestamp(2050, 12, 25))

    def test_get_school_holidays(self):
        """
        Test get_school_holidays
        """

        with self.assertRaises(NotImplementedError):
            Holidays.get_school_holidays(country='ES')

        result_df = Holidays.get_school_holidays('FR', years_list=[2021])

        # only get lines for the holidays
        self.assertEqual(len(result_df), 155)
        self.assertEqual(result_df["date"].min(), pd.Timestamp(2021, 1, 1).date())
        self.assertEqual(result_df["date"].max(), pd.Timestamp(2021, 12, 31).date())
        self.assertListEqual(list(result_df.columns),
                             ["date", "vacances_zone_a", "vacances_zone_b",
                              "vacances_zone_c", "school_holiday_name"]
                             )


class TestCalendar(unittest.TestCase):
    """
    This class aims at testing the functions of the Holidays class
    """

    def setUp(self):
        logging.captureWarnings(True)
        logging.disable(logging.INFO)

    def tearDown(self):
        logging.captureWarnings(False)
        logging.disable(logging.NOTSET)

    def test_get_lockdown(self):
        """
        Test get_lockdown
        """

        with self.assertRaises(NotImplementedError):
            Calendar.get_lockdown(country='ES')

        result_df = Calendar.get_lockdown('FR', years_list=[2020, 2021])

        self.assertEqual(len(result_df), 731)
        self.assertEqual(result_df.at[pd.Timestamp(2020, 1, 1), "lockdown"], 0)
        self.assertEqual(result_df.at[pd.Timestamp(2020, 4, 1), "lockdown"], 1)

    def test_get_public_holidays(self):
        """
        Test get_public_holidays
        """

        with self.assertRaises(NotImplementedError):
            Calendar.get_public_holidays(country='ES')

        result_df = Calendar.get_public_holidays(country='FR', years_list=[2020, 2021])

        self.assertEqual(len(result_df), 731)
        self.assertEqual(result_df.at[pd.Timestamp(2020, 1, 1), "public_holiday"], 1)
        self.assertEqual(result_df.at[pd.Timestamp(2021, 4, 2), "public_holiday"], 0)

    def test_get_number_school_areas_off(self):
        """
        Test get_number_school_areas_off
        """

        result_df = FrenchCalendar.get_number_school_areas_off(years_list=[2020, 2021])

        self.assertEqual(len(result_df), 731)
        self.assertEqual(result_df.at[pd.Timestamp(2020, 1, 1), "nb_school_areas_off"], 3)
        self.assertEqual(result_df.at[pd.Timestamp(2021, 2, 27), "nb_school_areas_off"], 2)

    def test_get_extra_long_weekend(self):
        """
        Test get_extra_long_weekend
        """

        with self.assertRaises(NotImplementedError):
            Calendar.get_extra_long_weekend(country='ES')

        result_df = Calendar.get_extra_long_weekend('FR', years_list=[2020, 2021])

        self.assertEqual(len(result_df), 731)
        self.assertEqual(result_df.at[pd.Timestamp(2020, 1, 1), "extra_long_weekend"], 0)
        self.assertEqual(result_df.at[pd.Timestamp(2021, 11, 12), "extra_long_weekend"], 1)
        self.assertEqual(len(result_df.loc[(result_df["extra_long_weekend"] == 1) & (result_df.index.year == 2021)]), 2)

    def test_feature_special_days(self):
        """
        Test feature_special_days
        """

        with self.assertRaises(NotImplementedError):
            Calendar.feature_special_days(country='ES')

        result_df = Calendar.feature_special_days('FR', years_list=[2020, 2021])

        self.assertEqual(len(result_df), 35088)
        self.assertEqual(result_df.index.freq.freqstr, "30T")

        test_timestamp = pd.to_datetime("2021-11-12 00:00:00+01")
        self.assertEqual(result_df.at[test_timestamp, "extra_long_weekend"], 1)
        self.assertEqual(result_df.at[test_timestamp, "nb_school_areas_off"], 0)
        self.assertEqual(result_df.at[test_timestamp, "public_holiday"], 0)
        self.assertEqual(result_df.at[test_timestamp, "lockdown"], 0)
