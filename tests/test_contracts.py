import unittest
from enda.contracts import Contracts
import pathlib
import os
from enda.contracts import Contracts


class TestContracts(unittest.TestCase):

    EXAMPLE_A_DIR = os.path.join(pathlib.Path(__file__).parent.absolute(), "example_a")
    CONTRACTS_PATH = os.path.join(EXAMPLE_A_DIR, "contracts.csv")

    def test_read_contracts_from_file(self):
        contracts = Contracts.read_contracts_from_file(TestContracts.CONTRACTS_PATH)
        self.assertEqual(contracts.shape, (7, 12))

    def test_check_contracts_dates(self):
        pass

    def test_compute_portfolio_by_day(self):
        pass

    def test_get_daily_portfolio_between_dates(self):
        pass

    def test_forecast_using_trend(self):
        pass
