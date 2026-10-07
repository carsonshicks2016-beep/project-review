import tempfile
import unittest
from pathlib import Path

import pandas as pd

import data.fetch_prices as fetch_prices


class FetchPricesCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_cache_dir = fetch_prices.CACHE_DIR
        self.old_db_path = fetch_prices.DB_PATH
        fetch_prices.CACHE_DIR = Path(self.tmp.name)
        fetch_prices.DB_PATH = Path(self.tmp.name) / "prices.db"

    def tearDown(self):
        fetch_prices.CACHE_DIR = self.old_cache_dir
        fetch_prices.DB_PATH = self.old_db_path
        self.tmp.cleanup()

    def test_cache_separates_intervals(self):
        df = pd.DataFrame({
            "date": ["2026-01-01 00:00:00"],
            "ticker": ["BTC-USD"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [10.0],
        })

        fetch_prices.save_to_db(df, interval="5m")

        intraday = fetch_prices.load_from_db("BTC-USD", interval="5m")
        daily = fetch_prices.load_from_db("BTC-USD", interval="1d")

        self.assertEqual(len(intraday), 1)
        self.assertTrue(daily.empty)

    def test_cache_upsert_preserves_other_tickers(self):
        first = pd.DataFrame({
            "date": ["2026-01-01"],
            "ticker": ["BTC-USD"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [10.0],
        })
        second = first.copy()
        second["ticker"] = "ETH-USD"
        second["close"] = 200.5

        fetch_prices.save_to_db(first, interval="1d")
        fetch_prices.save_to_db(second, interval="1d")

        cached = fetch_prices.load_from_db(interval="1d")
        self.assertEqual(set(cached["ticker"]), {"BTC-USD", "ETH-USD"})


if __name__ == "__main__":
    unittest.main()
