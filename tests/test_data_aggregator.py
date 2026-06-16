"""
test_data_aggregator.py
Unit tests for the DataAggregator service.
"""

import unittest
from pathlib import Path
import shutil
import tempfile
from src.services.reporting.data_aggregator import DataAggregator

class TestDataAggregator(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_aggregate_pipeline_stats_deduplication(self):
        """Verify that duplicate CSV files for same symbols are counted uniquely."""
        # Create some files
        files_to_create = [
            "SBIN_1Y_20260611.csv",
            "SBIN_1Y_20260613.csv",
            "TCS_1Y_20260613.csv",
            "TATASTEEL_1Y_20260329.csv",
            "TATASTEEL_1Y_20260613.csv",
        ]
        for fname in files_to_create:
            (self.test_dir / fname).touch()

        aggregator = DataAggregator(self.test_dir)
        stats = aggregator.aggregate_pipeline_stats()
        
        # SBIN, TCS, TATASTEEL = 3 unique symbols
        self.assertEqual(stats["extraction_count"], 3)

    def test_get_symbol_lists_deduplication(self):
        """Verify that get_symbol_lists returns deduplicated unique symbol list."""
        files_to_create = [
            "SBIN_1Y_20260611.csv",
            "SBIN_1Y_20260613.csv",
            "TCS_1Y_20260613.csv",
            "TATASTEEL_1Y_20260329.csv",
            "TATASTEEL_1Y_20260613.csv",
        ]
        for fname in files_to_create:
            (self.test_dir / fname).touch()

        aggregator = DataAggregator(self.test_dir)
        lists = aggregator.get_symbol_lists()
        
        expected_extraction = ["SBIN", "TATASTEEL", "TCS"]
        self.assertEqual(lists["extraction"], expected_extraction)

if __name__ == "__main__":
    unittest.main()
