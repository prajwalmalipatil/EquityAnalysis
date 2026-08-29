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

    def test_get_weekly_volume_trap_details_boundary_rounding(self):
        """Regression test for Finding 1: Ensure body ratio just below 0.30 (e.g. 0.29996) is extracted."""
        import pandas as pd
        from src.constants import vsa_constants as const

        filter_dir = self.test_dir / const.WEEKLY_VOLUME_TRAP_FILTER_DIR_NAME
        filter_dir.mkdir(parents=True)

        dates = pd.date_range("2026-06-01", periods=10, freq="B")
        df = pd.DataFrame({
            "Date": dates,
            "Open": [100.0] * 5 + [100.0] * 5,
            "High": [300.0] * 5 + [200.0] * 5,
            "Low": [100.0] * 5 + [100.0] * 5,
            # Week 1 close = 200, Week 2 close = 129.996 (Spread=100, Body=29.996 -> Body/Spread=0.29996 < 0.30)
            "Close": [200.0] * 5 + [100.0] * 4 + [129.996],
            "Volume": [200] * 5 + [400] * 5,
        })

        file_path = filter_dir / "TEST_VSA.xlsx"
        with pd.ExcelWriter(file_path) as writer:
            df.to_excel(writer, sheet_name="VSA_Analysis", index=False)

        aggregator = DataAggregator(self.test_dir)
        details = aggregator.get_weekly_volume_trap_details("TEST")
        self.assertIsNotNone(details, "Stock with body ratio 0.29996 should not be dropped by rounding")
        self.assertEqual(details["symbol"], "TEST")
        self.assertAlmostEqual(details["body_ratio"], 0.3, places=3)
        self.assertEqual(details["sentiment"], "Bearish")

    def test_get_monthly_volume_trap_details_boundary_rounding(self):
        """Regression test for Finding 1 (Monthly): Ensure body ratio just below 0.30 is extracted."""
        import pandas as pd
        from src.constants import vsa_constants as const

        filter_dir = self.test_dir / const.MONTHLY_VOLUME_TRAP_FILTER_DIR_NAME
        filter_dir.mkdir(parents=True)

        # 2 completed months (e.g. 2026-05 and 2026-06)
        dates_m1 = pd.date_range("2026-05-01", "2026-05-15", freq="B")
        dates_m2 = pd.date_range("2026-06-01", "2026-06-15", freq="B")
        dates = dates_m1.append(dates_m2)

        df = pd.DataFrame({
            "Date": dates,
            "Open": [100.0] * len(dates_m1) + [100.0] * len(dates_m2),
            "High": [300.0] * len(dates_m1) + [200.0] * len(dates_m2),
            "Low": [100.0] * len(dates_m1) + [100.0] * len(dates_m2),
            "Close": [200.0] * len(dates_m1) + [100.0] * (len(dates_m2) - 1) + [129.996],
            "Volume": [100] * len(dates_m1) + [200] * len(dates_m2),
        })

        file_path = filter_dir / "MONTHLY_TEST_VSA.xlsx"
        with pd.ExcelWriter(file_path) as writer:
            df.to_excel(writer, sheet_name="VSA_Analysis", index=False)

        aggregator = DataAggregator(self.test_dir)
        details = aggregator.get_monthly_volume_trap_details("MONTHLY_TEST")
        self.assertIsNotNone(details, "Monthly stock with body ratio 0.29996 should not be dropped")
        self.assertEqual(details["symbol"], "MONTHLY_TEST")

    def test_get_age_again_details_daily(self):
        """Test daily AgeAgain absorption extraction."""
        import pandas as pd
        from src.constants import vsa_constants as const

        aa_dir = self.test_dir / const.AGE_AGAIN_FILTER_DIR_NAME
        aa_dir.mkdir(parents=True)

        df = pd.DataFrame({
            "Date": ["2026-06-01", "2026-06-02"],
            "Open": [100.0, 105.0],
            "High": [120.0, 115.0],
            "Low": [95.0, 100.0],
            "Close": [110.0, 112.0],
            "Spread": [25.0, 15.0],        # contraction: 15 < 25
            "Volume": [1000, 2000],        # surge: 2000 > 1000
            "Close_Position": [0.6, 0.8],
        })

        file_path = aa_dir / "INFY_VSA.xlsx"
        with pd.ExcelWriter(file_path) as writer:
            df.to_excel(writer, sheet_name="VSA_Analysis", index=False)

        aggregator = DataAggregator(self.test_dir)
        details = aggregator.get_age_again_details("INFY")
        self.assertIsNotNone(details)
        self.assertEqual(details["symbol"], "INFY")
        self.assertEqual(details["scenario"], "Vol_Surge_Spread_Contraction")
        self.assertEqual(details["sentiment"], "Bullish")
        self.assertEqual(details["label"], "Absorption Signal")

    def test_get_weekly_age_again_details(self):
        """Test weekly AgeAgain extraction."""
        import pandas as pd
        from src.constants import vsa_constants as const

        aa_dir = self.test_dir / const.WEEKLY_AGE_AGAIN_FILTER_DIR_NAME
        aa_dir.mkdir(parents=True)

        dates = pd.date_range("2026-06-01", periods=10, freq="B")
        df = pd.DataFrame({
            "Date": dates,
            "Open": [100.0] * 10,
            "High": [130.0] * 5 + [115.0] * 5,  # W1 spread=30, W2 spread=15
            "Low": [100.0] * 10,
            "Close": [115.0] * 10,
            "Volume": [100] * 5 + [300] * 5,    # W1 vol=500, W2 vol=1500
        })

        file_path = aa_dir / "TATA_VSA.xlsx"
        with pd.ExcelWriter(file_path) as writer:
            df.to_excel(writer, sheet_name="VSA_Analysis", index=False)

        aggregator = DataAggregator(self.test_dir)
        details = aggregator.get_weekly_age_again_details("TATA")
        self.assertIsNotNone(details)
        self.assertEqual(details["symbol"], "TATA")
        self.assertEqual(details["scenario"], "Vol_Surge_Spread_Contraction")
        self.assertEqual(details["sentiment"], "Bullish")

    def test_get_monthly_age_again_details(self):
        """Test monthly AgeAgain effort without result extraction."""
        import pandas as pd
        from src.constants import vsa_constants as const

        aa_dir = self.test_dir / const.MONTHLY_AGE_AGAIN_FILTER_DIR_NAME
        aa_dir.mkdir(parents=True)

        dates_m1 = pd.date_range("2026-05-01", "2026-05-15", freq="B")
        dates_m2 = pd.date_range("2026-06-01", "2026-06-15", freq="B")
        dates = dates_m1.append(dates_m2)

        df = pd.DataFrame({
            "Date": dates,
            "Open": [100.0] * len(dates),
            "High": [110.0] * len(dates_m1) + [140.0] * len(dates_m2), # M1 spread=10, M2 spread=40 (expansion)
            "Low": [100.0] * len(dates),
            "Close": [105.0] * len(dates),
            "Volume": [300] * len(dates_m1) + [100] * len(dates_m2),   # M1 vol > M2 vol (drop)
        })

        file_path = aa_dir / "RELIANCE_VSA.xlsx"
        with pd.ExcelWriter(file_path) as writer:
            df.to_excel(writer, sheet_name="VSA_Analysis", index=False)

        aggregator = DataAggregator(self.test_dir)
        details = aggregator.get_monthly_age_again_details("RELIANCE")
        self.assertIsNotNone(details)
        self.assertEqual(details["symbol"], "RELIANCE")
        self.assertEqual(details["scenario"], "Vol_Drop_Spread_Expansion")
        self.assertEqual(details["sentiment"], "Bearish")
        self.assertEqual(details["label"], "Effort Without Result")

if __name__ == "__main__":
    unittest.main()
