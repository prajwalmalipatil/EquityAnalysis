"""
test_event_study.py
Unit tests for EventStudyEngine quantitative correlation and return windows.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
import pandas as pd

from src.services.macro_intelligence.event_study import EventStudyEngine
from src.services.macro_intelligence.models import (
    DerivedData,
    EventMetadata,
    ImpactAnalysis,
    MacroEvent,
    OfficialData,
)


class TestEventStudyEngine(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_event_study_calculation(self):
        """Verify returns window computation for index assets."""
        # 1. Create mock NIFTY 50 price data CSV
        # We need dates around the event date (e.g. 2023-01-11)
        dates = pd.date_range("2023-01-01", periods=30)
        # T0 will be at index 10 (2023-01-11). Close price starts at 100 and increases by 1 each day.
        prices = [100.0 + i for i in range(30)]
        df_prices = pd.DataFrame({"Date": dates, "Close": prices})

        # Save as safe symbol format: NIFTY50_1Y_20230101.csv
        file_path = self.test_dir / "NIFTY50_1Y_20230101.csv"
        df_prices.to_csv(file_path, index=False)

        # 2. Setup engine
        engine = EventStudyEngine(self.test_dir)

        # 3. Create mock MacroEvent at 2023-01-11
        event = MacroEvent(
            event_id="test_event_1",
            official_data=OfficialData(
                title="Event",
                content="Event content",
                category="Liquidity",
                publication_date="2023-01-11T10:00:00Z",
                source="RBI",
                official_url="http://rbi.org.in",
            ),
            derived_data=DerivedData(
                impact=ImpactAnalysis(
                    asset_classes=["Equity"],
                    sectors=["All"],
                    securities=["NIFTY 50"],
                    horizon="1-5 days",
                    direction="Positive",
                    severity="Medium",
                    importance=50,
                    confidence=50,
                )
            ),
            metadata=EventMetadata(
                processing_state="NEW",
                lifecycle_status="ACTIVE",
                created_at="2023-01-11T10:00:00Z",
                updated_at="2023-01-11T10:00:00Z",
            ),
        )

        # 4. Process
        study = engine.process(event)

        # Assert returns are computed
        self.assertIn("NIFTY 50", study.index_returns)
        returns = study.index_returns["NIFTY 50"]

        # T0 is index 10: price = 110.0
        # T-5 is index 5: price = 105.0. Return = ((110 / 105) - 1.0) * 100 = 4.76%
        self.assertAlmostEqual(returns.t_minus_5, 4.76, places=1)
        # T+5 is index 15: price = 115.0. Return = ((115 / 110) - 1.0) * 100 = 4.55%
        self.assertAlmostEqual(returns.t_plus_5, 4.55, places=1)


if __name__ == "__main__":
    unittest.main()
