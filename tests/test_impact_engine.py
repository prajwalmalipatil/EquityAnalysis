"""
test_impact_engine.py
Unit tests for RuleBasedImpactEngine keyword-to-impact classification logic.
"""

import unittest
from pathlib import Path

from src.services.macro_intelligence.impact_engine import RuleBasedImpactEngine
from src.services.macro_intelligence.models import (
    DerivedData,
    EventMetadata,
    MacroEvent,
    OfficialData,
)


class TestImpactEngine(unittest.TestCase):
    def test_repo_rate_hike_impact(self):
        """Verify impact classification for a repo rate hike."""
        engine = RuleBasedImpactEngine(Path("/tmp"))
        event = MacroEvent(
            event_id="test_repo_1",
            official_data=OfficialData(
                title="RBI Announces Repo Rate Hike",
                content="The Monetary Policy Committee decided to hike the repo rate by 25 bps.",
                category="Monetary Policy",
                publication_date="2023-01-01T10:00:00Z",
                source="RBI",
                official_url="http://rbi.org.in",
            ),
            derived_data=DerivedData(),
            metadata=EventMetadata(
                processing_state="NEW",
                lifecycle_status="ACTIVE",
                created_at="2023-01-01T10:00:00Z",
                updated_at="2023-01-01T10:00:00Z",
            ),
        )
        impact = engine.process(event)
        self.assertIn("Equity", impact.asset_classes)
        self.assertIn("Banking", impact.sectors)
        self.assertEqual(impact.direction, "Negative")
        self.assertEqual(impact.severity, "Critical")
        self.assertTrue(len(impact.securities) > 0)
        self.assertIn("HDFCBANK", impact.securities)

    def test_liquidity_injection_impact(self):
        """Verify impact classification for a liquidity injection OMO."""
        engine = RuleBasedImpactEngine(Path("/tmp"))
        event = MacroEvent(
            event_id="test_liq_1",
            official_data=OfficialData(
                title="RBI to inject liquidity via OMO",
                content="The central bank will inject liquidity into the system.",
                category="Liquidity",
                publication_date="2023-01-01T10:00:00Z",
                source="RBI",
                official_url="http://rbi.org.in",
            ),
            derived_data=DerivedData(),
            metadata=EventMetadata(
                processing_state="NEW",
                lifecycle_status="ACTIVE",
                created_at="2023-01-01T10:00:00Z",
                updated_at="2023-01-01T10:00:00Z",
            ),
        )
        impact = engine.process(event)
        self.assertIn("Bonds", impact.asset_classes)
        self.assertEqual(impact.direction, "Positive")
        self.assertEqual(impact.severity, "Medium")


if __name__ == "__main__":
    unittest.main()
