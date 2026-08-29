"""
test_consensus_engine.py
Unit tests for Multi-Timeframe Consensus Engine scoring and classification.
"""

import unittest
from pathlib import Path
import tempfile
import shutil
from src.services.vsa.consensus_engine_service import ConsensusEngineService
from src.models.vsa_models import EigenClassification

class TestConsensusEngine(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_neutral_sentiment_scoring(self):
        """Verify that Neutral sentiments are scored as 0.0 and do not default to Bearish."""
        service = ConsensusEngineService(self.test_dir)

        # Case 1: All timeframes are Neutral
        daily = [EigenClassification(
            symbol="TEST", gap_direction="Gap-Up", close_band="Strong",
            label="Label", sentiment="Neutral", volume_surge_pct=0.0,
            t_close_position=0.0, t1_close_position=0.0, delta_cp=0.0
        )]
        weekly = [EigenClassification(
            symbol="TEST", gap_direction="Gap-Up", close_band="Strong",
            label="Label", sentiment="Neutral", volume_surge_pct=0.0,
            t_close_position=0.0, t1_close_position=0.0, delta_cp=0.0
        )]
        monthly = [EigenClassification(
            symbol="TEST", gap_direction="Gap-Up", close_band="Strong",
            label="Label", sentiment="Neutral", volume_surge_pct=0.0,
            t_close_position=0.0, t1_close_position=0.0, delta_cp=0.0
        )]

        ratings = service.compute_consensus(daily, weekly, monthly)
        self.assertEqual(len(ratings), 1)
        r = ratings[0]
        self.assertEqual(r.score_percentage, 0.0)
        self.assertEqual(r.monthly_score, 0.0)
        self.assertEqual(r.weekly_score, 0.0)
        self.assertEqual(r.daily_score, 0.0)
        self.assertEqual(r.consensus_label, "Mixed Trend")
        self.assertEqual(r.star_rating, 3)

    def test_mixed_sentiment_scoring(self):
        """Verify mixed sentiments including Bullish, Bearish, and Neutral."""
        service = ConsensusEngineService(self.test_dir)

        # Case 2: Daily Bullish (25.0), Weekly Neutral (0.0), Monthly Bearish (-40.0)
        # Expected score: 25.0 + 0.0 - 40.0 = -15.0
        daily = [EigenClassification(
            symbol="TEST2", gap_direction="Gap-Up", close_band="Strong",
            label="Label", sentiment="Bullish", volume_surge_pct=0.0,
            t_close_position=0.0, t1_close_position=0.0, delta_cp=0.0
        )]
        weekly = [EigenClassification(
            symbol="TEST2", gap_direction="Gap-Up", close_band="Strong",
            label="Label", sentiment="Neutral", volume_surge_pct=0.0,
            t_close_position=0.0, t1_close_position=0.0, delta_cp=0.0
        )]
        monthly = [EigenClassification(
            symbol="TEST2", gap_direction="Gap-Up", close_band="Strong",
            label="Label", sentiment="Bearish", volume_surge_pct=0.0,
            t_close_position=0.0, t1_close_position=0.0, delta_cp=0.0
        )]

        ratings = service.compute_consensus(daily, weekly, monthly)
        self.assertEqual(len(ratings), 1)
        r = ratings[0]
        self.assertEqual(r.score_percentage, -15.0)
        self.assertEqual(r.daily_score, 25.0)
        self.assertEqual(r.weekly_score, 0.0)
        self.assertEqual(r.monthly_score, -40.0)
        self.assertEqual(r.consensus_label, "Cautious Bearish")
        self.assertEqual(r.star_rating, 2)

    def test_score_sentiment_pure_function(self):
        """Verify that _score_sentiment can be imported and executed as a pure function."""
        from src.services.vsa.consensus_engine_service import _score_sentiment
        self.assertEqual(_score_sentiment("Bullish", 10.0), 10.0)
        self.assertEqual(_score_sentiment("Bearish", 10.0), -10.0)
        self.assertEqual(_score_sentiment("Neutral", 10.0), 0.0)
        self.assertEqual(_score_sentiment("None", 10.0), 0.0)

    def test_age_again_supplemental_signal(self):
        """Verify that AgeAgain acts as a supplemental signal when Eigen is missing."""
        from src.models.vsa_models import AgeAgainClassification

        service = ConsensusEngineService(self.test_dir)

        # No Eigen signals for SYMBOL_AA
        daily_eigen = []
        weekly_eigen = []
        monthly_eigen = []

        aa_daily = [AgeAgainClassification(
            symbol="SYMBOL_AA", scenario="Vol_Surge_Spread_Contraction",
            label="Absorption Signal", sentiment="Bullish",
            t_volume=2000, t1_volume=1000, volume_pct=100.0,
            t_spread=10.0, t1_spread=20.0, spread_pct=-50.0,
            t_close=110.0, t_open=100.0, t_close_position=0.8
        )]

        ratings = service.compute_consensus(
            daily_eigen, weekly_eigen, monthly_eigen,
            age_again_daily=aa_daily
        )
        self.assertEqual(len(ratings), 1)
        r = ratings[0]
        self.assertEqual(r.symbol, "SYMBOL_AA")
        self.assertEqual(r.daily_sentiment, "Bullish")
        self.assertEqual(r.daily_score, 25.0)
        self.assertEqual(r.score_percentage, 25.0)
        self.assertEqual(r.consensus_label, "Cautious Bullish")


if __name__ == "__main__":
    unittest.main()
