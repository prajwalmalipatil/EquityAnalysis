"""
test_vsa_logic.py
Unit tests for mathematical indicators and pattern matching logic.
"""

import unittest
import numpy as np
import pandas as pd
from src.services.vsa.indicators import (
    calculate_spread, calculate_close_position, 
    calculate_moving_average, calculate_price_trend
)
from src.services.vsa.pattern_matcher import VSAClassicMatcher, AnomalyV2Matcher

class TestVSALogic(unittest.TestCase):
    def test_indicators(self):
        """Verify basic VSA math calculations."""
        high = np.array([100, 110, 120])
        low = np.array([90, 100, 110])
        close = np.array([95, 105, 115])
        
        spread = calculate_spread(high, low)
        self.assertTrue(np.array_equal(spread, [10, 10, 10]))
        
        pos = calculate_close_position(close, low, spread)
        self.assertTrue(np.array_equal(pos, [0.5, 0.5, 0.5]))
        
        ma = calculate_moving_average(close, period=2)
        self.assertEqual(ma[0], 95)
        self.assertEqual(ma[1], 100)

    def test_classic_vsa_matcher(self):
        """Verify classic VSA signal detection."""
        # Test No Demand
        signal = VSAClassicMatcher.match_no_demand(
            vol=50, vol_ma=100, is_up=True, trend=1
        )
        self.assertEqual(signal, "No Demand (Bearish Weakness)")
        
        # Test Climax (Bullish)
        signal = VSAClassicMatcher.match_climax(
            vol=300, vol_ma=100, spread=30, spread_ma=10, 
            close_pos=0.1, trend=-1
        )
        self.assertEqual(signal, "Selling Climax (Bullish Reversal)")

    def test_anomaly_v2_matcher(self):
        """Verify Anomaly V2 OHLC classification."""
        # Test Silent Accumulation (70.5% Win Rate)
        ohlc = {'open': 100, 'high': 105, 'low': 95, 'close': 97}
        prev_close = 96
        prev_open = 95
        drop_pct = -60
        
        result = AnomalyV2Matcher.classify(drop_pct, ohlc, prev_close, prev_open)
        self.assertEqual(result.pattern_name, "Silent Accumulation")
        self.assertEqual(result.sentiment, "Bullish")

if __name__ == "__main__":
    unittest.main()
