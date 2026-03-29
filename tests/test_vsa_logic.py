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
        # Note: indicators.py likely expects certain formats
        high = np.array([100.0, 110.0, 120.0])
        low = np.array([90.0, 100.0, 110.0])
        close = np.array([95.0, 105.0, 115.0])
        
        spread = calculate_spread(high, low)
        self.assertTrue(np.array_equal(spread, [10.0, 10.0, 10.0]))
        
        pos = calculate_close_position(close, low, spread)
        self.assertTrue(np.array_equal(pos, [0.5, 0.5, 0.5]))
        
        # calculate_moving_average might be slightly different in implementation
        # Let's verify its behavior
        ma = calculate_moving_average(close, period=2)
        # Assuming MA is same length as input
        self.assertAlmostEqual(ma[1], 100.0)

    def test_vsa_matcher(self):
        """Verify VSA signal detection using match_signal."""
        # Test Buying Climax
        # v_ratio > 2.0 and s_ratio > 1.8, is_up=True, close_pos < 0.3
        result = VSAClassicMatcher.match_signal(
            vol=300, vol_ma=100, spr=30, spr_ma=10,
            close_pos=0.2, price_trend="Up", is_up=True
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.pattern_name, "Buying Climax")
        self.assertEqual(result.sentiment, "Bearish")
        
        # Test Selling Climax
        # v_ratio > 2.0 and s_ratio > 1.8, is_up=False, close_pos > 0.7
        result = VSAClassicMatcher.match_signal(
            vol=300, vol_ma=100, spr=30, spr_ma=10,
            close_pos=0.8, price_trend="Down", is_up=False
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.pattern_name, "Selling Climax")
        self.assertEqual(result.sentiment, "Bullish")

    def test_anomaly_v2_matcher(self):
        """Verify Anomaly V2 OHLC classification logic."""
        # Test Silent Accumulation: drop_pct > 20, close > open, close > prev_close, close_pos > 0.7
        # close_pos = (103 - 95) / 10 = 0.8 > 0.7
        ohlc = {'open': 100, 'high': 105, 'low': 95, 'close': 103}
        prev_close = 101
        prev_open = 100
        drop_pct = 30
        
        result = AnomalyV2Matcher.classify(drop_pct, ohlc, prev_close, prev_open)
        self.assertEqual(result.pattern_name, "Silent Accumulation")
        self.assertEqual(result.sentiment, "Bullish")
        
        # Test Continuation Dump: drop_pct < -20, close < open, close < prev_close
        ohlc = {'open': 100, 'high': 102, 'low': 90, 'close': 95}
        prev_close = 101
        prev_open = 100
        drop_pct = -30
        result = AnomalyV2Matcher.classify(drop_pct, ohlc, prev_close, prev_open)
        self.assertEqual(result.pattern_name, "Continuation Dump")
        self.assertEqual(result.sentiment, "Bearish")

if __name__ == "__main__":
    unittest.main()
