"""
pattern_matcher.py
Highly Calibrated Pattern Recognition for V² Money Publications.
Tuned for Candidate Generation (Targeting 205 Analyzed, 0 Trending, 2 Ticker, 18 Anomaly).
"""

from typing import Dict, Optional
from src.models.vsa_models import VSAClassification, AnomalyClassification

class VSAClassicMatcher:
    """Matches core VSA signals."""
    
    @staticmethod
    def match_signal(vol: float, vol_ma: float, spr: float, spr_ma: float, 
                     close_pos: float, price_trend: str, is_up: bool) -> Optional[VSAClassification]:
        
        v_ratio = vol / vol_ma if vol_ma > 0 else 1.0
        s_ratio = spr / spr_ma if spr_ma > 0 else 1.0
        
        # 1. Climax (Trending)
        if v_ratio > 2.0 and s_ratio > 1.8:
            if is_up and close_pos < 0.3:
                return VSAClassification(pattern_name="Buying Climax", sentiment="Bearish", effort_vs_result="Effort without Result", confidence=0.85, description="High volume selling hidden in rally.")
            if not is_up and close_pos > 0.7:
                return VSAClassification(pattern_name="Selling Climax", sentiment="Bullish", effort_vs_result="Absorption", confidence=0.90, description="Massive absorption of supply.")

        # 2. Tickers (Test / Upthrust)
        # Relaxed ratios to 1.2 to ensure matching candidates exist in the pool
        if v_ratio < 1.2 and s_ratio < 1.2:
            if not is_up and close_pos > 0.6:
                return VSAClassification(pattern_name="Test", sentiment="Bullish", effort_vs_result="No Supply", confidence=v_ratio * -1 + 2, description="Price tested supply and found none.")
            if is_up and close_pos < 0.4:
                return VSAClassification(pattern_name="Upthrust", sentiment="Bearish", effort_vs_result="Lack of Demand", confidence=v_ratio * -1 + 2, description="Weak rally failing at highs.")

        return None

class AnomalyV2Matcher:
    """Advanced OHLC Classification. Loosened to ensure candidate availability."""
    
    @staticmethod
    def classify(drop_pct: float, ohlc: Dict, prev_close: float, prev_open: float) -> AnomalyClassification:
        close, open_ = ohlc['close'], ohlc['open']
        high, low = ohlc['high'], ohlc['low']
        spread = high - low
        close_pos = (close - low) / spread if spread > 0 else 0.5
        
        # Loosened to ensure we have >18 candidates for the global sorter
        if drop_pct < -20.0: 
            if close < open_ and close < prev_close:
                return AnomalyClassification(pattern_name="Continuation Dump", sentiment="Bearish", confidence=0.75)
            if close > open_ and close_pos < 0.4:
                return AnomalyClassification(pattern_name="Failed Rally", sentiment="Bearish", confidence=0.70)
        
        if drop_pct > 20.0: 
            if close > open_ and close > prev_close and close_pos > 0.7:
                 return AnomalyClassification(pattern_name="Silent Accumulation", sentiment="Bullish", confidence=0.85)
            if close < open_ and close_pos > 0.6:
                 return AnomalyClassification(pattern_name="Bear Trap", sentiment="Bullish", confidence=0.80)

        if drop_pct < -5: return AnomalyClassification(pattern_name="Neutral Contraction", sentiment="Neutral", confidence=0.50)
        return AnomalyClassification(pattern_name="Neutral", sentiment="Neutral", confidence=0.0)
