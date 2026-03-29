"""
pattern_matcher.py
Highly Calibrated Pattern Recognition for V² Money Publications.
Tuned for Candidate Generation (Targeting 205 Analyzed, 0 Trending, 2 Ticker, 18 Anomaly).
"""

from typing import Dict, Optional
from src.models.vsa_models import VSAClassification, AnomalyClassification
from src.constants import vsa_constants as const

class VSAClassicMatcher:
    """Matches core VSA signals."""
    
    @staticmethod
    def match_signal(vol: float, vol_ma: float, spr: float, spr_ma: float, 
                     close_pos: float, trend: int, is_up: bool, 
                     is_down: bool, prev_up: bool, 
                     support: float = 0.0, low: float = 0.0) -> Optional[VSAClassification]:
        
        v_ultra = vol_ma * const.ULTRA_HIGH_VOLUME
        v_high = vol_ma * const.HIGH_VOLUME
        v_low = vol_ma * const.LOW_VOLUME
        s_wide = spr_ma * const.WIDE_SPREAD
        s_norm = spr_ma * const.NORMAL_SPREAD
        s_narr = spr_ma * const.NARROW_SPREAD
        
        # 1. Climax (Ultra-High Volume + Wide Spread)
        if vol > v_ultra and spr > s_wide:
            if close_pos <= const.WEAK_CLOSE and trend <= 0:
                return VSAClassification(pattern_name="Selling Climax (Bullish Reversal)", sentiment="Bullish", effort_vs_result="Absorption", confidence=0.90, description="Massive absorption of supply in downtrend.")
            if close_pos >= const.STRONG_CLOSE and trend >= 0:
                return VSAClassification(pattern_name="Buying Climax (Bearish Reversal)", sentiment="Bearish", effort_vs_result="Effort without Result", confidence=0.85, description="Buying climax - exhausting demand at highs.")

        # 2. Upthrust (Bearish)
        if vol > v_high and spr > s_norm and close_pos <= const.WEAK_CLOSE and trend >= 0 and prev_up:
            return VSAClassification(pattern_name="Upthrust (Bearish)", sentiment="Bearish", effort_vs_result="Supply Entering", confidence=0.80, description="High volume rejection at highs.")

        # 3. No Demand (Bearish Weakness)
        if vol < v_low and is_up and trend >= 0:
             return VSAClassification(pattern_name="No Demand (Bearish Weakness)", sentiment="Bearish", effort_vs_result="Lack of Demand", confidence=0.70, description="Up bar on low volume - lack of buying interest.")

        # 4. Stopping Volume (Potential Reversal)
        if vol > v_high and spr < s_narr and const.MID_CLOSE_LOW <= close_pos <= const.MID_CLOSE_HIGH:
            return VSAClassification(pattern_name="Stopping Volume (Potential Reversal)", sentiment="Bullish", effort_vs_result="Absorption", confidence=0.75, description="High volume absorption in narrow range.")

        # 5. Test (Bullish) - Requires support context for Ticker parity
        if vol < v_low and is_down:
            # Legacy requires low near 20-day support
            if low <= support * (1 + const.SUPPORT_TOLERANCE):
                return VSAClassification(pattern_name="Test (Bullish)", sentiment="Bullish", effort_vs_result="No Supply", confidence=0.80, description="Low volume test of supply.")

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
