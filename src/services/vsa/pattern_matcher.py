"""
pattern_matcher.py
Advanced VSA and Anomaly V2 Pattern Recognition.
Includes Classic VSA (Climax, No Demand, Test, Upthrust) 
and Anomaly V2 (OHLC Classification).
"""

from typing import Dict, Optional
from src.models.vsa_models import VSAClassification, AnomalyClassification

class VSAClassicMatcher:
    """Matches core VSA signals based on volume and spread ratios."""
    
    @staticmethod
    def match_signal(vol: float, vol_ma: float, spr: float, spr_ma: float, 
                     close_pos: float, price_trend: str, is_up: bool) -> Optional[VSAClassification]:
        
        v_ratio = vol / vol_ma if vol_ma > 0 else 1.0
        s_ratio = spr / spr_ma if spr_ma > 0 else 1.0
        
        # 1. Climax (High Volume, High Spread)
        if v_ratio > 1.8 and s_ratio > 1.5:
            if is_up and close_pos < 0.4:
                return VSAClassification(
                    pattern_name="Buying Climax", sentiment="Bearish",
                    effort_vs_result="Effort without Result", confidence=0.85,
                    description="Institutional selling hidden in high-volume rally. Price unable to close high."
                )
            if not is_up and close_pos > 0.6:
                return VSAClassification(
                    pattern_name="Selling Climax", sentiment="Bullish",
                    effort_vs_result="Demand absorbing Supply", confidence=0.90,
                    description="Massive institutional absorption of panic selling. High probability bottom."
                )

        # 2. Test / No Supply (Low Volume, Low Spread)
        if v_ratio < 0.8 and s_ratio < 0.8:
            if not is_up and close_pos > 0.6:
                return VSAClassification(
                    pattern_name="Test", sentiment="Bullish",
                    effort_vs_result="No Supply", confidence=0.75,
                    description="Successful test of supply. Low volume indicates no institutional selling pressure."
                )
            if is_up and close_pos < 0.4:
                return VSAClassification(
                    pattern_name="No Demand", sentiment="Bearish",
                    effort_vs_result="Lack of Interest", confidence=0.70,
                    description="Price rising on low volume with weak close. Lack of institutional participation."
                )

        # 3. Upthrust (High Volume, Weak Close)
        if v_ratio > 1.2 and is_up and close_pos < 0.3:
            return VSAClassification(
                pattern_name="Upthrust", sentiment="Bearish",
                effort_vs_result="Hidden Weakness", confidence=0.80,
                description="Failed attempt to break higher. High volume with weak close indicates supply presence."
            )

        return None

class AnomalyV2Matcher:
    """Deep OHLC Classification for Volume Anomalies."""
    
    @staticmethod
    def classify(drop_pct: float, ohlc: Dict, prev_close: float, prev_open: float) -> AnomalyClassification:
        close = ohlc['close']
        open_ = ohlc['open']
        high = ohlc['high']
        low = ohlc['low']
        spread = high - low
        close_pos = (close - low) / spread if spread > 0 else 0.5
        
        # Bearish: Dump or Trap
        if drop_pct < -20: # Vol dropped significantly
            if close < open_ and close < prev_close:
                return AnomalyClassification(
                    pattern_name="Continuation Dump", sentiment="Bearish", confidence=0.70
                )
            if close > open_ and close_pos < 0.4:
                return AnomalyClassification(
                    pattern_name="Failed Rally", sentiment="Bearish", confidence=0.65
                )
        
        # Bullish: Accumulation or Absorption
        if drop_pct > 20: # Vol spiked significantly
            if close > open_ and close > prev_close and close_pos > 0.7:
                 return AnomalyClassification(
                    pattern_name="Silent Accumulation", sentiment="Bullish", confidence=0.80
                )
            if close < open_ and close_pos > 0.6:
                 return AnomalyClassification(
                    pattern_name="Bear Trap", sentiment="Bullish", confidence=0.75
                )

        if drop_pct < -5:
            return AnomalyClassification(
                pattern_name="Neutral Contraction", sentiment="Neutral", confidence=0.50
            )
            
        return AnomalyClassification(pattern_name="Neutral", sentiment="Neutral", confidence=0.0)
