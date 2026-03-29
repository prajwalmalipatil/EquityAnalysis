"""
pattern_matcher.py
Specialized logic for identifying Volume Spread Analysis (VSA) patterns 
and Anomaly V2 OHLC classifications.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from src.models.vsa_models import AnomalyClassification, VSAClassification
from src.constants import vsa_constants as const

class VSAClassicMatcher:
    """Matches core VSA signals based on volume and spread ratios."""
    
    @staticmethod
    def match_climax(vol: float, vol_ma: float, spread: float, spread_ma: float, 
                     close_pos: float, trend: int) -> Optional[VSAClassification]:
        """Matches Buying/Selling Climax patterns."""
        is_ultra_vol = vol > vol_ma * const.ULTRA_HIGH_VOLUME
        is_wide_spread = spread > spread_ma * const.WIDE_SPREAD
        
        if not (is_ultra_vol and is_wide_spread):
            return None
            
        if close_pos <= const.WEAK_CLOSE and trend <= 0:
            return VSAClassification(
                pattern_name="Selling Climax",
                effort_vs_result="Absorption",
                sentiment="Bullish Reversal",
                confidence=0.85,
                description="Smart money absorbing supply at local bottoms."
            )
        if close_pos >= const.STRONG_CLOSE and trend >= 0:
            return VSAClassification(
                pattern_name="Buying Climax",
                effort_vs_result="Distribution",
                sentiment="Bearish Reversal",
                confidence=0.85,
                description="Smart money offloading into retail FOMO."
            )
        return None

    @staticmethod
    def match_no_demand(vol: float, vol_ma: float, is_up: bool, trend: int) -> Optional[VSAClassification]:
        """Matches No Demand (Bearish Weakness)."""
        if vol < vol_ma * const.LOW_VOLUME and is_up and trend >= 0:
            return VSAClassification(
                pattern_name="No Demand",
                effort_vs_result="No_Demand",
                sentiment="Bearish Weakness",
                confidence=0.70,
                description="Asset is showing structural weakness with lack of interest from buyers."
            )
        return None

    @staticmethod
    def match_stopping_volume(vol: float, vol_ma: float, spread: float, 
                               spread_ma: float, close_pos: float) -> Optional[VSAClassification]:
        """Matches Stopping Volume (Potential Reversal)."""
        is_high_vol = vol > vol_ma * const.HIGH_VOLUME
        is_narrow_spread = spread < spread_ma * const.NARROW_SPREAD
        is_mid_close = const.MID_CLOSE_LOW <= close_pos <= const.MID_CLOSE_HIGH
        
        if is_high_vol and is_narrow_spread and is_mid_close:
            return VSAClassification(
                pattern_name="Stopping Volume",
                effort_vs_result="Absorption",
                sentiment="Potential Reversal",
                confidence=0.75,
                description="Strong effort to stop the fall; smart money buying detected."
            )
        return None


class AnomalyV2Matcher:
    """
    Classifies volume drops into high-confidence patterns using OHLC signatures.
    Logic extracted from the V2 backtest results.
    """
    
    @staticmethod
    def match_volume_spike(current: float, previous: float, threshold: float = 2.0) -> Optional[str]:
        """Simple spike detection (>200% of previous)."""
        if previous <= 0: return None
        if current > previous * threshold:
            return "Volume Spike"
        return None

    @staticmethod
    def match_volume_drop(current: float, previous: float, threshold: float = 0.5) -> Optional[str]:
        """Simple drop detection (<50% of previous)."""
        if previous <= 0: return None
        if current < previous * threshold:
            return "Volume Drop"
        return None

    @staticmethod
    def classify(drop_pct: float, ohlc: Dict[str, float], prev_close: float, prev_open: float) -> AnomalyClassification:
        """Main entry point for Anomaly V2 classification."""
        # Derived flags
        is_bearish_bar = ohlc['close'] < ohlc['open']
        close_above_prev = ohlc['close'] > prev_close
        prev_was_up = prev_close > prev_open
        
        total_range = ohlc['high'] - ohlc['low'] or 0.01
        close_pos = (ohlc['close'] - ohlc['low']) / total_range
        gap_pct = ((ohlc['open'] - prev_close) / prev_close) * 100 if prev_close > 0 else 0
        
        # Branch to specific classifiers to keep methods short
        bullish = AnomalyV2Matcher._check_bullish(drop_pct, close_pos, close_above_prev, 
                                                 prev_was_up, gap_pct, is_bearish_bar)
        if bullish:
            return bullish
            
        bearish = AnomalyV2Matcher._check_bearish(drop_pct, close_pos, close_above_prev, 
                                                 prev_was_up, gap_pct, is_bearish_bar)
        if bearish:
            return bearish
            
        return AnomalyClassification(
            pattern_name="Neutral Contraction",
            sentiment="Neutral",
            win_rate=50.0,
            description="Volume dropped but price action is indecisive."
        )

    @staticmethod
    def _check_bullish(drop: float, close_pos: float, cap: bool, pwu: bool, gap: float, ibb: bool) -> Optional[AnomalyClassification]:
        """Logic for Bullish setups."""
        if close_pos < 0.30 and drop < -50 and cap:
            return AnomalyClassification("Silent Accumulation", "Bullish", 70.5, "Smart money absorbing supply at lows.")
        if close_pos < 0.30 and pwu and cap:
            return AnomalyClassification("Pullback Absorption", "Bullish", 64.9, "Absorption after a previous up move.")
        if close_pos > 0.70 and gap < -0.3 and drop < -50:
            return AnomalyClassification("Gap Absorption", "Bullish", 63.6, "Institutional buying on a gap down.")
        if ibb and cap:
            return AnomalyClassification("Bear Trap", "Bullish", 63.0, "Big red candle failed to break prior support.")
        return None

    @staticmethod
    def _check_bearish(drop: float, close_pos: float, cap: bool, pwu: bool, gap: float, ibb: bool) -> Optional[AnomalyClassification]:
        """Logic for Bearish setups."""
        if close_pos < 0.30 and gap < -0.3 and -50 <= drop < -25:
            return AnomalyClassification("Continuation Dump", "Bearish", 32.6, "Further distribution likely.")
        if close_pos > 0.70 and ibb and not cap:
            return AnomalyClassification("Failed Rally", "Bearish", 37.5, "Strong start but weak finish indicates supply.")
        return None
