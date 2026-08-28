"""
data_aggregator.py
Scans processed VSA folders and aggregates full detail for professional reports.
"""

import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from src.constants import vsa_constants as const
from src.services.vsa.monthly_eigen_filter_service import MonthlyEigenFilterService
from src.services.vsa.monthly_volume_trap_filter_service import MonthlyVolumeTrapFilterService
from src.services.vsa.weekly_eigen_filter_service import WeeklyEigenFilterService
from src.services.vsa.weekly_volume_trap_filter_service import WeeklyVolumeTrapFilterService
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("data-aggregator")

class DataAggregator:
    """
    Service to collect and prepare data for reporting.
    Extracts deep metrics from Excel files for detailed tables.
    """
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def aggregate_pipeline_stats(self) -> Dict:
        """Collects high-level counts for the report summary."""
        return {
            "extraction_count": len(set(f.name.split("_")[0] for f in self.base_dir.glob("*.csv") if "_" in f.name)),
            "vsa": self._count_files(const.RESULTS_DIR_NAME),
            "trending": self._count_files(const.TRENDING_DIR_NAME),
            "anomaly": self._count_files(const.ANOMALY_DIR_NAME),
            "ticker": self._count_files(const.TICKER_DIR_NAME),
            "triggers": self._count_files(const.TRIGGERS_DIR_NAME),
            "eigen_filter": self._count_files(const.EIGEN_FILTER_DIR_NAME),
            "monthly_eigen": self._count_files(const.MONTHLY_EIGEN_FILTER_DIR_NAME),
            "weekly_eigen": self._count_files(const.WEEKLY_EIGEN_FILTER_DIR_NAME),
            "volume_trap": self._count_files(const.VOLUME_TRAP_FILTER_DIR_NAME),
            "weekly_volume_trap": self._count_files(const.WEEKLY_VOLUME_TRAP_FILTER_DIR_NAME),
            "monthly_volume_trap": self._count_files(const.MONTHLY_VOLUME_TRAP_FILTER_DIR_NAME),
        }

    def get_symbol_lists(self) -> Dict[str, List[str]]:
        """Returns simple lists of symbols for categorization."""
        return {
            "extraction": sorted(list(set(f.name.split("_")[0] for f in self.base_dir.glob("*.csv") if "_" in f.name))),
            "vsa": self._get_symbols(const.RESULTS_DIR_NAME),
            "trending": self._get_symbols(const.TRENDING_DIR_NAME),
            "anomaly": self._get_symbols(const.ANOMALY_DIR_NAME),
            "ticker": self._get_symbols(const.TICKER_DIR_NAME),
            "triggers": self._get_symbols(const.TRIGGERS_DIR_NAME),
            "eigen_filter": self._get_symbols(const.EIGEN_FILTER_DIR_NAME),
            "monthly_eigen": self._get_symbols(const.MONTHLY_EIGEN_FILTER_DIR_NAME),
            "weekly_eigen": self._get_symbols(const.WEEKLY_EIGEN_FILTER_DIR_NAME),
            "volume_trap": self._get_symbols(const.VOLUME_TRAP_FILTER_DIR_NAME),
            "weekly_volume_trap": self._get_symbols(const.WEEKLY_VOLUME_TRAP_FILTER_DIR_NAME),
            "monthly_volume_trap": self._get_symbols(const.MONTHLY_VOLUME_TRAP_FILTER_DIR_NAME),
        }

    def get_ticker_details(self, symbol: str) -> Optional[Dict]:
        """Deep extraction for Action Required ticker cards (VSA or Anomaly)."""
        df = self._read_latest(const.TICKER_DIR_NAME, symbol)
        if df is None: return None
        
        latest = df.iloc[-1]
        
        vsa_full = str(latest.get("Signal_Type", "No Signal"))
        if vsa_full != "No Signal":
            pattern_name = vsa_full.split(" (")[0] if " (" in vsa_full else vsa_full
            sentiment = vsa_full.split("(")[1].replace(")", "") if "(" in vsa_full else "Neutral"
            description = str(latest.get("Description", "Classic VSA signal detected."))
            confidence = float(latest.get("Confidence", 0.85))
            effort = str(latest.get("Effort_Result", "Neutral"))
        else:
            pattern_name = str(latest.get("Anomaly_V2", "No Signal"))
            sentiment = "Neutral"
            if any(w in pattern_name for w in ["Accumulation", "Absorption", "Trap"]):
                sentiment = "Bullish"
            elif any(w in pattern_name for w in ["Dump", "Failed"]):
                sentiment = "Bearish"
            
            description = f"Advanced Anomaly Detected: {pattern_name}. Structural shifts observed in volume/price relationship."
            confidence = 0.70
            effort = str(latest.get("Effort_Result", "Neutral"))
            
        return {
            "symbol": symbol,
            "pattern": pattern_name,
            "sentiment": sentiment,
            "effort": effort,
            "description": description,
            "spread_ratio": float(latest.get("Spread", 0) / latest.get("Spread_MA", 1)) if latest.get("Spread_MA", 0) > 0 else 1.0,
            "confidence": confidence
        }

    def get_trigger_details(self, symbol: str) -> Optional[Dict]:
        """Extraction for Vol Contraction + Spread Expansion table."""
        df = self._read_latest(const.TRIGGERS_DIR_NAME, symbol)
        if df is None: return None
        
        latest = df.iloc[-1]
        return {
            "symbol": symbol,
            "prev_vol": int(latest.get("Prev_Volume", 0)),
            "curr_vol": int(latest.get("Volume", 0)),
            "prev_spr": float(latest.get("Prev_Spread", 0)),
            "curr_spr": float(latest.get("Spread", 0)),
            "vol_pct": float(latest.get("Vol_Pct", 0)),
            "spr_pct": float(latest.get("Spr_Pct", 0))
        }

    def get_anomaly_details(self, symbol: str) -> Optional[Dict]:
        """Extraction for Anomaly V2 patterns with sentiment classification."""
        df = self._read_latest(const.ANOMALY_DIR_NAME, symbol)
        if df is None: return None
        
        latest = df.iloc[-1]
        v2_pattern = str(latest.get("Anomaly_V2", "Neutral"))
        sentiment = "Neutral"
        if any(w in v2_pattern for w in ["Accumulation", "Absorption", "Trap"]):
            sentiment = "Bullish"
        elif any(w in v2_pattern for w in ["Dump", "Failed"]):
            sentiment = "Bearish"
            
        return {
            "symbol": symbol,
            "pattern": v2_pattern,
            "prev_vol": int(latest.get("Prev_Volume", 0)),
            "curr_vol": int(latest.get("Volume", 0)),
            "drop_pct": float(latest.get("Vol_Pct", 0)),
            "sentiment": sentiment
        }

    def _count_files(self, folder: str) -> int:
        path = self.base_dir / folder
        return len(list(path.glob("*.xlsx"))) if path.exists() else 0

    def _get_symbols(self, folder: str) -> List[str]:
        path = self.base_dir / folder
        return [f.stem.replace("_VSA", "") for f in path.glob("*.xlsx")] if path.exists() else []

    def _read_latest(self, folder: str, symbol: str) -> Optional[pd.DataFrame]:
        path = self.base_dir / folder / f"{symbol}_VSA.xlsx"
        if not path.exists(): return None
        try:
            return pd.read_excel(path, sheet_name="VSA_Analysis")
        except (OSError, ValueError):
            return None

    def get_eigen_details(self, symbol: str) -> Optional[Dict]:
        """Extracts EigenFilter classification details from a processed Excel file."""
        df = self._read_latest(const.EIGEN_FILTER_DIR_NAME, symbol)
        if df is None or len(df) < 2:
            return None

        latest, prev = df.iloc[-1], df.iloc[-2]
        t_open = float(latest.get("Open", 0))
        t_close = float(latest.get("Close", 0))
        t1_close_val = float(prev.get("Close", 0))
        t_cp = float(latest.get("Close_Position", 0.5))
        t1_cp = float(prev.get("Close_Position", 0.5))
        t_spread = float(latest.get("Spread", 0))
        t_vol = int(latest.get("Volume", 0))
        t1_vol = int(prev.get("Volume", 0))

        gap_dir = "Gap-Up" if t_open > t1_close_val else "Gap-Down"
        close_band = "Strong" if t_cp >= const.EIGEN_CLOSE_UPPER_BAND else "Weak"
        delta_cp = round(t_cp - t1_cp, 4)
        vol_delta = round(((t_vol - t1_vol) / max(t1_vol, 1)) * 100, 1)

        label_map = {
            ("Gap-Up", "Strong"): ("Bullish Impulse Convergence", "Bullish"),
            ("Gap-Up", "Weak"): ("Contested Bullish Divergence", "Bullish"),
            ("Gap-Down", "Weak"): ("Bearish Impulse Convergence", "Bearish"),
            ("Gap-Down", "Strong"): ("Contested Bearish Divergence", "Bearish"),
        }
        label, sentiment = label_map.get((gap_dir, close_band), ("Unknown", "Neutral"))

        return {
            "symbol": symbol, "gap_dir": gap_dir, "close_band": close_band,
            "label": label, "sentiment": sentiment,
            "t_open": t_open, "t_close": t_close, "t_spread": t_spread,
            "t_cp": round(t_cp, 4), "t1_cp": round(t1_cp, 4),
            "delta_cp": delta_cp, "vol_delta_pct": vol_delta,
            "t_vol": t_vol, "t1_vol": t1_vol,
        }

    def get_weekly_eigen_details(self, symbol: str) -> Optional[Dict]:
        """Extracts Weekly EigenFilter classification details from a processed Excel file."""
        df = self._read_latest(const.WEEKLY_EIGEN_FILTER_DIR_NAME, symbol)
        if df is None:
            return None

        weekly = WeeklyEigenFilterService._consolidate_to_weekly(df)
        if weekly is None or len(weekly) < 2:
            return None

        latest, prev = weekly.iloc[-1], weekly.iloc[-2]
        t_open = float(latest["Open"])
        t_close = float(latest["Close"])
        t1_close_val = float(prev["Close"])
        t_cp = float(latest["Close_Position"])
        t1_cp = float(prev["Close_Position"])
        t_spread = float(latest["Spread"])
        t_vol = int(latest["Volume"])
        t1_vol = int(prev["Volume"])

        if t1_vol <= 0 or t_vol <= t1_vol:
            return None

        is_extreme = t_cp <= const.EIGEN_CLOSE_LOWER_BAND or t_cp >= const.EIGEN_CLOSE_UPPER_BAND
        if not is_extreme:
            return None

        gap_dir = None
        if t_open > t1_close_val and t_cp >= t1_cp:
            gap_dir = "Gap-Up"
        elif t_open < t1_close_val and t_cp <= t1_cp:
            gap_dir = "Gap-Down"
        if gap_dir is None:
            return None

        close_band = "Strong" if t_cp >= const.EIGEN_CLOSE_UPPER_BAND else "Weak"
        delta_cp = round(t_cp - t1_cp, 4)
        vol_delta = round(((t_vol - t1_vol) / max(t1_vol, 1)) * 100, 1)

        label_map = {
            ("Gap-Up", "Strong"): ("Bullish Impulse Convergence", "Bullish"),
            ("Gap-Up", "Weak"): ("Contested Bullish Divergence", "Bullish"),
            ("Gap-Down", "Weak"): ("Bearish Impulse Convergence", "Bearish"),
            ("Gap-Down", "Strong"): ("Contested Bearish Divergence", "Bearish"),
        }
        label, sentiment = label_map.get((gap_dir, close_band), ("Unknown", "Neutral"))

        latest_week = str(latest["YearWeek"])
        prev_week = str(prev["YearWeek"])

        return {
            "symbol": symbol, "gap_dir": gap_dir, "close_band": close_band,
            "label": label, "sentiment": sentiment,
            "t_open": t_open, "t_close": t_close, "t_spread": t_spread,
            "t_cp": round(t_cp, 4), "t1_cp": round(t1_cp, 4),
            "delta_cp": delta_cp, "vol_delta_pct": vol_delta,
            "t_vol": t_vol, "t1_vol": t1_vol,
            "latest_week": latest_week, "prev_week": prev_week,
        }

    def get_monthly_eigen_details(self, symbol: str) -> Optional[Dict]:
        """Extracts Monthly EigenFilter classification details from a processed Excel file."""
        df = self._read_latest(const.MONTHLY_EIGEN_FILTER_DIR_NAME, symbol)
        if df is None:
            return None

        monthly = MonthlyEigenFilterService._consolidate_to_monthly(df)
        if monthly is None or len(monthly) < 2:
            return None

        latest, prev = monthly.iloc[-1], monthly.iloc[-2]
        t_open = float(latest["Open"])
        t_close = float(latest["Close"])
        t1_close_val = float(prev["Close"])
        t_cp = float(latest["Close_Position"])
        t1_cp = float(prev["Close_Position"])
        t_spread = float(latest["Spread"])
        t_vol = int(latest["Volume"])
        t1_vol = int(prev["Volume"])

        if t1_vol <= 0 or t_vol <= t1_vol:
            return None

        is_extreme = t_cp <= const.EIGEN_CLOSE_LOWER_BAND or t_cp >= const.EIGEN_CLOSE_UPPER_BAND
        if not is_extreme:
            return None

        gap_dir = None
        if t_open > t1_close_val and t_cp >= t1_cp:
            gap_dir = "Gap-Up"
        elif t_open < t1_close_val and t_cp <= t1_cp:
            gap_dir = "Gap-Down"
        if gap_dir is None:
            return None

        close_band = "Strong" if t_cp >= const.EIGEN_CLOSE_UPPER_BAND else "Weak"
        delta_cp = round(t_cp - t1_cp, 4)
        vol_delta = round(((t_vol - t1_vol) / max(t1_vol, 1)) * 100, 1)

        label_map = {
            ("Gap-Up", "Strong"): ("Bullish Impulse Convergence", "Bullish"),
            ("Gap-Up", "Weak"): ("Contested Bullish Divergence", "Bullish"),
            ("Gap-Down", "Weak"): ("Bearish Impulse Convergence", "Bearish"),
            ("Gap-Down", "Strong"): ("Contested Bearish Divergence", "Bearish"),
        }
        label, sentiment = label_map.get((gap_dir, close_band), ("Unknown", "Neutral"))

        latest_month = str(latest["YearMonth"])
        prev_month = str(prev["YearMonth"])

        return {
            "symbol": symbol, "gap_dir": gap_dir, "close_band": close_band,
            "label": label, "sentiment": sentiment,
            "t_open": t_open, "t_close": t_close, "t_spread": t_spread,
            "t_cp": round(t_cp, 4), "t1_cp": round(t1_cp, 4),
            "delta_cp": delta_cp, "vol_delta_pct": vol_delta,
            "t_vol": t_vol, "t1_vol": t1_vol,
            "latest_month": latest_month, "prev_month": prev_month,
        }

    def get_volume_trap_details(self, symbol: str) -> Optional[Dict]:
        """Extracts daily Volume Trap classification details from a processed Excel file."""
        df = self._read_latest(const.VOLUME_TRAP_FILTER_DIR_NAME, symbol)
        if df is None or len(df) < 2:
            return None

        latest, prev = df.iloc[-1], df.iloc[-2]
        t_vol = int(latest.get("Volume", 0))
        t1_vol = int(prev.get("Volume", 0))
        t_spread = float(latest.get("Spread", 0))
        t1_spread = float(prev.get("Spread", 0))
        t_open = float(latest.get("Open", 0))
        t_close = float(latest.get("Close", 0))
        t_cp = float(latest.get("Close_Position", 0.5))

        if t1_vol <= 0 or t_spread <= 0:
            return None

        vol_pct = round(((t_vol - t1_vol) / max(t1_vol, 1)) * 100, 1)
        spread_pct = round(((t_spread - t1_spread) / max(t1_spread, 0.01)) * 100, 1)
        body_ratio = round(abs(t_close - t_open) / t_spread, 4) if t_spread > 0 else 0.0
        sentiment = "Bullish" if t_cp >= const.VOLUME_TRAP_SENTIMENT_MIDPOINT else "Bearish"

        return {
            "symbol": symbol, "sentiment": sentiment,
            "label": f"{sentiment} Volume Trap",
            "vol_delta_pct": vol_pct, "spread_delta_pct": spread_pct,
            "body_ratio": body_ratio, "t_cp": round(t_cp, 4),
            "t_open": t_open, "t_close": t_close,
            "t_vol": t_vol, "t1_vol": t1_vol,
            "t_spread": t_spread, "t1_spread": t1_spread,
        }

    def get_weekly_volume_trap_details(self, symbol: str) -> Optional[Dict]:
        """Extracts Weekly Volume Trap details by consolidating daily data into weekly candles."""
        df = self._read_latest(const.WEEKLY_VOLUME_TRAP_FILTER_DIR_NAME, symbol)
        if df is None:
            return None

        weekly = WeeklyVolumeTrapFilterService._consolidate_to_weekly(df)
        if weekly is None or len(weekly) < 2:
            return None

        latest, prev = weekly.iloc[-1], weekly.iloc[-2]
        t_vol = int(latest["Volume"])
        t1_vol = int(prev["Volume"])
        t_spread = float(latest["Spread"])
        t1_spread = float(prev["Spread"])
        t_open = float(latest["Open"])
        t_close = float(latest["Close"])
        t_cp = float(latest["Close_Position"])

        if t1_vol <= 0 or t_spread <= 0 or t_vol <= t1_vol or t_spread >= t1_spread:
            return None

        # Check body threshold on unrounded values to avoid premature rounding rejections
        body = abs(t_close - t_open)
        if body >= const.VOLUME_TRAP_BODY_RATIO_THRESHOLD * t_spread:
            return None

        body_ratio = round(body / t_spread, 4) if t_spread > 0 else 0.0
        vol_pct = round(((t_vol - t1_vol) / max(t1_vol, 1)) * 100, 1)
        spread_pct = round(((t_spread - t1_spread) / max(t1_spread, 0.01)) * 100, 1)
        sentiment = "Bullish" if t_cp >= const.VOLUME_TRAP_SENTIMENT_MIDPOINT else "Bearish"

        return {
            "symbol": symbol, "sentiment": sentiment,
            "label": f"{sentiment} Volume Trap",
            "vol_delta_pct": vol_pct, "spread_delta_pct": spread_pct,
            "body_ratio": body_ratio, "t_cp": round(t_cp, 4),
            "t_open": t_open, "t_close": t_close,
            "t_vol": t_vol, "t1_vol": t1_vol,
            "t_spread": t_spread, "t1_spread": t1_spread,
            "latest_week": str(latest["YearWeek"]),
            "prev_week": str(prev["YearWeek"]),
        }

    def get_monthly_volume_trap_details(self, symbol: str) -> Optional[Dict]:
        """Extracts Monthly Volume Trap details by consolidating daily data into monthly candles."""
        df = self._read_latest(const.MONTHLY_VOLUME_TRAP_FILTER_DIR_NAME, symbol)
        if df is None:
            return None

        monthly = MonthlyVolumeTrapFilterService._consolidate_to_monthly(df)
        if monthly is None or len(monthly) < 2:
            return None

        latest, prev = monthly.iloc[-1], monthly.iloc[-2]
        t_vol = int(latest["Volume"])
        t1_vol = int(prev["Volume"])
        t_spread = float(latest["Spread"])
        t1_spread = float(prev["Spread"])
        t_open = float(latest["Open"])
        t_close = float(latest["Close"])
        t_cp = float(latest["Close_Position"])

        if t1_vol <= 0 or t_spread <= 0 or t_vol <= t1_vol or t_spread >= t1_spread:
            return None

        # Check body threshold on unrounded values to avoid premature rounding rejections
        body = abs(t_close - t_open)
        if body >= const.VOLUME_TRAP_BODY_RATIO_THRESHOLD * t_spread:
            return None

        body_ratio = round(body / t_spread, 4) if t_spread > 0 else 0.0
        vol_pct = round(((t_vol - t1_vol) / max(t1_vol, 1)) * 100, 1)
        spread_pct = round(((t_spread - t1_spread) / max(t1_spread, 0.01)) * 100, 1)
        sentiment = "Bullish" if t_cp >= const.VOLUME_TRAP_SENTIMENT_MIDPOINT else "Bearish"

        return {
            "symbol": symbol, "sentiment": sentiment,
            "label": f"{sentiment} Volume Trap",
            "vol_delta_pct": vol_pct, "spread_delta_pct": spread_pct,
            "body_ratio": body_ratio, "t_cp": round(t_cp, 4),
            "t_open": t_open, "t_close": t_close,
            "t_vol": t_vol, "t1_vol": t1_vol,
            "t_spread": t_spread, "t1_spread": t1_spread,
            "latest_month": str(latest["YearMonth"]),
            "prev_month": str(prev["YearMonth"]),
        }

    def get_consensus_details(self) -> List[Dict]:
        """Extracts the sorted Consensus Ratings for the UI."""
        path = self.base_dir / const.CONSENSUS_RESULTS_DIR_NAME / "consensus_ratings.xlsx"
        if not path.exists(): return []
        
        try:
            df = pd.read_excel(path)
            df = df.fillna("None")
            df.columns = [c.lower() for c in df.columns]
            return df.to_dict('records')
        except (OSError, ValueError):
            return []
