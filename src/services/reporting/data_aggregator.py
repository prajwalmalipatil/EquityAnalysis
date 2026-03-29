"""
data_aggregator.py
Scans processed VSA folders and aggregates full detail for professional reports.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from src.constants import vsa_constants as const
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
            "vsa": self._count_files(const.RESULTS_DIR_NAME),
            "trending": self._count_files(const.TRENDING_DIR_NAME),
            "anomaly": self._count_files(const.ANOMALY_DIR_NAME),
            "ticker": self._count_files(const.TICKER_DIR_NAME),
            "triggers": self._count_files(const.TRIGGERS_DIR_NAME)
        }

    def get_symbol_lists(self) -> Dict[str, List[str]]:
        """Returns simple lists of symbols for categorization."""
        return {
            "trending": self._get_symbols(const.TRENDING_DIR_NAME),
            "anomaly": self._get_symbols(const.ANOMALY_DIR_NAME),
            "ticker": self._get_symbols(const.TICKER_DIR_NAME),
            "triggers": self._get_symbols(const.TRIGGERS_DIR_NAME)
        }

    def get_ticker_details(self, symbol: str) -> Optional[Dict]:
        """Deep extraction for Action Required ticker cards."""
        df = self._read_latest(const.TICKER_DIR_NAME, symbol)
        if df is None: return None
        
        latest = df.iloc[-1]
        return {
            "symbol": symbol,
            "pattern": str(latest.get("Signal_Type", "No Signal")),
            "description": "Asset is showing structural strength with institutional demand absorption.",
            "spread_ratio": float(latest.get("Spread", 0) / latest.get("Spread_MA", 1)),
            "confidence": 0.70  # Default or heuristic-based
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
        """Extraction for Anomaly V2 patterns."""
        df = self._read_latest(const.ANOMALY_DIR_NAME, symbol)
        if df is None: return None
        
        latest = df.iloc[-1]
        prev_vol = int(latest.get("Prev_Volume", 0))
        curr_vol = int(latest.get("Volume", 0))
        drop_pct = ((curr_vol - prev_vol) / prev_vol * 100) if prev_vol > 0 else 0
        
        return {
            "symbol": symbol,
            "pattern": str(latest.get("Anomaly_V2", "Neutral")),
            "prev_vol": prev_vol,
            "curr_vol": curr_vol,
            "drop_pct": drop_pct,
            "sentiment": "Bearish" if "Dump" in str(latest.get("Anomaly_V2")) or "Fall" in str(latest.get("Anomaly_V2")) else "Neutral"
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
        except Exception as e:
            logger.error("READ_EXCEL_FAILED", extra={"symbol": symbol, "error": str(e)})
            return None
