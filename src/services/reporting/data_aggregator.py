"""
data_aggregator.py
Scans processed VSA folders (Results, Trending, Anomaly, Ticker) and aggregates
data for the final automated report.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from ...constants import vsa_constants as const
from ...utils.observability import get_tenant_logger

logger = get_tenant_logger("data-aggregator")

class DataAggregator:
    """
    Service to collect and prepare data for reporting.
    Decouples folder scanning from HTML rendering.
    """
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def aggregate_pipeline_stats(self) -> Dict:
        """Collects high-level counts for the report summary."""
        stats = {
            "vsa": self._count_files(const.RESULTS_DIR_NAME),
            "trending": self._count_files(const.TRENDING_DIR_NAME),
            "anomaly": self._count_files(const.ANOMALY_DIR_NAME),
            "ticker": self._count_files(const.TICKER_DIR_NAME),
            "triggers": self._count_files(const.TRIGGERS_DIR_NAME)
        }
        return stats

    def get_symbol_lists(self) -> Dict[str, List[str]]:
        """Returns lists of symbols for each processed category."""
        return {
            "trending": self._get_symbols(const.TRENDING_DIR_NAME),
            "anomaly": self._get_symbols(const.ANOMALY_DIR_NAME),
            "ticker": self._get_symbols(const.TICKER_DIR_NAME),
            "triggers": self._get_symbols(const.TRIGGERS_DIR_NAME)
        }

    def _count_files(self, folder_name: str) -> int:
        folder = self.base_dir / folder_name
        if not folder.exists():
            return 0
        return len(list(folder.glob("*.xlsx")))

    def _get_symbols(self, folder_name: str) -> List[str]:
        folder = self.base_dir / folder_name
        if not folder.exists():
            return []
        return [f.stem.replace("_VSA", "") for f in folder.glob("*.xlsx")]

    def get_anomaly_details(self, symbol: str) -> Optional[Dict]:
        """Reads the latest anomaly data for a symbol."""
        # This mirrors get_anomaly_details from the original notifier
        path = self.base_dir / const.ANOMALY_DIR_NAME / f"{symbol}_VSA.xlsx"
        if not path.exists():
            return None
            
        try:
            df = pd.read_excel(path, sheet_name="VSA_Analysis")
            if df.empty:
                return None
            
            # Using tail() as per original logic for latest data
            latest = df.iloc[-1]
            return {
                "symbol": symbol,
                "drop_pct": float(latest.get("Drop_Pct", 0)),
                "pattern": str(latest.get("Signal_Type", "Neutral")),
                "sentiment": str(latest.get("Sentiment", "Neutral"))
            }
        except Exception as e:
            logger.error("AGGREGATOR_ANOMALY_ERR", extra={"symbol": symbol, "error": str(e)})
            return None
