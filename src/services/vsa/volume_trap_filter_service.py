"""
volume_trap_filter_service.py
Post-VSA volume trap anomaly classifier.
Scans Results/ folder and filters stocks matching the volume trap criteria.
"""

import shutil
import pandas as pd
from pathlib import Path
from typing import List, Optional

from src.constants import vsa_constants as const
from src.models.vsa_models import VolumeTrapClassification
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("volume-trap-filter-service")

MIN_ROWS_REQUIRED = 2


class VolumeTrapFilterService:
    """
    Scans processed VSA Excel files in Results/ and classifies stocks
    that meet the volume trap criteria.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results_dir = base_dir / const.RESULTS_DIR_NAME
        self.volume_trap_dir = base_dir / const.VOLUME_TRAP_FILTER_DIR_NAME
        self.volume_trap_dir.mkdir(parents=True, exist_ok=True)

    def scan_and_classify(self) -> List[VolumeTrapClassification]:
        """Reads each Excel in Results/, evaluates conditions, copies qualifying files."""
        if not self.results_dir.exists():
            logger.warning("VOLUME_TRAP_RESULTS_DIR_MISSING", extra={"path": str(self.results_dir)})
            return []

        results: List[VolumeTrapClassification] = []
        for xlsx_path in sorted(self.results_dir.glob("*.xlsx")):
            classification = self._process_single_file(xlsx_path)
            if classification is None:
                continue
            shutil.copy(xlsx_path, self.volume_trap_dir)
            results.append(classification)

        self._log_summary(results)
        return results

    def _process_single_file(self, path: Path) -> Optional[VolumeTrapClassification]:
        """Reads an Excel file, extracts T and T-1 rows, and evaluates conditions."""
        try:
            df = pd.read_excel(path, sheet_name="VSA_Analysis")
        except (OSError, ValueError):
            return None

        if len(df) < MIN_ROWS_REQUIRED:
            return None

        symbol = path.stem.replace("_VSA", "")
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        return self._evaluate_stock(symbol, latest, prev)

    def _evaluate_stock(
        self, symbol: str, latest: pd.Series, prev: pd.Series
    ) -> Optional[VolumeTrapClassification]:
        """Pure evaluation: checks volume, spread, and body relationship."""
        t_vol = float(latest.get("Volume", 0))
        t1_vol = float(prev.get("Volume", 0))
        t_spread = float(latest.get("Spread", 0))
        t1_spread = float(prev.get("Spread", 0))
        t_close = float(latest.get("Close", 0))
        t_open = float(latest.get("Open", 0))
        t_close_position = float(latest.get("Close_Position", 0.5))

        if t1_vol <= 0 or t1_spread <= 0 or t_spread <= 0:
            return None

        if not (
            t_vol > t1_vol and
            t_spread < t1_spread and
            abs(t_close - t_open) < const.VOLUME_TRAP_BODY_RATIO_THRESHOLD * t_spread
        ):
            return None

        sentiment = "Bullish" if t_close_position >= const.VOLUME_TRAP_SENTIMENT_MIDPOINT else "Bearish"
        label = f"{sentiment} Volume Trap"
        body_ratio = abs(t_close - t_open) / t_spread
        volume_pct = ((t_vol - t1_vol) / t1_vol) * 100
        spread_pct = ((t_spread - t1_spread) / t1_spread) * 100

        return VolumeTrapClassification(
            symbol=symbol,
            label=label,
            sentiment=sentiment,
            t_volume=int(t_vol),
            t1_volume=int(t1_vol),
            volume_pct=volume_pct,
            t_spread=t_spread,
            t1_spread=t1_spread,
            spread_pct=spread_pct,
            t_close=t_close,
            t_open=t_open,
            t_close_position=t_close_position,
            body_ratio=body_ratio
        )

    @staticmethod
    def _log_summary(results: List[VolumeTrapClassification]) -> None:
        """Logs summary counts by sentiment."""
        bullish = [r for r in results if r.sentiment == "Bullish"]
        bearish = [r for r in results if r.sentiment == "Bearish"]
        logger.info(
            f"VOLUME_TRAP_FILTER_COMPLETE: {len(results)} stocks qualified "
            f"(Bullish: {len(bullish)}, Bearish: {len(bearish)})"
        )
