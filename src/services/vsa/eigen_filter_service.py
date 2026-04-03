"""
eigen_filter_service.py
Post-VSA volume-amplitude OHLC divergence classifier.
Scans Results/ folder and filters stocks whose volume, gap direction,
close-position band, and close-position drift all confirm a directional eigenstate.
"""

import shutil
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple

from src.constants import vsa_constants as const
from src.models.vsa_models import EigenClassification
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("eigen-filter-service")

# Label matrix: (gap_direction, close_band) -> (label, sentiment)
_LABEL_MATRIX = {
    ("Gap-Up", "Strong"): ("Bullish Impulse Convergence", "Bullish"),
    ("Gap-Up", "Weak"):   ("Contested Bullish Divergence", "Bullish"),
    ("Gap-Down", "Weak"): ("Bearish Impulse Convergence", "Bearish"),
    ("Gap-Down", "Strong"): ("Contested Bearish Divergence", "Bearish"),
}

MIN_ROWS_REQUIRED = 2


class EigenFilterService:
    """
    Scans processed VSA Excel files in Results/ and classifies stocks
    that meet the volume-surge + gap-direction + close-position-drift criteria.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results_dir = base_dir / const.RESULTS_DIR_NAME
        self.eigen_dir = base_dir / const.EIGEN_FILTER_DIR_NAME
        self.eigen_dir.mkdir(parents=True, exist_ok=True)

    def scan_and_classify(self) -> List[EigenClassification]:
        """Reads each Excel in Results/, evaluates conditions, copies qualifying files."""
        if not self.results_dir.exists():
            logger.warning("EIGEN_RESULTS_DIR_MISSING", extra={"path": str(self.results_dir)})
            return []

        results: List[EigenClassification] = []
        for xlsx_path in sorted(self.results_dir.glob("*.xlsx")):
            classification = self._process_single_file(xlsx_path)
            if classification is None:
                continue
            shutil.copy(xlsx_path, self.eigen_dir)
            results.append(classification)

        self._log_summary(results)
        return results

    def _process_single_file(self, path: Path) -> Optional[EigenClassification]:
        """Reads an Excel file, extracts T and T-1 rows, and evaluates conditions."""
        try:
            df = pd.read_excel(path, sheet_name="VSA_Analysis")
        except Exception:
            return None

        if len(df) < MIN_ROWS_REQUIRED:
            return None

        symbol = path.stem.replace("_VSA", "")
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        return self._evaluate_stock(symbol, latest, prev)

    def _evaluate_stock(
        self, symbol: str, latest: pd.Series, prev: pd.Series
    ) -> Optional[EigenClassification]:
        """Pure evaluation: checks volume surge, gap, close band, and CP drift."""
        t_vol = float(latest.get("Volume", 0))
        t1_vol = float(prev.get("Volume", 0))

        # Condition 1: Volume surge
        if t_vol <= t1_vol or t1_vol <= 0:
            return None

        t_open = float(latest.get("Open", 0))
        t_close = float(latest.get("Close", 0))
        t1_close = float(prev.get("Close", 0))
        t_cp = float(latest.get("Close_Position", 0.5))
        t1_cp = float(prev.get("Close_Position", 0.5))
        t_spread = float(latest.get("Spread", 0))

        is_extreme_close = (
            t_cp <= const.EIGEN_CLOSE_LOWER_BAND or t_cp >= const.EIGEN_CLOSE_UPPER_BAND
        )
        if not is_extreme_close:
            return None

        gap_direction = self._detect_gap(t_open, t1_close, t_cp, t1_cp)
        if gap_direction is None:
            return None

        close_band = "Strong" if t_cp >= const.EIGEN_CLOSE_UPPER_BAND else "Weak"
        label, sentiment = _LABEL_MATRIX[(gap_direction, close_band)]
        volume_surge_pct = ((t_vol - t1_vol) / t1_vol) * 100
        delta_cp = t_cp - t1_cp

        return EigenClassification(
            symbol=symbol,
            gap_direction=gap_direction,
            close_band=close_band,
            label=label,
            sentiment=sentiment,
            volume_surge_pct=volume_surge_pct,
            t_close_position=t_cp,
            t1_close_position=t1_cp,
            delta_cp=delta_cp,
            t_open=t_open,
            t_close=t_close,
            t_spread=t_spread,
            t_volume=int(t_vol),
            t1_volume=int(t1_vol),
            t1_close=t1_close,
        )

    @staticmethod
    def _detect_gap(
        t_open: float, t1_close: float, t_cp: float, t1_cp: float
    ) -> Optional[str]:
        """Returns gap direction if conditions 2 or 3 are met, else None."""
        # Condition 2: Gap-Up with improving CP
        if t_open > t1_close and t_cp >= t1_cp:
            return "Gap-Up"

        # Condition 3: Gap-Down with deteriorating CP
        if t_open < t1_close and t_cp <= t1_cp:
            return "Gap-Down"

        return None

    @staticmethod
    def _log_summary(results: List[EigenClassification]) -> None:
        """Logs summary counts by label."""
        bullish = [r for r in results if r.sentiment == "Bullish"]
        bearish = [r for r in results if r.sentiment == "Bearish"]
        logger.info(
            f"EIGEN_FILTER_COMPLETE: {len(results)} stocks qualified "
            f"(Bullish: {len(bullish)}, Bearish: {len(bearish)})"
        )
