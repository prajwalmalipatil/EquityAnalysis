"""
age_again_filter_service.py
Post-VSA volume-spread structural anomaly classifier.
Scans Results/ folder and filters stocks matching either:
  Scenario A: T_vol > T1_vol AND T_spread < T1_spread  (Absorption - Bullish)
  Scenario B: T_vol < T1_vol AND T_spread > T1_spread  (Effort Without Result - Bearish)
"""

import shutil
import pandas as pd
from pathlib import Path
from typing import List, Optional

from src.constants import vsa_constants as const
from src.models.vsa_models import AgeAgainClassification
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("age-again-filter-service")

MIN_ROWS_REQUIRED = 2

_SCENARIO_LABELS = {
    "Vol_Surge_Spread_Contraction": ("Absorption Signal", "Bullish"),
    "Vol_Drop_Spread_Expansion":    ("Effort Without Result", "Bearish"),
}


class AgeAgainFilterService:
    """
    Scans processed VSA Excel files in Results/ and classifies stocks
    that meet the volume-spread structural anomaly criteria.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results_dir = base_dir / const.RESULTS_DIR_NAME
        self.age_again_dir = base_dir / const.AGE_AGAIN_FILTER_DIR_NAME
        self.age_again_dir.mkdir(parents=True, exist_ok=True)

    def scan_and_classify(self) -> List[AgeAgainClassification]:
        """Reads each Excel in Results/, evaluates conditions, copies qualifying files."""
        if not self.results_dir.exists():
            logger.warning("AGE_AGAIN_RESULTS_DIR_MISSING", extra={"path": str(self.results_dir)})
            return []

        results: List[AgeAgainClassification] = []
        for xlsx_path in sorted(self.results_dir.glob("*.xlsx")):
            classification = self._process_single_file(xlsx_path)
            if classification is None:
                continue
            shutil.copy(xlsx_path, self.age_again_dir)
            results.append(classification)

        self._log_summary(results)
        return results

    def _process_single_file(self, path: Path) -> Optional[AgeAgainClassification]:
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
    ) -> Optional[AgeAgainClassification]:
        """Pure evaluation: checks volume and spread relationship for both scenarios."""
        t_vol = float(latest.get("Volume", 0))
        t1_vol = float(prev.get("Volume", 0))
        t_spread = float(latest.get("Spread", 0))
        t1_spread = float(prev.get("Spread", 0))

        if t1_vol <= 0 or t1_spread <= 0:
            return None

        scenario = self._detect_scenario(t_vol, t1_vol, t_spread, t1_spread)
        if scenario is None:
            return None

        label, sentiment = _SCENARIO_LABELS[scenario]
        volume_pct = ((t_vol - t1_vol) / t1_vol) * 100
        spread_pct = ((t_spread - t1_spread) / t1_spread) * 100

        return AgeAgainClassification(
            symbol=symbol,
            scenario=scenario,
            label=label,
            sentiment=sentiment,
            t_volume=int(t_vol),
            t1_volume=int(t1_vol),
            volume_pct=volume_pct,
            t_spread=t_spread,
            t1_spread=t1_spread,
            spread_pct=spread_pct,
            t_close=float(latest.get("Close", 0)),
            t_open=float(latest.get("Open", 0)),
            t_close_position=float(latest.get("Close_Position", 0.5)),
        )

    @staticmethod
    def _detect_scenario(
        t_vol: float, t1_vol: float, t_spread: float, t1_spread: float
    ) -> Optional[str]:
        """Returns scenario key if either condition is met, else None."""
        # Scenario A: Volume surge + spread contraction (Absorption)
        if t_vol > t1_vol and t_spread < t1_spread:
            return "Vol_Surge_Spread_Contraction"
        # Scenario B: Volume drop + spread expansion (Effort Without Result)
        if t_vol < t1_vol and t_spread > t1_spread:
            return "Vol_Drop_Spread_Expansion"
        return None

    @staticmethod
    def _log_summary(results: List[AgeAgainClassification]) -> None:
        """Logs summary counts by scenario."""
        bullish = [r for r in results if r.sentiment == "Bullish"]
        bearish = [r for r in results if r.sentiment == "Bearish"]
        logger.info(
            f"AGE_AGAIN_FILTER_COMPLETE: {len(results)} stocks qualified "
            f"(Absorption: {len(bullish)}, Effort-Without-Result: {len(bearish)})"
        )
