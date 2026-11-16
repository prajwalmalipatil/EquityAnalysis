"""
gap_momentum_filter_service.py
Post-VSA gap-continuation classifier.
Scans Results/ and keeps stocks where day T expands on BOTH volume and spread
versus T-1 (the previous trading day) and the candle confirms the opening gap:
  Scenario A: T_open > T1_close AND T_close > T_open  (Gap-Up Continuation   - Bullish)
  Scenario B: T_open < T1_close AND T_close < T_open  (Gap-Down Continuation - Bearish)
"""

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.constants import vsa_constants as const
from src.models.vsa_models import GapMomentumClassification
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("gap-momentum-filter-service")

MIN_ROWS_REQUIRED = 2
PERCENT = 100
SENTIMENT_BULLISH = "Bullish"
SENTIMENT_BEARISH = "Bearish"

_SCENARIO_LABELS = {
    "Gap_Up_Green_Close": ("Gap-Up Continuation", SENTIMENT_BULLISH),
    "Gap_Down_Red_Close": ("Gap-Down Continuation", SENTIMENT_BEARISH),
}


class GapMomentumFilterService:
    """
    Scans processed VSA Excel files in Results/ and classifies stocks whose
    latest session shows volume + spread expansion with a confirmed opening gap.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results_dir = base_dir / const.RESULTS_DIR_NAME
        self.gap_momentum_dir = base_dir / const.GAP_MOMENTUM_FILTER_DIR_NAME
        self.gap_momentum_dir.mkdir(parents=True, exist_ok=True)

    def scan_and_classify(self) -> List[GapMomentumClassification]:
        """Reads each Excel in Results/, evaluates conditions, copies qualifying files."""
        if not self.results_dir.exists():
            logger.warning("GAP_MOMENTUM_RESULTS_DIR_MISSING", extra={"path": str(self.results_dir)})
            return []

        results: List[GapMomentumClassification] = []
        for xlsx_path in sorted(self.results_dir.glob("*.xlsx")):
            classification = self._process_single_file(xlsx_path)
            if classification is None:
                continue
            shutil.copy(xlsx_path, self.gap_momentum_dir)
            results.append(classification)

        self._log_summary(results)
        return results

    def _process_single_file(self, path: Path) -> Optional[GapMomentumClassification]:
        """Reads an Excel file and evaluates its T / T-1 rows."""
        try:
            df = pd.read_excel(path, sheet_name="VSA_Analysis")
        except (OSError, ValueError):
            return None

        symbol = path.stem.replace("_VSA", "")
        return self.classify_frame(symbol, df)

    @classmethod
    def classify_frame(cls, symbol: str, df: pd.DataFrame) -> Optional[GapMomentumClassification]:
        """Classifies the last two rows (T and T-1) of a chronologically ordered frame."""
        if len(df) < MIN_ROWS_REQUIRED:
            return None
        return cls.evaluate_stock(symbol, df.iloc[-1], df.iloc[-2])

    @classmethod
    def evaluate_stock(
        cls, symbol: str, latest: pd.Series, prev: pd.Series
    ) -> Optional[GapMomentumClassification]:
        """Pure evaluation of the volume, spread, gap and candle-colour conditions."""
        t_vol, t1_vol = _to_float(latest.get("Volume")), _to_float(prev.get("Volume"))
        t_spread, t1_spread = _to_float(latest.get("Spread")), _to_float(prev.get("Spread"))
        t_open, t_close = _to_float(latest.get("Open")), _to_float(latest.get("Close"))
        t1_close = _to_float(prev.get("Close"))

        if min(t1_vol, t1_spread, t1_close, t_open) <= 0:
            return None
        if not (t_vol > t1_vol and t_spread > t1_spread):
            return None

        scenario = cls._detect_scenario(t_open, t_close, t1_close)
        if scenario is None:
            return None

        label, sentiment = _SCENARIO_LABELS[scenario]
        return GapMomentumClassification(
            symbol=symbol,
            scenario=scenario,
            label=label,
            sentiment=sentiment,
            signal_date=_format_date(latest.get("Date")),
            t_open=t_open,
            t_close=t_close,
            t1_close=t1_close,
            gap_pct=_pct_change(t_open, t1_close),
            change_pct=_pct_change(t_close, t1_close),
            t_volume=int(t_vol),
            t1_volume=int(t1_vol),
            volume_pct=_pct_change(t_vol, t1_vol),
            t_spread=t_spread,
            t1_spread=t1_spread,
            spread_pct=_pct_change(t_spread, t1_spread),
        )

    @staticmethod
    def _detect_scenario(t_open: float, t_close: float, t1_close: float) -> Optional[str]:
        """Returns the scenario key when the candle colour confirms the gap, else None."""
        if t_open > t1_close and t_close > t_open:
            return "Gap_Up_Green_Close"
        if t_open < t1_close and t_close < t_open:
            return "Gap_Down_Red_Close"
        return None

    @staticmethod
    def to_details(classification: GapMomentumClassification) -> Dict:
        """Serialises a classification into the rounded dict published to data.json."""
        details = asdict(classification)
        for key in ("gap_pct", "change_pct", "volume_pct", "spread_pct"):
            details[key] = round(details[key], 2)
        for key in ("t_open", "t_close", "t1_close", "t_spread", "t1_spread"):
            details[key] = round(details[key], 2)
        return details

    @staticmethod
    def _log_summary(results: List[GapMomentumClassification]) -> None:
        """Logs summary counts by scenario."""
        bullish = sum(1 for r in results if r.sentiment == SENTIMENT_BULLISH)
        logger.info(
            f"GAP_MOMENTUM_FILTER_COMPLETE: {len(results)} stocks qualified "
            f"(Gap-Up Continuation: {bullish}, Gap-Down Continuation: {len(results) - bullish})"
        )


def _to_float(value) -> float:
    """Coerces a cell to float, treating missing / non-numeric values as 0."""
    number = pd.to_numeric(value, errors="coerce")
    return 0.0 if pd.isna(number) else float(number)


def _pct_change(current: float, base: float) -> float:
    """Percentage change of current versus a strictly positive base."""
    return ((current - base) / base) * PERCENT


def _format_date(value) -> str:
    """Formats the T-day date as YYYY-MM-DD, or '' when unavailable."""
    parsed = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")
