"""
weekly_age_again_filter_service.py
Consolidates daily OHLCV data into weekly candles and applies
AgeAgain Filter (volume-spread structural anomaly) classification logic on the weekly timeframe.
"""

import shutil
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

from src.constants import vsa_constants as const
from src.models.vsa_models import AgeAgainClassification
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("weekly-age-again-filter-service")

MIN_WEEKS_REQUIRED = 2

_SCENARIO_LABELS = {
    "Vol_Surge_Spread_Contraction": ("Absorption Signal", "Bullish"),
    "Vol_Drop_Spread_Expansion":    ("Effort Without Result", "Bearish"),
}


class WeeklyAgeAgainFilterService:
    """
    Consolidates daily data into weekly OHLCV candles, then applies
    the AgeAgain classification logic on the weekly timeframe.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results_dir = base_dir / const.RESULTS_DIR_NAME
        self.weekly_age_again_dir = base_dir / const.WEEKLY_AGE_AGAIN_FILTER_DIR_NAME
        self.weekly_age_again_dir.mkdir(parents=True, exist_ok=True)

    def consolidate_and_classify(self) -> List[AgeAgainClassification]:
        """Main entry: reads each Results/ Excel, consolidates to weekly, classifies."""
        if not self.results_dir.exists():
            logger.warning("WEEKLY_AGE_AGAIN_RESULTS_DIR_MISSING",
                           extra={"path": str(self.results_dir)})
            return []

        results: List[AgeAgainClassification] = []
        for xlsx_path in sorted(self.results_dir.glob("*.xlsx")):
            classification = self._process_single_file(xlsx_path)
            if classification is None:
                continue
            shutil.copy(xlsx_path, self.weekly_age_again_dir)
            results.append(classification)

        self._log_summary(results)
        return results

    def _process_single_file(self, path: Path) -> Optional[AgeAgainClassification]:
        """Reads daily data, consolidates to weekly, evaluates AgeAgain conditions."""
        try:
            df = pd.read_excel(path, sheet_name="VSA_Analysis")
        except (OSError, ValueError):
            return None

        weekly_df = self._consolidate_to_weekly(df)
        if weekly_df is None or len(weekly_df) < MIN_WEEKS_REQUIRED:
            return None

        symbol = path.stem.replace("_VSA", "")
        latest = weekly_df.iloc[-1]
        prev = weekly_df.iloc[-2]
        return self._evaluate_weekly(symbol, latest, prev)

    @staticmethod
    def _consolidate_to_weekly(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Aggregates daily OHLCV rows into completed weekly candles.

        The current in-progress week is always excluded to prevent
        partial-period bias. Only fully completed weeks are
        included in the consolidation.
        """
        if "Date" not in df.columns:
            return None

        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        if df.empty:
            return None

        df = df.sort_values("Date").reset_index(drop=True)
        # Group by ISO Year and Week
        df["YearWeek"] = df["Date"].dt.strftime("%G-W%V")

        # Exclude the current in-progress week — only completed weeks are valid
        now_ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
        current_period = now_ist.strftime("%G-W%V")
        df = df[df["YearWeek"] < current_period]
        if df.empty:
            return None

        weekly = df.groupby("YearWeek").agg(
            Open=("Open", "first"),
            High=("High", "max"),
            Low=("Low", "min"),
            Close=("Close", "last"),
            Volume=("Volume", "sum"),
        ).reset_index()

        weekly["Spread"] = weekly["High"] - weekly["Low"]
        weekly["Close_Position"] = weekly.apply(
            lambda r: (r["Close"] - r["Low"]) / r["Spread"]
            if r["Spread"] > 0 else 0.5,
            axis=1,
        )
        weekly = weekly.sort_values("YearWeek").reset_index(drop=True)
        return weekly

    def _evaluate_weekly(
        self, symbol: str, latest: pd.Series, prev: pd.Series
    ) -> Optional[AgeAgainClassification]:
        """Applies AgeAgain conditions on weekly candle data."""
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
        if t_vol > t1_vol and t_spread < t1_spread:
            return "Vol_Surge_Spread_Contraction"
        if t_vol < t1_vol and t_spread > t1_spread:
            return "Vol_Drop_Spread_Expansion"
        return None

    @staticmethod
    def _log_summary(results: List[AgeAgainClassification]) -> None:
        """Logs summary counts by scenario."""
        bullish = [r for r in results if r.sentiment == "Bullish"]
        bearish = [r for r in results if r.sentiment == "Bearish"]
        logger.info(
            f"WEEKLY_AGE_AGAIN_FILTER_COMPLETE: {len(results)} stocks qualified "
            f"(Absorption: {len(bullish)}, Effort-Without-Result: {len(bearish)})"
        )
