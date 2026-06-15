"""
weekly_eigen_filter_service.py
Consolidates daily OHLCV data into weekly candles and applies
EigenFilter (volume-amplitude OHLC divergence) analysis on the
weekly timeframe. Runs independently of the Daily EigenFilter.
"""

import shutil
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

from src.constants import vsa_constants as const
from src.models.vsa_models import EigenClassification
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("weekly-eigen-filter-service")

MIN_WEEKS_REQUIRED = 2

_LABEL_MATRIX = {
    ("Gap-Up", "Strong"): ("Bullish Impulse Convergence", "Bullish"),
    ("Gap-Up", "Weak"):   ("Contested Bullish Divergence", "Bullish"),
    ("Gap-Down", "Weak"): ("Bearish Impulse Convergence", "Bearish"),
    ("Gap-Down", "Strong"): ("Contested Bearish Divergence", "Bearish"),
}


class WeeklyEigenFilterService:
    """
    Consolidates daily data into weekly OHLCV candles, then applies
    the EigenFilter classification logic on the weekly timeframe.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results_dir = base_dir / const.RESULTS_DIR_NAME
        self.weekly_eigen_dir = base_dir / const.WEEKLY_EIGEN_FILTER_DIR_NAME
        self.weekly_eigen_dir.mkdir(parents=True, exist_ok=True)

    def consolidate_and_classify(self) -> List[EigenClassification]:
        """Main entry: reads each Results/ Excel, consolidates to weekly, classifies."""
        if not self.results_dir.exists():
            logger.warning("WEEKLY_EIGEN_RESULTS_DIR_MISSING",
                           extra={"path": str(self.results_dir)})
            return []

        results: List[EigenClassification] = []
        for xlsx_path in sorted(self.results_dir.glob("*.xlsx")):
            classification = self._process_single_file(xlsx_path)
            if classification is None:
                continue
            shutil.copy(xlsx_path, self.weekly_eigen_dir)
            results.append(classification)

        self._log_summary(results)
        return results

    def _process_single_file(self, path: Path) -> Optional[EigenClassification]:
        """Reads daily data, consolidates to weekly, evaluates EigenFilter conditions."""
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

        Weekly candle rules:
          Open   = first daily open of the week
          High   = max daily high of the week
          Low    = min daily low of the week
          Close  = last daily close of the week
          Volume = sum of daily volumes for the week
          Spread = High - Low (of the weekly candle)
          Close_Position = (Close - Low) / Spread
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
    ) -> Optional[EigenClassification]:
        """Applies identical EigenFilter conditions on weekly candle data."""
        t_vol = float(latest.get("Volume", 0))
        t1_vol = float(prev.get("Volume", 0))

        # Condition 1: Volume surge (current week > previous week)
        if t_vol <= t1_vol or t1_vol <= 0:
            return None

        t_open = float(latest.get("Open", 0))
        t_close = float(latest.get("Close", 0))
        t1_close = float(prev.get("Close", 0))
        t_cp = float(latest.get("Close_Position", 0.5))
        t1_cp = float(prev.get("Close_Position", 0.5))
        t_spread = float(latest.get("Spread", 0))

        # Condition 2: Extreme close position
        is_extreme_close = (
            t_cp <= const.EIGEN_CLOSE_LOWER_BAND
            or t_cp >= const.EIGEN_CLOSE_UPPER_BAND
        )
        if not is_extreme_close:
            return None

        # Condition 3: Gap direction + CP drift alignment
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
        """Returns gap direction if conditions are met, else None.

        For weekly candles:
          Gap-Up:   current week's open > previous week's close AND CP improved
          Gap-Down: current week's open < previous week's close AND CP deteriorated
        """
        if t_open > t1_close and t_cp >= t1_cp:
            return "Gap-Up"
        if t_open < t1_close and t_cp <= t1_cp:
            return "Gap-Down"
        return None

    @staticmethod
    def _log_summary(results: List[EigenClassification]) -> None:
        """Logs summary counts by sentiment."""
        bullish = [r for r in results if r.sentiment == "Bullish"]
        bearish = [r for r in results if r.sentiment == "Bearish"]
        logger.info(
            f"WEEKLY_EIGEN_FILTER_COMPLETE: {len(results)} stocks qualified "
            f"(Bullish: {len(bullish)}, Bearish: {len(bearish)})"
        )
