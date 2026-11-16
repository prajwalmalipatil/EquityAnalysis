"""
gap_up_rejection_filter_service.py
Post-VSA follow-through tracker for failed gap-up continuation.

A Gap-Up Rejection is three consecutive sessions:
  T-1 -> T : T is a Gap-Up Continuation (see GapMomentumFilterService)
  T+1      : opens above the T close, but closes below its own open (red)

The setup is reported on the trigger day (T+1) and then tracked for the next
TRACK_SESSIONS sessions, measured from the T+1 close.
"""

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.constants import vsa_constants as const
from src.models.vsa_models import GapUpRejectionClassification
from src.services.vsa.gap_momentum_filter_service import (
    SENTIMENT_BEARISH,
    GapMomentumFilterService,
    _format_date,
    _pct_change,
    _to_float,
)
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("gap-up-rejection-filter-service")

TRACK_SESSIONS = 10          # sessions followed after the trigger day
MIN_ROWS_REQUIRED = 3        # T-1, T and T+1
GAP_UP_SCENARIO = "Gap_Up_Green_Close"
LABEL = "Gap-Up Rejection"
STATUS_NEW, STATUS_TRACKING = "New", "Tracking"
_ROUNDED = (
    "signal_close", "trigger_open", "trigger_close", "gap_pct", "trigger_change_pct",
    "last_close", "return_pct", "max_gain_pct", "max_drawdown_pct",
)


class GapUpRejectionFilterService:
    """Finds each stock's most recent Gap-Up Rejection that is still inside its tracking window."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results_dir = base_dir / const.RESULTS_DIR_NAME
        self.output_dir = base_dir / const.GAP_UP_REJECTION_FILTER_DIR_NAME
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def scan_and_classify(self) -> List[GapUpRejectionClassification]:
        """Reads each Excel in Results/, evaluates the setup, copies qualifying files."""
        if not self.results_dir.exists():
            logger.warning("GAP_UP_REJECTION_RESULTS_DIR_MISSING", extra={"path": str(self.results_dir)})
            return []

        frames = {}
        for xlsx_path in sorted(self.results_dir.glob("*.xlsx")):
            try:
                frames[xlsx_path] = pd.read_excel(xlsx_path, sheet_name="VSA_Analysis")
            except (OSError, ValueError):
                continue

        # A stock whose history stops before the market's latest session is stale: its
        # "sessions since trigger" would be counted against old data, so it is skipped.
        last_dates = {path: _last_date(df) for path, df in frames.items()}
        market_date = max((d for d in last_dates.values() if d is not None), default=None)

        results: List[GapUpRejectionClassification] = []
        for xlsx_path, df in frames.items():
            if last_dates[xlsx_path] != market_date:
                continue
            classification = self.classify_frame(xlsx_path.stem.replace("_VSA", ""), df)
            if classification is None:
                continue
            shutil.copy(xlsx_path, self.output_dir)
            results.append(classification)

        new = sum(1 for r in results if r.status == STATUS_NEW)
        logger.info(
            f"GAP_UP_REJECTION_FILTER_COMPLETE: {len(results)} stocks qualified "
            f"(New: {new}, Tracking: {len(results) - new})"
        )
        return results

    @classmethod
    def classify_frame(cls, symbol: str, df: pd.DataFrame) -> Optional[GapUpRejectionClassification]:
        """Returns the most recent setup triggered within the tracking window of a chronological frame."""
        last = len(df) - 1
        for sessions_since in range(TRACK_SESSIONS + 1):
            trigger = last - sessions_since
            if trigger < MIN_ROWS_REQUIRED - 1:
                break
            if cls.is_trigger(symbol, df, trigger):
                return cls._build(symbol, df, trigger)
        return None

    @staticmethod
    def is_trigger(symbol: str, df: pd.DataFrame, trigger: int) -> bool:
        """True when row `trigger` is the T+1 of a Gap-Up Rejection."""
        signal = GapMomentumFilterService.evaluate_stock(symbol, df.iloc[trigger - 1], df.iloc[trigger - 2])
        if signal is None or signal.scenario != GAP_UP_SCENARIO:
            return False
        trigger_open = _to_float(df.iloc[trigger].get("Open"))
        trigger_close = _to_float(df.iloc[trigger].get("Close"))
        return trigger_open > signal.t_close and 0 < trigger_close < trigger_open

    @staticmethod
    def _build(symbol: str, df: pd.DataFrame, trigger: int) -> GapUpRejectionClassification:
        signal_row, trigger_row, latest = df.iloc[trigger - 1], df.iloc[trigger], df.iloc[-1]
        signal_close = _to_float(signal_row.get("Close"))
        trigger_open, trigger_close = _to_float(trigger_row.get("Open")), _to_float(trigger_row.get("Close"))
        last_close = _to_float(latest.get("Close"))

        # Excursions use the sessions after the trigger; High / Low fall back to Close when absent.
        after = df.iloc[trigger + 1:]
        highs = pd.to_numeric(after.get("High", after.get("Close")), errors="coerce").dropna()
        lows = pd.to_numeric(after.get("Low", after.get("Close")), errors="coerce").dropna()
        sessions_since = len(df) - 1 - trigger

        return GapUpRejectionClassification(
            symbol=symbol,
            label=LABEL,
            sentiment=SENTIMENT_BEARISH,
            status=STATUS_NEW if sessions_since == 0 else STATUS_TRACKING,
            signal_date=_format_date(signal_row.get("Date")),
            trigger_date=_format_date(trigger_row.get("Date")),
            sessions_since_trigger=sessions_since,
            signal_close=signal_close,
            trigger_open=trigger_open,
            trigger_close=trigger_close,
            gap_pct=_pct_change(trigger_open, signal_close),
            trigger_change_pct=_pct_change(trigger_close, trigger_open),
            last_close=last_close,
            return_pct=_pct_change(last_close, trigger_close),
            max_gain_pct=_pct_change(highs.max(), trigger_close) if len(highs) else 0.0,
            max_drawdown_pct=_pct_change(lows.min(), trigger_close) if len(lows) else 0.0,
            above_signal_close=last_close > signal_close,
        )

    @staticmethod
    def to_details(classification: GapUpRejectionClassification) -> Dict:
        """Serialises a classification into the rounded dict published to data.json."""
        details = asdict(classification)
        for key in _ROUNDED:
            details[key] = round(details[key], 2)
        details["track_sessions"] = TRACK_SESSIONS
        return details


def _last_date(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Date of the final row, or None when the frame has no usable date."""
    if "Date" not in df.columns or df.empty:
        return None
    parsed = pd.to_datetime(df["Date"].iloc[-1], errors="coerce")
    return None if pd.isna(parsed) else parsed
