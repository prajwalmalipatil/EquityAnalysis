"""
gap_up_watch_base.py
Shared logic for setups that start with a Gap-Up Continuation on T and are confirmed
by what the next session (T+1) does. Each setup is reported on its trigger day (T+1)
and then tracked for TRACK_SESSIONS sessions, measured from the T+1 close.

Subclasses define the T+1 condition, the classification they return and where the
qualifying files are copied.
"""

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.constants import vsa_constants as const
from src.models.vsa_models import GapMomentumClassification
from src.services.vsa.gap_momentum_filter_service import (
    GapMomentumFilterService,
    _format_date,
    _pct_change,
    _to_float,
)
from src.utils.observability import get_tenant_logger

TRACK_SESSIONS = 10          # sessions followed after the trigger day
MIN_ROWS_REQUIRED = 3        # T-1, T and T+1
GAP_UP_SCENARIO = "Gap_Up_Green_Close"
STATUS_NEW, STATUS_TRACKING = "New", "Tracking"
TRACKING_ROUNDED = (
    "signal_close", "trigger_open", "trigger_close", "gap_pct", "trigger_change_pct",
    "last_close", "return_pct", "max_gain_pct", "max_drawdown_pct",
)


class GapUpWatchService:
    """Finds each stock's most recent setup that is still inside its tracking window."""

    OUTPUT_DIR_NAME: str = ""
    LOG_PREFIX: str = ""
    ROUNDED: Tuple[str, ...] = TRACKING_ROUNDED

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results_dir = base_dir / const.RESULTS_DIR_NAME
        self.output_dir = base_dir / self.OUTPUT_DIR_NAME
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_tenant_logger(f"{self.LOG_PREFIX.lower().replace('_', '-')}-filter-service")

    def scan_and_classify(self) -> List:
        """Reads each Excel in Results/, evaluates the setup, copies qualifying files."""
        if not self.results_dir.exists():
            self.logger.warning(f"{self.LOG_PREFIX}_RESULTS_DIR_MISSING", extra={"path": str(self.results_dir)})
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

        results = []
        for xlsx_path, df in frames.items():
            if last_dates[xlsx_path] != market_date:
                continue
            classification = self.classify_frame(xlsx_path.stem.replace("_VSA", ""), df)
            if classification is None:
                continue
            shutil.copy(xlsx_path, self.output_dir)
            results.append(classification)

        new = sum(1 for r in results if r.status == STATUS_NEW)
        self.logger.info(
            f"{self.LOG_PREFIX}_FILTER_COMPLETE: {len(results)} stocks qualified "
            f"(New: {new}, Tracking: {len(results) - new})"
        )
        return results

    @classmethod
    def classify_frame(cls, symbol: str, df: pd.DataFrame):
        """Returns the most recent setup triggered within the tracking window of a chronological frame."""
        last = len(df) - 1
        for sessions_since in range(TRACK_SESSIONS + 1):
            trigger = last - sessions_since
            if trigger < MIN_ROWS_REQUIRED - 1:
                break
            if cls.is_trigger(symbol, df, trigger):
                return cls._build(symbol, df, trigger)
        return None

    @classmethod
    def is_trigger(cls, symbol: str, df: pd.DataFrame, trigger: int) -> bool:
        """True when row `trigger` is the T+1 of this setup."""
        signal = GapMomentumFilterService.evaluate_stock(symbol, df.iloc[trigger - 1], df.iloc[trigger - 2])
        if signal is None or signal.scenario != GAP_UP_SCENARIO:
            return False
        trigger_row = df.iloc[trigger]
        if not _to_float(trigger_row.get("Open")) > signal.t_close:
            return False
        return cls._t1_condition(signal, trigger_row)

    @classmethod
    def _t1_condition(cls, signal: GapMomentumClassification, trigger_row: pd.Series) -> bool:
        raise NotImplementedError

    @classmethod
    def _build(cls, symbol: str, df: pd.DataFrame, trigger: int):
        raise NotImplementedError

    @classmethod
    def to_details(cls, classification) -> Dict:
        """Serialises a classification into the rounded dict published to data.json."""
        details = asdict(classification)
        for key in cls.ROUNDED:
            details[key] = round(details[key], 2)
        details["track_sessions"] = TRACK_SESSIONS
        return details


def tracking_fields(df: pd.DataFrame, trigger: int) -> Dict:
    """Fields shared by every setup: the T / T+1 prices and the move since the T+1 close."""
    signal_row, trigger_row, latest = df.iloc[trigger - 1], df.iloc[trigger], df.iloc[-1]
    signal_close = _to_float(signal_row.get("Close"))
    trigger_open, trigger_close = _to_float(trigger_row.get("Open")), _to_float(trigger_row.get("Close"))
    last_close = _to_float(latest.get("Close"))

    # Excursions use the sessions after the trigger; High / Low fall back to Close when absent.
    after = df.iloc[trigger + 1:]
    highs = pd.to_numeric(after.get("High", after.get("Close")), errors="coerce").dropna()
    lows = pd.to_numeric(after.get("Low", after.get("Close")), errors="coerce").dropna()
    sessions_since = len(df) - 1 - trigger

    return {
        "status": STATUS_NEW if sessions_since == 0 else STATUS_TRACKING,
        "signal_date": _format_date(signal_row.get("Date")),
        "trigger_date": _format_date(trigger_row.get("Date")),
        "sessions_since_trigger": sessions_since,
        "signal_close": signal_close,
        "trigger_open": trigger_open,
        "trigger_close": trigger_close,
        "gap_pct": _pct_change(trigger_open, signal_close),
        "trigger_change_pct": _pct_change(trigger_close, trigger_open),
        "last_close": last_close,
        "return_pct": _pct_change(last_close, trigger_close),
        "max_gain_pct": _pct_change(highs.max(), trigger_close) if len(highs) else 0.0,
        "max_drawdown_pct": _pct_change(lows.min(), trigger_close) if len(lows) else 0.0,
        "above_signal_close": last_close > signal_close,
    }


def bar_spread(row: pd.Series) -> float:
    """High - Low of a bar, preferring the Spread column the VSA processor writes."""
    spread = _to_float(row.get("Spread"))
    return spread if spread > 0 else _to_float(row.get("High")) - _to_float(row.get("Low"))


def _last_date(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Date of the final row, or None when the frame has no usable date."""
    if "Date" not in df.columns or df.empty:
        return None
    parsed = pd.to_datetime(df["Date"].iloc[-1], errors="coerce")
    return None if pd.isna(parsed) else parsed
