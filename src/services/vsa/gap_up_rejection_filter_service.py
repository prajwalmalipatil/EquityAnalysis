"""
gap_up_rejection_filter_service.py
Post-VSA follow-through tracker for failed gap-up continuation.

A Gap-Up Rejection is three consecutive sessions:
  T-1 -> T : T is a Gap-Up Continuation (see GapMomentumFilterService)
  T+1      : opens above the T close, but closes below its own open (red)

The setup is reported on the trigger day (T+1) and then tracked for the next
TRACK_SESSIONS sessions, measured from the T+1 close.
"""

import pandas as pd

from src.constants import vsa_constants as const
from src.models.vsa_models import GapMomentumClassification, GapUpRejectionClassification
from src.services.vsa.gap_momentum_filter_service import SENTIMENT_BEARISH, _to_float
from src.services.vsa.gap_up_watch_base import (  # noqa: F401  (re-exported for callers and tests)
    STATUS_NEW,
    STATUS_TRACKING,
    TRACK_SESSIONS,
    GapUpWatchService,
    tracking_fields,
)

LABEL = "Gap-Up Rejection"


class GapUpRejectionFilterService(GapUpWatchService):
    """Gap-Up Continuation on T, then a T+1 that opened higher but closed red."""

    OUTPUT_DIR_NAME = const.GAP_UP_REJECTION_FILTER_DIR_NAME
    LOG_PREFIX = "GAP_UP_REJECTION"

    @classmethod
    def _t1_condition(cls, signal: GapMomentumClassification, trigger_row: pd.Series) -> bool:
        trigger_open, trigger_close = _to_float(trigger_row.get("Open")), _to_float(trigger_row.get("Close"))
        return 0 < trigger_close < trigger_open

    @classmethod
    def _build(cls, symbol: str, df: pd.DataFrame, trigger: int) -> GapUpRejectionClassification:
        return GapUpRejectionClassification(
            symbol=symbol, label=LABEL, sentiment=SENTIMENT_BEARISH, **tracking_fields(df, trigger))
