"""
gap_up_expansion_filter_service.py
Post-VSA follow-through tracker for gap-up continuation with range expansion.

A Gap-Up Expansion is three consecutive sessions:
  T-1 -> T : T is a Gap-Up Continuation (see GapMomentumFilterService)
  T+1      : opens above the T close and trades a wider spread (High - Low) than T

T+1 may close green or red; the candle colour is reported. The setup is reported on
the trigger day (T+1) and then tracked for the next TRACK_SESSIONS sessions,
measured from the T+1 close.
"""

import pandas as pd

from src.constants import vsa_constants as const
from src.models.vsa_models import GapMomentumClassification, GapUpExpansionClassification
from src.services.vsa.gap_momentum_filter_service import _to_float
from src.services.vsa.gap_up_watch_base import (  # noqa: F401  (re-exported for callers and tests)
    STATUS_NEW,
    STATUS_TRACKING,
    TRACK_SESSIONS,
    TRACKING_ROUNDED,
    GapUpWatchService,
    bar_spread,
    tracking_fields,
)

LABEL = "Gap-Up Expansion"


def candle_colour(open_: float, close: float) -> str:
    return "Green" if close > open_ else "Red" if close < open_ else "Doji"


class GapUpExpansionFilterService(GapUpWatchService):
    """Gap-Up Continuation on T, then a T+1 that opened higher with a wider spread than T."""

    OUTPUT_DIR_NAME = const.GAP_UP_EXPANSION_FILTER_DIR_NAME
    LOG_PREFIX = "GAP_UP_EXPANSION"
    ROUNDED = TRACKING_ROUNDED + ("signal_spread", "trigger_spread", "spread_ratio")

    @classmethod
    def _t1_condition(cls, signal: GapMomentumClassification, trigger_row: pd.Series) -> bool:
        return signal.t_spread > 0 and bar_spread(trigger_row) > signal.t_spread

    @classmethod
    def _build(cls, symbol: str, df: pd.DataFrame, trigger: int) -> GapUpExpansionClassification:
        signal_spread, trigger_spread = bar_spread(df.iloc[trigger - 1]), bar_spread(df.iloc[trigger])
        fields = tracking_fields(df, trigger)
        return GapUpExpansionClassification(
            symbol=symbol, label=LABEL, **fields,
            signal_spread=signal_spread, trigger_spread=trigger_spread,
            spread_ratio=trigger_spread / signal_spread,
            trigger_candle=candle_colour(fields["trigger_open"], fields["trigger_close"]),
        )
