"""
gap_up_expansion_email_renderer.py
Renders the Gap-Up Expansion section of the daily email (HTML + plain text).
Takes the `gap_up_expansion_filters` block of data.json, returns strings.

T+1 can close green or red; red ones also appear in the Gap-Up Rejection watch,
so the candle colour is shown in both tables.
"""

from typing import Dict, Optional

from src.services.reporting.gap_up_watch_email_renderer import (  # noqa: F401  (count_new re-exported)
    TRACKING_COLUMNS,
    WatchSection,
    count_new,
    num,
    render_html,
    render_text,
    session,
    text,
)

TITLE = "Gap-Up Expansion Watch"
CRITERIA = ("Gap-Up Continuation on T, then T+1 opened above the T close with a wider spread (High - Low) than T. "
            "Tracked for the next 10 sessions from the T+1 close. Red T+1 candles also appear in the Rejection watch.")
EMPTY_MESSAGE = "No new or active Gap-Up Expansion setups."

SECTION = WatchSection(
    title=TITLE, icon="📈", criteria=CRITERIA, empty_message=EMPTY_MESSAGE,
    accent="#38bdf8", title_color="#7dd3fc",
    new_heading="◆ New today", new_color="#38bdf8",
    new_columns=[
        ("T Close", num("signal_close", ",.2f")),
        ("T+1 Open", num("trigger_open", ",.2f")),
        ("T+1 Close", num("trigger_close", ",.2f")),
        ("Gap %", num("gap_pct", "+.2f")),
        ("Spread x T", num("spread_ratio", ".2f", suffix="x")),
        ("T+1 Candle", text("trigger_candle")),
    ],
    track_color="#a5b4fc",
    track_columns=[("Triggered", text("trigger_date")), ("Session", session), ("Candle", text("trigger_candle"))]
    + TRACKING_COLUMNS,
    new_sort_key="spread_ratio",
)


def render_gap_up_expansion_html(filters: Optional[Dict]) -> str:
    """Returns the HTML section, or '' when the indicator block is absent."""
    return render_html(SECTION, filters)


def render_gap_up_expansion_text(filters: Optional[Dict]) -> str:
    """Returns the plain-text section, or '' when the indicator block is absent."""
    return render_text(SECTION, filters)
