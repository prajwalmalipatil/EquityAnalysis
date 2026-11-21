"""
gap_up_rejection_email_renderer.py
Renders the Gap-Up Rejection section of the daily email (HTML + plain text).
Takes the `gap_up_rejection_filters` block of data.json, returns strings.
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

TITLE = "Gap-Up Rejection Watch"
CRITERIA = ("Gap-Up Continuation on T, then T+1 opened above the T close but closed red. "
            "Tracked for the next 10 sessions from the T+1 close.")
EMPTY_MESSAGE = "No new or active Gap-Up Rejection setups."

SECTION = WatchSection(
    title=TITLE, icon="⚠️", criteria=CRITERIA, empty_message=EMPTY_MESSAGE,
    accent="#f87171", title_color="#fca5a5",
    new_heading="▼ New today", new_color="#f87171",
    new_columns=[
        ("T Close", num("signal_close", ",.2f")),
        ("T+1 Open", num("trigger_open", ",.2f")),
        ("T+1 Close", num("trigger_close", ",.2f")),
        ("Gap %", num("gap_pct", "+.2f")),
        ("Open-Close %", num("trigger_change_pct", "+.2f")),
    ],
    track_color="#fbbf24",
    track_columns=[("Triggered", text("trigger_date")), ("Session", session)] + TRACKING_COLUMNS,
)


def render_gap_up_rejection_html(filters: Optional[Dict]) -> str:
    """Returns the HTML section, or '' when the indicator block is absent."""
    return render_html(SECTION, filters)


def render_gap_up_rejection_text(filters: Optional[Dict]) -> str:
    """Returns the plain-text section, or '' when the indicator block is absent."""
    return render_text(SECTION, filters)
