"""
gap_up_rejection_email_renderer.py
Renders the Gap-Up Rejection section of the daily email (HTML + plain text).
Pure presentation: takes the `gap_up_rejection_filters` block of data.json, returns strings.

Two tables: setups triggered in the latest session, and earlier setups still inside
their tracking window with the move since the trigger close.
"""

from html import escape
from typing import Callable, Dict, List, Optional, Tuple

TITLE = "Gap-Up Rejection Watch"
CRITERIA = ("Gap-Up Continuation on T, then T+1 opened above the T close but closed red. "
            "Tracked for the next 10 sessions from the T+1 close.")
EMPTY_MESSAGE = "No new or active Gap-Up Rejection setups."
MISSING = "--"
STATUS_NEW = "New"
DEFAULT_TRACK_SESSIONS = 10
NEW_COLOR, TRACK_COLOR = "#f87171", "#fbbf24"

_TH = "padding: 7px 8px;"
_TD = "padding: 7px 8px; color: #cbd5e1; text-align: right;"


def _num(key: str, spec: str) -> Callable[[Dict], str]:
    """Formatter for a numeric field; percentages carry a % suffix."""
    def fmt(item: Dict) -> str:
        value = item.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return MISSING
        return f"{value:{spec}}" + ("%" if key.endswith("_pct") else "")
    return fmt


def _session(item: Dict) -> str:
    since = item.get("sessions_since_trigger")
    return f"{since}/{item.get('track_sessions', DEFAULT_TRACK_SESSIONS)}" if isinstance(since, int) else MISSING


def _versus_signal_close(item: Dict) -> str:
    above = item.get("above_signal_close")
    return MISSING if above is None else ("Above" if above else "Below")


def _text(key: str) -> Callable[[Dict], str]:
    return lambda item: escape(str(item.get(key) or MISSING))


_Column = Tuple[str, Callable[[Dict], str]]
_NEW_COLUMNS: List[_Column] = [
    ("T Close", _num("signal_close", ",.2f")),
    ("T+1 Open", _num("trigger_open", ",.2f")),
    ("T+1 Close", _num("trigger_close", ",.2f")),
    ("Gap %", _num("gap_pct", "+.2f")),
    ("Open-Close %", _num("trigger_change_pct", "+.2f")),
]
_TRACK_COLUMNS: List[_Column] = [
    ("Triggered", _text("trigger_date")),
    ("Session", _session),
    ("T+1 Close", _num("trigger_close", ",.2f")),
    ("Last", _num("last_close", ",.2f")),
    ("Since %", _num("return_pct", "+.2f")),
    ("Max Up %", _num("max_gain_pct", "+.2f")),
    ("Max Down %", _num("max_drawdown_pct", "+.2f")),
    ("vs T Close", _versus_signal_close),
]


def render_gap_up_rejection_html(filters: Optional[Dict]) -> str:
    """Returns the HTML section, or '' when the indicator block is absent."""
    if filters is None:
        return ""
    new, tracking = _split(filters)
    body = (_html_table(new, "▼ New today", "Triggered in the latest session", NEW_COLOR, _NEW_COLUMNS)
            + _html_table(tracking, "◷ Still tracking", "Triggered earlier; move since the T+1 close", TRACK_COLOR, _TRACK_COLUMNS))
    if not new and not tracking:
        body = f'<p style="color: #94a3b8; margin: 10px 0 0 0; font-size: 13px;">{EMPTY_MESSAGE}</p>'
    return f"""
    <div style="background-color: #334155; padding: 20px; border-radius: 8px; margin: 30px 0; border-left: 4px solid #f87171;">
        <h3 style="color: #fca5a5; margin: 0 0 6px 0; font-size: 18px;">⚠️ {TITLE}</h3>
        <p style="color: #cbd5e1; margin: 0; font-size: 14px;">{_summary_line(new, tracking)}</p>
        <p style="color: #94a3b8; margin: 6px 0 4px 0; font-size: 12px;">{CRITERIA}</p>
        {body}
    </div>"""


def render_gap_up_rejection_text(filters: Optional[Dict]) -> str:
    """Returns the plain-text section, or '' when the indicator block is absent."""
    if filters is None:
        return ""
    new, tracking = _split(filters)
    lines = [f"\n⚠️ {TITLE}: {_summary_line(new, tracking, html=False)}", f"   {CRITERIA}\n"]
    if not new and not tracking:
        lines.append(f"   {EMPTY_MESSAGE}\n")
    lines.extend(_text_table(new, "New today", _NEW_COLUMNS))
    lines.extend(_text_table(tracking, "Still tracking", _TRACK_COLUMNS))
    return "\n".join(lines)


def count_new(filters: Optional[Dict]) -> int:
    """Number of setups triggered in the latest session."""
    return len(_split(filters or {})[0])


def _split(filters: Dict) -> Tuple[List[Dict], List[Dict]]:
    """(new today, still tracking): largest rejected gap first, then most recent trigger first."""
    items = [i for i in filters.get("daily", []) if isinstance(i, dict)]
    new = sorted((i for i in items if i.get("status") == STATUS_NEW), key=lambda i: i.get("gap_pct") or 0, reverse=True)
    tracking = sorted((i for i in items if i.get("status") != STATUS_NEW),
                      key=lambda i: (i.get("sessions_since_trigger") or 0, str(i.get("symbol"))))
    return new, tracking


def _summary_line(new: List[Dict], tracking: List[Dict], html: bool = True) -> str:
    count = f"<strong>{len(new)}</strong>" if html else str(len(new))
    return f"{count} new today, {len(tracking)} still tracking"


def _html_table(items: List[Dict], heading: str, rule: str, color: str, columns: List[_Column]) -> str:
    """Renders one table; '' when it has no rows."""
    if not items:
        return ""
    headers = "".join(f'<th style="{_TH} text-align: right;">{h}</th>' for h, _ in columns)
    rows = ""
    for index, item in enumerate(items):
        background = "#1e293b" if index % 2 == 0 else "#253347"
        cells = "".join(f'<td style="{_TD}">{fmt(item)}</td>' for _, fmt in columns)
        rows += f"""
                    <tr style="background-color: {background}; border-bottom: 1px solid #334155;">
                        <td style="padding: 7px 8px; font-weight: bold; color: {color};">{escape(str(item.get("symbol", MISSING)))}</td>{cells}
                    </tr>"""
    return f"""
        <div style="margin-top: 14px;">
            <div style="font-weight: bold; color: {color}; font-size: 13px; margin-bottom: 2px;">{heading} ({len(items)})</div>
            <div style="color: #94a3b8; font-size: 11px; margin-bottom: 6px;">{rule}</div>
            <table style="width: 100%; border-collapse: collapse; font-size: 12px; text-align: left; background-color: #1e293b;">
                <thead>
                    <tr style="background-color: #0f172a; color: #94a3b8; font-size: 10px; text-transform: uppercase;">
                        <th style="{_TH}">Symbol</th>{headers}
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>"""


def _text_table(items: List[Dict], heading: str, columns: List[_Column]) -> List[str]:
    """Renders one table as fixed-width text lines; [] when empty."""
    if not items:
        return []
    header = f"  {'SYMBOL':<12}" + "".join(f"{h.upper():>13}" for h, _ in columns)
    lines = [f"  {heading} ({len(items)})", header, f"  {'-' * (len(header) - 2)}"]
    for item in items:
        cells = "".join(f"{fmt(item):>13}" for _, fmt in columns)
        lines.append(f"  {str(item.get('symbol', MISSING)):<12}{cells}")
    lines.append("")
    return lines
