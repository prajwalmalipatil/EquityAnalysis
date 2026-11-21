"""
gap_up_watch_email_renderer.py
Shared email rendering for the gap-up watch sections (setups tracked for ten sessions
after their T+1 trigger). Each section shows two tables: setups triggered in the
latest session, and earlier setups still inside their tracking window.

Pure presentation: takes a `{"daily": [...]}` block of data.json, returns strings.
"""

from dataclasses import dataclass
from html import escape
from typing import Callable, Dict, List, Optional, Tuple

MISSING = "--"
STATUS_NEW = "New"
DEFAULT_TRACK_SESSIONS = 10

_TH = "padding: 7px 8px;"
_TD = "padding: 7px 8px; color: #cbd5e1; text-align: right;"

Column = Tuple[str, Callable[[Dict], str]]


def num(key: str, spec: str, suffix: str = "") -> Callable[[Dict], str]:
    """Formatter for a numeric field; percentages carry a % suffix."""
    def fmt(item: Dict) -> str:
        value = item.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return MISSING
        return f"{value:{spec}}" + ("%" if key.endswith("_pct") else suffix)
    return fmt


def text(key: str) -> Callable[[Dict], str]:
    return lambda item: escape(str(item.get(key) or MISSING))


def session(item: Dict) -> str:
    since = item.get("sessions_since_trigger")
    return f"{since}/{item.get('track_sessions', DEFAULT_TRACK_SESSIONS)}" if isinstance(since, int) else MISSING


def versus_signal_close(item: Dict) -> str:
    above = item.get("above_signal_close")
    return MISSING if above is None else ("Above" if above else "Below")


# Columns every tracking table ends with: the move since the T+1 close.
TRACKING_COLUMNS: List[Column] = [
    ("T+1 Close", num("trigger_close", ",.2f")),
    ("Last", num("last_close", ",.2f")),
    ("Since %", num("return_pct", "+.2f")),
    ("Max Up %", num("max_gain_pct", "+.2f")),
    ("Max Down %", num("max_drawdown_pct", "+.2f")),
    ("vs T Close", versus_signal_close),
]


@dataclass(frozen=True)
class WatchSection:
    title: str
    icon: str
    criteria: str
    empty_message: str
    accent: str             # left border of the section
    title_color: str
    new_heading: str
    new_color: str
    new_columns: List[Column]
    track_color: str
    track_columns: List[Column]
    new_sort_key: str = "gap_pct"


def render_html(section: WatchSection, filters: Optional[Dict]) -> str:
    """Returns the HTML section, or '' when the indicator block is absent."""
    if filters is None:
        return ""
    new, tracking = split(filters, section.new_sort_key)
    body = (_html_table(new, section.new_heading, "Triggered in the latest session", section.new_color, section.new_columns)
            + _html_table(tracking, "◷ Still tracking", "Triggered earlier; move since the T+1 close",
                          section.track_color, section.track_columns))
    if not new and not tracking:
        body = f'<p style="color: #94a3b8; margin: 10px 0 0 0; font-size: 13px;">{section.empty_message}</p>'
    return f"""
    <div style="background-color: #334155; padding: 20px; border-radius: 8px; margin: 30px 0; border-left: 4px solid {section.accent};">
        <h3 style="color: {section.title_color}; margin: 0 0 6px 0; font-size: 18px;">{section.icon} {section.title}</h3>
        <p style="color: #cbd5e1; margin: 0; font-size: 14px;">{_summary_line(new, tracking)}</p>
        <p style="color: #94a3b8; margin: 6px 0 4px 0; font-size: 12px;">{section.criteria}</p>
        {body}
    </div>"""


def render_text(section: WatchSection, filters: Optional[Dict]) -> str:
    """Returns the plain-text section, or '' when the indicator block is absent."""
    if filters is None:
        return ""
    new, tracking = split(filters, section.new_sort_key)
    lines = [f"\n{section.icon} {section.title}: {_summary_line(new, tracking, html=False)}", f"   {section.criteria}\n"]
    if not new and not tracking:
        lines.append(f"   {section.empty_message}\n")
    lines.extend(_text_table(new, "New today", section.new_columns))
    lines.extend(_text_table(tracking, "Still tracking", section.track_columns))
    return "\n".join(lines)


def count_new(filters: Optional[Dict]) -> int:
    """Number of setups triggered in the latest session."""
    return len(split(filters or {})[0])


def split(filters: Dict, new_sort_key: str = "gap_pct") -> Tuple[List[Dict], List[Dict]]:
    """(new today, still tracking): new sorted by `new_sort_key` descending, tracking by most recent trigger."""
    items = [i for i in filters.get("daily", []) if isinstance(i, dict)]
    new = sorted((i for i in items if i.get("status") == STATUS_NEW),
                 key=lambda i: i.get(new_sort_key) or 0, reverse=True)
    tracking = sorted((i for i in items if i.get("status") != STATUS_NEW),
                      key=lambda i: (i.get("sessions_since_trigger") or 0, str(i.get("symbol"))))
    return new, tracking


def _summary_line(new: List[Dict], tracking: List[Dict], html: bool = True) -> str:
    count = f"<strong>{len(new)}</strong>" if html else str(len(new))
    return f"{count} new today, {len(tracking)} still tracking"


def _html_table(items: List[Dict], heading: str, rule: str, color: str, columns: List[Column]) -> str:
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


def _text_table(items: List[Dict], heading: str, columns: List[Column]) -> List[str]:
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
