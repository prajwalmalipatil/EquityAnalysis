"""
gap_momentum_email_renderer.py
Renders the Gap Momentum indicator section of the daily email (HTML + plain text).
Pure presentation: takes the `gap_momentum_filters` block of data.json, returns strings.
"""

from html import escape
from typing import Dict, List, Optional, Tuple

TITLE = "Gap Momentum Insights"
CRITERIA = "Volume and spread both above T-1, with the candle confirming the opening gap."
EMPTY_MESSAGE = "No stocks met the criteria in the latest session."
MISSING = "--"

# (heading, sentiment, accent colour, rule shown to the reader)
_GROUPS: List[Tuple[str, str, str, str]] = [
    ("▲ Gap-Up Continuation", "Bullish", "#4ade80", "Open above T-1 close, closed green"),
    ("▼ Gap-Down Continuation", "Bearish", "#f87171", "Open below T-1 close, closed red"),
]
# (header, data key, format spec)
_COLUMNS: List[Tuple[str, str, str]] = [
    ("T-1 Close", "t1_close", ",.2f"),
    ("Open", "t_open", ",.2f"),
    ("Close", "t_close", ",.2f"),
    ("Gap %", "gap_pct", "+.2f"),
    ("Chg %", "change_pct", "+.2f"),
    ("Vol Δ%", "volume_pct", "+.1f"),
    ("Spread Δ%", "spread_pct", "+.1f"),
]
_TH = "padding: 7px 8px;"
_TD = "padding: 7px 8px; color: #cbd5e1; text-align: right;"


def render_gap_momentum_html(filters: Optional[Dict]) -> str:
    """Returns the HTML section, or '' when the indicator block is absent."""
    if filters is None:
        return ""
    items = _daily_items(filters)
    body = "".join(_html_group(items, *group) for group in _GROUPS)
    if not items:
        body = f'<p style="color: #94a3b8; margin: 10px 0 0 0; font-size: 13px;">{EMPTY_MESSAGE}</p>'

    return f"""
    <div style="background-color: #334155; padding: 20px; border-radius: 8px; margin: 30px 0; border-left: 4px solid #fb923c;">
        <h3 style="color: #fdba74; margin: 0 0 6px 0; font-size: 18px;">🚀 {TITLE}</h3>
        <p style="color: #cbd5e1; margin: 0; font-size: 14px;">{_summary_line(items)}</p>
        <p style="color: #94a3b8; margin: 6px 0 4px 0; font-size: 12px;">{CRITERIA}</p>
        {body}
    </div>"""


def render_gap_momentum_text(filters: Optional[Dict]) -> str:
    """Returns the plain-text section, or '' when the indicator block is absent."""
    if filters is None:
        return ""
    items = _daily_items(filters)
    lines = [f"\n🚀 {TITLE}: {_summary_line(items, html=False)}", f"   {CRITERIA}\n"]
    if not items:
        lines.append(f"   {EMPTY_MESSAGE}\n")
    for heading, sentiment, _, rule in _GROUPS:
        lines.extend(_text_group(_for_sentiment(items, sentiment), heading, rule))
    return "\n".join(lines)


def _daily_items(filters: Dict) -> List[Dict]:
    """Daily signals, strongest volume expansion first."""
    items = [i for i in filters.get("daily", []) if isinstance(i, dict)]
    return sorted(items, key=lambda i: i.get("volume_pct") or 0, reverse=True)


def _for_sentiment(items: List[Dict], sentiment: str) -> List[Dict]:
    return [i for i in items if i.get("sentiment") == sentiment]


def _summary_line(items: List[Dict], html: bool = True) -> str:
    """One-line count summary including the trading day the signals belong to."""
    count = f"<strong>{len(items)}</strong>" if html else str(len(items))
    dates = sorted({str(i.get("signal_date")) for i in items if i.get("signal_date")})
    session = f" for session {escape(dates[-1])}" if dates else ""
    up, down = (len(_for_sentiment(items, g[1])) for g in _GROUPS)
    return f"{count} stock{'' if len(items) == 1 else 's'} detected{session} (Gap-Up: {up}, Gap-Down: {down})"


def _fmt(item: Dict, key: str, spec: str) -> str:
    """Formats a numeric field; percentages carry a % suffix."""
    value = item.get(key)
    if not isinstance(value, (int, float)):
        return MISSING
    return f"{value:{spec}}" + ("%" if key.endswith("_pct") else "")


def _html_group(items: List[Dict], heading: str, sentiment: str, color: str, rule: str) -> str:
    """Renders one scenario table; '' when no stock falls in the scenario."""
    group = _for_sentiment(items, sentiment)
    if not group:
        return ""
    headers = "".join(f'<th style="{_TH} text-align: right;">{h}</th>' for h, _, _ in _COLUMNS)
    rows = "".join(_html_row(item, index, color) for index, item in enumerate(group))
    return f"""
        <div style="margin-top: 14px;">
            <div style="font-weight: bold; color: {color}; font-size: 13px; margin-bottom: 2px;">{heading} ({len(group)})</div>
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


def _html_row(item: Dict, index: int, color: str) -> str:
    background = "#1e293b" if index % 2 == 0 else "#253347"
    symbol = escape(str(item.get("symbol", MISSING)))
    cells = "".join(f'<td style="{_TD}">{_fmt(item, key, spec)}</td>' for _, key, spec in _COLUMNS)
    return f"""
                    <tr style="background-color: {background}; border-bottom: 1px solid #334155;">
                        <td style="padding: 7px 8px; font-weight: bold; color: {color};">{symbol}</td>{cells}
                    </tr>"""


def _text_group(group: List[Dict], heading: str, rule: str) -> List[str]:
    """Renders one scenario as fixed-width text lines; [] when empty."""
    if not group:
        return []
    header = f"  {'SYMBOL':<12}" + "".join(f"{h.upper():>11}" for h, _, _ in _COLUMNS)
    lines = [f"  {heading} ({len(group)}) - {rule}", header, f"  {'-' * (len(header) - 2)}"]
    for item in group:
        cells = "".join(f"{_fmt(item, key, spec):>11}" for _, key, spec in _COLUMNS)
        lines.append(f"  {str(item.get('symbol', MISSING)):<12}{cells}")
    lines.append("")
    return lines
