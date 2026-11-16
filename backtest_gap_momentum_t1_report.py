"""
backtest_gap_momentum_t1_report.py
Excel presentation layer for backtest_gap_momentum_t1.py.
One tear sheet per T+1 opening pattern, plus a summary, the trade list and chart data.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from backtest_gap_momentum import BEARISH, BULLISH
from backtest_gap_momentum_report import (
    AMBER, BLUE, CANVAS, CHART_ROWS, CORAL, GREY, NAVY, SLATE, TEAL,
    _bar, _data_sheet, _kpi_card, _line, write_table,
)
from backtest_gap_momentum_t1 import (
    ALL_SIGNALS, BASELINE, COST_ROUND_TRIP_PCT, HIGHER, HOLDS, INTRADAY, PATTERNS, SIGNIFICANT_T, TOP_TRADES,
)

BANNER_COLS = 24
SECOND_CHART_COL = 11      # every column left of this holds a table column at least 11 wide
SERIES = [BULLISH, BEARISH, BASELINE]
SERIES_COLORS = [TEAL, CORAL, GREY]
SHORT_HOLD = {label: label.replace("T+1 Open to ", "") for label, _, _ in HOLDS}

METHOD = [
    "T is the day a Gap Momentum signal fired. Each signal is split by how the next session (T+1) opened against the close of T.",
    "Entry is the T+1 open, the first price at which the split is known. Exits are the close of T+1, T+2, T+3, T+5 and T+10.",
    "Returns are raw price moves in percent: positive means the price rose. A reliably negative mean is a short-side edge.",
    "Baseline is every stock-day in the universe whose next session opened the same way, with or without a signal.",
    "Excess is the cohort mean minus the baseline mean. It isolates what the signal adds beyond the opening gap itself.",
    "t-stat (per trade) treats every signal as independent. t-stat (per day) averages all signals on a day first, because signals cluster on market-wide days.",
    f"A result is called significant when the per-day excess t-stat is at least {SIGNIFICANT_T:.0f} in absolute value.",
    f"Excess Net of Cost subtracts an assumed {COST_ROUND_TRIP_PCT:.2f}% round-trip cost from the absolute excess. All other figures are gross.",
    "Gap Filled: T+1 traded back to the T close during the session. Gap Held: T+1 closed on the same side of the T close as it opened.",
    "Prices are back-adjusted for splits, bonuses and demergers. One year of data is a single market regime; treat results as indicative.",
]


def _banner(ws: Worksheet, text: str) -> None:
    ws.merge_cells(start_row=1, start_column=1, end_row=2, end_column=BANNER_COLS)
    cell = ws.cell(row=1, column=1, value=f"  {text}")
    cell.font, cell.alignment = Font(size=16, bold=True, color="FFFFFF"), Alignment(vertical="center")
    for col in range(1, BANNER_COLS + 1):
        for row in (1, 2):
            ws.cell(row=row, column=col).fill = PatternFill("solid", fgColor=NAVY)


def _section(ws: Worksheet, row: int, title: str, note: str = "") -> int:
    """Writes a section heading (and optional reading note); returns the next free row."""
    ws.cell(row=row, column=1, value=title).font = Font(bold=True, size=12, color=NAVY)
    if note:
        ws.cell(row=row + 1, column=1, value=note).font = Font(italic=True, size=9, color=SLATE)
        return row + 2
    return row + 1


def _table(ws: Worksheet, row: int, title: str, df: pd.DataFrame, note: str = "", color: bool = True) -> int:
    top = _section(ws, row, title, note)
    _, last = write_table(ws, df, top=top, color_returns=color)
    return last + 3


# ───────────────────────────────── chart data ─────────────────────────────────
def _pivot(df: pd.DataFrame, index: str, value: str, columns: List[str]) -> pd.DataFrame:
    table = df.pivot_table(index=index, columns="Cohort", values=value, aggfunc="first", observed=True, sort=False)
    return table.reindex(columns=columns).reset_index()


def _chart_frames(tables: Dict) -> Dict[str, pd.DataFrame]:
    perf = tables["performance"].assign(Hold=lambda t: t["Holding Period"].map(SHORT_HOLD))
    mean = _pivot(perf, "Hold", "Mean %", [BULLISH, BEARISH])
    mean[BASELINE] = perf[perf["Cohort"] == ALL_SIGNALS]["Baseline Mean %"].values
    equity = tables["equity"].copy()
    equity["T+1 Date"] = equity["T+1 Date"].dt.strftime("%d-%b-%y")
    return {
        "mean": mean.rename(columns={"Hold": "Exit"}),
        "win": _pivot(perf, "Hold", "Win Rate %", [BULLISH, BEARISH]).rename(columns={"Hold": "Exit"}),
        "equity": equity[["T+1 Date"] + SERIES],
        "gap": _pivot(tables["gap"], "T+1 Gap Size", "Mean Open to Close %", SERIES),
        "fill": _pivot(tables["gap"], "T+1 Gap Size", "Gap Filled Intraday %", SERIES),
        "month": _pivot(tables["month"].sort_values("Month"), "Month", "Mean Open to Close %", SERIES),
    }


def _write_chart_data(ws: Worksheet, row: int, pattern: str, frames: Dict[str, pd.DataFrame]) -> Tuple[Dict, int]:
    ws.cell(row=row, column=1, value=pattern).font = Font(bold=True, size=12, color=NAVY)
    positions, row = {}, row + 2
    for name, df in frames.items():
        _, last = write_table(ws, df, top=row, color_returns=False)
        positions[name] = (row, max(last, row + 1), len(df.columns))
        row = last + 3
    return positions, row


# ───────────────────────────────── tear sheet ─────────────────────────────────
def _kpis(tables: Dict) -> List[Tuple[str, str, str, str]]:
    perf = tables["performance"]
    row = perf[(perf["Cohort"] == ALL_SIGNALS)].set_index("Holding Period")
    day, week = row.loc[HOLDS[0][0]], row.loc[HOLDS[3][0]]
    behaviour = tables["behaviour"].set_index("Cohort")
    signal, base = behaviour.loc[ALL_SIGNALS], behaviour.loc[BASELINE]
    return [
        ("SIGNALS IN THIS PATTERN", f"{int(day['Trades']):,}", f"across {int(day['Trade Days'])} trading days", BLUE),
        ("OPEN TO CLOSE, MEAN", f"{day['Mean %']:+.2f}%", f"baseline {day['Baseline Mean %']:+.2f}%", TEAL),
        ("OPEN TO CLOSE, WIN RATE", f"{day['Win Rate %']:.1f}%", f"median {day['Median %']:+.2f}%", TEAL),
        ("EXCESS t-STAT (PER DAY)", f"{day['Excess t-stat (per day)']:+.2f}", f"significant at |t| >= {SIGNIFICANT_T:.0f}", AMBER),
        ("GAP FILLED INTRADAY", f"{signal['Gap Filled Intraday %']:.1f}%", f"baseline {base['Gap Filled Intraday %']:.1f}%", CORAL),
        ("OPEN TO T+5 CLOSE, MEAN", f"{week['Mean %']:+.2f}%", f"excess {week['Excess vs Baseline %']:+.2f}%", BLUE),
    ]


def _tear_sheet(wb: Workbook, pattern: str, tables: Dict, ws_data: Worksheet, pos: Dict, meta: Dict) -> None:
    ws = wb.create_sheet(pattern)
    ws.sheet_view.showGridLines = False
    ws.sheet_view.zoomScale = 85
    rule = "T+1 open above T close" if pattern == HIGHER else "T+1 open below T close"
    _banner(ws, f"GAP MOMENTUM  |  {pattern.upper()}  ({rule})  |  "
                f"{meta['first_date']:%d %b %Y} - {meta['last_date']:%d %b %Y}")
    for col in range(1, BANNER_COLS + 1):
        for row in range(3, 9):
            ws.cell(row=row, column=col).fill = PatternFill("solid", fgColor=CANVAS)
    for index, card in enumerate(_kpis(tables)):
        _kpi_card(ws, 1 + index * 4, *card)

    row = _table(ws, 10, "1. Performance from the T+1 open", tables["performance"],
                 "Raw price return in percent, entry at the T+1 open. Excess = cohort mean minus baseline mean for the same exit.")
    row = _table(ws, row, "2. Anatomy of the T+1 session", tables["behaviour"],
                 "How the session after the signal traded, next to every stock-day that opened the same way.", color=False)
    row = _table(ws, row, "3. By size of the T+1 opening gap", tables["gap"],
                 "Open-to-close result split by how far T+1 opened from the T close.")
    row = _table(ws, row, "4. Stability by month", tables["month"],
                 "An edge that is real should not depend on one or two months.")

    row = _section(ws, row, "5. Charts")
    y = "Mean return %"
    charts = [
        _bar(ws_data, pos["mean"], "Mean return from the T+1 open, by exit", y, SERIES_COLORS),
        _bar(ws_data, pos["win"], "Win rate (price up) from the T+1 open, by exit", "Win rate %", SERIES_COLORS, number_format="0"),
        _line(ws_data, pos["equity"], "Cumulative T+1 open-to-close return (sum of daily average, %)", "Cumulative %", SERIES_COLORS),
        _bar(ws_data, pos["month"], "Mean T+1 open-to-close return by month", y, SERIES_COLORS),
        _bar(ws_data, pos["gap"], "Mean T+1 open-to-close return by opening gap size", y, SERIES_COLORS),
        _bar(ws_data, pos["fill"], "Share of gaps filled during T+1, by opening gap size", "Filled %", SERIES_COLORS, number_format="0"),
    ]
    for index, chart in enumerate(charts):
        anchor_col = 1 if index % 2 == 0 else SECOND_CHART_COL
        ws.add_chart(chart, f"{get_column_letter(anchor_col)}{row + (index // 2) * CHART_ROWS}")
    row += CHART_ROWS * 3 + 1

    row = _table(ws, row, f"6. Largest T+1 open-to-close gains (top {TOP_TRADES})", tables["best"])
    _table(ws, row, f"7. Largest T+1 open-to-close losses (bottom {TOP_TRADES})", tables["worst"])
    ws.freeze_panes = "B3"


def _summary_sheet(wb: Workbook, meta: Dict) -> None:
    ws = wb.create_sheet("Summary")
    ws.sheet_view.showGridLines = False
    _banner(ws, "GAP MOMENTUM  |  T+1 OPEN FOLLOW-THROUGH STUDY")
    facts = [
        ("Period", f"{meta['first_date']:%d %b %Y} - {meta['last_date']:%d %b %Y}"),
        ("Universe", f"{meta['stocks']} NSE stocks, {meta['stock_days']:,} stock-days"),
        ("Gap Momentum signals", f"{meta['signals']:,} ({meta['signals_with_t1']:,} with a T+1 session that opened away from the T close)"),
    ]
    for offset, (label, value) in enumerate(facts):
        ws.cell(row=4 + offset, column=1, value=label).font = Font(bold=True, color=SLATE)
        ws.cell(row=4 + offset, column=2, value=value)

    row = _table(ws, 8, "1. How often does T+1 open higher or lower?", meta["frequency"],
                 "Share of events by T+1 opening direction. Flat opens are excluded.", color=False)
    row = _table(ws, row, "2. Scorecard", meta["scorecard"],
                 "One line per pattern, signal type and exit. The verdict is driven by the per-day excess t-stat and the cost assumption.")
    row = _section(ws, row, "3. Method and definitions")
    for line in METHOD:
        ws.cell(row=row, column=1, value=f"-  {line}")
        row += 1
    ws.column_dimensions["A"].width = 28


def write_workbook(out_path: Path, signals: pd.DataFrame, tables: Dict[str, Dict], meta: Dict) -> None:
    wb = Workbook()
    wb.remove(wb.active)
    _summary_sheet(wb, meta)

    ws_data = wb.create_sheet("Chart Data")
    ws_data.cell(row=1, column=1, value="Source tables behind the charts.").font = Font(italic=True, color=SLATE)
    row = 3
    for pattern in PATTERNS:
        positions, row = _write_chart_data(ws_data, row, pattern, _chart_frames(tables[pattern]))
        _tear_sheet(wb, pattern, tables[pattern], ws_data, positions, meta)

    from backtest_gap_momentum_t1_path import SETUPS, add_path_sheets
    path_sheets = []
    for setup in SETUPS:
        names, row = add_path_sheets(wb, meta["paths"][setup.key], setup, ws_data, row, meta)
        path_sheets += names

    columns = ["Symbol", "Date", "Signal", "Pattern", "Gap %", "Volume x T-1", "Spread x T-1", "High", "Low", "Close",
               "T+1 Date", "T+1 Open", "T+1 High", "T+1 Low", "T+1 Close", "T+1 Gap %", "T+1 Gap Size",
               "Gap Filled", "Gap Held", "Closed Green", "Broke T Range", "Open to High %", "Open to Low %",
               "T+1 Close vs T Close %"] + [column for _, column, _ in HOLDS]
    trades = signals[columns].rename(columns={
        "Date": "T Date", "Gap %": "T Gap %", "High": "T High", "Low": "T Low", "Close": "T Close"})
    for flag in ["Gap Filled", "Gap Held", "Closed Green", "Broke T Range"]:
        trades[flag] = trades[flag].map({True: "Yes", False: "No"})
    _data_sheet(wb, "T+1 Trades", trades.sort_values(["T Date", "Symbol"]),
                "Every Gap Momentum signal with its T+1 session. Returns are raw price moves from the T+1 open. Blank = not available yet.")

    order = ["Summary"] + PATTERNS + path_sheets + ["T+1 Trades", "Chart Data"]
    wb._sheets = [wb[name] for name in order]
    wb.save(out_path)
