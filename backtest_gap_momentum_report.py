"""
backtest_gap_momentum_report.py
Excel presentation layer for backtest_gap_momentum.py.
Builds the data sheets and a chart dashboard styled like a BI report page.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference, Series
from openpyxl.chart.label import DataLabelList
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from backtest_gap_momentum import (
    BEARISH, BULLISH, EXCURSION_WINDOW, HORIZONS, MIN_SIGNALS_PER_STOCK, PRIMARY_HORIZON, TOP_STOCKS,
)

# Palette
NAVY, SLATE, CANVAS, CARD = "1F2A44", "374649", "F3F2F1", "FFFFFF"
TEAL, CORAL, AMBER, GREY, BLUE = "01B8AA", "FD625E", "F2C80F", "A6A6A6", "3599B8"
GREEN_TEXT, RED_TEXT = "107C41", "C0392B"

HEADER_FONT = Font(bold=True, color="FFFFFF", size=10)
HEADER_FILL = PatternFill("solid", fgColor=NAVY)
THIN = Side(style="thin", color="D9D9D9")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

DASH_COLS = 24            # dashboard grid: 24 equal columns
DASH_COL_WIDTH = 9.5
CHART_W, CHART_H = 21.4, 8.6   # cm; one chart spans 12 grid columns and ~17 rows
CHART_ROWS = 18

P = PRIMARY_HORIZON
RET_COL = f"Return {P}D %"


# ───────────────────────────── generic table writer ───────────────────────────
def _number_format(column: str, sample) -> str:
    if isinstance(sample, pd.Timestamp):
        return "dd-mmm-yyyy"
    if "%" in column:
        return '0.00"%"'
    if column in ("Raw Ratio", "Factor Applied"):
        return "0.0000"
    if " x T-1" in column:
        return '0.00"x"'
    if column == "Volume":
        return "#,##0"
    if isinstance(sample, float):
        return "#,##0.00"
    return "General"


def write_table(ws: Worksheet, df: pd.DataFrame, top: int = 1, left: int = 1, color_returns: bool = True) -> Tuple[int, int]:
    """Writes df as a styled table; returns (first_data_row, last_data_row)."""
    df = df.copy()
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype(str)
    for j, column in enumerate(df.columns):
        cell = ws.cell(row=top, column=left + j, value=column)
        cell.font, cell.fill, cell.border = HEADER_FONT, HEADER_FILL, BORDER
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for i, record in enumerate(df.itertuples(index=False), start=top + 1):
        for j, value in enumerate(record):
            if pd.isna(value):
                value = None
            elif isinstance(value, pd.Timestamp):
                value = value.to_pydatetime()
            elif hasattr(value, "item"):
                value = value.item()
            ws.cell(row=i, column=left + j, value=value)

    first, last = top + 1, top + len(df)
    for j, column in enumerate(df.columns):
        letter = get_column_letter(left + j)
        non_null = df[column].dropna()
        fmt = _number_format(column, non_null.iloc[0] if len(non_null) else None)
        for row in range(first, last + 1):
            ws[f"{letter}{row}"].number_format = fmt
        longest = max([len(str(column))] + [len(str(v)) for v in non_null.head(200)])
        ws.column_dimensions[letter].width = max(ws.column_dimensions[letter].width or 0, min(max(longest + 2, 11), 26))
        is_return = "%" in column and "Win" not in column and column not in ("Gap %", "Day Chg %")
        if color_returns and is_return and last >= first:
            rng = f"{letter}{first}:{letter}{last}"
            ws.conditional_formatting.add(rng, CellIsRule(operator="greaterThan", formula=["0"], font=Font(color=GREEN_TEXT)))
            ws.conditional_formatting.add(rng, CellIsRule(operator="lessThan", formula=["0"], font=Font(color=RED_TEXT)))
    ws.row_dimensions[top].height = 30
    return first, last


def _title(ws: Worksheet, row: int, text: str, size: int = 12, color: str = NAVY) -> None:
    cell = ws.cell(row=row, column=1, value=text)
    cell.font = Font(bold=True, size=size, color=color)


def _data_sheet(wb: Workbook, name: str, df: pd.DataFrame, note: str = "") -> Worksheet:
    ws = wb.create_sheet(name)
    top = 1
    if note:
        ws.cell(row=1, column=1, value=note).font = Font(italic=True, color=SLATE)
        top = 3
    first, last = write_table(ws, df, top=top)
    ws.freeze_panes = ws.cell(row=first, column=2)
    ws.auto_filter.ref = f"A{top}:{get_column_letter(len(df.columns))}{max(last, top)}"
    return ws


# ─────────────────────────────────── sheets ───────────────────────────────────
def _overview_sheet(wb: Workbook, signals: pd.DataFrame, tables: Dict, meta: Dict) -> None:
    ws = wb.active
    ws.title = "Overview"
    ws.sheet_view.showGridLines = False
    ws.column_dimensions["A"].width = 26
    _title(ws, 1, "Gap Momentum indicator - 1 year historical study", size=16)
    lines = [
        ("Data", f"{meta['stocks']} NSE stocks, {meta['stock_days']:,} stock-days, "
                 f"{meta['first_date']:%d-%b-%Y} to {meta['last_date']:%d-%b-%Y}"),
        ("Signals found", f"{len(signals):,} ({(signals['Signal'] == BULLISH).sum():,} gap-up, "
                          f"{(signals['Signal'] == BEARISH).sum():,} gap-down)"),
        ("Rule", "Day T volume AND spread (High - Low) both above T-1, and either"),
        ("", f"{BULLISH}: T opens above T-1 close and closes above its open (green)"),
        ("", f"{BEARISH}: T opens below T-1 close and closes below its open (red)"),
        ("How returns are measured", "Entry at the close of day T. Return N D = close N trading days later vs that close."),
        ("", "Gap-up signals are scored as longs. Gap-down signals are scored as shorts, so a positive return means the price fell."),
        ("", f"Best / Worst in {EXCURSION_WINDOW}D = most favourable / most adverse price reached in the next {EXCURSION_WINDOW} sessions (uses High / Low)."),
        ("Baseline", "The same holding period measured on every stock-day in the data, signal or not. Edge = signal average minus baseline."),
        ("Read with care", "No brokerage, slippage, taxes or short-selling constraints are included."),
        ("", "Signals cluster on market-wide days. 'Avg per Signal Day' scores each trading day once and is the more reliable number."),
        ("Corporate actions", f"{len(meta['actions'])} splits / bonuses / demergers were back-adjusted so they do not appear as fake gaps (see Corporate Actions sheet)."),
        ("", "One year is a single market regime. Recent signals have no result yet for the longer horizons and show blank."),
    ]
    if meta["missing"]:
        lines.append(("Not downloaded", ", ".join(meta["missing"])))
    for offset, (label, text) in enumerate(lines, start=3):
        ws.cell(row=offset, column=1, value=label).font = Font(bold=True, color=SLATE)
        ws.cell(row=offset, column=2, value=text)
    start = len(lines) + 5
    _title(ws, start, "Performance by holding period")
    write_table(ws, tables["horizons"], top=start + 1)
    ws.column_dimensions["A"].width = 26


def _outstanding_sheet(wb: Workbook, tables: Dict) -> None:
    ws = wb.create_sheet("Outstanding Returns")
    ws.sheet_view.showGridLines = False
    _title(ws, 1, f"Outstanding returns - ranked by {P}-day return after the signal close", size=14)
    row = 3
    blocks = [
        (f"Top {len(tables['top_bullish'])} {BULLISH} signals (long)", tables["top_bullish"]),
        (f"Top {len(tables['top_bearish'])} {BEARISH} signals (short - positive means price fell)", tables["top_bearish"]),
        (f"Most rewarding stocks (at least {MIN_SIGNALS_PER_STOCK} signals with a {P}D result)", _ranked_stocks(tables["stocks"])),
    ]
    for heading, df in blocks:
        _title(ws, row, heading)
        _, last = write_table(ws, df, top=row + 1)
        row = last + 3


def _ranked_stocks(stocks: pd.DataFrame) -> pd.DataFrame:
    eligible = stocks[stocks[f"Signals with {P}D result"] >= MIN_SIGNALS_PER_STOCK]
    return eligible.nlargest(TOP_STOCKS, f"Avg {P}D %").reset_index(drop=True)


# ───────────────────────────────── chart data ─────────────────────────────────
def _pivot(df: pd.DataFrame, index: str, value: str, order: List[str] = None) -> pd.DataFrame:
    by = "Scenario" if "Scenario" in df.columns else "Signal"
    wide = df.pivot(index=index, columns=by, values=value).reindex(columns=[BULLISH, BEARISH])
    if order:
        wide = wide.reindex([o for o in order if o in wide.index])
    wide.columns.name = None
    return wide.reset_index()


def _chart_frames(signals: pd.DataFrame, tables: Dict) -> Dict[str, pd.DataFrame]:
    hz = tables["horizons"]
    order = [f"{h}D" for h in HORIZONS]
    avg = _pivot(hz, "Horizon", "Avg Return %", order).rename(columns={BULLISH: "Gap-Up signal", BEARISH: "Gap-Down signal"})
    base = _pivot(hz, "Horizon", "Baseline Avg %", order)
    avg.insert(2, "Gap-Up baseline", base[BULLISH])
    avg["Gap-Down baseline"] = base[BEARISH]

    top_trades = pd.concat([tables["top_bullish"], tables["top_bearish"]]).nlargest(TOP_STOCKS, RET_COL)
    trades = pd.DataFrame({
        "Trade": top_trades["Symbol"] + " " + top_trades["Date"].dt.strftime("%d-%b-%y")
                 + top_trades["Signal"].map({BULLISH: " ▲", BEARISH: " ▼"}),
        f"{P}D Return %": top_trades[RET_COL],
    }).iloc[::-1]
    stocks = _ranked_stocks(tables["stocks"])[["Symbol", f"Avg {P}D %"]].iloc[::-1]
    months = tables["months"]
    return {
        "avg": avg,
        "win": _pivot(hz, "Horizon", "Win Rate %", order),
        "equity": tables["equity"].assign(Date=tables["equity"]["Date"].dt.strftime("%d-%b-%y")),
        "month_count": _pivot(months, "Month", "Signals").fillna(0),
        "month_avg": _pivot(months, "Month", f"Avg {P}D %"),
        "gap": _pivot(tables["gap"], "Gap Size", f"Avg {P}D %"),
        "gap_day": _pivot(tables["gap"], "Gap Size", "Avg per Signal Day %"),
        "distribution": tables["distribution"],
        "stocks": stocks,
        "trades": trades,
    }


def _chart_data_sheet(wb: Workbook, frames: Dict[str, pd.DataFrame]) -> Tuple[Worksheet, Dict[str, Tuple[int, int, int]]]:
    """Writes every chart's source table; returns {name: (header_row, last_row, n_cols)}."""
    ws = wb.create_sheet("Chart Data")
    ws.cell(row=1, column=1, value="Source tables behind the Dashboard charts.").font = Font(italic=True, color=SLATE)
    positions, row = {}, 3
    for name, df in frames.items():
        _, last = write_table(ws, df, top=row, color_returns=False)
        positions[name] = (row, max(last, row + 1), len(df.columns))
        row = last + 3
    return ws, positions


# ───────────────────────────────── dashboard ──────────────────────────────────
def _style_chart(chart, title: str, y_title: str, colors: List[str]) -> None:
    chart.title = title
    chart.width, chart.height = CHART_W, CHART_H
    chart.legend.position = "b"
    chart.y_axis.title = y_title
    chart.x_axis.delete = False
    chart.y_axis.delete = False
    for series, color in zip(chart.series, colors):
        series.graphicalProperties.solidFill = color
        series.graphicalProperties.line.solidFill = color


def _bar(ws_data, pos, title, y_title, colors, horizontal=False, stacked=False, number_format='0.00') -> BarChart:
    header, last, n_cols = pos
    chart = BarChart()
    chart.type = "bar" if horizontal else "col"
    if stacked:
        chart.grouping, chart.overlap = "stacked", 100
    chart.add_data(Reference(ws_data, min_col=2, max_col=n_cols, min_row=header, max_row=last), titles_from_data=True)
    chart.set_categories(Reference(ws_data, min_col=1, min_row=header + 1, max_row=last))
    _style_chart(chart, title, y_title, colors)
    chart.y_axis.numFmt = number_format
    chart.gapWidth = 60
    if n_cols == 2:
        chart.legend = None
        chart.dataLabels = DataLabelList()
        chart.dataLabels.showVal = True
        chart.dataLabels.showSerName = chart.dataLabels.showCatName = chart.dataLabels.showLegendKey = False
        chart.dataLabels.numFmt = number_format
    return chart


def _line(ws_data, pos, title, y_title, colors) -> LineChart:
    header, last, n_cols = pos
    chart = LineChart()
    chart.add_data(Reference(ws_data, min_col=2, max_col=n_cols, min_row=header, max_row=last), titles_from_data=True)
    chart.set_categories(Reference(ws_data, min_col=1, min_row=header + 1, max_row=last))
    _style_chart(chart, title, y_title, colors)
    for series, color in zip(chart.series, colors):
        series.graphicalProperties.line.width = 22000
        series.smooth = False
    chart.x_axis.tickLblSkip = max((last - header) // 10, 1)
    return chart


def _kpi_card(ws: Worksheet, col: int, label: str, value: str, sub: str, accent: str) -> None:
    """A 4-column wide card across rows 4-7."""
    for row in range(4, 8):
        for c in range(col, col + 4):
            ws.cell(row=row, column=c).fill = PatternFill("solid", fgColor=CARD)
        ws.cell(row=row, column=col).border = Border(left=Side(style="thick", color=accent))
    for row, text, font in [
        (4, label, Font(size=9, color=SLATE, bold=True)),
        (5, value, Font(size=20, color=NAVY, bold=True)),
        (7, sub, Font(size=9, color=SLATE)),
    ]:
        end_row = 6 if row == 5 else row
        ws.merge_cells(start_row=row, start_column=col, end_row=end_row, end_column=col + 3)
        cell = ws.cell(row=row, column=col, value=text)
        cell.font, cell.alignment = font, Alignment(horizontal="center", vertical="center")


def _kpis(signals: pd.DataFrame, tables: Dict, meta: Dict) -> List[Tuple[str, str, str, str]]:
    hz = tables["horizons"].set_index(["Scenario", "Horizon"])
    up, down = hz.loc[(BULLISH, f"{P}D")], hz.loc[(BEARISH, f"{P}D")]
    best = signals.loc[signals[RET_COL].idxmax()]
    return [
        ("SIGNALS FOUND", f"{len(signals):,}", f"{meta['stocks']} stocks, {meta['stock_days']:,} stock-days", BLUE),
        (f"GAP-UP WIN RATE ({P}D)", f"{up['Win Rate %']:.1f}%", f"{int(up['Signals']):,} signals, baseline {up['Baseline Win Rate %']:.1f}%", TEAL),
        (f"GAP-UP AVG RETURN ({P}D)", f"{up['Avg Return %']:+.2f}%", f"edge vs baseline {up['Edge vs Baseline %']:+.2f}%", TEAL),
        (f"GAP-DOWN WIN RATE ({P}D)", f"{down['Win Rate %']:.1f}%", f"{int(down['Signals']):,} signals, baseline {down['Baseline Win Rate %']:.1f}%", CORAL),
        (f"GAP-DOWN AVG RETURN ({P}D)", f"{down['Avg Return %']:+.2f}%", f"short side, edge {down['Edge vs Baseline %']:+.2f}%", CORAL),
        (f"BEST SIGNAL ({P}D)", f"{best[RET_COL]:+.1f}%", f"{best['Symbol']} {best['Date']:%d-%b-%y}", AMBER),
    ]


def _dashboard_sheet(wb: Workbook, ws_data: Worksheet, pos: Dict, signals: pd.DataFrame, tables: Dict, meta: Dict) -> None:
    ws = wb.create_sheet("Dashboard")
    ws.sheet_view.showGridLines = False
    ws.sheet_view.zoomScale = 85
    canvas = PatternFill("solid", fgColor=CANVAS)
    last_row = 9 + CHART_ROWS * 5 + 2
    for col in range(1, DASH_COLS + 3):
        ws.column_dimensions[get_column_letter(col)].width = 2 if col == 1 else DASH_COL_WIDTH
        for row in range(1, last_row):
            ws.cell(row=row, column=col).fill = canvas

    ws.merge_cells(start_row=1, start_column=2, end_row=2, end_column=DASH_COLS + 1)
    banner = ws.cell(row=1, column=2, value=(
        f"  GAP MOMENTUM  |  1-year pattern study  |  {meta['first_date']:%d %b %Y} - {meta['last_date']:%d %b %Y}"))
    banner.font, banner.alignment = Font(size=16, bold=True, color="FFFFFF"), Alignment(vertical="center")
    for col in range(2, DASH_COLS + 2):
        for row in (1, 2):
            ws.cell(row=row, column=col).fill = PatternFill("solid", fgColor=NAVY)
    ws.row_dimensions[5].height = ws.row_dimensions[6].height = 20

    for index, (label, value, sub, accent) in enumerate(_kpis(signals, tables, meta)):
        _kpi_card(ws, 2 + index * 4, label, value, sub, accent)

    pair = [TEAL, CORAL]
    charts = [
        _bar(ws_data, pos["avg"], "Average return after the signal vs all-days baseline", "Avg return %", [TEAL, "9FE3DD", CORAL, "FEC0BF"]),
        _bar(ws_data, pos["win"], "Win rate by holding period", "Win rate %", pair, number_format="0"),
        _line(ws_data, pos["equity"], "Cumulative next-day return (sum of daily average, %)", "Cumulative %", pair),
        _bar(ws_data, pos["distribution"], f"Distribution of {P}-day returns (number of signals)", "Signals", pair, number_format="0"),
        _bar(ws_data, pos["month_count"], "Signals per month", "Signals", pair, stacked=True, number_format="0"),
        _bar(ws_data, pos["month_avg"], f"Average {P}-day return by signal month", "Avg return %", pair),
        _bar(ws_data, pos["gap"], f"By opening gap size: average {P}-day return per signal", "Avg return %", pair),
        _bar(ws_data, pos["gap_day"], f"By opening gap size: average {P}-day return per trading day", "Avg return %", pair),
        _bar(ws_data, pos["stocks"], f"Top stocks by average {P}-day return (min {MIN_SIGNALS_PER_STOCK} signals)", "Avg return %", [BLUE], horizontal=True),
        _bar(ws_data, pos["trades"], f"Top individual signals by {P}-day return (▲ long, ▼ short)", "Return %", [AMBER], horizontal=True),
    ]
    for index, chart in enumerate(charts):
        row = 9 + (index // 2) * CHART_ROWS
        ws.add_chart(chart, f"{get_column_letter(2 + (index % 2) * 12)}{row}")


# ──────────────────────────────── entry point ─────────────────────────────────
def write_workbook(out_path: Path, signals: pd.DataFrame, tables: Dict, meta: Dict) -> None:
    wb = Workbook()
    _overview_sheet(wb, signals, tables, meta)

    signal_columns = ["Symbol", "Date", "Signal", "T-1 Close", "Open", "High", "Low", "Close", "Volume",
                      "Gap %", "Day Chg %", "Volume x T-1", "Spread x T-1", "Gap Size", "Volume Bucket"]
    signal_columns += [f"Return {h}D %" for h in HORIZONS] + [f"Best in {EXCURSION_WINDOW}D %", f"Worst in {EXCURSION_WINDOW}D %"]
    _data_sheet(wb, "All Signals", signals[signal_columns],
                "Every signal in the year. Returns are direction-adjusted: gap-down rows are shorts. Blank = result not available yet.")
    _outstanding_sheet(wb, tables)

    ws_data, positions = _chart_data_sheet(wb, _chart_frames(signals, tables))
    _dashboard_sheet(wb, ws_data, positions, signals, tables, meta)

    _data_sheet(wb, "By Stock", tables["stocks"], f"Per-stock scorecard. Return columns use the {P}-day horizon.")
    _data_sheet(wb, "By Month", tables["months"], f"Signals grouped by the month they fired. Return columns use the {P}-day horizon.")
    buckets = wb.create_sheet("Buckets")
    row = 1
    for heading, key in [("By opening gap size", "gap"), ("By volume expansion vs T-1", "volume"), ("By spread expansion vs T-1", "spread")]:
        _title(buckets, row, f"{heading} ({P}-day return)")
        _, last = write_table(buckets, tables[key], top=row + 1)
        row = last + 3

    if len(meta["actions"]):
        _data_sheet(wb, "Corporate Actions", meta["actions"].sort_values("Ex-Date"),
                    "Overnight opens beyond +/-20% were treated as corporate actions and earlier prices were back-adjusted.")
    else:
        wb.create_sheet("Corporate Actions")["A1"] = "No corporate actions detected."
    order = ["Overview", "All Signals", "Outstanding Returns", "Dashboard", "By Stock", "By Month", "Buckets", "Corporate Actions", "Chart Data"]
    wb._sheets = [wb[name] for name in order]
    wb.save(out_path)
