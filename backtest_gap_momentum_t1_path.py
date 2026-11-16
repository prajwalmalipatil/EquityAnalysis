"""
backtest_gap_momentum_t1_path.py
Ten-session path studies for setups that start with a Gap-Up Continuation signal on T
followed by a T+1 session that opened above the close of T.

  HIGHER_OPEN        entry at the T+1 open, path T+1 ... T+10
  HIGHER_OPEN_RED    T+1 also closed red (close below its open); that is only known at
                     the T+1 close, so entry is the T+1 close and the path is T+2 ... T+11

Each setup is compared with every stock-day that met the same T+1 conditions without
needing a signal (the baseline). Only events with all ten sessions available are used,
so every day of the path is measured on the same set of trades.

Each setup adds two sheets to the T+1 study workbook: a tear sheet and the per-trade paths.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from backtest_gap_momentum import BULLISH, GAP_BINS, GAP_LABELS, HIST_EDGES, HIST_LABELS
from backtest_gap_momentum_report import AMBER, BLUE, CANVAS, CHART_ROWS, CORAL, GREY, TEAL, _bar, _data_sheet, _kpi_card, _line
from backtest_gap_momentum_t1 import SIGNIFICANT_T, TOP_TRADES, _t_stat

PATH_DAYS = 10
SESSIONS = list(range(1, PATH_DAYS + 1))
THRESHOLDS = [2, 3, 5]            # % target / stop levels for the first-touch test
CHECKPOINTS = [1, 3, 5, 10]       # path sessions shown in the grouped tables
C, H, L = ([f"{p}{k}" for k in SESSIONS] for p in "CHL")   # close / high / low vs entry, in %


@dataclass(frozen=True)
class PathSetup:
    key: str
    sheet: str
    trades_sheet: str
    setup_label: str
    base_label: str
    banner: str
    entry_column: str        # T+1 price used as entry: "Open" or "Close"
    first_day: int           # first session of the path, as an offset from T
    red_t1: bool             # require T+1 to close below its open

    @property
    def entry(self) -> str:
        return f"T+1 {self.entry_column.lower()}"

    @property
    def days(self) -> List[str]:
        return [f"T+{self.first_day + i}" for i in range(PATH_DAYS)]


HIGHER_OPEN = PathSetup(
    key="higher_open", sheet="Gap-Up T+1 Higher 10D", trades_sheet="Gap-Up T+1 Higher Paths",
    setup_label="Gap-Up signal, T+1 opened higher", base_label="Baseline: any day, T+1 opened higher",
    banner="GAP-UP SIGNAL ON T, THEN T+1 OPENED HIGHER  |  10-SESSION PATH FROM THE T+1 OPEN",
    entry_column="Open", first_day=1, red_t1=False,
)
HIGHER_OPEN_RED = PathSetup(
    key="higher_open_red", sheet="Gap-Up T+1 High Red 10D", trades_sheet="Gap-Up T+1 High Red Paths",
    setup_label="Gap-Up signal, T+1 opened higher, closed red",
    base_label="Baseline: any day, T+1 opened higher, closed red",
    banner="GAP-UP SIGNAL ON T, T+1 OPENED HIGHER BUT CLOSED RED  |  NEXT 10 SESSIONS FROM THE T+1 CLOSE",
    entry_column="Close", first_day=2, red_t1=True,
)
SETUPS = [HIGHER_OPEN, HIGHER_OPEN_RED]


# ────────────────────────────── data preparation ──────────────────────────────
def build_paths(all_bars: pd.DataFrame, setup: PathSetup) -> pd.DataFrame:
    """One row per stock-day meeting the T+1 conditions, with the ten-session path vs the entry price."""
    bars = all_bars.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    by_stock = bars.groupby("Symbol", sort=False)
    t1_open, t1_close = by_stock["Open"].shift(-1), by_stock["Close"].shift(-1)
    entry = t1_open if setup.entry_column == "Open" else t1_close

    paths = bars[["Symbol", "Date", "Signal", "Close", "Gap %"]].copy()
    paths["T+1 Date"], paths["T+1 Open"], paths["T+1 Close"] = by_stock["Date"].shift(-1), t1_open, t1_close
    paths["T+1 Gap %"] = (t1_open / bars["Close"] - 1) * 100
    paths["T+1 Open to Close %"] = (t1_close / t1_open - 1) * 100
    paths["T Close vs Entry %"] = (bars["Close"] / entry - 1) * 100
    for i, session in enumerate(SESSIONS):
        for prefix, column in (("C", "Close"), ("H", "High"), ("L", "Low")):
            paths[f"{prefix}{session}"] = (by_stock[column].shift(-(setup.first_day + i)) / entry - 1) * 100

    keep = (t1_open > bars["Close"]) & paths[C[-1]].notna()
    if setup.red_t1:
        keep &= t1_close < t1_open
    paths = paths[keep].reset_index(drop=True)

    highs, lows = paths[H].to_numpy(), paths[L].to_numpy()
    paths["Max Gain %"], paths["Max Drawdown %"] = highs.max(axis=1), lows.min(axis=1)
    paths["Peak Session"], paths["Trough Session"] = highs.argmax(axis=1) + 1, lows.argmin(axis=1) + 1
    paths["T+1 Gap Size"] = pd.cut(paths["T+1 Gap %"], GAP_BINS, labels=GAP_LABELS, include_lowest=True)
    paths["Month"] = paths["T+1 Date"].dt.strftime("%Y-%m")
    return paths


def _session_returns(frame: pd.DataFrame) -> np.ndarray:
    """Return of each session alone: the first is from the entry, later ones are close-to-close."""
    growth = 1 + frame[C].to_numpy() / 100
    previous = np.hstack([np.ones((len(frame), 1)), growth[:, :-1]])
    return (growth / previous - 1) * 100


# ───────────────────────────────── statistics ─────────────────────────────────
def return_table(trades: pd.DataFrame, base: pd.DataFrame, setup: PathSetup) -> pd.DataFrame:
    """Cumulative and per-session return for each day of the path."""
    session = _session_returns(trades)
    by_day, base_by_day = trades.groupby("T+1 Date")[C].mean(), base.groupby("T+1 Date")[C].mean()
    excess_by_day = by_day - base_by_day.reindex(by_day.index)
    rows = []
    for i, column in enumerate(C):
        cum = trades[column]
        rows.append({
            "Day": setup.days[i], "Trades": len(cum),
            "Mean Cum Return %": cum.mean(), "Median Cum Return %": cum.median(), "Std Dev": cum.std(),
            "Win Rate %": (cum > 0).mean() * 100,
            "t-stat (per trade)": _t_stat(cum), "t-stat (per day)": _t_stat(by_day[column]),
            "Baseline Mean Cum %": base[column].mean(),
            "Excess vs Baseline %": cum.mean() - base[column].mean(),
            "Excess t-stat (per day)": _t_stat(excess_by_day[column]),
            "25th Pctl %": cum.quantile(0.25), "75th Pctl %": cum.quantile(0.75),
            "Mean Session Return %": session[:, i].mean(), "Session Win Rate %": (session[:, i] > 0).mean() * 100,
        })
    return pd.DataFrame(rows)


def behaviour_table(trades: pd.DataFrame, base: pd.DataFrame, setup: PathSetup) -> pd.DataFrame:
    """How the path traded relative to the entry and to the close of T."""
    rows = []
    for i in range(PATH_DAYS):
        row = {"Day": setup.days[i]}
        for label, frame in (("Setup", trades), ("Baseline", base)):
            t_close = frame["T Close vs Entry %"].to_numpy()
            row[f"{label}: Closed Above T Close %"] = (frame[C[i]].to_numpy() > t_close).mean() * 100
            row[f"{label}: T Close Breached So Far %"] = (frame[L[: i + 1]].to_numpy().min(axis=1) <= t_close).mean() * 100
            row[f"{label}: Avg Best Gain So Far %"] = frame[H[: i + 1]].to_numpy().max(axis=1).mean()
            row[f"{label}: Avg Worst Loss So Far %"] = frame[L[: i + 1]].to_numpy().min(axis=1).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def extremes_table(trades: pd.DataFrame, base: pd.DataFrame, setup: PathSetup) -> pd.DataFrame:
    """On which day the ten-session high and low were made (share of paths)."""
    table = pd.DataFrame({"Day": setup.days})
    for label, frame in ((setup.setup_label, trades), (setup.base_label, base)):
        for name, column in (("High", "Peak Session"), ("Low", "Trough Session")):
            share = frame[column].value_counts(normalize=True).reindex(SESSIONS, fill_value=0) * 100
            table[f"{name} Made: {label}"] = share.values
    return table


def first_touch_table(trades: pd.DataFrame, base: pd.DataFrame, setup: PathSetup) -> pd.DataFrame:
    """Which was hit first within ten sessions: +X% (target) or -X% (stop). Same-day ties count as stop."""
    rows = []
    for level in THRESHOLDS:
        for label, frame in ((setup.setup_label, trades), (setup.base_label, base)):
            up, down = frame[H].to_numpy() >= level, frame[L].to_numpy() <= -level
            first_up = np.where(up.any(axis=1), up.argmax(axis=1), PATH_DAYS)
            first_down = np.where(down.any(axis=1), down.argmax(axis=1), PATH_DAYS)
            target, stop = first_up < first_down, (first_down <= first_up) & (first_down < PATH_DAYS)
            rows.append({
                "Level": f"+/- {level}%", "Cohort": label, "Paths": len(frame),
                "Target First %": target.mean() * 100, "Stop First %": stop.mean() * 100,
                "Neither %": (~target & ~stop).mean() * 100,
                "Expectancy at Level %": (target.mean() - stop.mean()) * level,
            })
    return pd.DataFrame(rows)


def summary_table(trades: pd.DataFrame, base: pd.DataFrame, setup: PathSetup) -> pd.DataFrame:
    last = setup.days[-1]
    rows = []
    for label, frame in ((setup.setup_label, trades), (setup.base_label, base)):
        final = frame[C[-1]]
        rows.append({
            "Cohort": label, "Paths": len(frame), "Trade Days": frame["T+1 Date"].nunique(),
            "Avg T+1 Gap %": frame["T+1 Gap %"].mean(), "Avg T+1 Open to Close %": frame["T+1 Open to Close %"].mean(),
            f"Mean {last} Return %": final.mean(), f"Median {last} Return %": final.median(),
            "Win Rate %": (final > 0).mean() * 100,
            "Avg Max Gain %": frame["Max Gain %"].mean(), "Avg Max Drawdown %": frame["Max Drawdown %"].mean(),
            "Avg Peak Session (1-10)": frame["Peak Session"].mean(), "Avg Trough Session (1-10)": frame["Trough Session"].mean(),
            "Best %": final.max(), "Worst %": final.min(),
        })
    return pd.DataFrame(rows)


def grouped_table(trades: pd.DataFrame, key: str, setup: PathSetup) -> pd.DataFrame:
    grouped = trades.groupby(key, observed=True)
    table = grouped[C[-1]].agg(Trades="count", **{"Win Rate %": lambda r: (r > 0).mean() * 100})
    for k in CHECKPOINTS:
        table[f"Mean Cum {setup.days[k - 1]} %"] = grouped[f"C{k}"].mean()
    table["Avg Max Gain %"], table["Avg Max Drawdown %"] = grouped["Max Gain %"].mean(), grouped["Max Drawdown %"].mean()
    return table.reset_index()


def distribution_table(trades: pd.DataFrame, base: pd.DataFrame, setup: PathSetup) -> pd.DataFrame:
    """Share of paths by ten-session return band."""
    table = pd.DataFrame({f"{setup.days[-1]} Return Band (%)": HIST_LABELS})
    for label, frame in ((setup.setup_label, trades), (setup.base_label, base)):
        share = pd.cut(frame[C[-1]], HIST_EDGES, labels=HIST_LABELS).value_counts(normalize=True).reindex(HIST_LABELS) * 100
        table[label] = share.values
    return table


def trade_paths(trades: pd.DataFrame, setup: PathSetup) -> pd.DataFrame:
    out = trades[["Symbol", "Date", "Close", "Gap %", "T+1 Date", "T+1 Open", "T+1 Close", "T+1 Gap %",
                  "T+1 Open to Close %", "T+1 Gap Size"] + C + ["Max Gain %", "Max Drawdown %"]].copy()
    out["Peak Day"] = trades["Peak Session"].map(dict(zip(SESSIONS, setup.days)))
    out["Trough Day"] = trades["Trough Session"].map(dict(zip(SESSIONS, setup.days)))
    names = {"Date": "T Date", "Close": "T Close", "Gap %": "T Gap %", **{c: f"Cum {d} %" for c, d in zip(C, setup.days)}}
    return out.rename(columns=names).sort_values(["T Date", "Symbol"]).reset_index(drop=True)


def path_tables(all_bars: pd.DataFrame, setup: PathSetup) -> Dict[str, pd.DataFrame]:
    base = build_paths(all_bars, setup)
    trades = base[base["Signal"] == BULLISH]
    listing = trade_paths(trades, setup)
    final = f"Cum {setup.days[-1]} %"
    return {
        "summary": summary_table(trades, base, setup), "returns": return_table(trades, base, setup),
        "behaviour": behaviour_table(trades, base, setup), "extremes": extremes_table(trades, base, setup),
        "first_touch": first_touch_table(trades, base, setup), "distribution": distribution_table(trades, base, setup),
        "gap": grouped_table(trades, "T+1 Gap Size", setup), "month": grouped_table(trades, "Month", setup),
        "best": listing.nlargest(TOP_TRADES, final).reset_index(drop=True),
        "worst": listing.nsmallest(TOP_TRADES, final).reset_index(drop=True),
        "trades": listing,
    }


# ─────────────────────────────────── sheets ───────────────────────────────────
def _chart_frames(tables: Dict[str, pd.DataFrame], setup: PathSetup) -> Dict[str, pd.DataFrame]:
    returns, behaviour, touch = tables["returns"], tables["behaviour"], tables["first_touch"]
    rename = lambda cols: {c: c.replace("Setup: ", "Setup ").replace("Baseline: ", "Baseline ") for c in cols}
    excursion = [c for c in behaviour.columns if "So Far %" in c and "Avg" in c]
    support = [c for c in behaviour.columns if "T Close" in c and c.startswith("Setup")]
    setup_touch = touch[touch["Cohort"] == setup.setup_label]
    return {
        "path": returns[["Day", "Mean Cum Return %", "Baseline Mean Cum %", "Median Cum Return %"]].rename(
            columns={"Mean Cum Return %": "Setup mean", "Baseline Mean Cum %": "Baseline mean", "Median Cum Return %": "Setup median"}),
        "win": returns[["Day", "Win Rate %", "Session Win Rate %"]].rename(
            columns={"Win Rate %": "Cumulative return above zero", "Session Win Rate %": "Session closed up"}),
        "session": returns[["Day", "Mean Session Return %"]],
        "excursion": behaviour[["Day"] + excursion].rename(columns=rename(excursion)),
        "extremes": tables["extremes"].iloc[:, :3].rename(columns=lambda c: c.split(":")[0]),
        "support": behaviour[["Day"] + support].rename(columns=rename(support)),
        "distribution": tables["distribution"],
        "touch": setup_touch[["Level", "Target First %", "Stop First %", "Neither %"]],
    }


def _kpis(tables: Dict[str, pd.DataFrame], setup: PathSetup) -> List[Tuple[str, str, str, str]]:
    trades, base = tables["summary"].iloc[0], tables["summary"].iloc[1]
    last_row, last = tables["returns"].iloc[-1], setup.days[-1]
    final = f"Mean {last} Return %"
    peak, trough = (setup.days[int(round(trades[c])) - 1] for c in ("Avg Peak Session (1-10)", "Avg Trough Session (1-10)"))
    return [
        ("PATHS STUDIED", f"{int(trades['Paths']):,}", f"across {int(trades['Trade Days'])} trading days", BLUE),
        (f"MEAN RETURN AT {last}", f"{trades[final]:+.2f}%", f"baseline {base[final]:+.2f}%", TEAL),
        (f"WIN RATE AT {last}", f"{trades['Win Rate %']:.1f}%", f"baseline {base['Win Rate %']:.1f}%", TEAL),
        ("EXCESS t-STAT (PER DAY)", f"{last_row['Excess t-stat (per day)']:+.2f}", f"significant at |t| >= {SIGNIFICANT_T:.0f}", AMBER),
        ("AVG MAX GAIN IN 10 SESSIONS", f"{trades['Avg Max Gain %']:+.2f}%", f"peak around {peak} on average", TEAL),
        ("AVG MAX DRAWDOWN IN 10 SESSIONS", f"{trades['Avg Max Drawdown %']:+.2f}%", f"trough around {trough} on average", CORAL),
    ]


def add_path_sheets(wb: Workbook, tables: Dict[str, pd.DataFrame], setup: PathSetup,
                    ws_data: Worksheet, data_row: int, meta: Dict) -> Tuple[List[str], int]:
    """Adds a setup's tear sheet and per-trade sheet; returns (sheet names, next free chart-data row)."""
    from backtest_gap_momentum_t1_report import BANNER_COLS, SECOND_CHART_COL, _banner, _section, _table, _write_chart_data

    pos, next_row = _write_chart_data(ws_data, data_row, setup.sheet, _chart_frames(tables, setup))
    ws = wb.create_sheet(setup.sheet)
    ws.sheet_view.showGridLines = False
    ws.sheet_view.zoomScale = 85
    _banner(ws, f"{setup.banner}  |  {meta['first_date']:%d %b %Y} - {meta['last_date']:%d %b %Y}")
    for col in range(1, BANNER_COLS + 1):
        for row in range(3, 9):
            ws.cell(row=row, column=col).fill = PatternFill("solid", fgColor=CANVAS)
    for index, card in enumerate(_kpis(tables, setup)):
        _kpi_card(ws, 1 + index * 4, *card)

    first, last = setup.days[0], setup.days[-1]
    row = _table(ws, 10, "1. Setup against baseline", tables["summary"],
                 f"Entry at the {setup.entry}, tracked over {first} to {last}. The baseline meets the same T+1 conditions without a signal. Complete 10-session paths only.")
    row = _table(ws, row, "2. Return by day of the path", tables["returns"],
                 f"Cumulative return is measured from the {setup.entry} to that day's close. Session return is that day alone.")
    row = _table(ws, row, "3. Path behaviour by day", tables["behaviour"],
                 "T close = the signal day's close, which the T+1 open gapped above. Best gain / worst loss use intraday highs and lows up to that day.", color=False)
    row = _table(ws, row, "4. Which is hit first: target or stop?", tables["first_touch"],
                 "Within the 10 sessions, using daily highs and lows. A day that touches both counts as the stop. Expectancy = (target share - stop share) x level.", color=False)
    row = _table(ws, row, "5. Day on which the 10-session high and low were made (% of paths)", tables["extremes"], color=False)

    row = _section(ws, row, "6. Charts")
    charts = [
        _line(ws_data, pos["path"], f"Average path: cumulative return from the {setup.entry}", "Cumulative %", [TEAL, GREY, AMBER]),
        _bar(ws_data, pos["win"], "Win rate by day", "Win rate %", [TEAL, BLUE], number_format="0"),
        _bar(ws_data, pos["session"], "Mean return of each session", "Mean return %", [TEAL]),
        _line(ws_data, pos["excursion"], "Average best gain and worst loss so far", "% from entry", [TEAL, CORAL, "9FE3DD", "FEC0BF"]),
        _bar(ws_data, pos["extremes"], "Day the 10-session high and low were made (setup, % of paths)", "% of paths", [TEAL, CORAL], number_format="0"),
        _line(ws_data, pos["support"], "T close as support: closes above it vs breached so far (setup)", "% of paths", [TEAL, CORAL]),
        _bar(ws_data, pos["distribution"], f"Distribution of {last} returns (% of paths)", "% of paths", [TEAL, GREY], number_format="0"),
        _bar(ws_data, pos["touch"], "Target or stop first within 10 sessions (setup)", "% of paths", [TEAL, CORAL, GREY], number_format="0"),
    ]
    for index, chart in enumerate(charts):
        anchor_col = 1 if index % 2 == 0 else SECOND_CHART_COL
        ws.add_chart(chart, f"{get_column_letter(anchor_col)}{row + (index // 2) * CHART_ROWS}")
    row += CHART_ROWS * ((len(charts) + 1) // 2) + 1

    row = _table(ws, row, "7. By size of the T+1 opening gap", tables["gap"])
    row = _table(ws, row, "8. Stability by month of entry", tables["month"])
    row = _table(ws, row, f"9. Best {TOP_TRADES} paths by {last} return", tables["best"])
    _table(ws, row, f"10. Worst {TOP_TRADES} paths by {last} return", tables["worst"])
    ws.freeze_panes = "B3"

    _data_sheet(wb, setup.trades_sheet, tables["trades"],
                f"Every trade in the setup: {setup.setup_label}. Cum columns: return from the {setup.entry} to that day's close.")
    return [setup.sheet, setup.trades_sheet], next_row
