"""
backtest_gap_momentum_t1.py
Event study of the session AFTER a Gap Momentum signal.

T is the signal day. The study splits every signal by how T+1 opened against the
close of T, and measures what the price did from that open onwards:

  T+1 Opened Higher  -> T+1 open > T close
  T+1 Opened Lower   -> T+1 open < T close

The T+1 open is the first price at which the split is known, so every return is
measured from the T+1 open (no look-ahead). Returns are raw price moves: a positive
number means the price rose, so a reliably negative mean is a short-side edge.
The benchmark is every stock-day in the universe whose next session opened the
same way, whether or not a signal fired.

Usage:
  python backtest_gap_momentum_t1.py --data-dir equity_data --symbols @symbols.txt \
      --out reports/GapMomentum_T1_Open_Study.xlsx
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from backtest_gap_momentum import (
    BEARISH, BULLISH, GAP_BINS, GAP_LABELS, _read_symbols, enrich_bars, load_universe,
)

HIGHER, LOWER = "T+1 Opened Higher", "T+1 Opened Lower"
PATTERNS = [HIGHER, LOWER]
ALL_SIGNALS, BASELINE = "All Gap Momentum signals", "Baseline: all stock-days"
COHORTS = [ALL_SIGNALS, BULLISH, BEARISH]

# (label shown in the report, column name, sessions after T at which the trade is closed)
HOLDS = [
    ("T+1 Open to T+1 Close", "Ret T+1 Close %", 1),
    ("T+1 Open to T+2 Close", "Ret T+2 Close %", 2),
    ("T+1 Open to T+3 Close", "Ret T+3 Close %", 3),
    ("T+1 Open to T+5 Close", "Ret T+5 Close %", 5),
    ("T+1 Open to T+10 Close", "Ret T+10 Close %", 10),
]
INTRADAY = HOLDS[0][1]
COST_ROUND_TRIP_PCT = 0.10      # assumed brokerage + taxes + slippage for one round trip
SIGNIFICANT_T = 2.0             # |t| at or above this is treated as statistically significant
TOP_TRADES = 15


# ────────────────────────────── data preparation ──────────────────────────────
def add_next_session(all_bars: pd.DataFrame) -> pd.DataFrame:
    """Attaches the T+1 session and the forward closes to every stock-day."""
    bars = all_bars.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    by_stock = bars.groupby("Symbol", sort=False)
    nxt = {c: by_stock[c].shift(-1) for c in ["Date", "Open", "High", "Low", "Close"]}

    bars["T+1 Date"] = nxt["Date"]
    bars["T+1 Open"], bars["T+1 High"], bars["T+1 Low"], bars["T+1 Close"] = (
        nxt["Open"], nxt["High"], nxt["Low"], nxt["Close"])
    bars["T+1 Gap %"] = (nxt["Open"] / bars["Close"] - 1) * 100
    bars["Pattern"] = np.select([nxt["Open"] > bars["Close"], nxt["Open"] < bars["Close"]], PATTERNS, default="")

    for _, column, sessions in HOLDS:
        bars[column] = (by_stock["Close"].shift(-sessions) / nxt["Open"] - 1) * 100
    bars["Open to High %"] = (nxt["High"] / nxt["Open"] - 1) * 100
    bars["Open to Low %"] = (nxt["Low"] / nxt["Open"] - 1) * 100
    bars["T+1 Close vs T Close %"] = (nxt["Close"] / bars["Close"] - 1) * 100

    opened_higher = bars["Pattern"] == HIGHER
    # Gap fill: price traded back to the close of T at any point during T+1.
    bars["Gap Filled"] = np.where(opened_higher, nxt["Low"] <= bars["Close"], nxt["High"] >= bars["Close"])
    # Gap held: T+1 closed on the same side of the T close as it opened.
    bars["Gap Held"] = np.where(opened_higher, nxt["Close"] > bars["Close"], nxt["Close"] < bars["Close"])
    bars["Closed Green"] = nxt["Close"] > nxt["Open"]
    # Range break: T+1 closed beyond the extreme of T in the direction of the open.
    bars["Broke T Range"] = np.where(opened_higher, nxt["Close"] > bars["High"], nxt["Close"] < bars["Low"])
    bars["T+1 Gap Size"] = pd.cut(bars["T+1 Gap %"].abs(), GAP_BINS, labels=GAP_LABELS, include_lowest=True)
    bars["Month"] = bars["T+1 Date"].dt.strftime("%Y-%m")
    return bars[bars["T+1 Open"].notna() & (bars["Pattern"] != "")].reset_index(drop=True)


def cohort_frame(events: pd.DataFrame, pattern: str, cohort: str) -> pd.DataFrame:
    subset = events[events["Pattern"] == pattern]
    if cohort == BASELINE:
        return subset
    if cohort == ALL_SIGNALS:
        return subset[subset["Signal"] != ""]
    return subset[subset["Signal"] == cohort]


# ───────────────────────────────── statistics ─────────────────────────────────
def _t_stat(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) < 3 or values.std() == 0:
        return np.nan
    return values.mean() / (values.std() / np.sqrt(len(values)))


def performance_row(frame: pd.DataFrame, baseline: pd.DataFrame, column: str) -> Dict:
    """Return statistics for one cohort and one holding period, against the baseline."""
    returns = frame[column].dropna()
    if returns.empty:
        return {}
    by_day = frame.dropna(subset=[column]).groupby("T+1 Date")[column].mean()
    base_by_day = baseline.dropna(subset=[column]).groupby("T+1 Date")[column].mean()
    wins, losses = returns[returns > 0], returns[returns < 0]
    mean, base_mean = returns.mean(), baseline[column].mean()
    excess = mean - base_mean
    return {
        "Trades": len(returns), "Trade Days": len(by_day),
        "Mean %": mean, "Median %": returns.median(), "Std Dev": returns.std(),
        "Win Rate %": (returns > 0).mean() * 100,
        "t-stat (per trade)": _t_stat(returns),
        # Signals cluster on market-wide days, so each trading day is also scored once.
        "Mean per Day %": by_day.mean(), "t-stat (per day)": _t_stat(by_day),
        "Baseline Mean %": base_mean, "Excess vs Baseline %": excess,
        "Excess t-stat (per day)": _t_stat(by_day - base_by_day.reindex(by_day.index)),
        "Avg Win %": wins.mean(), "Avg Loss %": losses.mean(),
        "Payoff Ratio": wins.mean() / abs(losses.mean()) if len(losses) and len(wins) else np.nan,
        "Profit Factor": wins.sum() / abs(losses.sum()) if len(losses) else np.nan,
        "5th Pctl %": returns.quantile(0.05), "25th Pctl %": returns.quantile(0.25),
        "75th Pctl %": returns.quantile(0.75), "95th Pctl %": returns.quantile(0.95),
        "Best %": returns.max(), "Worst %": returns.min(),
        "Implied Side": "Long" if excess > 0 else "Short",
        "Excess Net of Cost %": abs(excess) - COST_ROUND_TRIP_PCT,
    }


def performance_table(events: pd.DataFrame, pattern: str) -> pd.DataFrame:
    baseline = cohort_frame(events, pattern, BASELINE)
    rows = []
    for cohort in COHORTS:
        frame = cohort_frame(events, pattern, cohort)
        for label, column, _ in HOLDS:
            stats = performance_row(frame, baseline, column)
            if stats:
                rows.append({"Cohort": cohort, "Holding Period": label, **stats})
    return pd.DataFrame(rows)


def behaviour_table(events: pd.DataFrame, pattern: str) -> pd.DataFrame:
    """What the T+1 session itself looked like."""
    break_label = "Closed Above T High %" if pattern == HIGHER else "Closed Below T Low %"
    rows = []
    for cohort in COHORTS + [BASELINE]:
        frame = cohort_frame(events, pattern, cohort)
        if frame.empty:
            continue
        rows.append({
            "Cohort": cohort, "Events": len(frame),
            "Avg T+1 Gap %": frame["T+1 Gap %"].mean(), "Median T+1 Gap %": frame["T+1 Gap %"].median(),
            "Gap Filled Intraday %": frame["Gap Filled"].mean() * 100,
            "Gap Held at Close %": frame["Gap Held"].mean() * 100,
            "Closed Green %": frame["Closed Green"].mean() * 100,
            break_label: frame["Broke T Range"].mean() * 100,
            "Avg Open to High %": frame["Open to High %"].mean(),
            "Avg Open to Low %": frame["Open to Low %"].mean(),
            "Avg Open to Close %": frame[INTRADAY].mean(),
            "Avg T+1 Close vs T Close %": frame["T+1 Close vs T Close %"].mean(),
        })
    return pd.DataFrame(rows)


def grouped_table(events: pd.DataFrame, pattern: str, key: str) -> pd.DataFrame:
    """Intraday result per cohort, split by a grouping column (gap size or month)."""
    frames = []
    for cohort in COHORTS + [BASELINE]:
        frame = cohort_frame(events, pattern, cohort).dropna(subset=[INTRADAY])
        if frame.empty:
            continue
        grouped = frame.groupby(key, observed=True)
        table = grouped[INTRADAY].agg(**{
            "Trades": "count", "Mean Open to Close %": "mean", "Median Open to Close %": "median",
            "Win Rate %": lambda r: (r > 0).mean() * 100, "t-stat (per trade)": _t_stat,
        })
        table["Gap Filled Intraday %"] = grouped["Gap Filled"].mean() * 100
        table[f"Mean {HOLDS[3][0]} %"] = grouped[HOLDS[3][1]].mean()
        table.insert(0, "Cohort", cohort)
        frames.append(table.reset_index())
    return pd.concat(frames, ignore_index=True)[lambda t: ["Cohort", key] + [c for c in t.columns if c not in ("Cohort", key)]]


def equity_curve(events: pd.DataFrame, pattern: str) -> pd.DataFrame:
    """Cumulative open-to-close return, equal-weighting every event on the same day."""
    curves = {}
    for cohort in [BULLISH, BEARISH, BASELINE]:
        frame = cohort_frame(events, pattern, cohort).dropna(subset=[INTRADAY])
        curves[cohort] = frame.groupby("T+1 Date")[INTRADAY].mean()
    return pd.DataFrame(curves).fillna(0.0).cumsum().rename_axis("T+1 Date").reset_index()


def extreme_trades(events: pd.DataFrame, pattern: str, best: bool) -> pd.DataFrame:
    columns = ["Symbol", "Date", "Signal", "Close", "T+1 Date", "T+1 Open", "T+1 High", "T+1 Low", "T+1 Close",
               "T+1 Gap %", "Open to High %", "Open to Low %"] + [c for _, c, _ in HOLDS]
    frame = cohort_frame(events, pattern, ALL_SIGNALS).dropna(subset=[INTRADAY])
    picked = frame.nlargest(TOP_TRADES, INTRADAY) if best else frame.nsmallest(TOP_TRADES, INTRADAY)
    return picked[columns].rename(columns={"Date": "T Date", "Close": "T Close"}).reset_index(drop=True)


def frequency_table(events: pd.DataFrame) -> pd.DataFrame:
    """How often T+1 opens higher or lower after each signal, against all stock-days."""
    rows = []
    for cohort in COHORTS + [BASELINE]:
        counts = {p: len(cohort_frame(events, p, cohort)) for p in PATTERNS}
        total = sum(counts.values())
        rows.append({
            "Cohort": cohort, "Events": total,
            "Opened Higher": counts[HIGHER], "Opened Higher %": counts[HIGHER] / total * 100,
            "Opened Lower": counts[LOWER], "Opened Lower %": counts[LOWER] / total * 100,
        })
    return pd.DataFrame(rows)


def verdict(row: pd.Series) -> str:
    t = row["Excess t-stat (per day)"]
    if pd.isna(t) or abs(t) < SIGNIFICANT_T:
        return "Not significant - indistinguishable from baseline"
    if row["Excess Net of Cost %"] <= 0:
        return f"Significant {row['Implied Side'].lower()} bias, but smaller than trading costs"
    return f"Significant {row['Implied Side'].lower()} edge after costs"


def scorecard(tables: Dict[str, Dict]) -> pd.DataFrame:
    """One line per pattern x signal x holding period, with a plain-language verdict."""
    keep = ["Trades", "Trade Days", "Mean %", "Median %", "Win Rate %", "Baseline Mean %", "Excess vs Baseline %",
            "Excess t-stat (per day)", "Implied Side", "Excess Net of Cost %"]
    frames = []
    for pattern in PATTERNS:
        perf = tables[pattern]["performance"]
        perf = perf[perf["Holding Period"].isin([HOLDS[0][0], HOLDS[3][0]])]
        part = perf[["Cohort", "Holding Period"] + keep].copy()
        part.insert(0, "T+1 Pattern", pattern)
        part["Verdict"] = perf.apply(verdict, axis=1).values
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────── main ─────────────────────────────────────
def run(data_dir: Path, symbols: List[str], out_path: Path) -> None:
    from backtest_gap_momentum_t1_path import SETUPS, path_tables
    from backtest_gap_momentum_t1_report import write_workbook  # presentation kept separate

    universe, _ = load_universe(data_dir, symbols)
    if not universe:
        raise SystemExit(f"No usable stock CSVs found in {data_dir}")
    all_bars = pd.concat([enrich_bars(s, df) for s, df in sorted(universe.items())], ignore_index=True)
    events = add_next_session(all_bars)

    tables: Dict[str, Dict] = {}
    for pattern in PATTERNS:
        tables[pattern] = {
            "performance": performance_table(events, pattern),
            "behaviour": behaviour_table(events, pattern),
            "gap": grouped_table(events, pattern, "T+1 Gap Size"),
            "month": grouped_table(events, pattern, "Month"),
            "equity": equity_curve(events, pattern),
            "best": extreme_trades(events, pattern, best=True),
            "worst": extreme_trades(events, pattern, best=False),
        }
    signals = events[events["Signal"] != ""]
    meta = {
        "stocks": len(universe), "stock_days": len(all_bars), "signals": int((all_bars["Signal"] != "").sum()),
        "signals_with_t1": len(signals), "first_date": all_bars["Date"].min(), "last_date": all_bars["Date"].max(),
        "frequency": frequency_table(events), "scorecard": scorecard(tables),
        "paths": {setup.key: path_tables(all_bars, setup) for setup in SETUPS},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_workbook(out_path, signals, tables, meta)

    pd.set_option("display.width", 250)
    print(meta["frequency"].round(2).to_string(index=False))
    print(meta["scorecard"].round(2).to_string(index=False))
    for pattern in PATTERNS:
        print(pattern)
        print(tables[pattern]["behaviour"].round(2).to_string(index=False))
    for setup in SETUPS:
        for name in ("summary", "returns", "first_touch", "behaviour", "gap", "month"):
            print(meta["paths"][setup.key][name].round(2).to_string(index=False))
    print(f"Workbook written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gap Momentum T+1 open follow-through study")
    parser.add_argument("--data-dir", default="equity_data")
    parser.add_argument("--symbols", default="@symbols.txt")
    parser.add_argument("--out", default="reports/GapMomentum_T1_Open_Study.xlsx")
    args = parser.parse_args()
    run(Path(args.data_dir), _read_symbols(args.symbols), Path(args.out))
