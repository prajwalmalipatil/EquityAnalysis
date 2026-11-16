"""
backtest_gap_momentum.py
Historical study of the Gap Momentum indicator across every extracted NSE CSV.

For every stock and every session T in the history it applies the same rule as
GapMomentumFilterService (volume and spread above T-1, candle confirming the gap),
then measures what the price did afterwards. Entry is the close of day T, because
the signal is only known once the session is complete.

  Gap-Up Continuation   -> measured as a long  (profit when price rises)
  Gap-Down Continuation -> measured as a short (profit when price falls)

Usage:
  python backtest_gap_momentum.py --data-dir equity_data --symbols @symbols.txt \
      --out reports/GapMomentum_Backtest.xlsx
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.services.vsa.file_loader import VSAFileLoader
from src.services.vsa.gap_momentum_filter_service import GapMomentumFilterService

HORIZONS = [1, 3, 5, 10, 20]          # trading days held after the signal close
PRIMARY_HORIZON = 5                   # horizon used for rankings and bucket tables
EXCURSION_WINDOW = 10                 # days used for best / worst price reached
MIN_SIGNALS_PER_STOCK = 5             # minimum signals for a stock to be ranked
TOP_TRADES = 25
TOP_STOCKS = 15
BULLISH, BEARISH = "Gap-Up Continuation", "Gap-Down Continuation"

GAP_BINS = [0, 0.25, 0.5, 1.0, 2.0, np.inf]
GAP_LABELS = ["< 0.25%", "0.25 - 0.5%", "0.5 - 1%", "1 - 2%", "> 2%"]
RATIO_BINS = [1.0, 1.5, 2.0, 3.0, np.inf]
RATIO_LABELS = ["1.0 - 1.5x", "1.5 - 2x", "2 - 3x", "> 3x"]
# Corporate actions: NSE history is not adjusted for splits / bonuses / demergers.
CORPORATE_ACTION_GAP = 0.20           # an overnight open beyond +/-20% is treated as a corporate action
SPLIT_RATIOS = [1 / 10, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4]   # price factor on the ex-date
SPLIT_TOLERANCE = 0.05                # raw ratio within 5% of a standard ratio => split / bonus

HIST_EDGES = [-np.inf, -10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10, np.inf]
HIST_LABELS = ["< -10", "-10 to -7.5", "-7.5 to -5", "-5 to -2.5", "-2.5 to 0",
               "0 to 2.5", "2.5 to 5", "5 to 7.5", "7.5 to 10", "> 10"]


# ────────────────────────────── data preparation ──────────────────────────────
def adjust_corporate_actions(symbol: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """Back-adjusts prices (and volume for splits / bonuses) so the series is continuous.

    A split or bonus snaps to the nearest standard ratio, which preserves the real
    opening gap of the ex-date. Anything else (a demerger) uses the raw open / previous
    close ratio, which makes the ex-date open flat so that day can never raise a signal.
    """
    raw_ratio = df["Open"] / df["Close"].shift(1)
    events = df.index[(raw_ratio - 1).abs() > CORPORATE_ACTION_GAP]
    if len(events) == 0:
        return df, []

    adjusted, actions = df.copy(), []
    for i in events:
        ratio = float(raw_ratio[i])
        nearest = min(SPLIT_RATIOS, key=lambda r: abs(ratio / r - 1))
        is_split = abs(ratio / nearest - 1) <= SPLIT_TOLERANCE
        factor = nearest if is_split else ratio
        earlier = adjusted.index < i
        adjusted.loc[earlier, ["Open", "High", "Low", "Close"]] *= factor
        if is_split:
            adjusted.loc[earlier, "Volume"] /= factor
        actions.append({
            "Symbol": symbol, "Ex-Date": df["Date"][i], "Prev Close (raw)": float(df["Close"][i - 1]),
            "Open (raw)": float(df["Open"][i]), "Raw Ratio": ratio, "Factor Applied": factor,
            "Treated As": "Split / bonus" if is_split else "Demerger / other",
        })
    return adjusted, actions


def load_universe(data_dir: Path, symbols: List[str]) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Loads the newest CSV per requested stock symbol (indices are not stocks, so skipped)."""
    wanted = set(symbols)
    latest_file: Dict[str, Path] = {}
    for path in sorted(data_dir.glob("*.csv")):
        symbol = path.stem.split("_")[0]
        if symbol in wanted:
            latest_file[symbol] = path          # sorted() => last one is the newest date

    universe, actions = {}, []
    for symbol, path in latest_file.items():
        df = VSAFileLoader.load_and_clean(path)
        if "Date" not in df.columns or len(df) < 2:
            continue
        df = df.drop_duplicates(subset="Date", keep="last").reset_index(drop=True)
        bars = df[["Date", "Open", "High", "Low", "Close", "Volume"]].astype({"Volume": float})
        universe[symbol], found = adjust_corporate_actions(symbol, bars)
        actions.extend(found)
    return universe, pd.DataFrame(actions)


def enrich_bars(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """Adds the signal flags and forward-return columns to one stock's daily bars."""
    bars = df.copy()
    bars.insert(0, "Symbol", symbol)
    bars["Spread"] = bars["High"] - bars["Low"]
    prev = bars.shift(1)

    valid_prev = (prev["Volume"] > 0) & (prev["Spread"] > 0) & (prev["Close"] > 0) & (bars["Open"] > 0)
    expansion = (bars["Volume"] > prev["Volume"]) & (bars["Spread"] > prev["Spread"])
    gap_up = (bars["Open"] > prev["Close"]) & (bars["Close"] > bars["Open"])
    gap_down = (bars["Open"] < prev["Close"]) & (bars["Close"] < bars["Open"])

    bars["Signal"] = np.select(
        [valid_prev & expansion & gap_up, valid_prev & expansion & gap_down], [BULLISH, BEARISH], default=""
    )
    bars["T-1 Close"] = prev["Close"]
    bars["Gap %"] = (bars["Open"] / prev["Close"] - 1) * 100
    bars["Day Chg %"] = (bars["Close"] / prev["Close"] - 1) * 100
    bars["Volume x T-1"] = bars["Volume"] / prev["Volume"]
    bars["Spread x T-1"] = bars["Spread"] / prev["Spread"]

    for h in HORIZONS:
        bars[f"Raw {h}D %"] = (bars["Close"].shift(-h) / bars["Close"] - 1) * 100

    fwd_high = bars["High"][::-1].rolling(EXCURSION_WINDOW, min_periods=1).max()[::-1].shift(-1)
    fwd_low = bars["Low"][::-1].rolling(EXCURSION_WINDOW, min_periods=1).min()[::-1].shift(-1)
    bars["Raw High %"] = (fwd_high / bars["Close"] - 1) * 100
    bars["Raw Low %"] = (fwd_low / bars["Close"] - 1) * 100
    return bars.iloc[1:]


def assert_matches_live_filter(symbol: str, df: pd.DataFrame, bars: pd.DataFrame) -> None:
    """Guards against drift: the vectorised rule must agree with the production filter."""
    frame = df.assign(Spread=df["High"] - df["Low"])
    live = GapMomentumFilterService.classify_frame(symbol, frame)
    expected = live.label if live else ""
    if bars.iloc[-1]["Signal"] != expected:
        raise AssertionError(f"{symbol}: backtest rule disagrees with GapMomentumFilterService")


def build_signals(all_bars: pd.DataFrame) -> pd.DataFrame:
    """Keeps the signal days and converts raw moves into direction-adjusted returns."""
    signals = all_bars[all_bars["Signal"] != ""].copy()
    direction = np.where(signals["Signal"] == BULLISH, 1.0, -1.0)
    for h in HORIZONS:
        signals[f"Return {h}D %"] = signals[f"Raw {h}D %"] * direction
    best = np.where(direction > 0, signals["Raw High %"], -signals["Raw Low %"])
    worst = np.where(direction > 0, signals["Raw Low %"], -signals["Raw High %"])
    signals[f"Best in {EXCURSION_WINDOW}D %"] = best
    signals[f"Worst in {EXCURSION_WINDOW}D %"] = worst
    signals["Month"] = signals["Date"].dt.strftime("%Y-%m")
    signals["Gap Size"] = pd.cut(signals["Gap %"].abs(), GAP_BINS, labels=GAP_LABELS, include_lowest=True)
    signals["Volume Bucket"] = pd.cut(signals["Volume x T-1"], RATIO_BINS, labels=RATIO_LABELS)
    signals["Spread Bucket"] = pd.cut(signals["Spread x T-1"], RATIO_BINS, labels=RATIO_LABELS)
    return signals.sort_values(["Date", "Symbol"]).reset_index(drop=True)


# ──────────────────────────────── aggregation ─────────────────────────────────
def horizon_table(signals: pd.DataFrame, all_bars: pd.DataFrame) -> pd.DataFrame:
    """Signal performance per scenario and horizon, next to the all-days baseline."""
    rows = []
    for scenario, sign in [(BULLISH, 1.0), (BEARISH, -1.0)]:
        subset = signals[signals["Signal"] == scenario]
        for h in HORIZONS:
            returns = subset[f"Return {h}D %"].dropna()
            baseline = (all_bars[f"Raw {h}D %"].dropna() * sign)
            per_day = subset.dropna(subset=[f"Return {h}D %"]).groupby("Date")[f"Return {h}D %"].mean()
            rows.append({
                "Scenario": scenario, "Horizon": f"{h}D", "Signals": len(returns),
                "Win Rate %": (returns > 0).mean() * 100 if len(returns) else np.nan,
                "Avg Return %": returns.mean(), "Median Return %": returns.median(),
                "Best %": returns.max(), "Worst %": returns.min(),
                "Signal Days": len(per_day), "Avg per Signal Day %": per_day.mean(),
                "Winning Days %": (per_day > 0).mean() * 100,
                "Baseline Avg %": baseline.mean(),
                "Baseline Win Rate %": (baseline > 0).mean() * 100,
                "Edge vs Baseline %": returns.mean() - baseline.mean(),
            })
    return pd.DataFrame(rows)


def group_table(signals: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    """Signal count, win rate and average return at the primary horizon per group."""
    col = f"Return {PRIMARY_HORIZON}D %"
    grouped = signals.dropna(subset=[col]).groupby(keys, observed=True)[col]
    table = grouped.agg(Signals="count", **{
        "Win Rate %": lambda r: (r > 0).mean() * 100,
        f"Avg {PRIMARY_HORIZON}D %": "mean", f"Median {PRIMARY_HORIZON}D %": "median",
        "Best %": "max", "Worst %": "min",
    })
    # Signals cluster on market-wide days, so also score each trading day once.
    per_day = signals.dropna(subset=[col]).groupby(keys + ["Date"], observed=True)[col].mean()
    by_day = per_day.groupby(level=list(range(len(keys))), observed=True).agg(**{
        "Signal Days": "count", "Avg per Signal Day %": "mean",
        "Winning Days %": lambda r: (r > 0).mean() * 100,
    })
    return table.join(by_day).reset_index()


def stock_table(signals: pd.DataFrame) -> pd.DataFrame:
    """Per-stock scorecard across both scenarios."""
    table = group_table(signals, ["Symbol"])
    counts = signals.groupby("Symbol")["Signal"].agg(
        **{"Gap-Up Signals": lambda s: (s == BULLISH).sum(), "Gap-Down Signals": lambda s: (s == BEARISH).sum()}
    ).reset_index()
    table = table.drop(columns=["Signal Days", "Avg per Signal Day %", "Winning Days %"])
    table = counts.merge(table, on="Symbol", how="left").rename(columns={"Signals": f"Signals with {PRIMARY_HORIZON}D result"})
    return table.sort_values(f"Avg {PRIMARY_HORIZON}D %", ascending=False).reset_index(drop=True)


def outstanding_trades(signals: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Top individual signals for a scenario, ranked by primary-horizon return."""
    col = f"Return {PRIMARY_HORIZON}D %"
    columns = ["Symbol", "Date", "Signal", "T-1 Close", "Open", "Close", "Gap %", "Volume x T-1", "Spread x T-1"]
    columns += [f"Return {h}D %" for h in HORIZONS] + [f"Best in {EXCURSION_WINDOW}D %", f"Worst in {EXCURSION_WINDOW}D %"]
    subset = signals[(signals["Signal"] == scenario)].dropna(subset=[col])
    return subset.nlargest(TOP_TRADES, col)[columns].reset_index(drop=True)


def equity_curve(signals: pd.DataFrame) -> pd.DataFrame:
    """Cumulative next-day return, equal-weighting every signal fired on the same date."""
    daily = signals.dropna(subset=["Return 1D %"]).pivot_table(
        index="Date", columns="Signal", values="Return 1D %", aggfunc="mean"
    ).reindex(columns=[BULLISH, BEARISH]).fillna(0.0)
    return daily.cumsum().reset_index()


def distribution(signals: pd.DataFrame) -> pd.DataFrame:
    """Histogram of primary-horizon returns per scenario."""
    col = f"Return {PRIMARY_HORIZON}D %"
    binned = pd.cut(signals[col], HIST_EDGES, labels=HIST_LABELS)
    table = pd.crosstab(binned, signals["Signal"]).reindex(index=HIST_LABELS, columns=[BULLISH, BEARISH], fill_value=0)
    return table.rename_axis(f"{PRIMARY_HORIZON}D Return Band (%)").reset_index()


# ─────────────────────────────────── main ─────────────────────────────────────
def run(data_dir: Path, symbols: List[str], out_path: Path) -> None:
    from backtest_gap_momentum_report import write_workbook  # presentation kept separate

    universe, actions = load_universe(data_dir, symbols)
    if not universe:
        raise SystemExit(f"No usable stock CSVs found in {data_dir}")

    frames = []
    for symbol, df in sorted(universe.items()):
        bars = enrich_bars(symbol, df)
        assert_matches_live_filter(symbol, df, bars)
        frames.append(bars)
    all_bars = pd.concat(frames, ignore_index=True)
    signals = build_signals(all_bars)

    tables = {
        "horizons": horizon_table(signals, all_bars),
        "stocks": stock_table(signals),
        "months": group_table(signals, ["Month", "Signal"]),
        "gap": group_table(signals, ["Signal", "Gap Size"]),
        "volume": group_table(signals, ["Signal", "Volume Bucket"]),
        "spread": group_table(signals, ["Signal", "Spread Bucket"]),
        "top_bullish": outstanding_trades(signals, BULLISH),
        "top_bearish": outstanding_trades(signals, BEARISH),
        "equity": equity_curve(signals),
        "distribution": distribution(signals),
    }
    meta = {
        "stocks": len(universe), "stock_days": len(all_bars),
        "first_date": all_bars["Date"].min(), "last_date": all_bars["Date"].max(),
        "missing": sorted(set(symbols) - set(universe)), "actions": actions,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_workbook(out_path, signals, tables, meta)
    print(f"stocks={meta['stocks']} stock_days={meta['stock_days']} signals={len(signals)} "
          f"range={meta['first_date']:%Y-%m-%d}..{meta['last_date']:%Y-%m-%d} missing={meta['missing']} corporate_actions={len(actions)}")
    print(tables["horizons"].round(2).to_string(index=False))
    print(f"Workbook written to {out_path}")


def _read_symbols(spec: str) -> List[str]:
    raw = Path(spec[1:]).read_text() if spec.startswith("@") else spec
    return [s.strip().upper() for s in raw.replace(",", "\n").splitlines() if s.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gap Momentum historical backtest")
    parser.add_argument("--data-dir", default="equity_data")
    parser.add_argument("--symbols", default="@symbols.txt")
    parser.add_argument("--out", default="reports/GapMomentum_Backtest.xlsx")
    args = parser.parse_args()
    run(Path(args.data_dir), _read_symbols(args.symbols), Path(args.out))
