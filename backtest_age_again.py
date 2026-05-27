"""
backtest_age_again.py
Historical backtest for the AgeAgain Filter across all available NSE CSVs.

Two scenarios:
  A) Vol_Surge_Spread_Contraction : T_vol > T1_vol AND T_spread < T1_spread  → Bullish (Absorption)
  B) Vol_Drop_Spread_Expansion    : T_vol < T1_vol AND T_spread > T1_spread  → Bearish (Effort Without Result)

Win = price moved in the predicted direction within the look-ahead window.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional


# ──────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────
@dataclass
class Trade:
    symbol: str
    date: str
    scenario: str       # "Vol_Surge_Spread_Contraction" | "Vol_Drop_Spread_Expansion"
    sentiment: str      # "Bullish" | "Bearish"
    t_vol: float
    t1_vol: float
    vol_pct: float
    t_spread: float
    t1_spread: float
    spread_pct: float
    t_cp: float
    close_price: float
    win_1d: bool = False
    win_3d: bool = False
    win_5d: bool = False
    ret_1d: float = 0.0
    ret_3d: float = 0.0
    ret_5d: float = 0.0
    max_fwd_5d_pct: float = 0.0


# ──────────────────────────────────────────────
# CSV loader — handles NSE format
# ──────────────────────────────────────────────
def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = [c.strip().strip('"').strip() for c in df.columns]

        # Priority-ordered mapping: first match wins per column
        col_map = {}
        for c in df.columns:
            cu = c.upper().replace(" ", "_")
            if c in col_map:
                continue
            if "DATE" in cu and "UPDATE" not in cu:
                col_map[c] = "Date"
            elif cu == "OPEN_PRICE" or cu == "OPEN":
                col_map[c] = "Open"
            elif cu == "HIGH_PRICE" or cu == "HIGH":
                col_map[c] = "High"
            elif cu == "LOW_PRICE" or cu == "LOW":
                col_map[c] = "Low"
            elif "CLOSE_PRICE" in cu or cu == "CLOSE":
                col_map[c] = "Close"
            elif "TOTAL_TRADED_QUANTITY" in cu or "TOTTRDQTY" in cu or cu == "VOLUME":
                col_map[c] = "Volume"
            elif cu == "SERIES":
                col_map[c] = "Series"

        df = df.rename(columns=col_map)

        # Drop any accidental duplicate columns, keeping first occurrence
        df = df.loc[:, ~df.columns.duplicated()]

        if "Series" in df.columns:
            df = df[df["Series"].astype(str).str.strip().str.upper() == "EQ"].copy()

        required = ["Date", "Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in required):
            return None

        for c in ["Open", "High", "Low", "Close", "Volume"]:
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(",", "").str.replace('"', "").str.strip(),
                errors="coerce"
            )

        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=required).sort_values("Date").reset_index(drop=True)
        return df if len(df) >= 10 else None

    except Exception:
        return None


# ──────────────────────────────────────────────
# Signal detection
# ──────────────────────────────────────────────
def _detect_scenario(t_vol, t1_vol, t_spread, t1_spread) -> Optional[str]:
    if t_vol > t1_vol and t_spread < t1_spread:
        return "Vol_Surge_Spread_Contraction"
    if t_vol < t1_vol and t_spread > t1_spread:
        return "Vol_Drop_Spread_Expansion"
    return None


# ──────────────────────────────────────────────
# Main backtest
# ──────────────────────────────────────────────
def run_backtest():
    data_dir = Path("equity_data")
    csvs = sorted(data_dir.glob("*.csv"))
    print(f"\n{'='*60}")
    print(f"  AgeAgain Filter — Historical Backtest")
    print(f"{'='*60}")
    print(f"  Tickers loaded : {len(csvs)}")

    trades: List[Trade] = []
    tickers_scanned = 0
    tickers_with_signals = 0

    for csv_path in csvs:
        df = _load_csv(csv_path)
        if df is None:
            continue

        symbol = csv_path.stem.split("_")[0]
        tickers_scanned += 1

        df["Spread"]       = df["High"] - df["Low"]
        df["Close_Pos"]    = np.where(
            df["Spread"] > 0, (df["Close"] - df["Low"]) / df["Spread"], 0.5
        )
        df["Prev_Vol"]     = df["Volume"].shift(1)
        df["Prev_Spread"]  = df["Spread"].shift(1)

        ticker_signals = 0
        # Need 5 forward candles; start at i=1 (need T-1)
        for i in range(1, len(df) - 5):
            row   = df.iloc[i]
            t_vol, t1_vol       = row["Volume"],     row["Prev_Vol"]
            t_spread, t1_spread = row["Spread"],     row["Prev_Spread"]

            if any(pd.isna(v) or v <= 0 for v in [t_vol, t1_vol, t_spread, t1_spread]):
                continue

            scenario = _detect_scenario(t_vol, t1_vol, t_spread, t1_spread)
            if scenario is None:
                continue

            sentiment   = "Bullish" if scenario == "Vol_Surge_Spread_Contraction" else "Bearish"
            t_close     = row["Close"]
            t_cp        = row["Close_Pos"]
            vol_pct     = ((t_vol - t1_vol) / t1_vol) * 100
            spread_pct  = ((t_spread - t1_spread) / t1_spread) * 100

            fwd = df.iloc[i + 1 : i + 6]
            c1, c3, c5 = fwd.iloc[0]["Close"], fwd.iloc[2]["Close"], fwd.iloc[4]["Close"]
            r1 = (c1 - t_close) / t_close
            r3 = (c3 - t_close) / t_close
            r5 = (c5 - t_close) / t_close

            if sentiment == "Bullish":
                win_1d, win_3d, win_5d = r1 > 0, r3 > 0, r5 > 0
                max_move = (fwd["High"].max() - t_close) / t_close
            else:
                win_1d, win_3d, win_5d = r1 < 0, r3 < 0, r5 < 0
                max_move = (t_close - fwd["Low"].min()) / t_close

            trades.append(Trade(
                symbol=symbol, date=str(row["Date"].date()),
                scenario=scenario, sentiment=sentiment,
                t_vol=t_vol, t1_vol=t1_vol, vol_pct=round(vol_pct, 1),
                t_spread=t_spread, t1_spread=t1_spread, spread_pct=round(spread_pct, 1),
                t_cp=round(t_cp, 4), close_price=t_close,
                win_1d=win_1d, win_3d=win_3d, win_5d=win_5d,
                ret_1d=round(r1 * 100, 2), ret_3d=round(r3 * 100, 2), ret_5d=round(r5 * 100, 2),
                max_fwd_5d_pct=round(max_move * 100, 2),
            ))
            ticker_signals += 1

        if ticker_signals > 0:
            tickers_with_signals += 1

    if not trades:
        print("  ❌ No historical signals found.")
        return

    df_t = pd.DataFrame([t.__dict__ for t in trades])
    bulls = df_t[df_t["sentiment"] == "Bullish"]
    bears = df_t[df_t["sentiment"] == "Bearish"]

    print(f"  Tickers scanned         : {tickers_scanned}")
    print(f"  Tickers with ≥1 signal  : {tickers_with_signals}")
    print(f"  Total signal occurrences: {len(df_t)}")
    print(f"    ► Absorption (Bullish) : {len(bulls)}")
    print(f"    ► Effort-W-R (Bearish) : {len(bears)}")

    # ── Helper ──────────────────────────────────────────────────
    def print_stats(chunk: pd.DataFrame, name: str):
        if chunk.empty:
            print(f"\n  {name}: No signals found.")
            return
        w1 = chunk["win_1d"].mean() * 100
        w3 = chunk["win_3d"].mean() * 100
        w5 = chunk["win_5d"].mean() * 100
        avg_r1 = chunk["ret_1d"].mean()
        avg_r3 = chunk["ret_3d"].mean()
        avg_r5 = chunk["ret_5d"].mean()
        mfe    = chunk["max_fwd_5d_pct"].mean()
        print(f"\n  {'─'*50}")
        print(f"  {name}  (n={len(chunk)})")
        print(f"  {'─'*50}")
        print(f"  1-Day  Win Rate : {w1:5.1f}%   Avg Ret: {avg_r1:+.2f}%")
        print(f"  3-Day  Win Rate : {w3:5.1f}%   Avg Ret: {avg_r3:+.2f}%")
        print(f"  5-Day  Win Rate : {w5:5.1f}%   Avg Ret: {avg_r5:+.2f}%")
        print(f"  Avg Max Favorable Excursion (5D): {mfe:+.2f}%")

    print_stats(bulls, "🟢 ABSORPTION SIGNAL  (Vol↑ + Spread↓)")
    print_stats(bears, "🔴 EFFORT WITHOUT RESULT  (Vol↓ + Spread↑)")

    # ── Comparative breakdown ────────────────────────────────────
    print(f"\n\n  {'='*50}")
    print(f"  CONTEXT ANALYSIS — Where Does Edge Come From?")
    print(f"  {'='*50}")

    # 1. Vol surge magnitude vs win rate (Bullish)
    if not bulls.empty:
        med_vol = bulls["vol_pct"].median()
        hi_vol  = bulls[bulls["vol_pct"] > med_vol]
        lo_vol  = bulls[bulls["vol_pct"] <= med_vol]
        print(f"\n  [Bullish] By Vol Surge Magnitude (median={med_vol:.1f}%):")
        print(f"    Surge > median : 5D Win {hi_vol['win_5d'].mean()*100:.1f}%  (n={len(hi_vol)})")
        print(f"    Surge ≤ median : 5D Win {lo_vol['win_5d'].mean()*100:.1f}%  (n={len(lo_vol)})")

    # 2. Spread contraction depth vs win rate (Bullish)
    if not bulls.empty:
        med_spr = bulls["spread_pct"].median()   # negative values = contraction
        deep    = bulls[bulls["spread_pct"] < med_spr]
        shallow = bulls[bulls["spread_pct"] >= med_spr]
        print(f"\n  [Bullish] By Spread Contraction Depth (median={med_spr:.1f}%):")
        print(f"    Deeper contraction : 5D Win {deep['win_5d'].mean()*100:.1f}%  (n={len(deep)})")
        print(f"    Shallower          : 5D Win {shallow['win_5d'].mean()*100:.1f}%  (n={len(shallow)})")

    # 3. Close position influence (Bullish)
    if not bulls.empty:
        strong_close = bulls[bulls["t_cp"] >= 0.70]
        weak_close   = bulls[bulls["t_cp"] < 0.70]
        print(f"\n  [Bullish] By Close Position:")
        print(f"    Strong close (CP≥0.70) : 5D Win {strong_close['win_5d'].mean()*100:.1f}%  (n={len(strong_close)})")
        print(f"    Weak/mid close (CP<0.70): 5D Win {weak_close['win_5d'].mean()*100:.1f}%  (n={len(weak_close)})")

    # 4. Vol drop magnitude vs win rate (Bearish)
    if not bears.empty:
        med_vol_b = bears["vol_pct"].abs().median()
        deep_drop = bears[bears["vol_pct"].abs() > med_vol_b]
        shlo_drop = bears[bears["vol_pct"].abs() <= med_vol_b]
        print(f"\n  [Bearish] By Vol Drop Magnitude (median abs={med_vol_b:.1f}%):")
        print(f"    Deeper drop    : 5D Win {deep_drop['win_5d'].mean()*100:.1f}%  (n={len(deep_drop)})")
        print(f"    Shallower drop : 5D Win {shlo_drop['win_5d'].mean()*100:.1f}%  (n={len(shlo_drop)})")

    # 5. Spread expansion magnitude vs win rate (Bearish)
    if not bears.empty:
        med_spr_b = bears["spread_pct"].median()
        wide_exp  = bears[bears["spread_pct"] > med_spr_b]
        narr_exp  = bears[bears["spread_pct"] <= med_spr_b]
        print(f"\n  [Bearish] By Spread Expansion (median={med_spr_b:.1f}%):")
        print(f"    Wider expansion   : 5D Win {wide_exp['win_5d'].mean()*100:.1f}%  (n={len(wide_exp)})")
        print(f"    Narrower expansion: 5D Win {narr_exp['win_5d'].mean()*100:.1f}%  (n={len(narr_exp)})")

    # ── Save full trade log ──────────────────────────────────────
    out = Path("backtest_age_again_results.csv")
    df_t.to_csv(out, index=False)
    print(f"\n  {'='*60}")
    print(f"  Full trade log saved → {out}")
    print(f"  {'='*60}\n")


if __name__ == "__main__":
    run_backtest()
