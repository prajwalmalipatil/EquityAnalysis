import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
import json
import datetime

@dataclass
class TradeRecord:
    symbol: str
    start_date: str
    completion_date: str
    trigger_pattern: str
    t_vol: float
    vol_surge_pct: float
    fwd_return_5b: Optional[float]
    win: Optional[bool]
    sentiment: str
    gauntlet_passed: bool

EIGEN_LOWER = 0.30
EIGEN_UPPER = 0.70

def _get_label(gap_dir, close_band):
    matrix = {
        ("Gap-Up", "Strong"): ("Bullish Impulse Convergence", "Bullish"),
        ("Gap-Up", "Weak"): ("Contested Bullish Divergence", "Bullish"),
        ("Gap-Down", "Weak"): ("Bearish Impulse Convergence", "Bearish"),
        ("Gap-Down", "Strong"): ("Contested Bearish Divergence", "Bearish"),
    }
    return matrix.get((gap_dir, close_band))

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe == "daily":
        return df.copy()
        
    df_temp = df.copy()
    if not isinstance(df_temp.index, pd.DatetimeIndex):
        df_temp.set_index("Date", inplace=True)
        
    rule = "W" if timeframe == "weekly" else "ME"
    agg_dict = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    resampled = df_temp.resample(rule).agg(agg_dict).dropna()
    resampled = resampled.reset_index()
    if 'Date' not in resampled.columns and resampled.index.name == 'Date':
        resampled = resampled.reset_index()
    elif 'index' in resampled.columns:
        resampled.rename(columns={'index': 'Date'}, inplace=True)
    return resampled

def evaluate_timeframe(df: pd.DataFrame, symbol: str) -> List[TradeRecord]:
    trades = []
    if len(df) < 5:
        return trades
        
    df = df.copy()
    df["Spread"] = df["High"] - df["Low"]
    df["Close_Position"] = np.where(df["Spread"] > 0, (df["Close"] - df["Low"]) / df["Spread"], 0.5)
    
    df["Prev_Close"] = df["Close"].shift(1)
    df["Prev_Vol"] = df["Volume"].shift(1)
    df["Prev_CP"] = df["Close_Position"].shift(1)
    
    for i in range(1, len(df)):
        t = df.iloc[i]
        
        t_vol = t["Volume"]
        t1_vol = t["Prev_Vol"]
        
        if pd.isna(t_vol) or pd.isna(t1_vol) or t1_vol <= 0 or t_vol <= t1_vol:
            continue
            
        t_open = t["Open"]
        t_close = t["Close"]
        t1_close = t["Prev_Close"]
        t_cp = t["Close_Position"]
        t1_cp = t["Prev_CP"]
        
        if pd.isna(t_cp) or pd.isna(t1_cp): continue
        
        is_extreme_close = (t_cp <= EIGEN_LOWER) or (t_cp >= EIGEN_UPPER)
        if not is_extreme_close:
            continue
            
        gap_dir = None
        if t_open > t1_close and t_cp >= t1_cp: gap_dir = "Gap-Up"
        elif t_open < t1_close and t_cp <= t1_cp: gap_dir = "Gap-Down"
        
        if not gap_dir: continue
        
        close_band = "Strong" if t_cp >= EIGEN_UPPER else "Weak"
        label_tuple = _get_label(gap_dir, close_band)
        if not label_tuple: continue
        
        label, sentiment = label_tuple
        label, sentiment = label_tuple
        
        start_date = str(df.iloc[i-1]["Date"]).split()[0]
        t_vol_float = float(t_vol)
        vol_surge_pct = float((t_vol/t1_vol - 1) * 100)
        
        # Evaluate 4-stage Gauntlet: LVLS -> HVHS -> LVLS -> HVHS
        gauntlet_passed = False
        completion_date = "--"
        fwd_return_5b = None
        win = None
        
        if i + 4 < len(df):
            # T1: LVLS
            t1_eval = df.iloc[i+1]
            t1_vol_eval = t1_eval["Volume"]
            t1_spread = t1_eval["Spread"]
            if t1_vol_eval < t_vol and t1_spread < t["Spread"]:
                # T2: HVHS
                t2_eval = df.iloc[i+2]
                t2_vol_eval = t2_eval["Volume"]
                t2_spread = t2_eval["Spread"]
                if t2_vol_eval > t1_vol_eval and t2_spread > t1_spread:
                    # T3: LVLS
                    t3_eval = df.iloc[i+3]
                    t3_vol_eval = t3_eval["Volume"]
                    t3_spread = t3_eval["Spread"]
                    if t3_vol_eval < t2_vol_eval and t3_spread < t2_spread:
                        # T4: HVHS
                        t4_eval = df.iloc[i+4]
                        t4_vol_eval = t4_eval["Volume"]
                        t4_spread = t4_eval["Spread"]
                        if t4_vol_eval > t3_vol_eval and t4_spread > t3_spread:
                            gauntlet_passed = True
                            completion_date = str(t4_eval["Date"]).split()[0]
                            
                            # Check forward 5 bars starting AFTER T+4
                            if i + 4 + 5 < len(df):
                                fwd_5 = df.iloc[i+5 : i+10]
                                t4_close = t4_eval["Close"]
                                if sentiment == "Bullish":
                                    fwd_return_5b = (fwd_5["Close"].max() - t4_close) / t4_close
                                    win = fwd_return_5b > 0
                                else:
                                    fwd_return_5b = (fwd_5["Close"].min() - t4_close) / t4_close
                                    win = fwd_return_5b < 0
        
        trades.append(TradeRecord(
            symbol=symbol,
            start_date=start_date,
            completion_date=completion_date,
            trigger_pattern=label,
            t_vol=t_vol_float,
            vol_surge_pct=vol_surge_pct,
            fwd_return_5b=float(fwd_return_5b * 100) if fwd_return_5b is not None else None,
            win=bool(win) if win is not None else None,
            sentiment=sentiment,
            gauntlet_passed=gauntlet_passed
        ))
        
    return trades

def run_backtest():
    data_dir = Path("equity_data")
    if not data_dir.exists():
        print("No equity_data dir found.")
        return
        
    all_csvs = list(data_dir.glob("*.csv"))
    print(f"Running historical backtest on {len(all_csvs)} tickers...")
    
    trades_daily = []
    trades_weekly = []
    trades_monthly = []
    
    for csv_file in all_csvs:
        symbol = csv_file.stem.split('_')[0]
        try:
            df = pd.read_csv(csv_file)
            if len(df) < 10: continue
            
            df.columns = [c.strip() for c in df.columns]
            if "Series" in df.columns:
                df = df[df["Series"] == "EQ"].copy()
                
            date_col = next((c for c in df.columns if c.upper() == "DATE"), None)
            open_col = next((c for c in df.columns if "OPEN" in c.upper()), None)
            high_col = next((c for c in df.columns if "HIGH" in c.upper()), None)
            low_col = next((c for c in df.columns if "LOW" in c.upper()), None)
            close_col = next((c for c in df.columns if "CLOSE" in c.upper() and "%" not in c.upper() and "PREV" not in c.upper()), None)
            vol_col = next((c for c in df.columns if "QTY" in c.upper() or "TOTAL TRADED QUANTITY" in c.upper() or "VOLUME" in c.upper()), None)
            
            if not all([date_col, open_col, high_col, low_col, close_col, vol_col]): continue
            
            for c in [open_col, high_col, low_col, close_col, vol_col]:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '').str.replace('-', ''), errors='coerce')
                
            df["Date"] = pd.to_datetime(df[date_col], errors='coerce')
            df.sort_values(by="Date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            df = df.rename(columns={
                open_col: "Open", high_col: "High", low_col: "Low", close_col: "Close", vol_col: "Volume"
            })
            
            df_daily = resample_ohlcv(df, "daily")
            df_weekly = resample_ohlcv(df, "weekly")
            df_monthly = resample_ohlcv(df, "monthly")
            
            trades_daily.extend(evaluate_timeframe(df_daily, symbol))
            trades_weekly.extend(evaluate_timeframe(df_weekly, symbol))
            trades_monthly.extend(evaluate_timeframe(df_monthly, symbol))
            
        except Exception as e:
            print(f"Error parsing {symbol}: {e}")
            
    # Calculate overall metrics
    all_trades = trades_daily + trades_weekly + trades_monthly
    if not all_trades:
        print("No trades found.")
        return
        
    total_eval = len(all_trades) # Number of T0 triggers
    completed_sequences = [t for t in all_trades if t.gauntlet_passed and t.win is not None]
    
    wins = len([t for t in completed_sequences if t.win])
    failures = len([t for t in completed_sequences if not t.win])
    win_rate = (wins / len(completed_sequences) * 100) if completed_sequences else 0
    
    # Sort completions by date descending
    def sort_completions(completions):
        return sorted([t.__dict__ for t in completions if t.gauntlet_passed and t.win], key=lambda x: x['completion_date'], reverse=True)

    results_payload = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "overall_metrics": {
            "total_sequences": total_eval,
            "win_rate": win_rate,
            "total_completed": wins,
            "total_failed": failures
        },
        "completions_daily": sort_completions(trades_daily),
        "completions_weekly": sort_completions(trades_weekly),
        "completions_monthly": sort_completions(trades_monthly)
    }
    
    out_dir = Path("dashboard")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "backtest_results.json", "w") as f:
        json.dump(results_payload, f, indent=2)
        
    print(f"Backtest completed! Generated completions for UI. Win Rate: {win_rate:.1f}%, Total: {total_eval}")

if __name__ == "__main__":
    run_backtest()
