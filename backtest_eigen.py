import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class TradeRecord:
    symbol: str
    date: str
    sentiment: str
    label: str
    t_vol: float
    vol_surge_pct: float
    gap_pct: float
    delta_cp: float
    t_cp: float
    close_price: float
    win_1d: bool = False
    win_3d: bool = False
    win_5d: bool = False
    max_fwd_move_5d_pct: float = 0.0

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

def run_backtest():
    data_dir = Path("equity_data")
    if not data_dir.exists():
        print("No equity_data dir found.")
        return
        
    all_csvs = list(data_dir.glob("*.csv"))
    print(f"Running historical backtest on {len(all_csvs)} tickers...")
    
    trades: List[TradeRecord] = []
    
    for csv_file in all_csvs:
        symbol = csv_file.stem.split('_')[0]
        try:
            df = pd.read_csv(csv_file)
            if len(df) < 10:
                continue
                
            # Clean column names
            df.columns = [c.strip() for c in df.columns]
            
            # Filter only Equity series if Series column exists
            if "Series" in df.columns:
                df = df[df["Series"] == "EQ"].copy()
                
            # Map columns
            date_col = next((c for c in df.columns if c.upper() == "DATE"), None)
            open_col = next((c for c in df.columns if "OPEN" in c.upper()), None)
            high_col = next((c for c in df.columns if "HIGH" in c.upper()), None)
            low_col = next((c for c in df.columns if "LOW" in c.upper()), None)
            close_col = next((c for c in df.columns if "CLOSE" in c.upper() and "%" not in c.upper() and "PREV" not in c.upper()), None)
            vol_col = next((c for c in df.columns if "QTY" in c.upper() or "TOTAL TRADED QUANTITY" in c.upper() or "VOLUME" in c.upper()), None)
            
            if not all([date_col, open_col, high_col, low_col, close_col, vol_col]):
                continue
                
            # Clean numeric columns strictly
            for c in [open_col, high_col, low_col, close_col, vol_col]:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '').str.replace('-', ''), errors='coerce')
                
            df["Date"] = pd.to_datetime(df[date_col], errors='coerce')
            df.sort_values(by="Date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            df["Open"] = df[open_col].astype(float)
            df["High"] = df[high_col].astype(float)
            df["Low"] = df[low_col].astype(float)
            df["Close"] = df[close_col].astype(float)
            df["Volume"] = df[vol_col].astype(float)
            
            df["Spread"] = df["High"] - df["Low"]
            df["Close_Position"] = np.where(df["Spread"] > 0, (df["Close"] - df["Low"]) / df["Spread"], 0.5)
            
            df["Prev_Close"] = df["Close"].shift(1)
            df["Prev_Vol"] = df["Volume"].shift(1)
            df["Prev_CP"] = df["Close_Position"].shift(1)
            
            # Iterate through historical to simulate live detection
            for i in range(1, len(df) - 5): # Ensure we have at least 5 days forward
                t = df.iloc[i]
                
                t_vol = t["Volume"]
                t1_vol = t["Prev_Vol"]
                
                # Condition 1
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
                
                fwd_5 = df.iloc[i+1 : i+6]
                ret_1d = (fwd_5.iloc[0]["Close"] - t_close) / t_close
                ret_3d = (fwd_5.iloc[2]["Close"] - t_close) / t_close
                ret_5d = (fwd_5.iloc[4]["Close"] - t_close) / t_close
                
                win_1d = ret_1d > 0 if sentiment == "Bullish" else ret_1d < 0
                win_3d = ret_3d > 0 if sentiment == "Bullish" else ret_3d < 0
                win_5d = ret_5d > 0 if sentiment == "Bullish" else ret_5d < 0
                
                max_move_5d = fwd_5["High"].max() if sentiment == "Bullish" else fwd_5["Low"].min()
                max_pct_move = (max_move_5d - t_close) / t_close
                
                gap_pct = (t_open - t1_close) / t1_close
                
                trades.append(TradeRecord(
                    symbol=symbol,
                    date=str(t["Date"].date()),
                    sentiment=sentiment,
                    label=label,
                    t_vol=t_vol,
                    vol_surge_pct=(t_vol/t1_vol - 1),
                    gap_pct=gap_pct,
                    delta_cp=t_cp - t1_cp,
                    t_cp=t_cp,
                    close_price=t_close,
                    win_1d=win_1d,
                    win_3d=win_3d,
                    win_5d=win_5d,
                    max_fwd_move_5d_pct=max_pct_move
                ))
                
        except Exception as e:
            print(f"Error parsing {symbol}: {e}")
            pass
            
    # Analytics
    if not trades:
        print("No historical trades found.")
        return
        
    df_trades = pd.DataFrame([t.__dict__ for t in trades])
    bulls = df_trades[df_trades["sentiment"] == "Bullish"]
    bears = df_trades[df_trades["sentiment"] == "Bearish"]

    
    print("\n" + "="*50)
    print("BACKTEST RESULTS: EIGEN FILTER")
    print(f"Total historical occurrences: {len(df_trades)}")
    print(f"  Bullish Signals: {len(bulls)}")
    print(f"  Bearish Signals: {len(bears)}")
    
    def print_stats(df_chunk, name):
        if len(df_chunk) == 0: return
        w1 = df_chunk["win_1d"].mean() * 100
        w3 = df_chunk["win_3d"].mean() * 100
        w5 = df_chunk["win_5d"].mean() * 100
        avg_move = df_chunk["max_fwd_move_5d_pct"].mean() * 100 if name=="Bullish" else df_chunk["max_fwd_move_5d_pct"].mean() * -100
        print(f"\n{name} PERFORMANCE:")
        print(f"  1-Day Win Rate: {w1:.1f}%")
        print(f"  3-Day Win Rate: {w3:.1f}%")
        print(f"  5-Day Win Rate: {w5:.1f}%")
        print(f"  Avg Max Favorable Excursion (5D): {avg_move:.2f}%")
        
    print_stats(bulls, "Bullish")
    print_stats(bears, "Bearish")
    
    # Context Analysis (Avoidable Observations)
    print("\nCONTEXT ANALYSIS (Finding Optimal Zones):")
    
    # Analyze by Volume Surge
    print("\nBy Volume Surge:")
    for quintile in [1, 2]:    
        high_vol = df_trades[df_trades["vol_surge_pct"] > df_trades["vol_surge_pct"].median()]
        w5_high = high_vol["win_5d"].mean() * 100
        low_vol = df_trades[df_trades["vol_surge_pct"] <= df_trades["vol_surge_pct"].median()]
        w5_low = low_vol["win_5d"].mean() * 100
        
    print(f"  Surge > Median ({df_trades['vol_surge_pct'].median()*100:.1f}%): Win Rate {w5_high:.1f}%")
    print(f"  Surge < Median ({df_trades['vol_surge_pct'].median()*100:.1f}%): Win Rate {w5_low:.1f}%")
    
    # Analyze by Close Position Absolute
    print("\nBy Close Position Extremity (Bullish CP > 0.85 or Bearish CP < 0.15):")
    extreme_cp = df_trades[
        ((df_trades["sentiment"] == "Bullish") & (df_trades["t_cp"] > 0.85)) | 
        ((df_trades["sentiment"] == "Bearish") & (df_trades["t_cp"] < 0.15))
    ]
    weak_cp = df_trades[~df_trades.index.isin(extreme_cp.index)]
    print(f"  Ultra-Extreme CP: Win Rate {extreme_cp['win_5d'].mean()*100:.1f}% (Count: {len(extreme_cp)})")
    print(f"  Borderline CP: Win Rate {weak_cp['win_5d'].mean()*100:.1f}% (Count: {len(weak_cp)})")
    
    # Divergence Analysis
    print("\nConvergence vs Divergence:")
    conv = df_trades[df_trades["label"].str.contains("Convergence")]
    div = df_trades[df_trades["label"].str.contains("Divergence")]
    print(f"  Pure Impulse Convergence: Win Rate {conv['win_3d'].mean()*100:.1f}% (Count: {len(conv)})")
    print(f"  Contested Divergence: Win Rate {div['win_3d'].mean()*100:.1f}% (Count: {len(div)})")
    
if __name__ == "__main__":
    run_backtest()
