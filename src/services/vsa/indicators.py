"""
indicators.py
Pure mathematical functions for calculating VSA-related indicators.
Uses NumPy for vectorized performance.
"""

import pandas as pd
import numpy as np
from typing import Tuple

def calculate_spread(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Calculates the price spread (High - Low)."""
    return high - low

def calculate_close_position(close: np.ndarray, low: np.ndarray, spread: np.ndarray) -> np.ndarray:
    """
    Calculates where price closed relative to the day's range.
    Returns value between 0 (at Low) and 1 (at High).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        pos = np.where(
            spread == 0, 
            0.5, 
            (close - low) / spread
        )
    return pos

def calculate_moving_average(data: np.ndarray, period: int) -> np.ndarray:
    """Calculates simple moving average for a series."""
    return pd.Series(data).rolling(window=period, min_periods=1).mean().to_numpy()

def calculate_price_trend(close: np.ndarray, close_ma: np.ndarray) -> np.ndarray:
    """
    Calculates trend state based on close price relative to its MA.
    2: Strong Up, 1: Up, -1: Down, -2: Strong Down
    """
    return np.select([
        close > close_ma * 1.02,
        close < close_ma * 0.98,
        close > close_ma,
        close < close_ma
    ], [2, -2, 1, -1], default=0)

def detect_bar_types(open_p: np.ndarray, close_p: np.ndarray, spread: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns boolean arrays for (IsUp, IsDown, IsDoji)."""
    is_up = close_p > open_p
    is_down = close_p < open_p
    is_doji = np.abs(close_p - open_p) <= spread * 0.1
    return is_up, is_down, is_doji

def calculate_effort_vs_result(df: pd.DataFrame) -> List[str]:
    """
    Port of legacy Effort vs Result logic.
    Categorizes the 'effort' (volume) vs the 'result' (price movement).
    """
    n = len(df)
    if n == 0: return []
    effort_results = ["Normal_Activity"]
    
    vol = df["Volume"].to_numpy()
    close = df["Close"].to_numpy()
    spread = df["Spread"].to_numpy()
    close_pos = df["Close_Position"].to_numpy()
    
    # Vectorized calculations for speed
    vol_ratio = vol[1:] / np.maximum(vol[:-1], 1)
    price_change = (close[1:] - close[:-1]) / np.maximum(close[:-1], 0.001)
    spread_ratio = spread[1:] / np.maximum(spread[:-1], 0.001)
    cp = close_pos[1:]
    
    for i in range(len(vol_ratio)):
        res = "Normal_Activity"
        vr, pc, sr, current_cp = vol_ratio[i], price_change[i], spread_ratio[i], cp[i]
        
        if vr > 1.5: # HIGH_VOLUME threshold from legacy
            if abs(pc) < 0.005:
                if current_cp > 0.6: res = "Absorption_Bullish"
                elif current_cp < 0.4: res = "Absorption_Bearish"
                else: res = "Professional_Activity"
            elif pc < -0.01: res = "Panic_Selling"
            elif pc > 0.01:
                if sr < 0.8: res = "Hidden_Buying"
                else: res = "Genuine_Buying"
        elif vr < 0.7: # LOW_VOLUME threshold
            if abs(pc) > 0.01:
                res = "No_Supply" if pc > 0 else "No_Demand"
            else:
                res = "Quiet_Market"
        
        effort_results.append(res)
        
    return effort_results
