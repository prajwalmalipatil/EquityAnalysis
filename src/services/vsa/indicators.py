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
