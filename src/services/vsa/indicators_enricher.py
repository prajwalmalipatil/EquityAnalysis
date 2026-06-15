"""
indicators_enricher.py
Enriches DataFrame with VSA indicators.
"""

import pandas as pd
from src.services.vsa.indicators import (
    calculate_spread,
    calculate_close_position,
    calculate_moving_average,
    calculate_price_trend,
    detect_bar_types,
    calculate_support_resistance,
)


class VSAIndicatorsEnricher:
    """Enriches stock price DataFrame with Volume Spread Analysis (VSA) indicators."""

    @staticmethod
    def enrich(df: pd.DataFrame) -> pd.DataFrame:
        """Calculates indicators and appends them as columns."""
        high, low, close = df["High"].values, df["Low"].values, df["Close"].values
        df["Spread"] = calculate_spread(high, low)
        df["Close_Position"] = calculate_close_position(close, low, df["Spread"].values)
        df["Volume_MA"] = calculate_moving_average(df["Volume"].values, 20)
        df["Spread_MA"] = calculate_moving_average(df["Spread"].values, 20)
        df["Close_MA"] = calculate_moving_average(close, 20)
        df["Price_Trend"] = calculate_price_trend(close, df["Close_MA"].values)
        df["IsUpBar"], df["IsDownBar"], _ = detect_bar_types(
            df["Open"].values, close, df["Spread"].values
        )

        # Support/Resistance context
        df["Support_20"], df["Resistance_20"] = calculate_support_resistance(low, high, 20)

        df["Prev_Volume"] = df["Volume"].shift(1).fillna(df["Volume"])
        df["Prev_Spread"] = df["Spread"].shift(1).fillna(df["Spread"])
        df["Vol_Pct"] = ((df["Volume"] - df["Prev_Volume"]) / df["Prev_Volume"]) * 100
        df["Spr_Pct"] = ((df["Spread"] - df["Prev_Spread"]) / df["Prev_Spread"]) * 100
        return df
