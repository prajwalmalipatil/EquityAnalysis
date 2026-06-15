"""
signal_applier.py
Applies VSA matches, forward validations, and anomalies to enriched DataFrame.
"""

import pandas as pd
from src.constants import vsa_constants as const
from src.services.vsa.indicators import calculate_effort_vs_result
from .pattern_matcher import VSAClassicMatcher, AnomalyV2Matcher


class VSASignalApplier:
    """Applies VSA matches, forward validations, and anomalies to enriched DataFrame."""

    @classmethod
    def apply_signals(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Main entrypoint to evaluate and apply VSA Signals on the DataFrame."""
        signals, efforts, confidences, descriptions = [], [], [], []
        anomaly_v2, validation_status, confirmed_fire = [], [], []

        lookup_days = const.DEFAULT_LOOKAHEAD

        for i in range(len(df)):
            row = df.iloc[i]
            prev_up = df.iloc[i - 1]["IsUpBar"] if i > 0 else False

            # 1. Base Signal Matching
            vsa_res = VSAClassicMatcher.match_signal(
                row["Volume"],
                row["Volume_MA"],
                row["Spread"],
                row["Spread_MA"],
                row["Close_Position"],
                row["Price_Trend"],
                row["IsUpBar"],
                row["IsDownBar"],
                prev_up,
                row["Support_20"],
                row["Low"],
            )

            if vsa_res:
                signals.append(vsa_res.pattern_name)
                efforts.append(vsa_res.effort_vs_result)
                confidences.append(vsa_res.confidence)
                descriptions.append(vsa_res.description)

                # Forward validation
                status = cls._forward_validate(
                    df, i, vsa_res.pattern_name, vsa_res.sentiment, lookup_days
                )
                validation_status.append(status)
            else:
                signals.append("No Signal")
                efforts.append("Neutral")
                confidences.append(0.0)
                descriptions.append("No institutional signal detected.")
                validation_status.append("N/A")

            # 2. Anomaly V2 logic
            anomaly_v2.append(cls._match_anomaly_v2(df, i))

            # 3. Fire Signal Logic (🔥)
            is_fire = row["Volume"] > (row["Volume_MA"] * 2.0) and "No" not in signals[-1]
            confirmed_fire.append("🔥" if is_fire else "")

        df["Signal_Type"] = signals
        df["Effort_Result"] = calculate_effort_vs_result(df)
        df["Validation_Status"] = validation_status
        df["Confirmed_Fire"] = confirmed_fire
        df["Confidence"] = confidences
        df["Description"] = descriptions
        df["Anomaly_V2"] = anomaly_v2
        return df

    @classmethod
    def _forward_validate(
        cls, df: pd.DataFrame, index: int, pattern_name: str, sentiment: str, lookup_days: int
    ) -> str:
        """Performs lookahead validation for a VSA pattern signal."""
        if index + lookup_days >= len(df):
            return "Pending ⏳"

        row = df.iloc[index]
        for step in range(1, lookup_days + 1):
            future_row = df.iloc[index + step]
            if cls._is_validation_successful(pattern_name, sentiment, row, future_row, step):
                return "Confirmed ✅"

        return "Failed ❌"

    @staticmethod
    def _is_validation_successful(
        sig: str, sentiment: str, row: pd.Series, f: pd.Series, step: int
    ) -> bool:
        """Determines if a forward validation condition is met for a given step."""
        if sig == "Upthrust (Bearish)":
            return bool(f["IsDownBar"] and f["Volume"] >= row["Volume"] * 0.8)

        if sig == "No Demand (Bearish Weakness)":
            return bool(
                step <= 2
                and (f["IsDownBar"] or (f["IsUpBar"] and f["Volume"] < f["Volume_MA"] * 0.8))
            )

        if sig == "Stopping Volume (Potential Reversal)":
            return bool(step >= 2 and f["Close"] > row["Close"] and f["Low"] >= row["Low"] * 0.99)

        if "Climax" in sig:
            if sentiment == "Bullish":
                return bool(f["IsUpBar"] and f["Close"] > row["Close"])
            return bool(f["IsDownBar"] and f["Close"] < row["Close"])

        if sig == "Test (Bullish)":
            return bool(f["Close"] > row["Close"] * 1.02)

        return False

    @staticmethod
    def _match_anomaly_v2(df: pd.DataFrame, index: int) -> str:
        """Classifies V2 anomalies for a given row index."""
        row = df.iloc[index]
        if index > 0:
            prev = df.iloc[index - 1]
            ohlc = {"open": row["Open"], "high": row["High"], "low": row["Low"], "close": row["Close"]}
            classif = AnomalyV2Matcher.classify(row["Vol_Pct"], ohlc, prev["Close"], prev["Open"])
            return classif.pattern_name
        return "Neutral"
