"""
processor_service.py
Refactored VSA Processing Engine.
Handles CSV loading, indicator calculation, and pattern matching.
"""

import pandas as pd
from pathlib import Path
import shutil
from typing import List, Dict, Optional
import os

from src.services.vsa.indicators import (
    calculate_spread, calculate_close_position, 
    calculate_moving_average, calculate_price_trend, 
    detect_bar_types
)
from .pattern_matcher import VSAClassicMatcher, AnomalyV2Matcher
from .formatters import ExcelFormatter
from src.constants import vsa_constants as const
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("vsa-processor-service")

class VSAProcessorService:
    """
    Coordinates the processing of raw NSE CSV files into analyzed XLSX reports.
    Strictly calibrated to match V2 Money counts and signal naming.
    """
    
    def __init__(self, output_base: Path):
        self.output_base = output_base
        self._ensure_dirs(clean=True)

    def _ensure_dirs(self, clean: bool = False):
        """Creates necessary subdirectories and optionally clears existing data."""
        dirs = [
            const.RESULTS_DIR_NAME, const.TRENDING_DIR_NAME, 
            const.ANOMALY_DIR_NAME, const.TICKER_DIR_NAME,
            const.TRIGGERS_DIR_NAME
        ]
        for d in dirs:
            dir_path = self.output_base / d
            if clean and dir_path.exists():
                shutil.rmtree(dir_path, ignore_errors=True)
            dir_path.mkdir(parents=True, exist_ok=True)

    def process_file(self, file_path: Path) -> bool:
        """Main pipeline for a single file."""
        try:
            # 1. Load and Standardize
            df = self._load_and_clean(file_path)
            if df.empty:
                return False
                
            # 2. Enrich Indicators
            df = self._enrich_indicators(df)
            
            # 3. Apply Signals
            df = self._apply_signals(df)
            
            # 4. Save and Format
            symbol = file_path.stem.split('_')[0]
            self._save_results(df, symbol)
            
            return True
        except Exception as e:
            logger.error("FILE_PROCESSING_FAILED", extra={"file": str(file_path), "error": str(e)})
            return False

    def _load_and_clean(self, path: Path) -> pd.DataFrame:
        """Extremely robust loader for NSE CSVs."""
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
            if df.empty: return pd.DataFrame()
            
            normalized = [str(c).strip().strip('"').strip("'").lower().replace(" ", "_") for c in df.columns]
            df.columns = normalized
            
            col_map = {
                "open": "Open", "open_price": "Open", "op": "Open",
                "high": "High", "high_price": "High", "hi": "High",
                "low": "Low", "low_price": "Low", "lo": "Low",
                "close": "Close", "close_price": "Close", "last_price": "Close",
                "volume": "Volume", "qty": "Volume", "tottrdqty": "Volume", "total_traded_quantity": "Volume",
                "date": "Date"
            }
            
            rename_dict = {col: col_map[col] for col in df.columns if col in col_map}
            df = df.rename(columns=rename_dict)
            df = df.loc[:, ~df.columns.duplicated()].copy()
            
            required = ["Open", "High", "Low", "Close", "Volume"]
            for col in required:
                if col not in df.columns: return pd.DataFrame()
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.strip(), errors='coerce')
                
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors='coerce', dayfirst=True)
                df = df.sort_values("Date").reset_index(drop=True)
                
            return df.dropna(subset=required).reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def _enrich_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates VSA indicators and essential change metrics."""
        high, low, close = df["High"].values, df["Low"].values, df["Close"].values
        df["Spread"] = calculate_spread(high, low)
        df["Close_Position"] = calculate_close_position(close, low, df["Spread"].values)
        
        df["Volume_MA"] = calculate_moving_average(df["Volume"].values, 20)
        df["Spread_MA"] = calculate_moving_average(df["Spread"].values, 20)
        df["Close_MA"] = calculate_moving_average(close, 20)
        
        df["Price_Trend"] = calculate_price_trend(close, df["Close_MA"].values)
        df["IsUpBar"], _, _ = detect_bar_types(df["Open"].values, close, df["Spread"].values)
        
        df["Prev_Volume"] = df["Volume"].shift(1).fillna(df["Volume"])
        df["Prev_Spread"] = df["Spread"].shift(1).fillna(df["Spread"])
        df["Vol_Pct"] = ((df["Volume"] - df["Prev_Volume"]) / df["Prev_Volume"]) * 100
        df["Spr_Pct"] = ((df["Spread"] - df["Prev_Spread"]) / df["Prev_Spread"]) * 100
        
        return df

    def _apply_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies both VSA and Anomaly logic using the upgraded matcher."""
        signals, efforts, confidences, descriptions, anomaly_v2 = [], [], [], [], []
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            vsa_res = VSAClassicMatcher.match_signal(
                row["Volume"], row["Volume_MA"], row["Spread"], row["Spread_MA"],
                row["Close_Position"], row["Price_Trend"], row["IsUpBar"]
            )
            
            if vsa_res:
                signals.append(f"{vsa_res.pattern_name} ({vsa_res.sentiment})")
                efforts.append(vsa_res.effort_vs_result)
                confidences.append(vsa_res.confidence)
                descriptions.append(vsa_res.description)
            else:
                signals.append("No Signal")
                efforts.append("Neutral")
                confidences.append(0.0)
                descriptions.append("No institutional signal detected.")
            
            if i > 0:
                prev_row = df.iloc[i-1]
                classification = AnomalyV2Matcher.classify(
                    drop_pct=row["Vol_Pct"], 
                    ohlc={"open": row["Open"], "high": row["High"], "low": row["Low"], "close": row["Close"]},
                    prev_close=prev_row["Close"], prev_open=prev_row["Open"]
                )
                anomaly_v2.append(classification.pattern_name)
            else:
                anomaly_v2.append("Neutral")
            
        df["Signal_Type"] = signals
        df["Effort_vs_Result"] = efforts
        df["Confidence"] = confidences
        df["Description"] = descriptions
        df["Anomaly_V2"] = anomaly_v2
        return df

    def _save_results(self, df: pd.DataFrame, symbol: str):
        """Saves results into standardized folders for aggregation."""
        res_dir = self.output_base / const.RESULTS_DIR_NAME
        out_path = res_dir / f"{symbol}_VSA.xlsx"
        
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name="VSA_Analysis", index=False)
            
        from openpyxl import load_workbook
        wb = load_workbook(out_path)
        ExcelFormatter.apply_standard_styling(wb, "VSA_Analysis")
        ExcelFormatter.add_vsa_legend(wb)
        wb.save(out_path)
        
        if df.empty: return
        latest = df.iloc[-1]
        
        # Determine Folder Membership
        has_vsa = latest["Signal_Type"] != "No Signal"
        is_anomaly = latest["Anomaly_V2"] != "Neutral"
        
        # Special Anomaly Count Calibration: Limit 'Neutral Contraction' to serious drops
        if latest["Anomaly_V2"] == "Neutral Contraction" and float(latest["Vol_Pct"]) > -15:
             is_anomaly = False
             
        if has_vsa:
            shutil.copy(out_path, self.output_base / const.TRENDING_DIR_NAME)
        if is_anomaly:
            shutil.copy(out_path, self.output_base / const.ANOMALY_DIR_NAME)
        if has_vsa: # Priority for Ticker is VSA
            shutil.copy(out_path, self.output_base / const.TICKER_DIR_NAME)
            
        # Triggers: Vol Down + Spread Up
        if len(df) > 1:
            prev_row = df.iloc[-2]
            if latest["Volume"] < prev_row["Volume"] and latest["Spread"] > prev_row["Spread"]:
                 shutil.copy(out_path, self.output_base / const.TRIGGERS_DIR_NAME)
