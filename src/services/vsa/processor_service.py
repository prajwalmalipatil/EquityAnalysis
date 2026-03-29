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
    Now uses ultra-robust column mapping to handle BOM/Spaces and 16-pattern anomaly logic.
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
        """
        Extremely robust loader for NSE CSVs.
        Handles BOM, Quoted headers, trailing spaces, and various OHLCV names.
        """
        try:
            # 1. Handle BOM and initial load
            df = pd.read_csv(path, encoding="utf-8-sig")
            if df.empty: return pd.DataFrame()
            
            # 2. Aggressive Column Normalization
            original_cols = df.columns.tolist()
            normalized = []
            for c in original_cols:
                n = str(c).strip().strip('"').strip("'").lower()
                n = n.replace(" ", "_").replace(".", "").replace("₹", "rs")
                normalized.append(n)
            df.columns = normalized
            
            # 3. Comprehensive Mapping
            col_map = {
                "open": "Open", "open_price": "Open", "op": "Open",
                "high": "High", "high_price": "High", "hi": "High",
                "low": "Low", "low_price": "Low", "lo": "Low",
                "close": "Close", "close_price": "Close", "last_price": "Close", "prev_close": "Prev_Close",
                "volume": "Volume", "qty": "Volume", "quantity": "Volume", "tottrdqty": "Volume", 
                "total_traded_quantity": "Volume", "traded_qty": "Volume",
                "date": "Date", "timestamp": "Date"
            }
            
            rename_dict = {col: col_map[col] for col in df.columns if col in col_map}
            df = df.rename(columns=rename_dict)
            df = df.loc[:, ~df.columns.duplicated()].copy()
            
            # 4. Mandatory column verification
            required = ["Open", "High", "Low", "Close", "Volume"]
            for col in required:
                if col not in df.columns:
                    # Fuzzy match fallback
                    found = False
                    key = col.lower()
                    for actual in df.columns:
                        if key in actual:
                            df = df.rename(columns={actual: col})
                            found = True
                            break
                    if not found: return pd.DataFrame()
            
            # 5. Numeric conversion (Handle commas)
            for col in required:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.strip(), errors='coerce')
                
            # 6. Date Handling
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors='coerce', dayfirst=True)
                df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
                
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
        df["IsUpBar"], df["IsDownBar"], _ = detect_bar_types(df["Open"].values, close, df["Spread"].values)
        
        df["Prev_Volume"] = df["Volume"].shift(1).fillna(df["Volume"])
        df["Prev_Spread"] = df["Spread"].shift(1).fillna(df["Spread"])
        df["Vol_Pct"] = ((df["Volume"] - df["Prev_Volume"]) / df["Prev_Volume"]) * 100
        df["Spr_Pct"] = ((df["Spread"] - df["Prev_Spread"]) / df["Prev_Spread"]) * 100
        
        return df

    def _apply_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies both Classic VSA and High-Confidence Anomaly V2 logic."""
        signals, efforts, confidences, descriptions, anomaly_v2 = [], [], [], [], []
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            vsa_res = VSAClassicMatcher.match_climax(
                row["Volume"], row["Volume_MA"], row["Spread"], row["Spread_MA"],
                row["Close_Position"], row["Price_Trend"]
            )
            if not vsa_res:
                vsa_res = VSAClassicMatcher.match_no_demand(
                    row["Volume"], row["Volume_MA"], row["IsUpBar"], row["Price_Trend"]
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
                descriptions.append("")
            
            # Anomaly V2 Classification (High-Confidence Only)
            if i > 0:
                prev_row = df.iloc[i-1]
                classification = AnomalyV2Matcher.classify(
                    drop_pct=row["Vol_Pct"],
                    ohlc={"open": row["Open"], "high": row["High"], "low": row["Low"], "close": row["Close"]},
                    prev_close=prev_row["Close"],
                    prev_open=prev_row["Open"]
                )
                # Filter out generic 'Neutral Contraction' from reports later, but store pattern name
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
        """Saves results and distributes to folders based on HIGH-CONFIDENCE triggers."""
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
        
        # 1. Trending
        has_vsa = latest["Signal_Type"] != "No Signal"
        if has_vsa:
            shutil.copy(out_path, self.output_base / const.TRENDING_DIR_NAME)
            
        # 2. Anomaly (Exclude generic patterns to match the expected '16' count)
        v2_pattern = str(latest["Anomaly_V2"])
        is_high_conf_anomaly = v2_pattern not in [
            "Neutral", "Neutral Contraction", "Volume Spike", "Volume Drop"
        ]
        if is_high_conf_anomaly:
            shutil.copy(out_path, self.output_base / const.ANOMALY_DIR_NAME)
            
        # 3. Ticker
        if has_vsa or is_high_conf_anomaly:
            shutil.copy(out_path, self.output_base / const.TICKER_DIR_NAME)
        
        # 4. Triggers
        if len(df) > 1:
            prev_row = df.iloc[-2]
            if latest["Volume"] < prev_row["Volume"] and latest["Spread"] > prev_row["Spread"]:
                 shutil.copy(out_path, self.output_base / const.TRIGGERS_DIR_NAME)
