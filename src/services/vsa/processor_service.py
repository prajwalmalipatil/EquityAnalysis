"""
processor_service.py
Orchestration service for VSA Analysis.
Coordinates data cleaning, indicator calculation, pattern matching, and file persistence.
"""

import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

from .indicators import (
    calculate_spread, calculate_close_position, 
    calculate_moving_average, calculate_price_trend, detect_bar_types
)
from .pattern_matcher import VSAClassicMatcher, AnomalyV2Matcher
from .formatters import ExcelFormatter
from src.constants import vsa_constants as const
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("vsa-processor-service")

class VSAProcessorService:
    """
    Service to process raw equity CSVs into analyzed VSA Excels.
    Uses pure indicator functions and pattern matchers for decoupling.
    """
    
    def __init__(self, output_base: Path):
        self.output_base = output_base
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Creates the necessary output structure."""
        dirs = [
            const.RESULTS_DIR_NAME, const.LOGS_DIR_NAME, 
            const.TRENDING_DIR_NAME, const.EFFORTS_DIR_NAME,
            const.ANOMALY_DIR_NAME
        ]
        for d in dirs:
            (self.output_base / d).mkdir(parents=True, exist_ok=True)

    def process_file(self, file_path: Path) -> bool:
        """Processes a single CSV file through the VSA pipeline."""
        try:
            # 1. Load and Clean
            df = self._load_and_clean(file_path)
            if df.empty:
                return False
                
            # 2. Enrich with Indicators
            df = self._enrich_indicators(df)
            
            # 3. Apply VSA Signals
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
        Standardizes CSV columns and validates OHLC data.
        Handles commas in numeric strings and case-insensitive column variations.
        """
        try:
            df = pd.read_csv(path)
            if df.empty:
                logger.warning("EMPTY_CSV_FILE", extra={"file": str(path)})
                return pd.DataFrame()
                
            # Normalize column names (Lower, Strip, Underscore)
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(".", "")
            
            # Map variations to canonical OHLCV
            col_map = {
                "open": "Open", "high": "High", "low": "Low", "close": "Close", 
                "volume": "Volume", "qty": "Volume", "date": "Date"
            }
            # Handle some NSE specific variations
            rename_dict = {}
            for col in df.columns:
                for k, v in col_map.items():
                    if k in col:
                        rename_dict[col] = v
                        break
            
            df = df.rename(columns=rename_dict)
            
            # Validation Logic
            required = ["Open", "High", "Low", "Close", "Volume"]
            if not all(col in df.columns for col in required):
                missing = [c for c in required if c not in df.columns]
                logger.warning("MISSING_COLUMNS", extra={"file": str(path), "missing": missing})
                return pd.DataFrame()
            
            # Date Parsing
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
                df = df.dropna(subset=["Date"])
                df["Date_Str"] = df["Date"].dt.strftime("%Y-%m-%d")
            
            # Numeric cleanup (Remove commas, then to_numeric)
            for col in required:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", "").str.strip(), 
                    errors='coerce'
                )
                
            return df.dropna(subset=required).reset_index(drop=True)
        except Exception as e:
            logger.error("LOAD_AND_CLEAN_FAILED", extra={"file": str(path), "error": str(e)})
            return pd.DataFrame()

    def _enrich_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates VSA indicators using vectorized functions from indicators.py."""
        high, low, close = df["High"].values, df["Low"].values, df["Close"].values
        vol = df["Volume"].values
        
        df["Spread"] = calculate_spread(high, low)
        df["Close_Position"] = calculate_close_position(close, low, df["Spread"].values)
        
        # MAs
        df["Volume_MA"] = calculate_moving_average(vol, 20)
        df["Spread_MA"] = calculate_moving_average(df["Spread"].values, 20)
        df["Close_MA"] = calculate_moving_average(close, 20)
        
        # Trends and Bars
        df["Price_Trend"] = calculate_price_trend(close, df["Close_MA"].values)
        df["IsUpBar"], df["IsDownBar"], _ = detect_bar_types(df["Open"].values, close, df["Spread"].values)
        
        return df

    def _apply_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Iterates and applies pattern matching logic."""
        signals = []
        
        # Classic VSA - Vectorized application would be better, but loop is safer for refactor consistency
        for i in range(len(df)):
            row = df.iloc[i]
            signal = VSAClassicMatcher.match_climax(
                row["Volume"], row["Volume_MA"], row["Spread"], row["Spread_MA"],
                row["Close_Position"], row["Price_Trend"]
            )
            if not signal:
                signal = VSAClassicMatcher.match_no_demand(
                    row["Volume"], row["Volume_MA"], row["IsUpBar"], row["Price_Trend"]
                )
            
            signals.append(signal or "No Signal")
            
        df["Signal_Type"] = signals
        return df

    def _save_results(self, df: pd.DataFrame, symbol: str):
        """Saves as Excel and applies formatting."""
        res_dir = self.output_base / const.RESULTS_DIR_NAME
        out_path = res_dir / f"{symbol}_VSA.xlsx"
        
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name="VSA_Analysis", index=False)
            
        # Formatting
        from openpyxl import load_workbook
        wb = load_workbook(out_path)
        ExcelFormatter.apply_standard_styling(wb, "VSA_Analysis")
        ExcelFormatter.add_vsa_legend(wb)
        wb.save(out_path)
