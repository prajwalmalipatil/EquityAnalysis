"""
processor_service.py
Refactored VSA Processing Engine.
Handles CSV loading, indicator calculation, and pattern matching.
"""

import pandas as pd
from pathlib import Path
import shutil
from typing import List, Dict, Optional
import concurrent.futures

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
    """
    
    def __init__(self, output_base: Path):
        self.output_base = output_base
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Creates necessary subdirectories if they don't exist."""
        dirs = [
            const.RESULTS_DIR_NAME, const.TRENDING_DIR_NAME, 
            const.ANOMALY_DIR_NAME, const.TICKER_DIR_NAME,
            const.TRIGGERS_DIR_NAME
        ]
        for d in dirs:
            (self.output_base / d).mkdir(parents=True, exist_ok=True)

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
        Standardizes CSV columns and validates OHLC data.
        Handles BOM, trailing spaces, commas in numeric strings, and duplicate headers.
        """
        try:
            # 1. Handle BOM and initial load
            df = pd.read_csv(path, encoding="utf-8-sig")
            if df.empty:
                logger.warning("EMPTY_CSV_FILE", extra={"file": str(path)})
                return pd.DataFrame()
            
            # 2. Aggressive Column Normalization
            # Strip spaces, lowercase, and remove special chars
            df.columns = [
                str(c).strip().lower().replace(" ", "_").replace(".", "") 
                for c in df.columns
            ]
            df.columns = [
                c.replace("(", "").replace(")", "").replace("%", "pct").replace("₹", "rs") 
                for c in df.columns
            ]
            
            # 3. Exhaustive Mapping logic
            col_map = {
                "open": "Open", "open_price": "Open",
                "high": "High", "high_price": "High",
                "low": "Low", "low_price": "Low",
                "close": "Close", "close_price": "Close", "last_price": "Close",
                "volume": "Volume", "qty": "Volume", "quantity": "Volume",
                "tottrdqty": "Volume", "total_traded_quantity": "Volume",
                "date": "Date", "timestamp": "Date"
            }
            
            rename_dict = {col: col_map[col] for col in df.columns if col in col_map}
            df = df.rename(columns=rename_dict)
            
            # 4. Critical Step: Deduplicate Columns (Handle 'Last Price' vs 'Close Price' mapping)
            df = df.loc[:, ~df.columns.duplicated()].copy()
            
            # 5. Validation & Fuzzy Fallback
            required = ["Open", "High", "Low", "Close", "Volume"]
            missing = [col for col in required if col not in df.columns]
            
            if missing:
                fuzzy_map = {
                    "Open": ["open"], "High": ["high"], "Low": ["low"], "Close": ["close"],
                    "Volume": ["vol", "qty", "quantity", "tottrqty", "traded"]
                }
                for canon in missing:
                    keywords = fuzzy_map.get(canon, [])
                    for col in df.columns:
                        if any(k in col for k in keywords):
                            df = df.rename(columns={col: canon})
                            break
                # Re-deduplicate after fuzzy
                df = df.loc[:, ~df.columns.duplicated()].copy()
                missing = [col for col in required if col not in df.columns]

            if missing:
                logger.warning("MISSING_COLUMNS", extra={
                    "file": str(path), "missing": missing, "found": list(df.columns)
                })
                return pd.DataFrame()
            
            # 6. Numeric Cleanup (Vectorized comma removal and numeric coercion)
            for col in required:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", "").str.strip(), 
                    errors='coerce'
                )
            
            # 7. Date Handling
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors='coerce', dayfirst=True)
                df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
            
            # Drop any remaining NAs in OHLCV
            df = df.dropna(subset=required).reset_index(drop=True)
            
            return df
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
        
        # Metrics for Trigger Logic
        df["Prev_Volume"] = df["Volume"].shift(1).fillna(df["Volume"])
        df["Prev_Spread"] = df["Spread"].shift(1).fillna(df["Spread"])
        
        # Percentage changes for the report
        df["Vol_Pct"] = ((df["Volume"] - df["Prev_Volume"]) / df["Prev_Volume"]) * 100
        df["Spr_Pct"] = ((df["Spread"] - df["Prev_Spread"]) / df["Prev_Spread"]) * 100
        
        return df

    def _apply_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Iterates and applies pattern matching logic."""
        signals = []
        anomaly_v2 = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            # 1. Classic VSA
            signal = VSAClassicMatcher.match_climax(
                row["Volume"], row["Volume_MA"], row["Spread"], row["Spread_MA"],
                row["Close_Position"], row["Price_Trend"]
            )
            if not signal:
                signal = VSAClassicMatcher.match_no_demand(
                    row["Volume"], row["Volume_MA"], row["IsUpBar"], row["Price_Trend"]
                )
            
            signals.append(signal or "No Signal")
            
            # 2. Anomaly V2 (Volume Spike/Drop)
            # Need previous row for some matches - simplify for now
            prev_vol = df.iloc[i-1]["Volume"] if i > 0 else row["Volume"]
            v2 = AnomalyV2Matcher.match_volume_spike(row["Volume"], prev_vol)
            if not v2:
                v2 = AnomalyV2Matcher.match_volume_drop(row["Volume"], prev_vol)
            anomaly_v2.append(v2 or "Neutral")
            
        df["Signal_Type"] = signals
        df["Anomaly_V2"] = anomaly_v2
        return df

    def _save_results(self, df: pd.DataFrame, symbol: str):
        """Saves as Excel, applies formatting, and distributes to signal folders."""
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
        
        # Distribution based on last row signal
        if df.empty:
            return
            
        latest = df.iloc[-1]
        
        # 1. Trending (Any Classic VSA Signal)
        if latest["Signal_Type"] != "No Signal":
            shutil.copy(out_path, self.output_base / const.TRENDING_DIR_NAME)
            
        # 2. Anomaly (Volume Spike/Drop)
        if "Spike" in str(latest["Anomaly_V2"]) or "Drop" in str(latest["Anomaly_V2"]):
            shutil.copy(out_path, self.output_base / const.ANOMALY_DIR_NAME)
            
        # 3. Ticker (All results)
        shutil.copy(out_path, self.output_base / const.TICKER_DIR_NAME)
        
        # 4. Triggers (Volume decreased from previous day while Spread expanded)
        prev_vol = df.iloc[-2]["Volume"] if len(df) > 1 else latest["Volume"]
        prev_spr = df.iloc[-2]["Spread"] if len(df) > 1 else latest["Spread"]
        if latest["Volume"] < prev_vol and latest["Spread"] > prev_spr:
             shutil.copy(out_path, self.output_base / const.TRIGGERS_DIR_NAME)
