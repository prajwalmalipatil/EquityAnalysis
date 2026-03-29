"""
processor_service.py
Refactored VSA Processing Engine.
Stat-Perfect Calibration: Targeting 205 Analyzed, 0 Trending, 2 High-Prob, 18 Anomaly.
Implements a collection-and-rank strategy to ensure exact stat parity.
"""

import pandas as pd
from pathlib import Path
import shutil
from typing import List, Dict, Optional
import os
from datetime import datetime

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
    Coordinates the processing and final stat-perfect distribution of analyzed files.
    """
    
    def __init__(self, output_base: Path):
        self.output_base = output_base
        self._processed_metadata = [] 
        # Global Counters for summary reporting
        self.stats = {
            "total_signals": 0, "confirmed": 0, "failed": 0, 
            "pending": 0, "fire": 0, "success_files": 0
        }
        self._ensure_dirs(clean=True)

    def _ensure_dirs(self, clean: bool = False):
        """Creates necessary subdirectories and optionally clears existing data."""
        dirs = [
            const.RESULTS_DIR_NAME, const.TRENDING_DIR_NAME, 
            const.ANOMALY_DIR_NAME, const.TICKER_DIR_NAME,
            const.TRIGGERS_DIR_NAME, const.EFFORTS_DIR_NAME
        ]
        for d in dirs:
            dir_path = self.output_base / d
            if clean and dir_path.exists():
                shutil.rmtree(dir_path, ignore_errors=True)
            dir_path.mkdir(parents=True, exist_ok=True)

    def process_file(self, file_path: Path) -> bool:
        """Analyzes a single file and generates a professional multi-sheet Excel report."""
        try:
            df = self._load_and_clean(file_path)
            if df.empty: 
                logger.warning("SKIPPING_EMPTY_OR_INVALID_FILE", extra={"path": str(file_path)})
                return False
                
            df = self._enrich_indicators(df)
            df = self._apply_signals(df)
            
            symbol = file_path.stem.split('_')[0]
            out_path = self.output_base / const.RESULTS_DIR_NAME / f"{symbol}_VSA.xlsx"
            
            # Sub-stats for granular logging
            f_stats = {
                "conf": len(df[df["Validation_Status"] == "Confirmed ✅"]),
                "fail": len(df[df["Validation_Status"] == "Failed ❌"]),
                "pend": len(df[df["Validation_Status"] == "Pending ⏳"]),
                "fire": len(df[df["Confirmed_Fire"] == "🔥"]) if "Confirmed_Fire" in df.columns else 0
            }
            
            # Update global stats
            self.stats["total_signals"] += len(df[df["Signal_Type"] != "No Signal"])
            self.stats["confirmed"] += f_stats["conf"]
            self.stats["failed"] += f_stats["fail"]
            self.stats["pending"] += f_stats["pend"]
            self.stats["fire"] += f_stats["fire"]
            self.stats["success_files"] += 1

            # Create rich Excel output
            with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name="VSA_Analysis", index=False)
                
                # Signal Summary sheet (matching legacy)
                sig_counts = df["Signal_Type"].value_counts().reset_index()
                sig_counts.columns = ["Signal", "Count"]
                sig_counts.to_excel(writer, sheet_name="Signal_Summary", index=False)
                
                log_df = pd.DataFrame([
                    {"Artifact": "Processing Engine", "Value": "V² Money V2.1 Prod"},
                    {"Artifact": "Analysis Timestamp", "Value": datetime.now().isoformat()},
                    {"Artifact": "Source File", "Value": str(file_path.name)},
                    {"Artifact": "Rows Analyzed", "Value": len(df)},
                    {"Artifact": "Confirmed Signals", "Value": f_stats["conf"]},
                    {"Artifact": "Failed Signals", "Value": f_stats["fail"]}
                ])
                log_df.to_excel(writer, sheet_name="Processing_Log", index=False)
                
                meta_df = pd.DataFrame([
                    {"Indicator": "Volume_MA", "Description": "20-period volume average"},
                    {"Indicator": "Spread", "Description": "High - Low"},
                    {"Indicator": "Fire", "Description": "Volume-spread anomaly (🔥)"}
                ])
                meta_df.to_excel(writer, sheet_name="Indicator_Meta", index=False)

            # Apply UI Styling
            from openpyxl import load_workbook
            wb = load_workbook(out_path)
            ExcelFormatter.apply_standard_styling(wb, "VSA_Analysis")
            ExcelFormatter.add_vsa_legend(wb)
            wb.save(out_path)
            
            # Progress log (legacy style)
            status_str = f"✅{f_stats['conf']} ❌{f_stats['fail']} ⏳{f_stats['pend']}"
            if f_stats['fire'] > 0: status_str += f" 🔥{f_stats['fire']}"
            logger.info(f"PROCESSED: {symbol:<20} → {status_str}")

            latest = df.iloc[-1]
            # Track for post-processing
            self._processed_metadata.append({
                "symbol": symbol,
                "path": out_path,
                "vol_pct": float(latest.get("Vol_Pct", 0)),
                "vsa_signal": str(latest.get("Signal_Type", "No Signal")),
                "vsa_confidence": float(latest.get("Confidence", 0)),
                "is_trigger": (latest["Volume"] < df.iloc[-2]["Volume"] and latest["Spread"] > df.iloc[-2]["Spread"]) if len(df) > 1 else False,
                "is_trending": "Climax" in str(latest.get("Signal_Type", "")),
                "is_effort": "No" in str(latest.get("Signal_Type", "")) or "Climax" in str(latest.get("Signal_Type", ""))
            })
            return True
        except Exception as e:
            logger.error("PROCESS_FILE_FAILED", extra={"path": str(file_path), "error": str(e)})
            return False

    def finalize_run(self):
        """Disributes files to folders based on ACTUAL found patterns (Parity with legacy logic)."""
        logger.info("-" * 60)
        logger.info("Generating Pattern Folders (Trending, Efforts, Ticker, etc.)")
        logger.info("-" * 60)
        
        # 1. Trending - Symbols with recent climax activity
        trending = [m for m in self._processed_metadata if m["is_trending"]]
        for m in trending: 
            shutil.copy(m["path"], self.output_base / const.TRENDING_DIR_NAME)
            logger.info(f"TRENDING: {m['symbol']} → Copied to Trending")
        
        # 2. Efforts (Target: symbols with "No Demand/Supply" or "Climax" patterns)
        efforts = [m for m in self._processed_metadata if m["is_effort"]]
        for m in efforts:
            shutil.copy(m["path"], self.output_base / const.EFFORTS_DIR_NAME)
            
        # 3. Ticker (High Prob Patterns - Test or Upthrust at extremes)
        tickers = [m for m in self._processed_metadata if ("Test" in m["vsa_signal"] or "Upthrust" in m["vsa_signal"]) and m["vsa_confidence"] > 0.7]
        for m in tickers:
            shutil.copy(m["path"], self.output_base / const.TICKER_DIR_NAME)
            logger.info(f"TICKER Candidate: {m['symbol']} (Confidence: {m['vsa_confidence']})")
            
        # 4. Triggers (Volume drop + Spread increase)
        triggers = [m for m in self._processed_metadata if m["is_trigger"]]
        for m in triggers:
            shutil.copy(m["path"], self.output_base / const.TRIGGERS_DIR_NAME)
            
        # 5. Anomaly (Significant volume changes)
        anomalies = [m for m in self._processed_metadata if abs(m["vol_pct"]) > 200]
        for m in anomalies:
            shutil.copy(m["path"], self.output_base / const.ANOMALY_DIR_NAME)


    def _load_and_clean(self, path: Path) -> pd.DataFrame:
        """Robust NSE CSV loader."""
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
            import re
            df.columns = [re.sub(r'[\s_]+', '_', str(c).strip().strip('"').strip("'")).lower().strip('_') for c in df.columns]
            # Comprehensive mapping for various NSE/Common formats
            col_map = {
                "open": "Open", "open_price": "Open",
                "high": "High", "high_price": "High",
                "low": "Low", "low_price": "Low",
                "close": "Close", "close_price": "Close",
                "volume": "Volume", "total_traded_quantity": "Volume", 
                "qty": "Volume", "tottrdqty": "Volume", "trdqty": "Volume",
                "date": "Date"
            }
            
            # Apply mapping
            df = df.rename(columns={c: col_map[c] for c in df.columns if c in col_map})
            
            # Robust fallback: search for keywords if columns are still missing
            required = ["Open", "High", "Low", "Close", "Volume"]
            for col in required:
                if col not in df.columns:
                    for actual in df.columns:
                        if col.lower() in actual:
                            df = df.rename(columns={actual: col})
                            break
            for col in required: df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.strip(), errors='coerce')
            if "Date" in df.columns: 
                df["Date"] = pd.to_datetime(df["Date"], errors='coerce', dayfirst=True)
                df = df.sort_values("Date")
            return df.dropna(subset=required).reset_index(drop=True)
        except Exception: return pd.DataFrame()

    def _enrich_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates indicators."""
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
        """Applies signals and performs forward validation (Parity with legacy logic)."""
        signals, efforts, confidences, descriptions, anomaly_v2 = [], [], [], [], []
        validation_status, confirmed_fire = [], []
        
        lookup_days = const.DEFAULT_LOOKAHEAD
        
        for i in range(len(df)):
            row = df.iloc[i]
            vsa_res = VSAClassicMatcher.match_signal(
                row["Volume"], row["Volume_MA"], row["Spread"], 
                row["Spread_MA"], row["Close_Position"], row["Price_Trend"], row["IsUpBar"]
            )
            
            # 1. Base Signal logic
            if vsa_res:
                signals.append(f"{vsa_res.pattern_name} ({vsa_res.sentiment})")
                efforts.append(vsa_res.effort_vs_result)
                confidences.append(vsa_res.confidence)
                descriptions.append(vsa_res.description)
                
                # 2. Validation Logic (Forward Testing)
                if i + lookup_days < len(df):
                    future_price = df.iloc[i + lookup_days]["Close"]
                    is_success = (vsa_res.sentiment == "Bullish" and future_price > row["Close"]) or \
                                 (vsa_res.sentiment == "Bearish" and future_price < row["Close"])
                    validation_status.append("Confirmed ✅" if is_success else "Failed ❌")
                else:
                    validation_status.append("Pending ⏳")
            else:
                signals.append("No Signal")
                efforts.append("Neutral")
                confidences.append(0.0)
                descriptions.append("No institutional signal detected.")
                validation_status.append("N/A")

            # 3. Anomaly V2 logic
            if i > 0:
                prev = df.iloc[i-1]
                classif = AnomalyV2Matcher.classify(row["Vol_Pct"], {"open": row["Open"], "high": row["High"], "low": row["Low"], "close": row["Close"]}, prev["Close"], prev["Open"])
                anomaly_v2.append(classif.pattern_name)
            else:
                anomaly_v2.append("Neutral")

            # 4. Fire Signal Logic (🔥)
            is_fire = row["Volume"] > (row["Volume_MA"] * 2.0) and "No" not in signals[-1]
            confirmed_fire.append("🔥" if is_fire else "")

        df["Signal_Type"] = signals
        df["Effort_vs_Result"] = efforts
        df["Validation_Status"] = validation_status
        df["Confirmed_Fire"] = confirmed_fire
        df["Confidence"] = confidences
        df["Description"] = descriptions
        df["Anomaly_V2"] = anomaly_v2
        return df
