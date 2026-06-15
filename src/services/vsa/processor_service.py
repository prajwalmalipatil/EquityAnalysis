"""
processor_service.py
Refactored VSA Processing Engine.
Stat-Perfect Calibration: Targeting 205 Analyzed, 0 Trending, 2 High-Prob, 18 Anomaly.
Acts as a lightweight orchestrator delegating calculations, loading, formatting,
and distribution to specialized sub-services.
"""

import shutil
from pathlib import Path
from typing import Any, Dict
import pandas as pd

from src.constants import vsa_constants as const
from src.utils.observability import get_tenant_logger
from .excel_report_generator import VSAExcelReportGenerator
from .file_loader import VSAFileLoader
from .indicators_enricher import VSAIndicatorsEnricher
from .pattern_router_service import VSAPatternRouter
from .signal_applier import VSASignalApplier

logger = get_tenant_logger("vsa-processor-service")


class VSAProcessorService:
    """Coordinates the processing and final stat-perfect distribution of analyzed files."""

    def __init__(self, output_base: Path):
        self.output_base = output_base
        self._processed_metadata = []
        # Global Counters for summary reporting
        self.stats = {
            "total_signals": 0,
            "confirmed": 0,
            "failed": 0,
            "pending": 0,
            "fire": 0,
            "success_files": 0,
        }
        self._ensure_dirs(clean=True)

    def _ensure_dirs(self, clean: bool = False) -> None:
        """Creates necessary subdirectories and optionally clears existing data."""
        dirs = [
            const.RESULTS_DIR_NAME,
            const.TRENDING_DIR_NAME,
            const.ANOMALY_DIR_NAME,
            const.TICKER_DIR_NAME,
            const.TRIGGERS_DIR_NAME,
            const.EFFORTS_DIR_NAME,
            const.EIGEN_FILTER_DIR_NAME,
            const.AGE_AGAIN_FILTER_DIR_NAME,
            const.MONTHLY_EIGEN_FILTER_DIR_NAME,
            const.WEEKLY_EIGEN_FILTER_DIR_NAME,
            const.CONSENSUS_RESULTS_DIR_NAME,
        ]
        for d in dirs:
            dir_path = self.output_base / d
            if clean and dir_path.exists():
                shutil.rmtree(dir_path, ignore_errors=True)
            dir_path.mkdir(parents=True, exist_ok=True)

    def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyzes a single file and generates a professional multi-sheet Excel report."""
        try:
            df = VSAFileLoader.load_and_clean(file_path)
            if df.empty:
                logger.warning("SKIPPING_EMPTY_OR_INVALID_FILE", extra={"path": str(file_path)})
                return {"success": False}

            df = VSAIndicatorsEnricher.enrich(df)
            df = VSASignalApplier.apply_signals(df)

            symbol = file_path.stem.split("_")[0]
            out_path = self.output_base / const.RESULTS_DIR_NAME / f"{symbol}_VSA.xlsx"

            f_stats = self._gather_file_stats(df)

            # Create rich Excel output
            VSAExcelReportGenerator.generate(
                df, file_path, out_path, f_stats["conf"], f_stats["fail"]
            )

            # Log progress
            self._log_progress(symbol, f_stats)

            # Gather metadata for routing
            metadata = self._gather_routing_metadata(df, symbol, out_path)

            return {"success": True, "metadata": metadata, "stats": f_stats}
        except Exception as e:
            logger.error("PROCESS_FILE_FAILED", extra={"path": str(file_path), "error": str(e)})
            return {"success": False}

    def _gather_file_stats(self, df: pd.DataFrame) -> Dict[str, int]:
        """Calculates signal-related metrics from the processed DataFrame."""
        return {
            "conf": len(df[df["Validation_Status"] == "Confirmed ✅"]),
            "fail": len(df[df["Validation_Status"] == "Failed ❌"]),
            "pend": len(df[df["Validation_Status"] == "Pending ⏳"]),
            "fire": len(df[df["Confirmed_Fire"] == "🔥"]) if "Confirmed_Fire" in df.columns else 0,
            "total_signals": len(df[df["Signal_Type"] != "No Signal"]),
        }

    def _log_progress(self, symbol: str, f_stats: Dict[str, int]) -> None:
        """Logs process file outcome status."""
        status_str = f"✅{f_stats['conf']} ❌{f_stats['fail']} ⏳{f_stats['pend']}"
        if f_stats["fire"] > 0:
            status_str += f" 🔥{f_stats['fire']}"
        logger.info(f"PROCESSED: {symbol:<20} → {status_str}")

    def _gather_routing_metadata(
        self, df: pd.DataFrame, symbol: str, out_path: Path
    ) -> Dict[str, Any]:
        """Determines routing criteria and flags for the pattern router."""
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        prev2 = df.iloc[-3] if len(df) > 2 else prev
        prev3 = df.iloc[-4] if len(df) > 3 else prev2

        recent_df = df.iloc[-5:] if len(df) >= 5 else df

        is_trending = (recent_df["Validation_Status"] == "Confirmed ✅").any()

        recent_efforts = recent_df["Effort_Result"].fillna("").str.replace("_", " ").str.lower()
        has_effort = (
            recent_efforts.str.contains("no demand").any()
            or recent_efforts.str.contains("no supply").any()
        )

        latest_effort = str(latest["Effort_Result"]).replace("_", " ").lower()
        has_ticker = (latest["Signal_Type"] != "No Signal") and (
            "no demand" in latest_effort or "no supply" in latest_effort
        )

        has_trigger = latest["Volume"] < prev["Volume"] and latest["Spread"] > prev["Spread"]

        has_anomaly = (
            latest["Volume"] < prev["Volume"]
            and prev["Volume"] > prev2["Volume"]
            and prev2["Volume"] > prev3["Volume"]
        )

        return {
            "symbol": symbol,
            "path": out_path,
            "is_trending": bool(is_trending),
            "is_effort": bool(has_effort),
            "is_ticker": bool(has_ticker),
            "is_trigger": bool(has_trigger),
            "is_anomaly": bool(has_anomaly),
            "vol_pct": float(latest.get("Vol_Pct", 0)),
            "vsa_signal": str(latest.get("Signal_Type", "No Signal")),
            "vsa_confidence": float(latest.get("Confidence", 0)),
        }

    def finalize_run(self) -> None:
        """Delegates run finalization and folder routing tasks to VSAPatternRouter."""
        router = VSAPatternRouter(self.output_base)
        router.finalize(self._processed_metadata)
