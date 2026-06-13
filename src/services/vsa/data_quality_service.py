import pandas as pd
import numpy as np
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("data-quality")

class DataQualityService:
    """
    Validates raw OHLCV CSV files before they are permitted to enter the analytical pipeline.
    Invalid files are moved to a quarantine directory.
    """
    REQUIRED_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.quarantine_dir = self.base_dir / "quarantine"
        self.stats = {
            "total_files": 0,
            "passed": 0,
            "quarantined": 0,
            "reasons": {}
        }

    def _record_quarantine(self, file_path: Path, reason: str):
        self.stats["quarantined"] += 1
        self.stats["reasons"][reason] = self.stats["reasons"].get(reason, 0) + 1
        logger.warning(f"QUARANTINED [{file_path.name}]: {reason}")
        
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(file_path), str(self.quarantine_dir / file_path.name))
        except Exception as e:
            logger.error(f"FAILED_TO_QUARANTINE {file_path.name}: {e}")

    def validate_file(self, file_path: Path) -> bool:
        """Returns True if valid, False if quarantined."""
        try:
            # We don't want to parse dates immediately if it's huge, but for validation we must.
            df = pd.read_csv(file_path)
        except Exception as e:
            self._record_quarantine(file_path, f"CSV_PARSE_ERROR: {str(e)}")
            return False

        # 1. Column Check
        missing_cols = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            self._record_quarantine(file_path, f"MISSING_COLUMNS: {missing_cols}")
            return False

        # 2. NaN Check
        if df[self.REQUIRED_COLUMNS].isnull().values.any():
            self._record_quarantine(file_path, "CONTAINS_NANS_IN_OHLCV")
            return False

        # 3. Duplicate Dates
        if df['Date'].duplicated().any():
            self._record_quarantine(file_path, "DUPLICATE_DATES")
            return False

        # 4. OHLC Logic
        invalid_ohlc = df[
            (df['High'] < df['Low']) | 
            (df['Open'] <= 0) | 
            (df['High'] <= 0) | 
            (df['Low'] <= 0) | 
            (df['Close'] <= 0)
        ]
        if not invalid_ohlc.empty:
            self._record_quarantine(file_path, "INVALID_OHLC_PRICES")
            return False

        # 5. Volume Check (Entire file cannot be 0 volume)
        if df['Volume'].sum() == 0 and len(df) > 0:
            self._record_quarantine(file_path, "ZERO_TOTAL_VOLUME")
            return False
            
        # 6. Minimum rows (Requires at least some history for eigen filters to work)
        if len(df) < 5:
            self._record_quarantine(file_path, "INSUFFICIENT_DATA")
            return False

        return True

    def run_gate(self) -> Dict[str, int]:
        logger.info("Starting Data Quality Gate...")
        csv_files = list(self.base_dir.glob("*.csv"))
        self.stats["total_files"] = len(csv_files)

        for file_path in csv_files:
            if file_path.parent.name == "quarantine":
                continue
                
            if self.validate_file(file_path):
                self.stats["passed"] += 1

        logger.info(f"Data Quality Gate complete. Passed: {self.stats['passed']}/{self.stats['total_files']}")
        return self.stats

from src.services.orchestration.registry import platform_registry, ResearchModule
platform_registry.register(ResearchModule(
    name="DataQualityGate",
    version="1.0.0",
    description="Validates raw OHLCV CSV files before they enter the analytical pipeline.",
    inputs=["CSV"],
    outputs=["CleanCSV"],
    dependencies=[]
))
