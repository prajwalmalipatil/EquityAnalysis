"""
file_loader.py
Robust NSE CSV loader and cleanser.
"""

import re
from pathlib import Path
import pandas as pd
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("vsa-file-loader")


class VSAFileLoader:
    """Handles loading and cleaning of stock historical data CSV files."""

    @staticmethod
    def load_and_clean(path: Path) -> pd.DataFrame:
        """Robust NSE CSV loader and cleanser with column mapping and chronological sort."""
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
            df.columns = [
                re.sub(r"[\s_]+", "_", str(c).strip().strip('"').strip("'")).lower().strip("_")
                for c in df.columns
            ]

            # Comprehensive mapping for various NSE/Common formats
            col_map = {
                "open": "Open",
                "open_price": "Open",
                "high": "High",
                "high_price": "High",
                "low": "Low",
                "low_price": "Low",
                "close": "Close",
                "close_price": "Close",
                "volume": "Volume",
                "total_traded_quantity": "Volume",
                "qty": "Volume",
                "tottrdqty": "Volume",
                "trdqty": "Volume",
                "date": "Date",
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

            for col in required:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", "").str.strip(),
                    errors="coerce",
                )

            # CRITICAL: Handle NSE Latest-First format by ensuring chronological sort
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
                df = df.dropna(subset=["Date", "Close"])  # Ensure we have valid temporal data
                df = df.sort_values("Date").reset_index(drop=True)

            return df.dropna(subset=required).reset_index(drop=True)
        except (OSError, ValueError, KeyError) as e:
            logger.error(f"LOAD_FAILED: {path.name} - {str(e)}")
            return pd.DataFrame()
