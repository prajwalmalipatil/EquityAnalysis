"""
vsa_models.py
Domain models and Data Transfer Objects for the VSA classification system.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass
class ProcessingResult:
    """Type-safe result container with integrity checks for inter-process communication."""
    name: str
    skipped: bool
    df_csv: Optional[str] = None
    log_csv: Optional[str] = None
    checksum_df: Optional[str] = None
    checksum_log: Optional[str] = None
    summary: Optional[Dict] = None
    latest_date: Optional[str] = None
    has_confirmed_recent: bool = False
    dropped_rows: int = 0
    reason: Optional[str] = None

@dataclass
class AnomalyClassification:
    """Classification of an anomaly based on OHLC conditions."""
    pattern_name: str
    sentiment: str  # 'Bullish', 'Bearish', 'Neutral'
    win_rate: float
    description: str

@dataclass
class VSAClassification:
    """Detailed VSA pattern classification with effort/result context."""
    pattern_name: str
    effort_vs_result: str
    sentiment: str
    confidence: float
    description: str

@dataclass
class VolumePriceAnomaly:
    """Represents a detected anomaly event."""
    symbol: str
    date: str
    volumes: list[int]
    drop_pct: float
    close_pos: str
    bar_dir: str
    gap_type: str
    classification: Optional[AnomalyClassification] = None
