"""
vsa_models.py
Domain models and Data Transfer Objects for the VSA classification system.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass
class AnomalyClassification:
    """Classification of an anomaly based on OHLC conditions."""
    pattern_name: str
    sentiment: str  # 'Bullish', 'Bearish', 'Neutral'
    confidence: float = 0.70
    description: str = ""

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
class EigenClassification:
    """Classifies a stock's volume-amplitude OHLC divergence eigenstate."""
    symbol: str
    gap_direction: str       # "Gap-Up" or "Gap-Down"
    close_band: str          # "Weak" (≤0.30) or "Strong" (≥0.70)
    label: str               # One of the 4 convergence/divergence labels
    sentiment: str           # "Bullish" or "Bearish"
    volume_surge_pct: float  # (T_vol - T1_vol) / T1_vol * 100
    t_close_position: float  # Today's close position (0-1)
    t1_close_position: float # Yesterday's close position (0-1)
    delta_cp: float          # t_close_position - t1_close_position
    t_open: float = 0.0
    t_close: float = 0.0
    t_spread: float = 0.0
    t_volume: int = 0
    t1_volume: int = 0
    t1_close: float = 0.0

