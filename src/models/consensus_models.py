"""
consensus_models.py
Domain models for the Multi-Timeframe Consensus Engine.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class ConsensusRating:
    """Represents the synthesized cross-timeframe rating for a stock."""
    symbol: str
    daily_sentiment: str        # "Bullish", "Bearish", or "None"
    weekly_sentiment: str       # "Bullish", "Bearish", or "None"
    monthly_sentiment: str      # "Bullish", "Bearish", or "None"
    score_percentage: float     # -100.0 to +100.0
    star_rating: int            # 1 to 5
    consensus_label: str        # e.g., "Strong Buy", "Cautious Bullish", etc.
    
    # Extra debug/metadata fields
    monthly_score: float = 0.0
    weekly_score: float = 0.0
    daily_score: float = 0.0
