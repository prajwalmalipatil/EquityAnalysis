"""
vsa_constants.py
Values for VSA thresholds and calculations.
"""

from pathlib import Path

# Volume thresholds
ULTRA_HIGH_VOLUME = 2.5
HIGH_VOLUME = 1.5
MEDIUM_VOLUME = 1.2
LOW_VOLUME = 0.7

# Spread thresholds
WIDE_SPREAD = 2.0
NORMAL_SPREAD = 1.2
NARROW_SPREAD = 0.8

# Close position thresholds
WEAK_CLOSE = 0.3
STRONG_CLOSE = 0.7
MID_CLOSE_LOW = 0.3
MID_CLOSE_HIGH = 0.7

# Support/Resistance tolerance
SUPPORT_TOLERANCE = 0.05

# Validation parameters
DEFAULT_LOOKAHEAD = 3
TRENDING_DAYS = 5

# Data Boundaries
MAX_PRICE = 1_000_000.0
MIN_PRICE = 0.01
MAX_VOLUME_LIMIT = 1_000_000_000_000
MIN_VOLUME_LIMIT = 0

# Folder configurations
RESULTS_DIR_NAME = "Results"
LOGS_DIR_NAME = "Logs"
TRENDING_DIR_NAME = "Trending"
EFFORTS_DIR_NAME = "Efforts"
TICKER_DIR_NAME = "Ticker"
TRIGGERS_DIR_NAME = "Triggers"
ANOMALY_DIR_NAME = "Anomaly"
EIGEN_FILTER_DIR_NAME = "EigenFilter"
AGE_AGAIN_FILTER_DIR_NAME = "AgeAgainFilter"
MONTHLY_EIGEN_FILTER_DIR_NAME = "MonthlyEigenFilter"
WEEKLY_EIGEN_FILTER_DIR_NAME = "WeeklyEigenFilter"
CONSENSUS_RESULTS_DIR_NAME = "ConsensusResults"

# EigenFilter thresholds
EIGEN_CLOSE_LOWER_BAND = 0.30
EIGEN_CLOSE_UPPER_BAND = 0.70

# Eigen Score Proxy Map
EIGEN_SCORE_MAP = {
    "Bullish": 0.8,
    "Neutral": 0.5,
    "Bearish": 0.2
}

# Standard columns
OHLCV_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

# Eigen Transition Engine (ETE)
ETE_EVENTS_DIR = "data/events"
ETE_SNAPSHOTS_DIR = "data/snapshots"
CHECKPOINT_INTERVAL = 500
ENGINE_VERSION = "1.0"

# Transition Rules
RULE_HVHS = "HVHS"
RULE_LVLS = "LVLS"
RULE_HVLS = "HVLS"
RULE_LVHS = "LVHS"

