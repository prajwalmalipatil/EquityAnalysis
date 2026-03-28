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

# Standard columns
OHLCV_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
