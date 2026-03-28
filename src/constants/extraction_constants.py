"""
extraction_constants.py
Centralized constants for the NSE Extraction Client.
"""

from pathlib import Path

# API Endpoint Constants
NSE_HOME_URL = "https://www.nseindia.com/"

# Request Headers
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Resilience Settings
MAX_WORKERS = 3
MAX_RETRIES = 5
BACKOFF_FACTOR = 3.0
JITTER_MIN = 0.8
JITTER_MAX = 2.0
MIN_DELAY = 1.0
MAX_DELAY = 3.0

# File Paths
DEFAULT_OUTPUT_DIR = Path("equity_data")
