"""
email_constants.py
Constants related to the automated email reporting system.
"""

# Default thresholds
DEFAULT_PORT = 465
SMTP_SERVER = "smtp.gmail.com"

# Sender and Recipient Configuration
# Note: Fallbacks if environment variables are missing
DEFAULT_RECIPIENT = "Prajwalmalipatil@gmail.com"

# Reporting thresholds for Outlook phrasing
TRENDING_THRESHOLD_HIGH = 25
TRENDING_THRESHOLD_MODERATE = 10
TICKER_THRESHOLD_HIGH = 5

# Styling Colors
COLOR_BULLISH = "#38a169"
COLOR_BEARISH = "#e53e3e"
COLOR_NEUTRAL = "#718096"
COLOR_BACKGROUND = "#f4f7f6"
COLOR_BORDER = "#edf2f7"
COLOR_TEXT = "#2d3748"
