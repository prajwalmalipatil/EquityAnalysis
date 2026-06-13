import json
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, List
import dateutil.parser
from src.services.macro_intelligence.models import ImpactAnalysis, MacroEvent, HistoricalAnalog
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("impact-engine")

class RuleBasedImpactEngine:
    """
    Deterministic rule-based engine to classify macroeconomic events
    and map them to specific asset classes, sectors, horizons, and directions.
    """
    
    SECTOR_TO_SECURITIES = {
        "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK"],
        "PSU Banks": ["SBIN", "PNB", "BOB", "CANBK", "UNIONBANK"],
        "NBFC": ["BAJFINANCE", "CHOLAFIN", "M&MFIN", "SHRIRAMFIN"],
        "Realty": ["DLF", "MACROTECH", "GODREJPROP", "OBEROIRLTY"],
        "Auto": ["TATAMOTORS", "M&M", "MARUTI", "BAJAJ-AUTO", "HEROMOTOCO"],
        "IT": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM"],
        "Pharma": ["SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "LUPIN"],
        "Fintech": ["PAYTM", "PBFINTECH"]
    }

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)

    def process(self, event: MacroEvent) -> ImpactAnalysis:
        """Determines impact based on deterministic keyword-to-impact mapping."""
        title = event.title.lower()
        summary = event.summary.lower()
        content = title + " " + summary

        # Default fallback
        asset_classes = ["Equity"]
        sectors = ["All"]
        horizon = "1-5 days"
        direction = "Neutral"
        category = event.category
        severity = "Informational"
        importance = 10
        confidence = 50

        if any(kw in content for kw in ["repo rate", "policy rate", "mpc", "monetary policy committee"]):
            asset_classes = ["Equity", "Bonds", "Currency"]
            sectors = ["Banking", "NBFC", "Realty", "Auto"]
            horizon = "1-5 days"
            direction = "Positive" if "cut" in content else ("Negative" if "hike" in content else "Neutral")
            category = "Monetary Policy"
            severity = "Critical"
            importance = 100
            confidence = 95

        elif any(kw in content for kw in ["crr", "slr", "reserve ratio"]):
            asset_classes = ["Equity", "Bonds"]
            sectors = ["Banking", "PSU Banks"]
            horizon = "Weekly"
            direction = "Positive" if ("cut" in content or "reduce" in content) else ("Negative" if ("hike" in content or "increase" in content) else "Neutral")
            category = "Monetary Policy"
            severity = "High"
            importance = 85
            confidence = 90

        elif any(kw in content for kw in ["nbfc", "non-banking", "housing finance"]):
            asset_classes = ["Equity"]
            sectors = ["NBFC", "Fintech"]
            horizon = "1-5 days"
            direction = "Negative" if ("strict" in content or "penalty" in content or "curb" in content) else ("Positive" if "relax" in content else "Neutral")
            category = "Regulatory"
            severity = "Medium"
            importance = 60
            confidence = 85

        elif any(kw in content for kw in ["forex", "exchange rate", "usd", "rupee"]):
            asset_classes = ["Currency", "Equity"]
            sectors = ["IT", "Pharma", "Auto"]
            horizon = "Intraday"
            direction = "Mixed"
            category = "Forex"
            severity = "Medium"
            importance = 50
            confidence = 80

        elif any(kw in content for kw in ["liquidity", "omo", "vrrr"]):
            asset_classes = ["Bonds", "Equity"]
            sectors = ["Banking"]
            horizon = "1-5 days"
            direction = "Positive" if "inject" in content else ("Negative" if "absorb" in content else "Neutral")
            category = "Liquidity"
            severity = "Medium"
            importance = 65
            confidence = 85

        elif any(kw in content for kw in ["gdp", "inflation", "cpi", "wpi"]):
            asset_classes = ["Equity", "Bonds", "Currency"]
            sectors = ["All"]
            horizon = "Weekly"
            direction = "Positive" if ("beat" in content or "growth" in content or "fall in inflation" in content) else "Negative"
            category = "Economic Data"
            severity = "High"
            importance = 80
            confidence = 90

        elif any(kw in content for kw in ["circular", "directive", "regulation", "guideline"]):
            asset_classes = ["Equity"]
            sectors = ["Banking", "NBFC"]
            horizon = "Monthly"
            direction = "Neutral"
            category = "Regulatory"
            severity = "Low"
            importance = 20
            confidence = 75

        securities = []
        for sec in sectors:
            securities.extend(self.SECTOR_TO_SECURITIES.get(sec, []))
        
        # Deduplicate
        securities = list(set(securities))

        return ImpactAnalysis(
            asset_classes=asset_classes,
            sectors=sectors,
            securities=securities,
            horizon=horizon,
            direction=direction,
            severity=severity,
            importance=importance,
            confidence=confidence
        )
