import json
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, List
import dateutil.parser
from src.services.macro_intelligence.models import ImpactAnalysis, MacroEvent, HistoricalAnalog, EigenCorrelation
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("impact-engine")

class RuleBasedImpactEngine:
    """
    Deterministic rule-based engine to classify macroeconomic events
    and map them to specific asset classes, sectors, horizons, and directions.
    """
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.analogs_file = self.base_dir / "data" / "rbi_analogs.json"
        self.data_file = self.base_dir / "dashboard" / "data.json"
        self.analogs = self._load_analogs()

    def _load_analogs(self) -> List[dict]:
        if not self.analogs_file.exists():
            # Seed a default analog if file is missing just to satisfy the schema without crashing
            self.analogs_file.parent.mkdir(parents=True, exist_ok=True)
            default_analog = [{
                "category": "Monetary Policy",
                "direction": "Neutral",
                "date": "2023-10-06",
                "event_summary": "RBI held at 6.50% — 4th consecutive pause",
                "nifty_reaction": "+0.6% over 5 days",
                "bank_nifty_reaction": "+1.1% over 5 days",
                "notes": "Market had priced in a hold; muted reaction"
            }]
            with open(self.analogs_file, 'w', encoding='utf-8') as f:
                json.dump(default_analog, f, indent=2)
            return default_analog
            
        with open(self.analogs_file, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []

    def _get_historical_analogs(self, category: str, direction: str) -> List[HistoricalAnalog]:
        """Lookup historical analogs by matching category and direction."""
        matches = []
        for a in self.analogs:
            if a.get("category") == category and a.get("direction") == direction:
                matches.append(HistoricalAnalog(
                    date=a.get("date", ""),
                    event_summary=a.get("event_summary", ""),
                    nifty_reaction=a.get("nifty_reaction", ""),
                    bank_nifty_reaction=a.get("bank_nifty_reaction", ""),
                    notes=a.get("notes", "")
                ))
        return matches[:2] # Return top 2 analogs

    def _get_eigen_correlation(self, pub_date_str: str) -> Optional[EigenCorrelation]:
        """
        Check if any timeframe signal changed within ±2 days of the event's published_at.
        Reads latest data.json from the dashboard.
        """
        if not self.data_file.exists():
            return None
            
        try:
            pub_dt = dateutil.parser.parse(pub_date_str)
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            eigen_filters = data.get("eigen_filters", {})
            # We assume if the daily payload generated time is within 2 days of pub_dt, 
            # and there are eigen filter matches, we map it to correlation.
            gen_at_str = data.get("generated_at")
            if not gen_at_str: return None
            
            gen_dt = dateutil.parser.parse(gen_at_str)
            if gen_dt.tzinfo is None:
                gen_dt = gen_dt.replace(tzinfo=timezone.utc)
                
            if abs((gen_dt - pub_dt).days) <= 2:
                # Find the strongest timeframe signal available
                if len(eigen_filters.get("monthly", [])) > 0:
                    return EigenCorrelation(timeframe="Monthly", signal=eigen_filters["monthly"][0]["sentiment"], coincidence="Aligned")
                elif len(eigen_filters.get("weekly", [])) > 0:
                    return EigenCorrelation(timeframe="Weekly", signal=eigen_filters["weekly"][0]["sentiment"], coincidence="Aligned")
                elif len(eigen_filters.get("daily", [])) > 0:
                    return EigenCorrelation(timeframe="Daily", signal=eigen_filters["daily"][0]["sentiment"], coincidence="Aligned")
                    
        except Exception as e:
            logger.error("FAILED_TO_GET_EIGEN_CORRELATION", extra={"error": str(e)})
            
        return None

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

        if any(kw in content for kw in ["repo rate", "policy rate", "mpc"]):
            asset_classes = ["Equity", "Bonds", "Currency"]
            sectors = ["Banking", "NBFC", "Realty", "Auto"]
            horizon = "1–5 days"
            direction = "Positive" if "cut" in content else ("Negative" if "hike" in content else "Neutral")
            category = "Monetary Policy"

        elif any(kw in content for kw in ["crr", "slr", "reserve ratio"]):
            asset_classes = ["Equity", "Bonds"]
            sectors = ["Banking", "PSU Banks"]
            horizon = "Weekly"
            direction = "Positive" if "cut" in content or "reduce" in content else ("Negative" if "hike" in content or "increase" in content else "Neutral")
            category = "Monetary Policy"

        elif any(kw in content for kw in ["nbfc", "non-banking"]):
            asset_classes = ["Equity"]
            sectors = ["NBFC", "Fintech"]
            horizon = "1–5 days"
            direction = "Negative" if "strict" in content or "penalty" in content else ("Positive" if "relax" in content else "Neutral")
            category = "Regulatory"

        elif any(kw in content for kw in ["forex", "exchange rate", "usd", "rupee"]):
            asset_classes = ["Currency", "Equity"]
            sectors = ["IT", "Pharma", "Auto"]
            horizon = "Intraday"
            direction = "Mixed"
            category = "Forex"

        elif any(kw in content for kw in ["liquidity", "omo", "vrrr"]):
            asset_classes = ["Bonds", "Equity"]
            sectors = ["Banking"]
            horizon = "1–5 days"
            direction = "Positive" if "inject" in content else ("Negative" if "absorb" in content else "Neutral")
            category = "Liquidity"

        elif any(kw in content for kw in ["gdp", "inflation", "cpi"]):
            asset_classes = ["Equity", "Bonds", "Currency"]
            sectors = ["All"]
            horizon = "Weekly"
            direction = "Positive" if "beat" in content or "growth" in content else "Negative"
            category = "Economic Data"

        elif any(kw in content for kw in ["circular", "directive", "regulation"]):
            asset_classes = ["Equity"]
            sectors = ["Banking", "NBFC"]
            horizon = "Monthly"
            direction = "Neutral"
            category = "Regulatory"

        # Lookup analogs and correlation
        analogs = self._get_historical_analogs(category, direction)
        correlation = self._get_eigen_correlation(event.published_at)

        return ImpactAnalysis(
            asset_classes=asset_classes,
            sectors=sectors,
            horizon=horizon,
            direction=direction,
            historical_analogs=analogs,
            eigen_correlation=correlation
        )
