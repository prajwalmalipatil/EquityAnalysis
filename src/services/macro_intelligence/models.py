from dataclasses import dataclass, asdict
from typing import List, Optional, Any, Dict

@dataclass
class EigenCorrelation:
    timeframe: str        # "Daily" | "Weekly" | "Monthly"
    signal: str           # "Bullish" | "Bearish" | "Neutral"
    coincidence: str      # "Aligned" | "Divergent" | "None"

    @classmethod
    def from_dict(cls, d: dict) -> 'EigenCorrelation':
        return cls(**d)

@dataclass
class HistoricalAnalog:
    date: str
    event_summary: str
    nifty_reaction: str        # e.g. "+1.8% over 5 days"
    bank_nifty_reaction: str
    notes: str

    @classmethod
    def from_dict(cls, d: dict) -> 'HistoricalAnalog':
        return cls(**d)

@dataclass
class ImpactAnalysis:
    asset_classes: List[str]  # ["Equity", "Bonds", "Currency"]
    sectors: List[str]        # ["Banking", "NBFC", "Realty", ...]
    horizon: str              # "Intraday" | "1–5 days" | "Weekly" | "Monthly" | "Long-term"
    direction: str            # "Positive" | "Negative" | "Neutral" | "Mixed"
    historical_analogs: List[HistoricalAnalog]
    eigen_correlation: Optional[EigenCorrelation] = None

    @classmethod
    def from_dict(cls, d: dict) -> 'ImpactAnalysis':
        analogs = [HistoricalAnalog.from_dict(a) for a in d.get("historical_analogs", [])]
        corr_dict = d.get("eigen_correlation")
        corr = EigenCorrelation.from_dict(corr_dict) if corr_dict else None
        return cls(
            asset_classes=d.get("asset_classes", []),
            sectors=d.get("sectors", []),
            horizon=d.get("horizon", ""),
            direction=d.get("direction", ""),
            historical_analogs=analogs,
            eigen_correlation=corr
        )

@dataclass
class MacroEvent:
    event_id: str        # dedup key component 1 — from RSS guid or generated
    url: str             # dedup key component 2
    published_at: str    # dedup key component 3 — ISO 8601
    title: str
    summary: str         # raw text from RSS or scraped excerpt
    category: str        # "Monetary Policy" | "Regulatory" | "Notification" | etc.
    source: str          # "RBI" (extensible: "SEBI", "FED")
    collected_at: str    # pipeline run timestamp

    # Populated by RuleBasedImpactEngine
    impact: Optional[ImpactAnalysis] = None

    # Populated by EventEnrichmentService (Phase 2 — LLM)
    ai_summary: Optional[str] = None
    ai_key_points: Optional[List[str]] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'MacroEvent':
        impact_dict = d.get("impact")
        impact = ImpactAnalysis.from_dict(impact_dict) if impact_dict else None
        return cls(
            event_id=d.get("event_id", ""),
            url=d.get("url", ""),
            published_at=d.get("published_at", ""),
            title=d.get("title", ""),
            summary=d.get("summary", ""),
            category=d.get("category", ""),
            source=d.get("source", ""),
            collected_at=d.get("collected_at", ""),
            impact=impact,
            ai_summary=d.get("ai_summary"),
            ai_key_points=d.get("ai_key_points")
        )
