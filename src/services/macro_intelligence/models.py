from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

@dataclass
class PreEventRegime:
    daily_eigen: str          # "Bullish", "Bearish", "Neutral"
    weekly_eigen: str
    monthly_eigen: str
    vix: Optional[float] = None
    breadth_pct: Optional[float] = None
    
    @classmethod
    def from_dict(cls, d: dict) -> 'PreEventRegime':
        return cls(
            daily_eigen=d.get("daily_eigen", "Neutral"),
            weekly_eigen=d.get("weekly_eigen", "Neutral"),
            monthly_eigen=d.get("monthly_eigen", "Neutral"),
            vix=d.get("vix"),
            breadth_pct=d.get("breadth_pct")
        )

@dataclass
class PostEventOutcome:
    daily_eigen: str
    weekly_eigen: str
    monthly_eigen: str
    vix: Optional[float] = None
    breadth_pct: Optional[float] = None

    @classmethod
    def from_dict(cls, d: dict) -> 'PostEventOutcome':
        return cls(
            daily_eigen=d.get("daily_eigen", "Neutral"),
            weekly_eigen=d.get("weekly_eigen", "Neutral"),
            monthly_eigen=d.get("monthly_eigen", "Neutral"),
            vix=d.get("vix"),
            breadth_pct=d.get("breadth_pct")
        )

@dataclass
class ReturnWindow:
    t_minus_5: Optional[float] = None
    t_minus_3: Optional[float] = None
    t_minus_1: Optional[float] = None
    t_plus_1: Optional[float] = None
    t_plus_3: Optional[float] = None
    t_plus_5: Optional[float] = None
    t_plus_10: Optional[float] = None
    t_plus_20: Optional[float] = None
    
    @classmethod
    def from_dict(cls, d: dict) -> 'ReturnWindow':
        return cls(**d)

@dataclass
class EventStudy:
    pre_event_regime: PreEventRegime
    post_event_outcome: Optional[PostEventOutcome] = None
    index_returns: Dict[str, ReturnWindow] = field(default_factory=dict)
    sector_returns: Dict[str, ReturnWindow] = field(default_factory=dict)
    stock_returns: Dict[str, ReturnWindow] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> 'EventStudy':
        pre = PreEventRegime.from_dict(d.get("pre_event_regime", {}))
        post_dict = d.get("post_event_outcome")
        post = PostEventOutcome.from_dict(post_dict) if post_dict else None
        
        idx_ret = {k: ReturnWindow.from_dict(v) for k, v in d.get("index_returns", {}).items()}
        sec_ret = {k: ReturnWindow.from_dict(v) for k, v in d.get("sector_returns", {}).items()}
        stk_ret = {k: ReturnWindow.from_dict(v) for k, v in d.get("stock_returns", {}).items()}
        
        return cls(pre, post, idx_ret, sec_ret, stk_ret)

@dataclass
class HistoricalAnalog:
    date: str
    event_summary: str
    nifty_reaction: str
    bank_nifty_reaction: str
    notes: str

    @classmethod
    def from_dict(cls, d: dict) -> 'HistoricalAnalog':
        return cls(**d)

@dataclass
class ImpactAnalysis:
    asset_classes: List[str]
    sectors: List[str]
    securities: List[str]      # e.g., ["HDFCBANK", "ICICIBANK"]
    horizon: str
    direction: str
    severity: str              # "Critical", "High", "Medium", "Low", "Informational"
    importance: int            # 0 - 100
    confidence: int            # 0 - 100
    historical_analogs: List[HistoricalAnalog] = field(default_factory=list)
    eigen_correlation: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, d: dict) -> 'ImpactAnalysis':
        analogs = [HistoricalAnalog.from_dict(a) for a in d.get("historical_analogs", [])]
        return cls(
            asset_classes=d.get("asset_classes", []),
            sectors=d.get("sectors", []),
            securities=d.get("securities", []),
            horizon=d.get("horizon", ""),
            direction=d.get("direction", ""),
            severity=d.get("severity", "Informational"),
            importance=d.get("importance", 0),
            confidence=d.get("confidence", 0),
            historical_analogs=analogs,
            eigen_correlation=d.get("eigen_correlation")
        )

@dataclass
class AISummarySnapshot:
    version: int
    timestamp: str
    prompt_version: str
    summary: str
    key_points: List[str]

    @classmethod
    def from_dict(cls, d: dict) -> 'AISummarySnapshot':
        return cls(
            version=d.get("version", 1),
            timestamp=d.get("timestamp", ""),
            prompt_version=d.get("prompt_version", ""),
            summary=d.get("summary", ""),
            key_points=d.get("key_points", [])
        )

@dataclass
class MacroEvent:
    event_id: str        # dedup key component 1
    url: str             # dedup key component 2
    published_at: str    # dedup key component 3 — ISO 8601
    title: str
    summary: str         # raw text from RSS or scraped excerpt
    category: str
    source: str          # e.g., "RBI"
    collected_at: str
    
    # Versioning & Lifecycle
    version: int = 1
    status: str = "Active"       # "Active", "Superseded", "Withdrawn", "Corrected"
    lifecycle: str = "Collected" # "Published", "Collected", "Processed", "Enriched", "Correlated", "Reported", "Archived"
    
    impact: Optional[ImpactAnalysis] = None
    event_study: Optional[EventStudy] = None
    
    # AI Snapshots (Never overwriting history)
    ai_snapshots: List[AISummarySnapshot] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'MacroEvent':
        impact_dict = d.get("impact")
        impact = ImpactAnalysis.from_dict(impact_dict) if impact_dict else None
        
        study_dict = d.get("event_study")
        study = EventStudy.from_dict(study_dict) if study_dict else None
        
        ai_snaps = [AISummarySnapshot.from_dict(s) for s in d.get("ai_snapshots", [])]
        
        return cls(
            event_id=d.get("event_id", ""),
            url=d.get("url", ""),
            published_at=d.get("published_at", ""),
            title=d.get("title", ""),
            summary=d.get("summary", ""),
            category=d.get("category", ""),
            source=d.get("source", ""),
            collected_at=d.get("collected_at", ""),
            version=d.get("version", 1),
            status=d.get("status", "Active"),
            lifecycle=d.get("lifecycle", "Collected"),
            impact=impact,
            event_study=study,
            ai_snapshots=ai_snaps
        )
