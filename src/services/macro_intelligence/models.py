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
    provider: str
    model: str
    generated_time: str
    confidence: int
    prompt_version: str
    response_version: str
    summary: str
    key_points: List[str]
    raw_ai_response: str

    @classmethod
    def from_dict(cls, d: dict) -> 'AISummarySnapshot':
        return cls(
            version=d.get("version", 1),
            provider=d.get("provider", "Unknown"),
            model=d.get("model", "Unknown"),
            generated_time=d.get("generated_time", d.get("timestamp", "")),
            confidence=d.get("confidence", 0),
            prompt_version=d.get("prompt_version", ""),
            response_version=d.get("response_version", ""),
            summary=d.get("summary", ""),
            key_points=d.get("key_points", []),
            raw_ai_response=d.get("raw_ai_response", "")
        )

@dataclass
class OfficialData:
    title: str
    publication_date: str
    category: str
    source: str
    official_url: str
    publication_time: Optional[str] = None
    effective_date: Optional[str] = None
    content: Optional[str] = None
    attachments: List[str] = field(default_factory=list)
    pdf_url: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> 'OfficialData':
        return cls(
            title=d.get("title", ""),
            publication_date=d.get("publication_date", d.get("published_at", "")),
            category=d.get("category", ""),
            source=d.get("source", "Unknown"),
            official_url=d.get("official_url", d.get("url", "")),
            publication_time=d.get("publication_time"),
            effective_date=d.get("effective_date"),
            content=d.get("content", d.get("summary", "")),
            attachments=d.get("attachments", []),
            pdf_url=d.get("pdf_url")
        )

@dataclass
class DerivedData:
    ai_summary: Optional[str] = None
    ai_theme: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    market_relevance: Optional[str] = None
    quality_score: int = 0
    impact: Optional[ImpactAnalysis] = None
    event_study: Optional[EventStudy] = None
    ai_snapshots: List[AISummarySnapshot] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> 'DerivedData':
        impact_dict = d.get("impact")
        study_dict = d.get("event_study")
        return cls(
            ai_summary=d.get("ai_summary"),
            ai_theme=d.get("ai_theme"),
            keywords=d.get("keywords", []),
            market_relevance=d.get("market_relevance"),
            quality_score=d.get("quality_score", 0),
            impact=ImpactAnalysis.from_dict(impact_dict) if impact_dict else None,
            event_study=EventStudy.from_dict(study_dict) if study_dict else None,
            ai_snapshots=[AISummarySnapshot.from_dict(s) for s in d.get("ai_snapshots", [])]
        )

@dataclass
class EventMetadata:
    processing_state: str
    lifecycle_status: str
    created_at: str
    updated_at: str
    schema_version: str = "1.0"
    supersedes_event_id: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> 'EventMetadata':
        return cls(
            processing_state=d.get("processing_state", d.get("lifecycle", "NEW")),
            lifecycle_status=d.get("lifecycle_status", d.get("status", "ACTIVE")),
            created_at=d.get("created_at", d.get("collected_at", "")),
            updated_at=d.get("updated_at", d.get("collected_at", "")),
            schema_version=d.get("schema_version", "1.0"),
            supersedes_event_id=d.get("supersedes_event_id")
        )

@dataclass
class MacroEvent:
    event_id: str
    official_data: OfficialData
    derived_data: DerivedData
    metadata: EventMetadata

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'MacroEvent':
        # Backward compatibility for old flat schema
        if "official_data" in d:
            official_data = OfficialData.from_dict(d.get("official_data", {}))
            derived_data = DerivedData.from_dict(d.get("derived_data", {}))
            metadata = EventMetadata.from_dict(d.get("metadata", {}))
        else:
            official_data = OfficialData.from_dict(d)
            derived_data = DerivedData.from_dict(d)
            metadata = EventMetadata.from_dict(d)
            
        return cls(
            event_id=d.get("event_id", ""),
            official_data=official_data,
            derived_data=derived_data,
            metadata=metadata
        )
