from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime

@dataclass(frozen=True)
class SearchDocument:
    doc_id: str
    tokens: List[str]

@dataclass(frozen=True)
class DashboardViewModel:
    """The flattened, UI-optimized DTO representing an event."""
    event_id: str
    title: str
    published: str
    display_date: str
    category: str
    priority: str
    summary: str
    confidence: int
    impact: str
    pdf: Optional[str]
    badges: List[str]
    url: str
    source: str
    attachments: List[Dict[str, str]]
    keywords: List[str]
    themes: List[str]
    processing_state: str
    related_events: List[str] = field(default_factory=list)
    supersedes: Optional[str] = None
    content: str = ""
    ai_summary: Optional[str] = None


@dataclass(frozen=True)
class TrendPoint:
    date: str
    value: float

@dataclass(frozen=True)
class TimeSeries:
    name: str
    points: List[TrendPoint]

@dataclass(frozen=True)
class DistributionBucket:
    label: str
    count: int

@dataclass(frozen=True)
class BusinessMetrics:
    events_per_day: TimeSeries
    category_distribution: List[DistributionBucket]
    high_priority_circulars: int
    upcoming_effective_dates: int

@dataclass(frozen=True)
class AIMetrics:
    confidence_distribution: List[DistributionBucket]
    theme_frequency: List[DistributionBucket]
    processing_success_rate: float
    avg_latency_ms: float

@dataclass(frozen=True)
class OperationalMetrics:
    collector_success_rate: float
    duplicate_rate: float
    publish_duration_ms: float
    attachment_success_rate: float
    feed_availability: float

@dataclass(frozen=True)
class CoverageMetrics:
    events_with_ai: int
    events_with_pdf: int
    events_with_attachments: int
    events_with_effective_date: int
    events_with_market_impact: int

@dataclass(frozen=True)
class QualityMetrics:
    events_with_pdf: int
    events_missing_attachment: int
    missing_effective_date: int
    ai_enrichment_coverage: float
    validation_failures: int
    avg_quality_score: float

@dataclass(frozen=True)
class AnalyticsReadModel:
    """The DTO representing pre-computed modular analytics for the UI."""
    version: int
    generated_at: str
    total_events: int
    business: BusinessMetrics
    ai: AIMetrics
    operational: OperationalMetrics
    quality: QualityMetrics
    coverage: CoverageMetrics


@dataclass(frozen=True)
class AnalyticsViewModel:
    """The presentation layer DTO for Analytics workspace, ready for JSON serialization."""
    schema_version: int
    generated_at: str
    pipeline_version: str
    event_count: int
    analytics: Dict[str, Any]

@dataclass(frozen=True)
class ManifestViewModel:
    """The DTO representing the dashboard manifest."""
    schema_version: int
    generated_at: str
    generator: str
    pipeline_version: str
    repository_version: str
    event_count: int
    analytics_version: int
    search_index_version: int
    artifacts: List[str]


@dataclass(frozen=True)
class DashboardBundle:
    """The complete package assembled by Builders."""
    manifest: ManifestViewModel
    analytics: AnalyticsViewModel
    events: List[DashboardViewModel]
