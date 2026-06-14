from typing import List, Dict, Any
from datetime import datetime, timezone
from collections import defaultdict
from src.services.macro_intelligence.models import MacroEvent
from src.services.macro_intelligence.read_models import (
    AnalyticsReadModel, 
    AnalyticsViewModel,
    ManifestViewModel, 
    DashboardBundle,
    DashboardViewModel
)
from src.services.macro_intelligence.relationship_models import Relationship
from src.services.macro_intelligence.dashboard_mapper import DashboardMapper
from src.services.macro_intelligence.interfaces import EventReadRepository

from dataclasses import asdict

class AnalyticsBuilder:
    """Formats the AnalyticsReadModel into a presentation-ready AnalyticsViewModel."""
    
    @staticmethod
    def build(read_model: AnalyticsReadModel, pipeline_version: str = "2.1.0") -> AnalyticsViewModel:
        # Convert nested dataclasses into pure dicts for the JSON payload
        analytics_dict = {
            "total_events": read_model.total_events,
            "business": asdict(read_model.business),
            "ai": asdict(read_model.ai),
            "operational": asdict(read_model.operational),
            "quality": asdict(read_model.quality),
            "coverage": asdict(read_model.coverage)
        }
        
        return AnalyticsViewModel(
            schema_version=1,
            generated_at=read_model.generated_at,
            pipeline_version=pipeline_version,
            event_count=read_model.total_events,
            analytics=analytics_dict
        )


class RelationshipBuilder:
    @staticmethod
    def build(relationships: List[Relationship]) -> List[Dict]:
        return [
            {
                "id": r.id,
                "source_event_id": r.source_event_id,
                "target_event_id": r.target_event_id,
                "type": r.type.value,
                "confidence": r.confidence.value,
                "provenance": r.provenance,
                "rule_version": r.rule_version,
                "resolver_version": r.resolver_version,
                "created_at": r.created_at
            }
            for r in relationships
        ]


class ManifestBuilder:
    """Builds the manifest containing pipeline and payload metadata."""
    
    @staticmethod
    def build(event_count: int, artifacts: Dict[str, Any]) -> ManifestViewModel:
        return ManifestViewModel(
            schema_version=2,
            generated_at=datetime.now(timezone.utc).isoformat(),
            generator="Macro Intelligence Pipeline",
            pipeline_version="2.1.0",
            repository_version="v1",
            event_count=event_count,
            analytics_version=1,
            search_index_version=1,
            artifacts=artifacts
        )
