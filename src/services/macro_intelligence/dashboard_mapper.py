from typing import List, Dict, Any
from src.services.macro_intelligence.models import MacroEvent
from src.services.macro_intelligence.read_models import DashboardViewModel
from datetime import datetime

class DashboardMapper:
    """Decouples backend macro models from the UI presentation layer."""
    
    @staticmethod
    def map_event(event: MacroEvent) -> DashboardViewModel:
        """Maps a backend MacroEvent to the DashboardViewModel flat structure."""
        
        confidence = 0
        impact = "Neutral"
        priority = "Informational"
        summary = event.official_data.content or ""
        keywords = []
        themes = []
        
        if event.derived_data:
            if event.derived_data.impact:
                impact = event.derived_data.impact.direction
                priority = event.derived_data.impact.severity
                confidence = event.derived_data.impact.confidence
            if event.derived_data.ai_summary:
                summary = event.derived_data.ai_summary
            if event.derived_data.keywords:
                keywords = event.derived_data.keywords
            if event.derived_data.ai_theme:
                themes = [event.derived_data.ai_theme]
                
        # Format display date "DD MMM YYYY"
        try:
            pub_date = datetime.fromisoformat(event.official_data.publication_date.replace("Z", "+00:00"))
            display_date = pub_date.strftime("%d %b %Y")
        except ValueError:
            display_date = event.official_data.publication_date[:10]
                
        return DashboardViewModel(
            event_id=event.event_id,
            title=event.official_data.title,
            published=event.official_data.publication_date,
            display_date=display_date,
            category=event.official_data.category,
            priority=priority,
            summary=summary,
            confidence=confidence,
            impact=impact,
            pdf=event.official_data.pdf_url,
            badges=[event.official_data.category, priority],
            url=event.official_data.official_url,
            source=event.official_data.source,
            attachments=event.official_data.attachments,
            keywords=keywords,
            themes=themes,
            processing_state=event.metadata.processing_state,
            related_events=event.metadata.related_event_ids or [],
            supersedes=event.metadata.supersedes_event_id,
            content=event.official_data.content or "",
            ai_summary=event.derived_data.ai_summary if event.derived_data else None
        )

    @staticmethod
    def map_events(events: List[MacroEvent]) -> List[DashboardViewModel]:
        return [DashboardMapper.map_event(e) for e in events]
