from typing import List, Optional
from src.services.macro_intelligence.models import MacroEvent
from src.services.macro_intelligence.interfaces import EventReadRepository
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("macro-query-service")

class MacroQueryService:
    """Read facade for UI and reporting components to fetch data from the repository."""
    def __init__(self, repository: EventReadRepository):
        self.repository = repository

    def get_latest_events(self, limit: int = 20) -> List[MacroEvent]:
        events = self.repository.get_active_events()
        # Sort by publication date descending
        events.sort(key=lambda x: x.official_data.publication_date, reverse=True)
        return events[:limit]

    def get_event(self, event_id: str) -> Optional[MacroEvent]:
        for e in self.repository.get_active_events():
            if e.event_id == event_id:
                return e
        return None

    def get_events_by_category(self, category: str) -> List[MacroEvent]:
        return [e for e in self.repository.get_active_events() if e.official_data.category == category]

    def get_high_priority_events(self) -> List[MacroEvent]:
        return [e for e in self.repository.get_active_events() 
                if e.derived_data.impact and e.derived_data.impact.severity in ["Critical", "High"]]

    def search_events(self, query: str) -> List[MacroEvent]:
        query = query.lower()
        results = []
        for e in self.repository.get_active_events():
            if (query in e.official_data.title.lower() or 
                (e.official_data.content and query in e.official_data.content.lower()) or
                (e.derived_data.ai_summary and query in e.derived_data.ai_summary.lower())):
                results.append(e)
        return results

    def get_all_events(self) -> List[MacroEvent]:
        events = self.repository.get_active_events()
        events.sort(key=lambda x: x.official_data.publication_date, reverse=True)
        return events
