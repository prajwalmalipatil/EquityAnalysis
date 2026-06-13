from src.services.macro_intelligence.models import MacroEvent

class EventEnrichmentService:
    """
    LLM Enrichment Stub.
    Designed to process MacroEvents and inject ai_summary and ai_key_points.
    Currently LLM is on hold, so this simply returns the event unchanged.
    """
    def __init__(self):
        pass

    def enrich(self, event: MacroEvent) -> MacroEvent:
        # LLM integration goes here.
        # For Phase 1, we just return the event.
        return event
