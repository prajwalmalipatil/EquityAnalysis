import sys
from pathlib import Path
from src.utils.observability import get_tenant_logger
from src.services.macro_intelligence.rbi_collector import RBICollector
from src.services.macro_intelligence.event_repository import EventRepository
from src.services.macro_intelligence.impact_engine import RuleBasedImpactEngine
from src.services.macro_intelligence.enrichment_service import EventEnrichmentService

logger = get_tenant_logger("main-macro")

def main():
    base_dir = Path(__file__).parent
    history_file = base_dir / "dashboard" / "history" / "rbi_events.jsonl"
    
    logger.info("STARTING_MACRO_INTELLIGENCE_PIPELINE")
    
    repo = EventRepository(history_file)
    collector = RBICollector()
    impact_engine = RuleBasedImpactEngine(base_dir)
    enricher = EventEnrichmentService()
    
    # 1. Continuity Engine: Get last timestamp
    events = repo.get_all_events()
    last_ts = ""
    if events:
        last_ts = max([e.published_at for e in events])
        logger.info("CONTINUITY_ENGINE_RESUMING", extra={"last_ts": last_ts})
    else:
        logger.info("CONTINUITY_ENGINE_FIRST_RUN")
        
    # 2. Collect
    try:
        new_raw_events = collector.fetch_since(last_ts)
    except Exception as e:
        logger.error("COLLECTOR_FAILED", extra={"error": str(e)})
        sys.exit(1)
        
    if not new_raw_events:
        logger.info("NO_NEW_MACRO_EVENTS")
        return

    saved_count = 0
    # 3. Process each event
    for event in new_raw_events:
        try:
            # Impact Engine
            event.impact = impact_engine.process(event)
            
            # LLM Enrichment (Stub)
            event = enricher.enrich(event)
            
            # Save
            if repo.save_event(event):
                saved_count += 1
        except Exception as e:
            logger.error("EVENT_PROCESSING_FAILED", extra={"event_id": event.event_id, "error": str(e)})
            
    logger.info("MACRO_INTELLIGENCE_PIPELINE_COMPLETE", extra={"new_events_saved": saved_count})

if __name__ == "__main__":
    main()
