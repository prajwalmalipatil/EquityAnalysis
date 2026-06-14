from typing import List, Optional
from src.utils.observability import get_tenant_logger
from src.services.macro_intelligence.config import MacroConfig
from src.services.macro_intelligence.interfaces import (
    OfficialSourceCollector,
    ValidatorInterface,
    EventRepositoryInterface,
    EnrichmentServiceInterface
)
from src.services.macro_intelligence.attachment_processor import AttachmentProcessor

logger = get_tenant_logger("macro-pipeline")

class MacroPipeline:
    """
    Central orchestrator for the Macro Intelligence Service.
    Coordinates decoupled components via dependency injection.
    """
    def __init__(
        self,
        config: MacroConfig,
        collectors: List[OfficialSourceCollector],
        validator: ValidatorInterface,
        repository: EventRepositoryInterface,
        attachment_processor: Optional[AttachmentProcessor] = None,
        enrichment: Optional[EnrichmentServiceInterface] = None
    ):
        self.config = config
        self.collectors = collectors
        self.validator = validator
        self.repository = repository
        self.attachment_processor = attachment_processor
        self.enrichment = enrichment

    def execute(self) -> None:
        """Runs the complete macro intelligence pipeline."""
        logger.info("PIPELINE_STARTED", extra={"version": self.config.version})
        
        try:
            total_collected = 0
            total_saved = 0
            
            # 1. Collect & Normalize
            # TODO: read last_ts from state file
            for collector in self.collectors:
                events = collector.fetch_since("")
                total_collected += len(events)
                
                for event in events:
                    # 2. Validate
                    if not self.validator.validate(event):
                        continue
                        
                    # 3. Process Attachments
                    if self.attachment_processor:
                        event = self.attachment_processor.process(event)
                        
                    # 4. Enrich
                    if self.enrichment:
                        event = self.enrichment.process(event)
                        
                    # 5. Deduplicate & Persist
                    saved = self.repository.save_event(event)
                    if saved:
                        total_saved += 1
                        
            logger.info("PIPELINE_EXECUTION_SUCCESSFUL", extra={"collected": total_collected, "saved": total_saved})
        except Exception as e:
            logger.error("PIPELINE_EXECUTION_FAILED", extra={"error": str(e)})
            raise

def run_macro_pipeline():
    import sys
    from pathlib import Path
    from src.services.macro_intelligence.config import load_config
    from src.services.macro_intelligence.validator import DefaultValidator
    from src.services.macro_intelligence.event_repository import EventRepository
    from src.services.macro_intelligence.rbi_collector import RBICollector
    from src.services.macro_intelligence.attachment_processor import AttachmentProcessor
    
    try:
        config_path = Path(__file__).parent / "config.yaml"
        config = load_config(config_path)
        
        pipeline = MacroPipeline(
            config=config,
            collectors=[RBICollector()],
            validator=DefaultValidator(),
            repository=EventRepository(config=config.storage),
            attachment_processor=AttachmentProcessor(config=config.storage)
        )
        pipeline.execute()
        return True
    except Exception as e:
        logger.error("Macro Intelligence execution failed", extra={"error": str(e)})
        return False

if __name__ == "__main__":
    success = run_macro_pipeline()
    sys.exit(0 if success else 1)
