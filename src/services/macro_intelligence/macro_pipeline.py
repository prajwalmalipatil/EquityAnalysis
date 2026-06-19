from typing import List, Optional
import sys
from datetime import datetime, timezone
from src.utils.observability import get_tenant_logger, get_metrics_tracker
from src.services.macro_intelligence.config import MacroConfig
from src.services.macro_intelligence.interfaces import (
    OfficialSourceCollector,
    ValidatorInterface,
    EventWriteRepository,
    EnrichmentServiceInterface
)
from src.services.macro_intelligence.attachment_processor import AttachmentProcessor
from src.services.macro_intelligence.state_manager import PipelineStateManager

logger = get_tenant_logger("macro-pipeline")
metrics = get_metrics_tracker()

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
        repository: EventWriteRepository,
        attachment_processor: Optional[AttachmentProcessor] = None,
        enrichment: Optional[EnrichmentServiceInterface] = None,
        state_manager: Optional[PipelineStateManager] = None
    ):
        self.config = config
        self.collectors = collectors
        self.validator = validator
        self.repository = repository
        self.attachment_processor = attachment_processor
        self.enrichment = enrichment
        self.state_manager = state_manager

    def execute(self) -> None:
        """Runs the complete macro intelligence pipeline."""
        logger.info("PIPELINE_STARTED", extra={"version": self.config.version})
        
        try:
            total_collected = 0
            total_saved = 0
            total_duplicates = 0
            
            with metrics.time_block("runtime_pipeline_total_duration_ms"):
                # 1. Collect & Normalize
                last_ts = self.state_manager.get_last_fetch_timestamp() if self.state_manager else ""
                for collector in self.collectors:
                    with metrics.time_block("collector_fetch_duration_ms", tags={"collector": type(collector).__name__}):
                        events = collector.fetch_since(last_ts)
                    total_collected += len(events)
                    metrics.increment("events_collected", len(events), tags={"collector": type(collector).__name__})
                    
                    for event in events:
                        # 2. Validate
                        if not self.validator.validate(event):
                            metrics.increment("events_validation_failed")
                            continue
                            
                        # 3. Process Attachments
                        if self.attachment_processor:
                            with metrics.time_block("attachment_processing_duration_ms"):
                                event = self.attachment_processor.process(event)
                            
                        # 4. Enrich
                        if self.enrichment:
                            with metrics.time_block("ai_enrichment_duration_ms"):
                                event = self.enrichment.process(event)
                            
                        # 5. Deduplicate & Persist
                        saved = self.repository.save_event(event)
                        if saved:
                            total_saved += 1
                            metrics.increment("events_persisted")
                        else:
                            total_duplicates += 1
                            metrics.increment("events_duplicate_rejected")
                            
            metrics.gauge("pipeline_run_collected", total_collected)
            metrics.gauge("pipeline_run_saved", total_saved)
            metrics.gauge("pipeline_run_duplicates", total_duplicates)
            
            if self.state_manager:
                self.state_manager.update_last_fetch_timestamp(
                    datetime.now(timezone.utc).isoformat()
                )
                        
            logger.info("PIPELINE_EXECUTION_SUCCESSFUL", extra={"collected": total_collected, "saved": total_saved, "duplicates": total_duplicates})
        except Exception as e:
            logger.error("PIPELINE_EXECUTION_FAILED", extra={"error": str(e)})
            raise

def load_env():
    import os
    from pathlib import Path
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

def run_macro_pipeline():
    load_env()
    import os
    from pathlib import Path
    from src.services.macro_intelligence.config import load_config
    from src.services.macro_intelligence.validator import DefaultValidator
    from src.services.macro_intelligence.event_repository import JSONEventWriteRepository
    from src.services.macro_intelligence.rbi_collector import RBICollector
    from src.services.macro_intelligence.attachment_processor import AttachmentProcessor
    from src.services.macro_intelligence.enrichment_service import EnrichmentService, GeminiProvider, DummyProvider
    from src.services.macro_intelligence.state_manager import PipelineStateManager
    
    try:
        config_path = Path(__file__).parent / "config.yaml"
        config = load_config(config_path)
        
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            provider = GeminiProvider(api_key=api_key)
        else:
            provider = DummyProvider()
            
        enrichment = EnrichmentService(provider=provider)
        state_manager = PipelineStateManager(state_file=config.storage.state_file)
        
        pipeline = MacroPipeline(
            config=config,
            collectors=[RBICollector()],
            validator=DefaultValidator(),
            repository=JSONEventWriteRepository(config=config),
            attachment_processor=AttachmentProcessor(config=config.storage),
            enrichment=enrichment,
            state_manager=state_manager
        )
        pipeline.execute()
        return True
    except Exception as e:
        logger.error("Macro Intelligence execution failed", extra={"error": str(e)})
        return False

if __name__ == "__main__":
    success = run_macro_pipeline()
    sys.exit(0 if success else 1)
