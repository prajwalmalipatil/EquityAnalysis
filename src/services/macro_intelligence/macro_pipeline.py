from typing import List, Optional
from src.utils.observability import get_tenant_logger
from src.services.macro_intelligence.config import MacroConfig
from src.services.macro_intelligence.interfaces import (
    OfficialSourceCollector,
    ValidatorInterface,
    EventRepositoryInterface,
    EnrichmentServiceInterface
)

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
        enrichment: Optional[EnrichmentServiceInterface] = None
    ):
        self.config = config
        self.collectors = collectors
        self.validator = validator
        self.repository = repository
        self.enrichment = enrichment

    def execute(self) -> None:
        """Runs the complete macro intelligence pipeline."""
        logger.info("PIPELINE_STARTED", extra={"version": self.config.version})
        
        try:
            # 1. Collect
            # 2. Normalize & canonicalize
            # 3. Validate
            # 4. Process Attachments
            # 5. Deduplicate & Persist
            # 6. Enrich
            logger.info("EMPTY_PIPELINE_EXECUTION_SUCCESSFUL")
        except Exception as e:
            logger.error("PIPELINE_EXECUTION_FAILED", extra={"error": str(e)})
            raise

if __name__ == "__main__":
    # Test empty execution to satisfy Milestone 1 exit criteria
    import sys
    from pathlib import Path
    from src.services.macro_intelligence.config import load_config
    from src.services.macro_intelligence.validator import DefaultValidator
    from src.services.macro_intelligence.event_repository import EventRepository
    from src.services.macro_intelligence.rbi_collector import RBICollector
    
    try:
        config_path = Path(__file__).parent / "config.yaml"
        config = load_config(config_path)
        
        pipeline = MacroPipeline(
            config=config,
            collectors=[RBICollector()],
            validator=DefaultValidator(),
            repository=EventRepository(config=config.storage)
        )
        pipeline.execute()
        print("Milestone 1 Pipeline execution successful!")
        sys.exit(0)
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        sys.exit(1)
