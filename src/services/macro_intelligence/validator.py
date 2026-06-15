from src.services.macro_intelligence.interfaces import ValidatorInterface
from src.services.macro_intelligence.models import MacroEvent
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("macro-validator")

class DefaultValidator(ValidatorInterface):
    """
    Validates that a MacroEvent has all required fields and correct formats
    before it is allowed to enter the repository.
    """
    
    def validate(self, event: MacroEvent) -> bool:
        try:
            # 1. Canonical ID validation
            if not event.event_id or len(event.event_id) != 16:
                logger.error("VALIDATION_FAILED: Invalid event_id", extra={"event_title": event.official_data.title})
                return False
                
            # 2. Official Data validation
            od = event.official_data
            if not od.title:
                logger.error("VALIDATION_FAILED: Missing title", extra={"event_id": event.event_id})
                return False
                
            if not od.official_url or not od.official_url.startswith("http"):
                logger.error("VALIDATION_FAILED: Invalid URL", extra={"event_id": event.event_id})
                return False
                
            if not od.publication_date:
                logger.error("VALIDATION_FAILED: Missing publication date", extra={"event_id": event.event_id})
                return False
                
            if not od.category:
                logger.error("VALIDATION_FAILED: Missing category", extra={"event_id": event.event_id})
                return False
                
            if not od.content:
                logger.warning("VALIDATION_WARNING: Missing content", extra={"event_id": event.event_id})
                # Content might be empty for some pure-attachment events, so we don't strictly fail it yet
                
            return True
            
        except (AttributeError, TypeError, ValueError) as e:
            logger.error("VALIDATION_ERROR_DURING_CHECKS", extra={"event_id": getattr(event, "event_id", "Unknown"), "error": str(e)})
            return False
