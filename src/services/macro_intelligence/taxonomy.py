from enum import Enum, auto

class MacroCategory(Enum):
    """Normalized categories for all macroeconomic events."""
    MONETARY_POLICY = "Monetary Policy"
    BANKING_REGULATION = "Banking Regulation"
    FOREX = "Forex & External Sector"
    PAYMENT_SYSTEMS = "Payment & Settlement Systems"
    FINANCIAL_STABILITY = "Financial Stability"
    ECONOMIC_PUBLICATIONS = "Economic Publications"
    GOVERNMENT_SECURITIES = "Government Securities"
    PUBLIC_ANNOUNCEMENT = "Public Announcement"
    UNKNOWN = "Unknown"

class ProcessingState(Enum):
    """The explicit processing stage of an event through the macro pipeline."""
    NEW = "NEW"
    COLLECTED = "COLLECTED"
    NORMALIZED = "NORMALIZED"
    VALIDATED = "VALIDATED"
    ATTACHMENTS_READY = "ATTACHMENTS_READY"
    ENRICHED = "ENRICHED"
    PUBLISHED = "PUBLISHED"
    FAILED = "FAILED"

class LifecycleStatus(Enum):
    """The regulatory status of the event."""
    ACTIVE = "ACTIVE"
    REVISED = "REVISED"
    SUPERSEDED = "SUPERSEDED"
    WITHDRAWN = "WITHDRAWN"
    UNKNOWN = "UNKNOWN"
