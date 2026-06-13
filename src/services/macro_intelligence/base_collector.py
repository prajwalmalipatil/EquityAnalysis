from abc import ABC, abstractmethod
from typing import List
from src.services.macro_intelligence.models import MacroEvent

class BaseMacroCollector(ABC):
    """Abstract base class for all Macro Intelligence data providers."""
    
    @abstractmethod
    def fetch_since(self, last_ts: str) -> List[MacroEvent]:
        """Fetch all official publications released after the last timestamp."""
        pass

    @abstractmethod
    def normalize(self, raw_item: dict) -> MacroEvent:
        """Convert a provider-specific raw item into a canonical MacroEvent."""
        pass
