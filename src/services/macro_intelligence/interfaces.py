from abc import ABC, abstractmethod
from typing import List, Optional
from src.services.macro_intelligence.models import MacroEvent

class OfficialSourceCollector(ABC):
    """Base interface for all official macroeconomic source collectors (RBI, SEBI, etc.)."""
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the official source (e.g., 'RBI')."""
        pass
        
    @abstractmethod
    def fetch_since(self, last_ts: str) -> List[MacroEvent]:
        """Fetches normalized events published after the given timestamp."""
        pass

class ValidatorInterface(ABC):
    """Interface for validating normalized events before persistence."""
    
    @abstractmethod
    def validate(self, event: MacroEvent) -> bool:
        """Returns True if the event is valid and safe to store."""
        pass

class EventRepositoryInterface(ABC):
    """Interface for the immutable event repository."""
    
    @abstractmethod
    def save_event(self, event: MacroEvent) -> bool:
        """Persist the event. Returns True if saved successfully, False if duplicate."""
        pass
        
    @abstractmethod
    def get_all_events(self) -> List[MacroEvent]:
        """Retrieve all events across history."""
        pass

class EnrichmentServiceInterface(ABC):
    """Interface for adding derived AI metadata to events."""
    
    @abstractmethod
    def process(self, event: MacroEvent) -> MacroEvent:
        """Appends derived metadata to the event without modifying official fields."""
        pass
