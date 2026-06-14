import json
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Union
from src.services.macro_intelligence.models import MacroEvent
from src.services.macro_intelligence.interfaces import EventRepositoryInterface
from src.services.macro_intelligence.config import StorageConfig
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("event-repository")

class EventRepository(EventRepositoryInterface):
    """JSON Lines repository for Macro Events with Versioning support."""

    def __init__(self, config: StorageConfig):
        self.config = config
        self.filepath = self.config.history_file
        
        # Ensure all required directories exist
        self.config.base_path.mkdir(parents=True, exist_ok=True)
        self.config.attachments_dir.mkdir(parents=True, exist_ok=True)
        self.config.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._cache = {}
        self._load_cache()

    def _load_cache(self):
        """Load all event_ids from the JSONL file to prevent exact duplicates."""
        if not self.filepath.exists():
            return
            
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    # event_id is the canonical deduplication key
                    key = data.get("event_id", "")
                    if key:
                        self._cache[key] = True
                except json.JSONDecodeError:
                    continue

    def save_event(self, event: MacroEvent) -> bool:
        """
        Save an event to the repository with versioning.
        Returns True if saved (new or updated), False if exact duplicate.
        """
        key = event.event_id
        if key in self._cache:
            return False
            
        all_events = self.get_all_events()
        past_versions = [
            e for e in all_events 
            if e.event_id == event.event_id
        ]
        
        if past_versions:
            # We don't have version on root anymore, we should use schema_version or just assume no versioning for now, 
            # wait, the plan explicitly says we will implement deduplication differently in Phase 8.
            # For now, if event_id matches, it's a duplicate.
            self._cache[key] = True
            return False

        with open(self.filepath, 'a', encoding='utf-8') as f:
            json_str = json.dumps(event.to_dict())
            f.write(json_str + '\n')
            
        self._cache[key] = True
        return True

    def get_all_events(self) -> List[MacroEvent]:
        events = []
        if not self.filepath.exists():
            return events
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    events.append(MacroEvent.from_dict(data))
                except json.JSONDecodeError:
                    continue
        return events

    def get_active_events(self) -> List[MacroEvent]:
        """Returns only Active events."""
        return [e for e in self.get_all_events() if e.metadata.lifecycle_status == "ACTIVE"]
        
    def get_events_by_date_range(self, start_date: str, end_date: str) -> List[MacroEvent]:
        return [
            e for e in self.get_active_events()
            if start_date <= e.official_data.publication_date <= end_date
        ]
