import json
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Union
from difflib import SequenceMatcher
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

    def _calculate_similarity(self, a: MacroEvent, b: MacroEvent) -> float:
        """Calculates a weighted similarity score between two events."""
        weights = self.config.deduplication.weights
        score = 0.0
        
        # Title Similarity (0.40)
        title_sim = SequenceMatcher(None, a.official_data.title.lower(), b.official_data.title.lower()).ratio()
        score += title_sim * weights.get("title", 0.40)
        
        # Publication Date Similarity (0.20)
        if a.official_data.publication_date[:10] == b.official_data.publication_date[:10]:
            score += weights.get("publication_date", 0.20)
            
        # Official URL Similarity (0.10)
        if a.official_data.official_url == b.official_data.official_url:
            score += weights.get("official_url", 0.10)
            
        # Content / Document Hash Similarity (0.20)
        content_a = a.official_data.content or ""
        content_b = b.official_data.content or ""
        if content_a and content_b:
            content_sim = SequenceMatcher(None, content_a.lower(), content_b.lower()).ratio()
            score += content_sim * weights.get("document_hash", 0.20)
            
        # Attachment Hash Similarity (0.10)
        # Not fully implemented until Phase 9 attaches actual hashes, so we just check count
        if len(a.official_data.attachments) == len(b.official_data.attachments) and a.official_data.attachments:
            score += weights.get("attachment_hash", 0.10)
            
        return score

    def save_event(self, event: MacroEvent) -> bool:
        """
        Save an event to the repository with versioning.
        Returns True if saved (new or updated), False if exact duplicate.
        """
        key = event.event_id
        if key in self._cache:
            return False
            
        all_events = self.get_all_events()
        
        # Advanced Weighted Deduplication
        threshold = self.config.deduplication.similarity_threshold
        for past_event in all_events:
            # Short-circuit exact match
            if past_event.event_id == event.event_id:
                self._cache[key] = True
                return False
                
            sim_score = self._calculate_similarity(event, past_event)
            if sim_score >= threshold:
                logger.info("ADVANCED_DEDUPLICATION_TRIGGERED", extra={
                    "new_title": event.official_data.title,
                    "past_title": past_event.official_data.title,
                    "score": sim_score
                })
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
