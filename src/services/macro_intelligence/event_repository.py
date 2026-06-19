import json
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Union
from difflib import SequenceMatcher
from filelock import FileLock
from src.services.macro_intelligence.models import MacroEvent
from src.services.macro_intelligence.interfaces import EventReadRepository, EventWriteRepository
from src.services.macro_intelligence.config import StorageConfig, MacroConfig
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("event-repository")

PUBLISHABLE_STATUSES = {"ACTIVE", "ENRICHED", "VALIDATED"}


class JSONEventReadRepository(EventReadRepository):
    """Read-optimized JSONL repository for queries and projections."""
    def __init__(self, config: StorageConfig):
        self.config = config
        self.filepath = self.config.history_file

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
        return [e for e in self.get_all_events() if e.metadata.lifecycle_status in PUBLISHABLE_STATUSES]



class JSONEventWriteRepository(EventWriteRepository):
    """Write-optimized JSONL repository for ingestion and deduplication."""
    def __init__(self, config: MacroConfig):
        self.config = config
        self.filepath = self.config.storage.history_file
        self._lock = FileLock(str(self.filepath) + ".lock", timeout=30)
        
        # Ensure directories exist
        self.config.storage.base_path.mkdir(parents=True, exist_ok=True)
        self.config.storage.attachments_dir.mkdir(parents=True, exist_ok=True)
        self.config.storage.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.config.storage.logs_dir.mkdir(parents=True, exist_ok=True)
        self.config.storage.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._cache = {}
        self._load_cache()
        
        self._events_cache: List[MacroEvent] = []
        self._cache_loaded = False

    def _ensure_cache(self):
        """Loads all events into memory once per pipeline run."""
        if self._cache_loaded:
            return
        reader = JSONEventReadRepository(self.config.storage)
        self._events_cache = reader.get_all_events()
        self._cache_loaded = True

    def _load_cache(self):
        if not self.filepath.exists():
            return
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    key = data.get("event_id", "")
                    if key:
                        self._cache[key] = True
                except json.JSONDecodeError:
                    continue

    def _calculate_similarity(self, a: MacroEvent, b: MacroEvent) -> float:
        # If the official URL is identical, it's the exact same event
        if a.official_data.official_url and b.official_data.official_url:
            url_a = a.official_data.official_url.strip().lower().rstrip('/')
            url_b = b.official_data.official_url.strip().lower().rstrip('/')
            if url_a == url_b:
                return 1.0

        weights = self.config.deduplication.weights
        score = 0.0
        
        title_sim = SequenceMatcher(None, a.official_data.title.lower(), b.official_data.title.lower()).ratio()
        score += title_sim * weights.get("title", 0.40)
        
        if a.official_data.publication_date[:10] == b.official_data.publication_date[:10]:
            score += weights.get("publication_date", 0.20)
            
        if a.official_data.official_url == b.official_data.official_url:
            score += weights.get("official_url", 0.10)
            
        content_a = a.official_data.content or ""
        content_b = b.official_data.content or ""
        if content_a and content_b:
            # Prevent SequenceMatcher timeouts on massive texts by truncating inputs
            content_sim = SequenceMatcher(None, content_a[:10000].lower(), content_b[:10000].lower()).ratio()
            score += content_sim * weights.get("document_hash", 0.20)
            
        if len(a.official_data.attachments) == len(b.official_data.attachments) and a.official_data.attachments:
            score += weights.get("attachment_hash", 0.10)
            
        return score

    def save_event(self, event: MacroEvent) -> bool:
        key = event.event_id
        if key in self._cache:
            return False
            
        self._ensure_cache()
            
        threshold = self.config.deduplication.similarity_threshold
        for past_event in self._events_cache:
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
    
        with self._lock:
            with open(self.filepath, 'a', encoding='utf-8') as f:
                json_str = json.dumps(event.to_dict())
                f.write(json_str + '\n')
                
        self._events_cache.append(event)
        self._cache[key] = True
        return True
