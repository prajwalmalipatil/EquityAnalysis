import json
import hashlib
from pathlib import Path
from typing import List, Optional, Dict
from src.services.macro_intelligence.models import MacroEvent

def make_dedup_key(event_id: str, url: str, published_at: str) -> str:
    """Generate a stable deduplication key for macro events."""
    raw = f"{event_id}|{url}|{published_at}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

class EventRepository:
    """JSON Lines repository for Macro Events with Versioning support."""

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._cache = {}
        self._load_cache()

    def _load_cache(self):
        """Load all dedup keys from the JSONL file to prevent exact duplicates."""
        if not self.filepath.exists():
            return
            
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    key = make_dedup_key(
                        data.get("event_id", ""),
                        data.get("url", ""),
                        data.get("published_at", "")
                    )
                    self._cache[key] = True
                except json.JSONDecodeError:
                    continue

    def save_event(self, event: MacroEvent) -> bool:
        """
        Save an event to the repository with versioning.
        Returns True if saved (new or updated), False if exact duplicate.
        """
        key = make_dedup_key(event.event_id, event.url, event.published_at)
        if key in self._cache:
            return False
            
        all_events = self.get_all_events()
        past_versions = [
            e for e in all_events 
            if (e.event_id == event.event_id and e.event_id) or (e.url == event.url and e.url)
        ]
        
        if past_versions:
            past_versions.sort(key=lambda x: x.version, reverse=True)
            latest = past_versions[0]
            
            if latest.title == event.title and latest.summary == event.summary and latest.published_at == event.published_at:
                self._cache[key] = True
                return False
                
            event.version = latest.version + 1
            self._rewrite_file_with_new_version(all_events, event, latest)
            self._cache[key] = True
            return True

        with open(self.filepath, 'a', encoding='utf-8') as f:
            json_str = json.dumps(event.to_dict())
            f.write(json_str + '\n')
            
        self._cache[key] = True
        return True

    def _rewrite_file_with_new_version(self, all_events: List[MacroEvent], new_event: MacroEvent, old_latest: MacroEvent):
        """Rewrites the JSONL file, setting older versions to 'Superseded' and appending the new event."""
        with open(self.filepath, 'w', encoding='utf-8') as f:
            for e in all_events:
                if e.event_id == old_latest.event_id and e.url == old_latest.url and e.version == old_latest.version:
                    e.status = "Superseded"
                f.write(json.dumps(e.to_dict()) + '\n')
            f.write(json.dumps(new_event.to_dict()) + '\n')

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
        return [e for e in self.get_all_events() if e.status == "Active"]
        
    def get_events_by_date_range(self, start_date: str, end_date: str) -> List[MacroEvent]:
        return [
            e for e in self.get_active_events()
            if start_date <= e.published_at <= end_date
        ]
