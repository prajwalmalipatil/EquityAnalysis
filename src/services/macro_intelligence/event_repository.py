import json
import hashlib
from pathlib import Path
from typing import List, Optional
from src.services.macro_intelligence.models import MacroEvent

def make_dedup_key(event_id: str, url: str, published_at: str) -> str:
    """Generate a stable deduplication key for macro events."""
    raw = f"{event_id}|{url}|{published_at}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

class EventRepository:
    """Append-only JSON Lines repository for Macro Events."""

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._cache = {}
        self._load_cache()

    def _load_cache(self):
        """Load all dedup keys from the JSONL file to prevent duplicates."""
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
        Save an event to the repository if it doesn't already exist.
        Returns True if saved, False if it was a duplicate.
        """
        key = make_dedup_key(event.event_id, event.url, event.published_at)
        if key in self._cache:
            return False
            
        with open(self.filepath, 'a', encoding='utf-8') as f:
            json_str = json.dumps(event.to_dict())
            f.write(json_str + '\n')
            
        self._cache[key] = True
        return True

    def get_all_events(self) -> List[MacroEvent]:
        """Retrieve all events from the repository."""
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

    def get_events_by_date_range(self, start_date: str, end_date: str) -> List[MacroEvent]:
        """
        Retrieve events published within a date range [start_date, end_date] inclusive.
        Dates should be string comparable (e.g. ISO 8601 YYYY-MM-DD).
        """
        return [
            e for e in self.get_all_events()
            if start_date <= e.published_at <= end_date
        ]
