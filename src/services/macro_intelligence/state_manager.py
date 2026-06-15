import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

class PipelineStateManager:
    """Persists pipeline run state (last fetch timestamp) to a JSON file."""

    def __init__(self, state_file: Path):
        self._state_file = state_file

    def get_last_fetch_timestamp(self) -> str:
        """Returns the last successful fetch timestamp, or empty string if none."""
        if not self._state_file.exists():
            return ""
        try:
            with open(self._state_file, 'r') as f:
                state = json.load(f)
            return state.get("last_fetch_ts", "")
        except (json.JSONDecodeError, IOError):
            return ""

    def update_last_fetch_timestamp(self, timestamp: str) -> None:
        """Saves the current fetch timestamp after a successful run."""
        state = {"last_fetch_ts": timestamp, "updated_at": datetime.now(timezone.utc).isoformat()}
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_file, 'w') as f:
            json.dump(state, f, indent=2)
