import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
from src.utils.observability import get_tenant_logger
from src.constants.vsa_constants import ETE_EVENTS_DIR
from src.services.reporting.view_builder_service import ViewBuilderService

logger = get_tenant_logger("replay-engine")

class ReplayEngine:
    """
    Reconstructs the live dashboard for any historical date using the immutable event repository.
    """
    def __init__(self, output_dir: Path, source_events_dir: Path = None):
        self.output_dir = Path(output_dir)
        self.source_events_dir = Path(source_events_dir) if source_events_dir else Path(ETE_EVENTS_DIR)
        self.replay_events_dir = self.output_dir / "temp_replay_events"

    def run_replay(self, target_date: str):
        """
        target_date: "YYYY-MM-DD"
        """
        logger.info(f"STARTING_REPLAY for Target Date: {target_date}")
        
        # 1. Setup shadow event directory
        if self.replay_events_dir.exists():
            shutil.rmtree(self.replay_events_dir)
        self.replay_events_dir.mkdir(parents=True, exist_ok=True)
        
        target_cutoff = f"{target_date}T23:59:59Z"
        
        # 2. Filter events
        events_filtered = 0
        events_included = 0
        
        for event_file in self.source_events_dir.glob("*_events.jsonl"):
            replay_file = self.replay_events_dir / event_file.name
            with open(event_file, 'r') as f_in, open(replay_file, 'w') as f_out:
                for line in f_in:
                    if not line.strip():
                        continue
                    try:
                        ev = json.loads(line)
                        if ev.get("timestamp", "") <= target_cutoff:
                            f_out.write(line)
                            events_included += 1
                        else:
                            events_filtered += 1
                    except json.JSONDecodeError:
                        pass
        
        logger.info(f"Events Included: {events_included}, Events Filtered Out (Future): {events_filtered}")
        
        # 3. Publish using shadow directory
        logger.info("Triggering ViewBuilder on Replay State...")
        builder = ViewBuilderService(base_output_dir=self.output_dir, events_dir=self.replay_events_dir)
        try:
            builder.publish()
            logger.info("REPLAY_SUCCESS: Dashboard generated.")
        finally:
            # Cleanup temp events
            if self.replay_events_dir.exists():
                shutil.rmtree(self.replay_events_dir)
