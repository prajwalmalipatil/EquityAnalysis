import argparse
import sys
from pathlib import Path
from src.services.orchestration.replay_engine import ReplayEngine
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("main-replay")

def main():
    parser = argparse.ArgumentParser(description="Research Time Machine - Replay Engine")
    parser.add_argument("--date", required=True, help="Target Date (YYYY-MM-DD)")
    parser.add_argument("--output", default="dashboard_replay", help="Output directory for the replay dashboard")
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    
    try:
        engine = ReplayEngine(output_dir)
        engine.run_replay(args.date)
    except Exception as e:
        logger.error("REPLAY_FAILED", extra={"error": str(e)})
        sys.exit(1)

if __name__ == "__main__":
    main()
