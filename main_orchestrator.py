import argparse
import sys
from pathlib import Path
from src.services.orchestration.pipeline_orchestrator import PipelineOrchestrator
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("main-orchestrator")

def main():
    parser = argparse.ArgumentParser(description="Pipeline Orchestrator (Phase A)")
    parser.add_argument("--folder", required=True, help="Folder containing CSV files")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--skip-macro", action="store_true", help="Skip the Macro Intelligence ingestion stage")
    parser.add_argument("--force-macro", action="store_true", help="Force run the Macro Intelligence ingestion even if skipped")
    args = parser.parse_args()

    data_dir = Path(args.folder).resolve()
    if not data_dir.exists():
        logger.error("DATA_DIR_NOT_FOUND", extra={"path": str(data_dir)})
        sys.exit(1)

    orchestrator = PipelineOrchestrator(
        data_dir=data_dir, 
        workers=args.workers,
        skip_macro=args.skip_macro,
        force_macro=args.force_macro
    )
    orchestrator.execute_dag()

if __name__ == "__main__":
    main()
