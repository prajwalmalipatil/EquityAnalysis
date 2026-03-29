"""
main_processor.py
New CLI entrypoint for VSA Analysis.
Uses the refactored Service/Client architecture with Dependency Injection.
"""

import argparse
import sys
import concurrent.futures
import multiprocessing
from pathlib import Path
from typing import List

from src.services.vsa.processor_service import VSAProcessorService
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("vsa-main")

def main():
    parser = argparse.ArgumentParser(description="Refactored VSA Processor")
    parser.add_argument("--folder", required=True, help="Folder containing CSV files")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1), 
                        help="Number of parallel workers")
    
    args = parser.parse_args()
    
    input_folder = Path(args.folder).resolve()
    if not input_folder.exists():
        logger.error("INPUT_FOLDER_NOT_FOUND", extra={"path": str(input_folder)})
        sys.exit(1)
        
    # Setup Service
    service = VSAProcessorService(output_base=input_folder)
    
    # Identify files
    csv_files = list(input_folder.glob("*.csv"))
    if not csv_files:
        logger.error("NO_CSV_FILES_FOUND", extra={"path": str(input_folder)})
        sys.exit(1)
        
    logger.info("VALIDATED_INPUTS", extra={"file_count": len(csv_files), "workers": args.workers})
    
    # Execute batch
    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(service.process_file, f): f for f in csv_files}
        for future in concurrent.futures.as_completed(futures):
            try:
                if future.result():
                    success_count += 1
            except Exception as e:
                logger.error("WORKER_EXECUTION_FAILED", extra={"error": str(e)})
    # Finalize and Distribute (Targeting: 205, 0, 2, 18)
    service.finalize_run()
                
    logger.info("VSA_PROCESSING_COMPLETE", extra={
        "total": len(csv_files),
        "success": success_count,
        "failed": len(csv_files) - success_count
    })

if __name__ == "__main__":
    main()
