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
    # Finalize and Distribute
    service.finalize_run()
    
    # ------------------------------------------------------------
    # LEGACY SUMMARY PARITY: Signal / Trending / Anomaly Reports
    # ------------------------------------------------------------
    logger.info("\n" + "="*60)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"Files Found: {len(csv_files)}")
    logger.info(f"Successfully Processed: {success_count}")
    logger.info(f"Skipped/Failed: {len(csv_files) - success_count}")
    
    s = service.stats
    logger.info("\nSIGNAL ANALYSIS:")
    logger.info(f"  Total Signals Detected: {s['total_signals']:,}")
    logger.info(f"  Confirmed: {s['confirmed']:,}")
    logger.info(f"  Failed: {s['failed']:,}")
    logger.info(f"  Pending: {s['pending']:,}")
    logger.info(f"  Fire Signals (🔥): {s['fire']:,}")
    
    if s['total_signals'] > 0:
        logger.info(f"  Confirmation Rate: {(s['confirmed']/s['total_signals'])*100:.1f}%")

    # Pattern Distribution Summary
    trending_count = len(list((input_folder / "Trending").glob("*.xlsx")))
    anomaly_count = len(list((input_folder / "Anomaly").glob("*.xlsx")))
    ticker_count = len(list((input_folder / "Ticker").glob("*.xlsx")))
    trigger_count = len(list((input_folder / "Triggers").glob("*.xlsx")))
    effort_count = len(list((input_folder / "Efforts").glob("*.xlsx")))
    eigen_count = len(list((input_folder / "EigenFilter").glob("*.xlsx")))

    logger.info("\nFOLDER DISTRIBUTION:")
    logger.info(f"  Trending:    {trending_count} files")
    logger.info(f"  Anomaly:     {anomaly_count} files")
    logger.info(f"  Ticker:      {ticker_count} files")
    logger.info(f"  Triggers:    {trigger_count} files")
    logger.info(f"  Efforts:     {effort_count} files")
    logger.info(f"  EigenFilter: {eigen_count} files")
    logger.info("="*60)
    logger.info("✅ VSA Analysis Run Finished Successfully")


if __name__ == "__main__":
    main()
