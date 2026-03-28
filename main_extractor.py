"""
main_extractor.py
New CLI entrypoint for NSE Data Extraction.
Uses the refactored Service/Client architecture with Dependency Injection.
"""

import argparse
import sys
from pathlib import Path

from src.dependencies import get_extraction_service
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("extractor-main")

def main():
    parser = argparse.ArgumentParser(description="Refactored NSE Data Extractor")
    parser.add_argument("--symbols", required=True, help="Comma-separated NSE symbols or @filepath")
    parser.add_argument("--out-dir", default="equity_data", help="Output directory")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--no-selenium", action="store_true", help="Disable selenium cookie warmup")
    parser.add_argument("--headless", action="store_true", default=True, help="Run browser in headless mode")
    
    args = parser.parse_args()
    
    # Resolve symbols
    symbols_input = args.symbols
    if symbols_input.startswith('@'):
        symbol_path = Path(symbols_input[1:])
        if not symbol_path.exists():
            logger.error("SYMBOL_FILE_NOT_FOUND", extra={"path": str(symbol_path)})
            sys.exit(1)
        symbols_raw = symbol_path.read_text()
    else:
        symbols_raw = symbols_input
        
    # Wire dependencies
    # Injection: ExtractionService <- NSEClient
    service = get_extraction_service()
    
    # Process symbols
    symbols = service.validate_symbols(symbols_raw)
    if not symbols:
        logger.error("NO_VALID_SYMBOLS_PROVIDED")
        sys.exit(1)
        
    logger.info("VALIDATED_SYMBOLS", extra={"count": len(symbols)})
    
    # Execute batch
    results = service.run_batch_extraction(
        symbols=symbols,
        output_dir=Path(args.out_dir),
        workers=args.workers,
        overwrite=args.overwrite
    )
    
    # Summary
    success_count = sum(1 for r in results if r.success)
    error_count = len(results) - success_count
    
    logger.info("EXTRACTION_COMPLETE", extra={
        "total": len(results),
        "success": success_count,
        "failed": error_count
    })

if __name__ == "__main__":
    main()
