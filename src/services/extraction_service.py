"""
extraction_service.py
Business logic layer for coordinating symbol data extraction from NSE.
Decouples symbols validation, date calculation, and file persistence from raw HTTP.
"""

import concurrent.futures
import time
import random
import re
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

from src.clients.nse_client import NSEClient
from src.models.extraction_models import ExtractionRequest, ExtractionResponse
from src.constants import extraction_constants as const
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("extraction-service")

class ExtractionService:
    """
    Orchestrates the extraction process.
    Injects an NSEClient (Infrastructure) to perform the work.
    """
    
    def __init__(self, nse_client: NSEClient):
        self.client = nse_client

    def compute_date_range(self, days_back: int = 364) -> Tuple[str, str]:
        """Calculates from/to dates for historical data."""
        to_dt = datetime.now().date()
        from_dt = to_dt - timedelta(days=days_back)
        return from_dt.strftime("%d-%m-%Y"), to_dt.strftime("%d-%m-%Y")

    def validate_symbols(self, symbols_raw: str) -> List[str]:
        """
        Validates and cleans input symbols.
        Supports comma-separated or newline-separated strings.
        """
        # Split by comma or whitespace/newlines
        raw_list = re.split(r'[,\s]+', symbols_raw)
        
        # Clean and validate NSE symbol format (Alphanumeric, underscores, hyphens, dots)
        cleaned = []
        pattern = re.compile(r'^[A-Z0-9_\-\.&]{1,25}$')
        
        for s in raw_list:
            s_up = s.strip().upper()
            if s_up and pattern.match(s_up):
                if s_up not in cleaned:
                    cleaned.append(s_up)
            elif s_up:
                logger.warning("INVALID_SYMBOL_FORMAT", extra={"symbol": s_up})
                
        return cleaned

    def extract_symbol_data(self, request: ExtractionRequest, overwrite: bool = False) -> ExtractionResponse:
        """Extracts data for a single symbol and saves it atomically."""
        symbol = request.symbol
        out_path = request.output_dir / f"{symbol}_1Y_{datetime.now().strftime('%Y%m%d')}.csv"
        
        if out_path.exists() and not overwrite:
            logger.info("SYMBOL_ALREADY_EXISTS_SKIPPING", extra={"symbol": symbol})
            return ExtractionResponse(symbol=symbol, success=True, file_path=out_path)
            
        try:
            resp = self.client.fetch_historical_data(symbol, request.from_date, request.to_date)
            
            # Atomic write
            request.output_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile('wb', delete=False, dir=str(request.output_dir), suffix='.tmp') as tf:
                tf.write(resp.content)
                temp_name = tf.name
                
            os.replace(temp_name, out_path)
            logger.info("SYMBOL_EXTRACTED_SUCCESSFULLY", extra={"symbol": symbol, "path": str(out_path)})
            
            # Polite pause to avoid IP blocks
            time.sleep(random.uniform(const.MIN_DELAY, const.MAX_DELAY))
            
            return ExtractionResponse(symbol=symbol, success=True, file_path=out_path)
        except Exception as e:
            logger.error("SYMBOL_EXTRACTION_FAILED", extra={"symbol": symbol, "error": str(e)})
            return ExtractionResponse(symbol=symbol, success=False, error=str(e))

    def run_batch_extraction(self, symbols: List[str], output_dir: Path, 
                             workers: int = 1, overwrite: bool = False) -> List[ExtractionResponse]:
        """Runs extraction for a list of symbols in parallel."""
        from_date, to_date = self.compute_date_range()
        requests_list = [
            ExtractionRequest(symbol=s, from_date=from_date, to_date=to_date, output_dir=output_dir)
            for s in symbols
        ]
        
        results = []
        max_workers = min(workers, const.MAX_WORKERS)
        
        logger.info("STARTING_BATCH_EXTRACTION", extra={"count": len(symbols), "workers": max_workers})
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.extract_symbol_data, req, overwrite): req.symbol 
                for req in requests_list
            }
            for future in concurrent.futures.as_completed(future_to_symbol):
                results.append(future.result())
                
        return results
