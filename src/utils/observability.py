"""
observability.py
Provides standardized structured logging for all domains.
Replaces basic print() or standard logging setups to meet observability-lib guidelines.
"""

import logging
import json
from datetime import datetime, timezone
import sys

def get_tenant_logger(name: str, tenant_id: str = "SYSTEM"):
    """
    Returns a configured logger that injects tenant_id and standardizes JSON logs
    for the Observability Council.
    """
    logger = logging.getLogger(name)
    
    # If it already has handlers, it might be cached. Return it.
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Custom formatter for structured JSON logging
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_obj = {
                "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "tenant_id": tenant_id,
                "message": record.getMessage()
            }
            if record.exc_info:
                log_obj["exception"] = self.formatException(record.exc_info)
            return json.dumps(log_obj)
            
    ch.setFormatter(JsonFormatter())
    logger.addHandler(ch)
    
    # Prevent propagation to root logger to avoid duplicates
    logger.propagate = False
    
    return logger
