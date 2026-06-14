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
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Custom formatter for structured JSON logging
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            # Include all standard fields
            log_obj = {
                "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "tenant_id": tenant_id,
                "message": record.getMessage()
            }
            
            # Include 'extra' fields (passed via logger.info(msg, extra={...}))
            # These are added directly to the record object by logging
            # We skip internal logging attributes
            internal_attrs = {
                'args', 'asctime', 'created', 'exc_info', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs', 'message',
                'msg', 'name', 'pathname', 'process', 'processName', 'relativeCreated',
                'stack_info', 'thread', 'threadName'
            }
            for key, value in record.__dict__.items():
                if key not in internal_attrs and not key.startswith('_'):
                    log_obj[key] = value

            if record.exc_info:
                log_obj["exception"] = self.formatException(record.exc_info)
            return json.dumps(log_obj)
            
    ch.setFormatter(JsonFormatter())
    logger.addHandler(ch)
    
    # Prevent propagation to root logger to avoid duplicates
    logger.propagate = False
    
    return logger

class MetricsTracker:
    """
    Standardized metrics tracking that emits machine-readable logs.
    Supports counters, gauges, and timing.
    """
    def __init__(self, tenant_id: str = "SYSTEM"):
        self.logger = get_tenant_logger("metrics", tenant_id)
        
    def increment(self, metric_name: str, value: int = 1, tags: dict = None):
        """Increment a counter metric."""
        self._emit_metric("counter", metric_name, value, tags)
        
    def gauge(self, metric_name: str, value: float, tags: dict = None):
        """Record an absolute value metric."""
        self._emit_metric("gauge", metric_name, value, tags)
        
    def timing(self, metric_name: str, duration_ms: float, tags: dict = None):
        """Record a duration metric."""
        self._emit_metric("timing", metric_name, duration_ms, tags)
        
    def _emit_metric(self, metric_type: str, name: str, value: float, tags: dict):
        extra = {
            "is_metric": True,
            "metric_type": metric_type,
            "metric_name": name,
            "metric_value": value,
            "tags": tags or {}
        }
        self.logger.info(f"METRIC [{metric_type.upper()}] {name}: {value}", extra=extra)

    from contextlib import contextmanager
    @contextmanager
    def time_block(self, metric_name: str, tags: dict = None):
        """Context manager to easily time code blocks."""
        import time
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000.0
            self.timing(metric_name, duration_ms, tags)

_METRICS_TRACKER = None

def get_metrics_tracker(tenant_id: str = "SYSTEM") -> MetricsTracker:
    global _METRICS_TRACKER
    if _METRICS_TRACKER is None:
        _METRICS_TRACKER = MetricsTracker(tenant_id)
    return _METRICS_TRACKER

