"""
http_client.py
Decorator functions and base classes for resilient HTTP and Selenium calls.
Complies with requirement to wrap all external calls with @with_retry.
"""

from functools import wraps
import time
import random
from typing import Callable, Any
from src.utils.observability import get_tenant_logger

logger = get_tenant_logger("http-client")

def with_retry(max_attempts: int = 3, base_delay: float = 1.0, 
               jitter: bool = True, fallback_value: Any = None):
    """
    Decorator to wrap external HTTP or Selenium calls with exponential backoff.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        logger.error(
                            "MAX_RETRIES_REACHED",
                            extra={"func": func.__name__, "error": str(e)}
                        )
                        if fallback_value is not None:
                            return fallback_value
                        raise e
                    
                    # Exponential Backoff with Jitter
                    delay = base_delay * (2 ** (attempts - 1))
                    if jitter:
                        delay *= random.uniform(0.8, 1.2)
                        
                    logger.warning(
                        "RETRYING_EXTERNAL_CALL",
                        extra={
                            "func": func.__name__, 
                            "attempt": attempts, 
                            "delay": round(delay, 2),
                            "error": str(e)
                        }
                    )
                    time.sleep(delay)
        return wrapper
    return decorator


def with_fallback(fallback_value: Any):
    """
    Decorator for non-critical reads. If the function raises an exception,
    return a safe default value rather than crashing the system.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    "TRIGGERED_FALLBACK",
                    extra={"func": func.__name__, "error": str(e), "fallback": fallback_value}
                )
                return fallback_value
        return wrapper
    return decorator
