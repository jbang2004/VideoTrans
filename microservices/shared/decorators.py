import logging
import functools
import asyncio
import time
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)

def handle_errors(custom_logger: Optional[logging.Logger] = None):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            actual_logger = custom_logger or (getattr(args[0], 'logger', logger) if args else logger)
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.time() - start_time
                actual_logger.debug(f"{func.__name__} completed in {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                actual_logger.error(f"{func.__name__} failed in {elapsed:.2f}s: {e}", exc_info=True)
                raise
        return wrapper
    return decorator
