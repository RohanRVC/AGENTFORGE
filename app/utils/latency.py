import time
import logging
import os
import asyncio
from functools import wraps

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/latency.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def measure_latency(name: str):
    """
    Decorator that works with BOTH sync and async FastAPI endpoints.
    Preserves FastAPI function signature (critical).
    """
    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            @wraps(func)  
            async def async_wrapper(*args, **kwargs):
                start = time.time()
                result = await func(*args, **kwargs)
                end = time.time()
                logging.info(f"[{name}] took {round(end - start, 4)} seconds")
                return result
            
            return async_wrapper

        else:

            @wraps(func) 
            def sync_wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                end = time.time()
                logging.info(f"[{name}] took {round(end - start, 4)} seconds")
                return result
            
            return sync_wrapper

    return decorator
