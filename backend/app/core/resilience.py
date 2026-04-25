import logging
from functools import wraps
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import time

logger = logging.getLogger(__name__)

# --- RETRY POLICY ---
# Retries up to 3 times with exponential backoff (2s, 4s, 8s)
# Specifically for transient connection errors or rate limits
standard_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)

# --- CIRCUIT BREAKER ---
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED" # CLOSED, OPEN, HALF-OPEN

    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF-OPEN"
                    logger.info(f"Circuit Breaker for {func.__name__} shifted to HALF-OPEN")
                else:
                    logger.warning(f"Circuit Breaker for {func.__name__} is OPEN. Blocking request.")
                    return f"Service {func.__name__} is temporarily unavailable (Circuit Breaker OPEN)."

            try:
                result = await func(*args, **kwargs)
                if self.state == "HALF-OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info(f"Circuit Breaker for {func.__name__} shifted to CLOSED")
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.error(f"Circuit Breaker for {func.__name__} shifted to OPEN after {self.failure_count} failures.")
                raise e
        return wrapper

# Individual breakers for different tool types
llm_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
tool_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
