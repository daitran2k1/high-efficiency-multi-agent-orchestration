import logging
import time
from contextlib import contextmanager

from app.config import settings


def configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


logger = logging.getLogger("bank_agent_orchestrator")


@contextmanager
def timed_operation(operation_name: str):
    started_at = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.info("%s completed in %.2f ms", operation_name, elapsed_ms)
