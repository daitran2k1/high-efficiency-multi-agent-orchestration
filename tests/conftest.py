import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import settings


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: calls the assignment's real model endpoint",
    )


@pytest.fixture(scope="session")
def real_model():
    if not settings.endpoint_ready:
        pytest.skip("Assignment endpoint is not configured in .env")

    from app.agents import get_model

    return get_model()
