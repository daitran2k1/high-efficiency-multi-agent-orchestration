from functools import lru_cache
from hashlib import sha256
from pathlib import Path

from app.config import settings

BASE_MANUAL_TEXT = """
[SECTION 1: CORE BANKING OPERATIONS]
1.1 Account Opening Procedures: All new accounts must undergo KYC (Know Your Customer) verification.
1.2 Transaction Limits: Standard retail accounts are capped at $10,000 per day for external transfers.
1.3 API Access: Technical integrations must use OAuth2.0 with a 3600s token expiry.

[SECTION 2: COMPLIANCE & REGULATORY]
2.1 AML Protocols: Any transaction exceeding $5,000 must be flagged for manual review if the pattern is irregular.
2.2 Data Privacy: Personal Identifiable Information (PII) must be encrypted at rest using AES-256.
2.3 Prohibited Activities: Cryptocurrency-related transactions are currently restricted for Tier 1 accounts.

[SECTION 3: TECHNICAL SPECIFICATIONS]
3.1 System Latency: All internal microservices must respond within 200ms.
3.2 Error Codes: ERR_AUTH_01 (Unauthorized), ERR_BAL_02 (Insufficient Funds).
3.3 Deployment: Blue-green deployment is mandatory for all production-facing compliance tools.
""".strip()


def build_simulated_manual() -> str:
    if settings.simulate_large_manual:
        return "\n\n".join(
            [BASE_MANUAL_TEXT] * max(settings.simulated_manual_repeat_count, 1)
        )
    return BASE_MANUAL_TEXT


@lru_cache(maxsize=1)
def load_manual() -> str:
    """
    Loads the 50-page Internal Operations & Compliance Manual.
    In a real scenario, this would read from a PDF or text file.
    For this assignment, we simulate a ~25,000 token document.
    """
    manual_path = Path(settings.manual_path)

    if manual_path.exists():
        with manual_path.open("r", encoding="utf-8") as f:
            return f.read()

    return build_simulated_manual()


@lru_cache(maxsize=1)
def get_manual_metadata() -> dict[str, str | int]:
    content = load_manual()
    return {
        "path": settings.manual_path,
        "sha256": sha256(content.encode("utf-8")).hexdigest(),
        "characters": len(content),
        "simulate_large_manual": settings.simulate_large_manual,
    }
