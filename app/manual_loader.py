import os


def load_manual() -> str:
    """
    Loads the 50-page Internal Operations & Compliance Manual.
    In a real scenario, this would read from a PDF or text file.
    For this assignment, we simulate a ~25,000 token document.
    """
    manual_path = os.getenv("MANUAL_PATH", "compliance_manual.txt")

    if os.path.exists(manual_path):
        with open(manual_path, "r", encoding="utf-8") as f:
            return f.read()

    # Simulation: Generating ~25k tokens of semi-realistic bank compliance text
    base_text = """
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

"""
    # Repeat the content to reach the desired token count simulation
    simulated_manual = base_text * 1  # Currently set to 1 for debugging
    return simulated_manual
