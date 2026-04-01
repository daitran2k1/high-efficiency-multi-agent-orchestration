from typing import Any

from app.llm_utils import extract_content
from app.prompts import build_router_messages

VALID_ROUTES = {
    "technical_specialist",
    "compliance_auditor",
    "support_concierge",
}


def normalize_route(decision_text: str) -> str:
    lowered = decision_text.strip().lower()
    if lowered in VALID_ROUTES:
        return lowered
    if "technical" in lowered:
        return "technical_specialist"
    if "compliance" in lowered or "auditor" in lowered:
        return "compliance_auditor"
    if "support" in lowered or "concierge" in lowered:
        return "support_concierge"
    return "support_concierge"


def decide_route(user_message: str, model: Any) -> str:
    response = model.invoke(build_router_messages(user_message))
    content = extract_content(response)
    return normalize_route(content)
