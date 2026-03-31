from dataclasses import dataclass

from app.prompts import build_router_messages

ROUTE_KEYWORDS = {
    "technical_specialist": {
        "api",
        "latency",
        "oauth",
        "token",
        "system",
        "integration",
        "error",
        "deployment",
        "microservice",
        "spec",
    },
    "compliance_auditor": {
        "policy",
        "regulatory",
        "regulation",
        "compliance",
        "allowed",
        "forbidden",
        "prohibited",
        "aml",
        "kyc",
        "privacy",
        "pii",
        "risk",
        "can i",
        "cannot",
    },
    "support_concierge": {
        "how do i",
        "how to",
        "steps",
        "guide",
        "walk me through",
        "explain",
        "summarize",
        "procedure",
        "help",
    },
}


@dataclass(frozen=True)
class RoutingDecision:
    route_name: str
    source: str


def normalize_route(decision_text: str) -> str:
    lowered = decision_text.strip().lower()
    if "technical" in lowered:
        return "technical_specialist"
    if "compliance" in lowered:
        return "compliance_auditor"
    if "support" in lowered or "concierge" in lowered:
        return "support_concierge"
    return "support_concierge"


def keyword_route(user_message: str) -> str | None:
    lowered = user_message.lower()
    scores = {
        route_name: sum(keyword in lowered for keyword in keywords)
        for route_name, keywords in ROUTE_KEYWORDS.items()
    }
    best_route = max(scores, key=lambda k: scores[k])
    if scores[best_route] == 0:
        return None
    return best_route


def decide_route(user_message: str, model) -> RoutingDecision:
    keyword_match = keyword_route(user_message)
    if keyword_match is not None:
        return RoutingDecision(route_name=keyword_match, source="keyword")

    response = model.invoke(build_router_messages(user_message))
    content = getattr(response, "content", str(response))
    return RoutingDecision(
        route_name=normalize_route(content),
        source="llm",
    )
