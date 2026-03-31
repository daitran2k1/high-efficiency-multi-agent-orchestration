from types import SimpleNamespace

from app.routing import decide_route, keyword_route, normalize_route


def test_keyword_router_prefers_compliance_for_policy_language():
    message = "Is this transfer allowed under AML policy for Tier 1 accounts?"
    assert keyword_route(message) == "compliance_auditor"


def test_keyword_router_prefers_technical_for_api_language():
    message = "What is the OAuth token expiry and API latency requirement?"
    assert keyword_route(message) == "technical_specialist"


def test_normalize_route_handles_supported_labels():
    assert normalize_route("technical_specialist") == "technical_specialist"
    assert normalize_route("Compliance Auditor") == "compliance_auditor"
    assert normalize_route("support concierge") == "support_concierge"


def test_decide_route_falls_back_to_llm_when_no_keyword_matches():
    class StubModel:
        def invoke(self, _messages):
            return SimpleNamespace(content="technical_specialist")

    decision = decide_route("Which expert should handle this?", StubModel())

    assert decision.route_name == "technical_specialist"
    assert decision.source == "llm"
