import pytest

from app.routing import VALID_ROUTES, decide_route, normalize_route


def test_normalize_route_handles_supported_labels():
    assert normalize_route("technical_specialist") == "technical_specialist"
    assert normalize_route("Compliance Auditor") == "compliance_auditor"
    assert normalize_route("support concierge") == "support_concierge"


@pytest.mark.endpoint
@pytest.mark.parametrize(
    ("prompt", "expected_route"),
    [
        (
            "What is the OAuth token expiry and latency requirement for internal services?",
            "technical_specialist",
        ),
        (
            "Can Tier 1 accounts process cryptocurrency-related transactions?",
            "compliance_auditor",
        ),
        (
            "Explain the account opening process for a new retail customer in simple steps.",
            "support_concierge",
        ),
    ],
)
def test_decide_route_calls_real_endpoint(real_model, prompt, expected_route):
    route_name = decide_route(prompt, real_model)

    assert route_name in VALID_ROUTES
    assert route_name == expected_route
