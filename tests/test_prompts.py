import pytest
from langchain_core.messages import HumanMessage

from app.agents import compliance_auditor, support_concierge, technical_specialist
from app.prompts import ExpertPromptTemplate


def test_expert_prompt_keeps_manual_as_first_message():
    template = ExpertPromptTemplate(
        role_name="Compliance Auditor",
        route_name="compliance_auditor",
        specific_instructions="Focus on policies.",
    )

    messages = template.build_messages(
        "MANUAL CONTENT",
        [HumanMessage(content="Can Tier 1 accounts use crypto?")],
    )

    assert messages[0].type == "system"
    assert "MANUAL CONTENT" in messages[0].content
    assert "ROLE: Compliance Auditor" in messages[1].content
    assert messages[2].content == "Can Tier 1 accounts use crypto?"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("expert", "user_message", "expected_substring"),
    [
        (
            technical_specialist,
            "What is the OAuth token expiry and latency requirement?",
            "3600",
        ),
        (
            compliance_auditor,
            "Can Tier 1 accounts process cryptocurrency-related transactions?",
            "restrict",
        ),
        (
            support_concierge,
            "Explain the account opening process in simple steps.",
            "kyc",
        ),
    ],
)
def test_expert_agents_call_real_endpoint(
    real_model, expert, user_message, expected_substring
):
    result = expert({"messages": [HumanMessage(content=user_message)]})
    response_text = result["messages"][-1].content.lower()

    assert response_text
    assert expected_substring in response_text
