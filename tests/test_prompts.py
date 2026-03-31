from langchain_core.messages import HumanMessage

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
