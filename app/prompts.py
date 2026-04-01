from dataclasses import dataclass
from typing import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

ROUTER_SYSTEM_PROMPT = (
    "You classify banking manual questions into exactly one route.\n"
    "Return only one label: technical_specialist, compliance_auditor, support_concierge.\n\n"
    "Route definitions:\n"
    "- technical_specialist: system specifications, API limits, integrations, errors, troubleshooting, infrastructure.\n"
    "- compliance_auditor: regulations, policy boundaries, can/cannot decisions, restrictions, risk and compliance rules.\n"
    "- support_concierge: step-by-step guidance, operational walkthroughs, simplified explanations for non-technical staff."
)


@dataclass(frozen=True)
class ExpertPromptTemplate:
    role_name: str
    route_name: str
    specific_instructions: str

    def build_messages(
        self, manual_content: str, conversation_messages: Sequence[BaseMessage]
    ) -> list[BaseMessage]:
        return [
            SystemMessage(
                content=f"BANK OPERATIONS & COMPLIANCE MANUAL:\n\n{manual_content}"
            ),
            SystemMessage(
                content=(
                    f"ROLE: {self.role_name}\n"
                    f"ROUTE: {self.route_name}\n"
                    f"INSTRUCTIONS: {self.specific_instructions}\n"
                    "Cite the relevant policy or specification in your answer when possible."
                )
            ),
            *conversation_messages,
        ]


def build_router_messages(user_message: str) -> list[BaseMessage]:
    return [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "Route this banking operations question to the best expert:\n"
                f"{user_message}"
            )
        ),
    ]
