import time
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from app.config import settings
from app.llm_utils import extract_content, extract_non_stream_response_metrics
from app.manual_loader import load_manual
from app.observability import logger, timed_operation
from app.prompts import ExpertPromptTemplate
from app.routing import decide_route
from app.state import AgentState

# Load the manual once at module level to avoid re-reading disk
MANUAL_CONTENT = load_manual()


@lru_cache(maxsize=1)
def get_model() -> ChatOpenAI:
    """
    Returns a client configured for the assignment's OpenAI-compatible endpoint.
    """
    default_headers = {}
    if settings.api_user_id:
        default_headers["X-User-Id"] = settings.api_user_id

    return ChatOpenAI(
        model=settings.model_name,
        base_url=settings.api_base_url,
        api_key=settings.api_key,
        default_headers=default_headers,
        temperature=0,
    )


class BaseExpert:
    """
    Base class for all expert agents, ensuring prefix alignment for KV Caching.
    """

    def __init__(self, prompt_template: ExpertPromptTemplate):
        self.prompt_template = prompt_template
        self.model = get_model()

    def _get_system_messages(self) -> list[Any]:
        """
        CRITICAL FOR CACHING:
        The large manual is placed FIRST in a SystemMessage.
        This ensures that regardless of which agent is called,
        the first ~25k tokens of the prompt are IDENTICAL.
        """
        return self.prompt_template.build_messages(MANUAL_CONTENT, [])

    def build_messages(self, state: AgentState) -> list[Any]:
        return self.prompt_template.build_messages(MANUAL_CONTENT, state["messages"])

    def invoke(self, state: AgentState) -> dict[str, Any]:
        messages = self.build_messages(state)
        response = self.model.invoke(messages)
        content = extract_content(response)
        metrics = extract_non_stream_response_metrics(response)
        return {
            "content": content,
            "metrics": metrics,
            "message": AIMessage(
                content=f"[{self.prompt_template.role_name}] {content}"
            ),
        }

    async def astream(self, state: AgentState) -> AsyncIterator[dict[str, Any]]:
        messages = self.build_messages(state)
        started_at = time.perf_counter()
        first_token_ms = None
        streamed_parts = []

        # Stream token events as chunks arrive, then emit one final summary event.
        async for chunk in self.model.astream(
            messages,
            stream_options={"include_usage": True},
        ):
            chunk_text = extract_content(chunk)
            if chunk_text and first_token_ms is None:
                first_token_ms = (time.perf_counter() - started_at) * 1000
            if chunk_text:
                streamed_parts.append(chunk_text)
                yield {
                    "type": "token",
                    "content": chunk_text,
                }

        total_stream_duration_ms = (time.perf_counter() - started_at) * 1000
        final_content = "".join(streamed_parts)
        yield {
            "type": "done",
            "content": final_content,
            "metrics": {
                "ttft_ms": first_token_ms,
                "total_stream_duration_ms": total_stream_duration_ms,
                "streamed_characters": len(final_content),
            },
            "message": AIMessage(
                content=f"[{self.prompt_template.role_name}] {final_content}"
            ),
        }

    def __call__(self, state: AgentState) -> dict[str, Any]:
        with timed_operation(f"expert:{self.prompt_template.route_name}"):
            result = self.invoke(state)

        return {
            "messages": [result["message"]],
            "response_metrics": result["metrics"],
        }


# Define the three specialized agents

technical_specialist = BaseExpert(
    ExpertPromptTemplate(
        role_name="Technical Specialist",
        route_name="technical_specialist",
        specific_instructions=(
            "You extract system specifications, API limits, and troubleshooting "
            "steps. Focus on technical feasibility and infrastructure details."
        ),
    )
)

compliance_auditor = BaseExpert(
    ExpertPromptTemplate(
        role_name="Compliance Auditor",
        route_name="compliance_auditor",
        specific_instructions=(
            "You interpret regulatory rules, 'Can/Cannot' constraints, and policy "
            "boundaries. Focus on risk mitigation and legal adherence."
        ),
    )
)

support_concierge = BaseExpert(
    ExpertPromptTemplate(
        role_name="Support Concierge",
        route_name="support_concierge",
        specific_instructions=(
            "You summarize complex procedures into step-by-step guides for "
            "non-technical staff. Use simple, empathetic language."
        ),
    )
)

EXPERTS = {
    "technical_specialist": technical_specialist,
    "compliance_auditor": compliance_auditor,
    "support_concierge": support_concierge,
}


# Router Logic
def router_node(state: AgentState) -> dict[str, str]:
    """
    Classifies the user's intent to route to the correct expert.
    This is a small, fast call.
    """
    last_message = extract_content(state["messages"][-1])
    with timed_operation("router"):
        route_name = decide_route(last_message, get_model())
    logger.info("Routed message to %s", route_name)
    return {"next_agent": route_name}
