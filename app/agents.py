from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from app.config import settings
from app.llm_utils import extract_content
from app.manual_loader import load_manual
from app.observability import logger, timed_operation
from app.prompts import ExpertPromptTemplate
from app.routing import decide_route

# Load the manual once at module level to avoid re-reading disk
MANUAL_CONTENT = load_manual()


def get_model():
    """
    Returns a client configured for the assignment's OpenAI-compatible endpoint.
    """
    default_headers = {}
    if settings.api_user_id:
        default_headers["X-User-Id"] = settings.api_user_id

    return ChatOpenAI(
        model=settings.model_name,
        base_url=settings.api_base_url,
        api_key=settings.api_key or "missing-api-key",
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

    def _get_system_messages(self):
        """
        CRITICAL FOR CACHING:
        The large manual is placed FIRST in a SystemMessage.
        This ensures that regardless of which agent is called,
        the first ~25k tokens of the prompt are IDENTICAL.
        """
        return self.prompt_template.build_messages(MANUAL_CONTENT, [])

    def __call__(self, state):
        with timed_operation(f"expert:{self.prompt_template.route_name}"):
            messages = self.prompt_template.build_messages(
                MANUAL_CONTENT, state["messages"]
            )
            response = self.model.invoke(messages)
            content = extract_content(response)

        return {
            "messages": [
                AIMessage(content=f"[{self.prompt_template.role_name}] {content}")
            ]
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


# Router Logic
def router_node(state):
    """
    Classifies the user's intent to route to the correct expert.
    This is a small, fast call.
    """
    last_message = state["messages"][-1].content
    with timed_operation("router"):
        route_name = decide_route(last_message, get_model())
    logger.info("Routed message to %s", route_name)
    return {"next_agent": route_name}
