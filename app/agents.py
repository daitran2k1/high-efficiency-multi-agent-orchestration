import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from app.manual_loader import load_manual

# Load variables from .env
load_dotenv()

# Load the manual once at module level to avoid re-reading disk
MANUAL_CONTENT = load_manual()


def get_model():
    """
    Returns a Chat instance configured based on environment variables.
    Supports "custom" (vLLM-style), "openai", or "google".

    NOTE: For the final submission, the "custom" provider should be used
    to point to the bank's high-efficiency endpoint.
    """
    provider = os.getenv(
        "MODEL_PROVIDER", "google"
    )  # Default to google for local testing

    if provider == "google":
        return ChatGoogleGenerativeAI(
            model=os.getenv("MODEL_NAME", "gemini-flash-latest"),
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

    elif provider == "openai":
        return ChatOpenAI(
            model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    # Fallback/Default for the specific bank endpoint (Assignment requirement)
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "Qwen/Qwen3-30B-A3B-Instruct-2507"),
        openai_api_base=os.getenv(
            "OPENAI_API_BASE", "http://localhost:30080/v1/chat/completions"
        ),
        openai_api_key=os.getenv("OPENAI_API_KEY", "dummy-key"),
        temperature=0,
    )


def extract_content(response) -> str:
    """
    Robustly extracts text content from various LLM response formats.
    Handles Gemini's list-of-dicts format.
    """
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Handle list of dicts (common in some Gemini/LangChain versions)
        text_parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            else:
                text_parts.append(str(part))
        return " ".join(text_parts)
    return str(content)


class BaseExpert:
    """
    Base class for all expert agents, ensuring prefix alignment for KV Caching.
    """

    def __init__(self, role_name: str, specific_instructions: str):
        self.role_name = role_name
        self.specific_instructions = specific_instructions
        self.model = get_model()

    def _get_system_messages(self):
        """
        CRITICAL FOR CACHING:
        The large manual is placed FIRST in a SystemMessage.
        This ensures that regardless of which agent is called,
        the first ~25k tokens of the prompt are IDENTICAL.
        """
        return [
            SystemMessage(
                content=f"BANK OPERATIONS & COMPLIANCE MANUAL:\n\n{MANUAL_CONTENT}"
            ),
            SystemMessage(
                content=f"ROLE: {self.role_name}\n\nINSTRUCTIONS: {self.specific_instructions}"
            ),
        ]

    def __call__(self, state):
        messages = self._get_system_messages() + state["messages"]
        response = self.model.invoke(messages)
        content = extract_content(response)

        # We append a helpful tag to identify which agent responded

        return {"messages": [AIMessage(content=f"[{self.role_name}] {content}")]}


# Define the three specialized agents

technical_specialist = BaseExpert(
    role_name="Technical Specialist",
    specific_instructions=(
        "You extract system specifications, API limits, and troubleshooting steps. "
        "Focus on technical feasibility and infrastructure details."
    ),
)

compliance_auditor = BaseExpert(
    role_name="Compliance Auditor",
    specific_instructions=(
        "You interpret regulatory rules, 'Can/Cannot' constraints, and policy boundaries. "
        "Focus on risk mitigation and legal adherence."
    ),
)

support_concierge = BaseExpert(
    role_name="Support Concierge",
    specific_instructions=(
        "You summarize complex procedures into step-by-step guides for non-technical staff. "
        "Use simple, empathetic language."
    ),
)


# Router Logic
def router_node(state):
    """
    Classifies the user's intent to route to the correct expert.
    This is a small, fast call.
    """
    model = get_model()
    last_message = state["messages"][-1].content

    prompt = f"""
    Analyze the user query and route it to one of these three experts:
    1. technical_specialist: For API, system specs, error codes.
    2. compliance_auditor: For policy, what is allowed/forbidden, regulatory.
    3. support_concierge: For general "how-to" and procedure summaries.

    Query: {last_message}

    Respond with ONLY the name of the expert.
    """

    response = model.invoke([HumanMessage(content=prompt)])

    decision = extract_content(response).strip().lower()

    # Simple mapping
    if "technical" in decision:
        return {"next_agent": "technical_specialist"}
    elif "compliance" in decision:
        return {"next_agent": "compliance_auditor"}
    else:
        return {"next_agent": "support_concierge"}
