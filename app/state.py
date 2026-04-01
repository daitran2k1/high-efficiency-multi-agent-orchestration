from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """
    The state of the multi-agent system.
    Using `add_messages` ensures that conversation history is appended correctly.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_agent: str  # Tracks which expert should handle the next turn
    response_metrics: dict[str, Any]
