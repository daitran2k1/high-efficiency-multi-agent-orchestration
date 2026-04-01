from typing import Any

from langgraph.graph import END, StateGraph

from app.agents import (
    compliance_auditor,
    router_node,
    support_concierge,
    technical_specialist,
)
from app.state import AgentState


def create_workflow() -> Any:
    """
    Creates and compiles the LangGraph state machine.
    Durable conversation persistence is handled separately in SQLite.
    """
    workflow = StateGraph(AgentState)

    # 1. Add nodes for each expert and the router
    workflow.add_node("router", router_node)
    workflow.add_node("technical_specialist", technical_specialist)
    workflow.add_node("compliance_auditor", compliance_auditor)
    workflow.add_node("support_concierge", support_concierge)

    # 2. Add edges for routing
    workflow.set_entry_point("router")

    # Routing logic: Based on the "next_agent" determined by the router
    workflow.add_conditional_edges(
        "router",
        lambda state: state["next_agent"],
        {
            "technical_specialist": "technical_specialist",
            "compliance_auditor": "compliance_auditor",
            "support_concierge": "support_concierge",
        },
    )

    # All experts transition to END after answering
    workflow.add_edge("technical_specialist", END)
    workflow.add_edge("compliance_auditor", END)
    workflow.add_edge("support_concierge", END)

    return workflow.compile()
