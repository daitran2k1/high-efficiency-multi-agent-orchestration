import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from app.agents import EXPERTS, get_model
from app.config import settings
from app.graph import create_workflow
from app.llm_utils import extract_content
from app.manual_loader import get_manual_metadata
from app.observability import configure_logging, logger, timed_operation
from app.persistence import conversation_store
from app.routing import decide_route
from app.state import AgentState

configure_logging()

app = FastAPI(title=settings.app_name)

# Initialize the LangGraph workflow
workflow = create_workflow()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    thread_id: str
    routed_agent: Optional[str] = None
    metrics: Optional[dict] = None


def build_thread_messages(
    thread_id: str, user_message: str
) -> tuple[dict[str, Any], list[BaseMessage]]:
    persisted_state = conversation_store.load_thread(thread_id)
    messages = list(persisted_state["messages"])
    messages.append(HumanMessage(content=user_message))
    return persisted_state, messages


def save_final_state(thread_id: str, final_state: dict) -> None:
    conversation_store.save_thread(
        thread_id=thread_id,
        messages=list(final_state["messages"]),
        next_agent=final_state.get("next_agent"),
    )


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    thread_id = request.thread_id or str(uuid.uuid4())

    try:
        with timed_operation("chat_request"):
            _, messages = build_thread_messages(thread_id, request.message)
            initial_state: AgentState = {"messages": messages}
            final_state = await workflow.ainvoke(initial_state)
            save_final_state(thread_id, final_state)
            last_message = extract_content(final_state["messages"][-1])

        return ChatResponse(
            response=last_message,
            thread_id=thread_id,
            routed_agent=final_state.get("next_agent"),
            metrics=final_state.get("response_metrics"),
        )
    except Exception as exc:
        logger.exception("Chat request failed for thread_id=%s", thread_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest) -> StreamingResponse:
    thread_id = request.thread_id or str(uuid.uuid4())

    async def event_stream() -> AsyncIterator[str]:
        try:
            _, messages = build_thread_messages(thread_id, request.message)
            state: AgentState = {"messages": messages}

            route_started_at = time.perf_counter()
            route_name = decide_route(request.message, get_model())
            route_elapsed_ms = (time.perf_counter() - route_started_at) * 1000
            expert = EXPERTS[route_name]

            yield sse_event(
                "route",
                {
                    "thread_id": thread_id,
                    "routed_agent": route_name,
                    "router_latency_ms": route_elapsed_ms,
                },
            )

            final_message = None
            final_metrics = None
            async for item in expert.astream(state):
                if item["type"] == "token":
                    yield sse_event("token", {"content": item["content"]})
                    continue

                final_message = item["message"]
                final_metrics = item["metrics"]
                final_state = {
                    "messages": [*messages, final_message],
                    "next_agent": route_name,
                    "response_metrics": final_metrics,
                }
                save_final_state(thread_id, final_state)
                yield sse_event(
                    "done",
                    {
                        "thread_id": thread_id,
                        "routed_agent": route_name,
                        "response": extract_content(final_message),
                        "metrics": final_metrics,
                    },
                )
        except Exception as exc:
            logger.exception(
                "Streaming chat request failed for thread_id=%s", thread_id
            )
            yield sse_event(
                "error",
                {
                    "thread_id": thread_id,
                    "detail": str(exc),
                },
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
async def healthcheck() -> dict[str, str | int | bool]:
    manual_metadata = get_manual_metadata()
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "model_name": settings.model_name,
        "api_base_url": settings.api_base_url,
        "endpoint_ready": settings.endpoint_ready,
        "manual_path": settings.manual_path,
        "manual_sha256": manual_metadata["sha256"],
        "manual_characters": manual_metadata["characters"],
        "state_db_path": settings.state_db_path,
        "simulate_large_manual": manual_metadata["simulate_large_manual"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
