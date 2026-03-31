import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.graph import create_workflow
from app.manual_loader import get_manual_metadata
from app.observability import configure_logging, logger, timed_operation

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


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # Use a persistent thread ID for conversation state
    thread_id = request.thread_id or str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}

    try:
        with timed_operation("chat_request"):
            initial_state = {"messages": [("user", request.message)]}
            final_state = await workflow.ainvoke(initial_state, config=config)
            last_message = final_state["messages"][-1].content

        return ChatResponse(
            response=last_message,
            thread_id=thread_id,
            routed_agent=final_state.get("next_agent"),
        )
    except Exception as exc:
        logger.exception("Chat request failed for thread_id=%s", thread_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
async def healthcheck():
    manual_metadata = get_manual_metadata()
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "model_provider": settings.model_provider,
        "manual_path": settings.manual_path,
        "manual_sha256": manual_metadata["sha256"],
        "manual_characters": manual_metadata["characters"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
