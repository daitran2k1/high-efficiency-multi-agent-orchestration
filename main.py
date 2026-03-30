import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.graph import create_workflow

app = FastAPI(title="Bank Agent Orchestrator")

# Initialize the LangGraph workflow
workflow = create_workflow()


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    thread_id: str


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # Use a persistent thread ID for conversation state
    thread_id = request.thread_id or str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Initial state for the conversation turn
        initial_state = {"messages": [("user", request.message)]}

        # Execute the LangGraph workflow
        final_state = await workflow.ainvoke(initial_state, config=config)

        # Extract the last message from the experts
        last_message = final_state["messages"][-1].content

        return ChatResponse(response=last_message, thread_id=thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
