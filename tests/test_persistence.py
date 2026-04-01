from langchain_core.messages import AIMessage, HumanMessage

from app.persistence import ConversationStore


def test_conversation_store_persists_messages(tmp_path):
    store = ConversationStore(str(tmp_path / "state.db"))
    messages = [
        HumanMessage(content="hello"),
        AIMessage(content="[Support Concierge] hi there"),
    ]

    store.save_thread("thread-1", messages, "support_concierge")
    loaded = store.load_thread("thread-1")

    assert [message.content for message in loaded["messages"]] == [
        "hello",
        "[Support Concierge] hi there",
    ]
    assert loaded["next_agent"] == "support_concierge"
