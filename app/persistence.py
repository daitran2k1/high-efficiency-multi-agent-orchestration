import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

from app.config import settings


class ConversationStore:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_threads (
                    thread_id TEXT PRIMARY KEY,
                    messages_json TEXT NOT NULL,
                    next_agent TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.commit()

    def load_thread(self, thread_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT messages_json, next_agent, updated_at
                FROM conversation_threads
                WHERE thread_id = ?
                """,
                (thread_id,),
            ).fetchone()

        if row is None:
            return {"messages": [], "next_agent": None, "updated_at": None}

        return {
            "messages": messages_from_dict(json.loads(row["messages_json"])),
            "next_agent": row["next_agent"],
            "updated_at": row["updated_at"],
        }

    def save_thread(
        self,
        thread_id: str,
        messages: list[BaseMessage],
        next_agent: str | None,
    ) -> None:
        messages_json = json.dumps(messages_to_dict(messages))
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversation_threads (thread_id, messages_json, next_agent)
                VALUES (?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    messages_json = excluded.messages_json,
                    next_agent = excluded.next_agent,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (thread_id, messages_json, next_agent),
            )
            connection.commit()


conversation_store = ConversationStore(settings.state_db_path)
