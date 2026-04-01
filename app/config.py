import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def normalize_api_base_url(raw_url: str | None) -> str:
    if not raw_url:
        return ""

    url = raw_url.strip().rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if url.endswith(suffix):
            return url[: -len(suffix)]
    return url


@dataclass(frozen=True)
class Settings:
    app_name: str = "Bank Agent Orchestrator"
    model_name: str = os.getenv("MODEL_NAME", "gpt-oss-20b-for-hiring").strip()
    api_base_url: str = normalize_api_base_url(os.getenv("API_BASE_URL"))
    api_key: str | None = os.getenv("API_KEY")
    api_user_id: str | None = os.getenv("API_USER_ID")
    manual_path: str = os.getenv("MANUAL_PATH", "compliance_manual.txt").strip()
    simulate_large_manual: bool = os.getenv("SIMULATE_LARGE_MANUAL", "0").strip() == "1"
    simulated_manual_repeat_count: int = int(
        os.getenv("SIMULATED_MANUAL_REPEAT_COUNT", "80").strip()
    )
    state_db_path: str = os.getenv(
        "STATE_DB_PATH", str(Path("data") / "conversation_state.db")
    ).strip()
    log_level: str = os.getenv("LOG_LEVEL", "INFO").strip().upper()

    @property
    def endpoint_ready(self) -> bool:
        api_key = self.api_key or ""
        api_user_id = self.api_user_id or ""

        return all(
            [
                api_key,
                api_user_id,
                self.api_base_url,
                "<provided" not in self.api_base_url,
                "<provided" not in api_key,
                "<provided" not in api_user_id,
            ]
        )


settings = Settings()
