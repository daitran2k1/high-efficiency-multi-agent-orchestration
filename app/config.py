import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_name: str = "Bank Agent Orchestrator"
    model_provider: str = os.getenv("MODEL_PROVIDER", "google").strip().lower()
    model_name: str = os.getenv("MODEL_NAME", "gemini-flash-latest").strip()
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_api_base: str = os.getenv(
        "OPENAI_API_BASE", "http://localhost:30080/v1/chat/completions"
    ).strip()
    manual_path: str = os.getenv("MANUAL_PATH", "compliance_manual.txt").strip()
    log_level: str = os.getenv("LOG_LEVEL", "INFO").strip().upper()


settings = Settings()
