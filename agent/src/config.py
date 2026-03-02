"""BugAgent — configuration loader.

Reads values from env vars (via pydantic-settings) and config.yaml.
All components import from here instead of touching os.environ directly.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ── LLM provider ──────────────────────────────────────────────────
    llm_provider: str = Field("ollama", alias="LLM_PROVIDER")
    ollama_base_url: str = Field("http://ollama:11434", alias="OLLAMA_BASE_URL")
    ollama_llm_model: str = Field("llama3.2", alias="OLLAMA_LLM_MODEL")
    ollama_embed_model: str = Field("nomic-embed-text", alias="OLLAMA_EMBED_MODEL")
    vllm_base_url: str = Field("http://vllm:8000/v1", alias="VLLM_BASE_URL")
    vllm_api_key: Optional[str] = Field(None, alias="VLLM_API_KEY")
    vllm_model: Optional[str] = Field(None, alias="VLLM_MODEL")
    llamacpp_base_url: str = Field("http://llamacpp:8080/v1", alias="LLAMACPP_BASE_URL")
    llamacpp_api_key: Optional[str] = Field(None, alias="LLAMACPP_API_KEY")
    llamacpp_model: Optional[str] = Field(None, alias="LLAMACPP_MODEL")
    openai_api_key: Optional[str] = Field(None, alias="OPENAI_API_KEY")
    openrouter_api_key: Optional[str] = Field(None, alias="OPENROUTER_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, alias="ANTHROPIC_API_KEY")
    cloud_llm_model: Optional[str] = Field(None, alias="CLOUD_LLM_MODEL")

    # ── Qdrant ────────────────────────────────────────────────────────
    qdrant_url: str = Field("http://qdrant:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field("telegram_bugs", alias="QDRANT_COLLECTION")

    # ── MCP endpoints ─────────────────────────────────────────────────
    telegram_mcp_url: str = Field("http://telegram-mcp:3000", alias="TELEGRAM_MCP_URL")
    taiga_mcp_url: str = Field("http://taiga-mcp:8001", alias="TAIGA_MCP_URL")

    # ── Detection thresholds ──────────────────────────────────────────
    rerank_score_threshold: float = Field(0.65, alias="RERANK_SCORE_THRESHOLD")
    bug_classify_threshold: float = Field(0.75, alias="BUG_CLASSIFY_THRESHOLD")

    # ── Reranker runtime ──────────────────────────────────────────────
    rerank_operator: str = Field("qdrant", alias="RERANK_OPERATOR")
    rerank_base_url: Optional[str] = Field(None, alias="RERANK_BASE_URL")
    rerank_model: Optional[str] = Field(None, alias="RERANK_MODEL")

    # ── Telegram ──────────────────────────────────────────────────────
    telegram_chat_ids_raw: str = Field("", alias="TELEGRAM_CHAT_IDS")

    @property
    def telegram_chat_ids(self) -> List[int]:
        return [int(x.strip()) for x in self.telegram_chat_ids_raw.split(",") if x.strip()]

    # ── RabbitMQ ──────────────────────────────────────────────────────
    rabbitmq_url: str = Field("amqp://bugagent:bugagent@rabbitmq:5672/", alias="RABBITMQ_URL")
    rabbitmq_exchange: str = Field("bugagent", alias="RABBITMQ_EXCHANGE")

    # ── Taiga ─────────────────────────────────────────────────────────
    taiga_project_slug: str = Field("", alias="TAIGA_PROJECT_SLUG")

    # ── Task tracker (generic) ────────────────────────────────────────
    task_tracker_provider: str = Field("taiga", alias="TASK_TRACKER_PROVIDER")
    task_tracker_project: Optional[str] = Field(None, alias="TASK_TRACKER_PROJECT")

    # ── Logging ───────────────────────────────────────────────────────
    log_level: str = Field("INFO", alias="LOG_LEVEL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def load_yaml_config(path: str = "/app/config.yaml") -> dict:
    candidates = [Path(path)]
    # Local/dev fallbacks (LangGraph CLI is often started from ./agent)
    candidates.extend([
        Path("config.yaml"),
        Path("../config.yaml"),
    ])

    for p in candidates:
        if p.exists():
            with p.open() as f:
                return yaml.safe_load(f) or {}
    return {}


def load_tags(path: str = "/app/tags.md") -> str:
    candidates = [Path(path), Path("tags.md"), Path("../tags.md")]
    for p in candidates:
        if p.exists():
            return p.read_text()
    return "# No tags file found"
