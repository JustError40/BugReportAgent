"""Configuration for the Telegram MCP server."""

from __future__ import annotations

from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Telegram Bot API
    bot_token: str = Field(..., alias="TELEGRAM_BOT_TOKEN")

    # Comma-separated chat IDs to accept messages from (empty = accept all)
    allowed_chat_ids_raw: str = Field("", alias="TELEGRAM_CHAT_IDS")

    @property
    def allowed_chat_ids(self) -> List[int]:
        return [int(x.strip()) for x in self.allowed_chat_ids_raw.split(",") if x.strip()]

    # RabbitMQ
    rabbitmq_url: str = Field("amqp://bugagent:bugagent@rabbitmq:5672/", alias="RABBITMQ_URL")
    rabbitmq_exchange: str = Field("bugagent", alias="RABBITMQ_EXCHANGE")
    service_name: str = Field("telegram-mcp", alias="SERVICE_NAME")
    service_heartbeat_enabled: bool = Field(True, alias="SERVICE_HEARTBEAT_ENABLED")
    service_heartbeat_interval_sec: int = Field(30, alias="SERVICE_HEARTBEAT_INTERVAL_SEC")
    service_heartbeat_routing_key: str = Field("service.health", alias="SERVICE_HEARTBEAT_ROUTING_KEY")

    # Topic filter state file
    topic_store_path: str = Field("/app/state/topics.json", alias="TOPIC_STORE_PATH")

    # MCP HTTP server
    mcp_host: str = Field("0.0.0.0", alias="MCP_HOST")
    mcp_port: int = Field(3000, alias="MCP_PORT")

    log_level: str = Field("INFO", alias="LOG_LEVEL")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
