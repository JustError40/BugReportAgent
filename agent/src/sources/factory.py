from __future__ import annotations

from typing import Dict

from src.config import get_settings, load_yaml_config
from src.sources.base import BaseSource
from src.sources.mcp_source import MCPSource
from src.sources.rabbitmq_source import RabbitMQSource


def build_source(source_cfg: Dict) -> BaseSource:
    cfg = get_settings()
    agent_cfg = load_yaml_config().get("agent", {})

    name = source_cfg.get("name", "unknown")
    transport = str(source_cfg.get("transport", "")).strip().lower()
    queue_name = source_cfg.get("queue_name")
    mcp_url = source_cfg.get("mcp_url") or source_cfg.get("url", "")

    if not transport:
        transport = "rabbitmq" if queue_name else "mcp"

    if transport == "rabbitmq":
        if not queue_name:
            raise ValueError(f"source '{name}' transport=rabbitmq requires queue_name")
        return RabbitMQSource(
            rabbitmq_url=source_cfg.get("rabbitmq_url") or cfg.rabbitmq_url,
            queue_name=queue_name,
            exchange_name=source_cfg.get("exchange_name") or cfg.rabbitmq_exchange,
            source_name=name,
            mcp_url=mcp_url or None,
            prefetch_count=int(source_cfg.get("prefetch_count", 10)),
        )

    if transport == "mcp":
        if not mcp_url:
            raise ValueError(f"source '{name}' transport=mcp requires mcp_url")
        return MCPSource(
            mcp_url=mcp_url,
            source_name=name,
            poll_interval=float(source_cfg.get("poll_interval_sec", agent_cfg.get("poll_interval_sec", 2))),
        )

    raise ValueError(f"Unsupported source transport '{transport}' for source '{name}'")
