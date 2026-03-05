"""BugAgent — application entrypoint.

Message sources are configured in config.yaml under the ``sources`` key.
Each entry needs ``mcp_url`` and ``queue_name`` for RabbitMQ-based delivery,
or just ``url`` for the legacy MCP polling mode.

    sources:
      - name: telegram
        mcp_url: http://telegram-mcp:3000   # send_message / get_file_url
        queue_name: telegram.messages       # RabbitMQ queue (push delivery)
      - name: discord                       # future messenger
        mcp_url: http://discord-mcp:3001
        queue_name: discord.messages

The agent creates one source per entry and runs them concurrently.
To add a new messenger: spin up its MCP+bot server and add a line to
config.yaml — no code changes needed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import time
from dataclasses import asdict
from typing import List

import aio_pika
import structlog

from src.agent.graph import agent_graph
from src.agent.error_router import route_error_notification
from src.agent.state import initial_state
from src.config import get_settings, load_yaml_config
from src.sources.base import BaseSource, UniversalMessage
from src.sources.factory import build_source
from src.trackers.base import TrackerDraft


def _configure_logging() -> None:
    cfg = get_settings()
    yaml_cfg = load_yaml_config()
    log_cfg = yaml_cfg.get("logging", {})
    level_name = cfg.log_level.upper()
    log_format = log_cfg.get("format", "json")

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
            if log_format == "json"
            else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level_name, logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


logger = structlog.get_logger(__name__)


async def _publish_heartbeat_loop() -> None:
    cfg = get_settings()
    if not cfg.service_heartbeat_enabled:
        return

    connection = await aio_pika.connect_robust(cfg.rabbitmq_url)
    try:
        channel = await connection.channel()
        exchange = await channel.declare_exchange(
            cfg.rabbitmq_exchange,
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )
        queue = await channel.declare_queue(cfg.service_heartbeat_routing_key, durable=True)
        await queue.bind(exchange, routing_key=cfg.service_heartbeat_routing_key)

        logger.info(
            "service_health.publisher_started",
            service=cfg.service_name,
            routing_key=cfg.service_heartbeat_routing_key,
            interval_sec=cfg.service_heartbeat_interval_sec,
        )

        while True:
            payload = {
                "service": cfg.service_name,
                "status": "ok",
                "timestamp": int(time.time()),
                "kind": "heartbeat",
            }
            await exchange.publish(
                aio_pika.Message(
                    body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    content_type="application/json",
                    delivery_mode=aio_pika.DeliveryMode.NOT_PERSISTENT,
                ),
                routing_key=cfg.service_heartbeat_routing_key,
            )
            await asyncio.sleep(max(5, cfg.service_heartbeat_interval_sec))
    finally:
        await connection.close()


async def _monitor_service_health_loop() -> None:
    cfg = get_settings()
    if not cfg.service_health_enabled:
        return

    connection = await aio_pika.connect_robust(cfg.rabbitmq_url)
    last_seen: dict[str, int] = {}
    stale_warned: set[str] = set()
    try:
        channel = await connection.channel()
        exchange = await channel.declare_exchange(
            cfg.rabbitmq_exchange,
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )
        queue = await channel.declare_queue(cfg.service_health_queue, durable=True)
        await queue.bind(exchange, routing_key=cfg.service_health_queue)

        logger.info(
            "service_health.monitor_started",
            queue=cfg.service_health_queue,
            stale_sec=cfg.service_health_stale_sec,
        )

        async def _on_msg(amqp_msg: aio_pika.IncomingMessage) -> None:
            async with amqp_msg.process(ignore_processed=True):
                try:
                    payload = json.loads(amqp_msg.body.decode("utf-8"))
                except Exception:
                    logger.warning("service_health.bad_payload", body=amqp_msg.body[:200])
                    return

                service = str(payload.get("service") or "unknown")
                ts = int(payload.get("timestamp") or time.time())
                status = str(payload.get("status") or "unknown")

                last_seen[service] = ts
                stale_warned.discard(service)
                logger.debug("service_health.heartbeat", service=service, status=status, timestamp=ts)

        await queue.consume(_on_msg, no_ack=False)

        while True:
            await asyncio.sleep(max(5, cfg.service_health_check_interval_sec))
            now = int(time.time())
            for service, ts in list(last_seen.items()):
                age = now - ts
                if age > cfg.service_health_stale_sec and service not in stale_warned:
                    logger.warning(
                        "service_health.stale",
                        service=service,
                        age_sec=age,
                        stale_threshold_sec=cfg.service_health_stale_sec,
                    )
                    stale_warned.add(service)
    finally:
        await connection.close()


async def process_message(msg, source: BaseSource | None = None) -> None:
    """Run the LangGraph pipeline for a single message."""
    state = initial_state(msg if isinstance(msg, dict) else msg.__dict__ if hasattr(msg, '__dict__') else msg)
    # Convert dataclass to plain dict if needed
    if hasattr(msg, '__dataclass_fields__'):
        state["message"] = asdict(msg)
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, agent_graph.invoke, state
        )
        if result.get("taiga_task_id") or result.get("task_tracker_item_id"):
            logger.info(
                "pipeline.created_task",
                task_id=result.get("taiga_task_id") or result.get("task_tracker_item_id"),
                url=result.get("taiga_task_url") or result.get("task_tracker_item_url"),
                tracker_item_id=result.get("task_tracker_item_id"),
                tracker_item_url=result.get("task_tracker_item_url"),
                message_id=state["message"].get("message_id") if isinstance(state["message"], dict) else state["message"].message_id,
            )
        elif result.get("skip_reason"):
            logger.debug("pipeline.skipped", reason=result["skip_reason"])
        elif result.get("error"):
            logger.error("pipeline.error", error=result["error"])
            parsed_msg = state["message"] if isinstance(state["message"], UniversalMessage) else UniversalMessage(**state["message"])
            draft_obj = result.get("draft")
            tracker_draft = None
            if draft_obj is not None:
                tracker_draft = TrackerDraft(
                    title=getattr(draft_obj, "title", "") or "",
                    description=getattr(draft_obj, "description", "") or "",
                    tags=list(getattr(draft_obj, "tags", []) or []),
                    project=getattr(draft_obj, "project_slug", "") or "",
                    status=getattr(draft_obj, "status", "New") or "New",
                    priority=getattr(draft_obj, "priority", "Normal") or "Normal",
                    item_type=getattr(draft_obj, "issue_type", "Bug") or "Bug",
                )
            await route_error_notification(
                source=source,
                message=parsed_msg,
                error_text=str(result.get("error")),
                task_id=result.get("task_tracker_item_id"),
                draft=tracker_draft,
            )
        else:
            logger.debug("pipeline.not_a_bug", confidence=result.get("bug_confidence"))

        for warning in (result.get("warnings") or []):
            warn_text = f"{warning.get('stage', 'pipeline')}: {warning.get('error', 'unknown warning')}"
            logger.warning("pipeline.warning", code=warning.get("code"), warning=warn_text)
            parsed_msg = state["message"] if isinstance(state["message"], UniversalMessage) else UniversalMessage(**state["message"])
            draft_obj = result.get("draft")
            tracker_draft = None
            if draft_obj is not None:
                tracker_draft = TrackerDraft(
                    title=getattr(draft_obj, "title", "") or "",
                    description=getattr(draft_obj, "description", "") or "",
                    tags=list(getattr(draft_obj, "tags", []) or []),
                    project=getattr(draft_obj, "project_slug", "") or "",
                    status=getattr(draft_obj, "status", "New") or "New",
                    priority=getattr(draft_obj, "priority", "Normal") or "Normal",
                    item_type=getattr(draft_obj, "issue_type", "Bug") or "Bug",
                )
            await route_error_notification(
                source=source,
                message=parsed_msg,
                error_text=warn_text,
                task_id=result.get("task_tracker_item_id"),
                draft=tracker_draft,
            )
    except Exception as exc:
        logger.error("pipeline.unhandled_exception", error=str(exc), exc_info=True)


async def main() -> None:
    _configure_logging()
    cfg = get_settings()
    yaml_cfg = load_yaml_config()
    agent_cfg = yaml_cfg.get("agent", {})
    max_concurrent = agent_cfg.get("max_concurrent_messages", 10)
    semaphore = asyncio.Semaphore(max_concurrent)

    # ── Build source list from config.yaml ────────────────────────────
    source_configs: List[dict] = yaml_cfg.get("sources", [])
    # Fallback: if nothing in config.yaml, use the telegram-mcp env var
    if not source_configs:
        source_configs = [
            {"name": "telegram", "mcp_url": cfg.telegram_mcp_url}
        ]

    sources = [build_source(s) for s in source_configs]

    logger.info(
        "bugagent.start",
        llm_provider=cfg.llm_provider,
        sources=[s._source_name for s in sources],
        tracker_provider=cfg.task_tracker_provider,
        project=cfg.task_tracker_project or cfg.taiga_project_slug,
    )

    loop = asyncio.get_running_loop()

    # Make the outer event loop available to sync graph nodes running in threads.
    # They use this to schedule coroutines safely via run_coroutine_threadsafe.
    import src.agent.nodes as _nodes_mod
    _nodes_mod._OUTER_LOOP = loop

    async def _shutdown():
        logger.info("bugagent.shutdown")
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(_shutdown()))

    async def _run_source(source: BaseSource) -> None:
        async def _process_with_semaphore(m: UniversalMessage) -> None:
            async with semaphore:
                await process_message(m, source=source)

        try:
            async with source:
                async for msg in source.poll():
                    asyncio.create_task(_process_with_semaphore(msg))
        except asyncio.CancelledError:
            pass  # normal shutdown
        except Exception as exc:
            logger.error("source.fatal_error", source=source._source_name, error=str(exc))

    tasks = [
        *[_run_source(s) for s in sources],
        _monitor_service_health_loop(),
        _publish_heartbeat_loop(),
    ]
    # Run all sources + health monitor concurrently
    await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
