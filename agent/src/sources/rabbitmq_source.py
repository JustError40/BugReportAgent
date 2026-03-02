"""RabbitMQ message source.

Consumes messages pushed by any messenger bot (telegram-mcp, discord-mcp, …)
from a durable RabbitMQ queue and yields ``UniversalMessage`` objects.

For outbound calls (send_message, get_file_url) it delegates to the
messenger's MCP HTTP endpoint via a plain httpx client — the same
JSON-RPC approach used by MCPSource.

Config.yaml entry:

    sources:
      - name: telegram
        mcp_url: http://telegram-mcp:3000    # for send_message / get_file_url
        queue_name: telegram.messages        # RabbitMQ routing key / queue name
"""

from __future__ import annotations

import json
from typing import AsyncIterator, Dict, Any, Optional

import aio_pika
import httpx
import structlog

from src.sources.base import BaseSource, UniversalMessage

logger = structlog.get_logger(__name__)


class RabbitMQSource(BaseSource):
    """
    Async RabbitMQ consumer implementing the BaseSource interface.

    • Declares a durable queue and binds it to the configured exchange.
    • Each message is ACKed only after processing (manual ack).
    • Outbound replies go through the messenger's MCP HTTP endpoint.
    """

    def __init__(
        self,
        rabbitmq_url: str,
        queue_name: str,
        exchange_name: str = "bugagent",
        source_name: str = "unknown",
        mcp_url: Optional[str] = None,
        prefetch_count: int = 10,
    ) -> None:
        self._rabbitmq_url = rabbitmq_url
        self._queue_name = queue_name
        self._exchange_name = exchange_name
        self._source_name = source_name
        self._mcp_base_url = mcp_url.rstrip("/") if mcp_url else None
        self._prefetch_count = prefetch_count

        self._connection: Optional[aio_pika.RobustConnection] = None
        self._channel: Optional[aio_pika.RobustChannel] = None
        self._queue: Optional[aio_pika.Queue] = None
        self._http: Optional[httpx.AsyncClient] = None

    # ── Context manager ───────────────────────────────────────────────

    async def __aenter__(self) -> "RabbitMQSource":
        self._connection = await aio_pika.connect_robust(self._rabbitmq_url)
        self._channel = await self._connection.channel()
        await self._channel.set_qos(prefetch_count=self._prefetch_count)

        exchange = await self._channel.declare_exchange(
            self._exchange_name, aio_pika.ExchangeType.DIRECT, durable=True
        )
        self._queue = await self._channel.declare_queue(self._queue_name, durable=True)
        await self._queue.bind(exchange, routing_key=self._queue_name)

        if self._mcp_base_url:
            self._http = httpx.AsyncClient(
                base_url=self._mcp_base_url,
                timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
            )

        logger.info(
            "rabbitmq_source.connected",
            queue=self._queue_name,
            exchange=self._exchange_name,
            source=self._source_name,
        )
        return self

    async def __aexit__(self, *_) -> None:
        if self._http:
            try:
                await self._http.aclose()
            except Exception:
                pass
        if self._connection:
            try:
                await self._connection.close()
            except Exception:
                pass
        logger.info("rabbitmq_source.disconnected", source=self._source_name)

    # ── Polling ───────────────────────────────────────────────────────

    async def poll(self) -> AsyncIterator[UniversalMessage]:  # type: ignore[override]
        """Consume messages from RabbitMQ queue indefinitely."""
        assert self._queue is not None

        async with self._queue.iterator() as queue_iter:
            async for amqp_msg in queue_iter:
                try:
                    raw: Dict[str, Any] = json.loads(amqp_msg.body)
                    msg = self._to_universal(raw)
                    logger.debug(
                        "rabbitmq_source.received",
                        msg_id=raw.get("id"),
                        chat_id=raw.get("chat_id"),
                        trigger=raw.get("is_manual_trigger"),
                    )
                    yield msg
                    await amqp_msg.ack()
                except Exception as exc:
                    logger.error(
                        "rabbitmq_source.parse_error",
                        error=str(exc),
                        body=amqp_msg.body[:200],
                    )
                    await amqp_msg.nack(requeue=False)

    # ── send_reply ────────────────────────────────────────────────────

    async def send_reply(
        self,
        chat_id: int,
        text: str,
        reply_to_message_id: Optional[int] = None,
        thread_id: Optional[int] = None,
    ) -> bool:
        if not self._http:
            logger.warning("rabbitmq_source.send_reply_skipped", reason="no mcp_url configured")
            return False
        args: Dict[str, Any] = {"chat_id": chat_id, "text": text}
        if reply_to_message_id:
            args["reply_to_message_id"] = reply_to_message_id
        if thread_id:
            args["thread_id"] = thread_id
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "send_message", "arguments": args},
        }
        try:
            resp = await self._http.post("/mcp", json=payload)
            resp.raise_for_status()
            return True
        except Exception as exc:
            logger.warning("rabbitmq_source.send_reply_failed", error=str(exc))
            return False

    async def get_file_url(self, file_id: str) -> Optional[str]:
        """Retrieve a download URL for a file via the messenger MCP endpoint."""
        if not self._http:
            return None
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "get_file_url", "arguments": {"file_id": file_id}},
        }
        try:
            resp = await self._http.post("/mcp", json=payload)
            resp.raise_for_status()
            body = resp.json()
            content = body.get("result", {}).get("content", [])
            if content:
                data = json.loads(content[0].get("text", "{}"))
                return data.get("url")
        except Exception as exc:
            logger.warning("rabbitmq_source.get_file_url_failed", error=str(exc))
        return None

    # ── conversion ────────────────────────────────────────────────────

    def _to_universal(self, raw: Dict[str, Any]) -> UniversalMessage:
        return UniversalMessage(
            id=raw.get("id", 0),
            message_id=raw.get("message_id", 0),
            chat_id=raw.get("chat_id", 0),
            text=raw.get("text", ""),
            sender_username=raw.get("sender_username"),
            sender_first_name=raw.get("sender_first_name"),
            date=raw.get("date", 0),
            reply_to_message_id=raw.get("reply_to_message_id"),
            media_file_ids=raw.get("media_file_ids", []),
            is_manual_trigger=raw.get("is_manual_trigger", False),
            thread_id=raw.get("thread_id"),
            original_text=raw.get("original_text"),
            source=self._source_name,
        )
