from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

import aio_pika
import structlog

logger = structlog.get_logger(__name__)


class ServiceHeartbeatPublisher:
    """Publish periodic service heartbeat events to RabbitMQ."""

    def __init__(
        self,
        rabbitmq_url: str,
        exchange_name: str,
        routing_key: str,
        service_name: str,
        interval_sec: int = 30,
    ) -> None:
        self._rabbitmq_url = rabbitmq_url
        self._exchange_name = exchange_name
        self._routing_key = routing_key
        self._service_name = service_name
        self._interval_sec = max(5, int(interval_sec))

        self._connection: Optional[aio_pika.RobustConnection] = None
        self._channel: Optional[aio_pika.RobustChannel] = None
        self._exchange: Optional[aio_pika.Exchange] = None
        self._task: Optional[asyncio.Task] = None
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        self._connection = await aio_pika.connect_robust(self._rabbitmq_url)
        self._channel = await self._connection.channel()
        self._exchange = await self._channel.declare_exchange(
            self._exchange_name,
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )
        queue = await self._channel.declare_queue(self._routing_key, durable=True)
        await queue.bind(self._exchange, routing_key=self._routing_key)

        self._task = asyncio.create_task(self._run(), name="heartbeat-publisher")
        logger.info(
            "heartbeat.started",
            service=self._service_name,
            exchange=self._exchange_name,
            routing_key=self._routing_key,
            interval_sec=self._interval_sec,
        )

    async def _run(self) -> None:
        assert self._exchange is not None
        while not self._stopping.is_set():
            payload = {
                "service": self._service_name,
                "status": "ok",
                "timestamp": int(time.time()),
                "kind": "heartbeat",
            }
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            await self._exchange.publish(
                aio_pika.Message(
                    body=body,
                    content_type="application/json",
                    delivery_mode=aio_pika.DeliveryMode.NOT_PERSISTENT,
                ),
                routing_key=self._routing_key,
            )
            await asyncio.sleep(self._interval_sec)

    async def stop(self) -> None:
        self._stopping.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._connection:
            await self._connection.close()
        logger.info("heartbeat.stopped", service=self._service_name)
