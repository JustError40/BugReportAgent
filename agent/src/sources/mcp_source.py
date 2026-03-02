"""
Generic MCP message source.

Connects to **any** MCP server that implements the Universal Messaging Contract
(see ``sources/base.py``) and yields ``UniversalMessage`` objects.

Usage
─────
    source = MCPSource(
        mcp_url="http://telegram-mcp:3000",
        source_name="telegram",
        poll_interval=2.0,
    )
    async with source:
        async for msg in source.poll():
            await handle(msg)

To add a new messenger the agent doesn't need to change at all — just
point ``mcp_url`` at a server that speaks the same contract.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Dict, Any, Optional

import httpx
import structlog

from src.sources.base import BaseSource, UniversalMessage

logger = structlog.get_logger(__name__)


class MCPSource(BaseSource):
    """
    Generic source that polls any Universal Messaging Contract MCP server.

    It uses a simple HTTP JSON-RPC call approach (POST to /mcp) which is
    the most portable and doesn't require a persistent SSE connection from
    the client side.

    If the target server only exposes SSE transport, set ``use_sse=True``
    and the class will use the MCP Python SDK's ``sse_client`` instead.
    """

    def __init__(
        self,
        mcp_url: str,
        source_name: str = "unknown",
        poll_interval: float = 2.0,
        cursor: int = 0,
        batch_limit: int = 100,
    ) -> None:
        self._base_url = mcp_url.rstrip("/")
        self._source_name = source_name
        self._poll_interval = poll_interval
        self._cursor = cursor
        self._batch_limit = batch_limit
        self._client: Optional[httpx.AsyncClient] = None
        self._running = False

    # ── Context manager ───────────────────────────────────────────────

    async def __aenter__(self) -> "MCPSource":
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
        )
        self._running = True
        logger.info("mcp_source.connected", url=self._base_url, source=self._source_name)
        return self

    async def __aexit__(self, *_) -> None:
        self._running = False
        if self._client:
            await self._client.aclose()

    # ── JSON-RPC helper ───────────────────────────────────────────────

    async def _call_tool(self, tool: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server via JSON-RPC over HTTP POST."""
        assert self._client is not None
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        }
        resp = await self._client.post("/mcp", json=payload)
        resp.raise_for_status()
        body = resp.json()
        if "error" in body:
            raise RuntimeError(f"MCP error from {tool}: {body['error']}")
        content = body.get("result", {}).get("content", [])
        if content and isinstance(content, list):
            return json.loads(content[0].get("text", "{}"))
        return body.get("result", {})

    # ── Polling ───────────────────────────────────────────────────────

    async def poll(self) -> AsyncIterator[UniversalMessage]:  # type: ignore[override]
        """Continuously poll the MCP server for new messages."""
        while self._running:
            try:
                result = await self._call_tool(
                    "get_updates",
                    {"cursor": self._cursor, "limit": self._batch_limit},
                )
                raw_messages = result.get("messages", [])
                if raw_messages:
                    self._cursor = result.get("next_cursor", self._cursor)
                    for raw in raw_messages:
                        yield self._to_universal(raw)
                else:
                    # No new messages — wait before next poll
                    await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                break
            except httpx.ReadTimeout:
                pass
            except Exception as exc:
                logger.error(
                    "mcp_source.poll_error",
                    source=self._source_name,
                    url=self._base_url,
                    error=str(exc),
                )
                await asyncio.sleep(5)

    # ── send_reply ────────────────────────────────────────────────────

    async def send_reply(
        self,
        chat_id: int,
        text: str,
        reply_to_message_id: Optional[int] = None,
    ) -> bool:
        args: Dict[str, Any] = {"chat_id": chat_id, "text": text}
        if reply_to_message_id:
            args["reply_to_message_id"] = reply_to_message_id
        try:
            await self._call_tool("send_message", args)
            return True
        except Exception as exc:
            logger.warning("mcp_source.send_reply_failed", error=str(exc))
            return False

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
