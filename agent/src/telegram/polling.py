"""Telegram long-polling via the telegram-mcp HTTP service.

The telegram-mcp Node.js service exposes a REST-ish JSON-RPC or REST API.
We use getUpdates polling to receive new messages.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Dict, Any, List, Optional

import httpx
import structlog

from src.agent.state import TelegramMessage
from src.config import get_settings, load_yaml_config

logger = structlog.get_logger(__name__)

MENTION_TRIGGERS = {"@bug-agent", "/bug"}


class TelegramPoller:
    """
    Async generator that yields TelegramMessage objects via long-polling.

    Connects to the telegram-mcp Node.js service which wraps the Telegram API.
    """

    def __init__(self) -> None:
        cfg = get_settings()
        yaml_cfg = load_yaml_config()
        agent_cfg = yaml_cfg.get("agent", {})
        self._base_url = cfg.telegram_mcp_url.rstrip("/")
        self._chat_ids: List[int] = cfg.telegram_chat_ids
        self._poll_interval: int = agent_cfg.get("poll_interval_sec", 2)
        self._offset: int = 0
        self._client: Optional[httpx.AsyncClient] = None
        self._running = False

    async def __aenter__(self) -> "TelegramPoller":
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=35.0)
        self._running = True
        return self

    async def __aexit__(self, *_) -> None:
        self._running = False
        if self._client:
            await self._client.aclose()

    # ── MCP tool call helper ──────────────────────────────────────────

    async def _call_tool(self, tool: str, args: Dict[str, Any]) -> Any:
        assert self._client is not None
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "id": 1,
            "params": {"name": tool, "arguments": args},
        }
        resp = await self._client.post("/mcp", json=payload, timeout=35.0)
        resp.raise_for_status()
        body = resp.json()
        if "error" in body:
            raise RuntimeError(f"telegram-mcp error: {body['error']}")
        import json
        content = body.get("result", {}).get("content", [])
        if content:
            return json.loads(content[0].get("text", "null"))
        return None

    # ── polling loop ──────────────────────────────────────────────────

    async def poll(self) -> AsyncIterator[TelegramMessage]:
        """Yield messages indefinitely until stopped."""
        logger.info("telegram.polling_start", chats=self._chat_ids)
        while self._running:
            try:
                updates = await self._call_tool("getUpdates", {
                    "offset": self._offset,
                    "timeout": self._poll_interval,
                    "allowed_updates": ["message"],
                })
                if updates:
                    for upd in updates:
                        self._offset = max(self._offset, upd.get("update_id", 0) + 1)
                        msg = self._parse_update(upd)
                        if msg:
                            yield msg
            except asyncio.CancelledError:
                break
            except httpx.ReadTimeout:
                pass  # normal for long-polling
            except Exception as exc:
                logger.error("telegram.poll_error", error=str(exc))
                await asyncio.sleep(5)

    def _parse_update(self, update: Dict[str, Any]) -> Optional[TelegramMessage]:
        raw_msg = update.get("message") or update.get("channel_post")
        if not raw_msg:
            return None

        chat_id: int = raw_msg.get("chat", {}).get("id", 0)
        if self._chat_ids and chat_id not in self._chat_ids:
            return None

        text: str = raw_msg.get("text") or raw_msg.get("caption") or ""
        if not text:
            return None  # skip media-only messages with no caption

        message_id: int = raw_msg.get("message_id", 0)
        sender = raw_msg.get("from") or {}
        username: Optional[str] = sender.get("username")
        first_name: Optional[str] = sender.get("first_name")
        date: int = raw_msg.get("date", 0)
        reply_to: Optional[int] = (
            raw_msg.get("reply_to_message", {}) or {}
        ).get("message_id")

        # Detect manual trigger (@bug-agent mention)
        is_manual = any(trigger.lower() in text.lower() for trigger in MENTION_TRIGGERS)

        # Collect media file ids
        media_ids: List[str] = []
        for photo in raw_msg.get("photo", []):
            fid = photo.get("file_id")
            if fid:
                media_ids.append(fid)
        doc = raw_msg.get("document")
        if doc and doc.get("file_id"):
            media_ids.append(doc["file_id"])

        return TelegramMessage(
            message_id=message_id,
            chat_id=chat_id,
            text=text,
            sender_username=username,
            sender_first_name=first_name,
            date=date,
            reply_to_message_id=reply_to,
            media_file_ids=media_ids,
            is_manual_trigger=is_manual,
        )
