"""
FastMCP server — exposes the Universal Messaging Contract.

Universal Messaging Contract
─────────────────────────────
Any messenger MCP server MUST implement these tools so the agent can
work with it without modification:

  send_message(chat_id, text, reply_to_message_id?)
      Send a text message to a chat.
      Response: { "ok": true, "message_id": int }

  get_file_url(file_id)
      Return a temporary download URL for a file/photo.
      Response: { "url": str, "mime_type": str }

  list_chats()
      Return the configured monitored chats.
      Response: { "chats": [{ "id": int, "type": str, "title": str }] }

Note: get_updates is NOT present — messages are delivered via RabbitMQ.
      The agent subscribes to the queue directly using RabbitMQSource.

Topic management tools (Forum supergroups):
  set_topic(chat_id, thread_id)   — add topic to watch list
  stop_topic(chat_id, thread_id)  — remove topic from watch list
  list_topics(chat_id)            — show current topic filter

Tools follow this contract so the agent's RabbitMQSource+MCPSource works
unchanged with any messenger backend (Discord, Slack, WhatsApp, …).
"""

from __future__ import annotations

from typing import Optional

import structlog
from mcp.server.fastmcp import FastMCP

from src.bot import get_bot, get_topic_store
from src.config import get_settings

logger = structlog.get_logger(__name__)

mcp = FastMCP(
    "telegram-mcp",
    instructions="Telegram adapter — implements the Universal Messaging Contract",
)


# ─────────────────────────────────────────────────────────────────────────────
# Tool: send_message
# ─────────────────────────────────────────────────────────────────────────────


@mcp.tool(
    "send_message",
    description="Send a text message to a Telegram chat. Optionally reply to a specific message.",
)
async def send_message(
    chat_id: int,
    text: str,
    reply_to_message_id: Optional[int] = None,
    thread_id: Optional[int] = None,
) -> dict:
    """
    Universal Messaging Contract — send_message.

    Returns:
        { "ok": true, "message_id": int }
    """
    bot = get_bot()
    kwargs: dict = {"chat_id": chat_id, "text": text}
    if reply_to_message_id:
        from aiogram.types import ReplyParameters
        kwargs["reply_parameters"] = ReplyParameters(message_id=reply_to_message_id)
    if thread_id:
        kwargs["message_thread_id"] = thread_id
    sent = await bot.send_message(**kwargs)
    logger.info("mcp.send_message", chat_id=chat_id, msg_id=sent.message_id)
    return {"ok": True, "message_id": sent.message_id}


# ─────────────────────────────────────────────────────────────────────────────
# Tool: get_file_url
# ─────────────────────────────────────────────────────────────────────────────


@mcp.tool(
    "get_file_url",
    description="Get a temporary HTTPS download URL for a Telegram file_id (photo, document, video, …).",
)
async def get_file_url(file_id: str) -> dict:
    """
    Universal Messaging Contract — get_file_url.

    Returns:
        { "url": str, "mime_type": str | null, "file_size": int | null }
    """
    bot = get_bot()
    file = await bot.get_file(file_id)
    token = get_settings().bot_token
    url = f"https://api.telegram.org/file/bot{token}/{file.file_path}"
    logger.debug("mcp.get_file_url", file_id=file_id)
    return {
        "url": url,
        "mime_type": None,          # Telegram doesn't expose mime type here
        "file_size": file.file_size,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool: list_chats
# ─────────────────────────────────────────────────────────────────────────────


@mcp.tool(
    "list_chats",
    description="Return the list of Telegram chat IDs this server is configured to monitor.",
)
async def list_chats() -> dict:
    """
    Universal Messaging Contract — list_chats.

    Returns:
        { "chats": [{ "id": int, "type": "group"|"channel"|"unknown" }] }
    """
    cfg = get_settings()
    chats = []
    for chat_id in cfg.allowed_chat_ids:
        try:
            info = await get_bot().get_chat(chat_id)
            chats.append({
                "id": chat_id,
                "type": info.type,
                "title": info.title or info.full_name or str(chat_id),
            })
        except Exception:
            chats.append({"id": chat_id, "type": "unknown", "title": str(chat_id)})
    return {"chats": chats}


# ─────────────────────────────────────────────────────────────────────────────
# Topic management tools
# ─────────────────────────────────────────────────────────────────────────────


@mcp.tool(
    "set_topic",
    description=(
        "Add a Forum supergroup topic thread_id to the watch list for a chat. "
        "Once at least one thread is registered, only messages from registered "
        "threads are forwarded to the agent."
    ),
)
def set_topic(chat_id: int, thread_id: int) -> dict:
    """Enable monitoring for a specific Forum topic thread."""
    ts = get_topic_store()
    ts.set_forum(chat_id, is_forum=True)
    ts.enable_thread(chat_id, thread_id)
    logger.info("mcp.set_topic", chat_id=chat_id, thread_id=thread_id)
    return {"ok": True, "chat_id": chat_id, "thread_id": thread_id, "action": "enabled"}


@mcp.tool(
    "stop_topic",
    description="Remove a Forum supergroup topic thread_id from the watch list.",
)
def stop_topic(chat_id: int, thread_id: int) -> dict:
    """Disable monitoring for a specific Forum topic thread."""
    ts = get_topic_store()
    ts.disable_thread(chat_id, thread_id)
    logger.info("mcp.stop_topic", chat_id=chat_id, thread_id=thread_id)
    return {"ok": True, "chat_id": chat_id, "thread_id": thread_id, "action": "disabled"}


@mcp.tool(
    "list_topics",
    description=(
        "Return the current topic filter state for a chat. "
        "Empty allowed_threads means all topics are accepted."
    ),
)
def list_topics(chat_id: int) -> dict:
    """Return topic filter state."""
    ts = get_topic_store()
    return {
        "chat_id": chat_id,
        "is_forum": ts.is_forum(chat_id),
        "allowed_threads": ts.get_allowed_threads(chat_id),
    }
