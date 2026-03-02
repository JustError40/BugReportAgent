"""
Universal message model and abstract source interface.

Any messenger (Telegram, Discord, Slack, …) must produce ``UniversalMessage``
objects.  The agent ONLY depends on ``BaseSource`` — it never imports
messenger-specific code.

Universal Messaging Contract (MCP side)
────────────────────────────────────────
A source MCP server must expose these tools:

  get_updates(cursor, limit)  → { messages: [...], next_cursor: int }
  send_message(chat_id, text, reply_to_message_id?)
  get_file_url(file_id)       → { url: str, ... }
  list_chats()                → { chats: [...] }

See telegram_mcp_server/src/server.py for the reference implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional


@dataclass
class UniversalMessage:
    """
    Messenger-agnostic message representation.

    Field semantics are intentionally generic so the same struct can be
    populated from Telegram, Discord, Slack, Matrix, etc.
    """

    # Opaque store ID issued by the source MCP server (used as cursor).
    id: int

    # Platform-native message identifier (e.g. Telegram message_id).
    message_id: int

    # Chat / channel / room identifier.
    chat_id: int

    # Plain text content (caption included).
    text: str

    # Sender handle (e.g. @username).
    sender_username: Optional[str]

    # Sender display name.
    sender_first_name: Optional[str]

    # Unix timestamp.
    date: int

    # If this message is a reply, the platform-native ID of the parent.
    reply_to_message_id: Optional[int] = None

    # Opaque file identifiers attached to the message.
    media_file_ids: List[str] = field(default_factory=list)

    # True when the user explicitly requested task creation
    # (e.g. replied with @bug-agent).
    is_manual_trigger: bool = False

    # Forum supergroup topic thread ID (Telegram topic / Discord thread).
    thread_id: Optional[int] = None

    # When is_manual_trigger=True: the text content of the ORIGINAL message
    # that the user replied to.  This is the text the agent should analyse
    # and store as a positive training example — NOT the trigger message.
    original_text: Optional[str] = None

    # Which source adapter produced this message (e.g. "telegram", "discord").
    source: str = "unknown"


class BaseSource(ABC):
    """Abstract async message source.

    Concrete implementations connect to an MCP server or directly to a
    platform's API and yield ``UniversalMessage`` objects.
    """

    @abstractmethod
    async def __aenter__(self) -> "BaseSource": ...

    @abstractmethod
    async def __aexit__(self, *args) -> None: ...

    @abstractmethod
    def poll(self) -> AsyncIterator[UniversalMessage]:
        """Yield messages indefinitely until the context is exited."""
        ...

    async def send_reply(
        self,
        chat_id: int,
        text: str,
        reply_to_message_id: Optional[int] = None,
    ) -> bool:
        """Send a reply to the source chat.

        Returns:
            True when delivery was accepted by the source connector,
            False when the source is unavailable or delivery failed.
        """
        return False
