"""Aiogram bot — receives Telegram updates and publishes to RabbitMQ.

Architecture
────────────
aiogram long-polling  →  handler  →  TopicStore filter  →  RabbitMQ publish
                                              ↑
                               /set_topic / /stop_topic commands

Supergroup / Forum topics
─────────────────────────
Telegram "Forum" supergroups carry a ``message_thread_id`` on every message.
By default all topics are accepted.  Once any ``/set_topic`` is issued inside
a forum chat, only registered threads are accepted going forward.

Commands (must be issued inside the target chat / topic):
  /set_topic   — add current topic thread to the monitored list
  /stop_topic  — remove current topic thread from the monitored list
  /status      — show current topic filter for this chat

Manual trigger
──────────────
When a user replies to a message and includes ``@bug-agent``:
  • ``is_manual_trigger = True``
  • ``original_text``  = text of the replied-to message (what should become a task)
  • ``text``           = the trigger message itself (may contain extra instructions)
This lets the agent create a task from the *original* message and store it as
a positive training example.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from typing import Optional

import aio_pika
import structlog
from aiogram import Bot, Dispatcher, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message

from src.config import get_settings
from src.topic_store import TopicStore

logger = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Message model
# ─────────────────────────────────────────────────────────────────────────────

MANUAL_TRIGGER_KEYWORDS = {"@bug-agent"}


@dataclass
class StoredMessage:
    """Serialisable message published to RabbitMQ."""

    id: int                         # filled by publisher (monotonic counter)
    message_id: int
    chat_id: int
    text: str
    sender_username: Optional[str]
    sender_first_name: Optional[str]
    date: int                       # unix timestamp
    thread_id: Optional[int] = None           # Telegram topic thread_id (Forum)
    reply_to_message_id: Optional[int] = None
    media_file_ids: list = field(default_factory=list)
    is_manual_trigger: bool = False
    # When is_manual_trigger=True: text of the message the user replied to.
    # The agent should use this as the content for task creation + training.
    original_text: Optional[str] = None
    source: str = "telegram"


# ─────────────────────────────────────────────────────────────────────────────
# RabbitMQ publisher
# ─────────────────────────────────────────────────────────────────────────────

class RabbitMQPublisher:
    """Thin wrapper that publishes JSON messages to a RabbitMQ exchange."""

    def __init__(self, url: str, exchange_name: str, routing_key: str) -> None:
        self._url = url
        self._exchange_name = exchange_name
        self._routing_key = routing_key
        self._connection: Optional[aio_pika.RobustConnection] = None
        self._channel: Optional[aio_pika.RobustChannel] = None
        self._exchange: Optional[aio_pika.Exchange] = None
        self._counter = 0
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        self._connection = await aio_pika.connect_robust(self._url)
        self._channel = await self._connection.channel()
        self._exchange = await self._channel.declare_exchange(
            self._exchange_name, aio_pika.ExchangeType.DIRECT, durable=True
        )
        # Ensure the queue exists and is bound
        queue = await self._channel.declare_queue(
            self._routing_key, durable=True
        )
        await queue.bind(self._exchange, routing_key=self._routing_key)
        logger.info(
            "rabbitmq.connected",
            exchange=self._exchange_name,
            routing_key=self._routing_key,
        )

    async def publish(self, msg: StoredMessage) -> None:
        async with self._lock:
            self._counter += 1
            msg.id = self._counter
        body = json.dumps(asdict(msg), ensure_ascii=False).encode()
        await self._exchange.publish(
            aio_pika.Message(
                body=body,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                content_type="application/json",
            ),
            routing_key=self._routing_key,
        )
        logger.debug(
            "rabbitmq.published",
            msg_id=msg.id,
            chat_id=msg.chat_id,
            trigger=msg.is_manual_trigger,
        )

    async def close(self) -> None:
        if self._connection:
            await self._connection.close()


# Module-level singletons (initialised in create_bot())
_publisher: Optional[RabbitMQPublisher] = None
_topic_store: Optional[TopicStore] = None
_bot_instance: Optional[Bot] = None
_bot_username: Optional[str] = None  # e.g. "bugagentbot" (without @)


def get_bot() -> Bot:
    if _bot_instance is None:
        raise RuntimeError("Bot not initialised; call create_bot() first")
    return _bot_instance


def get_topic_store() -> TopicStore:
    if _topic_store is None:
        raise RuntimeError("TopicStore not initialised; call create_bot() first")
    return _topic_store


# ─────────────────────────────────────────────────────────────────────────────
# Aiogram router
# ─────────────────────────────────────────────────────────────────────────────

router = Router()


def _is_chat_allowed(chat_id: int) -> bool:
    allowed = get_settings().allowed_chat_ids
    return not allowed or chat_id in allowed


def _extract_media(msg: Message) -> list:
    ids = []
    if msg.photo:
        ids.append(msg.photo[-1].file_id)
    if msg.document:
        ids.append(msg.document.file_id)
    if msg.video:
        ids.append(msg.video.file_id)
    if msg.audio:
        ids.append(msg.audio.file_id)
    return ids


# ── Admin commands ────────────────────────────────────────────────────────────

@router.message(Command("set_topic"))
async def cmd_set_topic(msg: Message) -> None:
    """Add the current topic thread to the monitored list for this chat."""
    ts = get_topic_store()
    thread_id = msg.message_thread_id
    chat_id = msg.chat.id

    if not thread_id:
        await msg.reply(
            "⚠️ This command must be used inside a <b>topic thread</b> of a Forum supergroup.",
            parse_mode=ParseMode.HTML,
        )
        return

    ts.set_forum(chat_id, is_forum=True)
    ts.enable_thread(chat_id, thread_id)
    logger.info("topic.enable", chat_id=chat_id, thread_id=thread_id)
    await msg.reply(
        f"✅ Topic <b>#{thread_id}</b> added to the watch list for this chat.\n"
        "I will now forward messages from this topic to the bug detector.",
        parse_mode=ParseMode.HTML,
    )


@router.message(Command("stop_topic"))
async def cmd_stop_topic(msg: Message) -> None:
    """Remove the current topic thread from the monitored list."""
    ts = get_topic_store()
    thread_id = msg.message_thread_id
    chat_id = msg.chat.id

    if not thread_id:
        await msg.reply("⚠️ Must be used inside a topic thread.", parse_mode=ParseMode.HTML)
        return

    ts.disable_thread(chat_id, thread_id)
    logger.info("topic.disable", chat_id=chat_id, thread_id=thread_id)
    await msg.reply(
        f"🛑 Topic <b>#{thread_id}</b> removed from the watch list.",
        parse_mode=ParseMode.HTML,
    )


@router.message(Command("status"))
async def cmd_status(msg: Message) -> None:
    """Show topic filter status for the current chat."""
    ts = get_topic_store()
    chat_id = msg.chat.id
    allowed = ts.get_allowed_threads(chat_id)
    is_forum = ts.is_forum(chat_id)

    if not is_forum:
        text = "ℹ️ Regular chat — all messages are forwarded."
    elif not allowed:
        text = "ℹ️ Forum supergroup — <b>all topics</b> are monitored (no filter set)."
    else:
        threads = ", ".join(f"#{t}" for t in allowed)
        text = f"ℹ️ Forum supergroup — monitoring topics: <b>{threads}</b>"

    await msg.reply(text, parse_mode=ParseMode.HTML)


# ── Main message handler ──────────────────────────────────────────────────────

def _has_bot_mention(msg: Message) -> bool:
    """Return True if the message mentions the bot by its actual Telegram username."""
    if not _bot_username:
        return False
    entities = msg.entities or msg.caption_entities or []
    text = msg.text or msg.caption or ""
    for ent in entities:
        if ent.type == "mention":
            mention = text[ent.offset : ent.offset + ent.length]  # e.g. "@bugagentbot"
            if mention.lstrip("@").lower() == _bot_username.lower():
                return True
    return False


@router.message()
async def on_message(msg: Message) -> None:
    """Forward qualifying messages to RabbitMQ."""
    chat_id = msg.chat.id
    thread_id: Optional[int] = msg.message_thread_id

    # ── 1. Chat-level filter ──────────────────────────────────────────
    if not _is_chat_allowed(chat_id):
        return

    # ── 2. Update forum flag if needed ───────────────────────────────
    ts = get_topic_store()
    is_forum_chat = getattr(msg.chat, "is_forum", False) or False
    if is_forum_chat:
        ts.set_forum(chat_id, is_forum=True)

    text = msg.text or msg.caption or ""

    # ── 3. Trigger / ping detection (runs BEFORE topic filter) ───────
    # Bot should always respond to a direct ping regardless of topic rules.
    is_trigger = (
        any(kw.lower() in text.lower() for kw in MANUAL_TRIGGER_KEYWORDS)
        or _has_bot_mention(msg)
    )

    if is_trigger and not msg.reply_to_message:
        # ── 3a. Ping — mention without a reply ───────────────────────
        allowed = ts.get_allowed_threads(chat_id)
        is_forum = ts.is_forum(chat_id)
        if is_forum and allowed:
            topics_str = ", ".join(f"#{t}" for t in allowed)
            topic_line = f"\n📌 Monitored topics: <b>{topics_str}</b>"
        elif is_forum:
            topic_line = "\n📌 All topics are monitored."
        else:
            topic_line = ""
        await msg.reply(
            "✅ <b>BugAgent is active</b> and monitoring this chat."
            + topic_line
            + "\n\nReply to a message with <code>@bug-agent</code> to create a task from it.",
            parse_mode=ParseMode.HTML,
        )
        return

    # ── 4. Topic filter — manual triggers always bypass ──────────────
    # A direct trigger (@mention + reply) must always reach the agent
    # regardless of which topic it comes from.
    if not is_trigger and not ts.is_message_allowed(chat_id, thread_id):
        return

    # ── 5. Skip purely empty messages ────────────────────────────────
    if not text and not msg.photo and not msg.document and not is_trigger:
        return

    # ── 6. Capture original text when triggered via reply ────────────
    original_text: Optional[str] = None
    media_ids = _extract_media(msg)
    if is_trigger and msg.reply_to_message:
        orig = msg.reply_to_message
        original_text = orig.text or orig.caption or ""
        # For manual trigger, attach media from the ORIGINAL message too.
        orig_media = _extract_media(orig)
        if orig_media:
            media_ids = list(dict.fromkeys([*media_ids, *orig_media]))

    stored = StoredMessage(
        id=0,  # set by publisher
        message_id=msg.message_id,
        chat_id=chat_id,
        text=text,
        sender_username=msg.from_user.username if msg.from_user else None,
        sender_first_name=msg.from_user.first_name if msg.from_user else None,
        date=int(msg.date.timestamp()) if msg.date else 0,
        thread_id=thread_id,
        reply_to_message_id=msg.reply_to_message.message_id if msg.reply_to_message else None,
        media_file_ids=media_ids,
        is_manual_trigger=is_trigger,
        original_text=original_text,
    )

    await _publisher.publish(stored)


# ─────────────────────────────────────────────────────────────────────────────
# Bot factory
# ─────────────────────────────────────────────────────────────────────────────

async def create_bot() -> tuple[Bot, Dispatcher, RabbitMQPublisher]:
    global _publisher, _topic_store, _bot_instance, _bot_username
    cfg = get_settings()

    _topic_store = TopicStore(cfg.topic_store_path)

    _publisher = RabbitMQPublisher(
        url=cfg.rabbitmq_url,
        exchange_name=cfg.rabbitmq_exchange,
        routing_key="telegram.messages",
    )
    await _publisher.connect()

    bot = Bot(token=cfg.bot_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    _bot_instance = bot

    me = await bot.get_me()
    _bot_username = me.username
    logger.info("bot.identity", username=_bot_username, id=me.id)

    dp = Dispatcher()
    dp.include_router(router)
    return bot, dp, _publisher


async def run_polling(bot: Bot, dp: Dispatcher) -> None:
    logger.info("bot.polling_start")
    try:
        await dp.start_polling(
            bot,
            allowed_updates=["message", "channel_post"],
        )
    finally:
        logger.info("bot.polling_stop")
        await bot.session.close()
