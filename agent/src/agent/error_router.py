"""Standard error classification and delivery fallbacks.

Routing policy:
1) try to notify in chat
2) if chat unavailable and task exists, append note to task
3) if both unavailable, log only
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import structlog

from src.sources.base import BaseSource, UniversalMessage
from src.trackers.base import TrackerDraft
from src.trackers.factory import get_task_tracker

logger = structlog.get_logger(__name__)


@dataclass
class StandardError:
    code: str
    title: str
    details: str


def classify_standard_error(error_text: str) -> StandardError:
    txt = (error_text or "").lower()
    if any(x in txt for x in ("401", "unauthorized", "token_not_valid", "forbidden", "403")):
        return StandardError("auth", "Ошибка авторизации", error_text)
    if any(x in txt for x in ("timeout", "timed out", "deadline exceeded")):
        return StandardError("timeout", "Ошибка таймаута", error_text)
    if any(x in txt for x in ("429", "rate limit", "too many requests")):
        return StandardError("rate_limit", "Превышен лимит запросов", error_text)
    if any(x in txt for x in ("404", "not found")):
        return StandardError("not_found", "Ресурс не найден", error_text)
    if any(x in txt for x in ("connection", "network", "dns", "refused", "unreachable")):
        return StandardError("network", "Сетевая ошибка", error_text)
    if any(x in txt for x in ("400", "validation", "invalid", "bad request", "valueerror")):
        return StandardError("validation", "Ошибка валидации", error_text)
    if any(x in txt for x in ("500", "502", "503", "server error", "internal")):
        return StandardError("server", "Внутренняя ошибка сервиса", error_text)
    return StandardError("unknown", "Неизвестная ошибка", error_text)


def _build_message(err: StandardError) -> str:
    return (
        f"⚠️ {err.title}\n"
        f"Код: `{err.code}`\n"
        f"Детали: {err.details}\n"
        "Я продолжу обработку при следующих сообщениях автоматически."
    )


def _build_task_note(err: StandardError) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        "\n\n---\n"
        f"⚠️ **Agent error** ({ts})\n"
        f"- code: `{err.code}`\n"
        f"- title: {err.title}\n"
        f"- details: {err.details}"
    )


async def route_error_notification(
    *,
    source: Optional[BaseSource],
    message: UniversalMessage,
    error_text: str,
    task_id: Optional[int] = None,
    draft: Optional[TrackerDraft] = None,
) -> str:
    """Route error notification with fallback chain.

    Returns: "chat" | "task" | "log"
    """
    err = classify_standard_error(error_text)
    text = _build_message(err)

    if source is not None:
        try:
            ok = await source.send_reply(
                chat_id=message.chat_id,
                text=text,
                reply_to_message_id=message.message_id,
            )
            if ok:
                logger.info("error.notify_chat", code=err.code, message_id=message.message_id)
                return "chat"
        except Exception as exc:
            logger.warning("error.notify_chat_failed", error=str(exc), code=err.code)

    if task_id and draft:
        try:
            tracker = get_task_tracker()
            await tracker.append_error_note(
                item_id=int(task_id),
                draft=draft,
                note=_build_task_note(err),
            )
            logger.info("error.notify_task", code=err.code, task_id=task_id)
            return "task"
        except Exception as exc:
            logger.warning("error.notify_task_failed", error=str(exc), code=err.code, task_id=task_id)

    logger.error("error.notify_log_only", code=err.code, details=err.details)
    return "log"
