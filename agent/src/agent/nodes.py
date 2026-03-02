"""LangGraph node implementations."""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict

import structlog

from src.agent.state import TaskDraft
from src.agent.runtime_overrides import get_runtime_override
from src.sources.base import UniversalMessage
from src.config import get_settings, load_tags, load_yaml_config
from src.mcp.messenger_mcp import MessengerMCPClient
from src.qdrant_store.client import QdrantStore
from src.trackers.base import TrackerDraft
from src.trackers.factory import get_task_tracker

logger = structlog.get_logger(__name__)

# Outer asyncio event loop — set by main.py so sync graph nodes running in
# executor threads can schedule coroutines on it via run_coroutine_threadsafe.
_OUTER_LOOP: asyncio.AbstractEventLoop | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_msg(state: dict) -> UniversalMessage:
    m = state["message"]
    if isinstance(m, dict):
        return UniversalMessage(**{k: v for k, v in m.items() if k in UniversalMessage.__dataclass_fields__})
    return m


def _get_llm():
    """Return a LangChain chat model depending on LLM_PROVIDER env var."""
    cfg = get_settings()
    llm_cfg = get_runtime_override("llm", {}) or {}
    provider = str(llm_cfg.get("provider", cfg.llm_provider)).lower().replace("-", "").replace("_", "")
    cloud_model = llm_cfg.get("cloud_model", cfg.cloud_llm_model)

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url=cfg.ollama_base_url,
            model=str(llm_cfg.get("ollama_model", cfg.ollama_llm_model)),
            temperature=0.1,
        )
    elif provider in {"vllm"}:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=str(llm_cfg.get("vllm_api_key", cfg.vllm_api_key or "EMPTY")),
            base_url=str(llm_cfg.get("vllm_base_url", cfg.vllm_base_url)),
            model=str(llm_cfg.get("vllm_model", cfg.vllm_model or cloud_model or "Qwen/Qwen2.5-7B-Instruct")),
            temperature=0.1,
        )
    elif provider in {"llamacpp", "llama.cpp", "llamacppserver"}:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=str(llm_cfg.get("llamacpp_api_key", cfg.llamacpp_api_key or "EMPTY")),
            base_url=str(llm_cfg.get("llamacpp_base_url", cfg.llamacpp_base_url)),
            model=str(llm_cfg.get("llamacpp_model", cfg.llamacpp_model or cloud_model or "local-model")),
            temperature=0.1,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=llm_cfg.get("openai_api_key", cfg.openai_api_key),
            model=str(cloud_model or "gpt-4o-mini"),
            temperature=0.1,
        )
    elif provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=llm_cfg.get("openrouter_api_key", cfg.openrouter_api_key),
            base_url="https://openrouter.ai/api/v1",
            model=str(cloud_model or "openai/gpt-4o-mini"),
            temperature=0.1,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            api_key=llm_cfg.get("anthropic_api_key", cfg.anthropic_api_key),
            model_name=str(cloud_model or "claude-3-haiku-20240307"),
            temperature=0.1,
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def _get_tracker_runtime_config() -> Dict[str, Any]:
    """Load task-tracker behavior flags from config.yaml/env with safe defaults."""
    yaml_cfg = load_yaml_config()
    cfg = get_settings()
    tracker_cfg = yaml_cfg.get("task_tracker", {}) if isinstance(yaml_cfg, dict) else {}
    taiga_cfg = yaml_cfg.get("taiga", {}) if isinstance(yaml_cfg, dict) else {}
    rt_tracker_cfg = get_runtime_override("task_tracker", {}) or get_runtime_override("tracker", {}) or {}

    project = str(rt_tracker_cfg.get("project", tracker_cfg.get("project", ""))).strip()
    if not project:
        project = str(cfg.task_tracker_project or cfg.taiga_project_slug or "").strip()

    return {
        "provider": str(rt_tracker_cfg.get("provider", tracker_cfg.get("provider", cfg.task_tracker_provider))).strip().lower(),
        "project": project,
        "work_item_type": str(
            rt_tracker_cfg.get("work_item_type", tracker_cfg.get("work_item_type", taiga_cfg.get("work_item_type", "issue")))
        ).strip().lower(),
        "placement": str(
            rt_tracker_cfg.get("placement", tracker_cfg.get("placement", taiga_cfg.get("placement", "kanban")))
        ).strip().lower(),
        "kanban_status": str(
            rt_tracker_cfg.get("kanban_status", tracker_cfg.get("kanban_status", taiga_cfg.get("kanban_status", "")))
        ).strip(),
        "backlog_status": str(
            rt_tracker_cfg.get("backlog_status", tracker_cfg.get("backlog_status", taiga_cfg.get("backlog_status", "")))
        ).strip(),
        "default_status": str(
            rt_tracker_cfg.get("default_status", tracker_cfg.get("default_status", taiga_cfg.get("default_status", "New")))
        ).strip(),
        "default_priority": str(
            rt_tracker_cfg.get("default_priority", tracker_cfg.get("default_priority", taiga_cfg.get("default_priority", "Normal")))
        ).strip(),
        "default_type": str(
            rt_tracker_cfg.get("default_type", tracker_cfg.get("default_type", taiga_cfg.get("default_type", "Bug")))
        ).strip(),
        "uniqueness_check_enabled": bool(
            rt_tracker_cfg.get(
                "uniqueness_check_enabled",
                tracker_cfg.get("uniqueness_check_enabled", taiga_cfg.get("uniqueness_check_enabled", True)),
            )
        ),
        "duplicate_title_similarity": float(
            rt_tracker_cfg.get(
                "duplicate_title_similarity",
                tracker_cfg.get("duplicate_title_similarity", taiga_cfg.get("duplicate_title_similarity", 0.92)),
            )
        ),
        "duplicate_scan_limit": int(
            rt_tracker_cfg.get("duplicate_scan_limit", tracker_cfg.get("duplicate_scan_limit", taiga_cfg.get("duplicate_scan_limit", 200)))
        ),
    }


def _to_tracker_draft(draft: TaskDraft) -> TrackerDraft:
    return TrackerDraft(
        title=draft.title,
        description=draft.description,
        tags=draft.tags,
        project=draft.project_slug,
        status=draft.status,
        priority=draft.priority,
        item_type=draft.issue_type,
    )


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _title_similarity(a: str, b: str) -> float:
    return float(difflib.SequenceMatcher(None, _normalize_text(a), _normalize_text(b)).ratio())


def _append_warning(state: dict, *, code: str, error: str, stage: str) -> dict:
    warnings = list(state.get("warnings") or [])
    warnings.append({"code": code, "error": error, "stage": stage})
    return {**state, "warnings": warnings}


# ─────────────────────────────────────────────────────────────────────────────
# Node: deduplicate
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate_node(state: dict) -> dict:
    """Skip messages we have already processed."""
    from src.dedup import SeenMessages
    msg = _get_msg(state)
    seen = SeenMessages()
    key = f"{msg.chat_id}:{msg.message_id}"
    if seen.contains(key):
        logger.info("dedup.skip", key=key)
        return {**state, "skip_reason": "already_processed"}
    seen.add(key)
    logger.info("dedup.new", key=key)
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Node: embed
# ─────────────────────────────────────────────────────────────────────────────

def embed_node(state: dict) -> dict:
    """Embed the message text using the configured embedding model."""
    msg = _get_msg(state)
    # For manual triggers, embed the original (replied-to) message, not the trigger
    text_to_embed = msg.original_text or msg.text
    store = QdrantStore()
    try:
        vector = store.embed_text(text_to_embed)
        logger.info("embed.done", chars=len(text_to_embed))
        return {**state, "embedding": vector}
    except Exception as exc:
        logger.error("embed.error", error=str(exc))
        return {**state, "error": f"Embedding failed: {exc}"}


# ─────────────────────────────────────────────────────────────────────────────
# Node: rerank
# ─────────────────────────────────────────────────────────────────────────────

def rerank_node(state: dict) -> dict:
    """Query Qdrant for similar examples and pick the best rerank score."""
    if state.get("error"):
        return state
    store = QdrantStore()
    msg = _get_msg(state)
    text_to_search = msg.original_text or msg.text
    try:
        results = store.rerank_search(
            query_text=text_to_search,
            query_vector=state["embedding"],
            limit=5,
        )
        best_score = results[0]["score"] if results else 0.0
        logger.info("rerank.done", score=best_score, hits=len(results))
        return {**state, "rerank_score": best_score, "similar_examples": results}
    except Exception as exc:
        logger.error("rerank.error", error=str(exc))
        # Don't abort—just use score 0 so manual triggers still work
        next_state = {**state, "rerank_score": 0.0, "similar_examples": []}
        return _append_warning(next_state, code="rerank_failed", error=str(exc), stage="rerank")


# ─────────────────────────────────────────────────────────────────────────────
# Node: llm_classify
# ─────────────────────────────────────────────────────────────────────────────

_CLASSIFY_SYSTEM = """\
You are a software bug triage assistant. 
Analyze the Telegram message below and decide if it describes a software bug
that should become a task in the issue tracker.

Respond ONLY with valid JSON in this exact schema:
{{
  "is_bug": true | false,
  "confidence": 0.0–1.0,
  "reason": "one-sentence explanation"
}}

Rules:
- Mark as bug if the message reports broken behaviour, crashes, incorrect output, security issues.
- Mark as NOT bug for feature requests, general questions, off-topic chat.
- confidence = your certainty (0.95 = very sure).
"""

_CLASSIFY_USER = """\
Message:
\"\"\"
{text}
\"\"\"

Similar labelled examples from memory (may be empty):
{examples}
"""


def llm_classify_node(state: dict) -> dict:
    """Use the LLM to classify whether the message is a bug report."""
    msg = _get_msg(state)
    examples_txt = json.dumps(
        [{"text": e.get("text", ""), "label": e.get("label", "?")} for e in (state.get("similar_examples") or [])],
        ensure_ascii=False,
        indent=2,
    )
    # For manual triggers, classify the original message, not the @-mention trigger
    text_for_classify = msg.original_text or msg.text
    prompt_user = _CLASSIFY_USER.format(text=text_for_classify, examples=examples_txt)
    try:
        llm = _get_llm()
        from langchain_core.messages import HumanMessage, SystemMessage
        response = llm.invoke([SystemMessage(content=_CLASSIFY_SYSTEM), HumanMessage(content=prompt_user)])
        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:-1])
        parsed = json.loads(raw)
        is_bug: bool = bool(parsed.get("is_bug", False))
        confidence: float = float(parsed.get("confidence", 0.0))
        reason: str = parsed.get("reason", "")
        threshold = float(
            get_runtime_override("detection.bug_classify_threshold", get_settings().bug_classify_threshold)
        )
        # Manual triggers always create a task — bypass confidence threshold
        if getattr(msg, 'is_manual_trigger', False):
            is_bug = True
            confidence = max(confidence, 1.0)
        logger.info("classify.done", is_bug=is_bug, confidence=confidence, reason=reason)
        return {
            **state,
            "is_bug_task": is_bug and confidence >= threshold,
            "bug_confidence": confidence,
            "classification_reason": reason,
        }
    except Exception as exc:
        logger.error("classify.error", error=str(exc))
        return {**state, "error": f"Classification failed: {exc}", "is_bug_task": False}


# ─────────────────────────────────────────────────────────────────────────────
# Node: build_draft
# ─────────────────────────────────────────────────────────────────────────────

_DRAFT_SYSTEM = """\
Ты — бот триажа багов. По сообщению из Telegram сформируй краткую и понятную задачу для Taiga.

Отвечай ТОЛЬКО валидным JSON в этой схеме:
{{
    "title": "короткий заголовок бага ≤80 символов, обязательно на русском",
    "description": "markdown-описание на русском: что произошло, шаги, ожидаемое vs фактическое",
    "tags": ["tag1", "tag2"]   // выбирай только из списка доступных тегов
}}

ВАЖНО:
- Заголовок и описание всегда пиши на русском языке.
- Даже если исходное сообщение не на русском, результат должен быть на русском.

Доступные теги:
{tags}
"""

_DRAFT_USER = """\
Текст сообщения:
\"\"\"
{text}
\"\"\"

Метаданные:
- Автор: {sender}
- Дата: {date}
- Ссылка: {url}
"""


def build_draft_node(state: dict) -> dict:
    """Ask the LLM to build a structured bug draft."""
    msg = _get_msg(state)
    cfg = get_settings()
    tags_md = load_tags()
    dt = datetime.fromtimestamp(msg.date, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tg_url = f"https://t.me/c/{abs(msg.chat_id)}/{msg.message_id}"
    sender = f"@{msg.sender_username}" if msg.sender_username else (msg.sender_first_name or "unknown")
    tracker_runtime = _get_tracker_runtime_config()
    status_for_placement = tracker_runtime["default_status"]
    if tracker_runtime["placement"] == "backlog" and tracker_runtime["backlog_status"]:
        status_for_placement = tracker_runtime["backlog_status"]
    elif tracker_runtime["placement"] == "kanban" and tracker_runtime["kanban_status"]:
        status_for_placement = tracker_runtime["kanban_status"]
    # For manual triggers, draft from the original message content
    text_for_draft = msg.original_text or msg.text
    prompt_user = _DRAFT_USER.format(text=text_for_draft, sender=sender, date=dt, url=tg_url)
    try:
        llm = _get_llm()
        from langchain_core.messages import HumanMessage, SystemMessage
        response = llm.invoke([
            SystemMessage(content=_DRAFT_SYSTEM.format(tags=tags_md)),
            HumanMessage(content=prompt_user),
        ])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:-1])
        parsed = json.loads(raw)
        title = parsed.get("title", f"Баг: {msg.text[:80]}")
        description = parsed.get("description", msg.text)
        tags = parsed.get("tags", [])
        # Append source footer
        footer = f"\n\n---\n**Источник:** {tg_url}  \n**Автор:** {sender}  \n**Дата:** {dt}"
        draft = TaskDraft(
            title=title,
            description=description + footer,
            tags=tags,
            project_slug=tracker_runtime["project"],
            status=status_for_placement,
            priority=tracker_runtime["default_priority"],
            issue_type=tracker_runtime["default_type"],
        )
        logger.info("draft.built", title=title, tags=tags)
        return {**state, "draft": draft}
    except Exception as exc:
        logger.error("draft.error", error=str(exc))
        return {**state, "error": f"Draft building failed: {exc}"}


# ─────────────────────────────────────────────────────────────────────────────
# Node: check_uniqueness
# ─────────────────────────────────────────────────────────────────────────────

def check_uniqueness_node(state: dict) -> dict:
    """Check whether draft already exists in tracker and skip duplicates."""
    draft: TaskDraft = state.get("draft")  # type: ignore[assignment]
    if not draft:
        return {**state, "error": "No draft available for uniqueness check"}

    tracker_runtime = _get_tracker_runtime_config()
    if not tracker_runtime["uniqueness_check_enabled"]:
        return state

    async def _check() -> Dict[str, Any]:
        tracker = get_task_tracker()
        candidates = await tracker.list_candidates(
            _to_tracker_draft(draft),
            limit=tracker_runtime["duplicate_scan_limit"],
        )

        best: Dict[str, Any] | None = None
        best_score = 0.0
        for item in candidates:
            subject = str(item.get("subject") or "")
            score = _title_similarity(draft.title, subject)
            if score > best_score:
                best_score = score
                best = item

        return {
            "best_score": best_score,
            "best": best or {},
            "is_duplicate": best_score >= float(tracker_runtime["duplicate_title_similarity"]),
        }

    try:
        if _OUTER_LOOP is not None and _OUTER_LOOP.is_running():
            fut = asyncio.run_coroutine_threadsafe(_check(), _OUTER_LOOP)
            check_result = fut.result(timeout=120)
        else:
            check_result = asyncio.run(_check())

        if check_result.get("is_duplicate"):
            existing = check_result.get("best") or {}
            logger.info(
                "duplicate.found",
                similarity=check_result.get("best_score"),
                existing_id=existing.get("id"),
                existing_ref=existing.get("ref"),
                existing_subject=existing.get("subject"),
            )
            return {
                **state,
                "skip_reason": "duplicate_in_tracker",
                "duplicate_of": {
                    "id": existing.get("id"),
                    "ref": existing.get("ref"),
                    "subject": existing.get("subject"),
                    "similarity": check_result.get("best_score"),
                },
            }

        logger.info("duplicate.not_found", similarity=check_result.get("best_score"))
        return state
    except Exception as exc:
        # Never block creation because of duplicate-check transport issues.
        logger.warning("duplicate.check_failed", error=str(exc))
        return _append_warning(state, code="duplicate_check_failed", error=str(exc), stage="check_uniqueness")


# ─────────────────────────────────────────────────────────────────────────────
# Node: create_task
# ─────────────────────────────────────────────────────────────────────────────

def create_task_node(state: dict) -> dict:
    """Create a task via configured tracker and store message as positive example."""
    import asyncio
    draft: TaskDraft = state.get("draft")  # type: ignore[assignment]
    if not draft:
        return {**state, "error": "No draft available"}

    tracker_runtime = _get_tracker_runtime_config()

    def _upsert_positive_example(
        *,
        taiga_id: Any,
        taiga_url: str,
        creation_error: str | None = None,
    ) -> None:
        """Persist a confirmed/manual bug example in Qdrant with a valid embedding."""
        msg = _get_msg(state)
        store = QdrantStore()
        training_text = msg.original_text or msg.text

        vector = state.get("embedding")
        if not vector:
            vector = store.embed_text(training_text)

        metadata: Dict[str, Any] = {
            "task_tracker_item_id": taiga_id,
            "task_tracker_url": taiga_url,
            "source": msg.source,
            "manual_trigger": bool(getattr(msg, "is_manual_trigger", False)),
        }
        if creation_error:
            metadata["creation_error"] = creation_error

        store.upsert_example(
            message_id=f"{msg.chat_id}:{msg.message_id}",
            text=training_text,
            vector=vector,
            label="bug",
            metadata=metadata,
        )

    async def _create():
        tracker = get_task_tracker()
        return await tracker.create_item(
            _to_tracker_draft(draft),
            placement=tracker_runtime["placement"],
        )

    async def _attach_media(task_id: int) -> list[dict[str, str]]:
        msg = _get_msg(state)
        if not msg.media_file_ids:
            return []

        tracker = get_task_tracker()
        warnings_local: list[dict[str, str]] = []
        async with MessengerMCPClient(msg.source) as mcp_client:
            for file_id in msg.media_file_ids:
                try:
                    file_url = await mcp_client.get_file_url(file_id)
                    filename = os.path.basename(file_url.split("?")[0]) or f"{file_id}.bin"
                    await tracker.attach_from_url(
                        item_id=task_id,
                        draft=_to_tracker_draft(draft),
                        file_url=file_url,
                        filename=filename,
                    )
                except Exception as exc:
                    logger.warning("task.attachment_failed", file_id=file_id, error=str(exc))
                    warnings_local.append(
                        {
                            "code": "attachment_failed",
                            "error": f"file_id={file_id}: {exc}",
                            "stage": "create_task",
                        }
                    )
        return warnings_local

    try:
        if _OUTER_LOOP is not None and _OUTER_LOOP.is_running():
            future = asyncio.run_coroutine_threadsafe(_create(), _OUTER_LOOP)
            result = future.result(timeout=120)
        else:
            result = asyncio.run(_create())
        task_id = result.get("id")
        task_url = result.get("url", "")
        logger.info("task.created", task_id=task_id, url=task_url)

        attach_warnings: list[dict[str, str]] = []
        if task_id:
            if _OUTER_LOOP is not None and _OUTER_LOOP.is_running():
                fut = asyncio.run_coroutine_threadsafe(_attach_media(int(task_id)), _OUTER_LOOP)
                attach_warnings = fut.result(timeout=180) or []
            else:
                attach_warnings = asyncio.run(_attach_media(int(task_id))) or []

        # Save as positive training example in Qdrant.
        _upsert_positive_example(taiga_id=task_id, taiga_url=task_url)

        out_state = {
            **state,
            "taiga_task_id": task_id,
            "taiga_task_url": task_url,
            "task_tracker_item_id": task_id,
            "task_tracker_item_url": task_url,
        }
        for warning in attach_warnings:
            out_state = _append_warning(
                out_state,
                code=str(warning.get("code") or "attachment_failed"),
                error=str(warning.get("error") or "unknown attachment error"),
                stage=str(warning.get("stage") or "create_task"),
            )
        return out_state
    except Exception as exc:
        logger.error("task.create_error", error=str(exc))
        msg = _get_msg(state)
        # Manual trigger from chat is treated as explicit positive label.
        if getattr(msg, "is_manual_trigger", False):
            try:
                _upsert_positive_example(taiga_id=None, taiga_url="", creation_error=str(exc))
                logger.info("qdrant.manual_bug_upserted_after_create_error")
            except Exception as upsert_exc:
                logger.error("qdrant.manual_bug_upsert_failed", error=str(upsert_exc))
        return {**state, "error": f"Task creation failed: {exc}"}
