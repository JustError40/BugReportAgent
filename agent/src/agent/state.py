"""LangGraph agent state definition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

# Re-export UniversalMessage so the rest of the agent can import from here
from src.sources.base import UniversalMessage  # noqa: F401

# Backwards-compatible alias (used in existing code)
TelegramMessage = UniversalMessage


@dataclass
class TaskDraft:
    """Tracker-agnostic task draft."""

    title: str
    description: str
    tags: List[str]
    attachments: List[str] = field(default_factory=list)   # local file paths
    project_slug: str = ""
    status: str = "New"
    priority: str = "Normal"
    issue_type: str = "Bug"


# Backward-compatible alias
TaigaDraft = TaskDraft


class AgentState(TypedDict, total=False):
    """LangGraph state schema.

    Explicit TypedDict prevents state from collapsing into a single
    ``__root__`` channel and avoids INVALID_CONCURRENT_GRAPH_UPDATE errors.
    """

    # Input
    message: TelegramMessage | Dict[str, Any]

    # Embeddings / retrieval
    embedding: List[float] | None
    rerank_score: float | None
    similar_examples: List[Dict[str, Any]] | None

    # Classification
    is_bug_task: bool | None
    bug_confidence: float | None
    classification_reason: str | None

    # Task creation
    draft: TaskDraft | None
    taiga_task_id: int | None
    taiga_task_url: str | None
    task_tracker_item_id: int | None
    task_tracker_item_url: str | None

    # Control flow
    skip_reason: str | None
    error: str | None
    warnings: List[Dict[str, Any]] | None


def initial_state(msg: TelegramMessage) -> dict:
    """Create a fresh AgentState for a new message."""
    return {
        "message": msg,
        "embedding": None,
        "rerank_score": None,
        "similar_examples": None,
        "is_bug_task": None,
        "bug_confidence": None,
        "classification_reason": None,
        "draft": None,
        "taiga_task_id": None,
        "taiga_task_url": None,
        "skip_reason": None,
        "error": None,
        "warnings": [],
    }
