"""LangGraph Studio entrypoint for interactive dashboard runs.

This graph is designed for `langgraph dev` and accepts runtime overrides from
the web UI to reconfigure MCP endpoints and agent logic on the fly.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agent.graph import agent_graph
from src.agent.runtime_overrides import runtime_overrides_context
from src.agent.state import initial_state
from src.sources.base import UniversalMessage


class StudioState(TypedDict, total=False):
    message: Dict[str, Any] | UniversalMessage
    text: str
    input: str
    chat_id: int
    sender_username: str
    sender_first_name: str
    media_file_ids: list[str]
    is_manual_trigger: bool
    thread_id: int
    original_text: str
    source: str
    runtime_overrides: Dict[str, Any]
    result: Dict[str, Any]


def _to_message(state: StudioState) -> UniversalMessage:
    raw = state.get("message")
    if isinstance(raw, UniversalMessage):
        return raw
    if isinstance(raw, dict):
        return UniversalMessage(**{k: v for k, v in raw.items() if k in UniversalMessage.__dataclass_fields__})

    now = int(time.time())
    text = str(state.get("text") or state.get("input") or "")
    return UniversalMessage(
        id=now,
        message_id=now,
        chat_id=int(state.get("chat_id", 0) or 0),
        text=text,
        sender_username=state.get("sender_username"),
        sender_first_name=state.get("sender_first_name", "studio"),
        date=now,
        reply_to_message_id=None,
        media_file_ids=list(state.get("media_file_ids") or []),
        is_manual_trigger=bool(state.get("is_manual_trigger", False)),
        thread_id=state.get("thread_id"),
        original_text=state.get("original_text"),
        source=str(state.get("source") or "telegram"),
    )


def _run_agent_pipeline(state: StudioState) -> StudioState:
    overrides = state.get("runtime_overrides")
    message = _to_message(state)

    with runtime_overrides_context(overrides if isinstance(overrides, dict) else {}):
        result = agent_graph.invoke(initial_state(message))

    return {
        **state,
        "message": asdict(message),
        "result": result,
        **result,
    }


def build_studio_graph() -> StateGraph:
    g = StateGraph(StudioState)
    g.add_node("run_pipeline", _run_agent_pipeline)
    g.add_edge(START, "run_pipeline")
    g.add_edge("run_pipeline", END)
    return g.compile()


studio_graph = build_studio_graph()
