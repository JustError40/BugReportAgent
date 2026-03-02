"""LangGraph agent graph definition.

Pipeline:
  embed → rerank_check → [skip | llm_classify] → [skip | build_draft] → create_task → done
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agent.runtime_overrides import get_runtime_override
from src.agent.state import AgentState
from src.agent.nodes import (
    build_draft_node,
    check_uniqueness_node,
    create_task_node,
    deduplicate_node,
    embed_node,
    llm_classify_node,
    rerank_node,
)


def _route_after_dedup(state: dict) -> str:
    return "skip" if state.get("skip_reason") else "embed"


def _route_after_rerank(state: dict) -> str:
    if state.get("skip_reason"):
        return "skip"
    if state["message"].get("is_manual_trigger") if isinstance(state["message"], dict) else getattr(state["message"], "is_manual_trigger", False):
        return "llm_classify"           # manual trigger → skip threshold
    score = state.get("rerank_score") or 0.0
    from src.config import get_settings
    threshold = float(
        get_runtime_override(
            "detection.rerank_score_threshold",
            get_settings().rerank_score_threshold,
        )
    )
    return "llm_classify" if score >= threshold else "skip"


def _route_after_classify(state: dict) -> str:
    if state.get("error"):
        return "done"
    return "build_draft" if state.get("is_bug_task") else "done"


def _route_after_build(state: dict) -> str:
    if state.get("error"):
        return "done"
    if state.get("skip_reason"):
        return "done"
    return "create_task"


def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("deduplicate", deduplicate_node)
    g.add_node("embed", embed_node)
    g.add_node("rerank", rerank_node)
    g.add_node("llm_classify", llm_classify_node)
    g.add_node("build_draft", build_draft_node)
    g.add_node("check_uniqueness", check_uniqueness_node)
    g.add_node("create_task", create_task_node)
    # terminal sink — no-op
    g.add_node("skip", lambda s: s)

    g.add_edge(START, "deduplicate")
    g.add_conditional_edges("deduplicate", _route_after_dedup, {"skip": "skip", "embed": "embed"})
    g.add_edge("embed", "rerank")
    g.add_conditional_edges(
        "rerank",
        _route_after_rerank,
        {"skip": "skip", "llm_classify": "llm_classify"},
    )
    g.add_conditional_edges(
        "llm_classify",
        _route_after_classify,
        {"build_draft": "build_draft", "done": END},
    )
    g.add_edge("build_draft", "check_uniqueness")
    g.add_conditional_edges(
        "check_uniqueness",
        _route_after_build,
        {"create_task": "create_task", "done": END},
    )
    g.add_edge("create_task", END)
    g.add_edge("skip", END)

    return g.compile()


# Singleton compiled graph
agent_graph = build_graph()
