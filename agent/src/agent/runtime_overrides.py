"""Runtime overrides for LangGraph Studio / CLI runs.

Allows changing MCP endpoints and agent logic parameters from web UI
without editing config files.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Dict, Iterator


_RUNTIME_OVERRIDES: ContextVar[Dict[str, Any]] = ContextVar("runtime_overrides", default={})


def get_runtime_overrides() -> Dict[str, Any]:
    value = _RUNTIME_OVERRIDES.get()
    return value if isinstance(value, dict) else {}


def get_runtime_override(path: str, default: Any = None) -> Any:
    """Read nested override by dot-path, e.g. ``mcp.taiga_url``."""
    current: Any = get_runtime_overrides()
    for key in (path or "").split("."):
        if not key:
            continue
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


@contextmanager
def runtime_overrides_context(overrides: Dict[str, Any] | None) -> Iterator[None]:
    token: Token = _RUNTIME_OVERRIDES.set(overrides or {})
    try:
        yield
    finally:
        _RUNTIME_OVERRIDES.reset(token)
