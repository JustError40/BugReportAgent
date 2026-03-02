"""Taiga MCP client — connects to the pytaiga-mcp streamable-HTTP endpoint."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx
import structlog
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from src.agent.runtime_overrides import get_runtime_override
from src.config import get_settings

logger = structlog.get_logger(__name__)


class TaigaMCPClient:
    """
    Async context-manager MCP client for the pytaiga-mcp service.

    Uses the MCP SDK's streamablehttp_client which handles the full
    MCP session lifecycle: initialize handshake, session ID tracking,
    and proper JSON-RPC framing.
    """

    def __init__(self) -> None:
        cfg = get_settings()
        taiga_mcp_url = str(
            get_runtime_override("mcp.taiga_url", get_runtime_override("mcp.taiga_mcp_url", cfg.taiga_mcp_url))
        )
        self._mcp_url = taiga_mcp_url.rstrip("/") + "/mcp"
        self._session: Optional[ClientSession] = None
        self._exit_stack: Optional[Any] = None

    async def __aenter__(self) -> "TaigaMCPClient":
        from contextlib import AsyncExitStack
        self._exit_stack = AsyncExitStack()
        read, write, _ = await self._exit_stack.enter_async_context(
            streamablehttp_client(self._mcp_url)
        )
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()
        return self

    async def __aexit__(self, *_) -> None:
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None

    # ── low-level tool call ───────────────────────────────────────────

    async def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        assert self._session is not None, "Use as async context manager"
        logger.debug("taiga_mcp.call", tool=tool_name, args=arguments)
        result = await self._session.call_tool(tool_name, arguments)

        def _normalize(payload: Any) -> Any:
            if isinstance(payload, dict):
                if "error" in payload and payload["error"]:
                    raise RuntimeError(str(payload["error"]))
                if "result" in payload and len(payload) == 1:
                    return payload["result"]
            return payload

        # Prefer structured payload when provided by MCP SDK
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            return _normalize(structured)

        content = getattr(result, "content", None) or []
        if not content:
            return {}

        parsed_items: List[Any] = []
        for item in content:
            text = getattr(item, "text", None)
            if text is None:
                continue
            try:
                parsed_items.append(json.loads(text))
            except json.JSONDecodeError:
                msg = text.strip()
                if msg.lower().startswith("error executing tool"):
                    raise RuntimeError(msg)
                parsed_items.append(msg)

        if not parsed_items:
            return {}
        if len(parsed_items) == 1:
            return _normalize(parsed_items[0])
        return _normalize(parsed_items)

    @staticmethod
    def _as_list(payload: Any) -> List[Dict[str, Any]]:
        """Normalize MCP payloads that may be wrapped in {'result': [...]} shape."""
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            nested = payload.get("result")
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
            if isinstance(nested, dict):
                return [nested]
            return [payload]
        return []

    async def _resolve_project_id_by_slug(self, project_slug: str) -> int:
        """Resolve a project slug to project ID.

        Prefer get_project_by_slug (server-side direct endpoint lookup),
        fallback to list_projects for compatibility with older servers.
        """
        try:
            project = await self._call_tool("get_project_by_slug", {"slug": project_slug})
            if isinstance(project, dict) and project.get("id"):
                return int(project["id"])
        except Exception:
            # Fallback below for mixed-version environments
            pass

        projects = self._as_list(await self._call_tool("list_projects", {"verbosity": "minimal"}))

        normalized_slug = project_slug.strip().lower()
        for project in projects:
            if not isinstance(project, dict):
                continue
            slug = str(project.get("slug") or "").strip().lower()
            if slug == normalized_slug:
                return int(project["id"])

        raise ValueError(f"Project with slug '{project_slug}' not found in list_projects")

    @staticmethod
    def _pick_id_by_name(items: List[Dict[str, Any]], preferred_name: str, kind: str) -> int:
        """Pick an item ID by case-insensitive name with fallback to the first item."""
        if not items:
            raise ValueError(f"No {kind} values available in Taiga")

        normalized = (preferred_name or "").strip().lower()
        if normalized:
            for item in items:
                name = str(item.get("name") or "").strip().lower()
                if name == normalized:
                    return int(item["id"])

        # Tolerant fallback for robustness across project setups
        logger.warning(
            "taiga_mcp.name_not_found_fallback",
            kind=kind,
            requested=preferred_name,
            fallback=items[0].get("name"),
        )
        return int(items[0]["id"])

    async def _resolve_issue_ids(
        self,
        project_slug: str,
        issue_type: str,
        status: str,
        priority: str,
        severity: str,
    ) -> Dict[str, int]:
        """Resolve project/type/status/priority/severity names to numeric IDs."""
        project_id = await self._resolve_project_id_by_slug(project_slug)

        statuses = self._as_list(await self._call_tool("get_issue_statuses", {"project_id": project_id}))
        priorities = self._as_list(
            await self._call_tool("get_issue_priorities", {"project_id": project_id})
        )
        severities = self._as_list(
            await self._call_tool("get_issue_severities", {"project_id": project_id})
        )
        types = self._as_list(await self._call_tool("get_issue_types", {"project_id": project_id}))

        return {
            "project_id": project_id,
            "status_id": self._pick_id_by_name(statuses, status, "issue status"),
            "priority_id": self._pick_id_by_name(priorities, priority, "issue priority"),
            "severity_id": self._pick_id_by_name(severities, severity, "issue severity"),
            "type_id": self._pick_id_by_name(types, issue_type, "issue type"),
        }

    async def _resolve_user_story_ids(self, project_slug: str, status: str) -> Dict[str, int]:
        """Resolve project and user story status names to numeric IDs."""
        project_id = await self._resolve_project_id_by_slug(project_slug)
        statuses = self._as_list(
            await self._call_tool("get_user_story_statuses", {"project_id": project_id})
        )

        return {
            "project_id": project_id,
            "status_id": self._pick_id_by_name(statuses, status, "user story status"),
        }

    # ── high-level helpers ────────────────────────────────────────────

    async def create_issue(
        self,
        project_slug: str,
        title: str,
        description: str,
        tags: List[str],
        issue_type: str = "Bug",
        status: str = "New",
        priority: str = "Normal",
        severity: str = "Normal",
    ) -> Dict[str, Any]:
        ids = await self._resolve_issue_ids(
            project_slug=project_slug,
            issue_type=issue_type,
            status=status,
            priority=priority,
            severity=severity,
        )

        kwargs: Dict[str, Any] = {"description": description}
        if tags:
            kwargs["tags"] = tags

        return await self._call_tool(
            "create_issue",
            {
                "project_id": ids["project_id"],
                "subject": title,
                "priority_id": ids["priority_id"],
                "status_id": ids["status_id"],
                "severity_id": ids["severity_id"],
                "type_id": ids["type_id"],
                "kwargs": json.dumps(kwargs),
            },
        )

    async def create_user_story(
        self,
        project_slug: str,
        subject: str,
        description: str,
        tags: List[str],
        status: str = "New",
        placement: str = "kanban",
    ) -> Dict[str, Any]:
        ids = await self._resolve_user_story_ids(project_slug=project_slug, status=status)

        kwargs: Dict[str, Any] = {
            "description": description,
            "status": ids["status_id"],
            # Keep item out of sprint by default; appears in backlog/kanban lanes
            "milestone": None,
        }
        if tags:
            kwargs["tags"] = tags

        if placement == "backlog":
            # Ensure deterministic placement when backlog mode is requested.
            kwargs["backlog_order"] = 1
        elif placement == "kanban":
            kwargs["kanban_order"] = 1

        return await self._call_tool(
            "create_user_story",
            {
                "project_id": ids["project_id"],
                "subject": subject,
                "kwargs": json.dumps(kwargs),
            },
        )

    async def list_issue_candidates(self, project_slug: str, limit: int = 200) -> List[Dict[str, Any]]:
        """List existing issues used for duplicate checks."""
        project_id = await self._resolve_project_id_by_slug(project_slug)
        items = self._as_list(
            await self._call_tool(
                "list_issues",
                {"project_id": project_id, "verbosity": "minimal"},
            )
        )
        return items[: max(1, limit)]

    async def list_user_story_candidates(
        self, project_slug: str, limit: int = 200
    ) -> List[Dict[str, Any]]:
        """List existing user stories used for duplicate checks."""
        project_id = await self._resolve_project_id_by_slug(project_slug)
        items = self._as_list(
            await self._call_tool(
                "list_user_stories",
                {"project_id": project_id, "verbosity": "minimal"},
            )
        )
        return items[: max(1, limit)]

    async def attach_issue_from_url(
        self,
        project_slug: str,
        issue_id: int,
        file_url: str,
        filename: str | None = None,
    ) -> Dict[str, Any]:
        """Attach remote file to an issue by URL."""
        project_id = await self._resolve_project_id_by_slug(project_slug)
        payload: Dict[str, Any] = {
            "project_id": project_id,
            "issue_id": issue_id,
            "file_url": file_url,
        }
        if filename:
            payload["filename"] = filename
        return await self._call_tool("attach_issue_from_url", payload)

    async def attach_user_story_from_url(
        self,
        project_slug: str,
        user_story_id: int,
        file_url: str,
        filename: str | None = None,
    ) -> Dict[str, Any]:
        """Attach remote file to a user story by URL."""
        project_id = await self._resolve_project_id_by_slug(project_slug)
        payload: Dict[str, Any] = {
            "project_id": project_id,
            "user_story_id": user_story_id,
            "file_url": file_url,
        }
        if filename:
            payload["filename"] = filename
        return await self._call_tool("attach_user_story_from_url", payload)

    async def get_project(self, slug: str) -> Dict[str, Any]:
        return await self._call_tool("get_project_by_slug", {"slug": slug})

    async def append_issue_note(self, issue_id: int, note: str) -> Dict[str, Any]:
        current = await self._call_tool("get_issue", {"issue_id": issue_id, "verbosity": "full"})
        description = str((current or {}).get("description") or "")
        updated = description + note
        return await self._call_tool(
            "update_issue",
            {
                "issue_id": issue_id,
                "kwargs": json.dumps({"description": updated}),
                "verbosity": "standard",
            },
        )

    async def append_user_story_note(self, user_story_id: int, note: str) -> Dict[str, Any]:
        current = await self._call_tool(
            "get_user_story",
            {"user_story_id": user_story_id, "verbosity": "full"},
        )
        description = str((current or {}).get("description") or "")
        updated = description + note
        return await self._call_tool(
            "update_user_story",
            {
                "user_story_id": user_story_id,
                "kwargs": json.dumps({"description": updated}),
                "verbosity": "standard",
            },
        )

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(self._mcp_url.replace("/mcp", "/health"))
                return resp.status_code == 200
        except Exception:
            return False
