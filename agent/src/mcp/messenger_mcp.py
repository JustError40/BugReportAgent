"""Generic MCP client for messenger adapters implementing get_file_url tool."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import httpx

from src.agent.runtime_overrides import get_runtime_override
from src.config import load_yaml_config


class MessengerMCPClient:
    def __init__(self, source_name: str) -> None:
        self._source_name = source_name
        self._base_url = self._resolve_source_mcp_url(source_name)
        self._client: Optional[httpx.AsyncClient] = None

    @staticmethod
    def _resolve_source_mcp_url(source_name: str) -> str:
        rt_url = get_runtime_override(f"mcp.sources.{source_name}")
        if rt_url:
            return str(rt_url).rstrip("/")

        cfg = load_yaml_config()
        for src in (cfg.get("sources", []) if isinstance(cfg, dict) else []):
            if str(src.get("name", "")).strip().lower() == source_name.strip().lower():
                mcp_url = src.get("mcp_url") or src.get("url")
                if mcp_url:
                    return str(mcp_url).rstrip("/")
        raise ValueError(f"No mcp_url configured for source '{source_name}'")

    async def __aenter__(self) -> "MessengerMCPClient":
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=60.0)
        return self

    async def __aexit__(self, *_) -> None:
        if self._client:
            await self._client.aclose()

    async def _call_tool(self, tool: str, args: Dict[str, Any]) -> Any:
        assert self._client is not None
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "id": 1,
            "params": {"name": tool, "arguments": args},
        }
        resp = await self._client.post("/mcp", json=payload)
        resp.raise_for_status()
        body = resp.json()
        if "error" in body:
            raise RuntimeError(f"{self._source_name}-mcp error: {body['error']}")
        content = body.get("result", {}).get("content", [])
        if content:
            return json.loads(content[0].get("text", "null"))
        return None

    async def get_file_url(self, file_id: str) -> str:
        data = await self._call_tool("get_file_url", {"file_id": file_id})
        if isinstance(data, dict) and data.get("url"):
            return str(data["url"])
        raise ValueError(f"No URL returned for file_id={file_id} by source={self._source_name}")
