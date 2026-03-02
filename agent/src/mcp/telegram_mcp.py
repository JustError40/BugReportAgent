"""Telegram MCP client — wraps the telegram-mcp Node.js service for non-polling calls."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx
import structlog

from src.config import get_settings

logger = structlog.get_logger(__name__)


class TelegramMCPClient:
    """Async helper to call telegram-mcp tools (file downloads, message sends, etc.)."""

    def __init__(self) -> None:
        cfg = get_settings()
        self._base_url = cfg.telegram_mcp_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "TelegramMCPClient":
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
            raise RuntimeError(f"telegram-mcp error: {body['error']}")
        content = body.get("result", {}).get("content", [])
        if content:
            return json.loads(content[0].get("text", "null"))
        return None

    async def get_file_url(self, file_id: str) -> str:
        """Resolve a Telegram file_id to a temporary download URL."""
        try:
            data = await self._call_tool("get_file_url", {"file_id": file_id})
            if isinstance(data, dict) and data.get("url"):
                return str(data["url"])
        except Exception:
            pass

        # Compatibility fallback for older servers exposing getFile tool.
        data = await self._call_tool("getFile", {"file_id": file_id})
        if isinstance(data, dict) and data.get("download_url"):
            return str(data["download_url"])
        raise ValueError(f"Unable to resolve download URL for file_id={file_id}")

    async def download_file(self, file_id: str, dest_path: str) -> str:
        """Download a Telegram file and save it locally. Returns local path."""
        result = await self._call_tool("getFile", {"file_id": file_id})
        file_path = result.get("file_path", "")
        if not file_path:
            raise ValueError(f"No file_path for file_id={file_id}")
        # The telegram-mcp service should provide a download URL or bytes
        download_url = result.get("download_url") or f"https://api.telegram.org/file/bot/{{}}/{file_path}"
        import aiofiles, httpx as _httpx
        async with _httpx.AsyncClient() as hc:
            r = await hc.get(download_url)
            r.raise_for_status()
            async with aiofiles.open(dest_path, "wb") as f:
                await f.write(r.content)
        return dest_path

    async def send_message(self, chat_id: int, text: str, reply_to: Optional[int] = None) -> Dict[str, Any]:
        args: Dict[str, Any] = {"chat_id": chat_id, "text": text}
        if reply_to:
            args["reply_to_message_id"] = reply_to
        return await self._call_tool("sendMessage", args)
