"""
Entrypoint — runs the aiogram long-polling bot and an HTTP server concurrently.

  asyncio event loop
  ├── bot_task     — aiogram dispatcher.start_polling()
  └── http_task    — uvicorn (Starlette app)

HTTP endpoints
──────────────
  GET  /health      — liveness probe
  POST /mcp         — JSON-RPC 2.0 tool dispatch (Universal Messaging Contract)
  GET  /sse         — MCP SSE transport (for full MCP SDK clients)
  POST /messages/   — MCP SSE transport messages
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging

import structlog
import uvicorn
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from src.bot import create_bot, run_polling
from src.config import get_settings
from src.server import mcp, send_message, get_file_url, list_chats, set_topic, stop_topic, list_topics


def _configure_logging() -> None:
    cfg = get_settings()
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, cfg.log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


# Map tool name → callable (the actual Python functions from server.py)
_TOOL_REGISTRY = {
    "send_message": send_message,
    "get_file_url": get_file_url,
    "list_chats": list_chats,
    "set_topic": set_topic,
    "stop_topic": stop_topic,
    "list_topics": list_topics,
}


def build_starlette_app() -> Starlette:
    """Build the HTTP app with both the simple /mcp JSON-RPC endpoint and the
    full MCP SSE transport for clients using the MCP SDK directly."""

    sse_transport = SseServerTransport("/messages/")

    async def handle_sse(request: Request):
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await mcp._mcp_server.run(
                streams[0],
                streams[1],
                mcp._mcp_server.create_initialization_options(),
            )

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def handle_mcp(request: Request) -> JSONResponse:
        """
        Lightweight JSON-RPC 2.0 tool dispatcher.

        Accepts:
          { "jsonrpc": "2.0", "id": 1,
            "method": "tools/call",
            "params": { "name": "<tool>", "arguments": {...} } }

        Returns standard JSON-RPC 2.0 response with result.content[0].text
        containing the JSON-serialised tool output.
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"jsonrpc": "2.0", "id": None,
                                 "error": {"code": -32700, "message": "Parse error"}})

        rpc_id = body.get("id")
        method = body.get("method", "")
        params = body.get("params", {})

        if method != "tools/call":
            return JSONResponse({"jsonrpc": "2.0", "id": rpc_id,
                                 "error": {"code": -32601, "message": f"Unknown method: {method}"}})

        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        fn = _TOOL_REGISTRY.get(tool_name)

        if fn is None:
            return JSONResponse({"jsonrpc": "2.0", "id": rpc_id,
                                 "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}})

        try:
            result = fn(**arguments)
            if inspect.iscoroutine(result):
                result = await result
            text = json.dumps(result, ensure_ascii=False)
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {"content": [{"type": "text", "text": text}]},
            })
        except Exception as exc:
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {"code": -32603, "message": str(exc)},
            })

    return Starlette(
        routes=[
            Route("/health", health),
            Route("/mcp", handle_mcp, methods=["POST"]),
            Route("/sse", handle_sse),
            Mount("/messages", app=sse_transport.handle_post_message),
        ]
    )


async def main() -> None:
    _configure_logging()
    cfg = get_settings()
    log = structlog.get_logger(__name__)

    log.info("telegram_mcp.start", host=cfg.mcp_host, port=cfg.mcp_port)

    bot, dp, publisher = await create_bot()
    app = build_starlette_app()

    config = uvicorn.Config(
        app=app,
        host=cfg.mcp_host,
        port=cfg.mcp_port,
        log_level=cfg.log_level.lower(),
    )
    server = uvicorn.Server(config)

    # Run bot polling and HTTP server concurrently
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(run_polling(bot, dp), name="bot-polling")
            tg.create_task(server.serve(), name="mcp-http")
    finally:
        await publisher.close()
        log.info("telegram_mcp.shutdown")


if __name__ == "__main__":
    asyncio.run(main())
