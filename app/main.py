"""FastAPI app factory + uvicorn entrypoint.

The app exposes an OpenAI-compatible surface (``/v1/models``,
``/v1/chat/completions``) backed by the Claude Code CLI, and mounts the in-process
MCP bridge at ``settings.mcp_path_prefix`` (default ``/mcp``) so Claude can reach
hermes' functions. The lifespan hook owns process-wide singletons: the MCP
session manager's task group and the :class:`ConversationManager`.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import logging
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request

from app.config import get_settings
from app.conversation import ConversationManager
from app.errors import OpenAIError
from app.mcp_bridge import McpBridge
from app.routes import chat, compat, health, models

logger = logging.getLogger("cci")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = app.state.settings
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("claude-code-interface starting on %s:%d", settings.host, settings.port)
    logger.info("default model=%s permission_mode=%s workdir=%s mcp_prefix=%s",
                settings.default_model, settings.permission_mode,
                settings.default_workdir, settings.mcp_path_prefix)

    mcp: McpBridge = app.state.mcp
    async with mcp.lifespan():
        manager = ConversationManager(mcp, settings)
        app.state.conv_manager = manager
        gc_task = asyncio.create_task(manager.gc_loop(), name="cci-gc")
        try:
            yield
        finally:
            gc_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await gc_task
            await manager.close_all()
            logger.info("claude-code-interface shutting down")


def create_app() -> FastAPI:
    settings = get_settings()

    # Safety interlock: the server drives the Claude CLI with
    # permission_mode=bypassPermissions by default, so an open, network-reachable
    # bind is remote code execution for anyone who can reach it. Refuse to start
    # on a non-loopback host unless an api_key is set to gate /v1 requests.
    if not settings.is_loopback_host() and not settings.api_key:
        raise RuntimeError(
            f"refusing to start: host={settings.host!r} is not loopback and no "
            "api_key is set — set CCI_API_KEY to require a bearer token, or bind "
            "to 127.0.0.1"
        )

    mcp = McpBridge()

    app = FastAPI(title="claude-code-interface", lifespan=lifespan)
    app.state.settings = settings
    app.state.mcp = mcp

    @app.exception_handler(OpenAIError)
    async def _openai_error_handler(_: Request, exc: OpenAIError):  # type: ignore[unused-ignore]
        return exc.json_response()

    @app.middleware("http")
    async def _auth(request: Request, call_next):
        # When an api_key is configured, every /v1 request must carry a matching
        # bearer token. The MCP mount is intentionally exempt: it is reached only
        # by the local Claude subprocess over loopback and carries no token.
        key = settings.api_key
        if key and request.url.path.startswith("/v1"):
            header = request.headers.get("authorization", "")
            token = header[7:] if header[:7].lower() == "bearer " else ""
            if not hmac.compare_digest(token, key):
                return OpenAIError(
                    "missing or invalid api key",
                    status_code=401,
                    type="invalid_request_error",
                    code="invalid_api_key",
                ).json_response()
        return await call_next(request)

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(chat.router)
    app.include_router(compat.router)

    # Mount the in-process MCP server; the dispatcher reads conv_id from the path.
    app.mount(settings.mcp_path_prefix, mcp.asgi_app())
    return app


app = create_app()


def run() -> None:
    """Console-script entrypoint (``claude-code-interface``)."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
