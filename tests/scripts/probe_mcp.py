#!/usr/bin/env python3
"""Probe the MCP bridge end-to-end against the live CLI (checklist #2/#3/#6).

Stands up a real in-process MCP server (one tool `get_weather`, NON-suspending —
returns canned JSON immediately) mounted at /mcp/<conv_id> via a single global
StreamableHTTPSessionManager, then launches `claude` pointed at it under
bypassPermissions and asks it to use the tool.

Confirms:
  #2  the mcp__hermes__get_weather call reaches our handler with NO control_request
  #3  --mcp-config type:"http" inline string parses; whether --allowed-tools needed
  #6  /mcp/<conv_id> routing through one mounted manager works

Env knobs:
  PROBE_ALLOWED=1   pass --allowed-tools "mcp__hermes__*" (default 0 = omit)
  PROBE_PORT=8799
"""
import asyncio
import contextlib
import os
import sys
from contextvars import ContextVar

import mcp.types as mtypes
import uvicorn
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount

from app.claude_session import STREAM_CLOSED, ClaudeSession
from app.events import (
    AssistantToolUse,
    ControlDialog,
    Error,
    Init,
    PermissionRequest,
    TextDelta,
    TurnDone,
)

PORT = int(os.environ.get("PROBE_PORT", "8799"))
ALLOWED = os.environ.get("PROBE_ALLOWED", "0") == "1"
CONV_ID = "probeconv"

# Shared probe state
HANDLER_CALLS: list[tuple[str, dict]] = []
current_conv_id: ContextVar[str] = ContextVar("current_conv_id", default="")

# ── MCP server ───────────────────────────────────────────────────────────────
server: Server = Server("hermes")


@server.list_tools()
async def list_tools() -> list[mtypes.Tool]:
    print(f"[mcp] list_tools (conv={current_conv_id.get()})", flush=True)
    return [
        mtypes.Tool(
            name="get_weather",
            description="Get the current weather for a city.",
            inputSchema={
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        )
    ]


@server.call_tool(validate_input=False)
async def call_tool(name: str, arguments: dict):
    conv = current_conv_id.get()
    print(f"[mcp] *** call_tool name={name} args={arguments} conv={conv} ***", flush=True)
    HANDLER_CALLS.append((name, arguments))
    return [mtypes.TextContent(type="text", text='{"temp_c": 21, "summary": "sunny"}')]


session_manager = StreamableHTTPSessionManager(app=server, json_response=False, stateless=False)


class McpDispatcher:
    """ASGI app mounted at /mcp; routes /mcp/<conv_id> to the one manager,
    setting the conv_id contextvar (captured into the server task)."""

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return
        path = scope.get("path", "")  # already stripped of /mcp by Mount
        conv_id = path.strip("/").split("/")[0] if path.strip("/") else ""
        current_conv_id.set(conv_id)
        # Rewrite to root so the transport sees its own base path.
        new_scope = dict(scope)
        new_scope["path"] = "/"
        new_scope["raw_path"] = b"/"
        await session_manager.handle_request(new_scope, receive, send)


@contextlib.asynccontextmanager
async def lifespan(app):
    async with session_manager.run():
        yield


def build_app() -> Starlette:
    async def health(request):
        return JSONResponse({"ok": True})

    return Starlette(
        lifespan=lifespan,
        routes=[Mount("/mcp", app=McpDispatcher()), Mount("/health", app=health)],
    )


# ── driver ───────────────────────────────────────────────────────────────────


async def drive_claude() -> bool:
    mcp_config = {
        "mcpServers": {
            "hermes": {"type": "http", "url": f"http://127.0.0.1:{PORT}/mcp/{CONV_ID}"}
        }
    }
    sess = ClaudeSession(
        claude_bin="claude",
        model="sonnet",
        permission_mode="bypassPermissions",
        workdir="/tmp/cci_probe_ws",
        mcp_config=mcp_config,
        allowed_tools=["mcp__hermes__get_weather", "mcp__hermes"] if ALLOWED else None,
        enable_tool_search=False,
    )
    print(f"[probe] allowed_tools_passed={ALLOWED}", flush=True)
    await sess.start()
    await sess.send_user_turn(
        "Use the get_weather tool to get the weather for Paris, then tell me the result in one sentence."
    )
    saw_hermes_tooluse = False
    saw_control = False
    final_text = ""
    while True:
        ev = await sess.next_event(timeout=120)
        if ev is None:
            print("[probe] TIMEOUT", flush=True)
            break
        if ev is STREAM_CLOSED:
            print("[probe] stream closed", flush=True)
            break
        if isinstance(ev, Init):
            print(f"[claude] init session={ev.session_id}", flush=True)
        elif isinstance(ev, TextDelta):
            final_text += ev.text
        elif isinstance(ev, AssistantToolUse):
            names = [b.name for b in ev.tool_uses]
            print(f"[claude] tool_use: {names}", flush=True)
            if any(b.is_hermes for b in ev.tool_uses):
                saw_hermes_tooluse = True
        elif isinstance(ev, (PermissionRequest, ControlDialog)):
            saw_control = True
            print(f"[claude] *** CONTROL REQUEST: {ev} ***", flush=True)
        elif isinstance(ev, TurnDone):
            print(f"[claude] TURN DONE stop={ev.stop_reason}", flush=True)
            break
        elif isinstance(ev, Error):
            print(f"[claude] ERROR: {ev.message}", flush=True)
            break
    await sess.aclose()

    print("\n========== PROBE RESULTS ==========", flush=True)
    print(f"handler invoked:        {len(HANDLER_CALLS)} time(s) -> {HANDLER_CALLS}", flush=True)
    print(f"assistant hermes call:  {saw_hermes_tooluse}", flush=True)
    print(f"control_request seen:   {saw_control}  (expect False under bypass)", flush=True)
    print(f"final text:             {final_text!r}", flush=True)
    ok = len(HANDLER_CALLS) >= 1 and not saw_control and "21" in final_text
    print("PROBE:", "PASS" if ok else "FAIL", flush=True)
    return ok


async def main() -> int:
    config = uvicorn.Config(build_app(), host="127.0.0.1", port=PORT, log_level="warning")
    server_obj = uvicorn.Server(config)
    serve_task = asyncio.create_task(server_obj.serve())
    for _ in range(100):
        if server_obj.started:
            break
        await asyncio.sleep(0.05)
    print(f"[probe] MCP server up on :{PORT}", flush=True)
    try:
        ok = await drive_claude()
    finally:
        server_obj.should_exit = True
        with contextlib.suppress(Exception):
            await asyncio.wait_for(serve_task, timeout=5)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
