"""``POST /v1/chat/completions`` — autonomous (no-tool) path.

This milestone handles requests with no external ``tools``: the full OpenAI
conversation is folded into one turn for a fresh ``claude`` subprocess, whose
streamed text becomes either an SSE stream or a single JSON completion. The
tool-passthrough path (suspend/resume via the MCP bridge) is layered on in later
milestones; for now any ``tools`` field is logged and ignored.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.claude_session import STREAM_CLOSED, ClaudeSession, prompt_session_kwargs as _prompt_kwargs
from app.config import ensure_workdir, get_settings, resolve_model
from app.conversation import (
    ConversationManager,
    DoneChunk,
    ErrorChunk,
    ExpiredContinuation,
    TextChunk,
    ToolBoundaryChunk,
    ToolCallsChunk,
)
from app.errors import OpenAIError, error_envelope
from app.events import AssistantToolUse, Error, TextDelta, TurnDone
from app.openai_models import ChatCompletionRequest
from app.textfilter import OutputFilter
from app.timing import TurnTimer
from app.translate import (
    DONE,
    completion_response,
    finish_chunk,
    fold_conversation,
    map_finish_reason,
    new_completion_id,
    now,
    role_chunk,
    split_system,
    sse,
    text_chunk,
    tool_calls_chunk,
    usage_from_turn,
)

logger = logging.getLogger("cci.chat")

router = APIRouter()

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    settings = get_settings()
    model = resolve_model(req.model, settings)
    try:
        workdir = settings.resolve_workdir(req.workdir)
    except ValueError as e:
        raise OpenAIError(str(e), status_code=400, param="workdir")
    ensure_workdir(workdir)
    effort = req.effort or settings.default_effort

    # ── tool-passthrough path (suspend/resume via the MCP bridge) ──────────
    if req.tools:
        return await _handle_tools(req, request, model, workdir, effort)

    # ── autonomous path (no external tools) ────────────────────────────────
    convo, system = split_system(req.messages)
    content = fold_conversation(convo)
    if not content.strip():
        raise OpenAIError("no user content in messages", status_code=400, param="messages")

    timing = settings.timing_log
    sess = ClaudeSession(
        claude_bin=settings.claude_bin,
        model=model,
        permission_mode=settings.permission_mode,
        workdir=workdir,
        effort=effort,
        enable_tool_search=settings.enable_tool_search,
        timing_log=timing,
        timing_label="autonomous",
        **_prompt_kwargs(settings, system),
    )
    await sess.start()
    await sess.send_user_turn(content)

    cid = new_completion_id()
    created = now()
    timeout = settings.request_timeout_s

    flatten = settings.flatten_markdown_tables
    if req.stream:
        return StreamingResponse(
            _stream(sess, cid, model, created, timeout, flatten, timing),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )
    body = await _collect(sess, cid, model, created, timeout, flatten, timing)
    return JSONResponse(content=body)


async def _stream(
    sess: ClaudeSession, cid: str, model: str, created: int, timeout: float,
    flatten_tables: bool = True, timing: bool = False,
) -> AsyncIterator[str]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    joiner = OutputFilter(flatten_tables=flatten_tables)
    timer = TurnTimer(timing, "autonomous")
    completion_tokens = 0
    try:
        yield sse(role_chunk(cid, model, created))
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                yield sse(error_envelope("upstream timeout", type="timeout_error"))
                break
            ev = await sess.next_event(timeout=remaining)
            if ev is None:
                yield sse(error_envelope("upstream timeout", type="timeout_error"))
                break
            if ev is STREAM_CLOSED:
                tail = joiner.flush()
                if tail:
                    yield sse(text_chunk(cid, model, created, tail))
                yield sse(finish_chunk(cid, model, created, "stop"))
                break
            if isinstance(ev, TextDelta):
                out = joiner.feed(ev.text)
                if out:
                    timer.first_token()
                    yield sse(text_chunk(cid, model, created, out))
            elif isinstance(ev, TurnDone):
                tail = joiner.flush()
                if tail:
                    yield sse(text_chunk(cid, model, created, tail))
                usage = usage_from_turn(ev)
                completion_tokens = usage.get("completion_tokens", 0)
                yield sse(
                    finish_chunk(
                        cid, model, created,
                        map_finish_reason(ev.stop_reason),
                        usage,
                    )
                )
                break
            elif isinstance(ev, Error):
                yield sse(error_envelope(ev.message, type="upstream_error"))
                break
            elif isinstance(ev, AssistantToolUse):
                # Built-in tool use runs internally; nothing to surface, but it
                # marks the seam between two assistant text segments so the next
                # one starts on a fresh blank line instead of gluing on.
                logger.debug("internal tool use: %s", [b.name for b in ev.tool_uses])
                joiner.tool_boundary()
                continue
            # Init / others: ignore.
        yield DONE
    finally:
        timer.done(completion_tokens)
        await sess.aclose()


async def _collect(
    sess: ClaudeSession, cid: str, model: str, created: int, timeout: float,
    flatten_tables: bool = True, timing: bool = False,
) -> dict:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    joiner = OutputFilter(flatten_tables=flatten_tables)
    timer = TurnTimer(timing, "autonomous")
    completion_tokens = 0
    chunks: list[str] = []
    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise OpenAIError("upstream timeout", status_code=504, type="timeout_error")
            ev = await sess.next_event(timeout=remaining)
            if ev is None:
                raise OpenAIError("upstream timeout", status_code=504, type="timeout_error")
            if ev is STREAM_CLOSED:
                chunks.append(joiner.flush())
                return completion_response(
                    cid, model, created, content="".join(chunks), finish_reason="stop"
                )
            if isinstance(ev, TextDelta):
                timer.first_token()
                chunks.append(joiner.feed(ev.text))
            elif isinstance(ev, TurnDone):
                chunks.append(joiner.flush())
                # Prefer the segment-joined text (it carries every text block plus
                # the seams that `result` flattens away); fall back to `result`
                # only when no deltas were captured.
                assembled = "".join(chunks)
                text = assembled if assembled.strip() else (ev.result_text or "")
                usage = usage_from_turn(ev)
                completion_tokens = usage.get("completion_tokens", 0)
                return completion_response(
                    cid, model, created,
                    content=text,
                    finish_reason=map_finish_reason(ev.stop_reason),
                    usage=usage,
                )
            elif isinstance(ev, Error):
                raise OpenAIError(ev.message, status_code=502, type="upstream_error")
            elif isinstance(ev, AssistantToolUse):
                logger.debug("internal tool use: %s", [b.name for b in ev.tool_uses])
                joiner.tool_boundary()
                continue
    finally:
        timer.done(completion_tokens)
        await sess.aclose()


# ── tool-passthrough path ───────────────────────────────────────────────────


async def _handle_tools(
    req: ChatCompletionRequest, request: Request, model: str, workdir, effort
):
    mgr: ConversationManager = request.app.state.conv_manager

    if mgr.is_continuation(req):
        try:
            conv = await mgr.resume(req)
        except ExpiredContinuation as e:
            raise OpenAIError(
                str(e), status_code=409, type="invalid_request_error", code="tool_result_expired"
            )
    else:
        convo, _ = split_system(req.messages)
        if not fold_conversation(convo).strip():
            raise OpenAIError("no user content in messages", status_code=400, param="messages")
        conv = await mgr.create(req, model=model, workdir=workdir, effort=effort)

    cid = new_completion_id()
    created = now()
    settings = get_settings()
    flatten = settings.flatten_markdown_tables
    timing = settings.timing_log

    if req.stream:
        return StreamingResponse(
            _tool_stream(mgr, conv, cid, model, created, flatten, timing),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )
    body = await _tool_collect(mgr, conv, cid, model, created, flatten, timing)
    return JSONResponse(content=body)


async def _tool_stream(mgr, conv, cid, model, created, flatten_tables: bool = True,
                       timing: bool = False):
    # Same OutputFilter the autonomous path uses, so the newline-seam and table
    # flattening apply on the tool path too.
    filt = OutputFilter(flatten_tables=flatten_tables)
    timer = TurnTimer(timing, "tool")
    completion_tokens = 0
    try:
        yield sse(role_chunk(cid, model, created))
        async for ch in mgr.run_turn(conv):
            if isinstance(ch, TextChunk):
                out = filt.feed(ch.text)
                if out:
                    timer.first_token()
                    yield sse(text_chunk(cid, model, created, out))
            elif isinstance(ch, ToolBoundaryChunk):
                filt.tool_boundary()
            elif isinstance(ch, ToolCallsChunk):
                tail = filt.flush()
                if tail:
                    yield sse(text_chunk(cid, model, created, tail))
                calls = [
                    {**pc.openai_tool_call(), "index": i} for i, pc in enumerate(ch.calls)
                ]
                yield sse(tool_calls_chunk(cid, model, created, calls))
                yield sse(finish_chunk(cid, model, created, "tool_calls"))
                break
            elif isinstance(ch, DoneChunk):
                tail = filt.flush()
                if tail:
                    yield sse(text_chunk(cid, model, created, tail))
                completion_tokens = (ch.usage or {}).get("completion_tokens", 0)
                yield sse(finish_chunk(cid, model, created, ch.finish_reason, ch.usage or None))
                break
            elif isinstance(ch, ErrorChunk):
                yield sse(error_envelope(ch.message, type="upstream_error"))
                break
        yield DONE
    finally:
        timer.done(completion_tokens)


async def _tool_collect(mgr, conv, cid, model, created, flatten_tables: bool = True,
                        timing: bool = False) -> dict:
    filt = OutputFilter(flatten_tables=flatten_tables)
    timer = TurnTimer(timing, "tool")
    completion_tokens = 0
    text_parts: list[str] = []
    try:
        async for ch in mgr.run_turn(conv):
            if isinstance(ch, TextChunk):
                if ch.text:
                    timer.first_token()
                text_parts.append(filt.feed(ch.text))
            elif isinstance(ch, ToolBoundaryChunk):
                filt.tool_boundary()
            elif isinstance(ch, ToolCallsChunk):
                text_parts.append(filt.flush())
                return completion_response(
                    cid, model, created,
                    content="".join(text_parts) or None,
                    finish_reason="tool_calls",
                    tool_calls=[pc.openai_tool_call() for pc in ch.calls],
                )
            elif isinstance(ch, DoneChunk):
                text_parts.append(filt.flush())
                completion_tokens = (ch.usage or {}).get("completion_tokens", 0)
                return completion_response(
                    cid, model, created,
                    content="".join(text_parts),
                    finish_reason=ch.finish_reason,
                    usage=ch.usage or None,
                )
            elif isinstance(ch, ErrorChunk):
                raise OpenAIError(ch.message, status_code=ch.status_code, type="upstream_error")
        raise OpenAIError("no response produced", status_code=502, type="upstream_error")
    finally:
        timer.done(completion_tokens)
