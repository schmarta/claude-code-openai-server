"""Translation between the OpenAI Chat Completions wire format and Claude Code.

Two directions:

* **OpenAI → Claude**: split out ``system`` messages (→ ``--append-system-prompt``)
  and fold the conversation into the content of a ``user`` turn. In autonomous
  mode a fresh subprocess handles one request, so the full history is folded into
  a single turn; the stateful continuation path (M6) sends only the latest turn.
* **Claude → OpenAI**: build SSE chunks and the non-streaming response body, map
  ``stop_reason`` → ``finish_reason``, and derive ``usage`` from the ``result`` line.
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from typing import Any, Optional

from app.events import TurnDone
from app.openai_models import ChatMessage

# ── ids / timestamps ─────────────────────────────────────────────────────────


def new_completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def now() -> int:
    return int(time.time())


# ── OpenAI message helpers ─────────────────────────────────────────────────—


def message_text(msg: ChatMessage) -> str:
    """Flatten an OpenAI message's ``content`` to plain text.

    Handles the string form and the content-parts list form
    (``[{"type":"text","text":...}, ...]``); non-text parts (images) are skipped.
    """
    c = msg.content
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    parts: list[str] = []
    for part in c:
        if isinstance(part, dict):
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
        elif isinstance(part, str):
            parts.append(part)
    return "".join(parts)


_IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
_MAX_IMAGE_BASE64_BYTES = 20 * 1024 * 1024


def _claude_message_content(msg: ChatMessage) -> str | list[dict[str, Any]]:
    """Preserve OpenAI data-URL images as Claude stream-json image blocks."""
    if not isinstance(msg.content, list):
        return message_text(msg)

    blocks: list[dict[str, Any]] = []
    has_image = False
    for part in msg.content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            blocks.append({"type": "text", "text": part["text"]})
            continue
        if part.get("type") != "image_url":
            continue

        image_url = part.get("image_url")
        url = image_url.get("url", "") if isinstance(image_url, dict) else ""
        header, sep, data = str(url).partition(",")
        media_type = header[5:].split(";", 1)[0].lower() if header.startswith("data:") else ""
        if (
            not sep
            or ";base64" not in header.lower()
            or media_type not in _IMAGE_MEDIA_TYPES
            or not data
            or len(data) > _MAX_IMAGE_BASE64_BYTES
        ):
            continue
        try:
            base64.b64decode(data, validate=True)
        except ValueError:
            continue
        blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        })
        has_image = True

    return blocks if has_image else message_text(msg)


def split_system(messages: list[ChatMessage]) -> tuple[list[ChatMessage], Optional[str]]:
    """Partition into (non-system messages, joined system prompt or None)."""
    system_parts: list[str] = []
    convo: list[ChatMessage] = []
    for m in messages:
        if m.role == "system":
            t = message_text(m)
            if t:
                system_parts.append(t)
        else:
            convo.append(m)
    system = "\n\n".join(system_parts) if system_parts else None
    return convo, system


def _role_label(role: str) -> str:
    return {"user": "User", "assistant": "Assistant", "tool": "Tool"}.get(role, role.capitalize())


# History is handed to a stateless subprocess as context, then the latest user
# message as the live turn. The preamble deliberately does NOT end on a role cue
# ("User:"/"Assistant:") and instructs the model not to continue the transcript:
# a bare model reading a role-tagged log will otherwise text-complete it and
# fabricate the next "User:" turn instead of stopping after its own reply.
_HISTORY_HEADER = "Conversation history so far (context only):"
_LIVE_GUARD = (
    'Reply only to the latest message below, as the assistant. Write a single '
    'reply and then stop — do not continue the transcript or emit any "User:" '
    'or "Assistant:" lines of your own.'
)


def fold_conversation(convo: list[ChatMessage]) -> str | list[dict[str, Any]]:
    """Fold a (system-stripped) OpenAI conversation into one Claude user turn.

    A lone trailing user message is sent as-is. Otherwise prior turns become a
    labelled context preamble — followed by an explicit guard and the latest
    message as the raw live turn — so a fresh, stateless subprocess has the
    context without reading it as a transcript to continue.
    """
    if not convo:
        return ""
    if len(convo) == 1 and convo[0].role == "user":
        return _claude_message_content(convo[0])

    lines: list[str] = []
    for m in convo[:-1]:
        text = message_text(m)
        if m.tool_calls:
            calls = ", ".join(
                f"{tc.function.name}({tc.function.arguments or ''})" for tc in m.tool_calls
            )
            text = (text + " " if text else "") + f"[called tools: {calls}]"
        if text:
            lines.append(f"{_role_label(m.role)}: {text}")

    last = convo[-1]
    last_content = _claude_message_content(last)

    # No usable prior context: send the latest message alone (nothing to fold).
    if not lines:
        return last_content if isinstance(last_content, list) else message_text(last)

    preamble = _HISTORY_HEADER + "\n" + "\n".join(lines) + "\n\n" + _LIVE_GUARD + "\n\n"
    if isinstance(last_content, list):
        return [{"type": "text", "text": preamble}, *last_content]
    return preamble + message_text(last)


# ── stop_reason / usage mapping ────────────────────────────────────────────—


def map_finish_reason(stop_reason: Optional[str]) -> str:
    """Claude ``stop_reason`` → OpenAI ``finish_reason``."""
    return {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }.get(stop_reason or "", "stop")


def usage_from_turn(turn: TurnDone) -> dict[str, Any]:
    """Derive an OpenAI ``usage`` object from a ``result`` line.

    The top-level ``usage`` on the ``result`` line reflects the *last* model
    iteration only; when the CLI reports a per-iteration breakdown we sum it so
    internal (built-in) tool turns are counted. Cache tokens count as prompt
    tokens. Never fabricates — unknown values stay 0. Surfaces ``total_cost_usd``
    as the non-standard ``cost_usd``.
    """
    u = turn.usage or {}
    iterations = u.get("iterations")
    if isinstance(iterations, list) and iterations:
        prompt = 0
        completion = 0
        for it in iterations:
            if not isinstance(it, dict):
                continue
            prompt += _int(it.get("input_tokens"))
            prompt += _int(it.get("cache_read_input_tokens"))
            prompt += _int(it.get("cache_creation_input_tokens"))
            completion += _int(it.get("output_tokens"))
    else:
        prompt = (
            _int(u.get("input_tokens"))
            + _int(u.get("cache_read_input_tokens"))
            + _int(u.get("cache_creation_input_tokens"))
        )
        completion = _int(u.get("output_tokens"))
    usage: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    if turn.total_cost_usd is not None:
        usage["cost_usd"] = turn.total_cost_usd
    return usage


def _int(v: Any) -> int:
    return v if isinstance(v, int) and not isinstance(v, bool) else 0


# ── SSE chunk builders ─────────────────────────────────────────────────────—

DONE = "data: [DONE]\n\n"


def sse(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _chunk(
    cid: str,
    model: str,
    created: int,
    *,
    delta: dict[str, Any],
    finish_reason: Optional[str] = None,
    usage: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    obj: dict[str, Any] = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        obj["usage"] = usage
    return obj


def role_chunk(cid: str, model: str, created: int) -> dict[str, Any]:
    return _chunk(cid, model, created, delta={"role": "assistant", "content": ""})


def text_chunk(cid: str, model: str, created: int, text: str) -> dict[str, Any]:
    return _chunk(cid, model, created, delta={"content": text})


def tool_calls_chunk(
    cid: str, model: str, created: int, tool_calls: list[dict[str, Any]]
) -> dict[str, Any]:
    return _chunk(cid, model, created, delta={"tool_calls": tool_calls})


def finish_chunk(
    cid: str,
    model: str,
    created: int,
    finish_reason: str,
    usage: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return _chunk(cid, model, created, delta={}, finish_reason=finish_reason, usage=usage)


# ── non-streaming response ─────────────────────────────────────────────────—


def completion_response(
    cid: str,
    model: str,
    created: int,
    *,
    content: Optional[str],
    finish_reason: str,
    usage: Optional[dict[str, Any]] = None,
    tool_calls: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    obj: dict[str, Any] = {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }
    if usage is not None:
        obj["usage"] = usage
    return obj
