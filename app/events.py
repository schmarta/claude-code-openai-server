"""Semantic events distilled from the `claude` CLI ``--output-format stream-json``
stream — a Python port of wisp's ``src/claude/event.rs`` ``parse_line``.

Parsing is deliberately **defensive and total**: ``parse_line`` matches on the
``"type"`` field of a decoded JSON object and reaches every nested field through
``dict.get``, so a blank line, non-JSON, or a renamed/missing field in a future
CLI version degrades to ``None`` ("ignore this line") rather than raising.

Differences from the wisp original (intentional, for the OpenAI bridge):

* ``assistant`` lines yield :class:`AssistantToolUse` carrying the **full list**
  of ``tool_use`` blocks (id/name/input), not a single one-line summary. The
  bridge must split hermes' functions (``mcp__hermes__*``) from Claude's own
  built-in tools and know the batch size, so it needs every block.
* :class:`TurnDone` carries ``usage`` / ``total_cost_usd`` / ``stop_reason`` /
  ``result`` text from the ``result`` line, for OpenAI ``usage`` + ``finish_reason``.
* ``AskUserQuestion`` is **not** suppressed (wisp suppressed its assistant echo
  for UI reasons); under ``bypassPermissions`` it should not surface, but if it
  ever does the bridge sees the raw block and can decide.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, Union

# hermes' external functions are exposed to Claude under this MCP tool-name
# prefix (server name "hermes"). Everything else in a tool_use block is a
# Claude Code built-in (Read/Edit/Bash/…) that runs internally.
HERMES_TOOL_PREFIX = "mcp__hermes__"


# ── Event types ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolUseBlock:
    """One assembled ``tool_use`` content block from a complete assistant message."""

    id: Optional[str]
    name: str
    input: dict[str, Any]

    @property
    def is_hermes(self) -> bool:
        return self.name.startswith(HERMES_TOOL_PREFIX)

    @property
    def hermes_function_name(self) -> str:
        """The bare OpenAI function name (``mcp__hermes__get_weather`` → ``get_weather``)."""
        if self.is_hermes:
            return self.name[len(HERMES_TOOL_PREFIX):]
        return self.name


@dataclass(frozen=True)
class Init:
    session_id: Optional[str]
    model: Optional[str]


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class AssistantToolUse:
    """A complete assistant message carrying one or more ``tool_use`` blocks."""

    tool_uses: list[ToolUseBlock]

    @property
    def hermes_calls(self) -> list[ToolUseBlock]:
        return [b for b in self.tool_uses if b.is_hermes]

    @property
    def builtin_calls(self) -> list[ToolUseBlock]:
        return [b for b in self.tool_uses if not b.is_hermes]


@dataclass(frozen=True)
class TurnDone:
    """A clean ``result`` line ending the turn."""

    result_text: Optional[str] = None
    stop_reason: Optional[str] = None
    usage: dict[str, Any] = field(default_factory=dict)
    total_cost_usd: Optional[float] = None
    session_id: Optional[str] = None


@dataclass(frozen=True)
class Error:
    message: str


@dataclass(frozen=True)
class PermissionRequest:
    request_id: str
    tool: str
    summary: str
    input: dict[str, Any]


@dataclass(frozen=True)
class QuestionRequest:
    request_id: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ControlDialog:
    request_id: str
    kind: str  # "user_dialog" | "elicitation"
    summary: str

    def cancel_behavior(self) -> str:
        """Protocol-mandated token to unblock the turn."""
        return "cancelled" if self.kind == "user_dialog" else "cancel"


ChatEvent = Union[
    Init,
    TextDelta,
    AssistantToolUse,
    TurnDone,
    Error,
    PermissionRequest,
    QuestionRequest,
    ControlDialog,
]


# ── Parser ──────────────────────────────────────────────────────────────────


def parse_line(line: str) -> Optional[ChatEvent]:
    """Parse one raw JSONL line from the CLI into a :data:`ChatEvent`, or ``None``.

    Total and panic-free: blank/non-JSON/unknown lines and missing fields all
    yield ``None``.
    """
    line = line.strip()
    if not line:
        return None
    try:
        v = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(v, dict):
        return None

    t = v.get("type")
    if not isinstance(t, str):
        return None

    if t == "system":
        if v.get("subtype") != "init":
            return None
        return Init(
            session_id=_as_str(v.get("session_id")),
            model=_as_str(v.get("model")),
        )

    if t == "assistant":
        return _parse_assistant(v)

    if t == "stream_event":
        return _parse_stream_event(v)

    if t == "result":
        return _parse_result(v)

    if t == "control_request":
        return _parse_control_request(v)

    return None


def _parse_assistant(v: dict[str, Any]) -> Optional[ChatEvent]:
    message = v.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if not isinstance(content, list):
        return None
    blocks: list[ToolUseBlock] = []
    for b in content:
        if not isinstance(b, dict):
            continue
        if b.get("type") != "tool_use":
            continue
        name = _as_str(b.get("name")) or "tool"
        raw_input = b.get("input")
        blocks.append(
            ToolUseBlock(
                id=_as_str(b.get("id")),
                name=name,
                input=raw_input if isinstance(raw_input, dict) else {},
            )
        )
    if not blocks:
        # Text/thinking-only assistant echo — text already arrived via deltas.
        return None
    return AssistantToolUse(tool_uses=blocks)


def _parse_stream_event(v: dict[str, Any]) -> Optional[ChatEvent]:
    event = v.get("event")
    if not isinstance(event, dict):
        return None
    if event.get("type") != "content_block_delta":
        return None
    delta = event.get("delta")
    if not isinstance(delta, dict):
        return None
    if delta.get("type") != "text_delta":
        return None
    text = delta.get("text")
    if not isinstance(text, str):
        return None
    return TextDelta(text)


def _parse_result(v: dict[str, Any]) -> Optional[ChatEvent]:
    is_error = v.get("is_error")
    if is_error is True:
        msg = (
            _as_str(v.get("result"))
            or _as_str(v.get("subtype"))
            or "the request failed"
        )
        return Error(msg)
    usage = v.get("usage")
    return TurnDone(
        result_text=_as_str(v.get("result")),
        stop_reason=_as_str(v.get("stop_reason")),
        usage=usage if isinstance(usage, dict) else {},
        total_cost_usd=_as_float(v.get("total_cost_usd")),
        session_id=_as_str(v.get("session_id")),
    )


def _parse_control_request(v: dict[str, Any]) -> Optional[ChatEvent]:
    request = v.get("request")
    if not isinstance(request, dict):
        return None
    request_id = _as_str(v.get("request_id"))
    if request_id is None:
        return None
    subtype = request.get("subtype")
    if subtype == "can_use_tool":
        tool = _as_str(request.get("tool_name")) or "tool"
        raw_input = request.get("input")
        tool_input = raw_input if isinstance(raw_input, dict) else {}
        if tool == "AskUserQuestion":
            return QuestionRequest(request_id=request_id, input=tool_input)
        return PermissionRequest(
            request_id=request_id,
            tool=tool,
            summary=tool_use_summary(tool, tool_input),
            input=tool_input,
        )
    if subtype == "request_user_dialog":
        return ControlDialog(
            request_id=request_id,
            kind="user_dialog",
            summary=_as_str(request.get("dialog_kind")) or "dialog",
        )
    if subtype == "elicitation":
        return ControlDialog(
            request_id=request_id,
            kind="elicitation",
            summary=_as_str(request.get("message")) or "elicitation",
        )
    return None


# ── helpers ─────────────────────────────────────────────────────────────────


def tool_use_summary(name: str, input: Optional[dict[str, Any]]) -> str:
    """One-line gist of a tool call's input (for logging built-in tool use)."""
    if not isinstance(input, dict):
        return name

    def pick(key: str) -> Optional[str]:
        val = input.get(key)
        return val if isinstance(val, str) else None

    if name in ("Read", "Edit", "Write", "MultiEdit", "NotebookEdit"):
        summary = pick("file_path")
    elif name == "Bash":
        summary = pick("command")
    elif name in ("Glob", "Grep"):
        summary = pick("pattern")
    elif name in ("WebFetch", "WebSearch"):
        summary = pick("url") or pick("query")
    else:
        summary = pick("file_path") or pick("command") or pick("pattern")
    summary = summary or name
    return _truncate(summary, 120)


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "…"


def _as_str(v: Any) -> Optional[str]:
    return v if isinstance(v, str) else None


def _as_float(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None
