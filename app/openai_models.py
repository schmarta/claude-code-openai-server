"""Pydantic models for the subset of the OpenAI Chat Completions API we serve.

Requests tolerate unknown fields (``extra="ignore"``) so clients like hermes can
send ``temperature``/``top_p``/``seed`` etc. without breaking us; we simply don't
act on the ones the CLI has no equivalent for. Two non-standard, optional fields
are recognised: ``workdir`` (per-request workspace override) and ``effort``
(Claude reasoning-effort level).
"""

from __future__ import annotations

import time
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


def _now() -> int:
    return int(time.time())


# ── Tool / function schema (request side) ──────────────────────────────────—


class FunctionDef(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    description: Optional[str] = None
    # JSON Schema object for the function arguments.
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class ToolDef(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["function"] = "function"
    function: FunctionDef


# ── Messages ────────────────────────────────────────────────────────────────


class FunctionCall(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: Optional[str] = None
    arguments: Optional[str] = None  # JSON-encoded string per OpenAI spec


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: Optional[str] = None
    type: Literal["function"] = "function"
    function: FunctionCall
    # Streaming deltas use an index to assemble fragmented tool calls.
    index: Optional[int] = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    # content may be a plain string, a list of content parts, or null (e.g. an
    # assistant message that only carries tool_calls).
    content: Union[str, list[Any], None] = None
    name: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None


# ── Request ─────────────────────────────────────────────────────────────────


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: Optional[str] = None
    messages: list[ChatMessage]
    stream: bool = False
    tools: Optional[list[ToolDef]] = None
    tool_choice: Optional[Union[str, dict[str, Any]]] = None
    # Accepted but not acted on (no clean CLI equivalent):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, list[str]]] = None
    # Non-standard CCI extensions:
    workdir: Optional[str] = None
    effort: Optional[str] = None


# ── Response (non-streaming) ──────────────────────────────────────────────—


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Non-standard: surfaced from the CLI `result` line when available.
    cost_usd: Optional[float] = None


class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=_now)
    model: str
    choices: list[Choice]
    usage: Optional[Usage] = None


# ── Response (streaming chunk) ──────────────────────────────────────────────


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=_now)
    model: str
    choices: list[ChunkChoice]
    usage: Optional[Usage] = None


# ── /v1/models ──────────────────────────────────────────────────────────────


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=_now)
    owned_by: str = "anthropic"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]
