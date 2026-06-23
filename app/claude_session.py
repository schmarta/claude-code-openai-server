"""Async ``claude`` CLI driver — a persistent subprocess per conversation.

A Python/asyncio port of wisp's ``src/claude/mod.rs``. One ``claude`` process
stays alive for the lifetime of a conversation, spoken to over the bidirectional
``stream-json`` protocol: user turns are written as JSON lines to stdin, and the
process streams JSONL reply lines on stdout, which a reader task parses into
:data:`~app.events.ChatEvent` values and pushes onto an :class:`asyncio.Queue`.

Empirically confirmed against CLI 2.1.183 (the wisp flag set, **without** ``-p``):
the process survives multiple stdin turns and accepts ``--permission-mode
bypassPermissions``. Each turn re-emits a ``system/init`` line; the consumer
treats one ``result`` line (→ ``TurnDone``/``Error``) as the turn boundary.

The control_response wire-format invariant is preserved verbatim: ``request_id``
**must** nest inside the ``response`` object, never at top level, or the turn
stalls forever. Under ``bypassPermissions`` no control requests should arrive,
but the path exists as a defensive fallback.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

from app.events import ChatEvent, parse_line

logger = logging.getLogger("cci.session")

# Sentinel placed on the event queue when the CLI's stdout reaches EOF (the
# process exited). Consumers stop waiting for turn events when they see it.
STREAM_CLOSED = object()

# StreamReader line buffer. Assistant ``tool_use`` inputs (e.g. a large Edit
# `new_string`) and `result` text can be sizeable; keep the cap generous and
# skip — rather than crash on — any pathologically long line.
_READ_LIMIT = 64 * 1024 * 1024


def prompt_session_kwargs(settings: Any, system_prompt: Optional[str]) -> dict[str, Any]:
    """System-prompt / built-in-tool kwargs for :class:`ClaudeSession`.

    In *bare model* mode the request's ``system_prompt`` REPLACES claude's
    default prompt, the dynamic context sections are excluded, and every native
    tool is stripped (only ``--mcp-config`` tools remain). Otherwise the prompt
    is appended and claude keeps its full default tool set.
    """
    if getattr(settings, "bare_model_mode", False):
        return {
            "system_prompt": system_prompt or settings.bare_model_system_prompt,
            "exclude_dynamic_sections": True,
            "builtin_tools": [],
        }
    return {"append_system_prompt": system_prompt}


class ClaudeSession:
    """Drives one persistent ``claude`` subprocess over stream-json."""

    def __init__(
        self,
        *,
        claude_bin: str,
        model: str,
        permission_mode: str,
        workdir: Union[str, Path],
        effort: Optional[str] = None,
        mcp_config: Optional[dict[str, Any]] = None,
        allowed_tools: Optional[list[str]] = None,
        append_system_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        exclude_dynamic_sections: bool = False,
        builtin_tools: Optional[list[str]] = None,
        enable_tool_search: bool = False,
        env: Optional[dict[str, str]] = None,
    ) -> None:
        self.claude_bin = claude_bin
        self.model = model
        self.permission_mode = permission_mode
        self.workdir = str(workdir)
        self.effort = effort
        self.mcp_config = mcp_config
        self.allowed_tools = allowed_tools
        self.append_system_prompt = append_system_prompt
        # When set, REPLACES claude's default (Claude Code) system prompt via
        # --system-prompt, instead of layering on top via --append-system-prompt.
        self.system_prompt = system_prompt
        # Drop the dynamic context blocks (env, cwd, git status, identity).
        self.exclude_dynamic_sections = exclude_dynamic_sections
        # Built-in tool allowlist for --tools. [] disables ALL native tools
        # (only --mcp-config tools remain); None leaves claude's defaults.
        self.builtin_tools = builtin_tools
        self.enable_tool_search = enable_tool_search
        self._env_overrides = env or {}

        self._proc: Optional[asyncio.subprocess.Process] = None
        self._queue: asyncio.Queue[Union[ChatEvent, object]] = asyncio.Queue()
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._stderr_task: Optional[asyncio.Task[None]] = None
        self._closed = False
        self.session_id: Optional[str] = None

    # ── command construction ────────────────────────────────────────────────

    def _build_args(self) -> list[str]:
        args = [
            self.claude_bin,
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--permission-prompt-tool", "stdio",
            "--model", self.model,
            "--permission-mode", self.permission_mode,
            "--add-dir", self.workdir,
        ]
        if self.effort:
            args += ["--effort", self.effort]
        if self.mcp_config is not None:
            args += ["--mcp-config", json.dumps(self.mcp_config), "--strict-mcp-config"]
        if self.allowed_tools:
            args += ["--allowed-tools", ",".join(self.allowed_tools)]
        # System prompt: --system-prompt fully replaces the default; otherwise
        # --append-system-prompt layers onto Claude Code's built-in prompt.
        if self.system_prompt is not None:
            args += ["--system-prompt", self.system_prompt]
        elif self.append_system_prompt:
            args += ["--append-system-prompt", self.append_system_prompt]
        if self.exclude_dynamic_sections:
            args += ["--exclude-dynamic-system-prompt-sections"]
        # Built-in tool selection. [] => "--tools ''" (none); a non-empty list
        # is passed as space-separated tokens (the CLI's variadic format).
        if self.builtin_tools is not None:
            args += ["--tools", *(self.builtin_tools or [""])]
        return args

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        if not self.enable_tool_search:
            env["ENABLE_TOOL_SEARCH"] = "false"
        env.update(self._env_overrides)
        return env

    # ── lifecycle ─────────────────────────────────────────────────────────—

    async def start(self) -> None:
        """Spawn the subprocess (idempotent) and launch reader/stderr tasks."""
        if self._proc is not None:
            return
        args = self._build_args()
        logger.info("spawning claude: model=%s mode=%s workdir=%s mcp=%s",
                    self.model, self.permission_mode, self.workdir,
                    bool(self.mcp_config))
        logger.debug("claude argv: %s", args)
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workdir,
                env=self._build_env(),
                limit=_READ_LIMIT,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                "claude CLI not found on PATH — install Claude Code and run `claude login`"
            ) from e
        self._reader_task = asyncio.create_task(self._read_stdout(), name="claude-stdout")
        self._stderr_task = asyncio.create_task(self._read_stderr(), name="claude-stderr")

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def _read_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        stdout = self._proc.stdout
        try:
            while True:
                try:
                    raw = await stdout.readline()
                except ValueError:
                    # A line longer than the buffer cap. asyncio's readline()
                    # catches the underlying LimitOverrunError and re-raises it
                    # as ValueError, having already cleared/advanced its buffer
                    # past the offending data — so we skip this one line and
                    # keep reading instead of killing the whole stream.
                    logger.warning(
                        "skipped oversized stdout line (> %d bytes)", _READ_LIMIT
                    )
                    continue
                except asyncio.IncompleteReadError:
                    break
                if not raw:
                    break
                line = raw.decode("utf-8", "replace")
                ev = parse_line(line)
                if ev is not None:
                    # Cache the session id from the first init line we see.
                    sid = getattr(ev, "session_id", None)
                    if sid and self.session_id is None:
                        self.session_id = sid
                    await self._queue.put(ev)
        finally:
            await self._queue.put(STREAM_CLOSED)

    async def _read_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        stderr = self._proc.stderr
        while True:
            try:
                raw = await stderr.readline()
            except (asyncio.LimitOverrunError, ValueError):
                continue
            if not raw:
                break
            msg = raw.decode("utf-8", "replace").rstrip()
            if msg:
                logger.debug("claude stderr: %s", msg)

    # ── writing ───────────────────────────────────────────────────────────—

    async def send_user_turn(self, content: Union[str, list[Any]]) -> None:
        """Write one ``{"type":"user",...}`` line to stdin (+ newline) and drain."""
        line = json.dumps(
            {"type": "user", "message": {"role": "user", "content": content}}
        )
        await self._write_line(line)

    async def send_control_response(self, request_id: str, inner: dict[str, Any]) -> None:
        """Answer a ``control_request`` — ``request_id`` nested inside ``response``.

        Wire-format invariant (wisp, verified against CLI 2.1.168): a top-level
        ``request_id`` is ignored and the turn stalls forever. Keep it nested.
        """
        line = json.dumps(
            {
                "type": "control_response",
                "response": {
                    "subtype": "success",
                    "request_id": request_id,
                    "response": inner,
                },
            }
        )
        await self._write_line(line)

    async def _write_line(self, line: str) -> None:
        if self._proc is None or self._proc.stdin is None:
            logger.warning("write to claude stdin but no running process")
            return
        try:
            self._proc.stdin.write(line.encode("utf-8") + b"\n")
            await self._proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as e:
            logger.warning("writing to claude stdin failed: %s", e)

    # ── reading ───────────────────────────────────────────────────────────—

    async def next_event(
        self, timeout: Optional[float] = None
    ) -> Union[ChatEvent, object, None]:
        """Pop the next event. Returns :data:`STREAM_CLOSED` at EOF, or ``None``
        if ``timeout`` elapses first."""
        try:
            if timeout is None:
                return await self._queue.get()
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    # ── shutdown ──────────────────────────────────────────────────────────—

    async def aclose(self) -> None:
        """Close stdin, terminate the process, and cancel reader/stderr tasks."""
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if proc is not None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except Exception:
                pass
            if proc.returncode is None:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    await proc.wait()
        for task in (self._reader_task, self._stderr_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
