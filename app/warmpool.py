"""Warm subprocess pool — pre-spawned, idle ``claude`` procs ready to adopt.

A fresh tool turn normally pays the full cold start: fork the ``claude`` Node
process, let it boot, HTTP-handshake the MCP bridge, and only then produce a
first token. Phase-0 timing showed ``create_subprocess_exec`` itself is ~10ms;
the rest of that cost is folded into the ~2.5s TTFT. The warm pool pre-pays it:
a small number of procs are spawned ahead of time so adoption skips straight to
sending the user turn.

The snag the design works around: a tool turn launches ``claude`` with
``--mcp-config`` pointing at *that conversation's* unique MCP URL, fixed at spawn
time, and with the system prompt fixed at spawn time too. A pooled proc can't
know the request yet. So each pooled entry:

* pre-mints its own ``conv_id`` (``poolN-…``) and registers an EMPTY
  :class:`~app.mcp_bridge.ConversationBridge`,
* spawns ``claude`` with ``--mcp-config`` already baked to ``/mcp/<conv_id>`` and
  a *canonical* system prompt / model / effort / workdir (the "signature"),
* on acquire, the caller late-binds ``bridge.tools`` **before** sending the first
  user turn — safe because ``list_tools`` is only consulted once the turn starts.

Matching: :meth:`acquire` returns a warm entry only when the request's signature
equals the pooled one; otherwise it returns ``None`` and the caller cold-spawns.
The pool *adapts* its canonical signature to live traffic, so the first request
after boot (whose system prompt the pool can't predict) is a cold spawn, and
steady-state same-signature traffic hits the pool.

Gated behind ``CCI_WARM_POOL_SIZE`` (default 0 = disabled), so it ships dark.
Pooled entries live ONLY in this pool — never in ``ConversationManager``'s
registry — so the GC never reaps an idle pooled proc.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.claude_session import ClaudeSession, prompt_session_kwargs as _prompt_kwargs
from app.config import Settings
from app.mcp_bridge import ConversationBridge, McpBridge
from app.openai_models import ToolDef

logger = logging.getLogger("cci.warmpool")


def tools_signature(tools: Optional[list[ToolDef]]) -> str:
    """A stable key for a tool set's schemas (name/description/parameters).

    Empirically (CLI 2.1.185) the spawned ``claude`` calls ``list_tools`` once at
    the MCP handshake — at startup, BEFORE any user turn — and caches the result.
    So a pooled proc must be spawned with its bridge ALREADY carrying the tools
    the request will use; late-binding an empty bridge yields a proc that thinks
    it has no tools. The tool set therefore becomes part of the pool signature:
    only a request whose tools match the pooled proc's baked-in schemas may adopt
    it; anything else cold-spawns. (Hermes' tool set is near-constant within a
    session, so steady-state hit rate stays high.)
    """
    if not tools:
        return ""
    items = sorted(
        (t.function.name, t.function.description or "",
         json.dumps(t.function.parameters or {}, sort_keys=True))
        for t in tools
    )
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Signature:
    """The spawn-time identity that must match for a pooled proc to be reusable.

    Everything that influences the spawned proc's view of the world: the argv
    (model/effort/workdir/system → prompt kwargs; ``Settings`` is constant so
    ``system`` determines the rest) PLUS the tool schemas, which claude lists at
    handshake time and cannot be changed afterward (see :func:`tools_signature`).
    The per-conversation MCP URL is intrinsic to pooling and excluded.
    """

    model: str
    effort: Optional[str]
    workdir: str
    system: Optional[str]
    tools_key: str = ""


@dataclass
class PooledEntry:
    conv_id: str
    session: ClaudeSession
    bridge: ConversationBridge
    signature: Signature


class WarmPool:
    def __init__(self, mcp: McpBridge, settings: Settings, size: int) -> None:
        self.mcp = mcp
        self.settings = settings
        self.size = max(0, size)
        self._entries: list[PooledEntry] = []
        self._lock = asyncio.Lock()
        self._counter = 0
        self._target: Signature = self._default_signature()
        # The actual tool schemas for the current target signature, baked into
        # each pooled bridge at spawn (claude lists them at handshake time).
        self._target_tools: list[ToolDef] = []
        self._refilling = False
        self._closed = False

    # ── helpers ───────────────────────────────────────────────────────────—

    def _default_signature(self) -> Signature:
        return Signature(
            model=self.settings.default_model,
            effort=self.settings.default_effort,
            workdir=str(self.settings.default_workdir),
            system=None,
        )

    def _mcp_url(self, conv_id: str) -> str:
        # Mirror ConversationManager._mcp_url: built from settings.port so the
        # URL the pooled claude dials self-matches whatever port we run on.
        prefix = self.settings.mcp_path_prefix.rstrip("/")
        return f"http://127.0.0.1:{self.settings.port}{prefix}/{conv_id}"

    def _next_conv_id(self) -> str:
        self._counter += 1
        return f"pool{self._counter}-{int(time.time())}"

    # ── lifecycle ─────────────────────────────────────────────────────────—

    async def start(self) -> None:
        if self.size <= 0:
            return
        logger.info("warmpool: starting (size=%d)", self.size)
        await self._refill()

    async def stop(self) -> None:
        async with self._lock:
            self._closed = True
            entries = self._entries
            self._entries = []
        for e in entries:
            await self._discard(e)
        logger.info("warmpool: stopped (%d procs drained)", len(entries))

    async def _discard(self, entry: PooledEntry) -> None:
        self.mcp.unregister(entry.conv_id)
        await entry.session.aclose()

    async def _spawn_entry(
        self, conv_id: str, sig: Signature, tools: list[ToolDef]
    ) -> PooledEntry:
        # Bridge is pre-populated with the target tools so claude's handshake-time
        # list_tools sees the correct schemas (see tools_signature).
        bridge = ConversationBridge(conv_id, list(tools))
        self.mcp.register(bridge)
        mcp_config = {
            "mcpServers": {"hermes": {"type": "http", "url": self._mcp_url(conv_id)}}
        }
        session = ClaudeSession(
            claude_bin=self.settings.claude_bin,
            model=sig.model,
            permission_mode=self.settings.permission_mode,
            workdir=Path(sig.workdir),
            effort=sig.effort,
            mcp_config=mcp_config,
            enable_tool_search=self.settings.enable_tool_search,
            timing_log=self.settings.timing_log,
            timing_label="pool",
            **_prompt_kwargs(self.settings, sig.system),
        )
        try:
            await session.start()
        except Exception:
            self.mcp.unregister(conv_id)
            raise
        logger.debug("warmpool: spawned %s (model=%s)", conv_id, sig.model)
        return PooledEntry(conv_id, session, bridge, sig)

    # ── refill ────────────────────────────────────────────────────────────—

    async def _refill(self) -> None:
        """Top the pool up to ``size`` entries of the current target signature,
        discarding dead and stale-signature entries. Re-entrant-safe via a flag
        so concurrent triggers don't over-spawn."""
        if self._closed or self.size <= 0:
            return
        async with self._lock:
            if self._refilling:
                return
            self._refilling = True
        try:
            while True:
                async with self._lock:
                    if self._closed:
                        break
                    target = self._target
                    target_tools = list(self._target_tools)
                    keep: list[PooledEntry] = []
                    drop: list[PooledEntry] = []
                    for e in self._entries:
                        if e.session.running and e.signature == target:
                            keep.append(e)
                        else:
                            drop.append(e)
                    self._entries = keep
                    need = self.size - len(keep)
                    conv_id = self._next_conv_id() if need > 0 else None
                # Discard stale/dead outside the lock.
                for e in drop:
                    reason = "dead" if not e.session.running else "stale-signature"
                    logger.debug("warmpool: discarding %s (%s)", e.conv_id, reason)
                    await self._discard(e)
                if conv_id is None:
                    break
                try:
                    entry = await self._spawn_entry(conv_id, target, target_tools)
                except Exception:
                    logger.exception("warmpool: spawn failed; aborting refill")
                    break
                async with self._lock:
                    if self._closed or entry.signature != self._target:
                        await self._discard(entry)
                        if self._closed:
                            break
                    else:
                        self._entries.append(entry)
        finally:
            async with self._lock:
                self._refilling = False

    # ── acquire ───────────────────────────────────────────────────────────—

    async def acquire(
        self, sig: Signature, tools: Optional[list[ToolDef]] = None
    ) -> Optional[PooledEntry]:
        """Pop a live warm entry matching ``sig``, or ``None`` to cold-spawn.

        Adapts the pool's canonical target (signature + the tool schemas to bake
        into future entries) to the request and kicks a background refill either
        way (top up on hit, re-target on miss)."""
        if self.size <= 0 or self._closed:
            return None
        popped: Optional[PooledEntry] = None
        async with self._lock:
            self._target = sig
            self._target_tools = list(tools or [])
            idx = next(
                (i for i, e in enumerate(self._entries)
                 if e.signature == sig and e.session.running),
                None,
            )
            if idx is not None:
                popped = self._entries.pop(idx)
        # Refill in the background; never block the request on (re)warming.
        asyncio.create_task(self._refill())
        if popped is None:
            return None
        if not popped.session.running:  # raced to death between check and pop
            await self._discard(popped)
            return None
        logger.info("warmpool: hit %s (model=%s)", popped.conv_id, sig.model)
        return popped
