"""Unit tests for the warm subprocess pool (Phase 2).

A FakeClaude stands in for ClaudeSession so we exercise the pool's fill / acquire
/ refill / signature-match / dead-proc / GC-isolation logic with no real CLI.
"""

import asyncio

import pytest

from app.conversation import Conversation, ConversationManager
from app.mcp_bridge import McpBridge
from app.openai_models import (
    ChatCompletionRequest,
    ChatMessage,
    FunctionDef,
    ToolDef,
)
from app.warmpool import Signature, WarmPool, tools_signature


class FakeClaude:
    """Records spawn kwargs; alive until aclose()."""

    instances: list["FakeClaude"] = []

    def __init__(self, **kw):
        self.kw = kw
        self.started = False
        self._alive = True
        self.sent: list = []
        FakeClaude.instances.append(self)

    async def start(self):
        self.started = True

    @property
    def running(self):
        return self.started and self._alive

    async def send_user_turn(self, content):
        self.sent.append(content)

    async def aclose(self):
        self._alive = False


@pytest.fixture(autouse=True)
def _patch_session(monkeypatch):
    FakeClaude.instances.clear()
    monkeypatch.setattr("app.warmpool.ClaudeSession", FakeClaude)


def make_settings(**over):
    from app.config import Settings

    base = dict(suspended_ttl_s=300, idle_session_ttl_s=900, gc_interval_s=30,
                request_timeout_s=30, permission_mode="bypassPermissions", port=8799,
                default_model="claude-opus-4-8")
    base.update(over)
    return Settings(**base)


def weather_tools():
    return [ToolDef(function=FunctionDef(
        name="get_weather", description="x",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}}))]


async def _settle(pred, tries=200):
    """Yield to the loop until pred() or we give up (for background refills)."""
    for _ in range(tries):
        if pred():
            return True
        await asyncio.sleep(0)
    return False


# ── basics ────────────────────────────────────────────────────────────────────


async def test_disabled_pool_acquire_returns_none():
    pool = WarmPool(McpBridge(), make_settings(), 0)
    await pool.start()
    assert await pool.acquire(pool._default_signature()) is None
    assert pool._entries == []


async def test_fills_to_size_and_registers_bridges():
    mcp = McpBridge()
    pool = WarmPool(mcp, make_settings(), 2)
    await pool.start()
    assert len(pool._entries) == 2
    for e in pool._entries:
        assert e.conv_id.startswith("pool")
        assert mcp.get(e.conv_id) is e.bridge  # bridge pre-registered
        assert e.bridge.tools == []            # empty until adopt
        assert e.session.running


async def test_acquire_hit_then_background_refill():
    mcp = McpBridge()
    pool = WarmPool(mcp, make_settings(), 2)
    await pool.start()
    sig = pool._default_signature()
    entry = await pool.acquire(sig)
    assert entry is not None and entry.signature == sig
    # popped → pool momentarily at 1, background refill tops it back to 2.
    assert await _settle(lambda: len(pool._entries) == 2)


async def test_signature_mismatch_misses_and_retargets():
    mcp = McpBridge()
    pool = WarmPool(mcp, make_settings(), 2)
    await pool.start()  # entries carry the default signature (system=None)
    other = Signature(model="claude-opus-4-8", effort=None,
                      workdir=str(make_settings().default_workdir), system="hermes prompt")
    miss = await pool.acquire(other)
    assert miss is None  # no entry matches the new signature → cold spawn
    # refill re-targets: pool eventually holds only `other`-signature entries.
    assert await _settle(lambda: len(pool._entries) == 2
                         and all(e.signature == other for e in pool._entries))
    hit = await pool.acquire(other)
    assert hit is not None and hit.signature == other


async def test_dead_proc_never_handed_out():
    mcp = McpBridge()
    pool = WarmPool(mcp, make_settings(), 1)
    await pool.start()
    pool._entries[0].session._alive = False  # proc died while idle
    sig = pool._default_signature()
    assert await pool.acquire(sig) is None  # dead one is not served
    # refill replaces it with a live one.
    assert await _settle(lambda: len(pool._entries) == 1 and pool._entries[0].session.running)


async def test_stop_drains_and_unregisters():
    mcp = McpBridge()
    pool = WarmPool(mcp, make_settings(), 2)
    await pool.start()
    ids = [e.conv_id for e in pool._entries]
    await pool.stop()
    assert pool._entries == []
    for cid in ids:
        assert mcp.get(cid) is None
    assert await pool.acquire(pool._default_signature()) is None  # closed


# ── integration with ConversationManager ──────────────────────────────────────


async def test_manager_creates_pool_only_when_enabled():
    assert ConversationManager(McpBridge(), make_settings(warm_pool_size=0)).pool is None
    assert ConversationManager(McpBridge(), make_settings(warm_pool_size=2)).pool is not None


async def test_gc_never_reaps_pooled_idle():
    mcp = McpBridge()
    mgr = ConversationManager(mcp, make_settings(warm_pool_size=2,
                                                 suspended_ttl_s=0, idle_session_ttl_s=0))
    await mgr.pool.start()
    assert await mgr.gc_once() == 0          # pooled procs invisible to GC
    assert len(mgr.pool._entries) == 2       # untouched


async def test_create_adopts_from_pool_and_binds_matching_tools():
    mcp = McpBridge()
    settings = make_settings(warm_pool_size=2)
    mgr = ConversationManager(mcp, settings)
    await mgr.pool.start()

    req = ChatCompletionRequest(
        tools=weather_tools(),
        messages=[ChatMessage(role="user", content="weather in Paris?")],
    )
    # Prime the pool to the request's signature (initial fill used the default,
    # empty-tools signature) so the adoption is a genuine hit, then let refill
    # spawn the matching procs.
    sig = Signature(model=settings.default_model, effort=settings.default_effort,
                    workdir=str(settings.default_workdir), system=None,
                    tools_key=tools_signature(req.tools))
    assert await mgr.pool.acquire(sig, req.tools) is None  # miss → re-target
    assert await _settle(lambda: len(mgr.pool._entries) == 2
                         and all(e.signature == sig for e in mgr.pool._entries))
    # The pooled bridges must already advertise the tools (claude lists them at
    # handshake time, before any user turn).
    assert all(e.bridge.tools for e in mgr.pool._entries)
    spawned_before = len(FakeClaude.instances)

    conv = await mgr.create(req, model=settings.default_model,
                            workdir=settings.default_workdir,
                            effort=settings.default_effort)

    assert conv.conv_id.startswith("pool")             # adopted, not cold-spawned
    assert conv.bridge.tools == req.tools              # exact request tools bound
    assert conv.session.sent == ["weather in Paris?"]  # turn sent after binding
    assert mgr._conversations[conv.conv_id] is conv
    # Reused a pre-spawned proc (refill may add more in the background).
    assert conv.session in FakeClaude.instances[:spawned_before]


async def test_create_cold_spawns_on_tool_set_mismatch():
    mcp = McpBridge()
    settings = make_settings(warm_pool_size=1)
    mgr = ConversationManager(mcp, settings)
    await mgr.pool.start()  # pool warmed for the default, empty-tools signature
    req = ChatCompletionRequest(
        tools=weather_tools(),  # different tool set than the pooled (empty) one
        messages=[ChatMessage(role="user", content="weather?")],
    )
    conv = await mgr.create(req, model=settings.default_model,
                            workdir=settings.default_workdir,
                            effort=settings.default_effort)
    assert conv.conv_id.startswith("conv")  # cold path, tool set didn't match


async def test_create_cold_spawns_on_pool_miss():
    mcp = McpBridge()
    settings = make_settings(warm_pool_size=1)
    mgr = ConversationManager(mcp, settings)
    await mgr.pool.start()

    # A request carrying a system prompt the pool didn't pre-spawn for → miss.
    req = ChatCompletionRequest(
        tools=weather_tools(),
        messages=[
            ChatMessage(role="system", content="be terse"),
            ChatMessage(role="user", content="weather?"),
        ],
    )
    conv = await mgr.create(req, model=settings.default_model,
                            workdir=settings.default_workdir,
                            effort=settings.default_effort)
    assert conv.conv_id.startswith("conv")  # cold path, not a pool id
    assert mcp.get(conv.conv_id) is conv.bridge
