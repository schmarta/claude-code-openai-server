"""Unit tests for ConversationManager + ConversationBridge with a fake session.

No live CLI: a FakeSession lets us drive the event stream deterministically and
assert the suspend/resume bookkeeping, classification, expired-continuation, GC,
and the MCP bridge's batch/resolve/fail-all plumbing.
"""

import asyncio

import pytest

from app.conversation import (
    CLOSED,
    RUNNING,
    SUSPENDED,
    Conversation,
    ConversationManager,
    DoneChunk,
    ErrorChunk,
    ExpiredContinuation,
    TextChunk,
    ToolCallsChunk,
)
from app.claude_session import STREAM_CLOSED
from app.events import Error, TextDelta, TurnDone
from app.mcp_bridge import ConversationBridge, McpBridge
from app.openai_models import (
    ChatCompletionRequest,
    ChatMessage,
    FunctionCall,
    FunctionDef,
    ToolCall,
    ToolDef,
)


_STREAM_CLOSED = STREAM_CLOSED


class FakeSession:
    """Stands in for ClaudeSession: events come from a preloaded list."""

    def __init__(self, events):
        self._events = list(events)
        self.sent_turns = []
        self.closed = False

    async def start(self):
        pass

    async def send_user_turn(self, content):
        self.sent_turns.append(content)

    async def next_event(self, timeout=None):
        if self._events:
            return self._events.pop(0)
        return _STREAM_CLOSED

    async def aclose(self):
        self.closed = True

    @property
    def running(self):
        return not self.closed


def make_settings(**over):
    from app.config import Settings

    base = dict(suspended_ttl_s=300, idle_session_ttl_s=900, gc_interval_s=30,
                request_timeout_s=30, permission_mode="bypassPermissions", port=8787)
    base.update(over)
    return Settings(**base)


def weather_tools():
    return [ToolDef(function=FunctionDef(
        name="get_weather",
        description="x",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
    ))]


# ── classification ───────────────────────────────────────────────────────────


def test_is_continuation():
    cont = ChatCompletionRequest(messages=[
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content=None, tool_calls=[
            ToolCall(id="call_1", function=FunctionCall(name="f", arguments="{}"))]),
        ChatMessage(role="tool", tool_call_id="call_1", content="result"),
    ])
    assert ConversationManager.is_continuation(cont) is True

    fresh = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
    assert ConversationManager.is_continuation(fresh) is False


def test_trailing_tool_messages_parallel():
    req = ChatCompletionRequest(messages=[
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content=None),
        ChatMessage(role="tool", tool_call_id="call_1", content="a"),
        ChatMessage(role="tool", tool_call_id="call_2", content="b"),
    ])
    trailing = ConversationManager._trailing_tool_messages(req)
    assert [m.tool_call_id for m in trailing] == ["call_1", "call_2"]


# ── bridge plumbing ──────────────────────────────────────────────────────————


async def test_bridge_dispatch_suspends_and_resolves():
    bridge = ConversationBridge("c1", weather_tools())

    async def caller():
        return await bridge.dispatch("get_weather", {"city": "Paris"})

    task = asyncio.create_task(caller())
    batch = await bridge.collect_batch(1)
    assert len(batch) == 1
    call = batch[0]
    assert call.name == "get_weather"
    assert call.id.startswith("call_")
    assert call.openai_tool_call()["function"]["arguments"] == '{"city": "Paris"}'
    # caller still blocked
    assert not task.done()
    assert bridge.resolve(call.id, '{"temp_c": 21}') is True
    result = await asyncio.wait_for(task, timeout=1)
    assert result == '{"temp_c": 21}'


async def test_bridge_parallel_batch():
    bridge = ConversationBridge("c1", weather_tools())
    t1 = asyncio.create_task(bridge.dispatch("get_weather", {"city": "Paris"}))
    t2 = asyncio.create_task(bridge.dispatch("get_time", {"tz": "UTC"}))
    batch = await bridge.collect_batch(2)
    assert {c.name for c in batch} == {"get_weather", "get_time"}
    for c in batch:
        bridge.resolve(c.id, "ok")
    assert await asyncio.wait_for(t1, 1) == "ok"
    assert await asyncio.wait_for(t2, 1) == "ok"


async def test_bridge_fail_all_unblocks():
    from app.mcp_bridge import ToolResultError

    bridge = ConversationBridge("c1", weather_tools())
    task = asyncio.create_task(bridge.dispatch("get_weather", {"city": "Paris"}))
    await bridge.collect_batch(1)
    bridge.fail_all("conversation closed")
    with pytest.raises(ToolResultError):
        await asyncio.wait_for(task, timeout=1)


def test_bridge_mcp_tools_schema():
    bridge = ConversationBridge("c1", weather_tools())
    tools = bridge.mcp_tools()
    assert len(tools) == 1
    assert tools[0].name == "get_weather"
    assert tools[0].inputSchema["properties"]["city"]["type"] == "string"


# ── full turn loop with fakes ─────────────────────────────────────────————————


async def test_run_turn_text_then_done():
    mcp = McpBridge()
    mgr = ConversationManager(mcp, make_settings())
    bridge = ConversationBridge("c1", [])
    sess = FakeSession([TextDelta("Hello"), TextDelta(" world"),
                        TurnDone(stop_reason="end_turn", usage={"input_tokens": 1, "output_tokens": 2})])
    conv = Conversation(conv_id="c1", session=sess, bridge=bridge, model="sonnet")
    mgr._conversations["c1"] = conv
    mcp.register(bridge)

    chunks = [c async for c in mgr.run_turn(conv)]
    texts = [c.text for c in chunks if isinstance(c, TextChunk)]
    dones = [c for c in chunks if isinstance(c, DoneChunk)]
    assert "".join(texts) == "Hello world"
    assert dones and dones[0].finish_reason == "stop"
    assert dones[0].usage["completion_tokens"] == 2
    assert conv.state == CLOSED
    assert sess.closed is True


async def test_run_turn_error_closes():
    mcp = McpBridge()
    mgr = ConversationManager(mcp, make_settings())
    bridge = ConversationBridge("c1", [])
    sess = FakeSession([Error("boom")])
    conv = Conversation(conv_id="c1", session=sess, bridge=bridge, model="sonnet")
    mgr._conversations["c1"] = conv
    mcp.register(bridge)
    chunks = [c async for c in mgr.run_turn(conv)]
    assert isinstance(chunks[0], ErrorChunk)
    assert conv.state == CLOSED


async def test_suspend_resume_roundtrip():
    """Drive a full suspend → resume cycle end-to-end with fakes."""
    mcp = McpBridge()
    mgr = ConversationManager(mcp, make_settings())
    bridge = ConversationBridge("c1", weather_tools())
    mcp.register(bridge)

    # The assistant tool_use event the parser would emit. We simulate the
    # subprocess by having a background coroutine call bridge.dispatch (as the
    # MCP handler would) right before the AssistantToolUse event is consumed.
    from app.events import AssistantToolUse, ToolUseBlock

    tool_use_ev = AssistantToolUse(tool_uses=[
        ToolUseBlock(id="toolu_1", name="mcp__hermes__get_weather", input={"city": "Paris"})])
    sess = FakeSession([tool_use_ev])  # then STREAM_CLOSED-> but we suspend first
    conv = Conversation(conv_id="c1", session=sess, bridge=bridge, model="sonnet")
    mgr._conversations["c1"] = conv

    # Background: the MCP handler suspends on the Future.
    dispatch_task = asyncio.create_task(bridge.dispatch("get_weather", {"city": "Paris"}))

    chunks = []
    async for c in mgr.run_turn(conv):
        chunks.append(c)
    # Should have yielded exactly one ToolCallsChunk and parked SUSPENDED.
    tc = [c for c in chunks if isinstance(c, ToolCallsChunk)]
    assert len(tc) == 1
    call = tc[0].calls[0]
    assert conv.state == SUSPENDED
    assert mgr._pending_index[call.id] == "c1"

    # Now resume with a continuation request carrying the tool result.
    cont = ChatCompletionRequest(
        tools=weather_tools(),
        messages=[
            ChatMessage(role="user", content="weather?"),
            ChatMessage(role="assistant", content=None, tool_calls=[
                ToolCall(id=call.id, function=FunctionCall(name="get_weather", arguments='{"city":"Paris"}'))]),
            ChatMessage(role="tool", tool_call_id=call.id, content='{"temp_c":21}'),
        ],
    )
    resumed = await mgr.resume(cont)
    assert resumed.conv_id == "c1"
    assert conv.state == RUNNING
    # The dispatch Future was resolved with the tool result.
    assert await asyncio.wait_for(dispatch_task, 1) == '{"temp_c":21}'
    # tool_call_id was consumed from the index.
    assert call.id not in mgr._pending_index


async def test_resume_expired_raises():
    mcp = McpBridge()
    mgr = ConversationManager(mcp, make_settings())
    cont = ChatCompletionRequest(messages=[
        ChatMessage(role="tool", tool_call_id="call_unknown", content="x")])
    with pytest.raises(ExpiredContinuation):
        await mgr.resume(cont)


# ── GC ───────────────────────────────────────────────────————————————————————


async def test_gc_reaps_suspended_past_ttl():
    mcp = McpBridge()
    mgr = ConversationManager(mcp, make_settings(suspended_ttl_s=0, idle_session_ttl_s=99999))
    bridge = ConversationBridge("c1", weather_tools())
    mcp.register(bridge)
    sess = FakeSession([])
    conv = Conversation(conv_id="c1", session=sess, bridge=bridge, model="sonnet", state=SUSPENDED)
    mgr._conversations["c1"] = conv

    # A pending call should be failed (not left hanging) on GC.
    task = asyncio.create_task(bridge.dispatch("get_weather", {"city": "Paris"}))
    await bridge.collect_batch(1)

    n = await mgr.gc_once()
    assert n == 1
    assert conv.state == CLOSED
    assert sess.closed is True
    from app.mcp_bridge import ToolResultError
    with pytest.raises(ToolResultError):
        await asyncio.wait_for(task, 1)


async def test_gc_keeps_fresh():
    mcp = McpBridge()
    mgr = ConversationManager(mcp, make_settings(suspended_ttl_s=300, idle_session_ttl_s=900))
    bridge = ConversationBridge("c1", [])
    mcp.register(bridge)
    conv = Conversation(conv_id="c1", session=FakeSession([]), bridge=bridge, model="sonnet", state=SUSPENDED)
    mgr._conversations["c1"] = conv
    assert await mgr.gc_once() == 0
    assert conv.state == SUSPENDED
