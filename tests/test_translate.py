"""Unit tests for app.translate (pure functions; no live CLI)."""

import json

from app.events import TurnDone
from app.openai_models import ChatMessage, FunctionCall, ToolCall
from app.translate import (
    DONE,
    completion_response,
    finish_chunk,
    fold_conversation,
    map_finish_reason,
    message_text,
    role_chunk,
    split_system,
    sse,
    text_chunk,
    tool_calls_chunk,
    usage_from_turn,
)


def test_message_text_string_and_parts():
    assert message_text(ChatMessage(role="user", content="hello")) == "hello"
    parts = [
        {"type": "text", "text": "a"},
        {"type": "image_url", "image_url": {"url": "x"}},
        {"type": "text", "text": "b"},
    ]
    assert message_text(ChatMessage(role="user", content=parts)) == "ab"
    assert message_text(ChatMessage(role="assistant", content=None)) == ""


def test_split_system():
    msgs = [
        ChatMessage(role="system", content="be terse"),
        ChatMessage(role="system", content="be correct"),
        ChatMessage(role="user", content="hi"),
    ]
    convo, system = split_system(msgs)
    assert system == "be terse\n\nbe correct"
    assert len(convo) == 1 and convo[0].role == "user"


def test_fold_single_user():
    convo = [ChatMessage(role="user", content="what is 2+2?")]
    assert fold_conversation(convo) == "what is 2+2?"


def test_fold_multiturn_transcript():
    convo = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="hello"),
        ChatMessage(role="user", content="bye"),
    ]
    folded = fold_conversation(convo)
    assert folded.startswith("Conversation so far:")
    assert "User: hi" in folded
    assert "Assistant: hello" in folded
    assert folded.rstrip().endswith("User: bye")


def test_fold_includes_assistant_tool_calls():
    convo = [
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="c1", function=FunctionCall(name="get_weather", arguments='{"city":"Paris"}'))],
        ),
        ChatMessage(role="user", content="and now?"),
    ]
    folded = fold_conversation(convo)
    assert "called tools: get_weather" in folded


def test_map_finish_reason():
    assert map_finish_reason("end_turn") == "stop"
    assert map_finish_reason("max_tokens") == "length"
    assert map_finish_reason("tool_use") == "tool_calls"
    assert map_finish_reason("stop_sequence") == "stop"
    assert map_finish_reason(None) == "stop"
    assert map_finish_reason("weird") == "stop"


def test_usage_from_turn_toplevel():
    turn = TurnDone(
        usage={
            "input_tokens": 3,
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 10,
            "output_tokens": 6,
        },
        total_cost_usd=0.05,
    )
    u = usage_from_turn(turn)
    assert u["prompt_tokens"] == 113
    assert u["completion_tokens"] == 6
    assert u["total_tokens"] == 119
    assert u["cost_usd"] == 0.05


def test_usage_from_turn_iterations_sum():
    turn = TurnDone(
        usage={
            "input_tokens": 3,
            "output_tokens": 6,
            "iterations": [
                {"input_tokens": 5, "output_tokens": 7, "cache_read_input_tokens": 1},
                {"input_tokens": 2, "output_tokens": 3},
            ],
        }
    )
    u = usage_from_turn(turn)
    # prompt = 5+1 + 2 = 8 ; completion = 7+3 = 10
    assert u["prompt_tokens"] == 8
    assert u["completion_tokens"] == 10
    assert "cost_usd" not in u


def test_usage_from_turn_empty():
    u = usage_from_turn(TurnDone())
    assert u == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_sse_and_done_format():
    s = sse({"a": 1})
    assert s == 'data: {"a": 1}\n\n'
    assert DONE == "data: [DONE]\n\n"


def test_chunk_shapes():
    rc = role_chunk("id1", "opus", 100)
    assert rc["object"] == "chat.completion.chunk"
    assert rc["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert rc["choices"][0]["finish_reason"] is None

    tc = text_chunk("id1", "opus", 100, "hello")
    assert tc["choices"][0]["delta"] == {"content": "hello"}

    fc = finish_chunk("id1", "opus", 100, "stop", {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})
    assert fc["choices"][0]["finish_reason"] == "stop"
    assert fc["choices"][0]["delta"] == {}
    assert fc["usage"]["total_tokens"] == 3

    toolc = tool_calls_chunk("id1", "opus", 100, [{"index": 0, "id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}}])
    assert toolc["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "f"


def test_completion_response_shape():
    body = completion_response(
        "id1", "opus", 100, content="hi", finish_reason="stop",
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "hi"}
    assert body["choices"][0]["finish_reason"] == "stop"
    # round-trips as JSON
    json.dumps(body)


def test_completion_response_with_tool_calls():
    tcs = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    body = completion_response("id1", "opus", 100, content=None, finish_reason="tool_calls", tool_calls=tcs)
    assert body["choices"][0]["message"]["tool_calls"] == tcs
    assert body["choices"][0]["finish_reason"] == "tool_calls"
