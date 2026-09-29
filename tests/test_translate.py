"""Unit tests for app.translate (pure functions; no live CLI)."""

import json

from app.events import TurnDone
from app.openai_models import ChatMessage, FunctionCall, ToolCall
from app.translate import (
    DONE,
    _CONTINUE_GUARD,
    _LIVE_GUARD,
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


def test_fold_single_user_preserves_data_image():
    data = "iVBORw0KGgo="
    convo = [ChatMessage(role="user", content=[
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
    ])]
    assert fold_conversation(convo) == [
        {"type": "text", "text": "what is this?"},
        {"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": data,
        }},
    ]


def test_fold_rejects_invalid_data_image():
    convo = [ChatMessage(role="user", content=[
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,not-valid!"}},
    ])]
    assert fold_conversation(convo) == "what is this?"


_PNG = "iVBORw0KGgo="


def _img_part(data: str = _PNG, media: str = "image/png") -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}}


def _image_blocks(folded) -> list[dict]:
    if not isinstance(folded, list):
        return []
    return [b for b in folded if isinstance(b, dict) and b.get("type") == "image"]


def _folded_text(folded) -> str:
    """All text of a fold, whether it came back as a string or as blocks."""
    if isinstance(folded, str):
        return folded
    return "".join(b.get("text", "") for b in folded if b.get("type") == "text")


def test_fold_keeps_an_image_from_an_earlier_turn():
    """An image sent in an earlier turn must survive the refold.

    Every fresh subprocess is fed the whole conversation, so an image that is
    flattened to text on the way in leaves the model answering follow-up
    questions about a picture it can no longer see — silently, and with a
    confident guess.
    """
    convo = [
        ChatMessage(role="user", content=[
            {"type": "text", "text": "what is in this photo?"},
            _img_part(),
        ]),
        ChatMessage(role="assistant", content="A red bicycle."),
        ChatMessage(role="user", content="what colour was it?"),
    ]
    folded = fold_conversation(convo)

    imgs = _image_blocks(folded)
    assert imgs, "image from an earlier turn was dropped from the fold"
    assert imgs[0]["source"]["data"] == _PNG
    text = _folded_text(folded)
    assert "User: what is in this photo?" in text
    assert "Assistant: A red bicycle." in text
    assert _LIVE_GUARD in text
    assert text.rstrip().endswith("what colour was it?")


def test_fold_keeps_history_images_in_transcript_order():
    """Each image stays attached to the turn that sent it: its block follows
    that turn's labelled line and precedes the next one."""
    convo = [
        ChatMessage(role="user", content=[{"type": "text", "text": "first"}, _img_part("iVBORw0KGgo=")]),
        ChatMessage(role="assistant", content="ok"),
        ChatMessage(role="user", content="last"),
    ]
    folded = fold_conversation(convo)
    assert isinstance(folded, list)
    kinds = [b.get("type") for b in folded]
    assert "image" in kinds
    i = kinds.index("image")
    before = "".join(b.get("text", "") for b in folded[:i] if b.get("type") == "text")
    after = "".join(b.get("text", "") for b in folded[i:] if b.get("type") == "text")
    assert before.rstrip().endswith("User: first")
    assert "Assistant: ok" in after


def test_fold_keeps_an_image_when_the_tail_is_not_a_user_turn():
    """The continuation shape (…user with an image → assistant tool_calls →
    tool result) folds entirely into context. The image must ride along there
    too, otherwise a rebuilt tool turn answers blind."""
    convo = [
        ChatMessage(role="user", content=[
            {"type": "text", "text": "identify this"},
            _img_part(),
        ]),
        ChatMessage(
            role="assistant", content=None,
            tool_calls=[ToolCall(id="c1", function=FunctionCall(name="lookup", arguments="{}"))],
        ),
        ChatMessage(role="tool", tool_call_id="c1", content="a red bicycle"),
    ]
    folded = fold_conversation(convo)

    assert _image_blocks(folded), "image dropped when folding a non-user tail"
    text = _folded_text(folded)
    assert "User: identify this" in text
    assert "Tool: a red bicycle" in text
    assert text.rstrip().endswith(_CONTINUE_GUARD)


def test_fold_without_history_images_stays_a_plain_string():
    """Regression guard: the block form is only used when there is an image to
    carry — an image-free conversation folds to exactly the same string as before."""
    convo = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="hello"),
        ChatMessage(role="user", content="bye"),
    ]
    assert isinstance(fold_conversation(convo), str)


def test_fold_with_history_image_and_empty_live_turn_emits_no_empty_block():
    """An empty final user message must not become an empty text block: the API
    rejects those, and the string path appended nothing for it either."""
    convo = [
        ChatMessage(role="user", content=[{"type": "text", "text": "look"}, _img_part()]),
        ChatMessage(role="assistant", content="ok"),
        ChatMessage(role="user", content="   "),
    ]
    folded = fold_conversation(convo)
    assert _image_blocks(folded), "image dropped"
    assert all(b.get("text", "x").strip() for b in folded if b["type"] == "text")


def test_fold_multiturn_transcript():
    convo = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="hello"),
        ChatMessage(role="user", content="bye"),
    ]
    folded = fold_conversation(convo)
    # Prior turns are carried as labelled context...
    assert "User: hi" in folded
    assert "Assistant: hello" in folded
    # ...but the folded content must NOT read as a continuable transcript. The
    # latest user message is the raw tail — no "User:" prefix, no dangling role
    # cue for the model to keep filling in (which made it fabricate a next
    # "User:" turn instead of stopping).
    assert folded.rstrip().endswith("bye")
    assert not folded.rstrip().endswith("User: bye")
    assert not folded.rstrip().endswith(":")
    # An explicit guard against continuing the transcript is present.
    assert _LIVE_GUARD in folded


def test_fold_multiturn_does_not_end_on_role_cue():
    """Regression: a dangling 'User:'/'Assistant:' cue at the tail invites the
    bare model to text-complete the transcript and emit a fabricated user turn."""
    convo = [
        ChatMessage(role="user", content="what is the capital of France?"),
        ChatMessage(role="assistant", content="Paris."),
        ChatMessage(role="user", content="and Germany?"),
    ]
    tail = fold_conversation(convo).rstrip().splitlines()[-1]
    assert tail == "and Germany?"


def test_fold_assistant_tail_folds_into_context():
    """A conversation ending on an assistant turn (prefill / continue pattern)
    must not present the assistant's own text as the live message to reply to:
    it is labelled into the context and a continue-style guard closes the fold."""
    convo = [
        ChatMessage(role="user", content="write a haiku"),
        ChatMessage(role="assistant", content="Autumn wind rises"),
    ]
    folded = fold_conversation(convo)
    assert "User: write a haiku" in folded
    assert "Assistant: Autumn wind rises" in folded
    assert folded.rstrip().endswith(_CONTINUE_GUARD)
    assert not folded.rstrip().endswith(":")


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


def test_fold_renders_tool_result_as_prose_and_drops_tool_call_id():
    """Characterize the KNOWN, ACCEPTED history-refold limitation for a full
    tool round-trip folded into a fresh subprocess: an assistant tool call
    becomes '[called tools: fn({...})]', the tool RESULT becomes a 'Tool: <text>'
    line, and the tool_call_id correlating the two is dropped — structured
    tool history is flattened to prose. Recorded as a regression characterization
    (previously the 'Tool:' path had no coverage), not a change of behavior."""
    convo = [
        ChatMessage(role="user", content="weather in Paris?"),
        ChatMessage(
            role="assistant", content=None,
            tool_calls=[ToolCall(id="call_abc123",
                                 function=FunctionCall(name="get_weather",
                                                       arguments='{"city":"Paris"}'))],
        ),
        ChatMessage(role="tool", tool_call_id="call_abc123",
                    content='{"temp_c":21,"summary":"sunny"}'),
        ChatMessage(role="user", content="and Berlin?"),
    ]
    folded = fold_conversation(convo)
    # assistant tool call rendered as prose
    assert '[called tools: get_weather({"city":"Paris"})]' in folded
    # tool result rendered as a plain 'Tool:' line
    assert 'Tool: {"temp_c":21,"summary":"sunny"}' in folded
    # the tool_call_id correlation is NOT preserved anywhere in the fold
    assert "call_abc123" not in folded
    # the whole prior turn is re-sent alongside the new one (re-billed each turn)
    assert "Paris" in folded and folded.rstrip().endswith("and Berlin?")


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
    d = u["prompt_tokens_details"]
    assert d == {"cached_tokens": 100, "cache_write_tokens": 10}
    assert d["cached_tokens"] + d["cache_write_tokens"] <= u["prompt_tokens"]


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
    assert u["prompt_tokens_details"] == {"cached_tokens": 1, "cache_write_tokens": 0}


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
