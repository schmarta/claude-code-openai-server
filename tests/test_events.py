"""Unit tests for app.events.parse_line.

Ported from wisp's src/claude/event.rs test vectors (the exact line shapes
captured from a live `claude --output-format stream-json --verbose
--include-partial-messages` run), adapted to our event types, plus vectors for
`mcp__hermes__*` tool calls and the usage-bearing `result` line captured live
from CLI 2.1.183.
"""

from app.events import (
    AssistantToolUse,
    ControlDialog,
    Error,
    Init,
    PermissionRequest,
    QuestionRequest,
    TextDelta,
    TurnDone,
    parse_line,
)

# ── line vectors ─────────────────────────────────────────────────────────────

INIT = '{"type":"system","subtype":"init","session_id":"abc-123","model":"claude-opus-4-8","tools":["Read"]}'
DELTA_H = '{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"H"}},"session_id":"abc-123"}'
DELTA_REST = '{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"i there."}}}'
RESULT_OK = '{"type":"result","subtype":"success","is_error":false,"result":"Hi there.","session_id":"abc-123"}'
RESULT_ERR = '{"type":"result","subtype":"error_during_execution","is_error":true,"result":"boom","session_id":"abc-123"}'

ASSISTANT_READ = '{"type":"assistant","message":{"model":"claude-opus-4-8","role":"assistant","content":[{"type":"tool_use","id":"toolu_01GeCxYcf9sXFDZSfQmpcgwk","name":"Read","input":{"file_path":"/Users/lucas/Projects/wisp/note.txt"}}]}}'
ASSISTANT_BASH = '{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"toolu_017FcpUw3HcHZ3gxBhoyMCp5","name":"Bash","input":{"command":"find /tmp -iname \'note*.txt\'","description":"Find note.txt"}}]}}'
STREAM_TOOL_START = '{"type":"stream_event","event":{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_x","name":"Read","input":{}}}}'
ASSISTANT_THINKING = '{"type":"assistant","message":{"role":"assistant","content":[{"type":"thinking","thinking":"hmm","signature":"x"}]}}'
ASSISTANT_TEXT = '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Hi there."}]}}'

# hermes function calls surface as mcp__hermes__<fn> tool_use blocks.
ASSISTANT_HERMES = '{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"toolu_h1","name":"mcp__hermes__get_weather","input":{"city":"Paris"}}]}}'
ASSISTANT_HERMES_PARALLEL = '{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"toolu_h1","name":"mcp__hermes__get_weather","input":{"city":"Paris"}},{"type":"tool_use","id":"toolu_h2","name":"mcp__hermes__get_time","input":{"tz":"UTC"}}]}}'
# A mixed step: a builtin Read AND a hermes call in one assistant message.
ASSISTANT_MIXED = '{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"toolu_r","name":"Read","input":{"file_path":"/tmp/x"}},{"type":"tool_use","id":"toolu_h","name":"mcp__hermes__lookup","input":{"q":"a"}}]}}'

# Captured live from CLI 2.1.183.
RESULT_USAGE = '{"type":"result","subtype":"success","is_error":false,"result":"PONG","stop_reason":"end_turn","session_id":"f360e668","total_cost_usd":0.0557505,"usage":{"input_tokens":3,"cache_creation_input_tokens":8428,"cache_read_input_tokens":16945,"output_tokens":6}}'

CONTROL_REQ_EDIT = '{"type":"control_request","request_id":"55ee4b08-485b-4e60-8490-fea16bb5505a","request":{"subtype":"can_use_tool","tool_name":"Edit","display_name":"Edit","input":{"file_path":"/private/var/probe.txt","old_string":"hello world","new_string":"HELLO world","replace_all":false},"tool_use_id":"toolu_01RYhouC"}}'
CONTROL_REQ_BASH = '{"type":"control_request","request_id":"aabbccdd","request":{"subtype":"can_use_tool","tool_name":"Bash","display_name":"Bash","input":{"command":"echo hello","description":"test"},"tool_use_id":"toolu_bash_01"}}'
CONTROL_REQ_ASK = '{"type":"control_request","request_id":"613575bc","request":{"subtype":"can_use_tool","tool_name":"AskUserQuestion","input":{"questions":[{"question":"Cats or dogs?","header":"Pref","options":[{"label":"Cats","description":"x"}],"multiSelect":false}]}}}'
CONTROL_REQ_DIALOG = '{"type":"control_request","request_id":"dlg-1","request":{"subtype":"request_user_dialog","dialog_kind":"exit_plan_mode","payload":{}}}'
CONTROL_REQ_ELICIT = '{"type":"control_request","request_id":"eli-1","request":{"subtype":"elicitation","message":"Pick a region"}}'
CONTROL_REQ_UNKNOWN = '{"type":"control_request","request_id":"x","request":{"subtype":"interrupt"}}'


# ── tests ────────────────────────────────────────────────────────────────────


def test_init():
    assert parse_line(INIT) == Init(session_id="abc-123", model="claude-opus-4-8")


def test_text_deltas():
    assert parse_line(DELTA_H) == TextDelta("H")
    assert parse_line(DELTA_REST) == TextDelta("i there.")


def test_result_ok_and_error():
    ev = parse_line(RESULT_OK)
    assert isinstance(ev, TurnDone)
    assert ev.result_text == "Hi there."
    assert parse_line(RESULT_ERR) == Error("boom")


def test_result_usage_fields():
    ev = parse_line(RESULT_USAGE)
    assert isinstance(ev, TurnDone)
    assert ev.stop_reason == "end_turn"
    assert ev.total_cost_usd == 0.0557505
    assert ev.usage["input_tokens"] == 3
    assert ev.usage["output_tokens"] == 6
    assert ev.usage["cache_read_input_tokens"] == 16945


def test_assistant_read_tool_use():
    ev = parse_line(ASSISTANT_READ)
    assert isinstance(ev, AssistantToolUse)
    assert len(ev.tool_uses) == 1
    b = ev.tool_uses[0]
    assert b.name == "Read"
    assert b.input["file_path"] == "/Users/lucas/Projects/wisp/note.txt"
    assert not b.is_hermes
    assert ev.hermes_calls == []
    assert len(ev.builtin_calls) == 1


def test_assistant_bash_tool_use():
    ev = parse_line(ASSISTANT_BASH)
    assert isinstance(ev, AssistantToolUse)
    assert ev.tool_uses[0].name == "Bash"
    assert ev.tool_uses[0].input["command"] == "find /tmp -iname 'note*.txt'"


def test_hermes_tool_use_split_and_name():
    ev = parse_line(ASSISTANT_HERMES)
    assert isinstance(ev, AssistantToolUse)
    b = ev.tool_uses[0]
    assert b.is_hermes
    assert b.hermes_function_name == "get_weather"
    assert b.input == {"city": "Paris"}
    assert len(ev.hermes_calls) == 1
    assert ev.builtin_calls == []


def test_hermes_parallel_calls_batch_size():
    ev = parse_line(ASSISTANT_HERMES_PARALLEL)
    assert isinstance(ev, AssistantToolUse)
    assert len(ev.hermes_calls) == 2
    assert [b.hermes_function_name for b in ev.hermes_calls] == ["get_weather", "get_time"]


def test_mixed_step_splits_builtin_and_hermes():
    ev = parse_line(ASSISTANT_MIXED)
    assert isinstance(ev, AssistantToolUse)
    assert len(ev.tool_uses) == 2
    assert len(ev.hermes_calls) == 1
    assert len(ev.builtin_calls) == 1
    assert ev.builtin_calls[0].name == "Read"
    assert ev.hermes_calls[0].hermes_function_name == "lookup"


def test_streamed_start_and_text_only_emit_nothing():
    assert parse_line(STREAM_TOOL_START) is None
    assert parse_line(ASSISTANT_THINKING) is None
    assert parse_line(ASSISTANT_TEXT) is None


def test_blank_nonjson_unknown():
    assert parse_line("") is None
    assert parse_line("   ") is None
    assert parse_line("not json at all") is None
    assert parse_line('{"type":"assistant","message":{}}') is None
    assert parse_line('{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"hmm"}}}') is None


def test_missing_fields_degrade_to_none():
    assert parse_line('{"type":"stream_event"}') is None
    assert parse_line('{"type":"system"}') is None
    assert parse_line("{}") is None
    assert parse_line("[1,2,3]") is None


def test_full_turn_reduces_to_expected_stream():
    transcript = [
        '{"type":"system","subtype":"hook_started","hook_name":"x"}',
        INIT,
        '{"type":"system","subtype":"status","status":"requesting"}',
        '{"type":"stream_event","event":{"type":"message_start","message":{}}}',
        '{"type":"stream_event","event":{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}}',
        DELTA_H,
        DELTA_REST,
        ASSISTANT_TEXT,
        '{"type":"stream_event","event":{"type":"message_stop"}}',
        '{"type":"rate_limit_event","rate_limit_info":{}}',
        RESULT_OK,
    ]
    events = [e for e in (parse_line(l) for l in transcript) if e is not None]
    assert events[0] == Init(session_id="abc-123", model="claude-opus-4-8")
    assert events[1] == TextDelta("H")
    assert events[2] == TextDelta("i there.")
    assert isinstance(events[3], TurnDone)
    assert len(events) == 4


def test_control_request_edit_permission():
    ev = parse_line(CONTROL_REQ_EDIT)
    assert isinstance(ev, PermissionRequest)
    assert ev.request_id == "55ee4b08-485b-4e60-8490-fea16bb5505a"
    assert ev.tool == "Edit"
    assert ev.summary == "/private/var/probe.txt"
    assert ev.input["new_string"] == "HELLO world"


def test_control_request_bash_summary():
    ev = parse_line(CONTROL_REQ_BASH)
    assert isinstance(ev, PermissionRequest)
    assert ev.tool == "Bash"
    assert ev.summary == "echo hello"


def test_control_request_ask_user_question():
    ev = parse_line(CONTROL_REQ_ASK)
    assert isinstance(ev, QuestionRequest)
    assert ev.request_id == "613575bc"
    assert ev.input["questions"][0]["question"] == "Cats or dogs?"


def test_control_request_dialog_and_elicitation():
    d = parse_line(CONTROL_REQ_DIALOG)
    assert isinstance(d, ControlDialog)
    assert d.kind == "user_dialog"
    assert d.summary == "exit_plan_mode"
    assert d.cancel_behavior() == "cancelled"

    e = parse_line(CONTROL_REQ_ELICIT)
    assert isinstance(e, ControlDialog)
    assert e.kind == "elicitation"
    assert e.summary == "Pick a region"
    assert e.cancel_behavior() == "cancel"


def test_control_request_unknown_subtype():
    assert parse_line(CONTROL_REQ_UNKNOWN) is None
