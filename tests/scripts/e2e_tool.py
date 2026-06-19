#!/usr/bin/env python3
"""M6 E2E: the full OpenAI tool loop (suspend/resume) against a running server.

Simulates what hermes does:
  1. POST with tools=[get_weather] + a prompt that needs it
     -> expect finish_reason "tool_calls" + a tool_call for get_weather(Paris)
  2. POST the same history + assistant(tool_calls) + tool(result)
     -> expect finish_reason "stop" + final answer mentioning the result

Runs the loop in both non-streaming and streaming modes.

Usage: .venv/bin/python tests/scripts/e2e_tool.py [base_url]
"""
import json
import sys

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8787"
MODEL = "sonnet"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }
]

PROMPT = "What is the weather in Paris right now? Use the get_weather tool, then answer in one sentence."
TOOL_RESULT = '{"temp_c": 21, "summary": "sunny"}'


def post(payload):
    return httpx.post(f"{BASE}/v1/chat/completions", json=payload, timeout=120)


def stream_post(payload):
    """Return (content, tool_calls, finish_reason) from an SSE stream."""
    content = ""
    finish = None
    # assemble tool calls by index
    tcs: dict[int, dict] = {}
    with httpx.stream("POST", f"{BASE}/v1/chat/completions", json=payload, timeout=120) as r:
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            ch = obj["choices"][0]
            delta = ch.get("delta", {})
            if delta.get("content"):
                content += delta["content"]
            for tc in delta.get("tool_calls", []) or []:
                idx = tc.get("index", 0)
                slot = tcs.setdefault(idx, {"id": None, "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function", {})
                if fn.get("name"):
                    slot["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    return content, [tcs[k] for k in sorted(tcs)], finish


def run_loop(streaming: bool) -> bool:
    label = "stream" if streaming else "non-stream"
    print(f"\n========== TOOL LOOP ({label}) ==========", flush=True)
    base_messages = [{"role": "user", "content": PROMPT}]

    # ── step 1: expect tool_calls ──
    payload1 = {"model": MODEL, "tools": TOOLS, "stream": streaming, "messages": base_messages}
    if streaming:
        content1, tool_calls, finish1 = stream_post(payload1)
    else:
        r1 = post(payload1)
        print("step1 status", r1.status_code, flush=True)
        body1 = r1.json()
        print(json.dumps(body1, indent=2)[:700], flush=True)
        choice1 = body1["choices"][0]
        finish1 = choice1["finish_reason"]
        tool_calls = choice1["message"].get("tool_calls") or []
        content1 = choice1["message"].get("content")

    print(f"step1 finish={finish1} tool_calls={tool_calls}", flush=True)
    if finish1 != "tool_calls" or not tool_calls:
        print(f"{label}: FAIL (no tool_calls)", flush=True)
        return False
    call = tool_calls[0]
    call_id = call["id"]
    fn_name = call["function"]["name"]
    fn_args = json.loads(call["function"]["arguments"] or "{}")
    if fn_name != "get_weather" or "city" not in fn_args:
        print(f"{label}: FAIL (wrong call: {fn_name} {fn_args})", flush=True)
        return False

    # ── step 2: return the tool result, expect final answer ──
    messages2 = base_messages + [
        {"role": "assistant", "content": content1, "tool_calls": tool_calls},
        {"role": "tool", "tool_call_id": call_id, "name": fn_name, "content": TOOL_RESULT},
    ]
    payload2 = {"model": MODEL, "tools": TOOLS, "stream": streaming, "messages": messages2}
    if streaming:
        content2, _, finish2 = stream_post(payload2)
    else:
        r2 = post(payload2)
        print("step2 status", r2.status_code, flush=True)
        body2 = r2.json()
        print(json.dumps(body2, indent=2)[:700], flush=True)
        choice2 = body2["choices"][0]
        finish2 = choice2["finish_reason"]
        content2 = choice2["message"].get("content") or ""

    print(f"step2 finish={finish2} content={content2!r}", flush=True)
    ok = finish2 == "stop" and ("21" in content2 or "sunny" in content2.lower())
    print(f"{label}:", "PASS" if ok else "FAIL", flush=True)
    return ok


if __name__ == "__main__":
    ns = run_loop(streaming=False)
    st = run_loop(streaming=True)
    print("\nE2E TOOL LOOP:", "PASS" if (ns and st) else "FAIL", flush=True)
    sys.exit(0 if (ns and st) else 1)
