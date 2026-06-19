#!/usr/bin/env python3
"""M4 E2E: hit a running server's /v1/chat/completions in both modes (no tools).

Usage: .venv/bin/python tests/scripts/e2e_autonomous.py [base_url]
Defaults to http://127.0.0.1:8787 .
"""
import json
import sys

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8787"
MODEL = "sonnet"


def test_nonstream() -> bool:
    print("\n=== non-stream ===", flush=True)
    r = httpx.post(
        f"{BASE}/v1/chat/completions",
        json={
            "model": MODEL,
            "stream": False,
            "messages": [{"role": "user", "content": "Reply with exactly the word PONG and nothing else."}],
        },
        timeout=120,
    )
    print("status", r.status_code, flush=True)
    body = r.json()
    print(json.dumps(body, indent=2)[:800], flush=True)
    choice = body["choices"][0]
    content = (choice["message"]["content"] or "").upper()
    ok = (
        r.status_code == 200
        and body["object"] == "chat.completion"
        and "PONG" in content
        and choice["finish_reason"] == "stop"
        and body.get("usage", {}).get("total_tokens", 0) >= 0
    )
    print("non-stream:", "PASS" if ok else "FAIL", flush=True)
    return ok


def test_stream() -> bool:
    print("\n=== stream ===", flush=True)
    content = ""
    saw_done = False
    finish = None
    saw_role = False
    with httpx.stream(
        "POST",
        f"{BASE}/v1/chat/completions",
        json={
            "model": MODEL,
            "stream": True,
            "messages": [{"role": "user", "content": "Reply with exactly the word PING and nothing else."}],
        },
        timeout=120,
    ) as r:
        for line in r.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                saw_done = True
                break
            obj = json.loads(data)
            delta = obj["choices"][0]["delta"]
            if delta.get("role"):
                saw_role = True
            if delta.get("content"):
                content += delta["content"]
            if obj["choices"][0]["finish_reason"]:
                finish = obj["choices"][0]["finish_reason"]
    print(f"accumulated={content!r} finish={finish} done={saw_done} role={saw_role}", flush=True)
    ok = "PING" in content.upper() and finish == "stop" and saw_done and saw_role
    print("stream:", "PASS" if ok else "FAIL", flush=True)
    return ok


if __name__ == "__main__":
    ns = test_nonstream()
    st = test_stream()
    print("\nE2E AUTONOMOUS:", "PASS" if (ns and st) else "FAIL", flush=True)
    sys.exit(0 if (ns and st) else 1)
