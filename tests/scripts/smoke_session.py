#!/usr/bin/env python3
"""M3 smoke test: drive a real `claude` subprocess for two turns via ClaudeSession.

Hits the live CLI (model sonnet, bypassPermissions). Run:
    .venv/bin/python tests/scripts/smoke_session.py
"""
import asyncio
import sys

from app.claude_session import STREAM_CLOSED, ClaudeSession
from app.events import AssistantToolUse, Error, Init, TextDelta, TurnDone


async def run_turn(sess: ClaudeSession, text: str) -> tuple[str, TurnDone | None]:
    print(f"\n>>> SEND: {text!r}", flush=True)
    await sess.send_user_turn(text)
    chunks: list[str] = []
    while True:
        ev = await sess.next_event(timeout=90)
        if ev is None:
            print("  [timeout]", flush=True)
            return "".join(chunks), None
        if ev is STREAM_CLOSED:
            print("  [stream closed]", flush=True)
            return "".join(chunks), None
        if isinstance(ev, Init):
            print(f"  init session={ev.session_id} model={ev.model}", flush=True)
        elif isinstance(ev, TextDelta):
            chunks.append(ev.text)
        elif isinstance(ev, AssistantToolUse):
            print(f"  tool_use: {[b.name for b in ev.tool_uses]}", flush=True)
        elif isinstance(ev, TurnDone):
            print(f"  TURN DONE stop={ev.stop_reason} cost={ev.total_cost_usd} "
                  f"usage_in={ev.usage.get('input_tokens')} out={ev.usage.get('output_tokens')}",
                  flush=True)
            return "".join(chunks), ev
        elif isinstance(ev, Error):
            print(f"  ERROR: {ev.message}", flush=True)
            return "".join(chunks), None


async def main() -> int:
    sess = ClaudeSession(
        claude_bin="claude",
        model="sonnet",
        permission_mode="bypassPermissions",
        workdir="/tmp/cci_probe_ws",
        enable_tool_search=False,
    )
    await sess.start()
    ok = True
    try:
        t1, d1 = await run_turn(sess, "Reply with exactly the word PONG and nothing else.")
        print(f"  collected text: {t1!r}", flush=True)
        ok &= "PONG" in t1.upper() and d1 is not None

        t2, d2 = await run_turn(sess, "Now reply with exactly the word PING and nothing else.")
        print(f"  collected text: {t2!r}", flush=True)
        ok &= "PING" in t2.upper() and d2 is not None

        print(f"\n  session still running after 2 turns: {sess.running}", flush=True)
        ok &= sess.running
    finally:
        await sess.aclose()
        print(f"  after aclose, running: {sess.running}", flush=True)

    print("\nSMOKE:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
