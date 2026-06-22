"""Reader-robustness tests for ClaudeSession._read_stdout.

An assistant turn can legitimately emit a single JSONL line larger than the
stream buffer (e.g. a huge base64 image block). asyncio's ``readline()`` catches
the underlying ``LimitOverrunError`` and re-raises it as ``ValueError``; the
reader must skip that one line and keep going rather than truncate the whole
response.
"""

from __future__ import annotations

import asyncio
import types

from app.claude_session import STREAM_CLOSED, ClaudeSession, _READ_LIMIT
from app.events import Init


def _bare_session(reader: asyncio.StreamReader) -> ClaudeSession:
    sess = object.__new__(ClaudeSession)
    sess._proc = types.SimpleNamespace(stdout=reader, returncode=None)
    sess._queue = asyncio.Queue()
    sess.session_id = None
    return sess


def _drain(q: asyncio.Queue) -> list:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_oversized_line_skipped_not_fatal():
    async def run():
        reader = asyncio.StreamReader(limit=64)  # tiny cap to force the overrun
        sess = _bare_session(reader)
        task = asyncio.create_task(sess._read_stdout())

        good1 = '{"type":"system","subtype":"init","session_id":"s1","model":"m"}\n'
        good2 = '{"type":"system","subtype":"init","session_id":"s2","model":"m"}\n'

        reader.feed_data(good1.encode())
        await asyncio.sleep(0.01)
        # One line far larger than the 64-byte cap, with no early newline.
        reader.feed_data(b"X" * 500 + b"\n")
        await asyncio.sleep(0.01)
        reader.feed_data(good2.encode())
        await asyncio.sleep(0.01)
        reader.feed_eof()
        await asyncio.wait_for(task, timeout=2)
        return _drain(sess._queue)

    events = asyncio.run(run())

    # Reader survived the oversized line and closed cleanly.
    assert events[-1] is STREAM_CLOSED
    inits = [e for e in events if isinstance(e, Init)]
    sids = {e.session_id for e in inits}
    # Both the line before AND the line after the oversized one parsed —
    # proving the oversized line was skipped, not fatal.
    assert "s1" in sids
    assert "s2" in sids


def test_clean_stream_parses_all_lines():
    async def run():
        reader = asyncio.StreamReader()
        sess = _bare_session(reader)
        task = asyncio.create_task(sess._read_stdout())
        for i in (1, 2, 3):
            reader.feed_data(
                f'{{"type":"system","subtype":"init","session_id":"s{i}","model":"m"}}\n'.encode()
            )
        reader.feed_eof()
        await asyncio.wait_for(task, timeout=2)
        return _drain(sess._queue)

    events = asyncio.run(run())
    assert events[-1] is STREAM_CLOSED
    assert {e.session_id for e in events if isinstance(e, Init)} == {"s1", "s2", "s3"}
    # session_id cached from the first init line.
