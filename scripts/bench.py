#!/usr/bin/env python3
"""Latency benchmark harness for cci-server (throwaway-port, self-contained).

Launches its OWN uvicorn instance on a throwaway port with ``CCI_TIMING_LOG=1``,
fires a fixed set of representative requests through ``/v1/chat/completions``,
parses the ``cci.timing`` log lines the server emits, and records p50/p95 of the
spawn / TTFT / total / throughput metrics to JSON.

It NEVER touches the live server. By design it sets ``CCI_PORT`` (not just
``--port``) so the per-conversation MCP bridge URL the spawned ``claude`` is told
to dial back self-matches the throwaway port — otherwise tool turns would report
"no tools" (see the gateway skill's pitfall). The launched server is killed by
PID on exit.

Scenarios (each repeated ``--iters`` times):
  * autonomous  — a short no-tool prompt (the autonomous path)
  * tool        — a tool-forcing prompt (the tool path, suspends on a tool call)
  * continuation— returns the tool result so the same conversation resumes

Usage:
  .venv/bin/python scripts/bench.py [--port 8799] [--iters 3] [--out bench-baseline.json]
                                    [--loop uvloop --http httptools] [--model opus]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent

WEATHER_TOOL = [
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

# Parses lines like:
#   ... cci.timing: timing event=turn label=tool ttft_ms=611 total_ms=4530 \
#       completion_tokens=88 tok_per_s=24.6
_TIMING_RE = re.compile(r"\btiming\s+(event=\S+.*)$")
_KV_RE = re.compile(r"(\w+)=(\S+)")


def _percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    # linear-interpolation percentile (numpy-compatible)
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


class ServerHandle:
    """A throwaway uvicorn instance, with its stderr scraped for timing lines."""

    def __init__(self, port: int, loop: str | None, http: str | None, model: str | None):
        self.port = port
        self.loop = loop
        self.http = http
        self.model = model
        self.proc: subprocess.Popen | None = None
        self.lines: list[str] = []
        self._reader: threading.Thread | None = None

    def start(self) -> None:
        env = dict(os.environ)
        env["CCI_PORT"] = str(self.port)
        env["CCI_HOST"] = "127.0.0.1"
        env["CCI_TIMING_LOG"] = "1"
        if self.model:
            env["CCI_DEFAULT_MODEL"] = self.model
        argv = [
            sys.executable, "-m", "uvicorn", "app.main:app",
            "--host", "127.0.0.1", "--port", str(self.port),
        ]
        if self.loop:
            argv += ["--loop", self.loop]
        if self.http:
            argv += ["--http", self.http]
        self.proc = subprocess.Popen(
            argv, cwd=str(ROOT), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        self._reader = threading.Thread(target=self._scrape, daemon=True)
        self._reader.start()

    def _scrape(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        for line in self.proc.stderr:
            self.lines.append(line.rstrip("\n"))

    def wait_ready(self, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        url = f"http://127.0.0.1:{self.port}/healthz"
        while time.monotonic() < deadline:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(
                    "server exited during startup:\n" + "\n".join(self.lines[-20:])
                )
            try:
                r = httpx.get(url, timeout=2)
                if r.status_code == 200:
                    break
            except Exception:
                time.sleep(0.2)
        else:
            raise RuntimeError("server did not become ready in time")
        # Confirm the boot log advertises the throwaway port (skill guardrail).
        boot = "\n".join(self.lines)
        if f"starting on 127.0.0.1:{self.port}" not in boot:
            raise RuntimeError(
                f"boot log did not confirm port {self.port}; refusing to bench "
                "(possible cross-talk with another instance)"
            )

    def timing_events(self) -> list[dict]:
        out: list[dict] = []
        for line in self.lines:
            m = _TIMING_RE.search(line)
            if not m:
                continue
            ev: dict = {}
            for k, v in _KV_RE.findall(m.group(1)):
                if v in ("na",):
                    ev[k] = None
                else:
                    try:
                        ev[k] = float(v) if ("." in v or v.isdigit() is False) else int(v)
                    except ValueError:
                        ev[k] = v
            out.append(ev)
        return out

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


def _post(base: str, payload: dict) -> dict:
    r = httpx.post(f"{base}/v1/chat/completions", json=payload, timeout=180)
    r.raise_for_status()
    return r.json()


def run_scenarios(base: str, iters: int) -> None:
    for i in range(iters):
        # ── autonomous (no tools) ──
        _post(base, {
            "messages": [{"role": "user", "content": "Reply with exactly the word: pong"}],
            "stream": False,
        })

        # ── tool path: step 1 forces a tool call ──
        msgs = [{"role": "user", "content":
                 "What is the weather in Paris? Use the get_weather tool, then answer in one sentence."}]
        body1 = _post(base, {"messages": msgs, "tools": WEATHER_TOOL, "stream": False})
        choice1 = body1["choices"][0]
        tcs = choice1["message"].get("tool_calls") or []
        if choice1.get("finish_reason") != "tool_calls" or not tcs:
            print(f"  iter {i}: WARN tool step did not produce a tool_call; "
                  f"finish={choice1.get('finish_reason')}", flush=True)
            continue
        call = tcs[0]

        # ── continuation: return the tool result, resume the same conversation ──
        msgs2 = msgs + [
            {"role": "assistant", "content": choice1["message"].get("content"),
             "tool_calls": tcs},
            {"role": "tool", "tool_call_id": call["id"],
             "name": call["function"]["name"], "content": '{"temp_c": 21, "summary": "sunny"}'},
        ]
        _post(base, {"messages": msgs2, "tools": WEATHER_TOOL, "stream": False})
        print(f"  iter {i + 1}/{iters} done", flush=True)


def summarize(events: list[dict]) -> dict:
    metrics = ("spawn_ms", "ttft_ms", "total_ms", "tok_per_s")

    def collect(evs: list[dict]) -> dict:
        samples = {m: [e[m] for e in evs if e.get(m) is not None] for m in metrics}
        return {
            m: {
                "n": len(vals),
                "p50": _percentile(vals, 0.50),
                "p95": _percentile(vals, 0.95),
                "min": min(vals) if vals else None,
                "max": max(vals) if vals else None,
            }
            for m, vals in samples.items()
        }

    by_label: dict[str, dict] = {}
    for label in sorted({e.get("label", "?") for e in events}):
        by_label[label] = collect([e for e in events if e.get("label") == label])
    return {
        "n_events": len(events),
        "overall": collect(events),
        "by_label": by_label,
        "raw_events": events,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--out", default="bench-baseline.json")
    ap.add_argument("--loop", default=None, help="uvicorn --loop (e.g. uvloop, asyncio)")
    ap.add_argument("--http", default=None, help="uvicorn --http (e.g. httptools, h11)")
    ap.add_argument("--model", default=None, help="CCI_DEFAULT_MODEL override for the run")
    ap.add_argument("--label", default=None, help="free-text label stored in the JSON")
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    srv = ServerHandle(args.port, args.loop, args.http, args.model)
    print(f"launching server on :{args.port} (loop={args.loop or 'auto'} "
          f"http={args.http or 'auto'} model={args.model or 'default'})", flush=True)
    srv.start()
    try:
        srv.wait_ready()
        print("server ready; running scenarios", flush=True)
        run_scenarios(base, args.iters)
        # Give the server a beat to flush the last turn's timing line.
        time.sleep(0.5)
        events = srv.timing_events()
    finally:
        srv.stop()

    result = {
        "config": {
            "port": args.port, "iters": args.iters, "loop": args.loop,
            "http": args.http, "model": args.model, "label": args.label,
        },
        **summarize(events),
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path} ({result['n_events']} timing events)", flush=True)
    for label, stats in result["by_label"].items():
        print(f"\n[{label}]", flush=True)
        for m, s in stats.items():
            if s["n"]:
                print(f"  {m:12s} n={s['n']:2d} p50={s['p50']:.1f} p95={s['p95']:.1f}",
                      flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
