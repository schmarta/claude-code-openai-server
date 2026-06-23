# claude-code-openai-server

Use **Claude Code** from any OpenAI-compatible client.

A small HTTP server that makes the `claude` CLI look like an OpenAI API (the way
LM Studio or Ollama do). Point any OpenAI client — hermes, the OpenAI Python SDK,
Open WebUI, etc. — at it and you get Claude Code, using your existing
`claude login` (no API key, no per-token billing).

Under the hood it drives `claude` as a persistent subprocess over its
`stream-json` protocol, one process per conversation.

## What it does

- **OpenAI surface:** `GET /v1/models` and `POST /v1/chat/completions`
  (streaming + non-streaming).
- **OpenAI function calling:** your client's `tools` are passed through to Claude
  via an in-process MCP bridge. Claude calls them, the server returns a normal
  `tool_calls` response, your client runs the tool and sends the result back, and
  the same Claude subprocess resumes — the whole multi-step tool loop is one
  conversation kept alive across continuations.
- **Claude's own built-in tools** (Read/Edit/Bash/…) run internally the whole
  time (unless bare mode strips them — see below).
- **Bare model mode** (default on): presents Claude as a plain model fronted by
  your client — replaces the system prompt, drops Claude Code's dynamic context,
  and exposes only the tools your client sends.
- **Markdown table flattening** (default on): rewrites pipe tables to fenced
  ASCII so they render in clients that don't support Markdown tables (Discord).
- **Ollama / llama.cpp compatibility shims** so capability-probing frontends
  (Open WebUI, etc.) accept the server.

## Requirements

- The `claude` CLI on your `PATH`, already logged in (`claude login`).
- Python 3.11+.

## Install

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

## Run

```bash
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8787
```

Or via the installed console script, which pins the fast event loop
(uvloop + httptools) explicitly:

```bash
.venv/bin/claude-code-interface
```

Configuration is entirely through `CCI_*` environment variables (or a `.env`
file). See the full list below.

## Configuration

Copy `.env.example` to `.env` and edit, or export the vars directly. Every field
is overridable via `CCI_<FIELD>`.

```bash
# claude-code-interface configuration (env prefix CCI_). Copy to .env or export.

# ── HTTP server ──────────────────────────────────────────────────────────────
CCI_HOST=127.0.0.1
CCI_PORT=8787

# ── Auth ─────────────────────────────────────────────────────────────────────
# Optional bearer token gating every /v1 request. Safe to leave UNSET only on a
# loopback bind. The server REFUSES to start on a non-loopback host unless this
# is set — it drives Claude with bypassPermissions, so an open bind with no key
# is remote code execution for anyone who can reach the port.
# CCI_API_KEY=choose-a-long-random-secret

# ── Claude CLI ───────────────────────────────────────────────────────────────
CCI_CLAUDE_BIN=claude
CCI_DEFAULT_MODEL=claude-opus-4-8
# CCI_DEFAULT_EFFORT=high
CCI_PERMISSION_MODE=bypassPermissions
CCI_ENABLE_TOOL_SEARCH=false

# ── Bare model mode ──────────────────────────────────────────────────────────
# true (default): strip Claude Code's identity + native tools so it behaves as a
# plain model fronted by your client — the request's system message REPLACES
# claude's default prompt, dynamic context (env/git/identity) is dropped, and
# only the MCP tools your client passes survive. false = legacy append + full
# native tool set.
CCI_BARE_MODEL_MODE=true
# CCI_BARE_MODEL_SYSTEM_PROMPT=You are a helpful AI assistant.

# ── Output formatting ────────────────────────────────────────────────────────
# Rewrite Markdown pipe tables to fenced monospace ASCII so they render in
# clients that don't support Markdown tables (e.g. Discord).
CCI_FLATTEN_MARKDOWN_TABLES=true

# ── Workspace (Claude's --add-dir) ───────────────────────────────────────────
# Per-request `workdir` overrides must resolve under one of
# CCI_ALLOWED_WORKDIR_ROOTS (a JSON list); empty => only the default is allowed.
CCI_DEFAULT_WORKDIR=~/cci-workspace
# CCI_ALLOWED_WORKDIR_ROOTS=["/home/lucas/Projects"]

# ── MCP bridge ───────────────────────────────────────────────────────────────
# Path the in-process MCP server (hermes' functions) mounts at; the spawned
# claude dials it back over loopback. Rarely needs changing.
# CCI_MCP_PATH_PREFIX=/mcp

# ── Lifecycle (seconds) ──────────────────────────────────────────────────────
CCI_REQUEST_TIMEOUT_S=600
CCI_SUSPENDED_TTL_S=300
CCI_IDLE_SESSION_TTL_S=900
CCI_GC_INTERVAL_S=30

# ── Warm subprocess pool ─────────────────────────────────────────────────────
# Pre-spawned, idle `claude` procs kept ready to adopt on a fresh turn, removing
# cold-start latency. 0 = disabled (default; ships dark). 1–2 is optimal for a
# single user; larger pools waste RAM (~200 MB per idle proc) and can spike tail
# latency through refill contention. See "Warm pool" below.
CCI_WARM_POOL_SIZE=0

# ── Logging ──────────────────────────────────────────────────────────────────
CCI_LOG_LEVEL=INFO
# Emit per-turn latency metrics (spawn_ms / ttft_ms / total_ms / tok_per_s) to
# the "cci.timing" logger at INFO. Off by default so normal operation pays
# nothing; scripts/bench.py turns it on for the throwaway benchmark instance.
CCI_TIMING_LOG=false
```

### Config reference

| Var | Default | Meaning |
|-----|---------|---------|
| `CCI_HOST` | `127.0.0.1` | Bind host. Non-loopback requires `CCI_API_KEY` (see Security). |
| `CCI_PORT` | `8787` | Bind port. Also used to build the per-conversation MCP callback URL. |
| `CCI_API_KEY` | _(unset)_ | Bearer token required on every `/v1` request when set. Mandatory for a non-loopback bind. |
| `CCI_CLAUDE_BIN` | `claude` | Path to the `claude` CLI. |
| `CCI_DEFAULT_MODEL` | `claude-opus-4-8` | Model used when the request names none / an unknown one. |
| `CCI_DEFAULT_EFFORT` | _(unset)_ | Reasoning effort passed to `--effort` (e.g. `high`). |
| `CCI_PERMISSION_MODE` | `bypassPermissions` | Claude CLI permission mode. |
| `CCI_ENABLE_TOOL_SEARCH` | `false` | When false, tool schemas are injected directly (always visible) instead of behind tool-search. |
| `CCI_BARE_MODEL_MODE` | `true` | Strip Claude Code identity + native tools; behave as a plain model. |
| `CCI_BARE_MODEL_SYSTEM_PROMPT` | `You are a helpful AI assistant.` | Fallback system prompt in bare mode when a request sends none. |
| `CCI_FLATTEN_MARKDOWN_TABLES` | `true` | Rewrite pipe tables to fenced ASCII. |
| `CCI_DEFAULT_WORKDIR` | `~/cci-workspace` | Working dir granted to Claude via `--add-dir`. |
| `CCI_ALLOWED_WORKDIR_ROOTS` | `[]` | JSON list of extra roots a per-request `workdir` may resolve under. |
| `CCI_MCP_PATH_PREFIX` | `/mcp` | Mount path for the in-process MCP bridge. |
| `CCI_REQUEST_TIMEOUT_S` | `600` | Per-turn upstream timeout. |
| `CCI_SUSPENDED_TTL_S` | `300` | How long a tool-suspended conversation may wait for its results before GC. |
| `CCI_IDLE_SESSION_TTL_S` | `900` | Idle conversation eviction age. |
| `CCI_GC_INTERVAL_S` | `30` | GC sweep interval. |
| `CCI_WARM_POOL_SIZE` | `0` | Pre-spawned idle `claude` procs (cold-start removal). 0 = off. |
| `CCI_LOG_LEVEL` | `INFO` | Log level. |
| `CCI_TIMING_LOG` | `false` | Emit per-turn latency metrics to the `cci.timing` logger. |

## Security

The server drives the Claude CLI with `permission_mode=bypassPermissions` by
default, which means **anyone who can reach the port can run arbitrary code as
your user.** Two interlocks guard this:

1. **Non-loopback bind requires an API key.** `create_app()` refuses to start
   when `CCI_HOST` is not a loopback address and `CCI_API_KEY` is unset:

   ```
   RuntimeError: refusing to start: host='0.0.0.0' is not loopback and no
   api_key is set — set CCI_API_KEY to require a bearer token, or bind to 127.0.0.1
   ```

2. **Bearer auth on `/v1`.** When `CCI_API_KEY` is set, every `/v1` request must
   carry `Authorization: Bearer <key>` (constant-time compared). The MCP mount is
   intentionally exempt — it is reached only by the local Claude subprocess over
   loopback and carries no token.

**Recommended:** keep the bind on `127.0.0.1`. If you must expose it (containers,
remote clients), set a long random `CCI_API_KEY` and put it behind TLS. Binding
`0.0.0.0` / a public interface with no key will refuse to boot — by design.

## Endpoints

**OpenAI**
- `POST /v1/chat/completions` — chat (streaming + non-streaming, tool calling)
- `GET /v1/models` — model list

**Health / info**
- `GET /healthz` — `{"status":"ok",…}` (a JSON 404 here means the server is up but the route moved)
- `GET /` — service info

**Ollama / llama.cpp compatibility** (for capability-probing frontends)
- `GET /api/tags`, `POST /api/show`, `GET /api/version`, `GET /version`
- `GET /v1/props`, `GET /props`, `GET /api/v1/models`

**Internal**
- `/mcp/<conv_id>` — in-process MCP bridge, dialed only by the spawned Claude over loopback

## Use it

Any OpenAI client, `base_url = http://127.0.0.1:8787/v1`, any `api_key` when none
is configured:

```bash
curl http://127.0.0.1:8787/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"opus","messages":[{"role":"user","content":"hello"}]}'
```

With `CCI_API_KEY` set, add the bearer header:

```bash
curl http://127.0.0.1:8787/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $CCI_API_KEY" \
  -d '{"model":"opus","messages":[{"role":"user","content":"hello"}]}'
```

From **hermes**, add a provider in `~/.hermes/config.yaml` and select it:

```yaml
providers:
  claude-code:
    name: Claude Code
    base_url: http://127.0.0.1:8787/v1
    api_key: any-string-accepted   # or your CCI_API_KEY if one is set
    api_mode: chat_completions
    default_model: opus
    models: [opus, sonnet, haiku]
```

Models: `opus`, `sonnet`, `haiku`, `fable`, `opusplan` (or any `claude-*` id).
Unknown ids fall back to `CCI_DEFAULT_MODEL`; the CLI resolves an alias like
`opus` to its current concrete version.

## Warm pool

A fresh turn normally pays the full cold start: fork the `claude` Node process,
let it boot, handshake the MCP bridge, then produce a first token (~2.5 s TTFT on
modest hardware). `CCI_WARM_POOL_SIZE=N` keeps `N` pre-spawned idle procs ready
to adopt, lifting that cost off the request's critical path.

- The pool holds **one signature at a time** (model + effort + workdir + system
  prompt + tool set). A request whose signature matches a pooled proc adopts it;
  anything else cold-spawns. The pool re-targets to live traffic.
- Each adopted turn kicks a **background refill**. Under back-to-back load that
  refill (a fresh Node boot) contends for CPU with the turn it's serving and can
  spike tail latency; with normal human-gapped traffic the refill finishes in the
  idle gap and you get the win cleanly.
- Each idle proc costs ~200 MB RAM. **1–2 is optimal for a single user.** Larger
  pools waste memory and amplify refill thrash when signatures interleave.

Disabled by default (`0`), so it ships dark.

## Deploying as a service (systemd user unit)

Example `~/.config/systemd/user/cci-server.service`:

```ini
[Unit]
Description=Claude Code OpenAI Server (CCI)
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/youruser/claude-code-openai-server
EnvironmentFile=/home/youruser/claude-code-openai-server/.env
Environment=PATH=/home/youruser/.local/bin:/usr/local/bin:/usr/bin:/bin
Environment=HOME=/home/youruser
ExecStart=/home/youruser/claude-code-openai-server/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8787
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now cci-server.service
loginctl enable-linger "$USER"   # so it keeps running after you log out (headless)
```

Notes:
- The unit reads config from `.env` via `EnvironmentFile`.
- It launches `python -m uvicorn` directly (not the `claude-code-interface`
  console script), and uvicorn already auto-selects uvloop when installed. To pin
  the fast loop deterministically, add `--loop uvloop --http httptools` to
  `ExecStart` or switch it to the console script.
- There is no `--reload`: editing files on disk does not affect the running
  process until you restart the unit.

## Test

```bash
.venv/bin/python -m pytest -q                     # unit tests (no CLI needed)
.venv/bin/python tests/scripts/e2e_autonomous.py  # live: text, needs a running server
.venv/bin/python tests/scripts/e2e_tool.py        # live: full tool loop
```

### Benchmark

`scripts/bench.py` launches its **own** throwaway uvicorn instance (never touches
a running server), fires representative autonomous / tool / continuation turns
with `CCI_TIMING_LOG=1`, and writes p50/p95 of spawn / TTFT / total / throughput
to JSON:

```bash
.venv/bin/python scripts/bench.py --port 8799 --iters 3 --out bench.json
# compare warm pool vs cold:
CCI_WARM_POOL_SIZE=2 .venv/bin/python scripts/bench.py --port 8799 --iters 6
```

It sets `CCI_PORT` (not just `--port`) so the per-conversation MCP callback URL
the spawned `claude` dials self-matches the throwaway port.

## How it works

```
OpenAI client ──HTTP /v1──▶ cci-server ──stream-json (stdin/stdout)──▶ claude CLI
       ▲                         │                                         │
       └──── tool_calls ─────────┤            mcp__hermes__* tool call     │
       │                         │◀──────── in-process MCP bridge ◀────────┘
       └──── tool result ───────▶┘            (/mcp/<conv_id>, loopback)
```

- **Autonomous turn** (no `tools`): the conversation is folded into one user turn
  for a fresh subprocess; streamed text becomes SSE or a single JSON completion.
- **Tool turn:** a conversation owns one subprocess + an MCP bridge. When Claude
  calls a tool, the subprocess blocks inside the MCP call; the server returns a
  `tool_calls` response and parks the conversation `SUSPENDED`. The next request
  (carrying the tool results) resolves the pending futures and the same
  subprocess resumes. Matching is by `tool_call_id`.
- A background GC reaps suspended-too-long and idle conversations.
