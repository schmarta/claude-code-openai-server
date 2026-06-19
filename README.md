# claude-code-interface

Use **Claude Code** from any OpenAI-compatible client.

This is a small HTTP server that makes the `claude` CLI look like an OpenAI API
(the way LM Studio or Ollama do). Point any OpenAI client — hermes, the OpenAI
Python SDK, etc. — at it and you get Claude Code, using your existing
`claude login` (no API key, no per-token billing).

## What it does

- `GET /v1/models` and `POST /v1/chat/completions` (streaming + non-streaming).
- **OpenAI function calling:** your client's tools are passed through to Claude.
  Claude calls them, the server returns a normal `tool_calls` response, your
  client runs the tool and sends the result back, Claude continues.
- Claude's own built-in tools (Read/Edit/Bash/…) run internally the whole time.

Under the hood it drives `claude` as a persistent subprocess over its
`stream-json` protocol.

## Requirements

- The `claude` CLI on your `PATH`, already logged in (`claude login`).
- Python 3.11.

## Install

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

## Run

```bash
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8787
```

Config is via `CCI_*` env vars (see `.env.example`) — e.g. `CCI_DEFAULT_MODEL=opus`,
`CCI_DEFAULT_WORKDIR=~/cci-workspace`.

## Use it

Any OpenAI client, `base_url = http://127.0.0.1:8787/v1`, any `api_key`:

```bash
curl http://127.0.0.1:8787/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"opus","messages":[{"role":"user","content":"hello"}]}'
```

From **hermes**, add a provider in `~/.hermes/config.yaml` and select it:

```yaml
providers:
  claude-code:
    name: Claude Code
    base_url: http://127.0.0.1:8787/v1
    api_key: any-string-accepted
    api_mode: chat_completions
    default_model: opus
    models: [opus, sonnet, haiku]
```

Models: `opus`, `sonnet`, `haiku`, `fable` (or any `claude-*` id).

## Test

```bash
.venv/bin/python -m pytest -q                    # unit tests (no CLI needed)
.venv/bin/python tests/scripts/e2e_autonomous.py # live: text, needs a running server
.venv/bin/python tests/scripts/e2e_tool.py       # live: full tool loop
```
