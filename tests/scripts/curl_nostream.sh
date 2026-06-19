#!/usr/bin/env bash
# Non-streaming text completion (no tools). Expect finish_reason "stop".
set -euo pipefail
BASE="${1:-http://127.0.0.1:8787}"
curl -sS "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer any-key-accepted' \
  -d '{
    "model": "sonnet",
    "stream": false,
    "messages": [{"role": "user", "content": "Reply with exactly the word PONG."}]
  }' | python3 -m json.tool
