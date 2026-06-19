#!/usr/bin/env bash
# Streaming text completion (no tools). Expect SSE chunks ending in data: [DONE].
set -euo pipefail
BASE="${1:-http://127.0.0.1:8787}"
curl -sS -N "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer any-key-accepted' \
  -d '{
    "model": "sonnet",
    "stream": true,
    "messages": [{"role": "user", "content": "Count from 1 to 5, one number per line."}]
  }'
echo
