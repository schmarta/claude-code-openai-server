# cci-server Latency Optimization Plan

**Goal:** Make `claude-code-openai-server` respond as fast as possible — cut time-to-first-token (TTFT) and total wall-clock per Hermes turn — without breaking the suspend/resume tool loop or the in-flight session that runs the agent itself.

**Architecture today:** Stateless OpenAI request → `ConversationManager` either creates a fresh conversation (spawns a brand-new `claude` Node subprocess + registers an MCP bridge) or resumes a suspended one. The subprocess lives only for the duration of ONE Hermes turn (across its tool continuations), then is torn down. Every new user turn pays a full cold start.

**Central constraint (from cci-server-gateway skill):** the code we edit is the code running us. No `--reload`, so disk edits are safe; a restart kills in-flight streams, so land changes via the detached-timer self-restart at end of turn. Lucas demands real runtime evidence, not just pytest.

---

## Where the time actually goes (hypotheses to confirm in Phase 0)

| # | Suspected cost | Path affected | Rough size | Confidence |
|---|----------------|---------------|-----------|------------|
| 1 | `claude` Node subprocess cold start every fresh turn (`ClaudeSession.start`) | both | ~1–3 s wall, pure overhead before any token | high |
| 2 | Model = `claude-opus-4-8` (slowest TTFT + tok/s) | both | seconds | high |
| 3 | MCP handshake: spawned claude must HTTP-handshake + `list_tools` before first tool turn acts | tool path | sub-second RTT | medium |
| 4 | Full conversation re-folded into one user turn each time → large input, no cross-turn prompt-cache reuse (fresh process) | both | grows with history | medium |
| 5 | `--verbose` stdout volume parsed line-by-line by the reader | both | small | low |
| 6 | Default asyncio loop (no uvloop/httptools) | server | small | low |

**Phase 0 must measure these before we touch anything** — Lucas wants proof, and we should not optimize blind.

---

## Phase 0 — Instrument & baseline (DO FIRST, no behavior change)

### Task 0.1: Add timing logs around the hot path
**Files:** `app/claude_session.py`, `app/routes/chat.py`, `app/conversation.py`

- In `ClaudeSession.start()` wrap the `create_subprocess_exec` in a `time.monotonic()` delta; log `spawn_ms`.
- In `_stream` / `_tool_stream`, record t0 at request entry and log `ttft_ms` when the first `TextChunk`/`TextDelta` is emitted, and `total_ms` at finish, plus token count for `tok_per_s`.
- For the tool path, log time from `session.start()` to first MCP `list_tools` and to first event.

Gate all of it behind `CCI_TIMING_LOG` (default off) so it's free in normal operation.

### Task 0.2: Capture a baseline harness
**Files:** `scripts/bench.py` (new)

A throwaway-port harness (per skill: `CCI_PORT=8799 CCI_HOST=127.0.0.1 .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8799`) that fires N representative requests through `/v1/chat/completions`:
- a short no-tool prompt (autonomous path),
- a tool-forcing prompt (tool path),
- a multi-turn continuation.

Record p50/p95 of spawn_ms, ttft_ms, total_ms, tok_per_s. **This is the before number every later phase is measured against.**

**Verify:** run bench against current `main`, save `bench-baseline.json`. Do NOT touch :8787.

---

## Phase 1 — Cheap, low-risk wins (land these first)

### Task 1.1: uvloop + httptools
**Files:** `app/main.py` (or the systemd `ExecStart`), `requirements`/`pyproject`

- `pip install uvloop httptools` into `.venv`.
- Launch uvicorn with `--loop uvloop --http httptools` (or `uvicorn.run(..., loop="uvloop")`).
- **Verify:** bench shows no regression; faster JSON/SSE throughput. Confirm boot log still prints `starting on …`.

### Task 1.2: Drop `--verbose` (test-gated)
**Files:** `app/claude_session.py:109` `_build_args`

- `--verbose` inflates stdout the reader must parse. Test whether `--include-partial-messages` still streams deltas WITHOUT `--verbose` on stream-json (it may be required together — verify empirically before removing).
- If partials survive: remove `--verbose`. If not: keep it, note the coupling.
- **Verify:** throwaway-port E2E — force a text→tool→text turn, confirm deltas still arrive and the seam still renders.

### Task 1.3: Make model the primary lever (config, not code)
**Files:** none — operational knob `CCI_DEFAULT_MODEL`

- opus-4-8 is the dominant wall-clock cost. `sonnet` is ~2–4× faster TTFT/tok-s; `haiku` faster still.
- This is a **quality/speed tradeoff Lucas owns** — present numbers from Phase 0 (opus vs sonnet vs haiku on the same prompts) and let him pick the default. Do NOT silently downgrade.
- Optional follow-up: per-request model routing already works (`resolve_model`) — Hermes can send `haiku` for trivial turns, `opus` for hard ones, if we wire it client-side.

**Decision point for Lucas:** what default model? (keep opus / sonnet default / haiku default / per-turn routing)

---

## Phase 2 — Warm subprocess pool (the big TTFT win)

**Idea:** eliminate cold start (#1) by keeping a small pool of pre-spawned, idle `claude` processes ready to adopt. This is the single largest TTFT reduction available.

**The snag:** a fresh tool turn launches claude with `--mcp-config` pointing at *that conversation's* unique MCP URL, fixed at spawn time. A pre-spawned process can't know the conv_id yet.

**Resolution — pre-allocate conv_ids:** the pool pre-mints conv_id `pooledN`, registers an empty `ConversationBridge`, and spawns claude with `--mcp-config` → `/mcp/pooledN` already baked in. When a request arrives:
1. Pop a warm process from the pool.
2. Populate its bridge's `tools` with `req.tools` (the bridge is already registered; tools are read lazily by `list_tools` on first call, so late-binding is safe as long as no tool call has happened yet — which it hasn't, the user turn isn't sent until adoption).
3. Send the folded user turn.
4. Refill the pool in the background.

**Why tool late-binding works:** `McpBridge._list_tools` reads `bridge.mcp_tools()` at call time, and the spawned claude only calls `list_tools` after it receives the first user turn. As long as we set `bridge.tools` *before* `send_user_turn`, claude sees the correct schema. Hermes' tool set is also near-constant across turns, lowering risk.

**Caveats to design around:**
- System prompt is also fixed at spawn. In bare mode it's effectively constant (`bare_model_system_prompt` / the standard Hermes system prompt) — pool only works if the per-request system prompt is stable. If Hermes varies it per request, the pool must spawn with the canonical prompt and we accept that, OR only pool when the request's system prompt matches the pooled one (fall back to cold spawn otherwise).
- Pool size small (2–3) to bound idle Node memory.
- Health: discard a pooled proc whose `running` is False; never hand out a dead one.
- GC must not reap pooled (idle-but-unused) processes.

### Task 2.1: `WarmPool` class
**Files:** `app/warmpool.py` (new), `app/conversation.py`, `app/main.py`

- `WarmPool(size, settings, mcp)` pre-spawns `size` `ClaudeSession`s with pre-registered bridges and canonical prompt/model.
- `async acquire(req_tools, system) -> (conv_id, session, bridge)`: returns a warm one if system prompt matches; else `None` (caller cold-spawns).
- Background refill task; lifespan-managed start/stop.
- Gate behind `CCI_WARM_POOL_SIZE` (default 0 = disabled, so it ships dark).

### Task 2.2: Wire `ConversationManager.create` to try the pool first
**Files:** `app/conversation.py:143` `create`

- Try `pool.acquire(...)`; on hit, skip `session.start()` (already running) and just set tools + send turn. On miss, current cold path unchanged.
- **Verify:** Phase-0 bench with `CCI_WARM_POOL_SIZE=2` vs baseline — expect spawn_ms ≈ 0 on pool hits, TTFT down by the cold-start delta. Capture `/proc` argv to prove the adopted process carries the right `--mcp-config` port (skill pitfall: throwaway must use `CCI_PORT` so the advertised MCP URL self-matches).

---

## Phase 3 — Cross-turn session reuse (architectural, higher risk — only if Phases 1–2 insufficient)

Today each user turn = fresh process + full history re-fold (#4). Claude Code supports session continuity (`--resume <session_id>`; we already capture `session_id` in `claude_session.py:206`). Keeping one subprocess alive across user turns would:
- skip cold start entirely on follow-ups,
- send only the *new* turn instead of re-folding all history → far smaller input → lower TTFT and cost,
- benefit from claude's own context retention.

**Why it's risky:** the whole `ConversationManager` is built around per-turn lifecycles; the OpenAI protocol is stateless so we'd need to key a live session to a logical chat (no stable id in the request today). This likely needs a Hermes-side conversation id header. **Defer unless measurements show history re-fold dominates.** Document as a known lever; don't build speculatively (YAGNI).

---

## Phase 4 — Prompt-cache verification (cheap if it works)

Anthropic prompt caching can make a fixed system-prompt prefix near-free on repeat. In bare mode the system prompt is constant.

### Task 4.1: Confirm whether Claude Code CLI sets cache breakpoints
- Inspect CLI behavior / docs for automatic `cache_control` on the system prompt under stream-json.
- If it caches: a warm pool + stable prompt already reaps it. If not and there's a flag, enable it.
- **Verify:** compare TTFT of 2nd identical-prefix request vs 1st; a cache hit shows a step-down in input processing time / `cache_read` tokens in usage.

---

## Rollout & verification discipline

1. Land Phase 0 first; commit; deploy via detached self-restart timer; collect baseline next turn.
2. Each later task: edit → `cd ~/claude-code-openai-server && .venv/bin/pytest -q` (was 82 passing) → throwaway-port E2E on 8799 (set `CCI_PORT`!) → bench diff → commit separately with descriptive message.
3. Push only when the batch is proven. Never restart `:8787` inline — use:
   ```
   systemd-run --user --on-active=15 --unit=cci-reload \
     systemctl --user restart cci-server.service
   ```
4. Report concrete before/after p50/p95 TTFT and total_ms — real runtime evidence, not pytest alone.

## Open questions for Lucas
- **Default model?** Biggest single lever. Keep opus (quality) or drop to sonnet/haiku, or per-turn routing?
- Acceptable idle memory for a warm pool (each idle `claude` ≈ a Node process)? Pool size 2–3 OK?
- Is the per-request system prompt Hermes sends stable across turns? (Determines warm-pool hit rate.)
- Appetite for Phase 3 (cross-turn session reuse) if Phases 1–2 don't get us far enough?

## Risks / tradeoffs
- Faster model = lower quality. Lucas's call.
- Warm pool late-binds tools/prompt — safe only while no tool call precedes adoption (it doesn't) and the system prompt matches; mismatches must fall back to cold spawn, not serve a wrong-prompt process.
- Removing `--verbose` may disable partial streaming — must be empirically verified, not assumed.
- Cross-turn reuse touches the core lifecycle/TOCTOU logic — highest blast radius; gate and test hard.
