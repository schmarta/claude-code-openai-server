"""Latency instrumentation for the hot path, gated behind ``CCI_TIMING_LOG``.

All output goes to the ``cci.timing`` logger at INFO in a stable, grep-friendly
``key=value`` shape so ``scripts/bench.py`` can parse it:

    timing event=spawn label=tool spawn_ms=842
    timing event=turn label=tool ttft_ms=611 total_ms=4530 completion_tokens=88 tok_per_s=24.6

When timing is disabled (the default) every helper is a cheap no-op: the
:class:`TurnTimer` records nothing and emits no log line, so normal operation
pays only a couple of branch checks.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger("cci.timing")


def log_spawn(enabled: bool, label: str, spawn_ms: float) -> None:
    """Emit the subprocess spawn cost for one ``ClaudeSession.start``."""
    if not enabled:
        return
    logger.info("timing event=spawn label=%s spawn_ms=%.0f", label, spawn_ms)


class TurnTimer:
    """Measures time-to-first-token and total wall-clock for one turn.

    ``t0`` is stamped at construction (request entry). :meth:`first_token` is
    called on the first text byte emitted to the client; :meth:`done` is called
    once the turn finishes and emits the summary line. Throughput is computed
    over the *generation* window (total − ttft) so spawn/queue time before the
    first token does not deflate tok/s.
    """

    def __init__(self, enabled: bool, label: str) -> None:
        self.enabled = enabled
        self.label = label
        self.t0 = time.monotonic() if enabled else 0.0
        self._ttft: Optional[float] = None
        self._fired = False

    def first_token(self) -> None:
        if self.enabled and self._ttft is None:
            self._ttft = time.monotonic() - self.t0

    def done(self, completion_tokens: int = 0) -> None:
        if not self.enabled or self._fired:
            return
        self._fired = True
        total = time.monotonic() - self.t0
        ttft = self._ttft
        # tok/s over the generation window; fall back to total if no token seam
        # was recorded (e.g. a tool-only turn with no assistant text).
        gen = total - ttft if ttft is not None else total
        tok_per_s = completion_tokens / gen if gen > 0 and completion_tokens else 0.0
        ttft_repr = f"{ttft * 1000:.0f}" if ttft is not None else "na"
        logger.info(
            "timing event=turn label=%s ttft_ms=%s total_ms=%.0f "
            "completion_tokens=%d tok_per_s=%.1f",
            self.label, ttft_repr, total * 1000, completion_tokens, tok_per_s,
        )
