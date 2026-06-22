"""Reassemble Claude's streamed assistant text for chat clients (e.g. Discord).

The ``claude`` CLI streams each assistant *text block* as a run of ``text_delta``
chunks with no trailing newline. When a built-in tool (Read/Edit/Bash/…) runs
between two text blocks the bridge surfaces nothing for the gap — the tool runs
internally — so the last delta of one block and the first delta of the next
arrive glued together::

    "...only works for one-off CLI runs.Found it — ~/.hermes/SOUL.md is ..."

Newlines appear to "randomly" disappear because the loss happens *only* at these
tool-call boundaries (within a single block, ``\\n`` rides through the deltas
fine). :class:`SegmentJoiner` watches for the boundaries via
:meth:`SegmentJoiner.tool_boundary` and guarantees exactly a blank line between
segments, without doubling newlines a block already supplies.
"""

from __future__ import annotations


class SegmentJoiner:
    """Insert a blank line between assistant text segments split by a tool use.

    Fed the raw text of each ``TextDelta`` via :meth:`feed`; told about each
    tool-call boundary via :meth:`tool_boundary`. Returns text ready to emit
    (possibly newline-prefixed). Stateful and total — never raises.
    """

    def __init__(self) -> None:
        self._emitted = False
        self._pending_break = False
        self._trailing_newlines = 0

    def tool_boundary(self) -> None:
        """Record that a built-in tool ran between two text segments."""
        if self._emitted:
            self._pending_break = True

    def feed(self, text: str) -> str:
        """Return ``text`` to emit, prefixed with a separator when one is due."""
        if not text:
            return ""
        if self._pending_break:
            self._pending_break = False
            # Aim for exactly one blank line (two newlines) at the seam, counting
            # newlines the previous segment already left and ones this one leads
            # with, so we never stack 3+ blank lines.
            leading = len(text) - len(text.lstrip("\n"))
            need = 2 - self._trailing_newlines - leading
            if need > 0:
                text = "\n" * need + text
        self._emitted = True
        stripped = text.rstrip("\n")
        if stripped == "":
            self._trailing_newlines += len(text)
        else:
            self._trailing_newlines = len(text) - len(stripped)
        return text

    def flush(self) -> str:
        """Drain any buffered tail. The joiner buffers nothing; always ``""``."""
        return ""
