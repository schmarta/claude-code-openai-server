"""Post-process Claude's streamed assistant text for chat clients (e.g. Discord).

Two streaming-safe transforms, combined in :class:`OutputFilter`:

1. **Segment joining** (:class:`SegmentJoiner`) — the ``claude`` CLI streams each
   assistant *text block* as a run of ``text_delta`` chunks with no trailing
   newline. When a built-in tool (Read/Edit/Bash/…) runs between two text blocks
   the bridge surfaces nothing for the gap, so the blocks arrive glued::

       "...only works for one-off CLI runs.Found it — ~/.hermes/SOUL.md is ..."

   Newlines appear to "randomly" disappear because the loss happens *only* at
   these tool-call boundaries. The joiner guarantees one blank line at the seam.

2. **Table flattening** (:class:`TableFlattener`) — GitHub-flavoured Markdown
   pipe tables don't render in Discord; they show as raw ``| a | b |`` noise.
   Whole table blocks are rewritten as a fixed-width ASCII grid inside a ```` ```
   ```` fence (Discord renders fences monospaced, so columns line up). Detection
   is fence-aware so ``|`` inside a code block is left untouched.

Both are fed text incrementally via ``feed`` and drained once at end-of-turn via
``flush``. They are total — they never raise on odd input.
"""

from __future__ import annotations

import re

# A separator-row cell: optional leading/trailing colon around one-or-more dashes.
_SEP_CELL = re.compile(r":?-+:?")


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


# ── table flattening ─────────────────────────────────────────────────────────


def _split_cells(line: str) -> list[str]:
    """Split one GFM table row into trimmed cell texts (outer pipes optional)."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _is_separator_row(line: str) -> bool:
    cells = _split_cells(line)
    if not cells:
        return False
    return all(c and _SEP_CELL.fullmatch(c) for c in cells)


def _alignments(sep_line: str) -> list[str]:
    aligns = []
    for c in _split_cells(sep_line):
        left, right = c.startswith(":"), c.endswith(":")
        aligns.append("c" if left and right else "r" if right else "l")
    return aligns


def _looks_like_table(rows: list[str]) -> bool:
    """A header row plus a separator row is the minimum for a real GFM table."""
    return len(rows) >= 2 and _is_separator_row(rows[1])


def render_ascii_table(rows: list[str]) -> str:
    """Render GFM pipe-table ``rows`` as a fenced, column-aligned ASCII grid."""
    header = _split_cells(rows[0])
    aligns = _alignments(rows[1])
    body = [_split_cells(r) for r in rows[2:]]
    ncols = max([len(header), len(aligns)] + [len(r) for r in body])

    def pad(r: list[str]) -> list[str]:
        return r + [""] * (ncols - len(r))

    header = pad(header)
    body = [pad(r) for r in body]
    aligns = (aligns + ["l"] * ncols)[:ncols]

    widths = [len(header[i]) for i in range(ncols)]
    for r in body:
        for i in range(ncols):
            widths[i] = max(widths[i], len(r[i]))

    def cell(val: str, i: int) -> str:
        w, a = widths[i], aligns[i]
        return val.rjust(w) if a == "r" else val.center(w) if a == "c" else val.ljust(w)

    def row(r: list[str]) -> str:
        return " | ".join(cell(r[i], i) for i in range(ncols))

    sep = "-+-".join("-" * widths[i] for i in range(ncols))
    lines = [row(header), sep] + [row(r) for r in body]
    return "```\n" + "\n".join(lines) + "\n```\n"


class TableFlattener:
    """Rewrite Markdown pipe tables into fenced ASCII as text streams through.

    Normal prose streams through with at most a line's leading whitespace
    buffered, so token-by-token streaming is preserved. Only consecutive
    pipe-leading lines (table candidates) are held until the block ends, then
    converted (if they form a real table) or emitted verbatim (if not). Lines
    inside a ```` ``` ```` fence are never treated as tables.
    """

    def __init__(self) -> None:
        self._line = ""          # chars of the current line, buffered
        self._decided: bool | None = None  # None=undecided, True=table cand., False=prose
        self._table: list[str] = []  # completed candidate lines of the current block
        self._in_fence = False

    def feed(self, text: str) -> str:
        out: list[str] = []
        for ch in text:
            if ch == "\n":
                self._newline(out)
            else:
                self._char(ch, out)
        return "".join(out)

    def _char(self, ch: str, out: list[str]) -> None:
        self._line += ch
        if self._decided is None:
            if ch.isspace():
                return  # still in leading whitespace
            if ch == "|" and not self._in_fence:
                self._decided = True  # table candidate; keep buffering silently
            else:
                self._decided = False
                out.append(self._flush_table())
                out.append(self._line)  # buffered leading ws + this char
        elif self._decided is False:
            out.append(ch)

    def _newline(self, out: list[str]) -> None:
        if self._decided is True:
            self._table.append(self._line)
        elif self._decided is False:
            out.append("\n")
        else:  # undecided: blank / whitespace-only line ends any pending table
            out.append(self._flush_table())
            out.append(self._line + "\n")
        if self._line.lstrip().startswith("```"):
            self._in_fence = not self._in_fence
        self._line = ""
        self._decided = None

    def _flush_table(self) -> str:
        if not self._table:
            return ""
        rows, self._table = self._table, []
        if _looks_like_table(rows):
            return render_ascii_table(rows)
        return "".join(r + "\n" for r in rows)

    def flush(self) -> str:
        out: list[str] = []
        if self._decided is True:
            self._table.append(self._line)
            out.append(self._flush_table())
        elif self._decided is None:
            out.append(self._flush_table())
            out.append(self._line)
        # decided False: the line was already streamed out
        self._line = ""
        self._decided = None
        return "".join(out)


# ── combined filter ──────────────────────────────────────────────────────────


class OutputFilter:
    """Segment-join, then optionally table-flatten, the assistant text stream."""

    def __init__(self, *, flatten_tables: bool = True) -> None:
        self._joiner = SegmentJoiner()
        self._tables = TableFlattener() if flatten_tables else None

    def feed(self, text: str) -> str:
        joined = self._joiner.feed(text)
        if not joined:
            return ""
        return self._tables.feed(joined) if self._tables else joined

    def tool_boundary(self) -> None:
        self._joiner.tool_boundary()

    def flush(self) -> str:
        # The joiner buffers nothing; only the table flattener may hold a tail.
        return self._tables.flush() if self._tables else ""
