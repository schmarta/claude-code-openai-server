"""Unit tests for app.textfilter."""

from app.textfilter import SegmentJoiner


def _run(joiner: SegmentJoiner, ops) -> str:
    """Replay a script of ("text", str) / ("tool",) ops; return joined output."""
    out = []
    for op in ops:
        if op[0] == "text":
            out.append(joiner.feed(op[1]))
        else:
            joiner.tool_boundary()
    out.append(joiner.flush())
    return "".join(out)


def test_glues_segments_get_blank_line():
    # The exact reported failure: "runs." then a tool, then "Found it".
    j = SegmentJoiner()
    result = _run(j, [
        ("text", "...one-off CLI runs."),
        ("tool",),
        ("text", "Found it — SOUL.md is the persona file."),
    ])
    assert result == "...one-off CLI runs.\n\nFound it — SOUL.md is the persona file."


def test_no_boundary_passes_through_unchanged():
    j = SegmentJoiner()
    assert _run(j, [("text", "hello "), ("text", "world")]) == "hello world"


def test_newline_within_block_preserved():
    j = SegmentJoiner()
    assert _run(j, [("text", "line1\nline2\nline3")]) == "line1\nline2\nline3"


def test_prev_segment_ending_in_one_newline():
    j = SegmentJoiner()
    # prev leaves one "\n"; seam adds one more → exactly one blank line.
    assert _run(j, [("text", "a\n"), ("tool",), ("text", "b")]) == "a\n\nb"


def test_prev_segment_ending_in_blank_line_no_extra():
    j = SegmentJoiner()
    assert _run(j, [("text", "a\n\n"), ("tool",), ("text", "b")]) == "a\n\nb"


def test_next_segment_leading_newline_not_doubled():
    j = SegmentJoiner()
    assert _run(j, [("text", "a"), ("tool",), ("text", "\nb")]) == "a\n\nb"


def test_next_segment_leading_blank_line_not_tripled():
    j = SegmentJoiner()
    assert _run(j, [("text", "a"), ("tool",), ("text", "\n\nb")]) == "a\n\nb"


def test_leading_tool_before_any_text_no_separator():
    j = SegmentJoiner()
    # A tool that runs before any assistant text must not produce a leading break.
    assert _run(j, [("tool",), ("text", "first words")]) == "first words"


def test_separator_spans_split_deltas():
    # The boundary's separator must land even when emit resumes across deltas.
    j = SegmentJoiner()
    assert _run(j, [("text", "end."), ("tool",), ("text", "next"), ("text", " more")]) == \
        "end.\n\nnext more"


def test_multiple_tool_boundaries():
    j = SegmentJoiner()
    result = _run(j, [
        ("text", "a"),
        ("tool",),
        ("text", "b"),
        ("tool",),
        ("text", "c"),
    ])
    assert result == "a\n\nb\n\nc"


def test_empty_feeds_ignored():
    j = SegmentJoiner()
    assert _run(j, [("text", ""), ("text", "x"), ("text", "")]) == "x"


# ── table flattening ─────────────────────────────────────────────────────────

from app.textfilter import OutputFilter, TableFlattener  # noqa: E402


def _content_lines(fenced: str) -> list[str]:
    """The lines of a single fenced block, fence markers stripped."""
    body = fenced.strip()
    assert body.startswith("```") and body.endswith("```")
    return body.splitlines()[1:-1]


def test_simple_table_becomes_fenced_ascii():
    src = "| Name | Age |\n|------|-----|\n| Alice | 30 |\n| Bob | 25 |\n"
    out = TableFlattener().feed(src) + TableFlattener().flush()
    f = TableFlattener()
    out = f.feed(src) + f.flush()
    assert out.startswith("```\n") and out.rstrip().endswith("```")
    # original GFM separator pipes are gone
    assert "|------|" not in out
    # a column-junction separator row exists
    assert "-+-" in out
    lines = _content_lines(out)
    # header + separator + 2 body rows
    assert len(lines) == 4
    assert "Name" in lines[0] and "Age" in lines[0]
    assert "Alice" in lines[2] and "Bob" in lines[3]
    # columns align: the " | " separators sit at the same offset in every data row
    data = [lines[0], lines[2], lines[3]]
    offsets = {ln.index(" | ") for ln in data}
    assert len(offsets) == 1


def test_right_alignment_from_separator():
    src = "| x | n |\n|:--|--:|\n| a | 5 |\n"
    f = TableFlattener()
    out = f.feed(src) + f.flush()
    lines = _content_lines(out)
    # right-aligned numeric column: '5' pushed to the right of its cell
    assert lines[2].rstrip().endswith("5")


def test_non_table_pipe_line_passes_through():
    # A lone pipe line with no separator row is not a table — leave it alone.
    src = "use a | b | c shell pipe here\n"
    f = TableFlattener()
    out = f.feed(src) + f.flush()
    assert out == src
    assert "```" not in out


def test_pipes_inside_code_fence_not_touched():
    src = "```\n| not | a table |\n|-----|---------|\n```\n"
    f = TableFlattener()
    out = f.feed(src) + f.flush()
    assert out == src


def test_prose_passes_through_unchanged():
    src = "Just a normal paragraph.\nSecond line, no pipes.\n"
    f = TableFlattener()
    assert f.feed(src) + f.flush() == src


def test_table_split_across_streamed_chunks():
    src = "| A | B |\n|---|---|\n| 1 | 2 |\n"
    whole = TableFlattener()
    expected = whole.feed(src) + whole.flush()
    piece = TableFlattener()
    got = "".join(piece.feed(ch) for ch in src) + piece.flush()
    assert got == expected
    assert got.startswith("```")


def test_table_with_no_trailing_newline_flushes():
    src = "| A | B |\n|---|---|\n| 1 | 2 |"  # no final newline
    f = TableFlattener()
    out = f.feed(src) + f.flush()
    assert out.startswith("```") and "1" in out and "2" in out


def test_outputfilter_combines_seam_and_table():
    flt = OutputFilter(flatten_tables=True)
    out = flt.feed("Here is the data.")
    flt.tool_boundary()
    out += flt.feed("| A | B |\n|---|---|\n| 1 | 2 |\n")
    out += flt.flush()
    assert out.startswith("Here is the data.\n\n")
    assert "```" in out and "-+-" in out


def test_outputfilter_disabled_leaves_tables_raw():
    flt = OutputFilter(flatten_tables=False)
    src = "| A | B |\n|---|---|\n| 1 | 2 |\n"
    out = flt.feed(src) + flt.flush()
    assert out == src
