"""Shared primitives: paper discovery, JSONL I/O, atomic writes, time."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

#: File extensions recognised as papers.
PAPER_SUFFIXES = frozenset({".pdf"})


def utc_now_iso() -> str:
    """Current time as an ISO-8601 UTC string (seconds precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def discover_papers(directory: os.PathLike | str) -> list[Path]:
    """Sorted paper files in *directory*; raises if none found."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"papers directory not found: {directory}")
    papers = sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in PAPER_SUFFIXES
    )
    if not papers:
        raise FileNotFoundError(f"no papers found in {directory}")
    return papers


def iter_jsonl(path: os.PathLike | str):
    """Yield parsed JSON objects from a JSONL file; silent if missing."""
    path = Path(path)
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl(path: os.PathLike | str) -> list[dict]:
    return list(iter_jsonl(path))


def write_text_atomic(path: os.PathLike | str, text: str) -> None:
    """Write *text* as UTF-8, atomically (tmp file + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def dump_jsonl(rows):
    """Serialise *rows* to JSONL text (one compact object per line)."""
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def dump_json(obj, indent: int = 2) -> str:
    return json.dumps(obj, indent=indent, ensure_ascii=False) + "\n"


def iter_lines_meta(path: os.PathLike | str):
    """Yield (line_number, feedback_id, text) for the *prose* lines of a
    feedback file.

    Skips markdown headings, blank lines, ``>`` blockquote meta lines
    (e.g. ``> scores: ...``), and the ``- Model:`` header bullet. Only
    content in a section body (its bullets included) is yielded. Line
    numbers are 1-based positions in the file; feedback_id is the id in
    the most recent ``## [id] ...`` section heading (or None).
    """
    path = Path(path)
    current_feedback_id: str | None = None
    in_section = False
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            stripped = line.strip()
            if stripped.startswith("## [") and "]" in stripped:
                current_feedback_id = (
                    stripped[4:].split("]", 1)[0].strip() or None
                )
                in_section = True
            elif stripped.startswith("#"):
                in_section = False
            elif stripped.startswith(">") or stripped.startswith(
                ("- Model:", "- DOI:")
            ):
                continue
            elif stripped and in_section:
                yield line_number, current_feedback_id, line.rstrip("\n")
