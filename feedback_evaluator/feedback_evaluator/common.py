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


def is_number(value) -> bool:
    """True for int/float but not bool (the usual Python gotcha)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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


def append_jsonl(path: os.PathLike | str, row: dict) -> None:
    """Append one JSON object as a single line to *path* (creating it).

    Line-buffered append so a long run can be killed and resumed: every
    completed record is durable before the next one starts.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()


def dump_json(obj, indent: int = 2) -> str:
    return json.dumps(obj, indent=indent, ensure_ascii=False) + "\n"
