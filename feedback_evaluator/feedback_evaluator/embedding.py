"""``feedback_evaluator embed`` — turn feedback lines into stored vectors.

Reads the markdown feedback files produced by ``eval``, embeds each prose
line (headings, blanks, ``>`` meta, and the ``- Model:`` header are
skipped) against a configurable embedding server, and stores one record
per line in ``embeddings/embeddings.jsonl`` — each record keeps its paper,
feedback section id, and 1-based line number so it aliases back to its
source — plus an ``embeddings/manifest.json`` describing the run.

The server speaks the OpenAI-compatible protocol:
``POST <base>/embeddings {"model": ..., "input": [lines...]}``
-> ``{"data": [{"index": i, "embedding": [...]}, ...]}``
Works with Ollama, vLLM, LM Studio, Groq, etc. Bearer auth is sent only
when an API key is provided.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import httpx

from .common import (
    dump_json,
    dump_jsonl,
    iter_lines_meta,
    utc_now_iso,
    write_text_atomic,
)

DEFAULT_BATCH_SIZE = 128
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 5


@dataclass(frozen=True)
class FeedbackLine:
    """One embeddable prose line and where it came from."""

    line_number: int  # 1-based position in the feedback file
    feedback_id: str | None  # enclosing ``## [id]`` section, if any
    text: str


@dataclass(frozen=True)
class EmbeddingRecord:
    """One stored vector, aliasable back to its source line."""

    id: str  # "<paper>#L<line_number>"
    paper: str  # feedback file stem == paper id
    doi: str | None  # the paper's DOI (from the feedback header, if any)
    feedback_id: str | None
    line_number: int
    line: str  # the exact text sent to the server
    model: str
    dim: int
    vector: list[float]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "paper": self.paper,
            "doi": self.doi,
            "feedback_id": self.feedback_id,
            "line_number": self.line_number,
            "line": self.line,
            "model": self.model,
            "dim": self.dim,
            "vector": self.vector,
        }


@dataclass(frozen=True)
class FileStat:
    paper: str
    lines: int
    sha256_16: str

    def to_dict(self) -> dict:
        return {
            "paper": self.paper,
            "lines": self.lines,
            "sha256_16": self.sha256_16,
        }


def iter_feedback_lines(path: Path) -> list[FeedbackLine]:
    """All embeddable prose lines of one feedback file, in order."""
    return [FeedbackLine(*meta) for meta in iter_lines_meta(path)]


def sha256_16(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def feedback_doi(path: Path) -> str | None:
    """The ``- DOI:`` value from the feedback file's header, if present."""
    for line in path.read_text(encoding="utf-8").splitlines()[:6]:
        stripped = line.strip()
        if stripped.startswith("- DOI:"):
            return stripped[len("- DOI:") :].strip() or None
    return None


def embed_batch(
    client: httpx.Client,
    url: str,
    *,
    api_key: str,
    model: str,
    texts: list[str],
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> list[list[float]]:
    """POST one batch of *texts*; retry with exponential backoff on
    429/5xx/transport errors. Returns vectors in input order."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {"input": texts}
    if model:
        payload["model"] = model

    for attempt in range(max_retries):
        try:
            resp = client.post(
                url, headers=headers, json=payload, timeout=timeout
            )
            resp.raise_for_status()
            data = resp.json()
            vectors = [None] * len(data["data"])
            for item in data["data"]:
                vectors[item["index"]] = item["embedding"]
            if len(vectors) != len(texts):
                raise KeyError(
                    f"server returned {len(vectors)} vectors for "
                    f"{len(texts)} lines"
                )
            return vectors
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            # Retryable: 429 rate-limit or 5xx.
            if code < 500 and code != 429:
                raise  # 4xx → fail fast
            if attempt >= max_retries - 1:
                raise RuntimeError(
                    f"embedding failed after {max_retries} tries: " f"{exc}"
                ) from exc
            delay = min(2**attempt, 30)
            print(
                f"  retry {attempt + 1}/{max_retries} in " f"{delay}s: {exc}",
                flush=True,
            )
            time.sleep(delay)
        except httpx.TransportError as exc:
            # Connection/timeout issues are transient: retry like 429/5xx.
            if attempt >= max_retries - 1:
                raise RuntimeError(
                    f"embedding failed after " f"{max_retries} tries: {exc}"
                ) from exc
            delay = min(2**attempt, 30)
            print(
                f"  retry {attempt + 1}/{max_retries} in " f"{delay}s: {exc}",
                flush=True,
            )
            time.sleep(delay)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"malformed embedding response: {exc}") from exc
    raise RuntimeError("unreachable")  # pragma: no cover


def embed_file(
    client: httpx.Client,
    url: str,
    *,
    api_key: str,
    model: str,
    feedback_file: Path,
    batch_size: int,
    doi: str | None = None,
) -> list[EmbeddingRecord]:
    """Embed every prose line of one feedback file."""
    paper = feedback_file.stem
    records: list[EmbeddingRecord] = []
    for lines in _chunks(iter_feedback_lines(feedback_file), batch_size):
        vectors = embed_batch(
            client,
            url,
            api_key=api_key,
            model=model,
            texts=[line.text for line in lines],
        )
        for line, vector in zip(lines, vectors):
            records.append(
                EmbeddingRecord(
                    id=f"{paper}#L{line.line_number}",
                    paper=paper,
                    doi=doi,
                    feedback_id=line.feedback_id,
                    line_number=line.line_number,
                    line=line.text,
                    model=model,
                    dim=len(vector),
                    vector=vector,
                )
            )
    return records


def _chunks(seq, size: int):
    it = iter(seq)
    while chunk := list(islice(it, size)):
        yield chunk


def run(
    *,
    feedback_dir,
    out_dir,
    embed_url,
    model: str = "",
    api_key: str = "",
    batch_size: int = DEFAULT_BATCH_SIZE,
    **_,
) -> int:
    """Embed all feedback files. Returns 0 on success, 1 on failure."""
    feedback_dir, out_dir = Path(feedback_dir), Path(out_dir)
    if not feedback_dir.is_dir():
        print(f"error: feedback directory not found: {feedback_dir}")
        return 1
    feedback_files = sorted(feedback_dir.glob("*.md"))
    if not feedback_files:
        print(f"error: no feedback files in {feedback_dir}")
        return 1

    url = embed_url.rstrip("/") + "/embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Embedding {len(feedback_files)} file(s) -> {out_dir} "
        f"(model={model or 'server default'}, batch={batch_size})"
    )

    all_records: list[EmbeddingRecord] = []
    file_stats: list[FileStat] = []
    try:
        with httpx.Client() as client:
            for file_path in feedback_files:
                records = embed_file(
                    client,
                    url,
                    api_key=api_key,
                    model=model,
                    feedback_file=file_path,
                    batch_size=batch_size,
                    doi=feedback_doi(file_path),
                )
                all_records.extend(records)
                file_stats.append(
                    FileStat(
                        paper=file_path.stem,
                        lines=len(records),
                        sha256_16=sha256_16(file_path),
                    )
                )
                print(f"  {file_path.name}: {len(records)} lines embedded")
    except (RuntimeError, httpx.HTTPError) as exc:
        print(f"error: {exc}")
        return 1

    if not all_records:
        print("error: no lines were embedded")
        return 1

    write_text_atomic(
        out_dir / "embeddings.jsonl",
        dump_jsonl(record.to_dict() for record in all_records),
    )
    manifest = {
        "created_at": utc_now_iso(),
        "server": embed_url,
        "model": model or None,
        "dim": all_records[0].dim,
        "batch_size": batch_size,
        "total_lines": len(all_records),
        "papers": sorted({r.paper for r in all_records}),
        "files": {s.paper: s.to_dict() for s in file_stats},
    }
    write_text_atomic(out_dir / "manifest.json", dump_json(manifest))
    print(
        f"\nWrote {len(all_records)} embeddings "
        f"(dim={all_records[0].dim}) to {out_dir / 'embeddings.jsonl'}"
    )
    print(f"Manifest at {out_dir / 'manifest.json'}")
    return 0
