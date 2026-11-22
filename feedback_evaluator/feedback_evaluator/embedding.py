"""``feedback_evaluator embed`` — turn evaluations into vectors.

Reads ``feedback/evaluations.jsonl`` (the per-prompt records `eval`
writes as it goes — one record per (paper, prompt), the unit of
evaluation) and embeds each record's feedback text against a
configurable embedding server. The human-readable ``feedback/*.md``
files are rendered for reference only; embed never reads them.

One record per evaluation is stored in
``embeddings/embeddings.jsonl``; each keeps its paper, feedback_id,
scores and citations, so every vector aliases back to its source —
plus an ``embeddings/manifest.json`` run summary.

The server speaks the OpenAI-compatible protocol:
``POST <base>/embeddings {"model": ..., "input": [texts...]}``
-> ``{"data": [{"index": i, "embedding": [...]}, ...]}``
Works with Ollama, vLLM, LM Studio, Groq, etc. Bearer auth is sent only
when an API key is provided.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

import httpx

from .common import (
    dump_json,
    dump_jsonl,
    iter_jsonl,
    utc_now_iso,
    write_text_atomic,
)

DEFAULT_BATCH_SIZE = 64
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 5


@dataclass(frozen=True)
class EmbeddingRecord:
    """One stored vector, aliasable back to its source evaluation.

    Carries the unit's scores and the paper's citations so
    ``compare`` can compute per-pair impact metrics without second
    lookups.
    """

    id: str  # "<paper_id>#<feedback_id>"
    paper: str  # the paper_id (paper file stem)
    feedback_id: str
    prompt_id: str | None  # the catalogue prompt this unit answered
    doi: str | None
    text: str  # the feedback body sent to the server
    originality_score: float | int | None
    overall_paper_score: float | int | None
    impact_score: float | int | None
    citations: int | None
    model: str
    dim: int
    vector: list[float]


def unit_records(path: Path) -> list[dict]:
    """The embeddable evaluation rows, in file order.

    First occurrence wins if a feedback_id repeats (the file is
    append-only, so duplicates would only come from manual edits).
    """
    rows: list[dict] = []
    seen: set[str] = set()
    for row in iter_jsonl(path):
        feedback_id = row.get("feedback_id")
        if feedback_id and feedback_id not in seen:
            seen.add(feedback_id)
            rows.append(row)
    return rows


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
                    f"{len(texts)} texts"
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
) -> int:
    """Embed every evaluation unit. Returns 0 on success, 1 on failure."""
    feedback_dir, out_dir = Path(feedback_dir), Path(out_dir)
    evaluations_file = feedback_dir / "evaluations.jsonl"
    if not evaluations_file.is_file():
        print(f"error: no evaluations at {evaluations_file}")
        return 1

    url = embed_url.rstrip("/") + "/embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = unit_records(evaluations_file)
    print(
        f"Embedding {len(rows)} unit(s) -> {out_dir} "
        f"(model={model or 'server default'}, batch={batch_size})"
    )

    all_records: list[EmbeddingRecord] = []
    try:
        with httpx.Client() as client:
            for chunk in _chunks(rows, batch_size):
                vectors = embed_batch(
                    client,
                    url,
                    api_key=api_key,
                    model=model,
                    texts=[row["feedback"] for row in chunk],
                )
                for row, vector in zip(chunk, vectors):
                    all_records.append(
                        EmbeddingRecord(
                            id=f"{row['paper_id']}#{row['feedback_id']}",
                            paper=row["paper_id"],
                            feedback_id=row["feedback_id"],
                            prompt_id=row.get("prompt_id"),
                            doi=row.get("doi"),
                            text=row["feedback"],
                            originality_score=row.get("originality_score"),
                            overall_paper_score=row.get("overall_paper_score"),
                            impact_score=row.get("impact_score"),
                            citations=row.get("citations"),
                            model=model,
                            dim=len(vector),
                            vector=vector,
                        )
                    )
    except (RuntimeError, httpx.HTTPError) as exc:
        print(f"error: {exc}")
        return 1

    if not all_records:
        print("error: no units were embedded")
        return 1

    units_per_paper: dict[str, int] = {}
    for record in all_records:
        units_per_paper[record.paper] = (
            units_per_paper.get(record.paper, 0) + 1
        )
    write_text_atomic(
        out_dir / "embeddings.jsonl",
        dump_jsonl(asdict(record) for record in all_records),
    )
    manifest = {
        "created_at": utc_now_iso(),
        "server": embed_url,
        "model": model or None,
        "dim": all_records[0].dim,
        "batch_size": batch_size,
        "total_units": len(all_records),
        "papers": sorted(units_per_paper),
        "units_per_paper": units_per_paper,
    }
    write_text_atomic(out_dir / "manifest.json", dump_json(manifest))
    print(
        f"\nWrote {len(all_records)} embeddings "
        f"(dim={all_records[0].dim}) to {out_dir / 'embeddings.jsonl'}"
    )
    print(f"Manifest at {out_dir / 'manifest.json'}")
    return 0
