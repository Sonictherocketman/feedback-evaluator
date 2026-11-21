"""``feedback_evaluator eval`` — critique papers with Claude Code agent sessions.

Each (paper, prompt) pair is evaluated by a full Claude Code agent session
driven by the Claude Agent SDK. The system prompt comes from
``prompts/system.md`` and the question prompts from the YAML catalogue
``prompts/catalog.yaml``. The agent reads the paper itself (PDFs included)
and ends its reply with a fenced ```json block of three scores.

Outputs: ``feedback/<paper>.md`` (one ``## [feedback_id]`` section per
prompt) and ``results/evaluations.jsonl`` (one record per completed pair,
keyed by paper_id / prompt_id / feedback_id, with the three scores).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

import anyio
import claude_agent_sdk
import yaml
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
)

from .common import (
    discover_papers,
    dump_jsonl,
    iter_jsonl,
    utc_now_iso,
    write_text_atomic,
)
from .doi import extract_doi

DEFAULT_MODEL = "sonnet"

#: The three scored dimensions, in output order.
SCORE_KEYS = ("originality_score", "overall_paper_score", "impact_score")
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass(frozen=True)
class Prompt:
    """One question from the catalogue."""

    id: str
    title: str
    prompt: str


@dataclass
class EvaluationRecord:
    """One (paper, prompt) result, as stored in evaluations.jsonl."""

    paper_id: str
    doi: str | None
    prompt_id: str
    feedback_id: str
    feedback: str
    originality_score: float | int | None
    overall_paper_score: float | int | None
    impact_score: float | int | None
    prompt_title: str | None
    model: str
    created_at: str
    warnings: list[str] = field(default_factory=list)
    format_issues: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def load_system_prompt(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"system prompt file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"system prompt file is empty: {path}")
    return text.strip()


def load_catalog(path: Path) -> list[Prompt]:
    """YAML catalogue -> prompts. Entries are {id, title, prompt} mappings
    or bare strings (auto-ID'd by 1-based position: first -> P01). Ids must
    be unique."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"prompt catalogue not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        raise ValueError(f"catalogue is empty: {path}")
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"catalogue must be a YAML list: {path}")

    prompts: list[Prompt] = []
    for position, item in enumerate(data, start=1):
        if isinstance(item, (str, Mapping)):
            prompt = _prompt_from_entry(item, position)
            if prompt is not None:
                prompts.append(prompt)
        else:
            raise ValueError(
                f"catalogue entry {position} must be a mapping "
                "with a string 'prompt' key, or a string"
            )
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    _require_unique_ids(prompts, path)
    return prompts


def _prompt_from_entry(item, position: int) -> Prompt | None:
    default_id = f"P{position:02d}"
    if isinstance(item, str):
        text = item.strip()
        return Prompt(default_id, text[:60], text) if text else None
    if not isinstance(item.get("prompt"), str):
        raise ValueError(
            f"catalogue entry {position} is missing a string "
            f"'prompt' field"
        )
    return Prompt(
        id=str(item.get("id") or default_id),
        title=str(item.get("title") or item["prompt"][:60]),
        prompt=item["prompt"],
    )


def _require_unique_ids(prompts: list[Prompt], path: Path) -> None:
    seen, duplicates = set(), set()
    for prompt in prompts:
        if prompt.id in seen:
            duplicates.add(prompt.id)
        seen.add(prompt.id)
    if duplicates:
        raise ValueError(
            f"duplicate prompt ids in {path}: {sorted(duplicates)}"
        )


def _is_number(value) -> bool:
    """True for int/float but not bool (the usual Python gotcha)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def coerce_score(value) -> float | int | None:
    """Best-effort numeric coercion; unusable values become None."""
    if _is_number(value):
        return value
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def extract_scores(text: str) -> tuple[dict | None, str]:
    """Pull the last fenced ```json scores block out of agent text.

    Returns (scores, body_without_block). The scores dict always carries
    all of SCORE_KEYS (missing values are None); None overall means no
    valid scores block was found.
    """
    for match in reversed(list(JSON_BLOCK_RE.finditer(text))):
        try:
            candidate = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and any(
            key in candidate for key in SCORE_KEYS
        ):
            scores = {
                key: coerce_score(candidate.get(key)) for key in SCORE_KEYS
            }
            body = (text[: match.start()] + text[match.end() :]).strip()
            return scores, body
    return None, text.strip()


def _extract_scores_dict(scores: dict | None) -> dict:
    """Unpack (scores|None) into keyword args for EvaluationRecord."""
    return {key: (scores or {}).get(key) for key in SCORE_KEYS}


def validate_feedback(record: EvaluationRecord) -> list[dict]:
    """HOOK: check a result record against the requested format.

    Returns [{"kind", "detail"}] issues; empty means acceptable. The
    pipeline logs and stores issues but keeps the record, so a stricter
    validator can replace this function without touching the writer.
    Extend the checks below as the feedback spec grows.
    """
    issues: list[dict] = []
    if not record.feedback.strip():
        issues.append({"kind": "empty", "detail": "feedback text is empty"})
    for key in SCORE_KEYS:
        value = getattr(record, key)
        if value is None:
            issues.append(
                {
                    "kind": "missing-score",
                    "detail": f"{key} is missing "
                    f"(no parseable scores block)",
                }
            )
        elif not _is_number(value):
            issues.append(
                {
                    "kind": "bad-score",
                    "detail": f"{key} not numeric: {value!r}",
                }
            )
    for key in ("originality_score", "overall_paper_score"):
        value = getattr(record, key)
        if _is_number(value) and not 0 <= value <= 5:
            issues.append(
                {"kind": "score-range", "detail": f"{key}={value} outside 0-5"}
            )
    return issues


def build_agent_options(
    *, cwd: str, model: str, system_prompt: str
) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        cwd=cwd,
        model=model,
        system_prompt=system_prompt,
        max_turns=40,
        permission_mode="dontAsk",
        allowed_tools=["Read", "Grep", "Glob", "Bash"],
        # The SDK reads the CLI's NDJSON stdout with a 1 MB per-message
        # buffer by default; a large paper read back as one tool result
        # exceeds it and raises CLIJSONDecodeError. Papers go to ~13 MB.
        max_buffer_size=64 * 1024 * 1024,
        # The CLI logs "unrecognized_model" for proxy model ids even when
        # routing works; drop just those lines, pass the rest through.
        stderr=_forward_stderr,
        env={"IS_SANDBOX": "1"},
    )


def _forward_stderr(line: str) -> None:
    if "unrecognized_model" not in line:
        print(f"[claude] {line}", file=sys.stderr)


def evaluate_one(
    prompt_text: str, read_ref: str, options: ClaudeAgentOptions
) -> tuple[str, list[str]]:
    """Run one agent session for one (paper, prompt) pair.

    ``read_ref`` is the paper file (text extraction preferred, else the
    PDF) as a path relative to the agent's working directory.

    Returns (feedback_text, warnings); raises if the agent produced
    nothing at all.
    """
    task = (
        f"Read the academic paper at the relative path `{read_ref}` "
        "(your working directory is the project root) and provide your "
        "evaluation of it. It is plain text extracted from the PDF, so "
        "figures are absent and some math may be approximate; evaluate "
        "what is there.\n\n"
        f"Question to answer:\n{prompt_text}\n\n"
        "You are running non-interactively as part of a batch: never ask "
        "for confirmation, never offer to re-run anything, and answer "
        "the question directly. End your response with the required "
        "fenced ```json scores block."
    )

    text_chunks: list[str] = []
    failed = False
    subtype: str = ""
    errors: list[str] = []

    async def run() -> None:
        async for message in claude_agent_sdk.query(
            prompt=task, options=options
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock) and block.text.strip():
                        text_chunks.append(block.text)
            elif isinstance(message, ResultMessage):
                nonlocal failed, subtype, errors
                failed = bool(message.is_error)
                if message.subtype:
                    subtype = message.subtype
                if message.errors:
                    errors = list(message.errors)

    anyio.run(run)
    warnings: list[str] = []
    if failed:
        warnings.append(f"agent failed: subtype={subtype}; errors={errors}")
    text = "\n\n".join(text_chunks).strip()
    if not text:
        raise RuntimeError("agent produced no output " f"(subtype={subtype})")
    return text, warnings


def score_line(record: EvaluationRecord) -> str:
    """The ``> scores: ...`` meta line for one record."""
    values = ", ".join(
        f"{key}={'n/a' if getattr(record, key) is None else getattr(record, key)}"
        for key in SCORE_KEYS
    )
    return f"> scores: {values}"


def _feedback_meta_header(paper_id: str, doi: str | None, model: str) -> str:
    """The ``- Model:``/``- DOI:`` meta block under the title (skipped by
    the line parser)."""
    lines = [f"# Feedback: {paper_id}", ""]
    if doi:
        lines += [f"- DOI: {doi}", "", f"- Model: {model}", ""]
    else:
        lines.append(f"- Model: {model}")
        lines.append("")
    return "\n".join(lines)


def render_feedback_file(
    paper_id: str,
    prompts: list[Prompt],
    records: list[EvaluationRecord],
    failed_pairs: set[tuple[str, str]],
    model: str,
    doi: str | None = None,
) -> str:
    """Markdown for one paper's feedback file: one section per prompt,
    body from this run's records; failed pairs get a retry note."""
    by_feedback_id = {
        record.feedback_id: record
        for record in records
        if record.paper_id == paper_id
    }
    lines = _feedback_meta_header(paper_id, doi, model).split("\n")
    for prompt in prompts:
        feedback_id = f"{paper_id}-{prompt.id}"
        lines += [f"## [{feedback_id}] {prompt.title}", ""]
        record = by_feedback_id.get(feedback_id)
        if record is not None:
            lines.extend(f"> warning: {w}" for w in record.warnings)
            lines.append("")
            lines.append(record.feedback)
            lines.append("")
            lines.append(score_line(record))
        elif (paper_id, prompt.id) in failed_pairs:
            lines.append("_Evaluation failed (re-run to retry.)_")
        lines.append("")
    return "\n".join(lines)


def run(
    *,
    papers_dir,
    system,
    catalog,
    feedback_dir,
    model,
    resume: bool = False,
    cwd: str = ".",
    out=print,
    **_,
) -> int:
    """Execute the evaluation. Returns 0 on success, 1 if any pair failed."""
    model = model or os.environ.get("CLAUDE_MODEL") or DEFAULT_MODEL
    papers = discover_papers(papers_dir)
    prompts = load_catalog(catalog)
    system_prompt = load_system_prompt(system)

    feedback_dir = Path(feedback_dir)
    feedback_dir.mkdir(parents=True, exist_ok=True)
    results_file = feedback_dir / "evaluations.jsonl"

    records: dict[str, EvaluationRecord] = {
        rec.feedback_id: rec
        for rec in (
            EvaluationRecord(
                **{
                    k: v
                    for k, v in stored.items()
                    if k in EvaluationRecord.__dataclass_fields__
                }
            )
            for stored in iter_jsonl(results_file)
        )
    }
    failed_pairs: set[tuple[str, str]] = set()

    options = build_agent_options(
        cwd=cwd, model=model, system_prompt=system_prompt
    )
    out(
        f"Evaluating {len(papers)} x {len(prompts)} "
        f"= {len(papers) * len(prompts)} agent run(s) with {model}"
    )

    for paper in papers:
        paper_id = paper.stem
        doi = extract_doi(paper)
        # pass the model the text extraction if it exists, else the PDF
        read_ref = (
            f"papers/text/{paper_id}.txt"
            if (papers_dir / "text" / f"{paper_id}.txt").is_file()
            else f"papers/{paper.name}"
        )
        out(
            f"== {paper.name} -> {paper_id}.md"
            + (f"  doi={doi}" if doi else "")
            + ("" if read_ref.endswith(".pdf") else "  (text)")
        )
        for prompt in prompts:
            feedback_id = f"{paper_id}-{prompt.id}"
            if resume and feedback_id in records:
                out(f"   [skip] {feedback_id}")
                continue
            started = time.monotonic()
            try:
                text, warnings = evaluate_one(prompt.prompt, read_ref, options)
            except Exception as exc:
                failed_pairs.add((paper_id, prompt.id))
                out(f"   [FAIL] {feedback_id}: {exc}")
                continue

            scores, body = extract_scores(text)
            scores_dict = _extract_scores_dict(scores)
            new_record = EvaluationRecord(
                paper_id=paper_id,
                doi=doi,
                prompt_id=prompt.id,
                feedback_id=feedback_id,
                feedback=body,
                **scores_dict,
                prompt_title=prompt.title,
                model=model,
                created_at=utc_now_iso(),
                warnings=warnings,
            )
            new_record.format_issues = validate_feedback(new_record)
            for issue in new_record.format_issues:
                out(
                    f"   [warn] {feedback_id}: {issue['kind']}: "
                    f"{issue['detail']}"
                )

            records[feedback_id] = new_record
            out(
                f"   [ok  ] {feedback_id}: {len(body)} chars in "
                f"{time.monotonic() - started:.0f}s, scores={scores}"
            )

        write_text_atomic(
            feedback_dir / f"{paper_id}.md",
            render_feedback_file(
                paper_id,
                prompts,
                records.values(),
                failed_pairs,
                model,
                doi=doi,
            ),
        )

    write_text_atomic(
        results_file, dump_jsonl(r.to_dict() for r in records.values())
    )
    out(
        f"\nDone. Failed: {len(failed_pairs)}; {len(records)} records in "
        f"{results_file}"
    )
    return 1 if failed_pairs else 0
