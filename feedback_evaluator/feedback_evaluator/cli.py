"""``feedback_evaluator`` — CLI dispatcher for the pipeline.

Stages: extract -> eval -> embed -> compare. All path options are
relative to the current directory; the Makefile drives the stages with
repo-root paths (data/pdfs, data/text, data/feedback, ...).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import compare, embedding, evaluator, extract


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="feedback_evaluator",
        description="Evaluate papers with a Claude Code agent, embed the "
        "feedback units, and compare the embeddings.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # Extract

    x = sub.add_parser(
        "extract", help="convert PDFs to text (papers/*.pdf -> text dir)"
    )
    x.add_argument("--papers-dir", type=Path, default=Path("papers"))
    x.add_argument("--text-dir", type=Path, default=Path("text"))

    # Eval

    e = sub.add_parser("eval", help="run agent evaluations on papers/")
    e.add_argument("--papers-dir", type=Path, default=Path("papers"))
    e.add_argument(
        "--system",
        type=Path,
        default=Path("prompts/system.md"),
        help="system prompt file (markdown)",
    )
    e.add_argument(
        "--catalog",
        type=Path,
        default=Path("prompts/catalog.yaml"),
        help="YAML prompt catalogue",
    )
    e.add_argument("--feedback-dir", type=Path, default=Path("feedback"))
    e.add_argument(
        "--data-catalog",
        type=Path,
        default=None,
        help="S2-enriched data catalogue JSONL (arxiv + S2 + pdf links), "
        "joined by pdf filename for doi/citations; optional",
    )
    e.add_argument(
        "--text-dir",
        type=Path,
        default=None,
        help="directory of extracted paper texts; a paper is passed as "
        "text when <text-dir>/<file>.txt exists, else as the PDF",
    )
    e.add_argument(
        "--model",
        default=None,
        help="tier alias sonnet/opus/haiku (default $CLAUDE_MODEL "
        f"or sonnet)",
    )
    e.add_argument(
        "--resume",
        action="store_true",
        help="skip pairs already in feedback/evaluations.jsonl",
    )

    # Embed

    b = sub.add_parser(
        "embed",
        help="embed each evaluation unit (evaluations.jsonl) via a server",
    )
    b.add_argument(
        "--feedback-dir",
        type=Path,
        default=Path("feedback"),
        help="directory holding evaluations.jsonl (one unit per line)",
    )
    b.add_argument("--out-dir", type=Path, default=Path("embeddings"))
    b.add_argument(
        "--embed-url",
        default=None,
        help="OpenAI-compatible base URL (default $EMBED_URL or "
        "http://localhost:11434/v1)",
    )
    b.add_argument(
        "--api-key",
        default=None,
        help="Bearer token (default $EMBED_API_KEY; empty = none)",
    )
    b.add_argument(
        "--model",
        default=None,
        help="embedding model name (default $EMBED_MODEL; "
        "empty = server default)",
    )
    b.add_argument("--batch-size", type=int, default=128)

    # Compare

    c = sub.add_parser(
        "compare", help="compare embeddings; write overlap report"
    )
    c.add_argument(
        "--embeddings-file",
        type=Path,
        default=Path("embeddings/embeddings.jsonl"),
    )
    c.add_argument("--out-dir", type=Path, default=Path("results"))
    c.add_argument(
        "--within-papers", action="store_true", help="only within-paper pairs"
    )
    return p


_DEFAULT_EMBED_URL = "http://localhost:11434/v1"


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "extract":
        return extract.run(papers_dir=args.papers_dir, text_dir=args.text_dir)

    if args.command == "eval":
        return evaluator.run(
            papers_dir=args.papers_dir,
            system=args.system,
            catalog=args.catalog,
            feedback_dir=args.feedback_dir,
            data_catalog=args.data_catalog,
            text_dir=args.text_dir,
            model=args.model,
            resume=args.resume,
        )

    if args.command == "embed":
        return embedding.run(
            feedback_dir=args.feedback_dir,
            out_dir=args.out_dir,
            embed_url=args.embed_url
            or _env("EMBED_URL")
            or _DEFAULT_EMBED_URL,
            model=args.model or _env("EMBED_MODEL"),
            api_key=args.api_key or _env("EMBED_API_KEY"),
            batch_size=args.batch_size,
        )

    return compare.run(
        embeddings_file=args.embeddings_file,
        out_dir=args.out_dir,
        within_papers=args.within_papers,
    )
