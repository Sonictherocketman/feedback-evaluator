"""``feedback_evaluator`` — CLI dispatcher for the eval / embed / compare pipeline.

All path options are relative to the current directory, so run from the
repository root where papers/, prompts/, feedback/, embeddings/ and
results/ live (see README).
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
        "feedback lines, and compare the embeddings.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # Extract

    x = sub.add_parser(
        "extract", help="convert papers/*.pdf to papers/text/*.txt"
    )
    x.add_argument("--papers-dir", type=Path, default=Path("papers"))

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
        "embed", help="embed feedback lines via an embedding server"
    )
    b.add_argument("--feedback-dir", type=Path, default=Path("feedback"))
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
    c.add_argument("--top", type=int, default=10)
    c.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="near-duplicate cosine threshold",
    )
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
        return extract.run(papers_dir=args.papers_dir)

    if args.command == "eval":
        return evaluator.run(
            papers_dir=args.papers_dir,
            system=args.system,
            catalog=args.catalog,
            feedback_dir=args.feedback_dir,
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
        top=args.top,
        threshold=args.threshold,
        within_papers=args.within_papers,
    )
