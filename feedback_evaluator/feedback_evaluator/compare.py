"""``feedback_evaluator compare`` — measure similarity across the embeddings.

Loads ``embeddings/embeddings.jsonl``, L2-normalizes the vectors with
numpy, and computes all pairwise cosine similarities. Reports overall
stats, the full pair list (each pair carries its paper ids/DOIs, its
prompt ids plus a ``same_prompt`` flag, and the human/LLM impact
metrics), and per-paper coherence. Writes ``results/overlap_report.json``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import is_number, iter_jsonl, utc_now_iso, write_text_atomic


def _statistics(scores: list[float]) -> dict:
    s = np.asarray(scores) if scores else np.array([np.nan])
    return {
        "mean": float(s.mean()),
        "median": float(np.median(s)),
        "p05": float(np.percentile(s, 5)),
        "p95": float(np.percentile(s, 95)),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def pair_impact(a_unit: dict, b_unit: dict, citations: dict) -> dict:
    """Impact metrics for two units, always populated where possible.

    * total_human_impact: sum of the two papers' citation counts (a
      paper-level value; known when either paper has a count).
    * total_llm_impact / x_impact: sum and absolute difference of the
      two units' impact scores, whenever both are numeric. Same-paper
      rows (different prompts) are the control for prompt-variability;
      cross-paper same-prompt rows are the treatment comparison — the
      ``same_paper``/``same_prompt`` row fields let the analysis stage
      filter either way.
    * same_prompt: bool, for convenient filtering.
    """
    a_cit = citations.get(a_unit["paper"])
    b_cit = citations.get(b_unit["paper"])
    human = None
    if a_cit is not None or b_cit is not None:
        human = (a_cit or 0) + (b_cit or 0)

    a_prompt = a_unit.get("prompt_id")
    b_prompt = b_unit.get("prompt_id")
    a_imp = a_unit.get("impact_score")
    b_imp = b_unit.get("impact_score")
    llm = x = None
    if is_number(a_imp) and is_number(b_imp):
        llm = round(a_imp + b_imp, 6)
        x = round(abs(a_imp - b_imp), 6)
    return {
        "total_human_impact": human,
        "total_llm_impact": llm,
        "x_impact": x,
        "same_prompt": bool(a_prompt) and a_prompt == b_prompt,
    }


def build_report(
    records: list[dict],
    *,
    within_papers: bool = False,
) -> dict:
    """Compute the overlap report over embedding records (each with
    'id', 'paper', 'vector'). Pure: returns a dict, writes nothing."""
    if len(records) < 2:
        raise ValueError("need at least 2 embeddings to compare")

    matrix = np.asarray([r["vector"] for r in records], dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.where(norms == 0, 1.0, norms)

    idx_by_paper: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        idx_by_paper[r["paper"]].append(i)
    n = len(records)
    papers = sorted(idx_by_paper)
    doi_by_paper = {
        paper: next(
            (
                r.get("doi")
                for r in records
                if r["paper"] == paper and r.get("doi")
            ),
            None,
        )
        for paper in papers
    }

    if not within_papers and n > 10000:
        raise ValueError(
            f"all-pair mode: {n} lines too many; " "use --within-papers"
        )

    # paper-level citation count (first non-null value for the paper)
    citations_by_paper: dict[str, int | None] = {}
    for r in records:
        if r.get("citations") is not None:
            citations_by_paper.setdefault(r["paper"], r["citations"])
    for paper in papers:
        citations_by_paper.setdefault(paper, None)

    pair_rows: list[dict] = []
    all_scores: list[float] = []

    def _compute_pair_scores(indices: np.ndarray) -> None:
        sub = matrix[indices]
        triu = np.triu_indices(len(indices), k=1)
        scores = np.einsum("md,md->m", sub[triu[0]], sub[triu[1]])
        for (a, b), s in zip(zip(*triu), scores):
            i, j = int(indices[a]), int(indices[b])
            s = float(s)
            a_rec, b_rec = records[i], records[j]
            same = a_rec["paper"] == b_rec["paper"]
            all_scores.append(s)
            # every pair gets the impact metrics (same-paper rows are the
            # control for prompt-variability; the prompt fields let the
            # analysis stage filter)
            row = {
                "a": a_rec["id"],
                "b": b_rec["id"],
                "paper_a": a_rec["paper"],
                "paper_b": b_rec["paper"],
                "doi_a": doi_by_paper.get(a_rec["paper"]),
                "doi_b": doi_by_paper.get(b_rec["paper"]),
                "prompt_a": a_rec.get("prompt_id"),
                "prompt_b": b_rec.get("prompt_id"),
                "same_paper": same,
                "cosine": round(s, 6),
            }
            row.update(pair_impact(a_rec, b_rec, citations_by_paper))
            pair_rows.append(row)

    if within_papers:
        for indices in idx_by_paper.values():
            if len(indices) >= 2:
                _compute_pair_scores(np.asarray(indices))
    else:
        _compute_pair_scores(np.arange(n))

    report = {
        "created_at": utc_now_iso(),
        "papers": papers,
        "papers_dois": {k: v for k, v in doi_by_paper.items()},
        "scope": "within-papers" if within_papers else "all-pairs",
        "n_pairs": len(all_scores),
        "cosine_stats": _statistics(all_scores),
    }

    report["pairs"] = sorted(pair_rows, key=lambda r: -r["cosine"])

    per_paper = {}
    for paper, indices in sorted(idx_by_paper.items()):
        stats = {"units": len(indices)}
        if len(indices) >= 2:
            sub = matrix[np.asarray(indices)]
            d = sub @ sub.T
            upper = np.triu_indices(len(indices), k=1)
            intra = d[upper]
            stats["intra_paper_mean"] = float(intra.mean())
            stats["intra_paper_max"] = float(intra.max())
            no_diag = np.where(np.eye(len(indices), dtype=bool), -np.inf, d)
            stats["mean_closest_neighbor"] = float(
                np.nanmax(no_diag, axis=1).mean()
            )
        per_paper[paper] = stats
    report["per_paper"] = per_paper
    return report


def print_report(report: dict) -> None:
    stats = report["cosine_stats"]
    print(f"\nCosine stats ({report['n_pairs']} pairs):")
    print(
        f"  mean={stats['mean']:.4f} median={stats['median']:.4f} "
        f"p05={stats['p05']:.4f} p95={stats['p95']:.4f} "
        f"min={stats['min']:.4f} max={stats['max']:.4f}"
    )


def run(
    *,
    embeddings_file,
    out_dir,
    within_papers=False,
) -> int:
    """Compute and write the overlap report. 0 on success, 1 on failure."""
    embeddings_file = Path(embeddings_file)
    if not embeddings_file.is_file():
        print(f"error: embeddings file not found: {embeddings_file}")
        return 1
    records = list(iter_jsonl(embeddings_file))
    print(
        f"Loaded {len(records)} embeddings "
        f"({'within-paper' if within_papers else 'all-pair'} mode)"
    )
    try:
        report = build_report(records, within_papers=within_papers)
    except ValueError as exc:
        print(f"error: {exc}")
        return 1
    print_report(report)

    out_path = Path(out_dir) / "overlap_report.json"
    write_text_atomic(
        out_path, json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"\nReport: {out_path}")
    return 0
