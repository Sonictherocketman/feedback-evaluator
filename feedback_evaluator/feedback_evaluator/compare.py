"""``feedback_evaluator compare`` — measure similarity across the embeddings.

Loads ``embeddings/embeddings.jsonl``, L2-normalizes the vectors with
numpy, and computes all pairwise cosine similarities. Reports overall
stats, cross-paper vs within-paper means (are the evaluations paper-
specific, or interchangeable boilerplate?), the most similar pairs,
near-duplicate pairs above a threshold, and per-paper coherence.
Writes ``results/overlap_report.json``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import iter_jsonl, utc_now_iso, write_text_atomic


def _add_cross_paper_comparison(
    report: dict, cross_scores: list[float], pair_rows: list[dict]
) -> None:
    cross = np.asarray(cross_scores)
    within = np.asarray([r["cosine"] for r in pair_rows if r["same_paper"]])

    if within.size and cross.mean() < within.mean():
        interp = "cross-paper less similar → paper-specific feedback"
    else:
        interp = "cross-paper more similar → generic/boilerplate"

    report["cross_paper_vs_within"] = {
        "cross_paper_mean": float(cross.mean()),
        "cross_paper_max": float(cross.max()),
        "within_paper_mean": float(within.mean()) if within.size else None,
        "interpretation": interp,
    }


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


def build_report(
    records: list[dict],
    *,
    top: int = 10,
    threshold: float = 0.95,
    within_papers: bool = False,
    source: str = "",
) -> dict:
    """Compute the overlap report over embedding records (each with
    'id', 'paper', 'line', 'vector'). Pure: returns a dict, writes nothing."""
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

    pair_rows: list[dict] = []
    all_scores: list[float] = []
    cross_scores: list[float] = []

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
            if not same:
                cross_scores.append(s)
            pair_rows.append(
                {
                    "a": a_rec["id"],
                    "b": b_rec["id"],
                    "paper_a": a_rec["paper"],
                    "paper_b": b_rec["paper"],
                    "doi_a": doi_by_paper.get(a_rec["paper"]),
                    "doi_b": doi_by_paper.get(b_rec["paper"]),
                    "same_paper": same,
                    "cosine": round(s, 6),
                }
            )

    if within_papers:
        for indices in idx_by_paper.values():
            if len(indices) >= 2:
                _compute_pair_scores(np.asarray(indices))
    else:
        _compute_pair_scores(np.arange(n))

    report = {
        "created_at": utc_now_iso(),
        "source": source,
        "n_lines": n,
        "papers": papers,
        "papers_dois": {k: v for k, v in doi_by_paper.items()},
        "scope": "within-papers" if within_papers else "all-pairs",
        "n_pairs": len(all_scores),
        "cosine_stats": _statistics(all_scores),
    }

    if cross_scores:
        _add_cross_paper_comparison(report, cross_scores, pair_rows)

    report["top_pairs"] = sorted(pair_rows, key=lambda r: -r["cosine"])[:top]
    near = [r for r in pair_rows if r["cosine"] >= threshold]
    report["near_duplicate_pairs"] = {
        "threshold": threshold,
        "count": len(near),
        "pairs": near,
    }

    per_paper = {}
    for paper, indices in sorted(idx_by_paper.items()):
        stats = {"lines": len(indices)}
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


def print_report(report: dict, records: list[dict]) -> None:
    by_id = {r["id"]: r for r in records}
    stats = report["cosine_stats"]
    print(f"\nCosine stats ({report['n_pairs']} pairs):")
    print(
        f"  mean={stats['mean']:.4f} median={stats['median']:.4f} "
        f"p05={stats['p05']:.4f} p95={stats['p95']:.4f} "
        f"min={stats['min']:.4f} max={stats['max']:.4f}"
    )
    cross_vs_within = report.get("cross_paper_vs_within")
    if cross_vs_within:
        print(
            f"  cross-paper mean: {cross_vs_within['cross_paper_mean']:.4f} "
            f"({cross_vs_within['interpretation']})"
        )

    top_pairs = report.get("top_pairs", [])
    if top_pairs:
        print(f"\nTop {len(top_pairs)} similar pairs:")
        for pair in top_pairs:
            print(f"  cos={pair['cosine']:.4f}  {pair['a']} <-> {pair['b']}")
            if pair["a"] in by_id:
                print(f"      A: {by_id[pair['a']]['line'][:96]}")
            if pair["b"] in by_id:
                print(f"      B: {by_id[pair['b']]['line'][:96]}")

    near_dup = report.get("near_duplicate_pairs", {})
    if near_dup.get("count"):
        print(
            f"\nNear-duplicate pairs (cos >= {near_dup['threshold']}): "
            f"{near_dup['count']}"
        )
        for pair in near_dup["pairs"][:10]:
            print(f"  {pair['cosine']:.4f}  {pair['a']} <-> {pair['b']}")
        if near_dup["count"] > 10:
            print("  ... and more (see JSON report)")

    print(f"\nPer-paper coherence ({report['scope']}):")
    for paper, stats in report["per_paper"].items():
        if "intra_paper_mean" in stats:
            print(
                f"  {paper}: mean={stats['intra_paper_mean']:.4f} "
                f"max={stats['intra_paper_max']:.4f} lines={stats['lines']}"
            )
        else:
            print(f"  {paper}: {stats['lines']} line, no pairs")


def run(
    *,
    embeddings_file,
    out_dir,
    top=10,
    threshold=0.95,
    within_papers=False,
    **kwargs,
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
        report = build_report(
            records,
            top=top,
            threshold=threshold,
            within_papers=within_papers,
            source=str(embeddings_file),
        )
    except ValueError as exc:
        print(f"error: {exc}")
        return 1
    print_report(report, records)

    out_path = Path(out_dir) / "overlap_report.json"
    write_text_atomic(
        out_path, json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"\nReport: {out_path}")
    return 0
