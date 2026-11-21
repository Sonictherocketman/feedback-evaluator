import json

import pytest

from feedback_evaluator import compare


def _rec(id, paper, text, vec, doi=None):
    return {
        "id": id,
        "paper": paper,
        "doi": doi,
        "line": text,
        "vector": vec,
    }


RECORDS = [
    _rec("p1#L1", "p1", "the method works", [1.0, 0.0, 0.0], "10.1/p1"),
    _rec("p1#L2", "p1", "the method works again", [1.0, 0.0, 0.1]),
    _rec("p2#L1", "p2", "completely different topic", [0.0, 1.0, 0.0]),
]


def test_build_report_all_pairs():
    report = compare.build_report(RECORDS, threshold=0.99, top=2)
    assert report["n_lines"] == 3
    assert report["n_pairs"] == 3
    assert report["scope"] == "all-pairs"
    # most similar pair is the two p1 lines (cos ~ 1/1.005 > 0.99)
    top = report["top_pairs"][0]
    assert {top["a"], top["b"]} == {"p1#L1", "p1#L2"}
    assert abs(top["cosine"] - 1 / (1 + 0.1**2) ** 0.5) < 1e-6
    assert top["same_paper"] is True
    # top pair is p1<->p1: the paper's DOI (from its first record) rides
    # along on both sides; papers without a DOI are None in the map
    assert top["doi_a"] == "10.1/p1" and top["doi_b"] == "10.1/p1"
    assert report["papers_dois"] == {"p1": "10.1/p1", "p2": None}
    # near-duplicates at 0.99: just that pair
    assert report["near_duplicate_pairs"]["count"] == 1
    # cross vs within
    xw = report["cross_paper_vs_within"]
    assert xw["cross_paper_mean"] == 0.0
    assert xw["within_paper_mean"] == pytest.approx(1 / (1 + 0.01) ** 0.5)
    assert "paper-specific" in xw["interpretation"]
    # per-paper coherence
    assert report["per_paper"]["p1"]["lines"] == 2
    assert report["per_paper"]["p1"]["intra_paper_mean"] > 0.99
    assert report["per_paper"]["p2"] == {"lines": 1}
    # normalization is L2 (vectors passed in were already unit-ish but
    # 2nd is not exactly unit; stats must stay within [-1, 1])
    assert (
        -1.0
        <= report["cosine_stats"]["min"]
        <= report["cosine_stats"]["max"]
        <= 1.0
    )


def test_build_report_within_papers_only():
    report = compare.build_report(RECORDS, within_papers=True)
    assert report["scope"] == "within-papers"
    assert report["n_pairs"] == 1  # only the p1 pair
    assert report.get("cross_paper_vs_within") is None


def test_build_report_errors():
    with pytest.raises(ValueError, match="at least 2"):
        compare.build_report(RECORDS[:1])


def test_run_writes_report(tmp_path):
    f = tmp_path / "embeddings.jsonl"
    with f.open("w") as fh:
        for r in RECORDS:
            fh.write(json.dumps(r) + "\n")
    rc = compare.run(
        embeddings_file=f, out_dir=tmp_path / "results", top=1, threshold=0.99
    )
    assert rc == 0
    report = json.load(open(tmp_path / "results" / "overlap_report.json"))
    assert report["n_pairs"] == 3
    assert report["top_pairs"][0]["a"] in {"p1#L1", "p1#L2"}
    assert "created_at" in report


def test_run_missing_file(tmp_path):
    assert (
        compare.run(
            embeddings_file=tmp_path / "nope.jsonl", out_dir=tmp_path / "r"
        )
        == 1
    )
