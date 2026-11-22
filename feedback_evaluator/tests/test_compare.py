import json

import pytest

from feedback_evaluator import compare


def _rec(
    id, paper, vec, doi=None, prompt_id=None, impact=None, citations=None
):
    return {
        "id": id,
        "paper": paper,
        "doi": doi,
        "vector": vec,
        "prompt_id": prompt_id,
        "impact_score": impact,
        "citations": citations,
    }


def test_build_report_all_pairs():
    records = [
        _rec("p1#P1", "p1", [1.0, 0.0, 0.0], "10.1/p1", "P1", 4.0, 100),
        _rec("p1#P2", "p1", [1.0, 0.0, 0.1], "10.1/p1", "P2", 3.5, 100),
        _rec("p2#P1", "p2", [0.0, 1.0, 0.0], None, "P1", 2.0, 40),
    ]
    report = compare.build_report(records)
    assert report["n_pairs"] == 3
    assert report["scope"] == "all-pairs"
    # report-level noise-free: no source / n_lines / near-dups
    for key in ("source", "n_lines", "near_duplicate_pairs"):
        assert key not in report
    # most similar pair is the two p1 units (cos ~ 1/1.005)
    top = report["pairs"][0]
    assert {top["a"], top["b"]} == {"p1#P1", "p1#P2"}
    assert abs(top["cosine"] - 1 / (1 + 0.1**2) ** 0.5) < 1e-6
    assert top["same_paper"] is True
    # a same-paper pair IS a control row: it carries the impact metrics
    # (both units have impact scores; the paper's own citations double)
    assert top["total_human_impact"] == 200
    assert top["total_llm_impact"] == 7.5
    assert top["x_impact"] == 0.5
    assert top["same_prompt"] is False
    assert {top["prompt_a"], top["prompt_b"]} == {"P1", "P2"}
    assert top["doi_a"] == "10.1/p1" and top["doi_b"] == "10.1/p1"
    assert report["papers_dois"] == {"p1": "10.1/p1", "p2": None}
    # per-paper coherence (units, not lines)
    assert report["per_paper"]["p1"]["units"] == 2
    assert report["per_paper"]["p1"]["intra_paper_mean"] > 0.99
    assert report["per_paper"]["p2"] == {"units": 1}
    # normalization is L2: stats stay within [-1, 1]
    assert (
        -1.0
        <= report["cosine_stats"]["min"]
        <= report["cosine_stats"]["max"]
        <= 1.0
    )


def test_pair_impact_metrics():
    # cross-paper units with impact scores: llm/x always computed
    a = {"paper": "p1", "prompt_id": "P1", "impact_score": 4.0}
    b = {"paper": "p2", "prompt_id": "P1", "impact_score": 2.0}
    citations = {"p1": 100, "p2": 40}
    assert compare.pair_impact(a, b, citations) == {
        "total_human_impact": 140,
        "total_llm_impact": 6.0,
        "x_impact": 2.0,
        "same_prompt": True,
    }

    # different prompts: same numbers, same_prompt is False
    b2 = {"paper": "p2", "prompt_id": "P2", "impact_score": 2.0}
    assert compare.pair_impact(a, b2, citations) == {
        "total_human_impact": 140,
        "total_llm_impact": 6.0,
        "x_impact": 2.0,
        "same_prompt": False,
    }

    # missing citations on one side: the other counts, llm still computed
    assert compare.pair_impact(a, b, {"p1": 100}) == {
        "total_human_impact": 100,
        "total_llm_impact": 6.0,
        "x_impact": 2.0,
        "same_prompt": True,
    }

    # a missing impact_score leaves llm/x as None (not a fake 0)
    no_imp = {"paper": "p1", "prompt_id": "P1"}
    assert (
        compare.pair_impact(no_imp, b, citations)["total_llm_impact"] is None
    )
    assert compare.pair_impact(no_imp, b, citations)["x_impact"] is None

    # neither paper has citations nor prompts: all None / same_prompt False
    assert compare.pair_impact({"paper": "p1"}, {"paper": "p2"}, {}) == {
        "total_human_impact": None,
        "total_llm_impact": None,
        "x_impact": None,
        "same_prompt": False,
    }


def test_report_rows_carry_prompt_and_impact():
    # the cross-paper same-prompt treatment row
    records = [
        _rec("p1#P1", "p1", [1.0, 0.0], "10.1/p1", "P1", 4.0, 100),
        _rec("p2#P1", "p2", [0.0, 1.0], None, "P1", 2.0, 40),
    ]
    report = compare.build_report(records)
    row = report["pairs"][0]
    assert row["same_paper"] is False
    assert row["same_prompt"] is True
    assert row["prompt_a"] == row["prompt_b"] == "P1"
    assert row["total_human_impact"] == 140
    assert row["total_llm_impact"] == 6.0
    assert row["x_impact"] == 2.0


def test_build_report_within_papers_only():
    records = [
        _rec("p1#P1", "p1", [1.0, 0.0, 0.0]),
        _rec("p1#P2", "p1", [1.0, 0.0, 0.1]),
        _rec("p2#P1", "p2", [0.0, 1.0, 0.0]),
    ]
    report = compare.build_report(records, within_papers=True)
    assert report["scope"] == "within-papers"
    assert report["n_pairs"] == 1  # only the p1 pair
    # even within-papers, the single row keeps its control fields
    assert report["pairs"][0]["same_paper"] is True
    assert report["pairs"][0]["same_prompt"] is False


def test_build_report_errors():
    with pytest.raises(ValueError, match="at least 2"):
        compare.build_report([_rec("a", "p", [1.0])])


def test_run_writes_report(tmp_path):
    records = [
        _rec("p1#P1", "p1", [1.0, 0.0, 0.0]),
        _rec("p1#P2", "p1", [1.0, 0.0, 0.1]),
        _rec("p2#P1", "p2", [0.0, 1.0, 0.0]),
    ]
    f = tmp_path / "embeddings.jsonl"
    with f.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    rc = compare.run(embeddings_file=f, out_dir=tmp_path / "results")
    assert rc == 0
    report = json.load(open(tmp_path / "results" / "overlap_report.json"))
    assert report["n_pairs"] == 3
    assert report["pairs"][0]["a"] in {"p1#P1", "p1#P2"}
    assert "created_at" in report
    assert "source" not in report and "n_lines" not in report


def test_run_missing_file(tmp_path):
    assert (
        compare.run(
            embeddings_file=tmp_path / "nope.jsonl", out_dir=tmp_path / "r"
        )
        == 1
    )
