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
    # per-paper coherence (units, not lines) + the per-paper impact view
    p1, p2 = report["per_paper"]["p1"], report["per_paper"]["p2"]
    assert p1["units"] == 2 and p1["intra_paper_mean"] > 0.99
    assert p1["avg_llm_impact"] == 3.75  # mean(4.0, 3.5) over the prompts
    assert p1["citations"] == 100
    assert p2 == {"units": 1, "avg_llm_impact": 2.0, "citations": 40}
    # unit-level view for the per-prompt chart
    units = {u["id"]: u for u in report["units"]}
    assert units["p1#P1"] == {
        "id": "p1#P1",
        "paper": "p1",
        "prompt_id": "P1",
        "impact_score": 4.0,
    }
    assert units["p2#P1"]["impact_score"] == 2.0
    # normalization is L2: stats stay within [-1, 1]
    assert (
        -1.0
        <= report["cosine_stats"]["min"]
        <= report["cosine_stats"]["max"]
        <= 1.0
    )


def test_cosine_split_within_vs_between_paper():
    # p1 has two near-identical units; p2 is orthogonal to both. So the one
    # same-paper pair (p1) is high-cosine and the two cross-paper pairs are
    # cosine 0: the within/between split is well defined.
    records = [
        _rec("p1#P1", "p1", [1.0, 0.0, 0.0], "10.1/p1", "P1", 4.0, 100),
        _rec("p1#P2", "p1", [1.0, 0.0, 0.1], "10.1/p1", "P2", 3.5, 100),
        _rec("p2#P1", "p2", [0.0, 1.0, 0.0], None, "P1", 2.0, 40),
    ]
    report = compare.build_report(records)
    split = report["cosine_split"]
    w = split["within_paper"]
    assert w["max"] > 0.99  # the two p1 units, near-identical
    b = split["between_paper"]
    assert b["max"] == 0.0 and b["min"] == 0.0  # p1 vs p2 are orthogonal
    # within mean / between mean is undefined as a mean ratio when between is
    # all zeros, so it is omitted
    assert "mean_within_vs_between" not in split


def test_cosine_split_ratio_when_between_nonzero():
    # p1 near-identical within; p2 offset but nonzero overlap with p1, so the
    # between mean > 0 and the ratio is present.
    records = [
        _rec("p1#P1", "p1", [1.0, 0.0, 0.0]),
        _rec("p1#P2", "p1", [1.0, 0.0, 0.1]),
        _rec("p2#P1", "p2", [0.7, 0.7, 0.0]),
    ]
    report = compare.build_report(records)
    split = report["cosine_split"]
    assert split["mean_within_vs_between"] > 1.0


def test_cosine_split_within_papers_only_has_no_between():
    records = [
        _rec("p1#P1", "p1", [1.0, 0.0, 0.0]),
        _rec("p1#P2", "p1", [1.0, 0.0, 0.1]),
        _rec("p2#P1", "p2", [0.0, 1.0, 0.0]),
    ]
    report = compare.build_report(records, within_papers=True)
    split = report["cosine_split"]
    assert "within_paper" in split
    assert "between_paper" not in split
    assert "mean_within_vs_between" not in split


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


def test_avg_impact_by_paper():
    # _rec(id, paper, vec, doi, prompt_id, impact, citations)
    records = [
        _rec("p1#P1", "p1", [1.0], "10.1", "P1", 4.0, 1),
        _rec("p1#P2", "p1", [1.0], "10.1", "P2", 6.0, 1),
        _rec("p2#P1", "p2", [1.0], None, "P1", 2.0, 1),
        # a unit with no score is ignored, not a 0
        _rec("p2#P2", "p2", [1.0], None, "P2", None, 1),
    ]
    got = compare.avg_impact_by_paper(records)
    assert got["p1"] == 5.0  # mean(4, 6)
    assert got["p2"] == 2.0  # only the scored unit counts


def test_paper_pairs_average_over_prompts():
    # p1's two prompts (impact 4, 6) average to 5.0; p2's (2, 3) to 2.5.
    # The vector pairs are orthogonal, so mean cross cos is 0.
    records = [
        _rec("p1#P1", "p1", [1.0, 0.0], "10.1/p1", "P1", 4.0, 100),
        _rec("p1#P2", "p1", [1.0, 0.0], "10.1/p1", "P2", 6.0, 100),
        _rec("p2#P1", "p2", [0.0, 1.0], None, "P1", 2.0, 40),
        _rec("p2#P2", "p2", [0.0, 1.0], None, "P2", 3.0, 40),
    ]
    report = compare.build_report(records)
    # one paper-pair row (p1, p2), sorted desc by mean_cosine
    row = report["paper_pairs"][0]
    assert len(report["paper_pairs"]) == 1
    assert row["paper_a"] == "p1" and row["paper_b"] == "p2"
    assert row["avg_llm_impact_a"] == 5.0
    assert row["avg_llm_impact_b"] == 2.5
    assert row["avg_x_llm_impact"] == 2.5
    # human side: citations are paper-level constants (no averaging),
    # x is the abs difference
    assert row["human_impact_a"] == 100  # p1's citation count
    assert row["human_impact_b"] == 40  # p2's citation count
    assert row["x_human_impact"] == 60
    assert row["mean_cosine"] == 0.0
    assert row["total_human_impact"] == 140
    assert row["doi_a"] == "10.1/p1" and row["doi_b"] is None
    # raw unit rows still present, untouched
    assert report["n_pairs"] == 6


def test_per_paper_uses_unscored_units_no_fake_impact():
    # an unscored unit counts towards `units` but not the impact average
    records = [
        _rec("p1#P1", "p1", [1.0], "10.1", "P1", 4.0, 100),
        _rec("p1#P2", "p1", [1.0], "10.1", "P2", None, 100),
    ]
    report = compare.build_report(records)
    stats = report["per_paper"]["p1"]
    assert stats["units"] == 2
    assert stats["avg_llm_impact"] == 4.0  # only the scored prompt counts
    assert stats["citations"] == 100
    assert stats["intra_paper_mean"] == 1.0  # identical vectors


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
