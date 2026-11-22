import pytest

from feedback_evaluator import evaluator
from feedback_evaluator.evaluator import EvaluationRecord, Prompt


def make_record(**overrides) -> EvaluationRecord:
    base = dict(
        paper_id="p",
        doi="10.1000/example.2024.1",
        prompt_id="a",
        feedback_id="p-a",
        feedback="some critique text",
        originality_score=3,
        overall_paper_score=4,
        impact_score=5,
        prompt_title="A",
        model="sonnet",
        created_at="t",
    )
    base.update(overrides)
    return EvaluationRecord(**base)


# ---------------------------------------------------------------- extract


def test_extract_scores_ok():
    text = (
        "Some prose about the paper.\n\n```json\n"
        '{"originality_score": 4, "overall_paper_score": "2.5", '
        '"impact_score": 3}\n```\n'
    )
    scores, body = evaluator.extract_scores(text)
    assert scores == {
        "originality_score": 4,
        "overall_paper_score": 2.5,
        "impact_score": 3,
    }
    assert "```json" not in body
    assert body.startswith("Some prose")


def test_extract_scores_missing_block():
    scores, body = evaluator.extract_scores("just prose, no block")
    assert scores is None
    assert body == "just prose, no block"


def test_extract_scores_skips_invalid_json_blocks():
    text = (
        "noise\n```json\n{not json}\n```\nmore\n```json\n"
        '{"originality_score": 1}\n```\n'
    )
    scores, body = evaluator.extract_scores(text)
    # the valid block wins; a missing key is present as None, not absent
    assert scores == {
        "originality_score": 1,
        "overall_paper_score": None,
        "impact_score": None,
    }
    # prose around the valid block remains
    assert body.strip() == "noise\n```json\n{not json}\n```\nmore"


def test_extract_scores_last_block_wins():
    text = (
        '```json\n{"originality_score": 1}\n```\n'
        'mid text\n```json\n{"overall_paper_score": 2}\n```\n'
    )
    scores, _ = evaluator.extract_scores(text)
    assert scores["originality_score"] is None
    assert scores["overall_paper_score"] == 2


# ------------------------------------------------------------------ codec


def test_coerce_score():
    assert evaluator.coerce_score(3) == 3
    assert evaluator.coerce_score(2.5) == 2.5
    assert evaluator.coerce_score("4") == 4.0
    assert evaluator.coerce_score(True) is None
    assert evaluator.coerce_score("n/a") is None
    assert evaluator.coerce_score(None) is None


# --------------------------------------------------------------- validate


def test_validate_feedback_clean():
    assert evaluator.validate_feedback(make_record()) == []


def test_validate_feedback_issues():
    record = make_record(
        feedback="   ",
        originality_score=None,
        overall_paper_score=-1,
        impact_score="oops",
    )
    kinds = {issue["kind"] for issue in evaluator.validate_feedback(record)}
    assert kinds == {"empty", "missing-score", "score-range", "bad-score"}


# ---------------------------------------------------------------- catalog


def test_load_catalog_yaml(tmp_path):
    path = tmp_path / "catalog.yaml"
    path.write_text(
        "- id: a\n  title: A\n  prompt: Do A.\n" "- Bare prompt line\n"
    )
    prompts = evaluator.load_catalog(path)
    assert prompts == [
        Prompt("a", "A", "Do A."),
        # bare string gets a position-based auto id (2nd entry -> P02)
        Prompt("P02", "Bare prompt line", "Bare prompt line"),
    ]


def test_load_catalog_single_mapping(tmp_path):
    path = tmp_path / "one.yaml"
    path.write_text("id: a\nprompt: Do A.\n")
    assert evaluator.load_catalog(path) == [Prompt("a", "Do A.", "Do A.")]


def test_load_catalog_errors(tmp_path):
    missing = tmp_path / "missing.yaml"
    missing.write_text("- {id: x, title: no prompt field here}\n")
    with pytest.raises(ValueError, match="'prompt'"):
        evaluator.load_catalog(missing)

    dup = tmp_path / "dup.yaml"
    dup.write_text("- {id: x, prompt: one}\n- {id: x, prompt: two}\n")
    with pytest.raises(ValueError, match="duplicate"):
        evaluator.load_catalog(dup)

    not_list = tmp_path / "scalar.yaml"
    not_list.write_text("just a bare string, not a list\n")
    with pytest.raises(ValueError, match="YAML list"):
        evaluator.load_catalog(not_list)

    with pytest.raises(FileNotFoundError):
        evaluator.load_catalog(tmp_path / "absent.yaml")


# ------------------------------------------------------------------- misc


def test_discover_papers(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    papers = evaluator.discover_papers(tmp_path)
    assert [p.name for p in papers] == ["a.pdf"]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        evaluator.discover_papers(empty)
    with pytest.raises(FileNotFoundError):
        evaluator.discover_papers(tmp_path / "missing")


def test_load_system_prompt(tmp_path):
    path = tmp_path / "system.md"
    path.write_text("  be a harsh reviewer \n")
    assert evaluator.load_system_prompt(path) == "be a harsh reviewer"
    empty = tmp_path / "empty.md"
    empty.write_text("   \n")
    with pytest.raises(ValueError):
        evaluator.load_system_prompt(empty)
    with pytest.raises(FileNotFoundError):
        evaluator.load_system_prompt(tmp_path / "absent.md")


def test_score_line():
    line = evaluator.score_line(make_record())
    assert line == (
        "> scores: originality_score=3, overall_paper_score=4, "
        "impact_score=5"
    )
    no_impact = make_record(impact_score=None)
    assert "impact_score=n/a" in evaluator.score_line(no_impact)


def test_render_feedback_file_no_doi():
    markdown = evaluator.render_feedback_file(
        "p", [Prompt("a", "A", "q")], [], set(), "sonnet", doi=None
    )
    assert "- DOI:" not in markdown
    assert "- Model: sonnet" in markdown


def test_render_feedback_file_shows_doi():
    markdown = evaluator.render_feedback_file(
        "p",
        [Prompt("a", "A", "q")],
        [],
        set(),
        "sonnet",
        doi="10.1038/s41586-023-06137-x",
    )
    assert "- DOI: 10.1038/s41586-023-06137-x" in markdown
    assert markdown.index("- DOI:") < markdown.index("- Model:")


def test_render_feedback_file():
    prompts = [
        Prompt("a", "Section A", "q a"),
        Prompt("b", "Section B", "q b"),
    ]
    records = [make_record(feedback_id="p-b", prompt_id="b")]
    markdown = evaluator.render_feedback_file(
        "p", prompts, records, {("p", "a")}, "sonnet"
    )
    assert markdown.startswith("# Feedback: p")
    assert "## [p-a] Section A" in markdown
    assert "## [p-b] Section B" in markdown
    assert evaluator.score_line(records[0]) in markdown
    assert "_Evaluation failed (re-run to retry.)_" in markdown


def test_render_feedback_file_shows_warnings():
    records = [make_record(warnings=["agent failed: subtype=x"])]
    markdown = evaluator.render_feedback_file(
        "p", [Prompt("a", "A", "q")], records, set(), "sonnet"
    )
    assert "> warning: agent failed: subtype=x" in markdown


def test_cli_eval_forwards_data_catalog_to_run(tmp_path, monkeypatch):
    """Regression: every CLI --flag for eval must reach evaluator.run as
    its named kwarg. A mismatch (e.g. data_catalog vs catalog_data) used
    to land silently in run()'s **_ grab-bag, dropping the catalogue so
    every record got doi=null / citations=null."""
    import json
    import shutil
    from pathlib import Path

    from feedback_evaluator import cli

    papers = tmp_path / "pdfs"
    papers.mkdir()
    src = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "pdfs"
        / "1410.6612v1.pdf"
    )
    shutil.copy(src, papers / "1410.6612v1.pdf")

    system = tmp_path / "system.md"
    system.write_text("You critique papers.\n")
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text("- prompt: rate the paper\n")

    data_catalog = tmp_path / "data.jsonl"
    data_catalog.write_text(
        json.dumps(
            {
                "pdf_path": str(papers / "1410.6612v1.pdf"),
                "doi": "10.48550/arXiv.1410.6612",
                "citation_count": 7,
            }
        )
        + "\n"
    )

    # stub out the agent (no SDK session) and the PDF-DOI fallback (so a
    # non-null doi proves the value came from --data-catalog, not the PDF)
    monkeypatch.setattr(
        evaluator,
        "evaluate_one",
        lambda prompt, read_ref, options: (
            "Fine work overall.\n\n```json\n"
            '{"originality_score": 3, "overall_paper_score": 4, '
            '"impact_score": 5}\n```',
            [],
        ),
    )
    monkeypatch.setattr(evaluator, "extract_doi", lambda path: "WRONG-DOI")

    rc = cli.main(
        [
            "eval",
            "--papers-dir",
            str(papers),
            "--system",
            str(system),
            "--catalog",
            str(catalog),
            "--feedback-dir",
            str(tmp_path / "feedback"),
            "--data-catalog",
            str(data_catalog),
        ]
    )
    assert rc == 0
    rows = [
        json.loads(line)
        for line in (tmp_path / "feedback" / "evaluations.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["doi"] == "10.48550/arXiv.1410.6612"
    assert rows[0]["citations"] == 7
