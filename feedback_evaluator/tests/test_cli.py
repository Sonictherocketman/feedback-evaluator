import pytest

from feedback_evaluator import cli
from feedback_evaluator.compare import build_report


def test_cli_parses_all_subcommands():
    p = cli.build_parser()
    args = p.parse_args(["eval", "--model", "opus", "--resume"])
    assert args.command == "eval" and args.model == "opus" and args.resume
    args = p.parse_args(
        [
            "embed",
            "--embed-url",
            "http://x/v1",
            "--api-key",
            "k",
            "--model",
            "m",
            "--batch-size",
            "4",
        ]
    )
    assert args.command == "embed" and args.api_key == "k"
    assert args.batch_size == 4
    args = p.parse_args(
        ["compare", "--top", "5", "--threshold", "0.9", "--within-papers"]
    )
    assert args.command == "compare" and args.top == 5
    assert args.threshold == 0.9 and args.within_papers
    with pytest.raises(SystemExit):
        p.parse_args([])  # subcommand required


def test_cli_eval_delegates(monkeypatch, tmp_path):
    """eval dispatches to evaluator.run with the right values."""
    from feedback_evaluator import evaluator

    calls = {}

    def fake_run(**kwargs):
        calls.update(kwargs)
        return 0

    monkeypatch.setattr(evaluator, "run", fake_run)
    rc = cli.main(["eval", "--papers-dir", str(tmp_path), "--model", "haiku"])
    assert rc == 0
    assert calls["papers_dir"] == tmp_path
    assert calls["model"] == "haiku"
    assert calls["system"].name == "system.md"
    assert calls["catalog"].name == "catalog.yaml"


def test_cli_embed_env_defaults(monkeypatch, tmp_path):
    from feedback_evaluator import embedding

    calls = {}

    def fake_run(**kwargs):
        calls.update(kwargs)
        return 0

    monkeypatch.setattr(embedding, "run", fake_run)
    monkeypatch.setenv("EMBED_URL", "http://envhost/v1")
    monkeypatch.setenv("EMBED_MODEL", "envmodel")
    monkeypatch.setenv("EMBED_API_KEY", "envkey")
    rc = cli.main(["embed"])
    assert rc == 0
    assert calls["embed_url"] == "http://envhost/v1"
    assert calls["model"] == "envmodel"
    assert calls["api_key"] == "envkey"


def test_cli_compare_delegates(monkeypatch, tmp_path):
    from feedback_evaluator import compare as cmp

    calls = {}

    def fake_run(**kwargs):
        calls.update(kwargs)
        return 0

    monkeypatch.setattr(cmp, "run", fake_run)
    rc = cli.main(["compare", "--top", "3"])
    assert rc == 0
    assert calls["top"] == 3
    assert calls["embeddings_file"].name == "embeddings.jsonl"

    # compare against build_report stays coherent: the report it writes is
    # what build_report returns.
    report = build_report(
        [
            {"id": "a", "paper": "p", "line": "x", "vector": [1, 0]},
            {"id": "b", "paper": "p", "line": "y", "vector": [0, 1]},
        ]
    )
    assert report["cosine_stats"]["mean"] == 0.0
