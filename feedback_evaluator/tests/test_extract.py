import itertools
import shutil
from pathlib import Path

from feedback_evaluator import extract

# small, fast real PDF in the corpus (used as the only fixture that
# needs to be a real PDF)
REPO_PAPERS = Path(__file__).resolve().parents[2] / "papers"
SMALL_PDF = REPO_PAPERS / "15306315.pdf"


def test_merge_pages_drops_blank_and_page_top_repeats():
    # a line repeated within 6 lines of the previous page's end is a
    # running header/footer: dropped
    page1 = "\n".join(["H", "first body line", "second body line"]) + "\n"
    page2 = "\n".join(["H", "third body line"]) + "\n"
    merged = extract.merge_pages([page1, page2])
    assert merged.splitlines() == [
        "H",
        "first body line",
        "second body line",
        "third body line",
    ]


def test_merge_pages_keeps_repeats_far_apart():
    # the repeat window is the last 6 lines of the *previous* page, so a
    # line that also appeared more than 6 lines earlier is content: kept
    page1 = "\n".join(["same", "b", "c", "d", "e", "f", "g", "h"]) + "\n"
    page2 = "\n".join(["same", "i"]) + "\n"
    assert extract.merge_pages([page1, page2]) == "\n".join(
        ["same", "b", "c", "d", "e", "f", "g", "h", "same", "i"]
    )


def test_run_extracts_and_skips(tmp_path):
    papers = tmp_path / "papers"
    papers.mkdir()
    shutil.copy(SMALL_PDF, papers / "15306315.pdf")

    rc = extract.run(papers_dir=papers)
    assert rc == 0
    out = papers / "text" / "15306315.txt"
    assert out.is_file() and out.stat().st_size > 1000

    # second run: nothing to do
    rc = extract.run(papers_dir=papers)
    assert rc == 0
    assert out.is_file()  # untouched


def test_run_salvages_pages_that_fail(tmp_path, monkeypatch):
    import pypdf

    calls = itertools.count()

    real = pypdf.PageObject.extract_text

    def flaky_extract(self, *args, **kwargs):
        if next(calls) % 7 == 0:  # every 7th page dies
            raise RuntimeError("simulated LimitReachedError")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pypdf.PageObject, "extract_text", flaky_extract)
    papers = tmp_path / "papers"
    papers.mkdir()
    shutil.copy(SMALL_PDF, papers / "15306315.pdf")

    lines: list[str] = []
    assert extract.run(papers_dir=papers, out=lines.append) == 0
    assert any("dropped" in line for line in lines)
    assert (papers / "text" / "15306315.txt").stat().st_size > 1000


def test_run_fails_on_bad_pdf(tmp_path):
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "broken.pdf").write_bytes(b"not a pdf")
    assert extract.run(papers_dir=papers) == 1


def test_run_dirs(tmp_path):
    assert extract.run(papers_dir=tmp_path / "missing") == 1
    empty = tmp_path / "empty"
    empty.mkdir()
    assert extract.run(papers_dir=empty) == 1
