import itertools
import shutil
from pathlib import Path

from feedback_evaluator import extract

# small, fast real PDF in the corpus (used as the only fixture that
# needs to be a real PDF)
PAPERS_DIR = Path(__file__).resolve().parents[2] / "data" / "pdfs"
SMALL_PDF = PAPERS_DIR / "1410.6612v1.pdf"


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
    text = tmp_path / "text"
    text.mkdir()
    shutil.copy(SMALL_PDF, papers / SMALL_PDF.name)

    rc = extract.run(papers_dir=papers, text_dir=text)
    assert rc == 0
    out = text / f"{SMALL_PDF.stem}.txt"
    assert out.is_file() and out.stat().st_size > 1000

    # second run: nothing to do
    rc = extract.run(papers_dir=papers, text_dir=text)
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
    text = tmp_path / "text"
    text.mkdir()
    shutil.copy(SMALL_PDF, papers / SMALL_PDF.name)

    lines: list[str] = []
    assert extract.run(papers_dir=papers, text_dir=text, out=lines.append) == 0
    assert any("dropped" in line for line in lines)
    assert (text / f"{SMALL_PDF.stem}.txt").stat().st_size > 1000


def test_run_fails_on_bad_pdf(tmp_path):
    papers = tmp_path / "papers"
    papers.mkdir()
    text = tmp_path / "text"
    text.mkdir()
    (papers / "broken.pdf").write_bytes(b"not a pdf")
    assert extract.run(papers_dir=papers, text_dir=text) == 1


def test_run_dirs(tmp_path):
    text = tmp_path / "text"
    text.mkdir()
    # papers dir missing
    assert extract.run(papers_dir=tmp_path / "missing", text_dir=text) == 1
    # empty papers dir
    empty = tmp_path / "empty"
    empty.mkdir()
    assert extract.run(papers_dir=empty, text_dir=text) == 1
    # text dir missing (extract no longer creates it)
    papers = tmp_path / "papers"
    papers.mkdir()
    shutil.copy(SMALL_PDF, papers / SMALL_PDF.name)
    assert extract.run(papers_dir=papers, text_dir=tmp_path / "no_text") == 1
