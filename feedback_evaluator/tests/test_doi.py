"""DOI extraction tests.

The expected values are real DOIs from the papers/ corpus (verified by
scanning it), not invented strings.
"""

from pathlib import Path

import pytest

from feedback_evaluator import doi

# repo root is two levels up from this tests/ directory
PAPERS_DIR = Path(__file__).resolve().parents[2] / "papers"


def test_doi_from_doiorg_link():
    # s41586-023-06137-x.pdf (Nature)
    text = "Title here. https://doi.org/10.1038/s41586-023-06137-x"
    assert doi.extract_doi_from_text(text) == "10.1038/s41586-023-06137-x"


def test_doi_from_label():
    # A-cognitive-fallacy-in-a-fish...pdf (Elsevier Cognition)
    text = "Published in Cognition. DOI: 10.1016/j.cognition.2026.106438."
    assert doi.extract_doi_from_text(text) == "10.1016/j.cognition.2026.106438"


def test_doi_bare_token():
    # 15306315.pdf (Proc. R. Soc. B)
    text = "Proc. R. Soc. B 272, 1955-1961.\n10.1098/rspb.2004.2746"
    assert doi.extract_doi_from_text(text) == "10.1098/rspb.2004.2746"


def test_doi_marker_wins_over_earlier_bare_citation():
    # a funder/reference token appears before the paper's own DOI link
    text = "10.13039/501100011033\nArticle DOI: 10.5281/zenodo.23945788"
    assert doi.extract_doi_from_text(text) == "10.5281/zenodo.23945788"


def test_doi_fused_tail_truncated():
    # bioRxiv preprint headers render a fused doi.org link + trailing 'doi'
    text = "https://doi.org/10.1101/2025.12.01.691628doi"
    assert doi.extract_doi_from_text(text) == "10.1101/2025.12.01.691628"


def test_doi_trailing_punctuation_stripped():
    assert (
        doi.extract_doi_from_text("x doi.org/10.1038/s41562-018-0474-5, y")
        == "10.1038/s41562-018-0474-5"
    )


def test_doi_none_when_absent():
    # 2611.24949v1.pdf is arXiv-only: no DOI in its header
    assert doi.extract_doi_from_text("arXiv:2611.24949v1 [stat.ME]") is None
    assert doi.extract_doi_from_text("") is None


def test_doi_extract_from_broken_or_missing_pdf(tmp_path):
    assert doi.extract_doi(tmp_path / "absent.pdf") is None
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf at all")
    assert doi.extract_doi(broken) is None


@pytest.mark.skipif(
    not PAPERS_DIR.is_dir(), reason="papers/ corpus not present"
)
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # (paper, its own title-page DOI)
        ("15306315.pdf", "10.1098/rspb.2004.2746"),
        ("s41586-023-06137-x.pdf", "10.1038/s41586-023-06137-x"),
        ("2611.25218.pdf", "10.5281/zenodo.23945788"),
        (
            "A-cognitive-fallacy-in-a-fish--Glass-catfish--"
            "like-humans--make-s_2026_Cogni.pdf",
            "10.1016/j.cognition.2026.106438",
        ),
        # arXiv-only preprint: no DOI
        ("1103.1676v1.pdf", None),
    ],
)
def test_doi_against_real_papers(name, expected):
    assert doi.extract_doi(PAPERS_DIR / name) == expected
