"""DOI extraction for paper PDFs.

Best effort: a paper's own DOI is on its title page, so only the first
few pages are read. Preprints without a DOI (arXiv-only) yield None.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pypdf import PdfReader

logging.getLogger("pypdf").setLevel(logging.ERROR)  # quiet font warnings

#: A bare DOI token: 10. + 4-9 digit registrant + "/" + non-space suffix.
DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s\"'<>\[\]{}|()]+)")
# Explicit markers are less likely to be somebody else's citation.
DOI_ORG_RE = re.compile(
    r"(?:dx\.)?doi\.org/\s*(10\.\d{4,9}/[^\s\)\]\"']+)", re.IGNORECASE
)
DOI_LABEL_RE = re.compile(
    r"\bdoi\b\s*:\s*(10\.\d{4,9}/[^\s\)\]\"']+)", re.IGNORECASE
)


def _clean(match: str) -> str | None:
    """Trim trailing punctuation; drop a fused 'doi' label if the rest
    still looks like a complete DOI ('...691628doi' -> '...691628')."""
    candidate = match.rstrip(".,;:)]}'\"")
    trimmed = re.sub(r"doi$", "", candidate, flags=re.IGNORECASE)
    if trimmed != candidate and re.fullmatch(r"10\.\d{4,9}/\S+", trimmed):
        return trimmed
    return candidate if re.fullmatch(r"10\.\d{4,9}/\S+", candidate) else None


def extract_doi_from_text(text: str) -> str | None:
    """Best-effort DOI from a stretch of text (e.g. a PDF's first pages).

    Prefers explicit markers (doi.org links, 'doi:' labels) over the
    first bare 10.xxxx/... token, since bare tokens in reference lists
    are usually citations to other papers.
    """
    for pattern in (DOI_ORG_RE, DOI_LABEL_RE, DOI_RE):
        match = pattern.search(text)
        if match:
            cleaned = _clean(match.group(1))
            if cleaned is not None:
                return cleaned
    return None


def extract_doi(path: Path, max_pages: int = 2) -> str | None:
    """The paper's own DOI from the first *max_pages* pages of a PDF.

    None when there is none or the PDF cannot be parsed.
    """
    try:
        reader = PdfReader(str(path))
        text = " ".join(
            (page.extract_text() or "") for page in reader.pages[:max_pages]
        )
    except Exception:  # noqa: BLE001 - best effort by design
        return None
    return extract_doi_from_text(text)
