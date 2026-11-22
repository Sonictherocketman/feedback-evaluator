"""``feedback_evaluator extract`` — turn the paper PDFs into plain text.

Each PDF in ``papers/`` becomes ``papers/text/<paper>.txt`` so the eval
stage can pass the model a text file instead of the raw PDF. Text layer
only: figures are dropped and equations survive as far as the PDF's
encoding allows (Unicode where available, ASCII where not). Papers whose
text file already exists and is non-empty are skipped, so the stage is
resumable and cheap to re-run.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader

from .common import write_text_atomic

logging.getLogger("pypdf").setLevel(logging.ERROR)  # quiet font warnings


def merge_pages(pages: list[str]) -> str:
    """Join per-page text into one document: strip each line, drop blank
    lines, and drop lines that repeat a line near the end of the earlier
    pages (running headers/footers)."""
    kept: list[str] = []
    for page in pages:
        lines = [line.rstrip() for line in page.splitlines() if line.strip()]
        tail = set(kept[-6:])
        kept.extend(line for line in lines if line not in tail)
    return "\n".join(kept)


def extract_pdf_text(path: Path) -> tuple[str, int]:
    """Plain text of a whole PDF, page by page.

    Returns (text, skipped_pages): a page that fails to parse (e.g. a
    dense figure page that trips pypdf's token limit) is dropped rather
    than failing the whole paper, since figures are acceptable loss.
    """
    reader = PdfReader(str(path))
    pages: list[str] = []
    skipped = 0
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - drop the page, keep the rest
            skipped += 1
    return merge_pages(pages), skipped


def run(*, papers_dir, text_dir, out=print, **_) -> int:
    """Convert every PDF in *papers_dir* to **text_dir**. 0 on success,
    1 if any PDF failed."""
    papers_dir = Path(papers_dir)
    if not papers_dir.is_dir():
        print(f"error: papers directory not found: {papers_dir}")
        return 1
    pdfs = sorted(papers_dir.glob("*.pdf"))
    if not pdfs:
        print(f"error: no PDFs found in {papers_dir}")
        return 1
    text_dir = Path(text_dir)
    if not text_dir.is_dir():
        print(f"error: text directory not found: {text_dir}")
        return 1

    done = skipped = errors = degraded = 0
    for pdf in pdfs:
        dest = text_dir / f"{pdf.stem}.txt"
        if dest.is_file() and dest.stat().st_size:
            out(f"[skip] {pdf.name} (text present)")
            skipped += 1
            continue
        try:
            text, dropped_pages = extract_pdf_text(pdf)
        except Exception as exc:
            out(f"[FAIL] {pdf.name}: {exc}")
            errors += 1
            continue
        if not text.strip() and dropped_pages:
            out(f"[FAIL] {pdf.name}: every page failed to parse")
            errors += 1
            continue
        write_text_atomic(dest, text)
        if dropped_pages:
            out(
                f"[warn] {pdf.name} -> text/{dest.name} ({len(text)} chars, "
                f"dropped {dropped_pages} page(s))"
            )
            degraded += 1
        else:
            out(f"[done] {pdf.name} -> text/{dest.name} ({len(text)} chars)")
        done += 1
    out(
        f"\nExtracted {done} ({degraded} partial), skipped {skipped}, "
        f"failed {errors} -> {text_dir}"
    )
    return 1 if errors else 0
