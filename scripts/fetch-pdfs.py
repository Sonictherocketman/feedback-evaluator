#!/usr/bin/env python3
"""Download arXiv PDFs for records in a JSONL file.

Usage:
    export CONTACT=you@example.com     # goes in the User-Agent
    python fetch_pdfs.py merged.jsonl merged_pdfs.jsonl pdfs/

Design, following arXiv's API terms of use and robots policy:
  * one connection, strictly sequential, >= 3.5 s between requests
  * export.arxiv.org host (the one arXiv asks programs to use)
  * pinned to the latest version listed in the record, so the file matches
    the metadata you sampled (re-runnable, citable)
  * stops immediately on HTTP 403 (arXiv treats continued requests after a
    403 as an attack); backs off on 429/5xx
  * resumable: existing, valid files are skipped
Adds to each record: pdf_path, pdf_version, pdf_sha256, pdf_bytes, pdf_status.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import httpx

HOST = "https://export.arxiv.org/pdf"
MIN_INTERVAL = 3.5
MAX_RETRIES = 5

client = httpx.Client(
    headers={"User-Agent": f"arxiv-sample-fetcher/0.1 (mailto:{os.environ.get('CONTACT', 'unset')})"},
    timeout=httpx.Timeout(60, connect=15),
    follow_redirects=True,
)
_last_call = 0.0


class Blocked(Exception):
    pass


def latest_version(rec):
    versions = rec.get("versions") or []
    return versions[-1]["version"] if versions else "v1"


def safe_name(arxiv_id):
    return arxiv_id.replace("/", "_")  # old-style ids like hep-th/9901001


def is_valid_pdf(path):
    try:
        with open(path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False


def fetch(url, dest):
    """Download url to dest. Returns (status, sha256, nbytes)."""
    global _last_call
    for attempt in range(MAX_RETRIES):
        wait = MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        try:
            with client.stream("GET", url) as r:
                if r.status_code == 403:
                    raise Blocked(url)
                if r.status_code == 404:
                    return "not_found", None, 0
                if r.status_code == 400:
                    # Malformed request (bad id/version); retrying can't help.
                    print(f"  HTTP 400 for {url}", file=sys.stderr)
                    return "bad_request", None, 0
                if r.status_code in (406, 429, 500, 502, 503, 504):
                    delay = float(r.headers.get("Retry-After", 5 * 2 ** attempt))
                    print(f"  HTTP {r.status_code}; sleeping {delay:.0f}s", file=sys.stderr)
                    time.sleep(delay)
                    continue
                if 400 <= r.status_code < 500:  # any other non-retryable 4xx
                    print(f"  HTTP {r.status_code} for {url}", file=sys.stderr)
                    return f"http_{r.status_code}", None, 0
                r.raise_for_status()
                tmp = dest.with_suffix(".part")
                h, n = hashlib.sha256(), 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes(65536):
                        f.write(chunk)
                        h.update(chunk)
                        n += len(chunk)
            if not is_valid_pdf(tmp):  # e.g. an HTML error/landing page
                tmp.unlink(missing_ok=True)
                return "not_pdf", None, 0
            tmp.replace(dest)
            return "ok", h.hexdigest(), n
        except httpx.TransportError as e:
            print(f"  network error ({e}); retry {attempt + 1}", file=sys.stderr)
            time.sleep(5 * 2 ** attempt)
    return "failed", None, 0


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main(src, dst, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(src, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    counts = {}
    try:
        for n, rec in enumerate(records, 1):
            ver = latest_version(rec)
            name = f"{safe_name(rec['id'])}{ver}.pdf"
            dest = outdir / name
            rec["pdf_version"] = ver

            if dest.exists() and is_valid_pdf(dest):
                status, sha, size = "ok", sha256_of(dest), dest.stat().st_size
            else:
                print(f"[{n}/{len(records)}] {name}", file=sys.stderr)
                status, sha, size = fetch(f"{HOST}/{rec['id']}{ver}", dest)

            rec.update(pdf_status=status, pdf_sha256=sha, pdf_bytes=size,
                       pdf_path=str(dest) if status == "ok" else None)
            counts[status] = counts.get(status, 0) + 1
    except Blocked as e:
        print(f"\nHTTP 403 from {e}. Stopping; do not retry until you've "
              "checked arXiv's access policy / contacted them.", file=sys.stderr)
    except KeyboardInterrupt:
        print("\ninterrupted; writing progress", file=sys.stderr)
    finally:
        # Records not reached keep no pdf_* fields; rerun to resume.
        with open(dst, "w", encoding="utf-8") as out:
            for rec in records:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"summary: {counts}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
