#!/usr/bin/env python3
"""Merge arXiv metadata (JSONL) with Semantic Scholar Graph API data (JSONL).

Usage:
    export S2_API_KEY=...        # optional but recommended
    python enrich_s2.py arxiv_sample.jsonl merged.jsonl

Lookup order per paper: DOI first, then ARXIV:<id> for anything the DOI
lookup missed. The S2 response is stored whole under "s2"; a few fields are
also hoisted to the top level for convenience.
"""
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import httpx

BASE = "https://api.semanticscholar.org/graph/v1/paper/batch"
BATCH_SIZE = 500          # documented max ids per batch call
MIN_INTERVAL = 1.1        # seconds between calls (intro key limit is 1 req/s)
MAX_RETRIES = 6

FIELDS = ",".join([
    "paperId", "externalIds", "url", "title", "abstract", "venue",
    "publicationVenue", "year", "publicationDate", "journal",
    "citationCount", "influentialCitationCount", "referenceCount",
    "isOpenAccess", "openAccessPdf", "s2FieldsOfStudy",
    "publicationTypes", "tldr",
])

headers = {}
if os.environ.get("S2_API_KEY"):
    headers["x-api-key"] = os.environ["S2_API_KEY"]
client = httpx.Client(headers=headers, timeout=120)

DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
MAX_SINGLE_FAILURES = 5   # more than this => systematic problem, not bad ids
_single_failures = 0


class BadRequest(Exception):
    """HTTP 400 from S2; message is the response body."""
_last_call = 0.0


def post_batch(ids):
    """POST one batch; returns a list aligned with `ids` (None = not found)."""
    global _last_call
    for attempt in range(MAX_RETRIES):
        wait = MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        try:
            r = client.post(BASE, params={"fields": FIELDS}, json={"ids": ids})
        except httpx.TransportError as e:
            print(f"  network error ({e}); retry {attempt + 1}", file=sys.stderr)
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            delay = float(r.headers.get("Retry-After", 2 ** attempt))
            print(f"  HTTP {r.status_code}; sleeping {delay:.0f}s", file=sys.stderr)
            time.sleep(delay)
            continue
        if r.status_code == 400:
            raise BadRequest(r.text[:500])
        r.raise_for_status()  # anything else unexpected: fail loudly
    raise RuntimeError("S2 batch failed after retries")


def resolve(chunk, keys, found):
    """Post one chunk; on HTTP 400, bisect to isolate the offending ids."""
    global _single_failures
    try:
        results = post_batch([keys[j] for j in chunk])
    except BadRequest as e:
        if len(chunk) == 1:
            _single_failures += 1
            print(f"  skipping {keys[chunk[0]]!r}: {e}", file=sys.stderr)
            if _single_failures > MAX_SINGLE_FAILURES:
                raise RuntimeError(
                    "Too many single-id 400s; likely a bad field name or "
                    "request shape, not bad ids. Last body: " + str(e))
            return
        print(f"  400 on {len(chunk)} ids ({e}); bisecting", file=sys.stderr)
        mid = len(chunk) // 2
        resolve(chunk[:mid], keys, found)
        resolve(chunk[mid:], keys, found)
        return
    for j, obj in zip(chunk, results):  # response order matches request
        if obj is not None:
            found[j] = obj


def lookup(keys):
    """keys: {record_index: s2_id_string}. Returns {record_index: s2_obj}."""
    idx = list(keys)
    found = {}
    for i in range(0, len(idx), BATCH_SIZE):
        chunk = idx[i:i + BATCH_SIZE]
        print(f"  batch {i // BATCH_SIZE + 1}: {len(chunk)} ids", file=sys.stderr)
        resolve(chunk, keys, found)
    return found


def first_doi(rec):
    # arXiv's doi field can hold several space-separated DOIs; take the first
    # one that looks like a DOI (junk values can make S2 reject the batch).
    for d in (rec.get("doi") or "").split():
        if DOI_RE.match(d):
            return d
    return None


def main(src, dst):
    with open(src, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"{len(records)} records", file=sys.stderr)

    match = {}  # record index -> ("doi" | "arxiv", s2 object)

    print("pass 1: DOI", file=sys.stderr)
    doi_keys = {i: f"DOI:{d}" for i, r in enumerate(records) if (d := first_doi(r))}
    for i, obj in lookup(doi_keys).items():
        match[i] = ("doi", obj)

    missing = [i for i in range(len(records)) if i not in match]
    if missing:
        print(f"pass 2: ARXIV id for {len(missing)} misses", file=sys.stderr)
        arxiv_keys = {i: f"ARXIV:{records[i]['id']}" for i in missing}
        for i, obj in lookup(arxiv_keys).items():
            match[i] = ("arxiv", obj)

    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(dst, "w", encoding="utf-8") as out:
        for i, rec in enumerate(records):
            how, s2 = match.get(i, (None, None))
            rec["s2_match"] = how
            rec["s2_retrieved_at"] = retrieved_at
            rec["citation_count"] = s2.get("citationCount") if s2 else None
            rec["s2"] = s2
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_doi = sum(1 for h, _ in match.values() if h == "doi")
    print(f"done: {len(match)}/{len(records)} matched "
          f"({n_doi} by DOI, {len(match) - n_doi} by arXiv id)", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
