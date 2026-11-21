import hashlib
import json

import httpx
import numpy as np
import pytest

from feedback_evaluator import embedding

FEEDBACK_MD = """# Feedback: p1

- Model: sonnet

## [p1-a] Section A

alpha body line one.
alpha body line two.

> scores: originality_score=3, overall_paper_score=4, impact_score=5

## [p1-b] Section B

beta body line.
"""


FEEDBACK_MD_WITH_DOI = FEEDBACK_MD.replace(
    "# Feedback: p1\n", "# Feedback: p1\n\n- DOI: 10.9999/sample.1\n", 1
)


def feedback_file(tmp_path, with_doi=False):
    path = tmp_path / "p1.md"
    path.write_text(FEEDBACK_MD_WITH_DOI if with_doi else FEEDBACK_MD)
    return path


def test_iter_feedback_lines(tmp_path):
    # prose lines only; headings, blanks, "> " meta, and the "- Model:"
    # header bullet are skipped; line numbers are 1-based
    got = embedding.iter_feedback_lines(feedback_file(tmp_path))
    assert got == [
        embedding.FeedbackLine(7, "p1-a", "alpha body line one."),
        embedding.FeedbackLine(8, "p1-a", "alpha body line two."),
        embedding.FeedbackLine(14, "p1-b", "beta body line."),
    ]


@pytest.fixture()
def fake_server():
    """Minimal OpenAI-compatible /embeddings server via stdlib http.server.

    Returns (base_url, handler_class, seen) where seen captures the last
    request's auth header and payload; handler_class.failed toggles a 500.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen: dict = {}

    def vec_for(text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        vector = np.array([b / 255.0 for b in h[:8]])
        return (vector / np.linalg.norm(vector)).tolist()

    class Handler(BaseHTTPRequestHandler):
        fail = False

        def do_POST(self):
            if Handler.fail:
                self.send_response(500)
                self.end_headers()
                return
            body = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
            )
            seen["auth"] = self.headers.get("Authorization")
            seen["payload"] = body
            out = {
                "data": [
                    {"index": i, "embedding": vec_for(t)}
                    for i, t in enumerate(body["input"])
                ]
            }
            payload = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", Handler, seen
    server.shutdown()


def test_feedback_doi(tmp_path):
    assert embedding.feedback_doi(feedback_file(tmp_path)) is None
    path = feedback_file(tmp_path, with_doi=True)
    assert embedding.feedback_doi(path) == "10.9999/sample.1"


def test_run_end_to_end(tmp_path, fake_server):
    base_url, handler, seen = fake_server
    feedback_dir = tmp_path / "feedback"
    feedback_dir.mkdir()
    feedback_file(feedback_dir)

    rc = embedding.run(
        feedback_dir=feedback_dir,
        out_dir=tmp_path / "embeddings",
        embed_url=base_url + "/v1",
        model="m1",
        api_key="k",
        batch_size=2,
    )
    assert rc == 0
    assert seen["auth"] == "Bearer k"

    rows = [
        json.loads(line)
        for line in open(tmp_path / "embeddings" / "embeddings.jsonl")
        if line.strip()
    ]
    assert [row["id"] for row in rows] == ["p1#L7", "p1#L8", "p1#L14"]
    assert rows[0]["paper"] == "p1" and rows[0]["feedback_id"] == "p1-a"
    assert rows[0]["line_number"] == 7
    assert all(row["doi"] is None for row in rows)
    assert rows[2]["feedback_id"] == "p1-b"
    assert all(row["dim"] == 8 and len(row["vector"]) == 8 for row in rows)
    assert rows[0]["model"] == "m1"

    manifest = json.load(open(tmp_path / "embeddings" / "manifest.json"))
    assert manifest["dim"] == 8
    assert manifest["total_lines"] == 3
    assert manifest["model"] == "m1"
    assert manifest["server"] == base_url + "/v1"
    assert manifest["files"]["p1"]["lines"] == 3
    assert manifest["papers"] == ["p1"]


def test_run_end_to_end_carries_doi(tmp_path, fake_server):
    base_url, _, _ = fake_server
    feedback_dir = tmp_path / "feedback"
    feedback_dir.mkdir()
    feedback_file(feedback_dir, with_doi=True)

    rc = embedding.run(
        feedback_dir=feedback_dir,
        out_dir=tmp_path / "embeddings",
        embed_url=base_url + "/v1",
        model="m1",
        api_key="",
    )
    assert rc == 0
    rows = [
        json.loads(line)
        for line in open(tmp_path / "embeddings" / "embeddings.jsonl")
        if line.strip()
    ]
    # the two extra header lines shift the prose line numbers down by 2
    assert [row["id"] for row in rows] == ["p1#L9", "p1#L10", "p1#L16"]
    assert all(row["doi"] == "10.9999/sample.1" for row in rows)


def test_run_no_feedback(tmp_path, fake_server):
    base_url, _, _ = fake_server
    empty_dir = tmp_path / "feedback"
    empty_dir.mkdir()
    assert (
        embedding.run(
            feedback_dir=empty_dir,
            out_dir=tmp_path / "out",
            embed_url=base_url,
            model="m",
            api_key="",
        )
        == 1
    )
    assert (
        embedding.run(
            feedback_dir=tmp_path / "missing",
            out_dir=tmp_path / "out",
            embed_url=base_url,
            model="m",
            api_key="",
        )
        == 1
    )


def test_embed_batch_retries_on_500_then_raises(fake_server):
    base_url, handler, _ = fake_server
    handler.fail = True
    with httpx.Client() as client:
        with pytest.raises(RuntimeError, match="after 2 tries"):
            embedding.embed_batch(
                client,
                base_url + "/v1/embeddings",
                api_key="",
                model="",
                texts=["x"],
                timeout=5.0,
                max_retries=2,
            )


def test_embed_batch_4xx_fails_immediately(fake_server):
    import time

    base_url, handler, _ = fake_server

    class Handler404(handler):
        def do_POST(self):
            self.send_response(404)
            self.end_headers()

    import threading
    from http.server import HTTPServer

    server = HTTPServer(("127.0.0.1", 0), Handler404)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        started = time.monotonic()
        with httpx.Client() as client:
            with pytest.raises(httpx.HTTPStatusError):
                embedding.embed_batch(
                    client,
                    f"http://127.0.0.1:{server.server_address[1]}"
                    "/v1/embeddings",
                    api_key="",
                    model="",
                    texts=["x"],
                    timeout=5.0,
                    max_retries=5,
                )
        # a 4xx must not wait through the retry ladder
        assert time.monotonic() - started < 1.0
    finally:
        server.shutdown()
