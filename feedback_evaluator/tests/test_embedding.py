import hashlib
import json

import httpx
import numpy as np
import pytest

from feedback_evaluator import embedding

UNIT_A = {
    "paper_id": "p1",
    "doi": "10.9999/a.1",
    "prompt_id": "P1",
    "feedback_id": "p1-P1",
    "feedback": "alpha body line one.\nalpha body line two.",
    "originality_score": 3,
    "overall_paper_score": 4,
    "impact_score": 5,
    "citations": 12,
    "model": "sonnet",
    "created_at": "t",
}
UNIT_B = {
    "paper_id": "p2",
    "doi": None,
    "prompt_id": "P2",
    "feedback_id": "p2-P2",
    "feedback": "beta body line.",
    "originality_score": None,
    "overall_paper_score": None,
    "impact_score": None,
    "citations": None,
    "model": "sonnet",
    "created_at": "t",
}


def write_evaluations(feedback_dir, units):
    path = feedback_dir / "evaluations.jsonl"
    path.write_text("".join(json.dumps(u) + "\n" for u in units))
    return path


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


def test_unit_records_dedupes_and_keeps_order(tmp_path):
    feedback_dir = tmp_path / "feedback"
    feedback_dir.mkdir()
    write_evaluations(feedback_dir, [UNIT_A, UNIT_B, UNIT_A])
    rows = embedding.unit_records(feedback_dir / "evaluations.jsonl")
    assert [row["feedback_id"] for row in rows] == ["p1-P1", "p2-P2"]


def test_run_end_to_end(tmp_path, fake_server):
    base_url, handler, seen = fake_server
    feedback_dir = tmp_path / "feedback"
    feedback_dir.mkdir()
    write_evaluations(feedback_dir, [UNIT_A, UNIT_B])

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
    # one record per evaluation unit, keyed <paper_id>#<feedback_id>
    assert [row["id"] for row in rows] == ["p1#p1-P1", "p2#p2-P2"]
    assert rows[0]["paper"] == "p1" and rows[0]["feedback_id"] == "p1-P1"
    assert rows[0]["text"] == UNIT_A["feedback"]
    # scores/metadata carried straight from the evaluation record
    assert rows[0]["doi"] == "10.9999/a.1"
    assert rows[0]["citations"] == 12
    assert rows[0]["prompt_id"] == "P1"
    assert rows[0]["impact_score"] == 5
    # a null-DOI unit stays null (no header fallback to fall through to)
    assert rows[1]["doi"] is None
    assert rows[1]["citations"] is None
    assert rows[1]["prompt_id"] == "P2"
    # the feedback/*.md is a human artifact: absent here, and unneeded
    assert list(feedback_dir.glob("*.md")) == []
    assert all(row["dim"] == 8 and len(row["vector"]) == 8 for row in rows)
    assert rows[0]["model"] == "m1"

    manifest = json.load(open(tmp_path / "embeddings" / "manifest.json"))
    assert manifest["dim"] == 8
    assert manifest["total_units"] == 2
    assert manifest["model"] == "m1"
    assert manifest["server"] == base_url + "/v1"
    assert manifest["units_per_paper"] == {"p1": 1, "p2": 1}
    assert manifest["papers"] == ["p1", "p2"]


def test_run_no_evaluations(tmp_path, fake_server):
    base_url, _, _ = fake_server
    empty_dir = tmp_path / "feedback"
    empty_dir.mkdir()
    # feedback dir without evaluations.jsonl
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
