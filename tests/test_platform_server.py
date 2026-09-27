"""Tests for `entail serve` (ROADMAP product track P2; LIBRARY_DESIGN.md 13.2, 13.5): the platform's local web server
over a record folder - its API, its page, what it refuses, and the live event stream. Pure Python, a server on a free
port of 127.0.0.1. Run: python tests/test_platform_server.py"""
import http.client
import json
import os
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail.platform import server  # noqa: E402

RUN = "777-1790000000"


def _lines():
    return [
        {"v": 2, "t": 100.0, "run": RUN, "pid": 1, "boundary": "load:vllm.attention", "verdict": "pass",
         "name": "ModelProps", "rule": "match"},
        {"v": 2, "t": 101.0, "run": RUN, "pid": 1, "boundary": "load:transformers.tokenizer", "verdict": "broken",
         "name": "Tokenization", "rule": "tokenizer_ids", "note": "probe <script>alert(1)</script> differs",
         "declared": {"name": "Tokenization", "kind": "X", "value": "ids A",
                      "source": {"kind": "file", "where": "tokenizer.json"}, "certainty": "declared"}},
        {"v": 2, "t": 102.0, "run": RUN, "pid": 2, "boundaries": {
            "container:vllm.allocate_slots": {"checks": 4, "passed": {"kv_needed": 4}}}},
    ]


def _folder():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "record-2026-09-28.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for x in _lines():
            f.write(json.dumps(x) + "\n")
    return d, path


def _start(folder):
    srv = server.make_server(folder, 0)
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    t.start()
    return srv, srv.server_address[1]


def _get(port, path, host=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("GET", path, headers={"Host": host or f"127.0.0.1:{port}"})
    r = c.getresponse()
    body = r.read()
    c.close()
    return r.status, r.getheader("Content-Type"), body


def test_the_api_answers_from_the_records():
    folder, _ = _folder()
    srv, port = _start(folder)
    try:
        assert srv.server_address[0] == "127.0.0.1"
        st, ctype, body = _get(port, "/api/runs")
        runs = json.loads(body)
        assert st == 200 and ctype.startswith("application/json") and runs["runs"][0]["run"] == RUN, runs
        assert runs["runs"][0]["state"] == "broken" and runs["runs"][0]["broken_at"] == "load:transformers.tokenizer"
        st, _, body = _get(port, f"/api/graph?run={RUN}")
        g = json.loads(body)
        by = {n["id"]: n for n in g["nodes"]}
        assert st == 200 and by["tokenizer"]["state"] == "broken" and by["cache"]["passed"] == 4, by
        st, _, body = _get(port, f"/api/node?run={RUN}&node=tokenizer")
        d = json.loads(body)
        assert st == 200 and d["decisions"][0]["rule"] == "tokenizer_ids" and "<script>" in d["decisions"][0]["note"]
        st, _, body = _get(port, f"/api/locate?run={RUN}")
        assert st == 200 and json.loads(body)["broken_at"] == "load:transformers.tokenizer"
        st, _, body = _get(port, "/api/graph")               # no run named: the newest launch
        assert st == 200 and json.loads(body)["decisions"] == 2
        st, _, _ = _get(port, "/api/graph?run=nope")
        assert st == 404
    finally:
        srv.stopping = True
        srv.shutdown()
        srv.server_close()


def test_the_page_and_what_it_refuses():
    folder, _ = _folder()
    srv, port = _start(folder)
    try:
        st, ctype, body = _get(port, "/")
        assert st == 200 and ctype.startswith("text/html") and b"/static/app.js" in body
        st, ctype, body = _get(port, "/static/app.js")
        assert st == 200 and b"textContent" in body and b"innerHTML" not in body   # record text never as markup
        assert _get(port, "/static/style.css")[0] == 200
        assert _get(port, "/static/../server.py")[0] == 404 and _get(port, "/server.py")[0] == 404
        # a page elsewhere reaching this server through a name that resolves to 127.0.0.1 (DNS rebinding)
        assert _get(port, "/api/runs", host="evil.example:80")[0] == 403
        assert _get(port, "/api/runs", host=f"localhost:{port}")[0] == 200
    finally:
        srv.stopping = True
        srv.shutdown()
        srv.server_close()


def test_the_event_stream_sends_the_graph_again_when_lines_arrive():
    folder, path = _folder()
    srv, port = _start(folder)
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("GET", f"/api/events?run={RUN}", headers={"Host": f"127.0.0.1:{port}"})
        r = c.getresponse()
        assert r.status == 200 and r.getheader("Content-Type").startswith("text/event-stream")

        def next_graph():
            data = None
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                line = r.fp.readline().decode("utf-8")
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                elif line == "\n" and data is not None:
                    return data
            raise AssertionError("no event")

        first = next_graph()
        assert {n["id"]: n["state"] for n in first["nodes"]}["cache"] == "pass"
        with open(path, "a", encoding="utf-8") as f:      # the engine says more: a KV decision that broke
            f.write(json.dumps({"v": 2, "t": 103.0, "run": RUN, "pid": 2, "boundary": "container:vllm.allocate_slots",
                                "verdict": "broken", "name": "KvExtent", "rule": "kv_needed"}) + "\n")
        second = next_graph()
        assert {n["id"]: n["state"] for n in second["nodes"]}["cache"] == "broken", second
        c.close()
    finally:
        srv.stopping = True
        srv.shutdown()
        srv.server_close()


def test_a_line_still_being_written_waits_for_its_newline():
    folder, path = _folder()
    store = server.Store(folder)
    store.refresh()
    n = len(store.lines)
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"v": 2, "run": "' + RUN + '", "pid": 3, "boundary": "kernel:triton", "verdict": "pa')
    store.refresh()
    assert len(store.lines) == n
    with open(path, "a", encoding="utf-8") as f:
        f.write('ss"}\n')
    store.refresh()
    assert len(store.lines) == n + 1 and store.lines[-1][1]["verdict"] == "pass"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
