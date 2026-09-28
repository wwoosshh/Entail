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


def _post(port, body, token=None, origin="self", host=None, path="/api/safe-mode"):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Host": host or f"127.0.0.1:{port}", "Content-Type": "application/json"}
    if origin:
        headers["Origin"] = f"http://127.0.0.1:{port}" if origin == "self" else origin
    if token is not None:
        headers["X-Entail-Token"] = token
    c.request("POST", path, body=body if isinstance(body, bytes) else json.dumps(body).encode(), headers=headers)
    r = c.getresponse()
    out = r.read()
    c.close()
    return r.status, out


def test_the_one_write_is_the_safety_mode_and_needs_the_page_token_and_origin():
    folder, _ = _folder()
    srv, port = _start(folder)
    try:
        st, _, body = _get(port, "/api/safe-mode")
        s = json.loads(body)
        assert st == 200 and s["mode"] == "auto" and s["set"] is False and s["paths"] == [], s
        page = _get(port, "/")[2].decode("utf-8")
        token = srv.safe.token
        assert f'content="{token}"' in page and "__ENTAIL_TOKEN__" not in page
        # refused: another site's page, no Origin, a wrong token, another address, a mode that is not one
        assert _post(port, {"mode": "all"}, token, origin="http://evil.example")[0] == 403
        assert _post(port, {"mode": "all"}, token, origin=None)[0] == 403
        assert _post(port, {"mode": "all"}, "not-the-token")[0] == 403
        assert _post(port, {"mode": "all"}, token, host="evil.example:80")[0] == 403
        assert _post(port, {"mode": "everything"}, token)[0] == 400
        assert _post(port, b"x" * 5000, token)[0] == 413
        # the platform's writes are exactly two (LIBRARY_DESIGN.md 13.2): any other path is not one
        for other in ("/api/runs", "/api/graph", "/api/settings", "/api/node", "/", "/api/safe-mode/../nodes"):
            assert _post(port, {"mode": "all"}, token, path=other)[0] == 404, other
        assert not os.path.exists(os.path.join(folder, "safe_mode.json"))
        # the write: the file the next start reads, and a new token (the old one works once)
        st, body = _post(port, {"mode": "all"}, token)
        d = json.loads(body)
        assert st == 200 and d["mode"] == "all" and d["set"] is True and d["token"] != token, d
        with open(os.path.join(folder, "safe_mode.json"), encoding="utf-8") as f:
            assert json.load(f)["mode"] == "all"
        from entail import safe_mode
        old = {k: os.environ.get(k) for k in ("ENTAIL_LOG_DIR", "ENTAIL_SAFE")}
        os.environ["ENTAIL_LOG_DIR"] = folder
        os.environ.pop("ENTAIL_SAFE", None)
        try:
            assert safe_mode.mode() == "all"
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        assert _post(port, {"mode": "off"}, token)[0] == 403              # used once
        st, body = _post(port, {"mode": "off"}, d["token"], origin=f"http://localhost:{port}")
        assert st == 200 and json.loads(body)["mode"] == "off"
        # the selective safe path's state, as the engine left it
        with open(os.path.join(folder, "safe_paths.json"), "w", encoding="utf-8") as f:
            json.dump({"k1": {"engine": "vllm", "model": "/m/nemotron", "status": "searching",
                              "candidates": ["speculative_decoding", "cuda_graphs"], "tried": [], "off": [],
                              "pairs": ["decode_prefill"], "updated": 5.0}}, f)
        s = json.loads(_get(port, "/api/safe-mode")[2])
        assert s["paths"][0]["key"] == "k1" and s["paths"][0]["status"] == "searching", s
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)    # no preflight is answered
        c.request("OPTIONS", "/api/safe-mode", headers={"Host": f"127.0.0.1:{port}", "Origin": "http://evil.example"})
        r = c.getresponse()
        assert r.status >= 400 and r.getheader("Access-Control-Allow-Origin") is None
        c.close()
    finally:
        srv.stopping = True
        srv.shutdown()
        srv.server_close()


def test_a_record_file_it_cannot_read_is_said_not_skipped():
    """P6: a broken link or an unreadable file under the record name is named in /api/runs (the E2 poller's broken
    links looked like "no records yet")."""
    folder, _ = _folder()
    os.makedirs(os.path.join(folder, "record-2026-09-29.jsonl"))       # a name the store reads, and cannot open
    srv, port = _start(folder)
    try:
        runs = json.loads(_get(port, "/api/runs")[2])
        assert runs["folder_found"] is True and runs["runs"], runs
        assert [u["file"] for u in runs["unreadable"]] == ["record-2026-09-29.jsonl"], runs["unreadable"]
        page = _get(port, "/static/app.js")[2]
        assert b"unreadable" in page and b"folder_found" in page
    finally:
        srv.stopping = True
        srv.shutdown()
        srv.server_close()
    srv, port = _start(os.path.join(folder, "not-there"))
    try:
        runs = json.loads(_get(port, "/api/runs")[2])
        assert runs["folder_found"] is False and runs["runs"] == [] and runs["unreadable"] == [], runs
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


def test_the_page_translates_only_rules_the_core_still_says():
    """The page reads the common rule texts in Korean (field test, entail#7). Every text it translates must still be
    one of the core's rules, or a renamed rule would quietly lose its translation."""
    import re
    from entail.contracts import RULES
    js = open(os.path.join(server.STATIC, "app.js"), encoding="utf-8").read()
    block = js[js.index("const RULE_KO = {"):]
    block = block[:block.index("};")]
    keys = re.findall(r'^\s*"([^"]+)":', block, re.M)
    assert len(keys) >= 10, keys
    assert all(k in RULES.values() for k in keys), [k for k in keys if k not in RULES.values()]
    # the facts the page offers to declare are the ones a manifest drafts (entail/manifest.py relevant_names)
    names = re.findall(r'"(\w+)"', js[js.index("const DECLARABLE"):].split(";", 1)[0])
    assert set(names) == {"Prediction", "LatentScale", "ModelProps", "Rotary", "Template"}, names


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
