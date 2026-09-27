"""Tests for custom nodes (ROADMAP product track P5; LIBRARY_DESIGN.md 13.7; entail/nodes.py): a developer's validator
at a point of their program - what it found as a core decision (fact Check), the core's guards around it (a failure,
a budget, nodes turned off), workshop packages through the entry point group entail.nodes, and the platform's node
and switch for it. Pure Python. Run: python tests/test_nodes.py"""
import http.client
import io
import json
import os
import sys
import tempfile
import threading
import time
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import core, nodes, record, tally  # noqa: E402
from entail.core import RoleError  # noqa: E402
from entail.platform import graph, server  # noqa: E402

_N = [0]


class Env:
    def __init__(self, **env):
        self.env = env

    def __enter__(self):
        keys = ("ENTAIL_LOG_DIR", "ENTAIL_RECORD", "ENTAIL_NODES", "ENTAIL_ON_BROKEN", "ENTAIL")
        self.old = {k: os.environ.get(k) for k in keys}
        self.dir = tempfile.mkdtemp()
        os.environ["ENTAIL_LOG_DIR"] = self.dir
        for k in ("ENTAIL_RECORD", "ENTAIL_NODES", "ENTAIL_ON_BROKEN"):
            os.environ.pop(k, None)
        os.environ.update(self.env)
        self.mode = core.mode()
        core.set_mode(self.env.get("ENTAIL", "load"))
        self.paths = list(sys.path)
        nodes.reset()
        tally.reset()
        return self

    def __exit__(self, *exc):
        record.close_files()
        core.set_mode(self.mode)
        sys.path[:] = self.paths
        nodes.reset()
        tally.reset()
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def records(self):
        record.close_files()
        out = []
        for name in os.listdir(self.dir):
            if name.startswith("record-"):
                out += [json.loads(x) for x in open(os.path.join(self.dir, name), encoding="utf-8")]
        return out


def quiet(fn, *a, **k):
    with redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _json_object():
    calls = []

    @nodes.validator("json_object")
    def json_object(value, keys=()):
        calls.append(value)
        try:
            obj = json.loads(value)
        except ValueError as e:
            return nodes.broken(f"not JSON: {e}")
        missing = [k for k in keys if k not in obj]
        return nodes.broken(f"missing keys {missing}") if missing else nodes.ok()

    return json_object, calls


def test_what_a_validator_finds_is_a_core_decision_once_per_finding():
    with Env() as e:
        v, calls = _json_object()
        good = '{"title": "a", "body": "b"}'
        assert nodes.check("demo.answer", good, v, keys=("title", "body")) == good     # the value comes back
        quiet(nodes.check, "demo.answer", '{"title": "a"}', v, keys=("title", "body"))
        quiet(nodes.check, "demo.answer", '{"title": "b"}', v, keys=("title", "body"))  # the same finding again
        quiet(nodes.check, "demo.answer", "not json", v, keys=("title", "body"))
        rows = [r for r in e.records() if r.get("boundary") == "node:demo.answer/json_object" and r.get("verdict")]
        assert [r["verdict"] for r in rows] == ["broken", "broken"], rows
        assert rows[0]["name"] == "Check" and "missing keys ['body']" in rows[0]["note"], rows[0]
        assert rows[0]["rule"].startswith("a custom node's validator") and rows[1]["note"].count("not JSON") == 1
        s = tally.stats("node:demo.answer/json_object")
        assert s["checks"] == 4 and s["broken"] == 3 and s["passed"] == {"node_check": 1}, s
        assert len(calls) == 4


def test_a_validator_that_cannot_tell_or_fails_or_is_slow_never_breaks_the_program():
    with Env() as e:
        @nodes.validator("needs_lang")
        def needs_lang(value, lang=None):
            return nodes.unknown("no language given") if lang is None else nodes.ok()

        quiet(nodes.check, "demo.docs", "text", needs_lang)
        n = [0]

        def flaky(value):
            n[0] += 1
            raise KeyError("oops")

        for _ in range(3):
            quiet(nodes.check, "demo.docs", "text", flaky)
        assert n[0] == 2                                   # left out after two failures

        @nodes.validator("slow", budget_ms=1)
        def slow(value):
            time.sleep(0.02)
            return True

        quiet(nodes.check, "demo.docs", "x", slow)
        quiet(nodes.check, "demo.docs", "x", slow)
        rows = e.records()
        unknown = [r for r in rows if r.get("verdict") == "unknown"]
        assert {r["boundary"] for r in unknown} == {"node:demo.docs/needs_lang", "node:demo.docs/flaky"}, unknown
        said = [r["text"] for r in rows if r.get("said") == "node:demo.docs/slow"]
        assert len(said) == 1 and "over its budget of 1 ms" in said[0], said
        assert tally.stats("node:demo.docs/slow")["checks"] == 1    # the second call did not run it
        quiet(nodes.check, "demo.docs", "x", lambda v: 42)          # not a result: unknown, said why
        assert any("returned int" in (r.get("note") or "") for r in e.records())
    with Env(ENTAIL="debug"):
        def boom(value):
            raise ValueError("bad")

        try:
            quiet(nodes.check, "demo.docs", "x", boom)
            raise AssertionError("debug mode must raise")
        except ValueError:
            pass


def test_the_stop_policy_stops_at_a_broken_check():
    with Env(ENTAIL_ON_BROKEN="stop") as e:
        v, _ = _json_object()
        try:
            quiet(nodes.check, "demo.answer", "no", v)
            raise AssertionError("must stop")
        except RoleError:
            pass
        assert [r["verdict"] for r in e.records() if r.get("verdict")] == ["refused"]


def test_nothing_runs_when_entail_is_off_or_the_node_is_turned_off():
    for env in ({"ENTAIL": "off"}, {"ENTAIL_NODES": "off"}):
        with Env(**env):
            v, calls = _json_object()
            nodes.check("demo.answer", "no", v)
            assert calls == []
    with Env() as e:
        v, calls = _json_object()
        with open(os.path.join(e.dir, "nodes.json"), "w", encoding="utf-8") as f:
            json.dump({"off": ["demo.answer"]}, f)
        nodes.check("demo.answer", "no", v)
        assert calls == []
        os.remove(os.path.join(e.dir, "nodes.json"))
        nodes._OFF["at"] = 0.0                              # (it looks again at most once a second)
        quiet(nodes.check, "demo.answer", "no", v)
        assert calls == ["no"]
        assert quiet(nodes.check, "Bad Name", 1, v) == 1 and calls == ["no"]    # not a node name: nothing checked


def test_watch_checks_what_a_function_returns():
    with Env() as e:
        v, calls = _json_object()

        @nodes.watch("demo.answer", v, keys=("title",))
        def ask(q):
            return '{"body": "x"}' if q == "bad" else '{"title": "t"}'

        assert ask("good") == '{"title": "t"}'
        assert quiet(ask, "bad") == '{"body": "x"}'
        assert [r["verdict"] for r in e.records() if r.get("verdict")] == ["broken"]
        assert calls == ['{"title": "t"}', '{"body": "x"}']


def _package(text, ep_name):
    _N[0] += 1
    root = tempfile.mkdtemp()
    mod = f"fake_nodes_{_N[0]}"
    with open(os.path.join(root, mod + ".py"), "w", encoding="utf-8") as f:
        f.write(text)
    info = os.path.join(root, f"{mod}-0.1.dist-info")
    os.makedirs(info)
    with open(os.path.join(info, "METADATA"), "w", encoding="utf-8") as f:
        f.write(f"Metadata-Version: 2.1\nName: {mod}\nVersion: 0.1\n")
    with open(os.path.join(info, "entry_points.txt"), "w", encoding="utf-8") as f:
        f.write(f"[entail.nodes]\n{ep_name} = {mod}\n")
    sys.path.insert(0, root)
    return mod


PKG = '''
name = "shop"
version = "0.1"
requires = ">=1.3,<3"


def short(value, limit=10):
    return len(value) <= limit


validators = {"short": short}
'''


def test_a_workshop_package_gives_validators_by_name_and_is_listed_to_attach():
    with Env() as e:
        _package(PKG, "shop")
        assert quiet(nodes.check, "demo.title", "a very long title", "short") == "a very long title"
        rows = [r for r in e.records() if r.get("verdict")]
        assert [r["boundary"] for r in rows] == ["node:demo.title/short"], rows
    with Env(ENTAIL_NODES="other"):
        mod = _package(PKG, "shop")
        infos = [i for i in quiet(nodes.packages) if i["entry_point"] == "shop"]
        assert not infos[0]["attached"] and mod not in sys.modules
        quiet(nodes.check, "demo.title", "long enough to fail", "short")   # no such validator here: said, not run
        assert tally.stats().get("node:demo.title/short") is None


def test_the_platform_shows_a_custom_node_and_turns_it_off():
    with Env() as e:
        v, calls = _json_object()
        quiet(nodes.check, "demo.answer", "no", v)
        nodes.check("demo.answer", '{"a": 1}', v)
        tally.write_summary({b: tally.stats(b) for b in tally.STATS})
        lines = e.records()
        g = graph.graph(lines)
        custom = [n for n in g["nodes"] if n["id"] == "node:demo.answer"]
        assert custom and custom[0]["custom"] and custom[0]["flow"] == "user" and custom[0]["state"] == "broken", g
        assert "node:demo.answer" in [f for f in g["flows"] if f["id"] == "user"][0]["nodes"]
        assert graph.engine_of("node:demo.answer/json_object") is None     # a point of the program, not an engine
        assert graph.summaries(graph.launches([("f", x) for x in lines]))[0]["engines"] == [], g
        d = graph.node_detail(lines, "node:demo.answer")
        assert d["custom"] and d["decisions"][0]["rule"].startswith("a custom node's validator"), d
        srv = server.make_server(e.dir, 0)
        t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        t.start()
        port = srv.server_address[1]
        try:
            def post(body, token):
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                c.request("POST", "/api/nodes", body=json.dumps(body).encode(),
                          headers={"Host": f"127.0.0.1:{port}", "Origin": f"http://127.0.0.1:{port}",
                                   "Content-Type": "application/json", "X-Entail-Token": token})
                r = c.getresponse()
                out = json.loads(r.read())
                c.close()
                return r.status, out

            assert post({"node": "demo.answer", "on": False}, "wrong")[0] == 403
            assert post({"node": "Bad Name", "on": False}, srv.safe.token)[0] == 400
            st, out = post({"node": "demo.answer", "on": False}, srv.safe.token)
            assert st == 200 and out["off"] == ["demo.answer"], out
            nodes._OFF["at"] = 0.0
            nodes.check("demo.answer", "no", v)             # the running program reads the switch
            assert calls == ["no", '{"a": 1}']
            st, out = post({"node": "demo.answer", "on": True}, out["token"])
            assert st == 200 and out["off"] == []
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            c.request("GET", "/api/nodes", headers={"Host": f"127.0.0.1:{port}"})
            assert json.loads(c.getresponse().read())["off"] == []
        finally:
            srv.stopping = True
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
