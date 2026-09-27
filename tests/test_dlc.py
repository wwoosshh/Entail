"""Tests for official DLCs (ROADMAP product track P4; LIBRARY_DESIGN.md 13.7; entail/dlc.py): found through the entry
point group entail.dlc, attached only in their core range, installed the core's way (a failure is recorded and the
program goes on), their entries in the start-up hook's table and their nodes in the platform. The DLCs here are
stand-in distributions written to a temporary folder on sys.path. Pure Python. Run: python tests/test_dlc.py"""
import importlib.util
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import __version__, core, dlc, record  # noqa: E402
from entail.platform import graph  # noqa: E402

SHIM = os.path.join(os.path.dirname(HERE), "entail", "adapters", "autoinstall", "sitecustomize.py")
_N = [0]


def _dist(module_text, ep_name="fake"):
    """A stand-in distribution: a module and its dist-info with one entail.dlc entry point, on sys.path."""
    _N[0] += 1
    root = tempfile.mkdtemp()
    mod = f"fake_dlc_{_N[0]}"
    with open(os.path.join(root, mod + ".py"), "w", encoding="utf-8") as f:
        f.write(module_text.replace("MOD", mod))
    info = os.path.join(root, f"{mod}-0.1.dist-info")
    os.makedirs(info)
    with open(os.path.join(info, "METADATA"), "w", encoding="utf-8") as f:
        f.write(f"Metadata-Version: 2.1\nName: {mod}\nVersion: 0.1\n")
    with open(os.path.join(info, "entry_points.txt"), "w", encoding="utf-8") as f:
        f.write(f"[entail.dlc]\n{ep_name} = {mod}\n")
    sys.path.insert(0, root)
    return root, mod


GOOD = '''
name = "fake"
version = "0.1"
requires = ">=1.3,<3"
engines = {"fakeengine": "1.0"}
targets = {"fake_engine_mod": ["MOD:install_ok", "MOD:install_bad"]}
facts = ()
nodes = [{"id": "fake_node", "flow": "image", "step": 5, "ko": "가짜", "en": "Fake", "patterns": ["^dlc:fake\\\\."]}]
CALLS = []


def install_ok():
    CALLS.append("ok")
    return 1


def install_bad():
    CALLS.append("bad")
    raise RuntimeError("boom")
'''


class Env:
    def __init__(self, **env):
        self.env = env

    def __enter__(self):
        keys = ("ENTAIL_LOG_DIR", "ENTAIL_DLC", "ENTAIL_RECORD")
        self.old = {k: os.environ.get(k) for k in keys}
        self.dir = tempfile.mkdtemp()
        os.environ["ENTAIL_LOG_DIR"] = self.dir
        os.environ.pop("ENTAIL_RECORD", None)
        os.environ.pop("ENTAIL_DLC", None)
        os.environ.update(self.env)
        self.mode = core.mode()
        core.set_mode("load")
        self.paths = list(sys.path)
        dlc.reset()
        return self

    def __exit__(self, *exc):
        record.close_files()
        core.set_mode(self.mode)
        sys.path[:] = self.paths
        dlc.reset()
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


def test_versions_are_compared_by_their_numbers():
    assert dlc.version_ok(">=1.3,<3", "1.3.0") and dlc.version_ok(">=1.3,<3", "2.0.0rc1")
    assert not dlc.version_ok(">=1.3,<3", "1.2.9") and not dlc.version_ok(">=1.3,<3", "3.0")
    assert dlc.version_ok("", "0.1") and dlc.version_ok("==1.3", "1.3.0") and not dlc.version_ok("!=1.3", "1.3")
    assert not dlc.version_ok("~=1.3", "1.3")             # a comparison it does not know holds nothing


def test_a_dlc_is_found_attached_and_gives_its_entries_and_nodes():
    with Env():
        _, mod = _dist(GOOD)
        found = [i for i in quiet(dlc.found) if i.get("name") == "fake"]
        assert len(found) == 1 and found[0]["attached"] and found[0]["version"] == "0.1", found
        assert dlc.targets()["fake_engine_mod"] == [(f"{mod}:install_ok", "fake"), (f"{mod}:install_bad", "fake")]
        mine = [n for n in dlc.nodes() if n["dlc"] == "fake"]      # (a real DLC installed here may add its own)
        assert [n["id"] for n in mine] == ["fake_node"], mine
        assert not [n for n in quiet(dlc.nodes, core_ids={"fake_node"}) if n["dlc"] == "fake"]   # a taken id is left out
    with Env(ENTAIL_DLC="off"):
        _dist(GOOD)
        assert dlc.found() == [] and dlc.targets() == {}


def test_a_dlc_outside_its_core_range_or_with_foreign_facts_is_not_attached_and_says_why():
    with Env() as e:
        _dist(GOOD.replace('requires = ">=1.3,<3"', 'requires = ">=9"'))
        _dist(GOOD.replace('name = "fake"', 'name = "fake2"').replace("facts = ()", 'facts = ("NoSuchFact",)'))
        found = {i["name"]: i for i in quiet(dlc.found) if i.get("name") in ("fake", "fake2")}
        assert not found["fake"]["attached"] and f"this is {__version__}" in found["fake"]["why"], found
        assert not found["fake2"]["attached"] and "vocabulary" in found["fake2"]["why"], found
        assert not any(name in ("fake", "fake2") for entries in dlc.targets().values() for _, name in entries)
        said = [r for r in e.records() if str(r.get("said", "")).startswith("dlc:")]
        assert {r["said"] for r in said} >= {"dlc:fake", "dlc:fake2"} and all("not attached" in r["text"] for r in said)


def test_the_core_installs_an_entry_and_a_failure_does_not_reach_the_program():
    with Env() as e:
        _, mod = _dist(GOOD)
        quiet(dlc.found)
        assert quiet(dlc.install, "fake", f"{mod}:install_ok") == 1
        assert quiet(dlc.install, "fake", f"{mod}:install_bad") == 0       # raised inside: recorded, not raised
        assert quiet(dlc.install, "fake", f"{mod}:install_bad") == 0
        assert quiet(dlc.install, "fake", f"{mod}:install_bad") == 0       # left out after two failures
        assert sys.modules[mod].CALLS == ["ok", "bad", "bad"]
        said = [r["text"] for r in e.records() if r.get("said") == "dlc:fake.install"]
        assert len(said) == 2 and "RuntimeError: boom" in said[0] and "left out" in said[1], said
    with Env() as e3:                                       # a DLC that decides a fact: unknown for it
        _, mod = _dist(GOOD.replace("facts = ()", 'facts = ("SafeMode",)'))
        quiet(dlc.found)
        quiet(dlc.install, "fake", f"{mod}:install_bad")
        rows = [r for r in e3.records() if r.get("boundary") == "dlc:fake.install"]
        assert len(rows) == 1 and rows[0]["verdict"] == "unknown" and rows[0]["name"] == "SafeMode", rows
    with Env():
        _, mod = _dist(GOOD)
        quiet(dlc.found)
        core.set_mode("debug")
        try:
            quiet(dlc.install, "fake", f"{mod}:install_bad")
            raise AssertionError("debug mode must raise")
        except RuntimeError:
            pass


def test_the_platform_shows_a_dlc_node_before_the_catch_all():
    with Env():
        _dist(GOOD)
        quiet(dlc.found)
        graph.reset_model()
        try:
            m = graph.model()
            ids = [n["id"] for n in m["nodes"]]
            assert "fake_node" in ids and ids.index("fake_node") < ids.index("other"), ids
            assert graph.node_of("dlc:fake.schedule") == "fake_node" and graph.node_of("load:vllm.attention") == "rotary"
        finally:
            graph.reset_model()


def test_the_start_up_hook_puts_a_dlc_s_entries_in_its_table():
    with Env():
        _, mod = _dist(GOOD)
        spec = importlib.util.spec_from_file_location("sitecustomize_dlc_test", SHIM)
        shim = importlib.util.module_from_spec(spec)
        old = os.environ.get("ENTAIL")
        os.environ["ENTAIL"] = "off"                        # build the table only; nothing is activated
        try:
            spec.loader.exec_module(shim)
        finally:
            if old is None:
                os.environ.pop("ENTAIL", None)
            else:
                os.environ["ENTAIL"] = old
        assert "fake_engine_mod" not in shim.TARGETS        # entail off: DLCs are not even looked up
        quiet(shim.add_dlcs)
        assert shim.TARGETS["fake_engine_mod"] == [f"{mod}:install_ok", f"{mod}:install_bad"]
        assert shim.DLC_OF[f"{mod}:install_ok"] == "fake"
        quiet(shim._install, f"{mod}:install_bad")         # through the core's wrapper: recorded, not raised
        assert sys.modules[mod].CALLS == ["bad"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
