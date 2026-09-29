"""Tests for the notes on inference engines entail does not watch (entail/unwatched.py; field test, entail#38): an
engine's module in the process gives one unknown at engine:<module>.unwatched, once per process, and the page puts
it in its own node. Stand-in modules only; no engine is imported. Run: python tests/test_unwatched.py"""
import io
import os
import sys
import types
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ["ENTAIL_LOG_DIR"] = "off"
from entail import core, load, unwatched  # noqa: E402
from entail.contracts import RULES, Verdict  # noqa: E402
from entail.platform import graph  # noqa: E402


def noted(mode="load", **modules):
    """unwatched.noted() with stand-in modules in sys.modules, in `mode`; (how many, the decisions, what it
    printed)."""
    was = {n: sys.modules.get(n) for n in modules}
    for n, version in modules.items():
        m = types.ModuleType(n)
        if version is not None:
            m.__version__ = version
        sys.modules[n] = m
    core.set_mode(mode)
    k = len(load.LEDGER.decisions)
    out = io.StringIO()
    try:
        with redirect_stdout(out):
            n = unwatched.noted()
    finally:
        core.set_mode("off")
        for name, m in was.items():
            if m is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = m
    return n, load.LEDGER.decisions[k:], out.getvalue()


def test_an_engine_entail_does_not_watch_is_said_once_per_process():
    """WhisperX's CTranslate2 (field test, #38): one unknown that names the engine and its version, and the run goes
    on; a second import in the same process says nothing more."""
    unwatched.reset()
    n, ds, printed = noted(ctranslate2="4.8.2")
    assert n == 1 and len(ds) == 1 and ds[0].verdict is Verdict.UNKNOWN and not ds[0].blocking, ds
    d = ds[0]
    assert d.contract.boundary == "engine:ctranslate2.unwatched" and d.rule == RULES["cannot_check"], d
    assert "this process loaded CTranslate2 4.8.2 (ctranslate2), which entail does not watch" in d.note, d.note
    assert "unknown at engine:ctranslate2.unwatched" in printed, printed
    n, ds, _ = noted(ctranslate2="4.8.2")
    assert n == 0 and ds == [], "once per process"
    unwatched.reset()
    n, ds, _ = noted(mode="debug", ctranslate2="4.8.2")   # debug stops at an unknown, not inside the import
    assert n == 1 and not ds[0].blocking, ds
    unwatched.reset()


def test_every_engine_of_the_table_present_is_said_and_nothing_else():
    """MinerU (#38) runs its document model with llama.cpp next to transformers; a module without a version is named
    without one, and a module that is not in the table is not said."""
    unwatched.reset()
    n, ds, _ = noted(llama_cpp=None, onnxruntime="1.22.0", some_library="1.0")
    assert n == 2 and [d.contract.boundary for d in ds] == ["engine:llama_cpp.unwatched",
                                                            "engine:onnxruntime.unwatched"], ds
    assert "loaded llama.cpp (llama-cpp-python) (llama_cpp)" in ds[0].note and "ONNX Runtime 1.22.0" in ds[1].note
    unwatched.reset()


def test_the_page_shows_it_in_a_node_of_its_own():
    """A run whose only record is this line (WhisperX on English audio, #38: nothing else reached) is listed, with
    the engine in its title and one node that could not be checked."""
    line = {"v": 2, "t": 1.0, "run": "1-1", "pid": 1, "boundary": "engine:ctranslate2.unwatched",
            "consumer": "ctranslate2", "name": "Coverage", "verdict": "unknown", "blocking": False,
            "rule": RULES["cannot_check"], "note": "this process loaded CTranslate2 4.8.2 (ctranslate2), which entail "
                                                   "does not watch: what it computes is not among these results"}
    assert graph.node_of(line["boundary"]) == "unwatched"
    g = graph.graph([line])
    assert [f["id"] for f in g["flows"]] == ["outside"] and g["engines"] == ["ctranslate2"], g["flows"]
    node, = g["nodes"]
    assert node["id"] == "unwatched" and node["state"] == "unknown" and node["ko"] == "보지 않는 엔진", node


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
