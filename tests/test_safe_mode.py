"""Tests for the two safety modes (ROADMAP product track P3; LIBRARY_DESIGN.md 13.6; safe_mode.py and the adapters
vllm_safe and sglang_safe on stand-in engine arguments). Pure Python. Run: python tests/test_safe_mode.py"""
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import core, load, record, safe_mode  # noqa: E402
from entail.adapters import sglang_safe, vllm_safe  # noqa: E402
from entail.contracts import RULES, Verdict  # noqa: E402
from entail.platform import graph  # noqa: E402

ALL_ON = {"cuda_graphs": True, "prefix_cache": True, "speculative_decoding": True, "custom_kernels": True}


class Env:
    """A fresh log folder and mode for one test; restores the environment after."""

    def __init__(self, safe=None):
        self.safe = safe

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in ("ENTAIL_LOG_DIR", "ENTAIL_SAFE", "ENTAIL_RECORD", "ENTAIL")}
        self.dir = tempfile.mkdtemp()
        os.environ["ENTAIL_LOG_DIR"] = self.dir
        os.environ.pop("ENTAIL_RECORD", None)
        if self.safe is None:
            os.environ.pop("ENTAIL_SAFE", None)
        else:
            os.environ["ENTAIL_SAFE"] = self.safe
        self.mode = core.mode()
        core.set_mode("load")
        safe_mode.LAST.clear()
        return self

    def __exit__(self, *exc):
        record.close_files()
        core.set_mode(self.mode)
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


def quiet(fn, *a):
    with redirect_stdout(io.StringIO()):
        return fn(*a)


def test_the_mode_comes_from_the_environment_then_the_folder_then_auto():
    with Env() as e:
        assert safe_mode.mode() == "auto"
        with open(os.path.join(e.dir, "safe_mode.json"), "w", encoding="utf-8") as f:
            json.dump({"mode": "all"}, f)
        assert safe_mode.mode() == "all"
        os.environ["ENTAIL_SAFE"] = "off"
        assert safe_mode.mode() == "off"


def test_the_explicit_mode_plans_every_optimization_that_is_on():
    with Env("all"):
        items = safe_mode.plan("vllm", "k", dict(ALL_ON, speculative_decoding=False))
        assert items == [("cuda_graphs", "all"), ("prefix_cache", "all"), ("custom_kernels", "all")], items
    with Env("off"):
        assert safe_mode.plan("vllm", "k", ALL_ON) == []
    with Env():
        assert safe_mode.plan("vllm", "k", ALL_ON) == []      # auto, nothing found yet


def test_the_selective_path_finds_the_cause_one_start_at_a_time():
    with Env() as e:
        safe_mode.started("vllm", "k1", ALL_ON, [])
        quiet(safe_mode.after_self_check, "vllm", ["decode_prefill"], "start:vllm.safe_mode", "c", "w")
        entry = safe_mode.load_store()["k1"]
        assert entry["status"] == "searching" and entry["candidates"] == ["speculative_decoding", "cuda_graphs"]
        assert safe_mode.plan("vllm", "k1", ALL_ON) == [("speculative_decoding", "path")]
        safe_mode.started("vllm", "k1", ALL_ON, ["speculative_decoding"])     # the next start, spec off
        quiet(safe_mode.after_self_check, "vllm", [], "start:vllm.safe_mode", "c", "w")
        entry = safe_mode.load_store()["k1"]
        assert entry["status"] == "found" and entry["off"] == ["speculative_decoding"]
        assert safe_mode.plan("vllm", "k1", ALL_ON) == [("speculative_decoding", "path")]   # and from then on
        said = [r for r in e.records() if r.get("said") == "start:vllm.safe_mode"]
        assert len(said) == 2 and "that is the cause" in said[-1]["text"], said


def test_the_selective_path_gives_up_when_every_candidate_was_tried():
    with Env():
        safe_mode.started("vllm", "k2", ALL_ON, [])
        quiet(safe_mode.after_self_check, "vllm", ["decode_prefill"], "start:vllm.safe_mode", "c", "w")
        safe_mode.started("vllm", "k2", ALL_ON, ["speculative_decoding"])
        assert quiet(safe_mode.after_self_check, "vllm", ["decode_prefill"], "start:vllm.safe_mode", "c", "w") == []
        assert safe_mode.plan("vllm", "k2", ALL_ON) == [("cuda_graphs", "path")]
        safe_mode.started("vllm", "k2", ALL_ON, ["cuda_graphs"])
        out = quiet(safe_mode.after_self_check, "vllm", ["decode_prefill"], "start:vllm.safe_mode", "c", "w")
        assert len(out) == 1 and out[0].verdict is Verdict.BROKEN and out[0].rule == RULES["safe_path_outside"]
        assert safe_mode.load_store()["k2"]["status"] == "outside" and safe_mode.plan("vllm", "k2", ALL_ON) == []
        assert quiet(safe_mode.after_self_check, "vllm", ["decode_prefill"], "start:vllm.safe_mode", "c", "w") == []


def test_nothing_to_turn_off_is_said_once_and_changes_nothing():
    with Env():
        safe_mode.started("vllm", "k3", {"prefix_cache": True}, [])
        assert quiet(safe_mode.after_self_check, "vllm", ["decode_prefill"], "start:vllm.safe_mode", "c", "w") == []
        assert safe_mode.load_store()["k3"]["status"] == "outside" and safe_mode.plan("vllm", "k3", ALL_ON) == []


def _vllm_args():
    return SimpleNamespace(model="/m/qwen", dtype="auto", quantization=None, tensor_parallel_size=1,
                           enforce_eager=False, enable_prefix_caching=None, speculative_config={"method": "ngram"},
                           compilation_config=SimpleNamespace(custom_ops=[]))


def test_the_vllm_adapter_turns_the_options_off():
    with Env("all") as e:
        args = _vllm_args()
        ds = quiet(vllm_safe._decide, args)
        assert [d.verdict for d in ds] == [Verdict.RESOLVED] * 4 and ds[0].rule == RULES["safe_mode"]
        assert args.enforce_eager is True and args.enable_prefix_caching is False
        assert args.speculative_config is None and args.compilation_config.custom_ops == ["none"]
        rows = [r for r in e.records() if r.get("boundary") == "start:vllm.safe_mode"]
        assert len(rows) == 4 and rows[0]["resolution"].startswith("enforce_eager") and rows[0]["v"] == 2
        assert graph.node_of("start:vllm.safe_mode") == "self_check"
        assert safe_mode.LAST["vllm"]["off"] == ["cuda_graphs", "prefix_cache", "speculative_decoding",
                                                 "custom_kernels"]
    with Env():
        args = _vllm_args()
        key, enabled = vllm_safe.read_choice(args)
        assert enabled == ALL_ON
        safe_mode.save_store({key: {"status": "found", "off": ["speculative_decoding"], "candidates": [],
                                    "tried": []}})
        ds = quiet(vllm_safe._decide, args)
        assert len(ds) == 1 and ds[0].rule == RULES["safe_path"] and args.speculative_config is None
        assert args.enforce_eager is False and args.compilation_config.custom_ops == []   # the rest untouched


def test_the_sglang_adapter_turns_the_options_off():
    with Env("all"):
        args = SimpleNamespace(model_path="/m/q", dtype="auto", quantization=None, tp_size=1,
                               disable_cuda_graph=False, disable_radix_cache=False, speculative_algorithm="EAGLE")
        ds = quiet(sglang_safe._decide, args)
        assert len(ds) == 3 and args.disable_cuda_graph is True and args.disable_radix_cache is True
        assert args.speculative_algorithm is None
    with Env("all"):
        already = SimpleNamespace(model_path="/m/q", dtype="auto", quantization=None, tp_size=1,
                                  disable_cuda_graph=True, disable_radix_cache=True, speculative_algorithm=None)
        assert quiet(sglang_safe._decide, already) == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
