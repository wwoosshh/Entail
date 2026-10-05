"""Tests for memory nothing wrote (ROADMAP M19 L7, entail/lifetime.py): a model made and loaded inside the load
window, with parameters a loader fills, partly fills or leaves alone; initializers and in-place copies count as
writes; integer and FP8 parameters; the report when the holding module runs, when a Triton launch reads the tensor,
when a compiled graph takes it; the stop policy. torch (CPU) only.
Run: python tests/test_lifetime.py
"""
import os
import sys
import tempfile

import torch
import torch.fx as fx

os.environ.setdefault("ENTAIL_LOG_DIR", tempfile.mkdtemp(prefix="entail_lifetime_test_"))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import graph_types as GT  # noqa: E402
from entail import kernel_check as KC  # noqa: E402
from entail import lifetime as LT  # noqa: E402


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Parameter(torch.empty(12, 8), requires_grad=False)       # loaded below
        self.proj = torch.nn.Parameter(torch.empty(8, 8), requires_grad=False)       # half its rows loaded
        self.bias = torch.nn.Parameter(torch.empty(8), requires_grad=False)          # not in the checkpoint
        self.norm = torch.nn.LayerNorm(8)                                            # its own initializer writes
        self.scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)         # made with a value
        self.packed = torch.nn.Parameter(torch.empty(4, 64, dtype=torch.int32), requires_grad=False)
        self.codes = torch.nn.Parameter(torch.empty(4, 64, dtype=torch.uint8), requires_grad=False)

    def forward(self, x):
        return self.norm(x @ self.qkv[:8].t() @ self.proj.t() + self.bias) * self.scale


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()
        self.head = torch.nn.Linear(8, 4)


def load(model):
    """A loader: copies what its checkpoint has into the parameters (through .data, which torch's version counter
    does not see, and in place)."""
    g = torch.Generator().manual_seed(0)
    model.block.qkv.data.copy_(torch.randn(12, 8, generator=g))
    model.block.proj.data[:4].copy_(torch.randn(4, 8, generator=g))
    model.block.packed.data.copy_(torch.randint(-2**31, 2**31 - 1, (4, 64), generator=g, dtype=torch.int64)
                                  .to(torch.int32))
    codes = torch.randint(0, 256, (4, 64), generator=g, dtype=torch.int64).to(torch.uint8)
    codes[0, 3] = 0xA5                       # a loaded byte that happens to equal the mark, alone: not a run
    model.block.codes.data.copy_(codes)


def main():
    before = LT._STATE["marked"]
    t = torch.empty(4)
    assert LT.unwritten(t) == 0 and LT._STATE["marked"] == before
    print("ok outside the window torch.empty is left as it is")

    with LT.load_window():
        model = Model()
        load(model)
    found = {name: (n, total) for name, n, total in LT.loaded(model)}
    assert found == {"block.proj": (32, 64), "block.bias": (8, 8)}, found
    print("ok after loading: the bias nothing loaded (8 of 8) and the half-loaded matrix (32 of 64) hold elements "
          "nothing wrote; loaded, initialized and made-with-a-value parameters do not:", found)

    with LT.load_window():
        m2 = Model()
        m2.block.packed.data[:2].copy_(torch.zeros(2, 64, dtype=torch.int32))
        f8 = torch.empty(16, 16, dtype=torch.float8_e4m3fn) if hasattr(torch, "float8_e4m3fn") else None
    n_int = LT.unwritten(m2.block.packed)
    assert n_int == 128, n_int
    assert LT.unwritten(m2.block.codes) == 256
    print("ok integer parameters: the unwritten 64-byte runs are counted (128 of 256 int32, 256 of 256 uint8)")
    if f8 is not None:
        assert LT.unwritten(f8) == 256
        f8[:8].copy_(torch.zeros(8, 16).to(torch.float8_e4m3fn))
        assert LT.unwritten(f8) == 128
        print("ok FP8: 256 unwritten, 128 after half of it is written")

    with LT.load_window():
        with LT.load_window():
            inner = Model()
        assert LT.loaded(inner) == [], "an inner window decides nothing: the outer one does"
    print("ok nested windows count as one")

    # the reader: the module that holds the parameter, when it runs
    reported = KC.stats().get("broken_reported", 0)
    x = torch.randn(2, 8)
    model.block(x)
    assert LT.stats().get("unwritten_read") == 2, LT.stats()
    assert KC.stats().get("broken_reported", 0) == reported + 2
    model.block(x)
    assert LT.stats().get("unwritten_read") == 2, "reported once per module"
    print("ok the module runs: its two parameters are reported as broken, once")

    os.environ["ENTAIL_ON_BROKEN"] = "stop"
    try:
        with LT.load_window():
            m3 = Model()
        LT.loaded(m3)
        try:
            m3.block(x)
            raise AssertionError("the stop policy should refuse the read")
        except KC.Broken as e:
            assert "never written" in str(e), e
            print("ok ENTAIL_ON_BROKEN=stop: the module is stopped before it runs:", str(e)[:100])
    finally:
        os.environ.pop("ENTAIL_ON_BROKEN", None)

    # a read while the engine assembles its model (M22.2): a trial call on a part the engine then replaces is
    # withdrawn, not reported; a part the model still holds is reported when the assembly ends
    reported = KC.stats().get("broken_reported", 0)
    withdrawn = LT.stats().get("unwritten_read_withdrawn", 0)
    with LT.assembly():
        with LT.load_window():
            m4 = Model()
            load(m4)
        LT.loaded(m4)
        m4.block(x)                                   # the trial call: its result is dropped
        assert KC.stats().get("broken_reported", 0) == reported, "held while the model is assembled"
        m4.block = Block()                            # the engine wires another part in its place
    assert KC.stats().get("broken_reported", 0) == reported, "a replaced part's trial read is not reported"
    assert LT.stats().get("unwritten_read_withdrawn", 0) == withdrawn + 2, LT.stats()
    print("ok a trial read while the engine assembles the model, of a part it then replaces: withdrawn")

    with LT.assembly():
        with LT.load_window():
            m5 = Model()
            load(m5)
        LT.loaded(m5)
        m5.block(x)
    assert KC.stats().get("broken_reported", 0) == reported + 2, KC.stats()
    print("ok the same read of a part the model keeps: reported when the assembly ends")

    # the reader: a Triton launch handed the tensor
    f = KC.fact_of(model.block.bias)
    assert f is not None and f.get("unwritten") == 8, f
    bad = KC._life_check("k", {"b": model.block.bias}, {"b": f}, set())
    assert bad and "never written" in bad[0], bad
    assert KC._life_check("k", {"b": model.block.bias}, {"b": f}, {"b"}) == [], "a launch that writes it reads nothing"
    assert KC._life_check("k", {"w": model.block.qkv}, {"w": KC.fact_of(model.block.qkv)}, set()) == []
    print("ok a launch that reads the bias: violation:", bad[0])

    # the reader: a compiled graph that takes it as an input
    g = fx.Graph()
    xin, bin_ = g.placeholder("x"), g.placeholder("b")
    xt = torch.zeros(2, 8)
    xin.meta["example_value"], bin_.meta["example_value"] = xt, model.block.bias
    out = g.call_function(torch.add, (xin, bin_))
    out.meta["example_value"] = xt
    g.output(out)
    v = GT.check_graph(fx.GraphModule(torch.nn.Module(), g), [xt, model.block.bias], KC.fact_of)
    assert v["verdict"] == "violation" and "never written" in v["violations"][0]["why"], v
    print("ok a graph that adds the bias: violation:", v["violations"][0]["why"])

    # a compiled model from the engine's cache would run with no reader seen: it is declined, so the graph is checked
    import types
    fake = types.ModuleType("vllm.compilation.decorators")
    fake._try_load_aot_compiled_fn = lambda self, path, *a, **k: "compiled"
    sys.modules["vllm.compilation.decorators"] = fake
    try:
        assert KC.install_compile_cache() == 1
        assert KC._UNWRITTEN[0] > 0
        assert fake._try_load_aot_compiled_fn(object(), "p") is None
        KC._UNWRITTEN[0], held = 0, KC._UNWRITTEN[0]
        KC._CACHE = {}
        assert fake._try_load_aot_compiled_fn(type("S", (), {"vllm_config": None})(), "p") is None, \
            "no verdict cached: declined once, as before"
        KC._UNWRITTEN[0] = held
    finally:
        sys.modules.pop("vllm.compilation.decorators", None)
    print("ok with parameters nothing wrote, a compiled model from the cache is declined (its graph is checked)")


if __name__ == "__main__":
    main()
    print("all ok")
