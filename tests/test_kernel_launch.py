"""Tests for the kernel-launch contract in the core (ROADMAP M17.3): a tensor strided in its innermost dimension
handed to a kernel that names no stride argument (broken), one that does but was not told this stride (unknown),
one that was told (pass); expanded views and size-1 dims left alone; the Triton adapter's binding, layout memo and
engine attribution on a fake JITFunction; the retrospective on sglang#21843's tensors. Needs torch (CPU).
Run: python tests/test_kernel_launch.py"""
import io
import os
import sys
from contextlib import redirect_stdout
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import core, kernel_launch_contract as klc, load  # noqa: E402
from entail.contracts import RULES, Verdict  # noqa: E402

try:
    import torch
except ImportError:      # the contract is exercised through torch strides; without torch there is nothing to test
    torch = None


def decided(fn):
    n = len(load.LEDGER.decisions)
    was = core.mode()
    core.set_mode("load")
    try:
        with redirect_stdout(io.StringIO()):
            out = fn()
    finally:
        core.set_mode(was)
    return out, load.LEDGER.decisions[n:]


def test_the_innermost_strided_dimension_is_found_and_expanded_or_size_one_dims_are_not():
    if torch is None:
        return
    a = torch.randn(8, 16)
    assert klc.innermost_stride(a) == (1, 16, 1)
    assert klc.innermost_stride(a.t()) == (1, 8, 16)
    assert klc.innermost_stride(a[:, :1]) == (0, 8, 16)            # size-1 innermost: the next dim counts
    assert klc.innermost_stride(torch.randn(1, 1)) is None
    assert klc.classify({"x": torch.randn(4).expand(3, 4)}) == []   # stride 0 is an expanded view


def test_a_kernel_without_stride_arguments_is_broken_with_them_unknown_or_told():
    if torch is None:
        return
    mixed = torch.randn(8, 32)
    b, a = torch.split(mixed.view(8, 16, 2), [1, 1], dim=2)
    a = a.reshape(8, 16)                                          # sglang#21843: strides (32, 2)
    assert a.stride() == (32, 2)
    out = klc.check("kernel:test.k", "test.k", "k", {"A_log": torch.randn(16), "a": a, "dt_bias": torch.randn(16)},
                    "k launch", record=False)
    assert len(out) == 1 and out[0].verdict is Verdict.BROKEN and out[0].rule == RULES["kernel_stride_assumed"], out
    assert "takes no stride argument" in out[0].note
    # sglang 0.5.20's kernel: told the row strides only -> the innermost stride 2 is among no integer argument
    out = klc.check("kernel:test.k", "test.k", "fused_gdn_gating",
                    {"A_log": torch.randn(16), "a": a, "b": b.reshape(8, 16), "dt_bias": torch.randn(16), "seq_len": 1,
                     "stride_a": 32, "stride_b": 32, "NUM_HEADS": 16, "beta": 1.0, "threshold": 20.0, "BLK_HEADS": 8},
                    "k launch", record=False)
    assert [d.verdict for d in out] == [Verdict.UNKNOWN, Verdict.UNKNOWN] and "cannot be told" in out[0].note, out
    # told the innermost stride -> pass
    out = klc.check("kernel:test.k", "test.k", "k", {"a": a, "stride_am": 32, "stride_ak": 2}, "k launch",
                    record=False)
    assert [d.verdict for d in out] == [Verdict.PASS] and "strides passed" in out[0].note
    # contiguous tensors -> pass, nothing to say
    out = klc.check("kernel:test.k", "test.k", "k", {"a": mixed, "n": 3}, "k launch", record=False)
    assert [d.verdict for d in out] == [Verdict.PASS] and out[0].note == ""


def test_the_triton_adapter_binds_arguments_memoises_layouts_and_names_the_engine():
    if torch is None:
        return
    from entail.adapters import triton_launch

    def fused_gdn_gating():
        pass

    fused_gdn_gating.__module__ = "sglang.kernels.ops.attention.fla.fused_gdn_gating"
    fn = SimpleNamespace(params=[SimpleNamespace(name=n) for n in ("g", "a", "b", "stride_a", "NUM_HEADS")],
                         fn=fused_gdn_gating)
    a = torch.randn(8, 32)[:, ::2]
    bound = triton_launch.read_choice(fn, (torch.empty(8, 16), a), {"b": a, "stride_a": 32, "NUM_HEADS": 16})
    assert list(bound) == ["g", "a", "b", "stride_a", "NUM_HEADS"] and bound["stride_a"] == 32
    assert triton_launch._where(fn) == ("sglang", "fused_gdn_gating", fused_gdn_gating.__module__)
    k1 = triton_launch._layout_key(fn, (torch.empty(8, 16), a), {"b": a})
    k2 = triton_launch._layout_key(fn, (torch.empty(8, 16), a.clone()[:, :]), {"b": a})
    assert k1 != k2 and triton_launch._layout_key(fn, (torch.empty(8, 16), a), {"b": a}) == k1
    _, rec = decided(lambda: triton_launch._decide(fn, (torch.empty(8, 16), a), {"b": a, "stride_a": 32,
                                                                                 "NUM_HEADS": 16}, k1))
    got = [d for d in rec if d.contract.boundary == "kernel:sglang.fused_gdn_gating"]
    assert got and all(d.verdict is Verdict.UNKNOWN for d in got), rec
    assert list(triton_launch.handles(fn)) == ["relay"], "the one repair: relaid copies (M19 L3.3b)"


def test_stride_like_names_and_a_stride_told_by_value_are_not_broken():
    """Review finding 1: vLLM's sparse indexer names its strides q_s0/q_s1, mxfp8 sxm/sxk, SGLang a_s0/sq_d; a
    kernel told the innermost stride under any name passes, one with stride-like names but not this value is
    unknown, and broken is kept for no stride-like name and no equal integer. Finding 2: integers count among the
    kernel's value parameters only (constexprs and launch options are not strides)."""
    if torch is None:
        return
    a = torch.randn(8, 32)[:, ::2]                                   # strides (32, 2)
    assert all(klc.stride_like(n) for n in ("stride_am", "q_s0", "weights_s1", "sxm", "sq_d", "s_k", "lda", "ld_a"))
    assert not any(klc.stride_like(n) for n in ("a", "seq_len", "NUM_HEADS", "scale", "softmax_scale", "seqlen_q"))
    out = klc.check("kernel:test.k", "test.k", "k", {"a": a, "q_s0": 32, "q_s1": 2}, "k launch", record=False)
    assert [d.verdict for d in out] == [Verdict.PASS], out                # told under a name without "stride"
    out = klc.check("kernel:test.k", "test.k", "k", {"a": a, "q_s0": 32, "q_s1": 4}, "k launch", record=False)
    assert [d.verdict for d in out] == [Verdict.UNKNOWN] and "q_s0, q_s1" in out[0].note
    out = klc.check("kernel:test.k", "test.k", "k", {"a": a, "n": 32, "inner": 2}, "k launch", record=False)
    assert [d.verdict for d in out] == [Verdict.PASS], "an integer equal to the stride is told, whatever its name"
    out = klc.check("kernel:test.k", "test.k", "k", {"a": a, "n": 32, "BLOCK": 2}, "k launch", record=False,
                    ints_from={"a", "n"})
    assert out[0].verdict is Verdict.BROKEN and "no integer argument equals 2" in out[0].note, out
    out = klc.check("kernel:test.k", "test.k", "k", {"a": a, "stride_a": 32, "num_warps": 2}, "k launch",
                    record=False, ints_from={"a", "stride_a"})
    assert [d.verdict for d in out] == [Verdict.UNKNOWN], "a launch option equal to the stride does not tell it"


def test_the_hook_decides_each_stride_pattern_once_and_caps_strided_patterns_per_kernel():
    """Review finding 3: the memo key is shape-free (rank and innermost stride per tensor), so a server's decode
    shapes share one pattern, a strided tensor at a new shape is still seen, and only strided patterns count
    toward the cap."""
    if torch is None:
        return
    from entail.adapters import triton_launch

    class FakeJIT:
        params = [SimpleNamespace(name="x", is_constexpr=False), SimpleNamespace(name="BLOCK", is_constexpr=True),
                  SimpleNamespace(name="stride_x_token", is_constexpr=True)]

        def fn(self):
            pass

    triton_launch.reset()
    try:
        jit = FakeJIT()
        assert triton_launch.value_params(jit) == ["x"]
        # a stride declared constexpr (causal_conv1d) still tells the stride; BLOCK does not (E2 on 99a4841 said
        # unknown twice on Nemotron's causal_conv1d launches because every constexpr was dropped)
        assert triton_launch.told_params(jit) == ["x", "stride_x_token"]
        x = torch.randn(11, 9728).t()                                    # (9728, 11) strides (1, 9728)
        out = klc.check("kernel:test.k", "test.k", "k", {"x": x, "BLOCK": 16, "stride_x_token": 9728}, "k launch",
                        record=False, ints_from=set(triton_launch.told_params(jit)))
        assert [d.verdict for d in out] == [Verdict.PASS], out
        looked = [triton_launch.should_look(jit, (torch.empty(2, n + 2),), {"BLOCK": 16}) for n in range(20)]
        assert sum(k is not None for k in looked) == 1, "contiguous launches of 20 shapes: one pattern"
        assert triton_launch._COUNT.get(id(jit), 0) == 0, "a contiguous pattern does not count toward the cap"
        big = torch.randn(4, 64)
        strided = [triton_launch.should_look(jit, (big[:, ::s],), {}) for s in range(2, 2 + triton_launch.LIMIT + 4)]
        assert sum(k is not None for k in strided) == triton_launch.LIMIT
        assert triton_launch._COUNT[id(jit)] == triton_launch.LIMIT
        again = triton_launch.should_look(jit, (torch.randn(16, 64)[:, ::2],), {})
        assert again is None, "the same strided pattern at another shape was decided already"
        assert triton_launch.stats()["kernels_seen"] == 1 and triton_launch.stats()["patterns_decided"] == 1 + triton_launch.LIMIT
    finally:
        triton_launch.reset()


def _gating(knows_inner=False, strides_told=True):
    """A launch emulated in torch as a Triton kernel reads memory: `a` through its pointer, the row stride it is told
    (or the row length when it is told none) and an inner stride of 1 unless it knows a's; it writes 2 * a into g."""
    def launch(args, kwargs):
        g, a = args[0], args[1]
        row = kwargs["stride_a"] if strides_told else a.shape[1]
        inner = a.stride(1) if knows_inner else 1
        g.copy_(torch.as_strided(a, a.shape, (row, inner), a.storage_offset()) * 2)
    return launch


def _fake(names):
    def gate():
        pass

    gate.__module__ = "fake_engine.kernels"
    return SimpleNamespace(params=[SimpleNamespace(name=n, is_constexpr=False) for n in names], fn=gate)


def test_a_launch_the_layout_of_which_cannot_be_told_is_run_twice_and_repaired():
    """M19 L3.3b: sglang#21843's shape, a strided innermost dimension the kernel is not told - run as given and with
    the tensor relaid (inner stride 1, the row stride it is told kept): the values differ, so the kernel reads the
    tensor as if it were contiguous; every later launch of that pattern runs on the relaid copy, written back."""
    if torch is None:
        return
    from entail.adapters import triton_launch as tl

    tl.reset()
    fn = _fake(["g", "a", "stride_a"])
    a = torch.randn(8, 32)[:, ::2]                           # strides (32, 2)
    g = torch.zeros(8, 16)
    key = tl._layout_key(fn, (g, a), {"stride_a": 32})
    _, ds = decided(lambda: tl._decide(fn, (g, a), {"stride_a": 32}, key, _gating()))
    got = [d for d in ds if d.contract.boundary == "kernel:fake_engine.gate"]
    assert len(got) == 1 and got[0].verdict is Verdict.RESOLVED and "reads a as if it were contiguous" in got[0].note, ds
    assert int(g.abs().sum()) == 0, "the comparison ran on copies: the caller's output is untouched"
    assert key in tl._REPAIR and tl.stats()["run_twice"] == 1
    tl.launch_relaid(_gating(), (g, a), {"stride_a": 32}, *tl._REPAIR[key])
    assert torch.equal(g, a * 2), "the repaired launch writes what a means"
    tl.reset()
    g = torch.zeros(8, 16)
    _, ds = decided(lambda: tl._decide(fn, (g, a), {"stride_a": 32}, key, _gating(knows_inner=True)))
    got = [d for d in ds if d.contract.boundary == "kernel:fake_engine.gate"]
    assert len(got) == 1 and got[0].verdict is Verdict.PASS and "reads a right" in got[0].note, ds
    assert key not in tl._REPAIR
    tl.reset()


def test_two_read_only_halves_of_one_buffer_are_copied_apart_and_repaired():
    """sglang#21843 exactly: a and b are every other column of one projection (one storage, only read); the kernel
    writes g. Sharing storage between tensors that are only read does not stop the comparison."""
    if torch is None:
        return
    from entail.adapters import triton_launch as tl

    def gdn(args, kwargs):
        g, a, b = args
        s = kwargs["stride_a"]
        ra = torch.as_strided(a, a.shape, (s, 1), a.storage_offset())
        rb = torch.as_strided(b, b.shape, (s, 1), b.storage_offset())
        g.copy_(ra + 2 * rb)

    tl.reset()
    fn = _fake(["g", "a", "b", "stride_a"])
    buf = torch.randn(8, 32)
    a, b = buf[:, 0::2], buf[:, 1::2]
    g = torch.zeros(8, 16)
    key = tl._layout_key(fn, (g, a, b), {"stride_a": 32})
    _, ds = decided(lambda: tl._decide(fn, (g, a, b), {"stride_a": 32}, key, gdn))
    got = [d for d in ds if d.contract.boundary == "kernel:fake_engine.gate"]
    assert len(got) == 1 and got[0].verdict is Verdict.RESOLVED, ds
    relay, written = tl._REPAIR[key]
    assert set(relay) == {("a", 1), ("a", 2)} and written == [], "a and b are read, not written: nothing copied back"
    tl.launch_relaid(gdn, (g, a, b), {"stride_a": 32}, relay, written)
    assert torch.allclose(g, a + 2 * b) and torch.equal(buf[:, 0::2], a), "the repaired launch; the inputs untouched"
    tl.reset()


def test_a_kernel_told_no_stride_is_relaid_wholly_contiguous():
    if torch is None:
        return
    from entail.adapters import triton_launch as tl

    tl.reset()
    fn = _fake(["g", "a"])
    a = torch.randn(8, 32)[:, ::2]
    g = torch.zeros(8, 16)
    key = tl._layout_key(fn, (g, a), {})
    _, ds = decided(lambda: tl._decide(fn, (g, a), {}, key, _gating(strides_told=False)))
    got = [d for d in ds if d.contract.boundary == "kernel:fake_engine.gate"]
    assert got[0].verdict is Verdict.RESOLVED and "wholly contiguous" in got[0].resolution, ds
    assert "worst ratio to the allowance 0;" not in got[0].note, "exact zeros must not hide the ratio (0/0)"
    tl.launch_relaid(_gating(strides_told=False), (g, a), {}, *tl._REPAIR[key])
    assert torch.equal(g, a * 2)
    tl.reset()


def test_launches_that_cannot_be_run_twice_are_decided_from_the_launch_alone():
    if torch is None:
        return
    from entail.adapters import triton_launch as tl

    tl.reset()
    fn = _fake(["g", "a", "stride_a"])
    x = torch.randn(16, 8).t()                                # transposed: keeping the row stride cannot relay it
    assert klc.relaid_strides(x) is None and klc.relaid_strides(x, whole=True) == (16, 1)
    g = torch.zeros(8, 16)
    _, ds = decided(lambda: tl._decide(fn, (g, x), {"stride_a": 1}, ("k", 1), _gating()))
    assert [d.verdict for d in ds if d.contract.boundary == "kernel:fake_engine.gate"] == [Verdict.UNKNOWN], ds
    buf = torch.randn(8, 32)
    _, ds = decided(lambda: tl._decide(fn, (buf[:, :16], buf[:, ::2]), {"stride_a": 32}, ("k", 2), _gating()))
    assert [d.verdict for d in ds if d.contract.boundary == "kernel:fake_engine.gate"] == [Verdict.UNKNOWN], \
        "tensors that share storage would be parted by copies"
    was = kernel_budget = None
    from entail import kernel_reference_contract as krc
    was = krc.BUDGET
    krc.BUDGET = 16
    try:
        a = torch.randn(8, 32)[:, ::2]
        _, ds = decided(lambda: tl._decide(fn, (torch.zeros(8, 16), a), {"stride_a": 32}, ("k", 3), _gating()))
    finally:
        krc.BUDGET = was
    assert [d.verdict for d in ds if d.contract.boundary == "kernel:fake_engine.gate"] == [Verdict.UNKNOWN]
    zeros = torch.zeros(8, 32)[:, ::2]
    assert tl.differential(_gating(), (torch.zeros(8, 16), zeros), {"stride_a": 32}, {("a", 1): False}) is None, \
        "all zeros: the comparison decides nothing"
    assert not tl._REPAIR
    tl.reset()


def test_relaid_keeps_outer_strides_and_values():
    if torch is None:
        return
    a = torch.randn(4, 6, 20)[:, :, ::2]                       # (4, 6, 10) strides (120, 20, 2)
    r = klc.relaid(a)
    assert r.stride() == (120, 20, 1) and torch.equal(r, a)
    w = klc.relaid(a, whole=True)
    assert w.is_contiguous() and torch.equal(w, a)
    assert klc.relaid(torch.randn(4, 6)) is None, "nothing to change"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
