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
    assert triton_launch.handles(fn) == {}


def test_the_hook_looks_at_a_kernel_for_its_first_layouts_only():
    if torch is None:
        return
    from entail.adapters import triton_launch

    calls = []

    class FakeJIT:
        params = [SimpleNamespace(name="x")]

        def fn(self):
            pass

        def run(self, *args, grid, warmup, **kwargs):
            calls.append(len(args))

    triton_launch.reset()
    orig = triton_launch._ORIG
    triton_launch._ORIG = FakeJIT.run
    try:
        run = triton_launch.__dict__.get("_wrapped_run")
        # install() builds the wrapper around triton's class; the same wrapper logic, on the fake, through a copy
        seen_before = len(triton_launch._SEEN)
        jit = FakeJIT()
        was = core.mode()
        core.set_mode("load")
        try:
            for n in range(triton_launch.LIMIT + 4):
                key = triton_launch._layout_key(jit, (torch.empty(2, n + 2),), {})
                if triton_launch._COUNT.get(id(jit), 0) < triton_launch.LIMIT and key not in triton_launch._SEEN:
                    triton_launch._SEEN.add(key)
                    triton_launch._COUNT[id(jit)] = triton_launch._COUNT.get(id(jit), 0) + 1
        finally:
            core.set_mode(was)
        assert triton_launch._COUNT[id(jit)] == triton_launch.LIMIT
        assert len(triton_launch._SEEN) - seen_before == triton_launch.LIMIT
        assert triton_launch.stats()["kernels_seen"] >= 1
    finally:
        triton_launch._ORIG = orig
        triton_launch.reset()
        _ = run


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
