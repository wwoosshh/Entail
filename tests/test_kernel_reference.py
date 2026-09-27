"""Tests for the kernel reference contract (ROADMAP M18.2; LIBRARY_DESIGN.md 11 M18): a custom op's dispatched
kernel against its own native definition on a slice of the real input, value by value, and the vLLM adapter on
stand-in modules (no vLLM: a stand-in CustomOp base with the same dispatch shape). CPU torch.
Run: python tests/test_kernel_reference.py"""
import io
import os
import sys
from contextlib import redirect_stdout
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import torch  # noqa: E402
from torch import nn  # noqa: E402

from entail import core, kernel_reference_contract as krc, load  # noqa: E402
from entail.adapters import vllm_kernel_reference as vk  # noqa: E402
from entail.contracts import RULES, Verdict  # noqa: E402
from entail.facts import KernelReference  # noqa: E402


class CustomOp(nn.Module):
    """vLLM's dispatch shape: forward() calls the method dispatch picked at construction."""

    def __init__(self, native=False):
        super().__init__()
        self._forward_method = self.forward_native if native else self.forward_cuda

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)


class SiluAndMul(CustomOp):
    """x[:, :d] * silu-gate, as vLLM's op; the "kernel" variants below stand in for a CUDA path."""

    def __init__(self, kind="faithful", native=False, eps=1e-6):
        self.kind = kind
        self.eps = eps
        super().__init__(native)

    def forward_native(self, x):
        d = x.shape[-1] // 2
        return nn.functional.silu(x[..., :d]) * x[..., d:]

    def forward_cuda(self, x):
        d = x.shape[-1] // 2
        if self.kind == "faithful":       # another op order, same values
            return x[..., d:] * (x[..., :d] * torch.sigmoid(x[..., :d]))
        if self.kind == "swapped":        # the halves the other way round: a layout bug
            return nn.functional.silu(x[..., d:]) * x[..., :d]
        if self.kind == "rounded":        # computed in bfloat16: rounding noise only
            return (nn.functional.silu(x[..., :d].to(torch.bfloat16)) * x[..., d:].to(torch.bfloat16)).to(x.dtype)
        if self.kind == "nan":            # one value lost: the commonest silent kernel failure
            out = self.forward_native(x).clone()
            out.view(-1)[3] = float("nan")
            return out
        if self.kind == "zeroed":         # every value but the largest zeroed: a per-element defect an
            out = self.forward_native(x).clone()                  # output-wide allowance would hide
            keep = out.abs().view(-1).argmax()
            mask = torch.zeros_like(out.view(-1), dtype=torch.bool)
            mask[keep] = True
            out.view(-1)[~mask] = 0
            return out
        if self.kind == "definition":     # the dispatched path is the definition itself
            return self.forward_native(x)
        raise AssertionError(self.kind)


class Quant(CustomOp):
    """Two outputs, the second a small scale: each output is held to its own size."""

    def __init__(self, double_scale=False):
        self.double_scale = double_scale
        super().__init__()

    def forward_native(self, x):
        scale = x.abs().amax(dim=-1, keepdim=True) / 100.0
        return x / scale, scale

    def forward_cuda(self, x):
        y, scale = self.forward_native(x)
        return (y, scale * 2) if self.double_scale else (y, scale)


class Rope(CustomOp):
    """positions [3, n] (MRoPE style) and query [n, d]: the slice must cut both along n. Its definition converts a
    cache to the query's dtype and keeps it, as vLLM's rotary does (_match_cos_sin_cache_dtype)."""

    def __init__(self, keep_cache=False):
        super().__init__()
        self.keep_cache = keep_cache
        self.register_buffer("cache", torch.ones(8, dtype=torch.float32))

    def forward_native(self, positions, query):
        cache = self.cache
        if cache.dtype != query.dtype:
            cache = cache.to(query.dtype)
            if self.keep_cache:
                self.cache = cache
        return query * cache * positions[0].unsqueeze(-1).to(query.dtype)

    def forward_cuda(self, positions, query):
        return query * self.cache.to(query.dtype) * positions[0].unsqueeze(-1).to(query.dtype)


class SizeTiled(CustomOp):
    """Right at launches of up to 64 rows, wrong above (a launch configuration keyed by the batch size, vllm#52576's
    shape): a 64-row slice takes the small launch and passes; the engine's own 256-row launch does not."""

    def forward_native(self, x):
        return x * 2.0

    def forward_cuda(self, x):
        return x * 2.0 if x.shape[0] <= 64 else x * 2.0 + 0.5 * x.flip(-1)


class CachedRope(CustomOp):
    """A rotary op with its cos/sin cache, one row per position. "doubled" reads the row of twice the position:
    right at position 0, which is all a warm-up gives it."""

    def __init__(self, kind="faithful"):
        self.kind = kind
        super().__init__()
        self.register_buffer("cos_sin_cache", torch.randn(100, 8))

    def forward_native(self, positions, query):
        return query * self.cos_sin_cache[positions]

    def forward_cuda(self, positions, query):
        p = positions if self.kind == "faithful" else (positions * 2) % 100
        return query * self.cos_sin_cache[p]


class Overrides(SiluAndMul):
    def forward(self, x):
        return self.forward_native(x)


class Indexer(SiluAndMul):
    """Holds an index buffer the engine writes: never re-run on a slice."""

    def __init__(self):
        super().__init__()
        self.topk_indices_buffer = torch.zeros(4, dtype=torch.int32)


class WithCache(SiluAndMul):
    def __init__(self):
        super().__init__()
        self.k_cache = SimpleNamespace(kv_cache=torch.zeros(2), prefix="layer.0")


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


def setup():
    vk.reset()
    torch.manual_seed(0)


def captured(m, x):
    """One call of the op inside a CUDA graph capture, at x's size: the graph holds the kernel there, and nothing
    held it to the definition at that size - so a later mismatch cannot be repaired for what the graph replays."""
    was = vk.capturing
    vk.capturing = lambda: True
    try:
        with redirect_stdout(io.StringIO()):
            m(x)
    finally:
        vk.capturing = was


def warm(fn):
    """fn() inside one of vLLM's dummy runs (profile run, warm-ups): the adapter decides on a probe made from it."""
    vk._STATE["dummy"] += 1
    try:
        return decided(fn)
    finally:
        vk._STATE["dummy"] -= 1


def test_the_fact_checks_its_numbers():
    f = KernelReference(op="SiluAndMul", max_abs_diff=0.5, floor=0.01, scale=3.0)
    assert f.op == "SiluAndMul" and f.floor == 0.01
    for bad in (dict(op="", max_abs_diff=0.0), dict(op="x", max_abs_diff=-1.0), dict(op="x", max_abs_diff=0.0, floor=-1)):
        try:
            KernelReference(**bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_slicing_cuts_the_token_dimension_wherever_it_is():
    q, pos = torch.randn(200, 8), torch.arange(600).reshape(3, 200)
    assert krc.rows_of((pos, q), {}) == 200, "the largest tensor names the token dimension, positions [3, n] share it"
    assert krc.rows_of((q, torch.zeros(5, dtype=torch.int32)), {}) is None, "metadata that does not share it: not cut"
    assert krc.rows_of((), {"flag": True}) is None
    assert krc.rows_of((torch.randn(8, 200),), {}, hint=200) == 8, "the engine's count names no first dimension: the largest tensor's"
    assert krc.rows_of((torch.randn(8, 200, 2), torch.randn(200, 4)), {}, hint=200) is None, "they must share it"
    a = krc.sliced((pos, q), 200, 16)
    assert tuple(a[0].shape) == (3, 16) and tuple(a[1].shape) == (16, 8)
    b = krc.sliced({"x": q.to(torch.bfloat16), "flag": True, "w": torch.randn(8), "s": torch.zeros(3, dtype=torch.int8)},
                   200, 16, torch.float32)
    assert tuple(b["x"].shape) == (16, 8) and b["x"].dtype == torch.float32 and b["flag"] is True
    assert b["w"].dtype == torch.float32 and b["s"].dtype == torch.int8, "only reduced-precision tensors are cast"
    assert tuple(b["w"].shape) == (8,), "a tensor without the token dimension is copied whole"
    assert a[1].data_ptr() != q.data_ptr(), "slices are clones"
    c = krc.cast(a)
    assert c[1].data_ptr() != a[1].data_ptr() and torch.equal(c[1], a[1]), "every run gets its own clone"
    assert krc.nbytes(a) == 3 * 16 * 8 + 16 * 8 * 4
    assert krc.uniform_rows([torch.ones(10, 4)]) and not krc.uniform_rows([torch.randn(10, 4)])
    assert not krc.uniform_rows([torch.ones(1, 4)]), "one row is one row"


def test_compare_value_by_value():
    r = torch.randn(16, 8)
    cmp = krc.compare(r + 0.001, r, r + 0.0005)
    assert abs(cmp.diff - 0.001) < 1e-6 and abs(cmp.floor - 0.0005) < 1e-6 and cmp.elements == 128
    assert cmp.scale == float(r.abs().max()) and cmp.violations == 0 and cmp.nonfinite == 0 and cmp.margin <= 1
    assert cmp.worst is not None and len(cmp.worst) == 6 and cmp.dtypes == ("float32",)
    cmp = krc.compare((r, None), (r, None))
    assert cmp.diff == 0.0 and cmp.floor is None and cmp.same_as_definition
    try:
        krc.compare(r, (r, r))
    except ValueError as e:
        assert "floating tensors" in str(e)
    else:
        raise AssertionError("mismatched outputs must not compare")
    bad = r.clone()
    bad.view(-1)[5] = float("nan")
    cmp = krc.compare(bad, r, r)
    assert cmp.nonfinite == 1 and cmp.violations == 0, "a NaN is a mismatch on its own, and never hides the rest"
    assert cmp.diff == 0.0, "the finite values still agree"
    cmp = krc.compare(bad, bad, bad)
    assert cmp.nonfinite == 0, "NaN on both sides at the same place is agreement"
    inf = r.clone()
    inf.view(-1)[2] = float("inf")
    assert krc.compare(inf, r, r).nonfinite == 1
    assert krc.compare(r, r, r, inputs=[r]).same_as_input, "an identity: the definition returned its input"
    assert "float8_e4m3fn" in krc.ULPS and krc.ULPS["float8_e4m3fn"] == 2.0 ** -3
    # one value zeroed among values of the same size is caught, whatever the output's largest value
    big = torch.randn(64, 8)
    big.view(-1)[0] = 300.0
    zeroed = big.clone()
    zeroed.view(-1)[7] = 0.0
    cmp = krc.compare(zeroed, big, big + 0.004)
    assert cmp.violations == 1 and cmp.worst[1] == 7, cmp


def test_check_passes_within_tolerance_and_breaks_beyond():
    B, C = "kernel:test.custom_op", "test.SiluAndMul"
    krc.reset(B)
    r = torch.randn(16, 8)
    ok = krc.compare(r + 1e-7, r, r + 1e-7)
    out, ds = decided(lambda: krc.check(B, C, "SiluAndMul", ok, "the kernel"))
    assert ds[0].verdict is Verdict.PASS and "worst ratio to the allowance" in ds[0].note
    assert ds[0].chosen.value.max_abs_diff == ok.diff and ds[0].declared.value.max_abs_diff == 0.0
    bad = krc.compare(r + 0.5, r, r + 1e-7)
    out, ds = decided(lambda: krc.check(B, C, "SiluAndMul", bad, "the kernel"))
    assert ds[0].verdict is Verdict.BROKEN and ds[0].rule == RULES["kernel_reference_mismatch"] and not ds[0].blocking
    assert "128 values beyond" in ds[0].note and "worst value at output 0" in ds[0].note, ds[0].note
    assert krc.stats(B)["broken"] == 1 and krc.stats(B)["checks"] == 2


def test_a_faithful_kernel_passes_once_and_the_wrapper_steps_aside():
    setup()
    m = SiluAndMul("faithful")
    orig = m._forward_method
    assert vk.wrap(m) and m._forward_method is not orig and not vk.wrap(m)
    x = torch.randn(300, 16)
    out, ds = decided(lambda: m(x))
    assert torch.equal(out, orig(x)), "the engine gets the kernel's own output"
    assert len(ds) == 1 and ds[0].verdict is Verdict.PASS and ds[0].name == "KernelReference", ds
    assert "64 rows" in ds[0].note and ds[0].chosen.value.op == "SiluAndMul"
    assert m._forward_method is orig, "after the decision the original is back"
    out, ds = decided(lambda: m(x))
    assert not ds, "decided once per op class, configuration and pattern"
    m2 = SiluAndMul("faithful")
    assert vk.wrap(m2)
    out, ds = decided(lambda: m2(x))
    assert not ds and m2._forward_method is not None and vk.stats()["decided"] == 1
    m3 = SiluAndMul("faithful", eps=1e-5)
    assert vk.wrap(m3)
    out, ds = decided(lambda: m3(x))
    assert len(ds) == 1, "another configuration of the same class is decided on its own"


def test_a_kernel_with_the_halves_swapped_is_broken():
    setup()
    m = SiluAndMul("swapped")
    vk.wrap(m)
    captured(m, torch.randn(100, 16))
    out, ds = decided(lambda: m(torch.randn(100, 16)))
    assert len(ds) == 1 and ds[0].verdict is Verdict.BROKEN and ds[0].rule == RULES["kernel_reference_mismatch"], ds
    assert "SiluAndMul" in ds[0].note and "max |kernel - definition|" in ds[0].note and "values beyond" in ds[0].note
    assert vk.stats()["broken"] == 1


def test_a_kernel_that_loses_a_value_to_nan_is_broken():
    setup()
    m = SiluAndMul("nan")
    vk.wrap(m)
    captured(m, torch.randn(100, 16))
    out, ds = decided(lambda: m(torch.randn(100, 16)))
    assert len(ds) == 1 and ds[0].verdict is Verdict.BROKEN and "non-finite" in ds[0].note, ds


def test_a_kernel_that_zeroes_values_is_broken_whatever_the_largest_value():
    setup()
    m = SiluAndMul("zeroed")
    vk.wrap(m)
    x = torch.randn(100, 16)
    x[0, 0] = 300.0
    captured(m, x)
    out, ds = decided(lambda: m(x))
    assert len(ds) == 1 and ds[0].verdict is Verdict.BROKEN and "values beyond" in ds[0].note, ds


def test_each_output_is_held_to_its_own_size():
    setup()
    m = Quant(double_scale=False)
    vk.wrap(m)
    out, ds = decided(lambda: m(torch.randn(100, 16)))
    assert len(ds) == 1 and ds[0].verdict is Verdict.PASS, ds
    setup()
    m = Quant(double_scale=True)
    vk.wrap(m)
    captured(m, torch.randn(100, 16))
    out, ds = decided(lambda: m(torch.randn(100, 16)))
    assert len(ds) == 1 and ds[0].verdict is Verdict.BROKEN and "output 1" in ds[0].note, ds


def test_rounding_noise_is_within_the_floor():
    """bfloat16 input, as an engine gives: the kernel (another op order) and the definition both round; the
    definition run in float32 is the reference, and its own bfloat16 noise is the floor the kernel is held to."""
    setup()
    m = SiluAndMul("faithful")
    vk.wrap(m)
    out, ds = decided(lambda: m(torch.randn(100, 16).to(torch.bfloat16)))
    assert len(ds) == 1 and ds[0].verdict is Verdict.PASS, ds
    assert ds[0].chosen.value.floor > 0 and "(typical" in ds[0].note
    setup()
    m = SiluAndMul("rounded")
    vk.wrap(m)
    out, ds = decided(lambda: m(torch.randn(100, 16).to(torch.bfloat16)))
    assert len(ds) == 1 and ds[0].verdict is Verdict.PASS, ds


def test_a_kernel_that_is_the_definition_is_said_so():
    setup()
    m = SiluAndMul("definition")
    vk.wrap(m)
    out, ds = decided(lambda: m(torch.randn(100, 16).to(torch.bfloat16)))
    assert len(ds) == 1 and ds[0].verdict is Verdict.PASS and "reproduces the definition bitwise" in ds[0].note, ds
    assert "calls the definition itself" in ds[0].note, ds[0].note
    assert "forward_native" not in vars(m), "the spy on the definition is gone after the comparison"
    setup()
    m = SiluAndMul("faithful")
    vk.wrap(m)
    out, ds = decided(lambda: m(torch.randn(100, 16).to(torch.bfloat16)))
    assert "calls the definition" not in ds[0].note, "an independent kernel is not said to call the definition"


def test_a_definition_that_cannot_run_is_unknown_and_the_output_still_flows():
    setup()
    m = SiluAndMul("faithful")          # the kernel works; the definition refuses to run
    m.forward_native = lambda x: (_ for _ in ()).throw(NotImplementedError("no native"))
    vk.wrap(m)
    x = torch.randn(10, 16)
    try:
        out, ds = decided(lambda: m(x))
    except NotImplementedError:
        raise AssertionError("the comparison must never raise into the engine")
    assert ds and ds[-1].verdict is Verdict.UNKNOWN and "NotImplementedError" in ds[-1].note, ds
    out, ds = decided(lambda: m(x))
    assert not ds, "said once, not on every call"


def test_native_dispatch_forward_overrides_and_engine_state_are_not_candidates():
    setup()
    assert not vk.is_candidate(SiluAndMul("faithful", native=True))
    assert not vk.is_candidate(Overrides("faithful"))
    assert not vk.is_candidate(nn.Linear(2, 2))
    assert not vk.is_candidate(Indexer()) and "index buffer" in vk.stateful(Indexer())
    assert not vk.is_candidate(WithCache()) and "KV cache" in vk.stateful(WithCache())
    assert vk.stateful(SiluAndMul("faithful")) is None and vk.stateful(Rope()) is None
    s = vk.stats()
    assert s["native"] == 1 and s["overrides_forward"] == 1 and s["stateful"] == 2


def test_positions_of_shape_3_by_n_are_sliced_with_the_query_and_the_ops_state_is_put_back():
    setup()
    m = Rope(keep_cache=True)
    vk.wrap(m)
    cache = m.cache
    pos, q = torch.arange(3 * 500).reshape(3, 500) + 1, torch.randn(500, 8).to(torch.bfloat16)
    out, ds = decided(lambda: m(pos, q))
    assert len(ds) == 1 and ds[0].verdict is Verdict.PASS, ds
    assert m.cache is cache and m.cache.dtype == torch.float32, "the definition's converted cache did not stay"
    assert "put back" in ds[0].note, ds[0].note
    stores = vk.snapshot(m)
    m.cache = m.cache.to(torch.bfloat16)
    m.extra = torch.zeros(2)
    assert vk.restore(stores) == 2 and m.cache is cache and not hasattr(m, "extra")


def test_a_dummy_or_identity_input_decides_nothing_and_the_next_call_decides():
    """A profile run feeds zeros or one repeated row, and rotary at position 0 is the identity: such a call says
    nothing about the kernel, so the wrapper stays on and the next real input decides; after TRIES such calls it
    is given up. Calls inside the engine's dummy runs are not even counted."""
    r = torch.randn(4, 4)
    assert krc.vacuous(krc.compare(torch.zeros(4, 4), torch.zeros(4, 4), torch.zeros(4, 4))) is not None
    assert krc.vacuous(krc.compare(r, r, r, inputs=[r])) is not None, "an identity"
    assert krc.vacuous(krc.compare(r + 1, r, r, inputs=[r])) is None, "a kernel that differs on it is decided"
    assert krc.vacuous(krc.compare(r, r, r + 1e-7)) is None
    setup()
    m = SiluAndMul("swapped")
    vk.wrap(m)
    out, ds = decided(lambda: m(torch.zeros(100, 16)))
    assert not ds and vk.stats()["decided"] == 0
    out, ds = decided(lambda: m(torch.ones(100, 16)))
    assert not ds, "one repeated row decides nothing either"
    tried = dict(vk._TRIED)
    captured(m, torch.randn(100, 16))
    assert vk._TRIED == tried, "a capture-time call is noted, not compared or counted"
    out, ds = decided(lambda: m(torch.randn(100, 16)))
    assert len(ds) == 1 and ds[0].verdict is Verdict.BROKEN and "call 3" in ds[0].note, ds
    setup()
    m = Rope()
    vk.wrap(m)
    out, ds = decided(lambda: m(torch.ones(3, 50, dtype=torch.long), torch.randn(50, 8)))
    assert not ds, "positions 1 with a cache of ones: the identity"
    setup()
    m = SiluAndMul("faithful")
    vk.wrap(m)
    was = vk.TRIES
    vk.TRIES = 3
    try:
        for _ in range(vk.TRIES - 1):
            out, ds = decided(lambda: m(torch.zeros(100, 16)))
            assert not ds
        out, ds = decided(lambda: m(torch.zeros(100, 16)))
        assert len(ds) == 1 and ds[0].verdict is Verdict.UNKNOWN and "first 3 real calls" in ds[0].note, ds
    finally:
        vk.TRIES = was


def test_under_a_stop_policy_the_decision_raises_once_and_the_wrapper_steps_aside():
    setup()
    m = SiluAndMul("swapped")
    orig = m._forward_method
    vk.wrap(m)
    captured(m, torch.randn(100, 16))       # unrepairable, so the mismatch is broken - and a stop policy stops there
    os.environ["ENTAIL_ON_BROKEN"] = "stop"
    try:
        try:
            decided(lambda: m(torch.randn(100, 16)))
        except core.RoleError:
            pass
        else:
            raise AssertionError("a stop policy must stop at the mismatch")
        assert m._forward_method is orig, "decided: the original is back"
        out, ds = decided(lambda: m(torch.randn(100, 16)))
        assert not ds, "the second call compares nothing"
    finally:
        del os.environ["ENTAIL_ON_BROKEN"]


def test_the_wrapper_steps_aside_inside_torch_compile():
    """Custom ops enabled under torch.compile put the wrapper inside vLLM's full-graph compile: traced, it hands
    the call straight to the kernel, so the graph holds the kernel alone and nothing here breaks the compile."""
    setup()
    m = SiluAndMul("swapped")
    vk.wrap(m)
    required = sys.platform.startswith("linux")
    try:
        f = torch.compile(m, fullgraph=True, backend="eager")
        out, ds = decided(lambda: f(torch.randn(100, 16)))
    except Exception as e:  # noqa: BLE001 - Dynamo unavailable here (Windows, an old torch)
        if required:
            raise
        print("skip: torch.compile unavailable:", type(e).__name__)
        return
    assert not ds, "traced: nothing compared, nothing recorded"
    assert torch.equal(out, m.forward_cuda(torch.randn(0, 16))) or out.shape == (100, 8)


def test_arguments_that_cannot_be_cut_are_not_cloned_whole():
    setup()
    m = SiluAndMul("faithful")
    vk.wrap(m)
    was = krc.BUDGET
    krc.BUDGET = 1024
    try:
        out, ds = decided(lambda: m(torch.randn(2, 300, 16)))    # a batch first: rows = n = 2 cuts nothing
    finally:
        krc.BUDGET = was
    assert len(ds) == 1 and ds[0].verdict is Verdict.UNKNOWN and "too large to compare whole" in ds[0].note, ds


def test_an_in_place_kernel_is_compared_on_the_input_it_was_given():
    class InPlace(CustomOp):
        def forward_native(self, x):
            return x * 2

        def forward_cuda(self, x):
            x.mul_(2)
            return x

    setup()
    m = InPlace()
    vk.wrap(m)
    out, ds = decided(lambda: m(torch.randn(100, 16)))
    assert len(ds) == 1 and ds[0].verdict is Verdict.PASS, ds


def test_a_mismatch_is_sent_to_the_definition_when_graphs_are_off():
    """M19 L3: resolved by the meaning-keeping consumer, from the very call that was compared."""
    setup()
    was = vk.graphs_off
    vk.graphs_off = lambda: True
    try:
        m = SiluAndMul("swapped")
        vk.wrap(m)
        x = torch.randn(100, 16)
        out, ds = decided(lambda: m(x))
        assert len(ds) == 1 and ds[0].verdict is Verdict.RESOLVED and ds[0].rule == RULES["resolved"], ds
        assert "forward_native" in (ds[0].resolution or ""), ds[0].resolution
        assert torch.equal(out, m.forward_native(x)), "the call that was compared already gets the definition"
        assert m._forward_method == m.forward_native, "later calls go to the definition"
        out2, ds2 = decided(lambda: m(torch.randn(50, 16)))
        assert not ds2 and torch.equal(out2.shape and out2, out2)
        assert vk.stats()["resolved"] == 1 and vk.stats()["sent_to_definition"] == 1, vk.stats()
    finally:
        vk.graphs_off = was


def test_with_graphs_or_under_refuse_a_mismatch_stays_broken():
    setup()
    m = SiluAndMul("swapped")             # a graph captured the kernel at a size nothing held it to the definition
    vk.wrap(m)
    x = torch.randn(100, 16)
    captured(m, torch.randn(8, 16))
    out, ds = decided(lambda: m(x))
    assert ds[0].verdict is Verdict.BROKEN and "not repaired: CUDA graphs were captured" in ds[0].note, ds[0].note
    assert torch.equal(out, SiluAndMul("swapped").forward_cuda(x)), "unrepaired: the engine gets the kernel's output"
    setup()
    m = SiluAndMul("swapped")             # nothing captured yet: the switch reaches every later call and capture
    vk.wrap(m)
    out, ds = decided(lambda: m(x))
    assert ds[0].verdict is Verdict.RESOLVED and torch.equal(out, m.forward_native(x)), ds
    setup()
    was, pol = vk.graphs_off, core.policy()
    vk.graphs_off = lambda: True
    core.set_policy("refuse")
    try:
        m = SiluAndMul("swapped")
        vk.wrap(m)
        out, ds = decided(lambda: m(x))
        assert ds[0].verdict is Verdict.BROKEN and m._forward_method != m.forward_native, ds
    finally:
        vk.graphs_off = was
        core.set_policy(pol)


def test_every_runner_method_that_feeds_dummy_input_is_marked():
    """vLLM 0.30's second GPU runner captures and profiles its graphs outside _dummy_run (capture_model,
    profile_cudagraph_memory): calls made there are the engine's own dummy input too."""
    import types
    setup()
    name = "vllm.v1.worker.gpu.model_runner"
    seen = {}

    class GPUModelRunner:
        def _dummy_run(self):
            seen["_dummy_run"] = vk._STATE["dummy"]

        def capture_model(self):
            seen["capture_model"] = vk._STATE["dummy"]

        def profile_cudagraph_memory(self):
            seen["profile_cudagraph_memory"] = vk._STATE["dummy"]

        def execute_model(self):
            seen["execute_model"] = vk._STATE["dummy"]

    mod = types.ModuleType(name)
    GPUModelRunner.__module__ = name
    mod.GPUModelRunner = GPUModelRunner
    was = sys.modules.get(name)
    sys.modules[name] = mod
    try:
        assert vk.install_dummy_run() == 1 and vk.install_dummy_run() == 0
        r = GPUModelRunner()
        for m in ("_dummy_run", "capture_model", "profile_cudagraph_memory", "execute_model"):
            getattr(r, m)()
        assert seen == {"_dummy_run": 1, "capture_model": 1, "profile_cudagraph_memory": 1, "execute_model": 0}, seen
        assert vk._STATE["dummy"] == 0
        vk.uninstall()
        r.capture_model()
        assert seen["capture_model"] == 0, "uninstall puts every method back"
    finally:
        if was is None:
            del sys.modules[name]
        else:
            sys.modules[name] = was


def test_a_warm_up_call_is_decided_on_a_probe_before_any_capture():
    """M19 L3.3a: vLLM's warm-up feeds one row over and over; the probe keeps its shape and makes up the values, so
    the mismatch is decided - and repaired - before a graph is captured, and the capture records the definition."""
    setup()
    m = SiluAndMul("swapped")
    vk.wrap(m)
    x = torch.ones(100, 16)
    out, ds = warm(lambda: m(x))
    assert len(ds) == 1 and ds[0].verdict is Verdict.RESOLVED, ds
    assert "probe of 100 rows made from vLLM's warm-up call of 100 rows" in ds[0].note, ds[0].note
    assert torch.equal(out, m.forward_native(x)), "the warm-up itself already runs the definition"
    assert m._forward_method == m.forward_native, "a capture from here on records the definition"
    assert vk.stats()["warm_decided"] == 1
    a = krc.probed((x,), 100, 64, krc.generator("cpu"))
    b = krc.probed((x,), 100, 64, krc.generator("cpu"))
    assert tuple(a[0].shape) == (64, 16) and not krc.uniform_rows(a) and torch.equal(a[0], b[0]), "seeded"
    cols = torch.ones(100, 32)[:, ::2]
    p = krc.probed((cols, torch.zeros(100, dtype=torch.long)), 100, 64, krc.generator("cpu"))
    assert p[0].stride() == cols.stride() and p[1].dtype == torch.long and int(p[1].abs().sum()) == 0, \
        "strides kept; an integer the op alone knows the range of keeps the engine's values without a fill"


def test_warm_ups_are_decided_once_per_size_class():
    setup()
    m = SiluAndMul("faithful")
    vk.wrap(m)
    for n in (1, 2, 3, 4, 100, 120, 128):
        out, ds = warm(lambda: m(torch.ones(n, 16)))
    assert vk.stats()["warm_decided"] == 4, vk.stats()
    assert vk._PASSED[vk.config_of(m, "SiluAndMul")] == {1, 2, 4, 128}
    out, ds = decided(lambda: m(torch.randn(50, 16)))
    assert len(ds) == 1 and ds[0].verdict is Verdict.PASS and "call 1" in ds[0].note, "the first real call still decides"


def test_a_kernel_wrong_only_at_large_launches_is_caught_at_that_size():
    setup()
    m = SizeTiled()
    vk.wrap(m)
    out, ds = decided(lambda: m(torch.randn(256, 16)))
    assert ds[0].verdict is Verdict.PASS, "the real call's 64-row slice takes the small launch: missed"
    setup()
    m = SizeTiled()
    vk.wrap(m)
    out, ds = warm(lambda: m(torch.ones(32, 16)))
    assert ds[0].verdict is Verdict.PASS
    captured(m, torch.randn(32, 16))       # the graph of class 32 holds a kernel verified at that size
    out, ds = warm(lambda: m(torch.ones(256, 16)))
    assert ds[0].verdict is Verdict.RESOLVED and "probe of 256 rows" in ds[0].note, ds
    setup()
    m = SizeTiled()
    vk.wrap(m)
    captured(m, torch.randn(512, 16))      # a graph at a size nothing verified
    out, ds = warm(lambda: m(torch.ones(256, 16)))
    assert ds[0].verdict is Verdict.BROKEN and "not repaired" in ds[0].note, ds


def test_rotary_positions_are_drawn_from_the_cache_so_a_warm_up_decides():
    for kind, verdict in (("doubled", Verdict.RESOLVED), ("faithful", Verdict.PASS)):
        setup()
        m = CachedRope(kind)
        vk.wrap(m)
        out, ds = warm(lambda: m(torch.zeros(50, dtype=torch.long), torch.ones(50, 8)))
        assert len(ds) == 1 and ds[0].verdict is verdict, (kind, ds)


def test_a_repair_reaches_every_module_of_its_configuration():
    setup()
    m1, m2, m3 = SiluAndMul("swapped"), SiluAndMul("swapped"), SiluAndMul("swapped", eps=1e-5)
    for m in (m1, m2, m3):
        vk.wrap(m)
    x = torch.randn(100, 16)
    out, ds = decided(lambda: m1(x))
    assert ds[0].verdict is Verdict.RESOLVED, ds
    assert m2._forward_method == m2.forward_native, "the layer that shares the configuration is switched too"
    out2, ds2 = decided(lambda: m2(x))
    assert not ds2 and torch.equal(out2, m2.forward_native(x))
    assert m3._forward_method != m3.forward_native, "another configuration is its own decision"


def test_instrument_walks_a_model_and_says_what_it_skipped():
    setup()
    model = nn.Sequential(SiluAndMul("faithful"), nn.Linear(8, 8), SiluAndMul("faithful", native=True), Indexer())
    out, ds = decided(lambda: vk.instrument(model))
    assert out == 1 and vk.stats()["instrumented"] == 1 and vk.stats()["stateful"] == 1
    assert not ds, "nothing to say without vLLM's registry: no unreached ops, one rank, eager"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    core.set_mode("off")
