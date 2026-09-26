"""Tests for the kernel reference contract (ROADMAP M18.2; LIBRARY_DESIGN.md 11 M18): a custom op's dispatched
kernel against its own native definition on a slice of the real input, and the vLLM adapter on stand-in modules
(no vLLM: a stand-in CustomOp base with the same dispatch shape). CPU torch. Run: python tests/test_kernel_reference.py"""
import io
import os
import sys
from contextlib import redirect_stdout

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

    def __init__(self, kind="faithful", native=False):
        self.kind = kind
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
        raise AssertionError(self.kind)


class Rope(CustomOp):
    """positions [3, n] (MRoPE style) and query [n, d]: the slice must cut both along n."""

    def forward_native(self, positions, query):
        return query * positions[0].unsqueeze(-1).to(query.dtype)

    def forward_cuda(self, positions, query):
        return self.forward_native(positions, query)


class Overrides(SiluAndMul):
    def forward(self, x):
        return self.forward_native(x)


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
    assert krc.rows_of((pos, q), {}) == 3 or krc.rows_of((q, pos), {}) == 200
    a = krc.sliced((pos, q), 200, 16)
    assert tuple(a[0].shape) == (3, 16) and tuple(a[1].shape) == (16, 8)
    b = krc.sliced({"x": q, "flag": True, "w": torch.randn(8)}, 200, 16, torch.float64)
    assert tuple(b["x"].shape) == (16, 8) and b["x"].dtype == torch.float64 and b["flag"] is True
    assert tuple(b["w"].shape) == (8,), "a tensor without the token dimension is copied whole"
    assert a[1].data_ptr() != q.data_ptr(), "slices are clones"


def test_compare_and_tolerance():
    r = torch.randn(16, 8)
    diff, floor, scale, n = krc.compare(r + 0.001, r, r + 0.0005)
    assert abs(diff - 0.001) < 1e-6 and abs(floor - 0.0005) < 1e-6 and n == 128 and scale == float(r.abs().max())
    diff, floor, scale, n = krc.compare((r, None), (r, None))
    assert diff == 0.0 and floor is None
    try:
        krc.compare(r, (r, r))
    except ValueError as e:
        assert "floating tensors" in str(e)
    else:
        raise AssertionError("mismatched outputs must not compare")
    assert krc.tolerance("torch.bfloat16", 0.01, 2.0) == krc.FACTOR * 0.01 + krc.ATOL_ULPS * 2.0 ** -7 * 2.0
    assert krc.tolerance("bfloat16", None, 2.0) == krc.ATOL_ULPS * 2.0 ** -7 * 2.0


def test_check_passes_within_tolerance_and_breaks_beyond():
    B, C = "kernel:test.custom_op", "test.SiluAndMul"
    krc.reset(B)
    out, ds = decided(lambda: krc.check(B, C, "SiluAndMul", "torch.bfloat16", 0.02, 0.01, 2.0, 128, "the kernel"))
    assert ds[0].verdict is Verdict.PASS and "allowed" in ds[0].note and ds[0].chosen.value.max_abs_diff == 0.02
    out, ds = decided(lambda: krc.check(B, C, "SiluAndMul", "torch.bfloat16", 1.5, 0.01, 2.0, 128, "the kernel"))
    assert ds[0].verdict is Verdict.BROKEN and ds[0].rule == RULES["kernel_reference_mismatch"] and not ds[0].blocking
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
    assert not ds, "decided once per op class and pattern"
    m2 = SiluAndMul("faithful")
    assert vk.wrap(m2)
    out, ds = decided(lambda: m2(x))
    assert not ds and m2._forward_method is not None and vk.stats()["decided"] == 1


def test_a_kernel_with_the_halves_swapped_is_broken():
    setup()
    m = SiluAndMul("swapped")
    vk.wrap(m)
    out, ds = decided(lambda: m(torch.randn(100, 16)))
    assert len(ds) == 1 and ds[0].verdict is Verdict.BROKEN and ds[0].rule == RULES["kernel_reference_mismatch"], ds
    assert "SiluAndMul" in ds[0].note and "max |kernel - definition|" in ds[0].note
    assert vk.stats()["broken"] == 1


def test_rounding_noise_is_within_the_floor():
    """bfloat16 input, as an engine gives: the kernel (another op order) and the definition both round; the
    definition run in float32 is the reference, and its own bfloat16 noise is the floor the kernel is held to."""
    setup()
    m = SiluAndMul("faithful")
    vk.wrap(m)
    out, ds = decided(lambda: m(torch.randn(100, 16).to(torch.bfloat16)))
    assert len(ds) == 1 and ds[0].verdict is Verdict.PASS, ds
    assert ds[0].chosen.value.floor > 0 and "bfloat16" not in ds[0].note or True
    assert ds[0].chosen.value.max_abs_diff <= krc.tolerance("torch.bfloat16", ds[0].chosen.value.floor,
                                                             ds[0].chosen.value.scale)


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


def test_native_dispatch_and_forward_overrides_are_not_candidates():
    setup()
    assert not vk.is_candidate(SiluAndMul("faithful", native=True))
    assert not vk.is_candidate(Overrides("faithful"))
    assert not vk.is_candidate(nn.Linear(2, 2))
    assert vk.stats()["native"] == 1 and vk.stats()["overrides_forward"] == 1


def test_positions_of_shape_3_by_n_are_sliced_with_the_query():
    setup()
    m = Rope()
    vk.wrap(m)
    pos, q = torch.arange(3 * 500).reshape(3, 500), torch.randn(500, 8)
    out, ds = decided(lambda: m(pos, q))
    assert len(ds) == 1 and ds[0].verdict is Verdict.PASS, ds


def test_instrument_walks_a_model():
    setup()
    model = nn.Sequential(SiluAndMul("faithful"), nn.Linear(8, 8), SiluAndMul("faithful", native=True))
    assert vk.instrument(model) == 1 and vk.stats()["instrumented"] == 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    core.set_mode("off")
