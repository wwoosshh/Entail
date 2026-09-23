"""Tests for the signed vLLM repack boundary (ROADMAP M4.2): load.weights_written on the weights the adapter reads,
with fake layers, so no engine is needed.

The step's job is to leave each weight in the layout its kernel reads without moving a value. So each test builds the
good weight and the broken one from the same sizes: a wrong orientation, dtype or scale granularity, a strided view
where the kernel reads packed values, rows that moved while the shape stayed. The signatures are data
(data/signatures.json); the verdicts are the core's. Run: python tests/test_vllm_layout.py
"""
import io
import os
import sys
from contextlib import redirect_stdout

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
# These tests check what stops, so they run under the policy that stopped before M5.4 (ENTAIL_ON_BROKEN=stop,
# unknown meaning-changing facts required); the default, which reports and goes on, is tested in test_report.py.
os.environ["ENTAIL_ON_BROKEN"] = "stop"
os.environ["ENTAIL_UNKNOWN"] = "require"
from entail import core, load  # noqa: E402
from entail.adapters.vllm_layout import handles  # noqa: E402
from entail.contracts import Verdict  # noqa: E402
from entail.core import RoleError  # noqa: E402
from entail.facts import Certainty  # noqa: E402
from entail.policies import Policy  # noqa: E402

STOPS = dict(on_broken="stop", on_unknown_meaning_changing="require")   # the policy before M5.4

IN, OUT = 256, 768
UNQ, FP8 = "vllm.quant_method.UnquantizedLinearMethod", "vllm.quant_method.Fp8PerTensorOnlineLinearMethod"
B = "load:vllm.quant_method.process_weights_after_loading"
LOAD, DEBUG = Policy(mode="load", **STOPS), Policy(mode="debug")
core.set_mode("load")


def w(tensor, producer=UNQ, layer="l", scale=None, sizes=(IN, OUT)):
    return load.Weight(layer, producer, tensor, sizes[0], sizes[1], scale)


def fp8(shape):
    return torch.zeros(*shape).to(torch.float8_e4m3fn)


def written(weights, policy=LOAD, step=None):
    """What the adapter does around the step: sample the weights, run the step, decide."""
    before = load.sample_weights(weights)
    if step is not None:
        step()
    return load.weights_written(B, weights, before, policy)


def verdicts(ds):
    return [(d.name, d.verdict.value) for d in ds]


def one_refusal(ds, name="Layout"):
    bad = [d for d in ds if d.verdict is not Verdict.PASS]
    assert len(bad) == 1 and bad[0].name == name and bad[0].verdict is Verdict.REFUSED and bad[0].blocking, \
        verdicts(ds)
    return bad[0]


def test_unquantized_passes_and_the_layout_travels_with_the_weight():
    weight = torch.zeros(OUT, IN, dtype=torch.bfloat16)
    ds = written([w(weight)])
    assert verdicts(ds) == [("Coverage", "pass"), ("Layout", "pass")] and "1 weight(s)" in ds[1].note, ds
    fact = core.envelopes_of(weight)["Layout"]
    assert fact.source.kind == "boundary" and fact.source.where == \
        "vllm.quant_method.UnquantizedLinearMethod.process_weights_after_loading.writes.weight", fact
    assert fact.certainty is Certainty.VERIFIED
    assert (fact.value.orientation, fact.value.dtype, fact.value.scale_granularity) == ("out_in", "bfloat16",
                                                                                         "unscaled"), fact.value


def test_unquantized_transposed_is_refused():
    weight = torch.zeros(IN, OUT, dtype=torch.bfloat16)
    d = one_refusal(written([w(weight)]))
    assert d.rule == "the declaration contradicts the data" and d.observed.value.orientation == "in_out", d
    assert "Layout" not in core.envelopes_of(weight)   # nothing false is attached


def test_fp8_passes():
    weight = fp8((OUT, IN)).t()   # online/fp8.py stores qweight.t(): shape (in, out), strides (1, in)
    ds = written([w(weight, FP8, scale=torch.ones(1))])
    assert verdicts(ds) == [("Coverage", "pass"), ("Layout", "pass")], ds
    assert core.envelopes_of(weight)["Layout"].value.scale_granularity == "per_tensor"


def test_fp8_missing_transpose_is_refused():
    """The defect the ledger showed nothing can detect once the axis facts are erased: the step forgot the .t()."""
    d = one_refusal(written([w(fp8((OUT, IN)), FP8, scale=torch.ones(1))]))
    assert (d.observed.value.orientation, d.observed.value.kind) == ("out_in", "dense"), d.observed


def test_fp8_wrong_dtype_and_scale_is_refused_with_both_in_the_record():
    weight = torch.zeros(OUT, IN, dtype=torch.bfloat16).t()
    d = one_refusal(written([w(weight, FP8, scale=torch.ones(OUT))]))
    assert (d.observed.value.dtype, d.observed.value.scale_granularity) == ("bfloat16", "per_channel"), d.observed


def test_fp8_without_a_scale_is_refused():
    d = one_refusal(written([w(fp8((OUT, IN)).t(), FP8)]))
    assert d.observed.value.scale_granularity == "unscaled", d.observed


def test_a_strided_view_is_refused_or_packed_under_use_data():
    """Same shape, dtype and values; only the strides differ (benchmark case 03, entail/audits/D_LEDGER.md 4-1)."""
    model = torch.nn.Module()
    model.proj = torch.nn.Module()
    model.proj.weight = torch.nn.Parameter(torch.randn(IN, OUT).t(), requires_grad=False)
    assert not model.proj.weight.is_contiguous()
    d = one_refusal(written([w(model.proj.weight, layer="proj")]))
    assert d.observed.value.kind == "strided", d
    use_data = Policy(mode="load", **STOPS, on_false_declaration="use_data")
    ds = written([w(model.proj.weight, layer="proj")], use_data)
    (r,) = [d for d in ds if d.verdict is Verdict.RESOLVED]
    assert r.handle == "layout.contiguous" and r.target == ("proj",) and not r.blocking, r
    values = model.proj.weight.detach().clone()
    with redirect_stdout(io.StringIO()):
        load.resolve(load.enforce(ds), handles(model))
    assert model.proj.weight.is_contiguous() and torch.equal(model.proj.weight, values)
    assert "(make the tensor contiguous" in core.envelopes_of(model.proj.weight)["Layout"].source.where


def test_a_shape_that_fits_neither_orientation_is_refused():
    d = one_refusal(written([w(torch.zeros(OUT, IN + 8, dtype=torch.bfloat16))]))
    assert "is neither (out, in)" in d.note, d.note


def test_rows_moved_by_the_step_are_refused():
    """The silent case of D_LEDGER.md result 4: shape, dtype and scale as declared, every row moved by one."""
    weight = torch.randn(OUT, IN, dtype=torch.bfloat16)
    ds = written([w(weight)], step=lambda: weight.copy_(torch.roll(weight, 1, 0)))
    d = one_refusal(ds, "Coverage")
    assert d.observed.value.taken < d.observed.value.total and "flat index" in d.observed.value.left[0], d
    assert ("Layout", "pass") in verdicts(ds)


def test_a_weight_replaced_by_one_of_another_shape_has_no_value_where_it_was():
    weight = torch.randn(OUT, IN, dtype=torch.bfloat16)
    before = load.sample_weights([w(weight)])
    ds = load.weights_written(B, [w(weight.t().contiguous())], before, LOAD)   # the transpose, packed
    (c,) = [d for d in ds if d.name == "Coverage"]
    assert c.verdict is Verdict.REFUSED and c.observed.value.taken == 0, c.observed
    assert c.note.startswith("1 weight(s): l: 16 of 16 sampled values moved (first: it went from"), c.note


def test_a_square_weight_transposed_is_caught_by_the_values_only():
    """A square weight does not show its orientation; the sampled values do."""
    weight = torch.randn(IN, IN, dtype=torch.bfloat16)
    one_refusal(written([w(weight, sizes=(IN, IN))], step=lambda: weight.copy_(weight.t().clone())), "Coverage")


def test_an_untouched_weight_passes_both():
    weight = torch.randn(OUT, IN, dtype=torch.bfloat16)
    before = load.sample_weights([w(weight)])
    ds = load.weights_written(B, [w(weight), w(weight.clone(), layer="m")], before, LOAD)
    assert sorted(verdicts(ds)) == [("Coverage", "pass"), ("Coverage", "unknown"), ("Layout", "pass")], ds
    (u,) = [d for d in ds if d.verdict is Verdict.UNKNOWN]   # "m" was not sampled: reported, not passed
    assert "no values were sampled" in u.note and not u.blocking, u


def test_an_unsigned_method_is_reported_not_passed_and_stops_only_in_debug():
    weights = [w(torch.zeros(OUT, IN), "vllm.quant_method.SomeNewQuantMethod", layer=f"l{i}") for i in range(3)]
    (d,) = written(weights)
    assert d.verdict is Verdict.UNKNOWN and not d.blocking and d.note.startswith("3 weight(s) (first: l0)"), d
    assert "has no signature" in d.note
    (d,) = written(weights, DEBUG)
    assert d.blocking


def test_methods_that_hold_no_weight_are_left_out():
    ds = written([w(None, "vllm.quant_method.Fp8KVCacheMethod"),
                  w(torch.zeros(10, 4), "vllm.quant_method.UnquantizedEmbeddingMethod")])
    assert ds == [], ds


def test_a_layer_without_sizes_is_checked_on_the_rest_and_says_what_it_did_not_read():
    ds = written([w(torch.zeros(OUT, IN, dtype=torch.bfloat16), sizes=(None, None))])
    assert sorted(verdicts(ds)) == [("Coverage", "pass"), ("Layout", "pass"), ("Layout", "unknown")], ds
    assert any("orientation was not read" in d.note for d in ds)


def test_a_packed_integer_weight_is_not_read_as_dense():
    for dtype, text in ((torch.int32, "not in the vocabulary"), (torch.uint8, "may hold packed values")):
        ds = written([w(torch.zeros(OUT, IN // 8, dtype=dtype))])
        (d,) = [d for d in ds if d.name == "Layout"]
        assert d.verdict is Verdict.UNKNOWN and text in d.note, d


def test_repeated_outcomes_are_one_decision_that_names_the_first_weights():
    weights = [w(torch.zeros(IN, OUT, dtype=torch.bfloat16), layer=f"l{i}") for i in range(5)]
    d = one_refusal(written(weights))
    assert d.note.startswith("5 weight(s): l0: orientation out_in -> in_out; l1: ") and "and 2 more" in d.note, d.note
    model = torch.nn.Module()
    for i in range(4):
        sub = torch.nn.Module()
        sub.weight = torch.nn.Parameter(torch.randn(IN, OUT).t(), requires_grad=False)
        model.add_module(f"p{i}", sub)
    strided = [w(model.get_submodule(f"p{i}").weight, layer=f"p{i}") for i in range(4)]
    ds = written(strided, Policy(mode="load", **STOPS, on_false_declaration="use_data"))
    (r,) = [d for d in ds if d.verdict is Verdict.RESOLVED]
    assert r.target == ("p0", "p1", "p2", "p3") and r.resolution == "make the tensor contiguous (4 weight(s))", r
    with redirect_stdout(io.StringIO()):
        load.resolve(load.enforce(ds), handles(model))
    assert all(model.get_submodule(f"p{i}").weight.is_contiguous() for i in range(4))


def test_enforce_stops_on_a_refusal():
    ds = written([w(torch.zeros(IN, OUT, dtype=torch.bfloat16))])
    try:
        with redirect_stdout(io.StringIO()):
            load.enforce(ds)
    except RoleError as e:
        assert "the declaration contradicts the data" in str(e), e
    else:
        raise AssertionError("a refused weight must stop the load")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
