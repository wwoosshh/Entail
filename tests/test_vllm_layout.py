"""Tests for the D-arm layout check, with fake layers so no engine is needed.

The check's job is to notice a repack that produced the wrong orientation, dtype or scale granularity, so each
test builds the good layer and the broken one from the same sizes.
Run: python tests/test_vllm_layout.py
"""
import os
import sys
from types import SimpleNamespace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail.adapters.vllm_layout import check_layer  # noqa: E402

IN, OUT = 2560, 9728


def layer(weight, **kw):
    return SimpleNamespace(weight=weight, input_size_per_partition=IN, output_size_per_partition=OUT, **kw)


def test_unquantized_ok_and_broken():
    good = layer(torch.zeros(OUT, IN, dtype=torch.bfloat16))
    assert check_layer("l", good, "UnquantizedLinearMethod") == []
    transposed = layer(torch.zeros(IN, OUT, dtype=torch.bfloat16))
    out = check_layer("l", transposed, "UnquantizedLinearMethod")
    assert len(out) == 1 and "out_in" in out[0], out


def test_fp8_ok():
    good = layer(torch.zeros(IN, OUT, dtype=torch.float8_e4m3fn), weight_scale=torch.zeros(1))
    assert check_layer("l", good, "Fp8PerTensorOnlineLinearMethod") == []


def test_fp8_missing_transpose():
    """The exact defect the ledger showed nothing can detect once the axis facts are erased."""
    bad = layer(torch.zeros(OUT, IN, dtype=torch.float8_e4m3fn), weight_scale=torch.zeros(1))
    out = check_layer("l", bad, "Fp8PerTensorOnlineLinearMethod")
    assert len(out) == 1 and "in_out" in out[0], out


def test_fp8_wrong_dtype_and_scale():
    bad = layer(torch.zeros(IN, OUT, dtype=torch.bfloat16), weight_scale=torch.zeros(OUT))
    out = check_layer("l", bad, "Fp8PerTensorOnlineLinearMethod")
    assert len(out) == 2, out
    assert any("dtype" in c for c in out) and any("per_tensor" in c for c in out), out


def test_a_strided_weight_is_caught():
    """Same shape, same dtype, same values: only the strides differ (benchmark case 03)."""
    packed = torch.zeros(IN, OUT, dtype=torch.bfloat16)
    strided = packed.t().as_strided((OUT, IN), (1, OUT))
    assert tuple(strided.shape) == (OUT, IN) and not strided.is_contiguous()
    out = check_layer("l", layer(strided), "UnquantizedLinearMethod")
    assert len(out) == 1 and "strided view" in out[0], out


def test_unknown_method_is_reported_not_ignored():
    l = layer(torch.zeros(OUT, IN, dtype=torch.bfloat16))
    out = check_layer("l", l, "SomeNewQuantMethod")
    assert len(out) == 1 and "declares no layout" in out[0], out


def test_skipped_kinds():
    l = layer(torch.zeros(OUT, IN, dtype=torch.bfloat16))
    assert check_layer("l", l, "Fp8KVCacheMethod") == []
    assert check_layer("l", l, "UnquantizedEmbeddingMethod") == []


def test_layer_without_sizes_is_left_alone():
    l = SimpleNamespace(weight=torch.zeros(3, 4))
    assert check_layer("l", l, "UnquantizedLinearMethod") == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
