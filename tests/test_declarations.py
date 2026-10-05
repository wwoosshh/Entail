"""Tests for entail/declarations.py (ROADMAP M19 L6 step 1, M22.4): what an engine declares about the values it makes,
read from its table into meanings - for vLLM's table and SGLang's, through each adapter's own reading point. A layer
of the kind the table names (a class called LinearBase in its MRO), weight parameters that say their output and input
axes (the parameter classes both engines use), a block-FP8 weight and its scale; an activation quantizer that returns
a value and its scale made together. CPU tensors (the adapters read only the GPU's; here the reader is told not to).
Run: python tests/test_declarations.py
"""
import json
import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import declarations as D  # noqa: E402
from entail import kernel_check as KC  # noqa: E402

DATA = os.path.join(os.path.dirname(HERE), "entail", "data")


class Declared(torch.nn.Parameter):
    """A weight parameter that says which of its axes is the output and which the input (as vLLM's and SGLang's
    parameter classes do)."""

    def __new__(cls, data, output_dim=None, input_dim=None):
        p = super().__new__(cls, data, requires_grad=False)
        p.output_dim, p.input_dim = output_dim, input_dim
        return p


class LinearBase(torch.nn.Module):
    def __init__(self, out_f=256, in_f=512):
        super().__init__()
        self.weight = Declared(torch.zeros(out_f, in_f, dtype=torch.float8_e4m3fn), output_dim=0, input_dim=1)
        self.weight_scale_inv = Declared(torch.ones(out_f // 128, in_f // 128), output_dim=0, input_dim=1)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = LinearBase()
        self.norm = torch.nn.LayerNorm(512)


def table(name):
    with open(os.path.join(DATA, f"{name}_declarations.json"), encoding="utf-8") as f:
        return json.load(f)


def process(model, *rest):
    """What an engine's weight processing does here: makes new tensors of the same shapes (a requantization)."""
    model.proj.weight = torch.nn.Parameter(model.proj.weight.data.clone(), requires_grad=False)
    model.proj.weight_scale_inv = torch.nn.Parameter(model.proj.weight_scale_inv.data.clone(), requires_grad=False)
    return model


def check_weight(model, engine):
    fw, fs = KC.fact_of(model.proj.weight), KC.fact_of(model.proj.weight_scale_inv)
    assert fw is not None and fw["names"] == ["feature", "hidden"] and fw["kind"] == "value", (engine, fw)
    assert fs is not None and fs["names"] == ["feature", "hidden"] and fs["kind"] == "scale" and \
        fs["groups"] == [128, 128], (engine, fs)
    assert fw["pair"] == fs["serial"] and fs["pair"] == fw["serial"], (engine, fw, fs)
    assert KC.fact_of(model.norm.weight)["life"] == "const", "every other parameter is a constant after loading"


def main():
    for engine in ("vllm", "sglang"):
        r = D.Declarations(table(engine), cuda_only=False)
        model = Model()
        r.around(process)(model, "cuda:0")
        check_weight(model, engine)
        print(f"ok {engine}'s table: a block-FP8 weight and its scale, processed into new tensors, get their axes "
              f"(feature, hidden) and their pairing (scale groups 128 x 128); the other parameters are constants")

    # SGLang's activation quantizer: the value and its scale made together
    r = D.Declarations(table("sglang"), cuda_only=False)
    spec = r.choice("functions")["sglang.kernels.ops.quantization.fp8_kernel:per_token_group_quant_fp8"]

    def per_token_group_quant_fp8(x, group_size, eps=1e-10, column_major_scales=False):
        q = x.to(torch.float8_e4m3fn)
        return q, torch.ones(x.shape[0], x.shape[1] // group_size)

    quant = r.wrap_function(per_token_group_quant_fp8, spec)
    q, s = quant(torch.randn(6, 512), 128)
    fq, fs = KC.fact_of(q), KC.fact_of(s)
    assert fq["names"] == ["token", "hidden"] and fs["names"] == ["token", "hidden"] and fs["groups"] == [1, 128] \
        and fq["pair"] == fs["serial"], (fq, fs)
    print("ok SGLang's activation quantizer: the value (token, hidden) and its scale (groups of 128 along hidden) are "
          "one issue")
    q2, s2 = quant(torch.randn(6, 512), group_size=512)
    assert KC.fact_of(s2)["groups"] == [1, 512], KC.fact_of(s2)
    print("ok the group size is read from the call (512: one scale per token)")

    # each adapter's own reading point
    from entail.adapters import sglang_declarations as SD
    from entail.adapters import vllm_declarations as VD

    mod = types.ModuleType("sglang.srt.model_loader.loader")

    class DefaultModelLoader:
        @staticmethod
        def postprocess_weights(model, target_device):
            return process(model, target_device)

        @staticmethod
        def load_weights_and_postprocess(model, weights, target_device):
            DefaultModelLoader.postprocess_weights(model, target_device)

    mod.DefaultModelLoader = DefaultModelLoader
    sys.modules[mod.__name__] = mod
    SD._reader().cuda_only = False
    try:
        assert SD.install_loader() == 1 and SD.install_loader() == 0, "installed once"
        assert isinstance(DefaultModelLoader.__dict__["postprocess_weights"], staticmethod)
        model = Model()
        DefaultModelLoader.load_weights_and_postprocess(model, None, "cuda:0")
        check_weight(model, "sglang adapter")
        print("ok SGLang's adapter reads at DefaultModelLoader.postprocess_weights (a static method, kept one)")
    finally:
        SD.uninstall()
        sys.modules.pop(mod.__name__, None)

    vmod = types.ModuleType("vllm.model_executor.model_loader.utils")
    vmod.process_weights_after_loading = process
    sys.modules[vmod.__name__] = vmod
    VD._reader().cuda_only = False
    try:
        assert VD.install_loader() == 1
        model = Model()
        vmod.process_weights_after_loading(model, None, "cuda:0")
        check_weight(model, "vllm adapter")
        print("ok vLLM's adapter reads at model_loader.utils.process_weights_after_loading, as before")
    finally:
        VD.uninstall()
        sys.modules.pop(vmod.__name__, None)

    # a tensor that already has a meaning keeps it
    t = torch.zeros(4, 512)
    KC.attach(t, ["token", "hidden"])
    r = D.Declarations(table("sglang"), cuda_only=False)
    assert r.attach(t, ["feature", "hidden"]) is False and KC.fact_of(t)["names"] == ["token", "hidden"]
    print("ok a tensor that already has a meaning keeps it")


if __name__ == "__main__":
    main()
