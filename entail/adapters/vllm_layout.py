"""D-arm check: after vLLM has repacked the weights, do they still match the layout the kernel expects?

The ledger (adapters/vllm_ledger.py) showed that the axis facts are erased exactly where the tensor is
transposed, so nothing downstream can verify the orientation. This check puts the fact back: each quantisation
method declares the layout its kernel reads, and the check compares that declaration against the tensors the
layer actually holds, using the layer's own input/output sizes as the ground truth.

It runs once, after process_weights_after_loading, and raises RoleError on a mismatch. What it can catch is a
repack that produced the wrong orientation, dtype or scale granularity - the silent kind, because the shapes
still multiply and the model still answers.

Declared layouts come from reading the kernels; each row says where. Unknown methods are reported, not ignored.
"""
import os
import time

from .. import core
from ..core import RoleError

# quant-method class name -> (weight dtype name or None for "model dtype", orientation, scale attr, scale kind,
#                             what post-load processing does to the weight, why)
#   orientation "out_in": weight.shape == (out_features, in_features)   (the usual torch.nn.Linear layout)
#   orientation "in_out": weight.shape == (in_features, out_features)   (transposed for the kernel)
#   scale kind "per_tensor" (numel 1), "per_channel" (numel == out_features), None (no scale)
#   post "identity": process_weights_after_loading must leave the weight exactly as it found it. Both methods
#     here quantise while the checkpoint is read, so by then the weight is final; a method that repacks (Marlin,
#     AWQ) would declare its own transform instead.
LAYOUTS = {
    "UnquantizedLinearMethod": (None, "out_in", None, None, "identity",
                                "torch.nn.functional.linear reads (out, in)"),
    "Fp8PerTensorOnlineLinearMethod": ("float8_e4m3fn", "in_out", "weight_scale", "per_tensor", "identity",
                                       "online/fp8.py stores qweight.t() with one scale per tensor"),
}
# Which methods hand the weight to a kernel that reads it row by row, so it must be packed rather than a
# strided view. This is the fact behind benchmark case 03 (SGLang #31641: a kernel given a sliced, strided
# tensor read it as if it were packed); shape and dtype are identical, only the strides tell.
#
# The first version of this set also listed the fp8 method and was wrong: a healthy fp8 server produced 144
# complaints, because `online/fp8.py` stores `qweight.t()` - shape (in, out) with strides (1, in) - on purpose
# and its kernel reads it that way. The measurement on a healthy run is what fixed the declaration.
PACKED = {"UnquantizedLinearMethod"}
SAMPLE_K = 16  # elements per weight kept to check the declared post-load transform
SKIP = ("UnquantizedEmbeddingMethod", "Fp8KVCacheMethod")  # not linear weights; sizes mean something else


def _sizes(layer):
    i = getattr(layer, "input_size_per_partition", None)
    o = getattr(layer, "output_size_per_partition", None)
    if o is None:
        sizes = getattr(layer, "output_partition_sizes", None)
        o = sum(sizes) if sizes else None
    return i, o


def check_layer(name, layer, method_name):
    """Return a list of complaints for one layer."""
    if method_name in SKIP:
        return []
    weight = getattr(layer, "weight", None)
    if weight is None:
        return []
    in_f, out_f = _sizes(layer)
    if in_f is None or out_f is None:
        return []
    if method_name not in LAYOUTS:
        return [f"{name}: {method_name} declares no layout, so the weight shape {tuple(weight.shape)} "
                f"was not checked"]
    import torch

    want_dtype, orientation, scale_attr, scale_kind, _post, why = LAYOUTS[method_name]
    out = []
    if want_dtype is not None and weight.dtype != getattr(torch, want_dtype):
        out.append(f"{name}: {method_name} declares dtype {want_dtype}, weight is {weight.dtype} ({why})")
    want_shape = (out_f, in_f) if orientation == "out_in" else (in_f, out_f)
    if tuple(weight.shape) != want_shape:
        out.append(f"{name}: {method_name} declares {orientation} layout {want_shape}, "
                   f"weight is {tuple(weight.shape)} ({why})")
    if method_name in PACKED and not weight.is_contiguous():
        out.append(f"{name}: {method_name} reads the weight row by row, so it must be packed, but it is a "
                   f"strided view (shape {tuple(weight.shape)}, strides {tuple(weight.stride())})")
    if scale_attr:
        scale = getattr(layer, scale_attr, None)
        if scale is None:
            out.append(f"{name}: {method_name} declares a {scale_kind} {scale_attr}, but the layer has none")
        else:
            n = scale.numel()
            want = 1 if scale_kind == "per_tensor" else out_f
            if n != want:
                out.append(f"{name}: {method_name} declares a {scale_kind} {scale_attr} ({want} value(s)), "
                           f"the layer holds {n}")
    return out


def sample(weight):
    """A few evenly spread elements of a weight, as float32, so a later copy can be compared against it.

    Shapes and dtypes only say what kind of tensor this is; these values say which tensor it is. That is what
    catches a repack that kept the shape but moved the data (a wrong packing order, a shifted row).
    """
    import torch

    n = weight.numel()
    if n == 0:
        return None
    # integer arithmetic on purpose: linspace computes in float32, and past 2**24 elements it can round an
    # index up past the end of the tensor, which on CUDA is a device-side assert that kills the process.
    step = max(1, n // SAMPLE_K)
    idx = (torch.arange(SAMPLE_K, dtype=torch.long, device=weight.device) * step).clamp_(max=n - 1)
    vals = weight.detach().reshape(-1).index_select(0, idx).to(torch.float32).tolist()
    return {"idx": idx, "vals": vals, "shape": tuple(weight.shape), "dtype": weight.dtype}


def check_post_transform(name, weight, before, declared):
    """Compare the weight against the sample taken before post-processing, under the declared transform."""
    if before is None or weight is None or declared != "identity":
        return []
    if tuple(weight.shape) != before["shape"] or weight.dtype != before["dtype"]:
        return [f"{name}: post-load processing is declared as identity, but the weight went from "
                f"{before['shape']} {before['dtype']} to {tuple(weight.shape)} {weight.dtype}"]
    now = weight.detach().reshape(-1).index_select(0, before["idx"]).to("cpu").to(float).tolist()
    bad = [k for k, (a, b) in enumerate(zip(before["vals"], now)) if a != b]
    if bad:
        return [f"{name}: post-load processing is declared as identity, but {len(bad)} of {SAMPLE_K} sampled "
                f"values moved (first at flat index {int(before['idx'][bad[0]])}: "
                f"{before['vals'][bad[0]]:.6g} -> {now[bad[0]]:.6g})"]
    return []


def install():
    """Wrap process_weights_after_loading so the layouts are checked once, right after the repack."""
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
    from vllm.model_executor.model_loader import utils as loader_utils

    orig = loader_utils.process_weights_after_loading

    def wrapped(model, model_config, target_device, *a, **kw):
        active = core.mode() in ("load", "debug")
        before = {}
        if active:
            for name, module in model.named_modules():
                qm = getattr(module, "quant_method", None)
                if isinstance(qm, QuantizeMethodBase) and type(qm).__name__ in LAYOUTS:
                    w = getattr(module, "weight", None)
                    if w is not None:
                        before[name] = sample(w)
        out = orig(model, model_config, target_device, *a, **kw)
        if active:
            complaints, checked = [], 0
            t0 = time.perf_counter()
            for name, module in model.named_modules():
                qm = getattr(module, "quant_method", None)
                if isinstance(qm, QuantizeMethodBase):
                    checked += 1
                    method = type(qm).__name__
                    complaints += check_layer(name or "<root>", module, method)
                    if name in before and method in LAYOUTS:
                        complaints += check_post_transform(name, getattr(module, "weight", None),
                                                           before[name], LAYOUTS[method][4])
            if os.environ.get("ENTAIL_VERBOSE"):
                print(f"[entail] layout check: {checked} layers, {len(complaints)} complaints, "
                      f"{(time.perf_counter() - t0) * 1e3:.1f} ms", flush=True)
            if complaints:
                head = complaints[:5]
                more = f"\n  ... and {len(complaints) - len(head)} more" if len(complaints) > len(head) else ""
                raise RoleError("weights do not match the declared layout:\n  " + "\n  ".join(head) + more)
        return out

    loader_utils.process_weights_after_loading = wrapped
    return 1
