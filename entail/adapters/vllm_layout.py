"""Adapter v2 for the step where vLLM leaves its weights in the layout their kernels read (LIBRARY_DESIGN.md 4.6,
4.8; ROADMAP M4.2; entail/audits/D_LEDGER.md results 3, 4 and 4-1).

After loading, every layer's quantisation method runs process_weights_after_loading, and its kernel (apply) then
reads the weight as that step left it. The ledger showed that the facts about which axis is which are erased exactly
there: fp8 stores the transpose, and nothing downstream says so. entail owns the wrapper it installs around that
step, so it signs it. Each method's signature - the layout the step leaves the weight in, which its kernel reads,
and that it moves no value - is data (data/signatures.json); load.weights_written compares it with the tensors and
puts the confirmed layout on each weight, so it travels with the value.

  hook         vllm.model_executor.model_loader.utils.process_weights_after_loading: every layer's quantisation
               method runs its step there.
  read_choice  for every layer with a quantisation method: the method (the producer), the weight, the layer's own
               in and out features, and its scale tensor.
  handles      "layout.contiguous": pack the weights of the layers named (a resolution that only the use_data
               policy reaches: by default a signature the data contradicts is refused).
load.sample_weights takes a few values of each weight before the step and load.weights_written decides after it;
load.enforce records the decisions and stops on a blocking one.
"""
from .. import core, load, policies
from .base import Hook

engine = "vllm"
versions = "0.30.0"


def hooks():
    return [Hook("vllm.model_executor.model_loader.utils.process_weights_after_loading", "load")]


def read_choice(model):
    """A load.Weight for every layer that has a quantisation method; the producer is named after the method."""
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

    out = []
    for name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        if not isinstance(method, QuantizeMethodBase):
            continue
        out_features = getattr(module, "output_size_per_partition", None)
        if out_features is None:   # a fused layer that only lists its parts
            parts = getattr(module, "output_partition_sizes", None)
            out_features = sum(parts) if parts else None
        scale = getattr(module, "weight_scale", None)   # vLLM's two names for a weight's scale
        if scale is None:
            scale = getattr(module, "weight_scale_inv", None)
        out.append(load.Weight(name or "<root>", f"{engine}.quant_method.{type(method).__name__}",
                               getattr(module, "weight", None), getattr(module, "input_size_per_partition", None),
                               out_features, scale))
    return out


def handles(model):
    def contiguous(layers):
        for layer in layers:
            weight = model.get_submodule("" if layer == "<root>" else layer).weight
            weight.data = weight.data.contiguous()
        return layers

    return {"layout.contiguous": contiguous}


def install():
    """Wrap process_weights_after_loading: sample before the step, decide after it."""
    from vllm.model_executor.model_loader import utils as loader_utils

    orig = loader_utils.process_weights_after_loading
    boundary, consumer = f"load:{engine}.quant_method.process_weights_after_loading", f"{engine}.quant_method"

    def wrapped(model, model_config, target_device, *a, **kw):
        active = core.mode() in ("load", "debug")
        before = load.safely(boundary, consumer, "Coverage", lambda: load.sample_weights(read_choice(model)),
                             {}) if active else {}
        out = orig(model, model_config, target_device, *a, **kw)
        if active:
            def decide():
                done = load.enforce(load.weights_written(boundary, read_choice(model), before, policies.current()))
                load.resolve(done, handles(model))

            load.safely(boundary, consumer, "Layout", decide)
        return out

    loader_utils.process_weights_after_loading = wrapped
    return 1
