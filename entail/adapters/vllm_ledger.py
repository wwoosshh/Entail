"""D-arm measurement: follow LAYOUT facts through vLLM's weight post-processing and record where they are lost.

vLLM does carry format facts, but it carries them in the *type* of each parameter: ModelWeightParameter,
PackedvLLMParameter, GroupQuantScaleParameter and friends hold output_dim, input_dim, packed_dim, packed_factor
and marlin_tile_size (model_executor/parameter.py). The question this ledger answers is how much of that is
still attached once the weights have been repacked for the kernel that will read them.

Two hooks, both installed by the sitecustomize shim so they are in place before any consumer binds the names:
  process_weights_after_loading (model_loader/utils.py) - snapshot every quantised layer before and after
  replace_parameter (quantization/utils/layer_utils.py) - the re-registration itself

This is a ledger, not a check: it never raises. Set ENTAIL_LEDGER to a file path to switch it on; each
process writes <path>.<pid>.json.
"""
import atexit
import json
import os

ROLE_ATTRS = ("output_dim", "input_dim", "packed_dim", "packed_factor", "marlin_tile_size")
LEDGER = {"layers": [], "replace_parameter": [], "pid": os.getpid()}
_written = False


def _snapshot(param):
    facts = {}
    for a in ROLE_ATTRS:
        try:
            v = getattr(param, a)
        except (AttributeError, Exception):  # properties on these classes can raise when unset
            continue
        facts[a] = repr(v)
    return {"class": type(param).__name__, "dtype": str(getattr(param, "dtype", None)),
            "shape": list(getattr(param, "shape", [])), "facts": facts}


def _write():
    global _written
    path = os.environ.get("ENTAIL_LEDGER")
    if not path or _written:
        return
    _written = True
    with open(f"{path}.{os.getpid()}.json", "w", encoding="utf-8") as f:
        json.dump(LEDGER, f, ensure_ascii=False, indent=1)


def install_loader():
    """Wrap process_weights_after_loading so every quantised layer is snapshotted before and after."""
    import torch
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
    from vllm.model_executor.model_loader import utils as loader_utils

    orig = loader_utils.process_weights_after_loading

    def snapshot_model(model):
        out = {}
        for name, module in model.named_modules():
            qm = getattr(module, "quant_method", None)
            if not isinstance(qm, QuantizeMethodBase):
                continue
            params = {n: _snapshot(p) for n, p in module.named_parameters(recurse=False)}
            if params:
                out[name or "<root>"] = {"quant_method": type(qm).__name__, "params": params}
        return out

    def wrapped(model, model_config, target_device, *a, **kw):
        before = snapshot_model(model)
        out = orig(model, model_config, target_device, *a, **kw)
        after = snapshot_model(model)
        for layer, b in before.items():
            a_ = after.get(layer, {"params": {}})
            row = {"layer": layer, "quant_method": b["quant_method"], "params": {}}
            for pname, bp in b["params"].items():
                ap = a_["params"].get(pname)
                row["params"][pname] = {
                    "before": bp,
                    "after": ap,
                    "class_changed": None if ap is None else (bp["class"] != ap["class"]),
                    "facts_lost": sorted(set(bp["facts"]) - set(ap["facts"])) if ap else sorted(bp["facts"]),
                    "gone": ap is None,
                }
            LEDGER["layers"].append(row)
        _write()
        return out

    loader_utils.process_weights_after_loading = wrapped
    atexit.register(_write)
    return 1


def _patch_replace(module, where):
    """Wrap one module's replace_parameter so each re-registration is recorded with what it kept."""
    orig = module.replace_parameter

    def wrapped(mod, name, new):
        before = _snapshot(getattr(mod, name, None)) if hasattr(mod, name) else None
        out = orig(mod, name, new)
        after = _snapshot(getattr(mod, name, None)) if hasattr(mod, name) else None
        if before is not None and after is not None:
            LEDGER["replace_parameter"].append({
                "from": where, "module": type(mod).__name__, "name": name, "before": before, "after": after,
                "class_changed": before["class"] != after["class"],
                "facts_lost": sorted(set(before["facts"]) - set(after["facts"]))})
        return out

    module.replace_parameter = wrapped
    atexit.register(_write)
    return 1


def install_replace():
    """vllm/model_executor/layers/quantization/utils/layer_utils.py"""
    from vllm.model_executor.layers.quantization.utils import layer_utils

    return _patch_replace(layer_utils, "quantization.utils.layer_utils")


def install_replace_core():
    """vllm/model_executor/utils.py holds a second function of the same name, and the online quantisation
    methods use that one. Both have to be wrapped or the ledger silently records nothing."""
    from vllm.model_executor import utils as me_utils

    return _patch_replace(me_utils, "model_executor.utils")
