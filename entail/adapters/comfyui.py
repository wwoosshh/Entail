"""Adapter: ComfyUI. A LoRA has to reach the model it is applied to.

ComfyUI maps each LoRA module to a weight of the loaded model. A module with no counterpart is skipped with one
console line ("lora key not loaded") and the workflow goes on and succeeds, so a LoRA made for another base model
does nothing while the UI shows nothing wrong. Measured on a real install (ComfyUI v0.34.1): an Anima LoRA in an
SDXL workflow logged 840 such lines, finished "success", and moved the image by 0.8 of 255 on average against no
LoRA at all; the matching SDXL LoRA moved it by 35.2.

The check runs just before the patch is applied, in comfy.sd.load_lora_for_models, which the built-in LoRA nodes
and most custom LoRA loaders call. It counts how many of the LoRA's modules have a counterpart in the model (and in
the text encoder when one is given):
  none  the LoRA can change nothing and there is nothing to convert, so the workflow stops with the reason - what
        the LoRA declares it was trained for, and which model it met (RoleError; ComfyUI shows it as a node error)
  some  one line saying how many are left out; the workflow goes on
  all   nothing is said (ENTAIL_VERBOSE=1 prints a line anyway)
"""
import contextvars
import os
import re

from .. import core

_ORIG_APPLY = None
_ORIG_NODE = None
_name = contextvars.ContextVar("entail_lora_name", default=None)

# What follows a module name in the LoRA formats ComfyUI reads (kohya, peft/diffusers, LyCORIS and friends).
_PART = re.compile(r"\.(?:lora_up|lora_down|lora_A|lora_B|lora_mid|lora\.up|lora\.down|alpha|dora_scale|hada_\w+|"
                   r"lokr_\w+|diff|diff_b|set_weight|w_norm|b_norm|reshape_weight|lora_linear_layer)(?:\.|$)")
_TEXT = ("lora_te", "text_encoder", "text_model", "te_", "te1_", "te2_", "clip_l", "clip_g", "t5xxl", "lora_te1",
         "lora_te2", "lora_te3")


def module_of(key):
    """The module a LoRA tensor belongs to: 'lora_unet_x.lora_down.weight' -> 'lora_unet_x'."""
    m = _PART.search(key)
    return key[:m.start()] if m else key.rsplit(".", 1)[0]


def coverage(lora_keys, model_map_keys, clip_map_keys):
    """Pure counting, so it can be tested without ComfyUI.

    lora_keys: tensor names of the (converted) LoRA; *_map_keys: the module names ComfyUI can map for the model and
    for the text encoder (empty when that side is not being patched). Returns the module counts per side."""
    modules = {module_of(k) for k in lora_keys}
    text = {m for m in modules if m.startswith(_TEXT)}
    model = modules - text
    model_map, clip_map = set(model_map_keys), set(clip_map_keys)
    return {"model_total": len(model), "model_matched": len(model & model_map),
            "text_total": len(text), "text_matched": len(text & clip_map)}


def verdict(cov, applied, sides, name, declared, target):
    """None when everything reaches; ('violation', msg) when nothing does; ('partial', msg) otherwise.

    applied: how many weights ComfyUI's own mapping would patch; sides: which of 'model' and 'text' are patched."""
    label = f"LoRA {name}" if name else "a LoRA"
    trained = f" It declares it was trained for {declared}." if declared else ""
    if applied == 0:
        if "model" in sides and cov["model_total"] == 0 and cov["text_total"]:
            why = "it only carries text-encoder modules and was loaded without a text encoder"
        else:
            total = (cov["model_total"] if "model" in sides else 0) + (cov["text_total"] if "text" in sides else 0)
            why = f"none of its {total} modules match a weight of the loaded {target} model"
        return ("violation", f"{label} cannot reach this model: {why}.{trained} It would change nothing. Use the "
                             f"version made for {target}, or remove it.")
    left = []
    if "model" in sides and cov["model_matched"] < cov["model_total"]:
        left.append(f"{cov['model_total'] - cov['model_matched']} of {cov['model_total']} model modules")
    if "text" in sides and cov["text_matched"] < cov["text_total"]:
        left.append(f"{cov['text_total'] - cov['text_matched']} of {cov['text_total']} text-encoder modules")
    if left:
        return ("partial", f"{label}: {' and '.join(left)} have no counterpart in the loaded {target} model and are "
                           f"left out (ComfyUI skips them).{trained}")
    return None


def _declared(meta):
    if not meta:
        return None
    parts = [meta.get("modelspec.architecture"), meta.get("ss_base_model_version")]
    parts = [p for p in parts if p]
    return " / ".join(dict.fromkeys(parts)) or None


def install():
    """Wrap comfy.sd.load_lora_for_models. Returns 1, or 0 if already installed."""
    global _ORIG_APPLY
    import importlib

    # This runs the moment comfy.sd finishes executing, before Python binds it as an attribute of the comfy
    # package, so an attribute access through the package would fail here ("cannot access submodule 'sd'").
    # Take the module objects directly.
    sd = importlib.import_module("comfy.sd")
    lora_mod = importlib.import_module("comfy.lora")
    convert = importlib.import_module("comfy.lora_convert")

    if _ORIG_APPLY is not None:
        return 0
    _ORIG_APPLY = sd.load_lora_for_models

    def load_lora_for_models(model, clip, lora, strength_model, strength_clip, *a, **kw):
        if core.mode() in ("load", "debug"):
            said = None
            try:
                sides = []
                model_map = clip_map = {}
                if model is not None and strength_model != 0:
                    model_map = lora_mod.model_lora_keys_unet(model.model, {})
                    sides.append("model")
                if clip is not None and strength_clip != 0:
                    clip_map = lora_mod.model_lora_keys_clip(clip.cond_stage_model, {})
                    sides.append("text")
                if sides:
                    converted = convert.convert_lora(lora)
                    applied = len(lora_mod.load_lora(converted, {**model_map, **clip_map}, log_missing=False))
                    cov = coverage(converted.keys(), model_map.keys(), clip_map.keys())
                    meta = kw.get("lora_metadata", a[0] if a else None)
                    target = type(getattr(getattr(model, "model", None), "model_config", None)).__name__ \
                        if model is not None else "text encoder"
                    said = verdict(cov, applied, sides, _name.get(), _declared(meta), target)
                    if said is None and os.environ.get("ENTAIL_VERBOSE"):
                        print(f"[entail] LoRA {_name.get() or ''}: all {cov['model_total']} model and "
                              f"{cov['text_total']} text-encoder modules reach the {target} model", flush=True)
            except Exception as e:  # noqa: BLE001 - a check that cannot run must never break the workflow
                print(f"[entail] could not check the LoRA ({type(e).__name__}: {e}); applying it unchecked", flush=True)
                said = None
            if said and said[0] == "violation":
                raise core.RoleError(said[1])
            if said:
                print(f"[entail] {said[1]}", flush=True)
        return _ORIG_APPLY(model, clip, lora, strength_model, strength_clip, *a, **kw)

    sd.load_lora_for_models = load_lora_for_models
    return 1


def install_nodes():
    """Remember which file the built-in LoRA nodes are loading, for the message. 0 if this is not ComfyUI."""
    global _ORIG_NODE
    import sys

    nodes = sys.modules.get("nodes")
    loader = getattr(nodes, "LoraLoader", None)
    if loader is None or _ORIG_NODE is not None:
        return 0
    _ORIG_NODE = loader.load_lora

    def load_lora(self, model, clip, lora_name, *a, **kw):
        token = _name.set(lora_name)
        try:
            return _ORIG_NODE(self, model, clip, lora_name, *a, **kw)
        finally:
            _name.reset(token)

    loader.load_lora = load_lora
    return 1


def uninstall():
    global _ORIG_APPLY, _ORIG_NODE
    import sys

    n = 0
    if _ORIG_APPLY is not None:
        sys.modules["comfy.sd"].load_lora_for_models = _ORIG_APPLY
        _ORIG_APPLY = None
        n += 1
    loader = getattr(sys.modules.get("nodes"), "LoraLoader", None)
    if _ORIG_NODE is not None and loader is not None:
        loader.load_lora = _ORIG_NODE
        _ORIG_NODE = None
        n += 1
    return n
