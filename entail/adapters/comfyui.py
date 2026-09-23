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

Second check: the prediction type. For SD1/SD2/SDXL checkpoints ComfyUI decides v-prediction from one marker, a
`v_pred` key in the file (comfy/supported_models.py). A merge or conversion that drops it makes ComfyUI sample a
v-prediction model as eps, and the run still "succeeds": measured with NoobAI-XL-Vpred, the images came out as
coloured noise or completely black, 89-189 of 255 away from the right ones on average. So the declaration is
checked against the model itself, on the first model call of each sampling - the noisiest step, so no extra
forward pass is needed: an eps model predicts the noise in its input back (cosine ~1.00 measured on 8
checkpoints), a v-prediction model predicts the image (~0.01). A separate probe pass was tried first; it changed
the kernels picked for the first image after start-up (0.6/255 on average), so the check only watches now. When the behaviour contradicts what ComfyUI decided from the marker, the model is
sampled the way it behaves - exactly what a ModelSamplingDiscrete node would do - and one line says so. When it
contradicts a sampling node the workflow itself set, or the sigmas were already computed (custom samplers), it stops
with the reason instead: an explicit choice is not overridden silently.
"""
import contextvars
import importlib
import os
import re

from .. import core
from . import _shared

_ORIG_APPLY = None
_ORIG_NODE = None
_ORIG_SAMPLE = None
_ORIG_SAMPLE_CUSTOM = None
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


# Halfway between what theory predicts - eps: cos = sigma_t (0.99 at the last timestep, >0.9 from the middle up),
# v: ~0 at every timestep - so not fitted to data. Measured with a separate probe at t=999: eps 0.9997-0.9999 on
# 8 checkpoints, v -0.013..0.039 (issue_track/comfyui_field_test/VPRED_PROTOCOL.md).
BOUNDARY = 0.5
MIN_T = 500  # below this the eps value (sigma_t) nears the boundary; the first call of a low-denoise img2img is skipped
_seen = {}  # id(base model) -> 'eps' | 'v_prediction', as measured on its first sampling


def behaves_like(cos):
    return "eps" if cos > BOUNDARY else "v_prediction"


def sampling_kind(ms, ms_mod):
    """'eps' | 'v_prediction' for the plain discrete schedules this check has been measured on; None otherwise."""
    if not isinstance(ms, ms_mod.ModelSamplingDiscrete) or isinstance(ms, ms_mod.ModelSamplingDiscreteEDM):
        return None
    names = {c.__name__ for c in type(ms).__mro__}
    if isinstance(ms, (ms_mod.EDM, ms_mod.X0)) or "LCM" in names or "ModelSamplingDiscreteDistilled" in names:
        return None
    if isinstance(ms, ms_mod.V_PREDICTION):
        return "v_prediction"
    return "eps" if isinstance(ms, ms_mod.EPS) else None


def raw_output(kind, x, sigma, denoised, sigma_data=1.0):
    """The network output that ComfyUI turned into `denoised` (inverting calculate_denoised of EPS / V_PREDICTION)."""
    s = sigma.reshape(sigma.shape[:1] + (1,) * (x.ndim - 1)).to(x.dtype)
    if kind == "eps":
        return (x - denoised) / s
    d2 = sigma_data ** 2
    return (x * d2 / (s ** 2 + d2) - denoised) * (s ** 2 + d2) ** 0.5 / (s * sigma_data)


def first_call_cos(kind, x, sigma, denoised, sigma_data=1.0):
    """cos(network output, network input) for one model call, as the probe of VPRED_PROTOCOL measures it."""
    import torch.nn.functional as F

    s = sigma.reshape(sigma.shape[:1] + (1,) * (x.ndim - 1)).to(x.dtype)
    xin = x / (s ** 2 + sigma_data ** 2) ** 0.5
    out = raw_output(kind, x, sigma, denoised, sigma_data)
    return F.cosine_similarity(out.float().flatten(), xin.float().flatten(), dim=0).item()


class _Mismatch(Exception):
    def __init__(self, cos):
        super().__init__(cos)
        self.cos = cos


def _switched(model, actual, ms_mod):
    class ModelSamplingAdvanced(ms_mod.ModelSamplingDiscrete,
                                ms_mod.V_PREDICTION if actual == "v_prediction" else ms_mod.EPS):
        pass

    patched = model.clone()
    patched.add_object_patch("model_sampling", ModelSamplingAdvanced(model.model.model_config, zsnr=None))
    return patched


def _contradiction(model, declared, actual, cos, custom, ms_mod):
    """What to do when the model does not behave as it was set up: the model to sample, or RoleError."""
    target = type(model.model.model_config).__name__
    probe = f"{cos:.2f}" if cos is not None else "measured on an earlier run"
    seen = (f"the model behaves like {actual} (probe {probe}; eps models give about 1.00, v-prediction models "
            f"about 0.00)")
    if "model_sampling" in model.object_patches:
        raise core.RoleError(f"This workflow sets the model's sampling to {declared} (ModelSamplingDiscrete or a "
                             f"similar node), but {seen}. Sampled this way the image comes out as noise or black. "
                             f"Set it to {actual}, or remove that node.")
    why = "it found no 'v_pred' marker in the file" if declared == "eps" else "the file carries a 'v_pred' marker"
    if custom or core.policy() == "refuse":
        where = ("The sigmas for this sampler were already computed for that setting, so entail cannot switch it "
                 "here. " if custom else "")
        raise core.RoleError(f"ComfyUI set this {target} checkpoint up as {declared} because {why}, but {seen}. "
                             f"{where}Add ModelSamplingDiscrete({actual}) right after the model loader.")
    _shared.note({"engine": "comfyui", "where": "prediction type", "from": declared, "to": actual, "probe": cos},
                 f"prediction type: ComfyUI set this {target} checkpoint up as {declared} because {why}, but {seen}; "
                 f"sampling it as {actual}, as a ModelSamplingDiscrete({actual}) node would")
    return _switched(model, actual, ms_mod)


def _sample_checked(orig, model, args, kwargs, custom, ms_mod):
    """Sample, watching the first model call. No extra forward pass: the first call already sees the noisiest
    input, so its own output is the probe. A contradiction found there costs that one call and a restart."""
    try:
        ms = model.get_model_object("model_sampling")
        declared = sampling_kind(ms, ms_mod)
        applies = declared is not None and type(model.model.diffusion_model).__name__ == "UNetModel"
    except Exception:  # noqa: BLE001
        applies = False
    if not applies:
        return orig(model, *args, **kwargs)
    base = model.model
    known = _seen.get(id(base))
    if known == declared:
        return orig(model, *args, **kwargs)
    if known is not None:  # measured on an earlier run of this model: no need to watch again
        return orig(_contradiction(model, declared, known, None, custom, ms_mod), *args, **kwargs)

    watched = model.clone()
    before = watched.model_options.get("model_function_wrapper")
    state = {"done": False}
    sigma_data = float(getattr(ms, "sigma_data", 1.0))

    def wrapper(apply_model, a):
        out = before(apply_model, a) if before else apply_model(a["input"], a["timestep"], **a["c"])
        if not state["done"]:
            state["done"] = True
            try:
                t = float(ms.timestep(a["timestep"]).max())
                cos = first_call_cos(declared, a["input"], a["timestep"], out, sigma_data) if t >= MIN_T else None
            except Exception as e:  # noqa: BLE001 - never break sampling because the check could not run
                print(f"[entail] could not check the prediction type ({type(e).__name__}: {e})", flush=True)
                cos = None
            if cos is not None:
                _seen[id(base)] = behaves_like(cos)
                if behaves_like(cos) != declared:
                    raise _Mismatch(cos)
                if os.environ.get("ENTAIL_VERBOSE"):
                    print(f"[entail] {type(base.model_config).__name__} model behaves like {declared} as set up "
                          f"(probe {cos:.2f})", flush=True)
        return out

    watched.set_model_unet_function_wrapper(wrapper)
    try:
        return orig(watched, *args, **kwargs)
    except _Mismatch as m:
        actual = "eps" if declared == "v_prediction" else "v_prediction"
        return orig(_contradiction(model, declared, actual, m.cos, custom, ms_mod), *args, **kwargs)


def install_sampling():
    """Wrap comfy.sample.sample and sample_custom. Returns 1, or 0 if already installed."""
    global _ORIG_SAMPLE, _ORIG_SAMPLE_CUSTOM
    sample_mod = importlib.import_module("comfy.sample")
    ms_mod = importlib.import_module("comfy.model_sampling")
    if _ORIG_SAMPLE is not None:
        return 0
    _ORIG_SAMPLE, _ORIG_SAMPLE_CUSTOM = sample_mod.sample, sample_mod.sample_custom

    def sample(model, *a, **kw):
        if core.mode() in ("load", "debug"):
            return _sample_checked(_ORIG_SAMPLE, model, a, kw, False, ms_mod)
        return _ORIG_SAMPLE(model, *a, **kw)

    def sample_custom(model, *a, **kw):
        if core.mode() in ("load", "debug"):
            return _sample_checked(_ORIG_SAMPLE_CUSTOM, model, a, kw, True, ms_mod)
        return _ORIG_SAMPLE_CUSTOM(model, *a, **kw)

    sample_mod.sample, sample_mod.sample_custom = sample, sample_custom
    return 1


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
    global _ORIG_APPLY, _ORIG_NODE, _ORIG_SAMPLE, _ORIG_SAMPLE_CUSTOM
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
    if _ORIG_SAMPLE is not None:
        mod = sys.modules["comfy.sample"]
        mod.sample, mod.sample_custom = _ORIG_SAMPLE, _ORIG_SAMPLE_CUSTOM
        _ORIG_SAMPLE = _ORIG_SAMPLE_CUSTOM = None
        n += 1
    return n
