"""Adapter v2 for ComfyUI (LIBRARY_DESIGN.md 4.8; ROADMAP M6.2). Where ComfyUI decides, what it decided, and how to
make it use something else; the rules are the core's (load.prediction, load.latent_scale, load.lora).

  hooks        comfy.sd.load_checkpoint_guess_config and load_diffusion_model (a file path) and the state-dict loaders
               they call, load_state_dict_guess_config and load_diffusion_model_state_dict (the tensor names and
               metadata in hand): what the checkpoint declares - its header, and a manifest found by the file's hash
               when ENTAIL_MANIFESTS names folders - is kept with the model it builds (load.remember).
               comfy.sd.load_lora_for_models: a LoRA against the model and the text encoder it is applied to.
               comfy.sample.sample and sample_custom: the prediction and the latent scale the sampling uses, at every
               run (nodes can change them between runs); the same decision for the same model is recorded once.
               nodes.LoraLoader.load_lora: only the file name, for the message.
  read_choice  the prediction the model's sampling object is set up for (which of ComfyUI's prediction classes, and
               zsnr on its discrete schedule), whether a node set it (an object patch: the user's explicit choice),
               the scale and shift of the model's latent format; for a LoRA, its modules for the parts it is given to
               (a strength other than 0) and the ones ComfyUI's own key maps find there (after ComfyUI's own format
               conversion).
  handles      switch_prediction: sample a clone with the declared prediction, the object patch a
               ModelSamplingDiscrete node makes. Only on ComfyUI's discrete schedule, for eps and v, and not once the
               sigmas were computed (sample_custom, or sigmas given). Nothing sets the latent scale here.
Measured on ComfyUI 0.34.1 (issue_track/comfyui_field_test/): the file-declared v model (M7), a LoRA made for another
base model. ComfyUI's own defect (Comfy-Org/ComfyUI#16490) is repaired in comfyui_repair.py, marked as the engine's.
"""
import contextvars
import functools
import importlib
import inspect

from .. import core, load, policies, readers
from ..facts import LatentScale, Prediction
from .base import Hook

engine = "comfyui"
versions = "0.34.1"
_ORIG = {}   # (module or class, attribute) -> original, for uninstall
_PATH = contextvars.ContextVar("entail_comfyui_path", default=None)   # the file a path loader is reading
_LORA = contextvars.ContextVar("entail_comfyui_lora", default=None)   # the LoRA file a LoRA node is loading


def hooks():
    return [Hook("comfy.sd.load_checkpoint_guess_config", "load"), Hook("comfy.sd.load_diffusion_model", "load"),
            Hook("comfy.sd.load_state_dict_guess_config", "load"),
            Hook("comfy.sd.load_diffusion_model_state_dict", "load"), Hook("comfy.sd.load_lora_for_models", "load"),
            Hook("comfy.sample.sample", "load"), Hook("comfy.sample.sample_custom", "load")]


def _active():
    return core.mode() in ("load", "debug")


def _patch(owner, attr, new):
    if (owner, attr) in _ORIG:
        return 0
    _ORIG[(owner, attr)] = getattr(owner, attr)
    setattr(owner, attr, new)
    return 1


def read_choice(kind, *args):
    """What ComfyUI chose, as fact values.
      ("prediction", model_sampling, comfy.model_sampling)  -> Prediction, or None for a class not read here
      ("discrete", model_sampling, comfy.model_sampling)    -> whether it is the discrete schedule with eps or v
      ("latent_scale", latent_format)                       -> LatentScale, or None when a per-channel mean and std
                                                               take part (not a scale and a shift)
      ("lora", LoRA keys, model key map keys or None, text-encoder key map keys or None)
                                                            -> (given, taken, carried) module names; None marks a part
                                                               the LoRA is not applied to"""
    if kind == "prediction":
        ms, m = args
        zsnr = bool(getattr(ms, "zsnr", False)) if isinstance(ms, m.ModelSamplingDiscrete) else None
        for cls, value in ((m.EDM, "edm"), (m.V_PREDICTION, "v"), (getattr(m, "V_PREDICTION_DDPM", ()), "v"),
                           (m.X0, "x0"), (m.CONST, "flow"), (getattr(m, "COSMOS_RFLOW", ()), "flow"), (m.EPS, "eps")):
            if cls and isinstance(ms, cls):
                return Prediction(value, zsnr)
        return None
    if kind == "discrete":
        ms, m = args
        return (isinstance(ms, m.ModelSamplingDiscrete) and not isinstance(ms, m.ModelSamplingDiscreteEDM)
                and isinstance(ms, m.EPS) and not isinstance(ms, (m.X0, m.EDM)))
    if kind == "latent_scale":
        (lf,) = args
        if any(getattr(lf, n, None) is not None for n in ("latents_mean", "latents_std")):
            return None
        scale, shift = getattr(lf, "scale_factor", None), getattr(lf, "shift_factor", None)
        return LatentScale(float(scale), shift=None if shift is None else float(shift)) if scale else None
    if kind == "lora":
        keys, model_keys, text_keys = args
        carried = readers.lora_modules(keys)
        text = {m for m in carried if readers.is_text_module(m)}
        given, taken = set(), set()
        if model_keys is not None:
            given |= carried - text
            taken |= (carried - text) & set(model_keys)
        if text_keys is not None:
            given |= text
            taken |= text & set(text_keys)
        return given, taken, carried
    raise ValueError(f"comfyui.read_choice: unknown kind {kind!r}")


def handles(model, ms_mod):
    """switch_prediction(target): a clone of the model patcher sampled as `target` (eps or v), as a
    ModelSamplingDiscrete node sets it up."""
    def switch_prediction(target):
        class ModelSamplingAdvanced(ms_mod.ModelSamplingDiscrete,
                                    ms_mod.V_PREDICTION if target.kind == "v" else ms_mod.EPS):
            pass

        patched = model.clone()
        patched.add_object_patch("model_sampling", ModelSamplingAdvanced(model.model.model_config, zsnr=target.zsnr))
        return patched

    return {"switch_prediction": switch_prediction}


# --- loading: what the checkpoint declares stays with the model ----------------------------------------------------

def _path_loader(orig):
    @functools.wraps(orig)
    def load_path(path, *a, **kw):
        token = _PATH.set(path)
        try:
            return orig(path, *a, **kw)
        finally:
            _PATH.reset(token)

    return load_path


def _state_dict_loader(orig):
    sig = inspect.signature(orig)

    @functools.wraps(orig)
    def load_state_dict(sd, *a, **kw):
        if not _active():
            return orig(sd, *a, **kw)
        path = _PATH.get()
        header = None
        if path is None:   # a custom loader that has the state dict and metadata, but no file
            try:
                header = (list(sd.keys()), sig.bind_partial(sd, *a, **kw).arguments.get("metadata"),
                          "a state dict ComfyUI loaded")
            except Exception:  # noqa: BLE001 - reading what is declared must never break loading
                header = None
        out = orig(sd, *a, **kw)
        patcher = out[0] if isinstance(out, tuple) else out
        model = getattr(patcher, "model", None)
        if model is not None and (path is not None or header is not None):
            load.safely("load:comfyui.checkpoint", "comfyui.loader", "Prediction",
                        lambda: load.remember(model, load.declared(path, header=header)))
        return out

    return load_state_dict


def install():
    """Hooks on comfy.sd: the loaders, and the LoRA contract. Returns hooks installed."""
    # This runs the moment comfy.sd finishes executing, before Python binds it as an attribute of the comfy package,
    # so the modules are taken directly ("cannot access submodule 'sd'" otherwise).
    sd_mod = importlib.import_module("comfy.sd")
    lora_mod = importlib.import_module("comfy.lora")
    convert = importlib.import_module("comfy.lora_convert")
    n = 0
    for name in ("load_checkpoint_guess_config", "load_diffusion_model"):
        if callable(getattr(sd_mod, name, None)):
            n += _patch(sd_mod, name, _path_loader(getattr(sd_mod, name)))
    for name in ("load_state_dict_guess_config", "load_diffusion_model_state_dict"):
        if callable(getattr(sd_mod, name, None)):
            n += _patch(sd_mod, name, _state_dict_loader(getattr(sd_mod, name)))
    orig_lora = sd_mod.load_lora_for_models

    def load_lora_for_models(model, clip, lora, strength_model, strength_clip, *a, **kw):
        if _active():
            def work():
                model_keys = lora_mod.model_lora_keys_unet(model.model, {}).keys() \
                    if model is not None and strength_model != 0 else None
                text_keys = lora_mod.model_lora_keys_clip(clip.cond_stage_model, {}).keys() \
                    if clip is not None and strength_clip != 0 else None
                if model_keys is None and text_keys is None:
                    return
                given, taken, carried = read_choice("lora", convert.convert_lora(lora).keys(), model_keys, text_keys)
                meta = kw.get("lora_metadata", a[0] if a else None)
                target = type(model.model.model_config).__name__ if model is not None else "text encoder"
                load.enforce(load.lora(engine, f"{_LORA.get() or 'a LoRA'} on the {target} model", given, taken,
                                       base=readers.lora_base(meta), policy=policies.current(), carried=carried))

            load.safely("load:comfyui.lora", "comfyui.lora_loader", "Coverage", work)
        return orig_lora(model, clip, lora, strength_model, strength_clip, *a, **kw)

    return n + _patch(sd_mod, "load_lora_for_models", load_lora_for_models)


def install_nodes():
    """Remember which file the built-in LoRA nodes are loading, for the message. 0 if this is not ComfyUI."""
    import sys

    loader = getattr(sys.modules.get("nodes"), "LoraLoader", None)
    if loader is None:
        return 0
    orig = loader.load_lora

    def load_lora(self, model, clip, lora_name, *a, **kw):
        token = _LORA.set(lora_name)
        try:
            return orig(self, model, clip, lora_name, *a, **kw)
        finally:
            _LORA.reset(token)

    return _patch(loader, "load_lora", load_lora)


# --- sampling: the prediction and the latent scale in use -----------------------------------------------------------

def _decide(model, precomputed, ms_mod):
    """The decisions for this sampling run, and the model patcher to sample with."""
    base = model.model
    facts = load.remembered(base) or load.Declared()
    ms = model.get_model_object("model_sampling")
    where = type(base.model_config).__name__
    switchable = () if precomputed or not read_choice("discrete", ms, ms_mod) else ("eps", "v")
    why = "" if switchable else ("the sigmas were computed before sampling, from the set-up it had" if precomputed
                                 else "it is not ComfyUI's discrete eps/v schedule, which is what entail switches")
    policy = policies.current()
    decisions = load.prediction(engine, "comfyui.model_sampling", facts, read_choice("prediction", ms, ms_mod),
                                explicit="model_sampling" in model.object_patches, can_switch=switchable,
                                policy=policy, where=where, note=why)
    decisions += load.latent_scale(engine, "comfyui.latent_format", facts,
                                   read_choice("latent_scale", model.get_model_object("latent_format")),
                                   can_switch=False, policy=policy, where=where)
    load.enforce(decisions, once_for=base)
    return load.resolve(decisions, handles(model, ms_mod)).get("switch_prediction", model)


def install_sampling():
    """Wrap comfy.sample.sample and sample_custom. Returns hooks installed."""
    sample_mod = importlib.import_module("comfy.sample")
    ms_mod = importlib.import_module("comfy.model_sampling")
    n = 0
    for fname, custom in (("sample", False), ("sample_custom", True)):
        orig = getattr(sample_mod, fname)
        sig = inspect.signature(orig)

        def run(model, *a, _orig=orig, _custom=custom, _sig=sig, **kw):
            if not _active():
                return _orig(model, *a, **kw)
            precomputed = _custom or _sig.bind_partial(model, *a, **kw).arguments.get("sigmas") is not None
            chosen = load.safely("load:comfyui.prediction", "comfyui.model_sampling", "Prediction",
                                 lambda: _decide(model, precomputed, ms_mod), default=model)
            return _orig(chosen, *a, **kw)

        n += _patch(sample_mod, fname, run)
    return n


def uninstall():
    n = 0
    for (owner, attr), orig in list(_ORIG.items()):
        setattr(owner, attr, orig)
        n += 1
    _ORIG.clear()
    return n
