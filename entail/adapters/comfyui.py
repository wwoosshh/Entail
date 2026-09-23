"""Adapter: ComfyUI. Where to hook, how to read what ComfyUI decided, how to make it use something else.

The checks themselves are entail's own and engine-independent:
  coverage.py   a LoRA's modules that reach nothing in the model (MAPPING)
  declared.py   what a checkpoint says about itself (metadata, marker keys), read from the state dict and metadata
                ComfyUI already has in hand and then drops
  contract.py   that declaration against what ComfyUI set up; behaviour.py gives evidence when nothing is declared
  ownership.py  a schedule stays with the object that set it
This file only connects them to ComfyUI v0.34 (measured there: issue_track/comfyui_field_test/):

  LoRA            comfy.sd.load_lora_for_models; ComfyUI's own key maps say what the model can take. None taken:
                  stop with the reason (measured: an Anima LoRA on SDXL changed the image by 0.8/255 and the run
                  "succeeded"). Some left out: one line.
  declarations    comfy.sd.load_state_dict_guess_config / load_diffusion_model_state_dict receive the metadata
                  and ComfyUI reads only the `v_pred` / `ztsnr` keys of it; the declaration is kept on the model.
  prediction type comfy.sample.sample / sample_custom. What ComfyUI set up is the model's model_sampling; switching
                  is an object patch, as a ModelSamplingDiscrete node does. With nothing declared, the first model
                  call of the sampling is watched (no extra pass) and its behaviour is the evidence; a contradiction
                  there costs that call and a restart. Custom samplers computed their sigmas already: stop instead.
  schedules       ComfyUI-specific repair of its dynamic VRAM loader, which backs buffers up by path
                  (Comfy-Org/ComfyUI#16490): the setters of comfy.model_sampling record, ModelPatcherDynamic's
                  restore hands backups to their owner, BaseModel.apply_model checks at the first call.
"""
import contextvars
import functools
import importlib
import inspect
import os

from .. import behaviour, contract, core, coverage, declared, ownership
from ..facts import Prediction

_ORIG = {}  # (module or class, attribute) -> original, for uninstall
_SETTERS = []  # (class, name, original) from ownership.wrap_setters
_name = contextvars.ContextVar("entail_lora_name", default=None)
_seen = {}  # id(base model) -> Prediction its first sampling behaved like
WHERE = "ComfyUI"


def _active():
    return core.mode() in ("load", "debug")


def _patch(owner, attr, new):
    if (owner, attr) in _ORIG:
        return 0
    _ORIG[(owner, attr)] = getattr(owner, attr)
    setattr(owner, attr, new)
    return 1


def _target(model):
    return type(getattr(getattr(model, "model", None), "model_config", None)).__name__


# LoRA --------------------------------------------------------------------------------------------------------------

def lora_reach(lora_keys, model_keys, text_keys):
    """How much of a LoRA reaches the model and the text encoder: {'model': Coverage, 'text': Coverage}."""
    mods = declared.lora_modules(lora_keys) or {declared.lora_module(k) for k in lora_keys}
    text = {m for m in mods if declared.is_text_module(m)}
    return {"model": coverage.count(mods - text, model_keys), "text": coverage.count(text, text_keys)}


def lora_verdict(reach, applied, sides, name, base, target):
    """None when everything reaches; ('violation', msg) when nothing does; ('partial', msg) otherwise.
    applied: how many weights ComfyUI's own mapping would patch; sides: which of 'model' and 'text' are patched."""
    label = f"LoRA {name}" if name else "a LoRA"
    trained = f" It declares it was trained for {base}." if base else ""
    if applied == 0:
        if "model" in sides and reach["model"].total == 0 and reach["text"].total:
            why = "it only carries text-encoder modules and was loaded without a text encoder"
        else:
            total = sum(reach[s].total for s in sides)
            why = f"none of its {total} modules match a weight of the loaded {target} model"
        return ("violation", f"{label} cannot reach this model: {why}.{trained} It would change nothing. Use the "
                             f"version made for {target}, or remove it.")
    left = [f"{reach[s].total - reach[s].taken} of {reach[s].total} {'text-encoder' if s == 'text' else s} modules"
            for s in sides if not reach[s].all_taken]
    if left:
        return ("partial", f"{label}: {' and '.join(left)} have no counterpart in the loaded {target} model and are "
                           f"left out (ComfyUI skips them).{trained}")
    return None


def install():
    """Hooks on comfy.sd: the LoRA check, and keeping each checkpoint's own declarations. Returns hooks installed."""
    # This runs the moment comfy.sd finishes executing, before Python binds it as an attribute of the comfy package,
    # so the modules are taken directly ("cannot access submodule 'sd'" otherwise).
    sd = importlib.import_module("comfy.sd")
    lora_mod = importlib.import_module("comfy.lora")
    convert = importlib.import_module("comfy.lora_convert")
    orig_apply = getattr(sd, "load_lora_for_models")

    def load_lora_for_models(model, clip, lora, strength_model, strength_clip, *a, **kw):
        if _active():
            said = None
            try:
                sides, model_map, clip_map = [], {}, {}
                if model is not None and strength_model != 0:
                    model_map = lora_mod.model_lora_keys_unet(model.model, {})
                    sides.append("model")
                if clip is not None and strength_clip != 0:
                    clip_map = lora_mod.model_lora_keys_clip(clip.cond_stage_model, {})
                    sides.append("text")
                if sides:
                    converted = convert.convert_lora(lora)
                    applied = len(lora_mod.load_lora(converted, {**model_map, **clip_map}, log_missing=False))
                    reach = lora_reach(converted.keys(), model_map.keys(), clip_map.keys())
                    meta = kw.get("lora_metadata", a[0] if a else None)
                    base = declared.from_header(converted.keys(), meta).get("Base")
                    target = _target(model) if model is not None else "text encoder"
                    said = lora_verdict(reach, applied, sides, _name.get(), base, target)
                    if said is None and os.environ.get("ENTAIL_VERBOSE"):
                        print(f"[entail] LoRA {_name.get() or ''}: all {reach['model'].total} model and "
                              f"{reach['text'].total} text-encoder modules reach the {target} model", flush=True)
            except Exception as e:  # noqa: BLE001 - a check that cannot run must never break the workflow
                print(f"[entail] could not check the LoRA ({type(e).__name__}: {e}); applying it unchecked", flush=True)
                said = None
            if said and said[0] == "violation":
                raise core.RoleError(said[1])
            if said:
                print(f"[entail] {said[1]}", flush=True)
        return orig_apply(model, clip, lora, strength_model, strength_clip, *a, **kw)

    n = _patch(sd, "load_lora_for_models", load_lora_for_models)
    for fname in ("load_state_dict_guess_config", "load_diffusion_model_state_dict"):
        if callable(getattr(sd, fname, None)):
            n += _patch(sd, fname, _keeping_declarations(getattr(sd, fname)))
    return n


def _keeping_declarations(orig):
    """Wrap a state-dict loader so the checkpoint's own declarations stay with the model it builds."""
    sig = inspect.signature(orig)

    @functools.wraps(orig)
    def load(sd, *a, **kw):
        decl = None
        if _active():
            try:
                meta = sig.bind_partial(sd, *a, **kw).arguments.get("metadata")
                decl = declared.from_header(sd.keys(), meta)
            except Exception:  # noqa: BLE001 - reading declarations must never break loading
                decl = None
        out = orig(sd, *a, **kw)
        patcher = out[0] if isinstance(out, tuple) else out
        base = getattr(patcher, "model", None)
        if decl and base is not None:
            core.tag(base, decl)
            declared.announce(decl, f"{WHERE} {_target(patcher)}")
        return out

    return load


def install_nodes():
    """Remember which file the built-in LoRA nodes are loading, for the message. 0 if this is not ComfyUI."""
    import sys

    loader = getattr(sys.modules.get("nodes"), "LoraLoader", None)
    if loader is None:
        return 0
    orig = loader.load_lora

    def load_lora(self, model, clip, lora_name, *a, **kw):
        token = _name.set(lora_name)
        try:
            return orig(self, model, clip, lora_name, *a, **kw)
        finally:
            _name.reset(token)

    return _patch(loader, "load_lora", load_lora)


# Prediction type ---------------------------------------------------------------------------------------------------

def sampling_kind(ms, ms_mod):
    """What ComfyUI set up, as a Prediction; None for schedules this check has not been measured on (EDM, X0,
    distilled/LCM, flow models)."""
    if not isinstance(ms, ms_mod.ModelSamplingDiscrete) or isinstance(ms, ms_mod.ModelSamplingDiscreteEDM):
        return None
    names = {c.__name__ for c in type(ms).__mro__}
    if isinstance(ms, (ms_mod.EDM, ms_mod.X0)) or "LCM" in names or "ModelSamplingDiscreteDistilled" in names:
        return None
    kind = "v" if isinstance(ms, ms_mod.V_PREDICTION) else "eps" if isinstance(ms, ms_mod.EPS) else None
    return Prediction(kind, getattr(ms, "zsnr", None)) if kind else None


def _switched(model, fact, ms_mod):
    """The model sampled as `fact`, the way a ModelSamplingDiscrete node does it (an object patch on a clone)."""
    class ModelSamplingAdvanced(ms_mod.ModelSamplingDiscrete, ms_mod.V_PREDICTION if fact.kind == "v" else ms_mod.EPS):
        pass

    patched = model.clone()
    patched.add_object_patch("model_sampling", ModelSamplingAdvanced(model.model.model_config, zsnr=fact.zsnr))
    return patched


class _Mismatch(Exception):
    def __init__(self, cos, seen):
        super().__init__(cos)
        self.cos, self.seen = cos, seen


def _sample_checked(orig, model, args, kwargs, custom, ms_mod):
    try:
        ms = model.get_model_object("model_sampling")
        used = sampling_kind(ms, ms_mod)
        applies = used is not None and type(model.model.diffusion_model).__name__ == "UNetModel"
    except Exception:  # noqa: BLE001
        applies = False
    if not applies:
        return orig(model, *args, **kwargs)
    base = model.model
    where = f"{WHERE} {type(base.model_config).__name__}"

    def resolve(fact):
        if custom:
            raise core.RoleError(f"{where}: the sigmas of this sampler were already computed for {used}, so entail "
                                 f"cannot switch it here. Add ModelSamplingDiscrete({'v_prediction' if fact.kind == 'v' else 'eps'}) right after the model loader.")
        return _switched(model, fact, ms_mod)

    check = functools.partial(contract.reconcile, Prediction, where=where, resolve=resolve,
                              explicit="model_sampling" in model.object_patches, what="prediction type")
    decl = core.facts_of(base).get("Declared")
    if decl is not None and decl.get(Prediction) is not None:
        return orig(check(decl, used)[1] or model, *args, **kwargs)
    if id(base) in _seen:  # behaviour measured on an earlier run of this model: no need to watch again
        return orig(check(None, used, evidence=(_seen[id(base)], "measured on an earlier run"))[1] or model,
                    *args, **kwargs)

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
                cos = behaviour.first_call_cos(used.kind, a["input"], a["timestep"], out, sigma_data) \
                    if t >= behaviour.MIN_T else None
            except Exception as e:  # noqa: BLE001 - never break sampling because the check could not run
                print(f"[entail] could not check the prediction type ({type(e).__name__}: {e})", flush=True)
                cos = None
            if cos is not None:
                seen = behaviour.behaves_like(cos)
                _seen[id(base)] = seen
                if not contract.agrees(seen, used, compare_zsnr=False):
                    raise _Mismatch(cos, seen)
                if os.environ.get("ENTAIL_VERBOSE"):
                    print(f"[entail] {where} model behaves like {seen} as set up (probe {cos:.2f})", flush=True)
        return out

    watched.set_model_unet_function_wrapper(wrapper)
    try:
        return orig(watched, *args, **kwargs)
    except _Mismatch as m:
        switched = check(None, used, evidence=(m.seen, f"first model call, cosine {m.cos:.2f}"))[1]
        return orig(switched or model, *args, **kwargs)


def install_sampling():
    """Wrap comfy.sample.sample and sample_custom. Returns hooks installed."""
    sample_mod = importlib.import_module("comfy.sample")
    ms_mod = importlib.import_module("comfy.model_sampling")
    n = 0
    for fname, custom in (("sample", False), ("sample_custom", True)):
        orig = getattr(sample_mod, fname)

        def run(model, *a, _orig=orig, _custom=custom, **kw):
            if _active():
                return _sample_checked(_orig, model, a, kw, _custom, ms_mod)
            return _orig(model, *a, **kw)

        n += _patch(sample_mod, fname, run)
    return n


# Schedules (ComfyUI-specific repair, Comfy-Org/ComfyUI#16490) ------------------------------------------------------

def install_schedule_record():
    """Every sampling object ComfyUI or a node makes keeps a copy of the schedule its own setter registered."""
    import torch

    ms_mod = importlib.import_module("comfy.model_sampling")
    if _SETTERS:
        return 0
    classes = [c for c in vars(ms_mod).values()
               if isinstance(c, type) and issubclass(c, torch.nn.Module) and c.__module__ == ms_mod.__name__]
    _SETTERS.extend(ownership.wrap_setters(classes, ("set_sigmas", "set_parameters")))
    return len(_SETTERS)


def install_buffer_guard():
    """ModelPatcherDynamic backs buffers up by path and writes them back at the next load. Returns 1, or 0 when that
    loader is not there (legacy loading, another ComfyUI) or the guard is already in place."""
    mp = importlib.import_module("comfy.model_patcher")
    resolve = getattr(importlib.import_module("comfy.utils"), "resolve_attr", None)
    cls = getattr(mp, "ModelPatcherDynamic", None)
    if cls is None or not callable(resolve) or not all(callable(getattr(cls, n, None))
                                                       for n in ("load", "restore_loaded_backups")):
        return 0
    orig_load, orig_restore = cls.load, cls.restore_loaded_backups
    where = f"{WHERE} sampling schedule"

    def load(self, *a, **kw):
        out = orig_load(self, *a, **kw)
        try:
            ownership.remember_owners(self.model, self.backup_buffers, resolve)
        except Exception:  # noqa: BLE001 - bookkeeping only
            pass
        return out

    def restore_loaded_backups(self):
        try:
            moved = ownership.return_foreign(self.model, self.backup_buffers, resolve)
            if moved:
                ownership.say_returned(moved, where)
        except Exception as e:  # noqa: BLE001 - never break loading because the guard could not run
            print(f"[entail] could not guard the buffer restore ({type(e).__name__}: {e})", flush=True)
        return orig_restore(self)

    return 1 if _patch(cls, "load", load) + _patch(cls, "restore_loaded_backups", restore_loaded_backups) else 0


def install_schedule_check():
    """At the first model call after a model's schedule buffers were (re)placed, check them against what the
    sampling object registered. The guard should leave nothing to find; this covers loaders it does not know."""
    import weakref

    cls = getattr(importlib.import_module("comfy.model_base"), "BaseModel", None)
    if cls is None or not callable(getattr(cls, "apply_model", None)):
        return 0
    orig = cls.apply_model

    def apply_model(self, *a, **kw):
        try:
            ms = self._modules.get("model_sampling")
            sig = ms._buffers.get("sigmas") if ms is not None else None
            last = self.__dict__.get("_entail_checked")
            if sig is not None and (last is None or last[0]() is not ms or last[1]() is not sig):
                found = ownership.drift(ms)
                if found:
                    ownership.put_back(ms, found, f"{WHERE} sampling schedule")
                self.__dict__["_entail_checked"] = (weakref.ref(ms), weakref.ref(ms._buffers["sigmas"]))
        except Exception:  # noqa: BLE001 - a check that cannot run must never break sampling
            pass
        return orig(self, *a, **kw)

    return _patch(cls, "apply_model", apply_model)


def uninstall():
    n = 0
    for cls, name, fn in _SETTERS:
        setattr(cls, name, fn)
        n += 1
    _SETTERS.clear()
    for (owner, attr), orig in list(_ORIG.items()):
        setattr(owner, attr, orig)
        n += 1
    _ORIG.clear()
    return n
