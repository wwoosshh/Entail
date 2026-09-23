"""Adapter: diffusers. Where to hook and how to switch; the checks are entail's own, the same ones ComfyUI gets.

  single-file pipelines  FromSingleFileMixin.from_single_file builds the scheduler from a default config and reads none
                         of the checkpoint's own statements (diffusers 0.40, loaders/single_file_utils.py: prediction_type
                         falls back to epsilon). The file's declarations (declared.py) are compared with the scheduler's
                         prediction_type (contract.py). Switching is a scheduler rebuilt with the declared prediction
                         type, which is what diffusers tells users to do by hand. A scheduler or prediction_type passed
                         by the caller is explicit and is not overridden.
  LoRA                   load_lora_weights: the LoRA's modules against the modules that received the adapter
                         (coverage.py). None received: stop. Some left out: one line.
"""
import importlib
import os

from .. import contract, core, coverage, declared
from ..facts import Prediction

_ORIG = {}
WHERE = "diffusers"
_NAMES = {"v": "v_prediction", "eps": "epsilon", "x0": "sample"}


def _active():
    return core.mode() in ("load", "debug")


def used_prediction(scheduler):
    """What a diffusers scheduler will do, as a Prediction (None when its config does not say)."""
    cfg = getattr(scheduler, "config", None) or {}
    kind = declared.prediction_kind(cfg.get("prediction_type", ""))
    zsnr = cfg.get("rescale_betas_zero_snr")
    return Prediction(kind, zsnr if isinstance(zsnr, bool) else None) if kind else None


def _switch(pipe, fact):
    over = {"prediction_type": _NAMES[fact.kind]}
    if fact.zsnr is not None and "rescale_betas_zero_snr" in pipe.scheduler.config:
        over["rescale_betas_zero_snr"] = fact.zsnr
    pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config, **over)
    return pipe


def install():
    """Wrap FromSingleFileMixin.from_single_file (every pipeline class inherits it). Returns hooks installed."""
    mod = importlib.import_module("diffusers.loaders.single_file")
    cls = mod.FromSingleFileMixin
    if (cls, "from_single_file") in _ORIG:
        return 0
    raw = cls.__dict__["from_single_file"].__func__

    def from_single_file(klass, pretrained_model_link_or_path, **kwargs):
        pipe = raw(klass, pretrained_model_link_or_path, **kwargs)
        if _active() and isinstance(pretrained_model_link_or_path, str) and os.path.isfile(pretrained_model_link_or_path):
            sched = getattr(pipe, "scheduler", None)
            used = used_prediction(sched) if sched is not None else None
            decl = declared.from_file(pretrained_model_link_or_path)
            if decl and used is not None:
                where = f"{WHERE} {klass.__name__}"
                declared.announce(decl, where)
                contract.reconcile(Prediction, decl, used, where=where, resolve=lambda f: _switch(pipe, f),
                                   explicit="scheduler" in kwargs or "prediction_type" in kwargs, what="prediction type")
        return pipe

    _ORIG[(cls, "from_single_file")] = cls.__dict__["from_single_file"]
    cls.from_single_file = classmethod(from_single_file)
    return 1


def _adapted_modules(pipe, name):
    """'component.module' for every module of the pipeline's components that holds adapter `name`."""
    out = set()
    for comp in ("unet", "transformer", "text_encoder", "text_encoder_2", "text_encoder_3"):
        model = getattr(pipe, comp, None)
        if model is None or not hasattr(model, "named_modules"):
            continue
        for mod_name, mod in model.named_modules():
            lora_a = getattr(mod, "lora_A", None)
            if lora_a is not None and hasattr(lora_a, "keys") and name in lora_a.keys():
                out.add(f"{comp}.{mod_name}")
    return out


def _given_modules(state_dict):
    """'component.module' for every module a (diffusers-format) LoRA state dict carries weights for."""
    mods = set()
    for key in state_dict:
        mod = declared.lora_module(key)
        for suffix in (".lora", ".lora_linear_layer"):
            if mod.endswith(suffix):
                mod = mod[: -len(suffix)]
        mods.add(mod)
    return mods


def install_lora():
    """Wrap load_lora_weights of the LoRA loader mixins. Returns hooks installed."""
    mod = importlib.import_module("diffusers.loaders.lora_pipeline")
    n = 0
    for cls_name in ("StableDiffusionXLLoraLoaderMixin", "StableDiffusionLoraLoaderMixin"):
        cls = getattr(mod, cls_name, None)
        if cls is None or (cls, "load_lora_weights") in _ORIG:
            continue
        orig = cls.__dict__.get("load_lora_weights")
        if orig is None:
            continue

        def load_lora_weights(self, pretrained_model_name_or_path_or_dict, adapter_name=None, *a, _orig=orig, **kw):
            if not _active():
                return _orig(self, pretrained_model_name_or_path_or_dict, adapter_name, *a, **kw)
            try:
                given = _given_modules(self.lora_state_dict(pretrained_model_name_or_path_or_dict, **kw)[0])
            except Exception:  # noqa: BLE001 - the count is ours; loading is diffusers' and must go on
                given = None
            before = set(self.get_list_adapters().get("unet", [])) if hasattr(self, "get_list_adapters") else set()
            out = _orig(self, pretrained_model_name_or_path_or_dict, adapter_name, *a, **kw)
            if given:
                names = adapter_name and {adapter_name} or \
                    set(self.get_list_adapters().get("unet", [])) - before if hasattr(self, "get_list_adapters") else set()
                taken = set().union(*(_adapted_modules(self, nm) for nm in names)) if names else set()
                what = f"LoRA {pretrained_model_name_or_path_or_dict if isinstance(pretrained_model_name_or_path_or_dict, str) else ''} modules".replace("  ", " ")
                coverage.check(coverage.count(given, taken), f"{WHERE} {type(self).__name__}", what)
            return out

        _ORIG[(cls, "load_lora_weights")] = orig
        cls.load_lora_weights = load_lora_weights
        n += 1
    return n


def uninstall():
    n = 0
    for (owner, attr), orig in list(_ORIG.items()):
        setattr(owner, attr, orig)
        n += 1
    _ORIG.clear()
    return n
