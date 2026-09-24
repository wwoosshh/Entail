"""Adapter v2 for diffusers (LIBRARY_DESIGN.md 4.8; ROADMAP M6.2). Where diffusers decides, what it decided, and how to
make it use something else; the rules are the core's (load.prediction, load.latent_scale, load.lora).

  hooks        FromSingleFileMixin.from_single_file: a pipeline built from one checkpoint file. diffusers configures its
               scheduler without reading the file's declarations - prediction_type falls back to epsilon
               (loaders/single_file_utils.py) - so a file that declares v is sampled as eps (fd-m7, market I04).
               DiffusionPipeline.from_pretrained, for a local diffusers folder: its scheduler_config and vae/config.json
               are the declaration.
               DiffusionPipeline.__setattr__ for "vae" on a pipeline already built: a VAE put in later. A VAE file read
               on its own is taken for Stable Diffusion 1.5's, because the two VAEs have the same keys (fd-vae).
               load_lora_weights of every LoRA loader mixin in loaders/lora_pipeline.py.
  read_choice  the scheduler's prediction_type and rescale_betas_zero_snr; the VAE config's scaling_factor and
               shift_factor; a LoRA's modules (in diffusers' own conversion of its state dict) and the modules that
               hold the adapter afterwards; whether the caller passed the scheduler or prediction_type (explicit).
  handles      switch_prediction: the scheduler rebuilt from its config with the declared prediction_type (and
               rescale_betas_zero_snr where the scheduler has it), what diffusers' docs tell users to do by hand;
               set_latent_scale: the VAE's config given the declared scaling_factor (and shift_factor).
A pipeline from the Hub is not read (only local files and folders are): reported as not checked.
"""
import importlib
import os

from .. import core, load, policies, readers
from ..facts import LatentScale, Prediction
from .base import Hook

engine = "diffusers"
versions = "0.40.0"
_ORIG = {}


def hooks():
    return [Hook("diffusers.loaders.single_file.FromSingleFileMixin.from_single_file", "load"),
            Hook("diffusers.pipelines.pipeline_utils.DiffusionPipeline.from_pretrained", "load"),
            Hook("diffusers.pipelines.pipeline_utils.DiffusionPipeline.__setattr__", "load"),
            Hook("diffusers.loaders.lora_pipeline.*LoraLoaderMixin.load_lora_weights", "load")]


def _active():
    return core.mode() in ("load", "debug")


def read_choice(kind, *args):
    """What diffusers chose, as fact values.
      ("prediction", scheduler)        -> Prediction, or None when its config does not name a prediction type
      ("latent_scale", vae)            -> LatentScale, or None when its config has no scaling_factor
      ("lora_given", lora state dict)  -> the modules a LoRA carries weights for, as 'component.module'
      ("lora_taken", pipeline, names)  -> the modules that hold one of the adapters `names`, as 'component.module'"""
    if kind == "prediction":
        (scheduler,) = args
        cfg = getattr(scheduler, "config", None) or {}
        kind_ = readers.prediction_kind(cfg.get("prediction_type", ""))
        zsnr = cfg.get("rescale_betas_zero_snr")
        return Prediction(kind_, zsnr if isinstance(zsnr, bool) else None) if kind_ else None
    if kind == "latent_scale":
        (vae,) = args
        cfg = getattr(vae, "config", None) or {}
        scale, shift = cfg.get("scaling_factor"), cfg.get("shift_factor")
        return LatentScale(float(scale), shift=None if shift is None else float(shift)) if scale else None
    if kind == "lora_given":
        (state_dict,) = args
        mods = set()
        for mod in readers.lora_modules(state_dict):
            for suffix in (".lora", ".lora_linear_layer"):
                if mod.endswith(suffix):
                    mod = mod[: -len(suffix)]
            mods.add(mod)
        return mods
    if kind == "lora_taken":
        pipe, names = args
        out = set()
        for comp in ("unet", "transformer", "text_encoder", "text_encoder_2", "text_encoder_3"):
            model = getattr(pipe, comp, None)
            if model is None or not hasattr(model, "named_modules"):
                continue
            for mod_name, mod in model.named_modules():
                lora_a = getattr(mod, "lora_A", None)
                if lora_a is not None and hasattr(lora_a, "keys") and set(names) & set(lora_a.keys()):
                    out.add(f"{comp}.{mod_name}")
        return out
    raise ValueError(f"diffusers_adapter.read_choice: unknown kind {kind!r}")


def handles(pipe):
    names = {"v": "v_prediction", "eps": "epsilon", "x0": "sample"}

    def switch_prediction(target):
        over = {"prediction_type": names[target.kind]}
        if target.zsnr is not None and "rescale_betas_zero_snr" in pipe.scheduler.config:
            over["rescale_betas_zero_snr"] = target.zsnr
        pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config, **over)
        return pipe

    def set_latent_scale(target):
        over = {"scaling_factor": target.scale}
        if target.shift is not None:
            over["shift_factor"] = target.shift
        pipe.vae.register_to_config(**over)
        return pipe

    return {"switch_prediction": switch_prediction, "set_latent_scale": set_latent_scale}


# --- the contracts at each hook -------------------------------------------------------------------------------------

def _check_pipeline(pipe, path, kwargs, where):
    """After a pipeline was built from a local file or folder: its prediction and its VAE's scale against what the
    file or folder declares; the declarations stay with the pipeline for a VAE put in later."""
    facts = load.declared(path)
    load.remember(pipe, facts)
    policy = policies.current()
    decisions = []
    sched = getattr(pipe, "scheduler", None)
    if sched is not None:
        decisions += load.prediction(engine, "diffusers.scheduler", facts, read_choice("prediction", sched),
                                     explicit="scheduler" in kwargs or "prediction_type" in kwargs,
                                     can_switch=("eps", "v", "x0"), policy=policy, where=where)
    if getattr(pipe, "vae", None) is not None:
        decisions += _vae_decisions(pipe, facts, policy, where)
    load.enforce(decisions)
    load.resolve(decisions, handles(pipe))


def _vae_decisions(pipe, facts, policy, where):
    return load.latent_scale(engine, "diffusers.vae", facts, read_choice("latent_scale", pipe.vae), policy=policy,
                             where=where)


def _classmethod(cls, name, make):
    raw = cls.__dict__[name].__func__
    _ORIG[(cls, name)] = cls.__dict__[name]
    setattr(cls, name, classmethod(make(raw)))
    return 1


def install():
    """Wrap FromSingleFileMixin.from_single_file (every pipeline class inherits it). Returns hooks installed."""
    cls = importlib.import_module("diffusers.loaders.single_file").FromSingleFileMixin
    if (cls, "from_single_file") in _ORIG:
        return 0

    def make(raw):
        def from_single_file(klass, pretrained_model_link_or_path, **kwargs):
            pipe = raw(klass, pretrained_model_link_or_path, **kwargs)
            if _active():
                path = pretrained_model_link_or_path
                if isinstance(path, str) and os.path.isfile(path):
                    load.safely("load:diffusers.single_file", "diffusers.scheduler", "Prediction",
                                lambda: _check_pipeline(pipe, path, kwargs, f"{klass.__name__}.from_single_file"))
                else:
                    load.enforce([load.cannot_check("load:diffusers.single_file", "diffusers.scheduler", "Prediction",
                                                    f"{path!r} is not a local file; entail reads only local files")])
            return pipe

        return from_single_file

    return _classmethod(cls, "from_single_file", make)


def install_pipeline():
    """Wrap DiffusionPipeline.from_pretrained (a local folder) and __setattr__ ("vae" on a built pipeline)."""
    cls = importlib.import_module("diffusers.pipelines.pipeline_utils").DiffusionPipeline
    if (cls, "__setattr__") in _ORIG:
        return 0

    def make(raw):
        def from_pretrained(klass, pretrained_model_name_or_path, **kwargs):
            pipe = raw(klass, pretrained_model_name_or_path, **kwargs)
            if _active():
                path = pretrained_model_name_or_path
                if isinstance(path, (str, os.PathLike)) and os.path.isdir(path):
                    load.safely("load:diffusers.pipeline", "diffusers.scheduler", "Prediction",
                                lambda: _check_pipeline(pipe, os.fspath(path), kwargs,
                                                        f"{klass.__name__}.from_pretrained"))
                else:
                    load.enforce([load.cannot_check("load:diffusers.pipeline", "diffusers.scheduler", "Prediction",
                                                    f"{path!r} is not a local folder; entail reads only local "
                                                    f"folders")])
            return pipe

        return from_pretrained

    n = _classmethod(cls, "from_pretrained", make)
    orig_setattr = cls.__setattr__

    def __setattr__(self, name, value):
        orig_setattr(self, name, value)
        if name == "vae" and value is not None and _active():
            facts = load.remembered(self)
            if facts is not None:   # a pipeline entail saw being built: its declarations are known
                def work():
                    decisions = _vae_decisions(self, facts, policies.current(), f"{type(self).__name__}.vae")
                    load.enforce(decisions)
                    load.resolve(decisions, handles(self))

                load.safely("load:diffusers.latent_scale", "diffusers.vae", "LatentScale", work)

    _ORIG[(cls, "__setattr__")] = orig_setattr
    cls.__setattr__ = __setattr__
    return n + 1


def install_lora():
    """Wrap load_lora_weights of every LoRA loader mixin that defines it. Returns hooks installed."""
    mod = importlib.import_module("diffusers.loaders.lora_pipeline")
    n = 0
    for cls in vars(mod).values():
        orig = cls.__dict__.get("load_lora_weights") if isinstance(cls, type) else None
        if orig is None or (cls, "load_lora_weights") in _ORIG:
            continue

        def load_lora_weights(self, pretrained_model_name_or_path_or_dict, adapter_name=None, *a, _orig=orig, **kw):
            if not _active():
                return _orig(self, pretrained_model_name_or_path_or_dict, adapter_name, *a, **kw)
            given = load.safely("load:diffusers.lora", "diffusers.lora_loader", "Coverage",
                                lambda: _lora_given(self, pretrained_model_name_or_path_or_dict, kw))
            before = _adapters(self)
            out = _orig(self, pretrained_model_name_or_path_or_dict, adapter_name, *a, **kw)
            if given:
                source = pretrained_model_name_or_path_or_dict
                where = f"LoRA {source}" if isinstance(source, (str, os.PathLike)) else "a LoRA state dict"

                def work():
                    names = {adapter_name} if adapter_name else _adapters(self) - before
                    load.enforce(load.lora(engine, f"{where} on {type(self).__name__}", given,
                                           read_choice("lora_taken", self, names), policy=policies.current()))

                load.safely("load:diffusers.lora", "diffusers.lora_loader", "Coverage", work)
            return out

        _ORIG[(cls, "load_lora_weights")] = orig
        cls.load_lora_weights = load_lora_weights
        n += 1
    return n


def _lora_given(pipe, source, kw):
    """As load_lora_weights reads it: lora_state_dict edits a dict it is given, and does not take low_cpu_mem_usage."""
    source = source.copy() if isinstance(source, dict) else source
    sd = pipe.lora_state_dict(source, **{k: v for k, v in kw.items() if k not in ("low_cpu_mem_usage", "hotswap")})
    return read_choice("lora_given", sd[0] if isinstance(sd, tuple) else sd)


def _adapters(pipe):
    """The adapter names the pipeline lists, or none. Without the PEFT backend diffusers raises here, before the
    loader would raise the same thing itself: that is the loader's to say, not entail's (principle 12; found in M6.3)."""
    get = getattr(pipe, "get_list_adapters", None)
    try:
        listed = get() if callable(get) else None
    except Exception:  # noqa: BLE001 - reading what the engine holds must never break its call
        return set()
    return set().union(*listed.values()) if listed else set()


def uninstall():
    n = 0
    for (owner, attr), orig in list(_ORIG.items()):
        setattr(owner, attr, orig)
        n += 1
    _ORIG.clear()
    return n
