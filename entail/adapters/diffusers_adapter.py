"""Adapter v2 for diffusers (LIBRARY_DESIGN.md 4.8; ROADMAP M6.2). Where diffusers decides, what it decided, and how to
make it use something else; the rules are the core's (load.prediction, load.latent_scale, load.lora).

  hooks        FromSingleFileMixin.from_single_file: a pipeline built from one checkpoint file. diffusers configures its
               scheduler without reading the file's declarations - prediction_type falls back to epsilon
               (loaders/single_file_utils.py) - so a file that declares v is sampled as eps (fd-m7, market I04).
               DiffusionPipeline.from_pretrained, for a local diffusers folder: its scheduler_config and vae/config.json
               are the declaration.
               The prediction is decided where the model samples, not at the load (field test, entail#21, #24): the
               pipeline class's __call__ (the scheduler it is about to sample with; rebuilt there when it differs),
               and a one-time forward pre-hook on its denoiser for a program that runs the model in a sampling loop of
               its own (InvokeAI): said as unknown there, nothing changed.
               DiffusionPipeline.__setattr__ for "vae" on a pipeline already built: a VAE put in later. A VAE file read
               on its own is taken for Stable Diffusion 1.5's, because the two VAEs have the same keys (fd-vae).
               load_lora_weights of every LoRA loader mixin in loaders/lora_pipeline.py, and peft's
               set_peft_model_state_dict, the loader's last step, which it calls for each model the LoRA reaches.
  read_choice  the scheduler's prediction_type and rescale_betas_zero_snr; the VAE config's scaling_factor and
               shift_factor; for a LoRA, what the loader handed peft at its last step - under the names it
               converted the LoRA to - and the keys peft found no place for in the model (its unexpected keys), and,
               when nothing got that far, the LoRA's modules as lora_state_dict reads them; whether the caller passed
               the scheduler or prediction_type (explicit).
  handles      switch_prediction: the scheduler rebuilt from its config with the declared prediction_type (and
               rescale_betas_zero_snr where the scheduler has it), what diffusers' docs tell users to do by hand;
               set_latent_scale: the VAE's config given the declared scaling_factor (and shift_factor).
A pipeline from the Hub is not read (only local files and folders are): reported as not checked.
"""
import contextvars
import functools
import importlib
import os
import weakref

from .. import core, load, policies, readers
from ..facts import LatentScale, Prediction
from .base import Hook

engine = "diffusers"
versions = "0.40.0"
_ORIG = {}
_HANDED = contextvars.ContextVar("entail_diffusers_lora", default=None)   # this load: [(model, keys, unexpected)]
_STATE = weakref.WeakKeyDictionary()   # pipeline -> its prediction check, waiting for where the pipeline samples
_IN_CALL = contextvars.ContextVar("entail_diffusers_call", default=False)   # inside a pipeline's own __call__
_INHERITED = object()   # a __call__ the pipeline class inherited: uninstall removes the wrapper instead of restoring


def hooks():
    return [Hook("diffusers.loaders.single_file.FromSingleFileMixin.from_single_file", "load"),
            Hook("diffusers.pipelines.pipeline_utils.DiffusionPipeline.from_pretrained", "load"),
            Hook("diffusers.pipelines.pipeline_utils.DiffusionPipeline.__setattr__", "load"),
            Hook("diffusers.loaders.lora_pipeline.*LoraLoaderMixin.load_lora_weights", "load"),
            Hook("peft.set_peft_model_state_dict", "load")]


def _active():
    return core.mode() in ("load", "debug")


def read_choice(kind, *args):
    """What diffusers chose, as fact values.
      ("prediction", scheduler)        -> Prediction, or None when its config does not name a prediction type
      ("latent_scale", vae)            -> LatentScale, or None when its config has no scaling_factor
      ("lora_given", lora state dict)  -> the modules a LoRA carries weights for, as 'component.module'
      ("lora_handed", pipeline, [(model, keys, unexpected keys)])
                                       -> (given, taken): the modules the loader handed peft, and those of them the
                                          model had a place for, as 'component.module'
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
    if kind == "lora_handed":
        pipe, handed = args
        comps = {id(getattr(pipe, c, None)): c for c in ("unet", "transformer", "text_encoder", "text_encoder_2",
                                                          "text_encoder_3")}
        given, taken = set(), set()
        for model, keys, unexpected in handed:
            comp = comps.get(id(model), type(model).__name__)
            mods, left = readers.lora_modules(keys), readers.lora_modules(unexpected)
            given |= {f"{comp}.{m}" for m in mods}
            taken |= {f"{comp}.{m}" for m in mods - left}
        return given, taken
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
    """After a pipeline was built from a local file or folder: its VAE's scale against what the file or folder
    declares, now; its prediction where it samples. The declarations stay with the pipeline, for a VAE put in later
    and for the sampling."""
    facts = load.declared(path)
    load.remember(pipe, facts)
    if getattr(pipe, "vae", None) is not None:
        decisions = _vae_decisions(pipe, facts, policies.current(), where)
        load.enforce(decisions)
        load.resolve(decisions, handles(pipe))
    if getattr(pipe, "scheduler", None) is not None:
        explicit = "scheduler" in kwargs or "prediction_type" in kwargs
        _await_sampling(pipe, where, pipe.scheduler if explicit else None)


def _await_sampling(pipe, where, explicit):
    """Decide the prediction where the pipeline samples, not at the load. A program may set up a sampler of its own
    after the load: SD.Next already samples a v checkpoint as v, and a scheduler entail rebuilt at the load changed
    its images (field test, entail#21); InvokeAI builds its sampler from its own model settings, so a scheduler
    rebuilt at the load never reached the sampling but left its zero-SNR schedule under Invoke's epsilon, and the
    default sampler stopped (entail#24). `explicit`: the scheduler the caller passed at the load (never replaced)."""
    state = {"where": where, "explicit": explicit, "hook": None}
    _STATE[pipe] = state
    _wrap_call(type(pipe))
    denoiser = next((m for m in (getattr(pipe, "unet", None), getattr(pipe, "transformer", None))
                     if hasattr(m, "register_forward_pre_hook")), None)
    if denoiser is None:
        return
    ref = weakref.ref(pipe)

    def own_loop(module, args):
        """The denoiser's first forward outside the pipeline's call: a sampling loop of the program's own."""
        owner = ref()
        if owner is None or _IN_CALL.get():
            return None
        _unhook(_STATE.get(owner))
        if _active():
            load.safely("load:diffusers.prediction", "diffusers.sampler", "Prediction",
                        lambda: _decide_own_loop(owner))
        return None

    state["hook"] = denoiser.register_forward_pre_hook(own_loop)


def _unhook(state):
    handle = state.pop("hook", None) if state else None
    if handle is not None:
        handle.remove()


def _set_up(sched):
    cfg = getattr(sched, "config", None) or {}
    return (f"the scheduler: {type(sched).__name__}, prediction_type {cfg.get('prediction_type')!r}, "
            f"rescale_betas_zero_snr {cfg.get('rescale_betas_zero_snr')!r}")


def _decide_call(pipe, state):
    """At a pipeline call: the scheduler it is about to sample with, against the declaration. One set up for another
    prediction is rebuilt for the declared one before the call; the one the caller passed at the load is reported,
    not replaced."""
    _unhook(state)
    sched = getattr(pipe, "scheduler", None)
    if sched is None:
        return
    decisions = load.prediction(engine, "diffusers.scheduler", load.remembered(pipe) or load.Declared(),
                                read_choice("prediction", sched), explicit=sched is state["explicit"],
                                can_switch=("eps", "v", "x0"), policy=policies.current(), where=state["where"],
                                note=_set_up(sched))
    load.enforce(decisions, once_for=sched)
    load.resolve(decisions, handles(pipe))


def _decide_own_loop(pipe):
    """The model ran in the program's own sampling loop, with a scheduler entail does not see: the declaration is
    said, and nothing is changed."""
    state = _STATE.get(pipe) or {}
    note = ("the model ran outside the call of the pipeline it was loaded with - in the program's own sampling loop, "
            "or another pipeline made from its parts - whose scheduler entail does not see; nothing was changed. If "
            "the images come out as noise or washed out, set the program's prediction type for this model to the "
            "declared one (InvokeAI: the model's settings)")
    load.enforce(load.prediction(engine, "diffusers.sampler", load.remembered(pipe) or load.Declared(), None,
                                 can_switch=(), policy=policies.current(), where=state.get("where", ""), note=note))


def _wrap_call(cls):
    """The pipeline class's __call__, wrapped once: each pipeline class defines its own sampling. A call inside a
    call (a pipeline that runs another) is decided by the outer one."""
    if (cls, "__call__") in _ORIG or not callable(getattr(cls, "__call__", None)):
        return
    orig = cls.__call__
    _ORIG[(cls, "__call__")] = cls.__dict__.get("__call__", _INHERITED)

    @functools.wraps(orig)
    def __call__(self, *a, **kw):
        state = _STATE.get(self) if _active() and not _IN_CALL.get() else None
        if state is None:
            return orig(self, *a, **kw)
        load.safely("load:diffusers.prediction", "diffusers.scheduler", "Prediction", lambda: _decide_call(self, state))
        token = _IN_CALL.set(True)
        try:
            return orig(self, *a, **kw)
        finally:
            _IN_CALL.reset(token)

    cls.__call__ = __call__


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


def local_snapshot(name, kwargs):
    """The local folder a hub id was downloaded to - the huggingface_hub cache diffusers has just filled - or None
    when it is not there (M11.6; 1.0 checked nothing for a pipeline named by its hub id). Only the local cache is
    asked: nothing is downloaded here."""
    if not isinstance(name, str) or os.path.exists(name):
        return None
    try:
        from huggingface_hub import snapshot_download

        folder = snapshot_download(repo_id=name, revision=kwargs.get("revision"), cache_dir=kwargs.get("cache_dir"),
                                   local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached, or no huggingface_hub: reported as not checked by the caller
        return None
    return folder if isinstance(folder, str) and os.path.isdir(folder) else None


def install_pipeline():
    """Wrap DiffusionPipeline.from_pretrained (a local folder, or a hub id found in the local cache) and __setattr__
    ("vae" on a built pipeline)."""
    cls = importlib.import_module("diffusers.pipelines.pipeline_utils").DiffusionPipeline
    if (cls, "__setattr__") in _ORIG:
        return 0

    def make(raw):
        def from_pretrained(klass, pretrained_model_name_or_path, **kwargs):
            pipe = raw(klass, pretrained_model_name_or_path, **kwargs)
            if _active():
                path = pretrained_model_name_or_path
                folder = os.fspath(path) if isinstance(path, (str, os.PathLike)) and os.path.isdir(path) else \
                    local_snapshot(path, kwargs)
                if folder is not None:
                    load.safely("load:diffusers.pipeline", "diffusers.scheduler", "Prediction",
                                lambda: _check_pipeline(pipe, folder, kwargs, f"{klass.__name__}.from_pretrained"))
                else:
                    load.enforce([load.cannot_check("load:diffusers.pipeline", "diffusers.scheduler", "Prediction",
                                                    f"{path!r} is not a local folder and is not in the local "
                                                    f"huggingface_hub cache; entail reads only local folders")])
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
            source = pretrained_model_name_or_path_or_dict
            before = _adapters(self)
            token = _HANDED.set([])
            try:
                out = _orig(self, source, adapter_name, *a, **kw)
            finally:
                handed = _HANDED.get()
                _HANDED.reset(token)
            where = f"LoRA {source}" if isinstance(source, (str, os.PathLike)) else "a LoRA state dict"

            def work():
                if not handed and read_choice("lora_taken", self,
                                              {adapter_name} if adapter_name else _adapters(self) - before):
                    load.enforce([load.cannot_check("load:diffusers.lora", "diffusers.lora_loader", "Coverage",
                                                    f"{where}: loaded without peft's set_peft_model_state_dict "
                                                    f"(hotswap?), so what reached the model was not read")])
                    return
                given, taken = read_choice("lora_handed", self, handed)
                carried = () if handed else _lora_given(self, source, kw)   # nothing got that far: what it has
                load.enforce(load.lora(engine, f"{where} on {type(self).__name__}", given, taken,
                                       policy=policies.current(), carried=carried))

            load.safely("load:diffusers.lora", "diffusers.lora_loader", "Coverage", work)
            return out

        _ORIG[(cls, "load_lora_weights")] = orig
        cls.load_lora_weights = load_lora_weights
        n += 1
    return n + _install_peft()


def _install_peft():
    """peft.set_peft_model_state_dict, which diffusers imports from peft each time it loads a LoRA into a model."""
    try:
        peft = importlib.import_module("peft")
    except ImportError:   # without peft diffusers loads no LoRA at all, and says so itself
        return 0
    if (peft, "set_peft_model_state_dict") in _ORIG:
        return 0
    orig = peft.set_peft_model_state_dict

    def set_peft_model_state_dict(model, peft_model_state_dict, *a, **kw):
        out = orig(model, peft_model_state_dict, *a, **kw)
        handed = _HANDED.get()
        if handed is not None:
            try:
                handed.append((model, list(peft_model_state_dict), list(getattr(out, "unexpected_keys", None) or ())))
            except Exception:  # noqa: BLE001 - reading what the loader did must never break it
                pass
        return out

    _ORIG[(peft, "set_peft_model_state_dict")] = orig
    peft.set_peft_model_state_dict = set_peft_model_state_dict
    return 1


def _lora_given(pipe, source, kw):
    """The modules a LoRA carries, as lora_state_dict reads it, for a LoRA none of which reached the loader's last
    step: it edits a dict it is given, and does not take low_cpu_mem_usage. (Its names are not always the loader's
    final ones - a kohya LoRA still has attention-processor names there, and peft's target list holds name endings,
    not modules - which is why a LoRA the loader took is read at set_peft_model_state_dict; found in M6.3.)"""
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
        if orig is _INHERITED:
            delattr(owner, attr)
        else:
            setattr(owner, attr, orig)
        n += 1
    _ORIG.clear()
    for state in list(_STATE.values()):
        _unhook(state)
    _STATE.clear()
    return n
