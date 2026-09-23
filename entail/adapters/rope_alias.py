"""Adapter: a RoPE value given under its transformers-4 name after the config is built still reaches the rotary
embedding.

transformers 5 keeps the RoPE settings in `rope_parameters`. A checkpoint that spells them the old way
(`rope_theta`, `rope_scaling` in config.json) is converted while the config is built, but a value given after
that goes through code that does not convert it. Measured (audits/ALIAS.md, audits/override_probe.py):

  AutoConfig.from_pretrained(p, rope_theta=x)          ignored, silently (it comes back as an unused kwarg)
  cfg.rope_theta = x                                   ignored, silently (nothing reads the attribute)
  vLLM hf_overrides / SGLang json_model_override_args  {"rope_theta": x} ignored, silently
  cfg.rope_scaling = {...}, and the same key through either engine's override
      replaces rope_parameters and drops rope_theta with it. transformers then fails to build the model; the
      engines build it with whatever their model file falls back to. Measured end to end on vLLM: Llama 3.2 falls
      to 10,000 (GSM8K 379 -> 279 of 500), Qwen3 happens to fall to its own 1,000,000 (issue_track/rope_override).

Both engines apply their overrides with setattr on the transformers config, so one hook here covers all three.

Resolution (policy `resolve`): convert the value the way transformers converts the same key when it reads it
from config.json - the meaning a model card relies on when it tells users to pass the key at launch - and say
so in one line. Where the key lands is asked of the config class itself (_landing), so a per-layer-type RoPE
(Gemma 3, where rope_theta is the full-attention base) is written where the class puts it, not where we guess.
Under `refuse`, raise instead.

Not touched: classes that declare the old name as a field (esm, evolla, cohere2_moe, deepseek_v4, sapiens2 ...)
or map it through attribute_map - their models read the attribute itself - and configs whose rope_parameters
has not been built yet (the write happens during construction, where the conversion still runs).
"""
import copy
import dataclasses
import functools

from .. import core
from . import _shared

LEGACY = ("rope_theta", "rope_scaling")
_MARK_THETA = 12345.678  # values no real checkpoint uses, to see where the class puts each key
_MARK_FACTOR = 7.25
_MISSING = object()
_ORIG_SETATTR = None
_ORIG_FROM_DICT = None
_probing = False
_active = False  # classes created while installed keep our wrapper in their @strict closure; this turns it off


def _declares(cls, key):
    try:
        return key in {f.name for f in dataclasses.fields(cls)}
    except TypeError:
        return False


def _built(rp):
    """rope_parameters after conversion: rope_theta sits at the top, or in every per-layer-type entry."""
    if not isinstance(rp, dict) or not rp:
        return False
    if "rope_theta" in rp:
        return True
    entries = [v for v in rp.values() if isinstance(v, dict)]
    return bool(entries) and all("rope_theta" in v for v in entries)


def applies(cfg, key):
    """Would a write of `key` on this config be lost? (The class reads rope_parameters, which is already built.)"""
    cls = type(cfg)
    if key not in LEGACY or key in (getattr(cls, "attribute_map", None) or {}) or _declares(cls, key):
        return False
    return _built(cfg.__dict__.get("rope_parameters"))


@functools.lru_cache(maxsize=None)
def _landing(cls, key):
    """Which entries of a per-layer-type rope_parameters take `key` when it comes from config.json.

    Asked of the class: build one with its defaults and the key set to a marker, and see where the marker is.
    Returns a tuple of layer types, or None when the class cannot be built that way."""
    global _probing
    marker = _MARK_THETA if key == "rope_theta" else {"rope_type": "linear", "factor": _MARK_FACTOR}
    _probing = True
    try:
        inst = cls(**{key: copy.deepcopy(marker)})
    except Exception:  # noqa: BLE001 - a class that cannot be built with defaults gives no answer
        return None
    finally:
        _probing = False
    rp = inst.__dict__.get("rope_parameters")
    if not isinstance(rp, dict):
        return None
    hit = (lambda d: d.get("rope_theta") == _MARK_THETA) if key == "rope_theta" else \
        (lambda d: d.get("factor") == _MARK_FACTOR)
    found = tuple(sorted(k for k, v in rp.items() if isinstance(v, dict) and hit(v)))
    return found or None


def _standardized(cfg, rp):
    """Run the class's own standardisation on a copy of the config carrying `rp`, and return the result."""
    global _probing
    _probing = True
    try:
        tmp = copy.copy(cfg)
        tmp.__dict__["rope_parameters"] = rp
        tmp.standardize_rope_params()
        return tmp.__dict__["rope_parameters"]
    finally:
        _probing = False


def _merged_entry(value, old):
    """A rope_scaling value as one rope_parameters entry: the old spelling never carried the base, so the base
    (and the partial rotary factor) of the entry it replaces is kept, as the config.json route keeps it."""
    new = {"rope_type": "default"} if value is None else dict(value)
    if "rope_type" not in new and "type" in new:
        new["rope_type"] = new["type"]
    for k in ("rope_theta", "partial_rotary_factor"):
        if k not in new and k in old:
            new[k] = old[k]
    return new


def _plan(cfg, key, value):
    """What the write means under the config.json reading, without changing anything yet.

    Returns (value to store, entries to update in place, message). The message is None when nothing would be
    lost; then the write goes through as it is."""
    rp = cfg.__dict__["rope_parameters"]
    name = type(cfg).__name__
    flat = "rope_theta" in rp
    where = None if flat else _landing(type(cfg), key)
    if not flat and where is None:
        raise core.RoleError(
            f"{name}.{key} was set after the config was built. transformers 5 reads RoPE only from rope_parameters,"
            f" which here is per layer type ({', '.join(sorted(rp))}), and entail could not tell which layer"
            f" types the old name stands for. Set rope_parameters directly.")
    if key == "rope_theta":
        if cfg.__dict__.get("rope_theta", _MISSING) == value:
            return value, [], None  # the same value stated again (vLLM's patch_rope_parameters does this)
        targets = [rp] if flat else [rp[t] for t in where]
        before = targets[0].get("rope_theta")
        if all(t.get("rope_theta") == value for t in targets):
            return value, [], None
        at = "rope_parameters['rope_theta']" if flat else f"rope_parameters[{'/'.join(where)}]['rope_theta']"
        return value, targets, (f"{name}.rope_theta={value} was given after the config was built, where"
                                f" transformers 5 no longer reads it; wrote it to {at} (was {before}),"
                                f" as config.json would")
    # rope_scaling: the property setter replaces rope_parameters wholesale
    if flat:
        new = _standardized(cfg, _merged_entry(value, rp))
    else:
        new = copy.deepcopy(rp)
        for t in where:
            new[t] = _merged_entry(value, rp[t])
        new = _standardized(cfg, new)
    if flat and value is not None and all(value.get(k) == new.get(k) for k in new):
        return value, [], None  # the value already says everything the config.json route would add
    lost = rp.get("rope_theta") if flat else {t: rp[t].get("rope_theta") for t in where}
    return new, [], (f"{name}.rope_scaling was given after the config was built; it replaces rope_parameters and"
                     f" would have dropped rope_theta={lost}, leaving the base to whatever the engine's model file"
                     f" falls back to (10000 in vLLM's llama.py and SGLang's get_rope_config); kept it, as"
                     f" config.json would")


def _wrap(orig):
    def __setattr__(self, key, value):
        if key in LEGACY and _active and not _probing and core.mode() in ("load", "debug") and applies(self, key):
            store, entries, msg = _plan(self, key, value)
            if msg:
                if core.policy() == "refuse":
                    raise core.RoleError(f"{msg.split(';')[0]}; transformers 5 reads RoPE only from"
                                         f" rope_parameters, so the value would be lost")
                before = copy.deepcopy(self.__dict__.get("rope_parameters"))
                for e in entries:  # in place, so anything already holding the dict sees the change
                    e["rope_theta"] = value
                value = store
                after = value if key == "rope_scaling" else self.__dict__.get("rope_parameters")
                _shared.note({"engine": "transformers-config", "where": key, "from": before,
                              "to": copy.deepcopy(after)}, msg)
        orig(self, key, value)

    __setattr__.entail_wrapped = orig
    return __setattr__


def _subclasses(cls):
    seen, todo = [], list(cls.__subclasses__())
    while todo:
        c = todo.pop()
        if c not in seen:
            seen.append(c)
            todo.extend(c.__subclasses__())
    return seen


_WRAPPED = []  # (class, its own __setattr__ before we wrapped it)


def install():
    """Wrap PreTrainedConfig.__setattr__ and from_dict. Returns 1, or 0 if already installed.

    Every config class is decorated with huggingface_hub's @strict, which copies the __setattr__ it inherits into
    the class when the class is created (original_setattr = cls.__setattr__). A class created before this runs
    therefore never sees a change to the base class, so those are wrapped one by one; classes created later
    pick the wrapped base up by themselves. (The autoinstall shim runs before any model config class exists.)"""
    global _ORIG_SETATTR, _ORIG_FROM_DICT, _active
    try:
        from transformers.configuration_utils import PreTrainedConfig
    except ImportError:  # transformers 4.x: rope_theta and rope_scaling are still the names models read
        return 0

    if _ORIG_SETATTR is not None:
        return 0
    _active = True
    _ORIG_SETATTR = PreTrainedConfig.__setattr__
    _ORIG_FROM_DICT = PreTrainedConfig.__dict__["from_dict"].__func__
    for cls in _subclasses(PreTrainedConfig):
        own = cls.__dict__.get("__setattr__")
        if own is not None and not hasattr(own, "entail_wrapped"):
            _WRAPPED.append((cls, own))
            cls.__setattr__ = _wrap(own)
    __setattr__ = _wrap(_ORIG_SETATTR)

    def from_dict(cls, config_dict, **kwargs):
        if core.mode() not in ("load", "debug"):
            return _ORIG_FROM_DICT(cls, config_dict, **kwargs)
        legacy = {k: kwargs.pop(k) for k in LEGACY if k in kwargs}
        if not legacy:
            return _ORIG_FROM_DICT(cls, config_dict, **kwargs)
        want_unused = kwargs.get("return_unused_kwargs", False)
        out = _ORIG_FROM_DICT(cls, config_dict, **kwargs)
        config, unused = out if want_unused else (out, None)
        for k, v in legacy.items():
            if applies(config, k) or hasattr(config, k):
                setattr(config, k, v)  # converted by the hook above, or set as from_dict itself would
            elif want_unused:
                unused[k] = v  # what from_dict does with a key the class does not know
        return (config, unused) if want_unused else config

    PreTrainedConfig.__setattr__ = __setattr__
    PreTrainedConfig.from_dict = classmethod(from_dict)
    return 1


def uninstall():
    global _ORIG_SETATTR, _ORIG_FROM_DICT, _active
    if _ORIG_SETATTR is None:
        return 0
    _active = False
    from transformers.configuration_utils import PreTrainedConfig

    PreTrainedConfig.__setattr__ = _ORIG_SETATTR
    PreTrainedConfig.from_dict = classmethod(_ORIG_FROM_DICT)
    for cls, own in _WRAPPED:
        cls.__setattr__ = own
    _WRAPPED.clear()
    _ORIG_SETATTR = _ORIG_FROM_DICT = None
    _landing.cache_clear()
    return 1
