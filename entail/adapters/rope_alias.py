"""Adapter v2: a RoPE value written under its transformers-4 name after the config is built (LIBRARY_DESIGN.md 4.8;
ROADMAP M3.3). The rule is load.rotary_write's; this file only knows how transformers handles the old names.

transformers 5 keeps the RoPE settings in `rope_parameters`. A checkpoint that spells them the old way (`rope_theta`,
`rope_scaling` in config.json) is converted while the config is built, but a value given after that goes through
code that does not convert it. Measured (audits/ALIAS.md, audits/override_probe.py):

  AutoConfig.from_pretrained(p, rope_theta=x)          ignored, silently (it comes back as an unused kwarg)
  cfg.rope_theta = x                                   ignored, silently (nothing reads the attribute)
  vLLM hf_overrides / SGLang json_model_override_args  {"rope_theta": x} ignored, silently
  cfg.rope_scaling = {...}, and the same key through either engine's override
      replaces rope_parameters and drops rope_theta with it. Measured end to end on vLLM: Llama 3.2 falls to 10,000
      (GSM8K 379 -> 279 of 500), Qwen3 happens to fall to its own 1,000,000 (issue_track/rope_override).
Both engines apply their overrides with setattr on the transformers config, so one hook here covers all three.

  hooks        PreTrainedConfig.__setattr__ (and every config class created before the hook, since huggingface_hub's
               @strict copies __setattr__ into each class), and from_dict, which passes old-name kwargs to setattr.
  read_choice  for one write: what it means - the rope_parameters config.json would give for the same key (_file_route)
               - and what the model will read if transformers performs it as it does (_engine_route), as Rotary facts
               per layer type the key lands on. A write of the value the attribute already holds is a restatement
               (vLLM's patch_rope_parameters writes config.rope_theta back with the value it just read), not a new
               declaration, and is let through.
  handle       rope_write_as_file: write it the way config.json would.
Where the key lands in a per-layer-type rope_parameters (Gemma 3) is asked of the config class itself (_landing).
Not touched: classes that declare the old name as a field or map it through attribute_map (their models read the
attribute itself), and configs whose rope_parameters has not been built yet (the conversion still runs then).
"""
import copy
import dataclasses
import functools

from .. import core, load, policies
from ..readers import rotary_of
from .base import Hook

engine = "transformers"
versions = "5.12.1, 5.16.1, 5.17.0"
LEGACY = ("rope_theta", "rope_scaling")
_MARK_THETA = 12345.678  # values no real checkpoint uses, to see where the class puts each key
_MARK_FACTOR = 7.25
_MISSING = object()
_ORIG_SETATTR = None
_ORIG_FROM_DICT = None
_probing = False
_active = False  # classes created while installed keep our wrapper in their @strict closure; this turns it off


def hooks():
    return [Hook("transformers.configuration_utils.PreTrainedConfig.__setattr__", "load"),
            Hook("transformers.configuration_utils.PreTrainedConfig.from_dict", "load")]


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
    """Does the model read this key from rope_parameters, already built? (Otherwise the write is not ours.)"""
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


def _file_route(cfg, key, value, where):
    """rope_parameters as config.json would build them from this key: (the new dict, entries to update in place)."""
    rp = cfg.__dict__["rope_parameters"]
    flat = where is None
    if key == "rope_theta":
        targets = [rp] if flat else [rp[t] for t in where]
        new = copy.deepcopy(rp)
        for t in ([new] if flat else [new[t] for t in where]):
            t["rope_theta"] = value
        return new, targets
    if flat:
        return _standardized(cfg, _merged_entry(value, rp)), []
    new = copy.deepcopy(rp)
    for t in where:
        new[t] = _merged_entry(value, rp[t])
    return _standardized(cfg, new), []


def _engine_route(cfg, key, value):
    """rope_parameters after transformers performs the write as it does: rope_theta is stored as an attribute
    nothing reads; rope_scaling replaces rope_parameters wholesale."""
    global _probing
    rp = cfg.__dict__["rope_parameters"]
    if key == "rope_theta":
        return copy.deepcopy(rp)
    _probing = True
    try:
        tmp = copy.deepcopy(cfg)
        _ORIG_SETATTR(tmp, key, copy.deepcopy(value))
        return tmp.__dict__.get("rope_parameters")
    finally:
        _probing = False


def read_choice(cfg, key, value):
    """What one write means and what the model would read after it, per layer type the key lands on:
    {"pairs": [(layer type or "", meant Rotary, as-engine Rotary)], "problems": [...], "store", "targets"} - the
    last two are what the handle writes. None when the write is a restatement. {"unplaced": ...} when the class's
    per-layer RoPE cannot be told apart (then nobody can say what the old name stands for)."""
    if cfg.__dict__.get(key, _MISSING) == value and key == "rope_theta":
        return None
    rp = cfg.__dict__["rope_parameters"]
    where = None if "rope_theta" in rp else _landing(type(cfg), key)
    if "rope_theta" not in rp and where is None:
        return {"unplaced": f"rope_parameters is per layer type ({', '.join(sorted(rp))}) and the class does not "
                            f"say which layer types {key} stands for; set rope_parameters directly"}
    meant, targets = _file_route(cfg, key, value, where)
    as_engine = _engine_route(cfg, key, value)
    out, problems = [], []
    for scope in (where or ("",)):
        m = meant.get(scope) if scope else meant
        # the engine's write leaves per-layer entries (rope_theta) or one flat dict for every layer (rope_scaling)
        e = as_engine.get(scope) if scope and isinstance(as_engine, dict) and isinstance(as_engine.get(scope), dict) \
            else as_engine
        rm, pm = rotary_of(m)
        re_, pe = rotary_of(e) if isinstance(e, dict) else (None, [])
        problems += pm + pe
        out.append((scope, rm, re_))
    return {"pairs": out, "problems": problems, "store": meant if key == "rope_scaling" else value,
            "targets": targets}


def handles(cfg, key, plan):
    def write_as_file(_target):
        for e in plan["targets"]:   # in place, so anything already holding the dict sees the change
            e["rope_theta"] = plan["store"]
        return plan["store"]
    return {"rope_write_as_file": write_as_file}


def _decide(cfg, key, value):
    """The value to store for this write: unchanged, or converted as config.json would (load.rotary_write decides)."""
    plan = read_choice(cfg, key, value)
    if plan is None:
        return value
    policy = policies.current()
    boundary, consumer, owner = f"load:{engine}.config.{key}", f"{engine}.rotary_embedding", type(cfg).__name__
    if "unplaced" in plan:
        load.enforce([load.cannot_check(boundary, consumer, "Rotary", f"{owner}.{key}: {plan['unplaced']}", policy,
                                        meaning_changing=True)])
        return value
    decisions = []
    for scope, meant, as_engine in plan["pairs"]:
        if meant is None:
            decisions.append(load.cannot_check(boundary, consumer, "Rotary", "; ".join(plan["problems"]) or
                                               "not representable in vocabulary v1", policy, meaning_changing=True))
            continue
        decisions += load.rotary_write(engine, owner, key, meant, as_engine, scope, policy, config=cfg)
    done = load.resolve(decisions, handles(cfg, key, plan))
    load.enforce(decisions)
    return done.get("rope_write_as_file", value)


def _wrap(orig):
    def __setattr__(self, key, value):
        if key in LEGACY and _active and not _probing and core.mode() in ("load", "debug") and applies(self, key):
            value = load.safely(f"load:{engine}.config.{key}", f"{engine}.rotary_embedding", "Rotary",
                                lambda: _decide(self, key, value), value)
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
                setattr(config, k, v)  # decided by the hook above, or set as from_dict itself would
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
