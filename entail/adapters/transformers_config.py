"""Adapter v2: which keys of a config.json the model's config class takes (LIBRARY_DESIGN.md 4.8; ROADMAP M3.3;
test problem rolebench 15). transformers keeps a key its class does not know as a plain attribute and says nothing,
so a misspelt key (rope_scale for rope_scaling) runs with the default instead. vLLM and SGLang build their model
configs through the same transformers classes, so this one hook covers all three engines.

  hook         PreTrainedConfig.from_dict: the config dict as read from config.json, and the object built from it.
  read_choice  for the top level and a nested text_config: the raw dict, the fields the class knows (a default
               instance of the class, cached per class), and what the object holds of them.
  handles      none: a key the class does not know cannot be put anywhere by the library.
load.config_keys decides (a key neither known nor surviving by its value -> refused); load.enforce records.
"""
import functools

from .. import core, load, policies
from .base import Hook

engine = "transformers"
versions = "5.12.1, 5.16.1, 5.17.0"
_ORIG = None
_MINE = None   # our wrapper, so uninstall restores only what it replaced (rope_alias wraps from_dict too)


def hooks():
    return [Hook("transformers.configuration_utils.PreTrainedConfig.from_dict", "load")]


def handles():
    return {}


@functools.lru_cache(maxsize=None)
def _known(cls):
    """The fields a config class declares: the keys of a default instance's to_dict(), and the standard names the
    class renames on the way in (`attribute_map`: GPT-2 stores hidden_size as n_embd; M12.1), or None if it cannot
    be built."""
    try:
        known = set(cls().to_dict())
    except Exception:  # noqa: BLE001 - a class that cannot be built with defaults gives no answer
        return None
    renamed = getattr(cls, "attribute_map", None)
    if isinstance(renamed, dict):
        known |= set(renamed)
    return frozenset(known)


def read_choice(config, raw):
    """[(prefix, raw dict, fields the class knows, what the object holds)] for the top level and text_config; None
    when a class cannot say which fields it knows."""
    scopes = []
    for prefix, obj, d in (("", config, raw), ("text_config.", getattr(config, "text_config", None),
                                             raw.get("text_config") if isinstance(raw, dict) else None)):
        if obj is None or not isinstance(d, dict):
            continue
        known = _known(type(obj))
        if known is None:
            return None
        scopes.append((prefix, d, known, obj.to_dict()))
    return scopes


def install():
    global _ORIG, _MINE
    try:
        from transformers.configuration_utils import PreTrainedConfig
    except ImportError:
        return 0
    if _ORIG is not None:
        return 0
    _ORIG = PreTrainedConfig.__dict__["from_dict"].__func__

    def from_dict(cls, config_dict, **kwargs):
        out = _ORIG(cls, config_dict, **kwargs)
        if core.mode() not in ("load", "debug") or not isinstance(config_dict, dict):
            return out
        config = out[0] if isinstance(out, tuple) else out

        def decide():
            # the file's own _name_or_path can be stale (a checkpoint copied from another: ms-marco-MiniLM-L-6-v2's
            # config names L-12-v2), so it is quoted as the file's claim, not used as the model's name (M15 review)
            own = config_dict.get("_name_or_path")
            where = f"{type(config).__name__} config.json" + (f" (the file names itself {own})" if own else "")
            scopes = read_choice(config, config_dict)
            policy = policies.current()
            if scopes is None:
                decisions = [load.cannot_check(f"load:{engine}.config", f"{engine}.config", "Coverage",
                                               f"{type(config).__name__} cannot be built with defaults to list its "
                                               f"fields", policy)]
            else:
                decisions = load.config_keys(engine, scopes, where, policy)
            # an engine builds the same config several times in one process (vLLM: the model config, then again for
            # the tokenizer and the scheduler; transformers: AutoConfig, then the model class), and each build said
            # the same unread keys again - 69 unknown lines became 306 over 81 runs (M15.6 E2). The same decision for
            # the same class and file name is recorded once per process; a different decision still is.
            load.enforce(decisions, once_for=(type(config).__name__, own))

        load.safely(f"load:{engine}.config", f"{engine}.config", "Coverage", decide)
        return out

    _MINE = from_dict
    PreTrainedConfig.from_dict = classmethod(from_dict)
    return 1


def uninstall():
    global _ORIG, _MINE
    if _ORIG is None:
        return 0
    from transformers.configuration_utils import PreTrainedConfig

    if PreTrainedConfig.__dict__["from_dict"].__func__ is _MINE:
        PreTrainedConfig.from_dict = classmethod(_ORIG)
    _ORIG = _MINE = None
    _known.cache_clear()
    return 1
