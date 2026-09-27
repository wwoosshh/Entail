"""Adapter v2 for SGLang's server arguments: where the two safety modes turn an optimization off (LIBRARY_DESIGN.md
13.6; ROADMAP product track P3).

  hook         sglang.srt.server_args.ServerArgs.__post_init__: the arguments the Engine and the server are built
               from, before SGLang derives anything from them.
  read_choice  the configuration's key, and which optimizations of data/safe_mode.json these arguments leave on
  handles      safe_mode: set the options that turn one off
safe_mode decides what to turn off. The selective safe path needs the engine's path check, which on SGLang runs only
with ENTAIL_PATHS=1 (sglang_paths); the explicit safe mode (ENTAIL_SAFE=all) needs nothing else.
"""
from .. import core, load, safe_mode
from .base import Hook

engine = "sglang"
versions = "0.5.20"
BOUNDARY = "start:sglang.safe_mode"
CONSUMER = "sglang.server_args"
_ORIG = {}


def hooks():
    return [Hook("sglang.srt.server_args.ServerArgs.__post_init__", "start")]


def read_choice(args):
    """(configuration key, {feature: on}) for these server arguments."""
    try:
        import sglang

        version = getattr(sglang, "__version__", "?")
    except ImportError:
        version = "?"
    key = safe_mode.config_key(engine, {"version": version, "model": str(getattr(args, "model_path", "")),
                                        "dtype": str(getattr(args, "dtype", "")),
                                        "quantization": str(getattr(args, "quantization", None)),
                                        "tp": getattr(args, "tp_size", 1)})
    return key, {f: any(getattr(args, option, None) != safe for option, safe in options)
                 for f, options in safe_mode.features(engine).items()}


def handles(args=None):
    return {"safe_mode": lambda target: [setattr(args, option, safe) for option, safe in target]}


def _decide(args):
    key, enabled = read_choice(args)
    items = safe_mode.plan(engine, key, enabled)
    where = f"SGLang server arguments for {getattr(args, 'model_path', 'the model')}"
    decisions = safe_mode.decisions(engine, BOUNDARY, CONSUMER, items, where)
    if decisions:
        load.enforce(decisions, once_for=args)
        load.resolve(decisions, handles(args))
    safe_mode.started(engine, key, enabled, [f for f, _ in items], str(getattr(args, "model_path", "")))
    return decisions


def install():
    try:
        from sglang.srt.server_args import ServerArgs
    except ImportError:
        return 0
    if "post_init" in _ORIG:
        return 0
    orig = _ORIG["post_init"] = ServerArgs.__post_init__

    def __post_init__(self):
        if core.mode() in ("load", "debug"):
            load.safely(BOUNDARY, CONSUMER, "SafeMode", lambda: _decide(self))
        return orig(self)

    ServerArgs.__post_init__ = __post_init__
    return 1


def uninstall():
    if "post_init" in _ORIG:
        from sglang.srt.server_args import ServerArgs

        ServerArgs.__post_init__ = _ORIG.pop("post_init")
        return 1
    return 0


def stats():
    return {}


def reset():
    safe_mode.LAST.pop(engine, None)
