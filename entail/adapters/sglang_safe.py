"""Adapter v2 for SGLang's server arguments: where the two safety modes turn an optimization off (LIBRARY_DESIGN.md
13.6; ROADMAP product track P3).

  hook         sglang.srt.server_args.ServerArgs.resolve_once (0.5.20): the arguments the Engine and the server are
               built from, just before SGLang resolves them - resolution seals the record, and a child process gets
               it resolved already. ServerArgs is a msgspec Struct there: its __post_init__ is empty and is looked up
               when the class is made, so replacing it later is never called (found live, P3). Versions without
               resolve_once (a dataclass whose __post_init__ resolves) are hooked at __post_init__.
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
    return [Hook("sglang.srt.server_args.ServerArgs.resolve_once", "start")]


def _version():
    try:
        import sglang

        return getattr(sglang, "__version__", "?")
    except ImportError:
        return "?"


def read_options(args):
    """({feature: [the table's options these arguments have]}, {feature: [the ones they lack]})."""
    return safe_mode.missing(engine, lambda option: hasattr(args, option))


def read_choice(args):
    """(configuration key, {feature: on}) for these server arguments; a feature this SGLang has no option for is
    not in it."""
    key = safe_mode.config_key(engine, {"version": _version(), "model": str(getattr(args, "model_path", "")),
                                        "dtype": str(getattr(args, "dtype", "")),
                                        "quantization": str(getattr(args, "quantization", None)),
                                        "tp": getattr(args, "tp_size", 1)})
    have, _ = read_options(args)
    return key, {f: any(getattr(args, option, None) != safe for option, safe in options) for f, options in have.items()}


def handles(args=None):
    return {"safe_mode": lambda target: [setattr(args, option, safe) for option, safe in target]}


def _decide(args):
    key, enabled = read_choice(args)
    have, lack = read_options(args)
    items = safe_mode.plan(engine, key, enabled)
    where = f"SGLang server arguments for {getattr(args, 'model_path', 'the model')}"
    decisions = safe_mode.decisions(engine, BOUNDARY, CONSUMER, items, where, options=have)
    if safe_mode.mode() == "all":
        decisions = decisions + safe_mode.cannot_turn(engine, BOUNDARY, CONSUMER, lack, have, _version())
    if decisions:
        load.enforce(decisions, once_for=args)
        load.resolve(decisions, handles(args))
    safe_mode.started(engine, key, enabled, [f for f, _ in items], str(getattr(args, "model_path", "")))
    return decisions


def _unresolved(args) -> bool:
    return not getattr(args, "_resolution_finished", False) and not getattr(args, "_resolution_failed", False)


def install():
    try:
        from sglang.srt.server_args import ServerArgs
    except ImportError:
        return 0
    if _ORIG:
        return 0
    if hasattr(ServerArgs, "resolve_once"):
        orig = _ORIG["resolve_once"] = ServerArgs.resolve_once

        def resolve_once(self, *args, **kwargs):
            if core.mode() in ("load", "debug") and _unresolved(self):
                load.safely(BOUNDARY, CONSUMER, "SafeMode", lambda: _decide(self))
            return orig(self, *args, **kwargs)

        ServerArgs.resolve_once = resolve_once
        return 1
    orig = _ORIG["__post_init__"] = ServerArgs.__post_init__

    def __post_init__(self, *args, **kwargs):
        if core.mode() in ("load", "debug"):
            load.safely(BOUNDARY, CONSUMER, "SafeMode", lambda: _decide(self))
        return orig(self, *args, **kwargs)

    ServerArgs.__post_init__ = __post_init__
    return 1


def uninstall():
    if not _ORIG:
        return 0
    from sglang.srt.server_args import ServerArgs

    for name, fn in list(_ORIG.items()):
        setattr(ServerArgs, name, fn)
        _ORIG.pop(name)
    return 1


def stats():
    return {}


def reset():
    safe_mode.LAST.pop(engine, None)
