"""Adapter v2 for vLLM's engine arguments: where the two safety modes turn an optimization off (LIBRARY_DESIGN.md 13.6;
ROADMAP product track P3).

  hook         vllm.engine.arg_utils.EngineArgs.create_engine_config: the arguments an offline LLM and the OpenAI
               server (AsyncEngineArgs) both turn into the engine's configuration, before any of it is built.
  read_choice  the configuration's key, and which optimizations of data/safe_mode.json these arguments leave on
               (an option whose value is not the value that turns it off)
  handles      safe_mode: set the options that turn one off (dotted paths: compilation_config.custom_ops;
               ir_op_priority.<op> puts the op's definition first, in the kernel config when the user set it there)
safe_mode decides what to turn off (ENTAIL_SAFE; the selective safe path's store); vllm_paths tells it what the
engine's paths came to.
"""
from .. import core, load, safe_mode
from .base import Hook

engine = "vllm"
versions = "0.30.0"
BOUNDARY = "start:vllm.safe_mode"
CONSUMER = "vllm.engine_args"
_ORIG = {}


def hooks():
    return [Hook("vllm.engine.arg_utils.EngineArgs.create_engine_config", "start")]


def _get(obj, path):
    for name in path.split("."):
        if obj is None:
            return None
        obj = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
    return obj


def _set(obj, path, value):
    *head, last = path.split(".")
    for name in head:
        nxt = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if nxt is None:
            nxt = {}
            if isinstance(obj, dict):
                obj[name] = nxt
            else:
                setattr(obj, name, nxt)
        obj = nxt
    if isinstance(obj, dict):
        obj[last] = value
    else:
        setattr(obj, last, value)


def _ir(args, path):
    """(where an IR op's priority list is, the list): the kernel config's when the user set it there (vLLM refuses a
    priority given in both places), else the engine arguments' own."""
    op = path.split(".", 1)[1]
    inner = _get(args, f"kernel_config.ir_op_priority.{op}")
    if inner:
        return f"kernel_config.ir_op_priority.{op}", list(inner)
    return path, list(_get(args, path) or [])


def _on(args, option, safe):
    if option.startswith("ir_op_priority."):   # off when the definition comes first; vLLM appends its own after
        return _ir(args, option)[1][:1] != safe[:1]
    current = _get(args, option)
    if safe is False:          # enable_prefix_caching: None is the engine's default, which is on
        return current is not False
    if isinstance(safe, list):
        return list(current or []) != safe
    return current != safe


def _turn(args, option, safe):
    if option.startswith("ir_op_priority."):
        where, current = _ir(args, option)
        _set(args, where, list(safe) + [p for p in current if p not in safe])
    else:
        _set(args, option, list(safe) if isinstance(safe, list) else safe)


def read_choice(args):
    """(configuration key, {feature: on}) for these engine arguments."""
    try:
        import vllm

        version = getattr(vllm, "__version__", "?")
    except ImportError:
        version = "?"
    key = safe_mode.config_key(engine, {"version": version, "model": str(getattr(args, "model", "")),
                                        "dtype": str(getattr(args, "dtype", "")),
                                        "quantization": str(getattr(args, "quantization", None)),
                                        "tp": getattr(args, "tensor_parallel_size", 1)})
    return key, {f: any(_on(args, option, safe) for option, safe in options)
                 for f, options in safe_mode.features(engine).items()}


def handles(args=None):
    return {"safe_mode": lambda target: [_turn(args, option, safe) for option, safe in target]}


def _decide(args):
    key, enabled = read_choice(args)
    items = safe_mode.plan(engine, key, enabled)
    where = f"vLLM engine arguments for {getattr(args, 'model', 'the model')}"
    decisions = safe_mode.decisions(engine, BOUNDARY, CONSUMER, items, where)
    if decisions:
        load.enforce(decisions, once_for=args)
        load.resolve(decisions, handles(args))
    safe_mode.started(engine, key, enabled, [f for f, _ in items], str(getattr(args, "model", "")))
    return decisions


def install():
    try:
        from vllm.engine.arg_utils import EngineArgs
    except ImportError:
        return 0
    if "create" in _ORIG:
        return 0
    orig = _ORIG["create"] = EngineArgs.create_engine_config

    def create_engine_config(self, *args, **kwargs):
        if core.mode() in ("load", "debug"):
            load.safely(BOUNDARY, CONSUMER, "SafeMode", lambda: _decide(self))
        return orig(self, *args, **kwargs)

    EngineArgs.create_engine_config = create_engine_config
    return 1


def uninstall():
    if "create" in _ORIG:
        from vllm.engine.arg_utils import EngineArgs

        EngineArgs.create_engine_config = _ORIG.pop("create")
        return 1
    return 0


def stats():
    return {}


def reset():
    safe_mode.LAST.pop(engine, None)
