"""Adapter v2 for vLLM's custom ops against their own native definitions (LIBRARY_DESIGN.md 11 M18; ROADMAP M18.2;
kernel_reference_contract.py; vllm#42016).

  hook         vllm.model_executor.model_loader.utils.process_weights_after_loading: the model is built. Every
               CustomOp module whose dispatched forward is a kernel path (custom_op.py keeps it as _forward_method;
               it is forward_native when custom ops are off, as under torch.compile by default) gets that method
               wrapped. The wrapper compares once per (op class, input pattern) per process, on the first call, and
               then puts the original back, so the steady state costs nothing.
  read_choice  the kernel's and the definition's outputs on a slice of the real input (ROWS rows of the token
               dimension), the definition run in the input dtype and in float32; the rule is in the core.
  handles      none: a kernel that differs from its definition is reported. Switching the op to its native path is
               a repair to be measured (cost, CUDA graphs) before it is offered.
Not compared: ops that override forward() (the mamba mixers, static sink attention: stateful, not row-wise), calls
made while a CUDA graph is being captured (nothing may be launched there), ops with no tensor argument, and ops
whose definition raises on the input (said unknown once). vLLM caches rotary modules process-wide, so one instance
serves every layer; each instance is wrapped once.
"""
from .. import core, kernel_reference_contract
from .base import Hook

engine = "vllm"
versions = "0.30.0"
BOUNDARY = "kernel:vllm.custom_op"
CONSUMER = "vllm.custom_op"
_ORIG = None
_DECIDED = set()       # (op class name, input pattern) decided in this process
_WRAPPED = {}          # id(module) -> (module, original _forward_method)
_STATS = {"instrumented": 0, "native": 0, "overrides_forward": 0}


def hooks():
    return [Hook("vllm.model_executor.model_loader.utils.process_weights_after_loading", "kernel")]


def handles():
    return {}


def _overrides_forward(module) -> bool:
    """Whether a class between the module's own and CustomOp defines forward (the dispatch is bypassed there)."""
    for cls in type(module).__mro__:
        if cls.__name__ == "CustomOp":
            return False
        if "forward" in cls.__dict__:
            return True
    return False


def _is_native(fwd) -> bool:
    name = getattr(fwd, "__name__", "")
    inner = getattr(fwd, "__wrapped__", None)
    return name == "forward_native" or getattr(inner, "__name__", "") == "forward_native"


def is_candidate(module) -> bool:
    """A CustomOp instance dispatching to a kernel path: it has the dispatched method and a native definition."""
    fwd = getattr(module, "_forward_method", None)
    if fwd is None or not callable(getattr(module, "forward_native", None)):
        return False
    if not any(cls.__name__ == "CustomOp" for cls in type(module).__mro__):
        return False
    if _overrides_forward(module):
        _STATS["overrides_forward"] += 1
        return False
    if _is_native(fwd):
        _STATS["native"] += 1
        return False
    return True


def pattern(args, kwargs) -> tuple:
    def one(x):
        if hasattr(x, "dtype") and hasattr(x, "dim"):
            return ("t", str(x.dtype), int(x.dim()))
        if isinstance(x, (tuple, list)):
            return (type(x).__name__,) + tuple(one(y) for y in x)
        return (type(x).__name__,)

    return tuple(one(a) for a in args) + tuple((k, one(v)) for k, v in sorted(kwargs.items()))


def capturing() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        return False


def read_choice(module, orig, args, kwargs):
    """(kernel output, definition output in the input dtype, definition output in float32 or None, rows) on a
    slice of the arguments; every slice is a clone, so the engine's tensors are untouched."""
    import torch

    cut = kernel_reference_contract.sliced
    n = kernel_reference_contract.rows_of(args, kwargs)
    if n is None:
        return None
    rows = min(kernel_reference_contract.ROWS, n)
    with torch.no_grad():
        k_out = orig(*cut(args, n, rows), **cut(kwargs, n, rows))
        n_out = module.forward_native(*cut(args, n, rows), **cut(kwargs, n, rows))
        try:
            r_out = module.forward_native(*cut(args, n, rows, torch.float32), **cut(kwargs, n, rows, torch.float32))
        except Exception:  # noqa: BLE001 - a definition that refuses float32 (a kernel-backed native path)
            r_out = None
    return k_out, n_out, r_out, rows


def _decide(module, name, orig, args, kwargs):
    from .. import load

    got = read_choice(module, orig, args, kwargs)
    consumer = f"vllm.{name}"
    if got is None:
        load.enforce([load.cannot_check(BOUNDARY, consumer, "KernelReference",
                                        f"{name}: no tensor argument to slice, so its kernel is not compared")])
        return
    k_out, n_out, r_out, rows = got
    outs = kernel_reference_contract.tensors_of(k_out)
    if not outs:
        load.enforce([load.cannot_check(BOUNDARY, consumer, "KernelReference",
                                        f"{name}: the kernel returns no floating tensor to compare")])
        return
    reference, native = (r_out, n_out) if r_out is not None else (n_out, None)
    diff, floor, scale, elements = kernel_reference_contract.compare(k_out, reference, native)
    where = f"{name}'s dispatched {getattr(orig, '__name__', 'kernel')} ({type(module).__module__}) on {rows} rows " \
            f"of its first call"
    kernel_reference_contract.check(BOUNDARY, consumer, name, str(outs[0].dtype), diff, floor, scale, elements, where)


def wrap(module) -> bool:
    """Wrap the module's dispatched forward so its first call (per op class and input pattern) is compared with
    the definition; afterwards the original is put back on this module. Returns whether it was wrapped."""
    if id(module) in _WRAPPED or not is_candidate(module):
        return False
    orig = module._forward_method
    name = type(module).__name__

    def run(*args, **kwargs):
        out = orig(*args, **kwargs)
        try:
            key = (name, pattern(args, kwargs))
            if key in _DECIDED:
                module._forward_method = orig
                return out
            if capturing():
                return out
            _DECIDED.add(key)
            module._forward_method = orig
            from .. import load

            load.safely(BOUNDARY, f"vllm.{name}", "KernelReference", lambda: _decide(module, name, orig, args, kwargs))
        except Exception:  # noqa: BLE001 - never the engine's problem (principle 12)
            pass
        return out

    module._forward_method = run
    _WRAPPED[id(module)] = (module, orig)
    _STATS["instrumented"] += 1
    return True


def instrument(model) -> int:
    """Wrap every candidate custom op of the model. Returns how many were wrapped; the counts (wrapped, dispatching
    to the native definition, overriding forward) go to the record as one line, so a run where nothing was
    compared says why."""
    n = 0
    for _, m in model.named_modules():
        if wrap(m):
            n += 1
    try:
        import os

        from .. import load

        load._write({"pid": os.getpid(), "boundary": BOUNDARY, "kernel_reference": {
            "instrumented": n, "native": _STATS["native"], "overrides_forward": _STATS["overrides_forward"],
            "model": type(model).__name__}})
    except Exception:  # noqa: BLE001 - a record that cannot be written is only a missing line
        pass
    return n


def install():
    global _ORIG
    try:
        from vllm.model_executor.model_loader import utils as loader_utils
    except ImportError:
        return 0
    if _ORIG is not None:
        return 0
    _ORIG = loader_utils.process_weights_after_loading

    def wrapped(model, model_config, target_device, *a, **kw):
        out = _ORIG(model, model_config, target_device, *a, **kw)
        if core.mode() in ("load", "debug"):
            from .. import load

            load.safely(BOUNDARY, CONSUMER, "KernelReference", lambda: instrument(model))
        return out

    loader_utils.process_weights_after_loading = wrapped
    return 1


def uninstall():
    global _ORIG
    for module, orig in list(_WRAPPED.values()):
        module._forward_method = orig
    _WRAPPED.clear()
    if _ORIG is None:
        return 0
    from vllm.model_executor.model_loader import utils as loader_utils

    loader_utils.process_weights_after_loading = _ORIG
    _ORIG = None
    return 1


def stats():
    out = dict(_STATS)
    out["decided"] = len(_DECIDED)
    out.update(kernel_reference_contract.stats(BOUNDARY))
    return out


def reset():
    for module, orig in list(_WRAPPED.values()):
        module._forward_method = orig
    _WRAPPED.clear()
    _DECIDED.clear()
    for k in _STATS:
        _STATS[k] = 0
    kernel_reference_contract.reset(BOUNDARY)
