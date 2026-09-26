"""Adapter v2 for Triton kernel launches, engine-independent (LIBRARY_DESIGN.md 4.8; ROADMAP M17.3;
kernel_launch_contract.py).

  hook         triton.runtime.jit.JITFunction.run: every `kernel[grid](*args, **kwargs)` of every engine in the
               process goes through it (3.7 and 3.8). The parameters' names come from the kernel's own signature
               (JITFunction.params), so the launch's arguments are bound to names without knowing the kernel.
  read_choice  the bound arguments of one launch: parameter name -> argument.
  handles      none: a launch is not repaired here; a strided tensor the kernel cannot know about is reported.
Each (kernel, layout of its tensor arguments) is decided once per process, and a kernel is looked at for its
first LIMIT distinct layouts only: reading the strides of every launch cost 4.8% at batch 32 on vLLM's CUDA-graph
path (the kernels outside the graph launch through Python at every step), the cap keeps the steady state at a
dictionary lookup. A warm-up launch (autotuning) is not looked at. The boundary is kernel:<engine>.<kernel name>,
the engine read from the kernel's module (sglang, vllm, ...). Principle 6 (no checks while a CUDA graph is
captured): a capture-time launch is the first launch of its layout, decided once, before replay - the graph
replays carry no Python.
"""
from .. import core, kernel_launch_contract
from .base import Hook

engine = "triton"
versions = "3.7.1, 3.8.0"
_ORIG = None
_SEEN = set()
_COUNT = {}        # id(kernel) -> distinct layouts decided; past LIMIT the kernel's launches are not looked at
LIMIT = 8          # S4: reading every launch's strides cost 4.8% at batch 32 on vLLM's graph path (M17.3 first run)


def hooks():
    return [Hook("triton.runtime.jit.JITFunction.run", "kernel")]


def _names(fn):
    try:
        return [p.name for p in fn.params]
    except AttributeError:
        return []


def read_choice(fn, args, kwargs):
    """parameter name -> argument for one launch (positional by the kernel's signature, then keywords)."""
    bound = dict(zip(_names(fn), args))
    bound.update(kwargs)
    return bound


def handles(fn):
    return {}


def _layout_key(fn, args, kwargs):
    parts = []
    for i, v in enumerate(list(args) + list(kwargs.values())):
        if hasattr(v, "stride") and hasattr(v, "shape"):
            try:
                parts.append((i, tuple(v.shape), tuple(v.stride())))
            except (TypeError, RuntimeError):
                parts.append((i, None, None))
    return (id(fn), tuple(parts))


def _where(fn):
    f = getattr(fn, "fn", fn)
    module = getattr(f, "__module__", "") or ""
    eng = module.split(".", 1)[0] or "kernel"
    return eng, getattr(f, "__name__", "kernel"), module


def _decide(fn, args, kwargs, key):
    eng, name, module = _where(fn)
    bound = read_choice(fn, args, kwargs)
    kernel_launch_contract.check(f"kernel:{eng}.{name}", f"{eng}.{name}", name, bound, f"{module}.{name} launch",
                                 owner=key)


def install():
    global _ORIG
    try:
        from triton.runtime.jit import JITFunction
    except ImportError:
        return 0
    if _ORIG is not None:
        return 0
    _ORIG = JITFunction.run

    def run(self, *args, grid, warmup, **kwargs):
        if not warmup and core.mode() in ("load", "debug") and _COUNT.get(id(self), 0) < LIMIT:
            key = _layout_key(self, args, kwargs)
            if key not in _SEEN:
                _SEEN.add(key)
                _COUNT[id(self)] = _COUNT.get(id(self), 0) + 1
                from .. import load

                load.safely("kernel:triton", "triton.launch", "Layout", lambda: _decide(self, args, kwargs, key))
        return _ORIG(self, *args, grid=grid, warmup=warmup, **kwargs)

    JITFunction.run = run
    return 1


def uninstall():
    global _ORIG
    if _ORIG is None:
        return 0
    from triton.runtime.jit import JITFunction

    JITFunction.run = _ORIG
    _ORIG = None
    _SEEN.clear()
    _COUNT.clear()
    return 1


def stats():
    return {"kernels_seen": len(_COUNT), "layouts_decided": len(_SEEN)}


def reset():
    _SEEN.clear()
    _COUNT.clear()
