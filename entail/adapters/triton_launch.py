"""Adapter v2 for Triton kernel launches, engine-independent (LIBRARY_DESIGN.md 4.8; ROADMAP M17.3;
kernel_launch_contract.py).

  hook         triton.runtime.jit.JITFunction.run: every `kernel[grid](*args, **kwargs)` of a @triton.jit kernel
               launched eagerly, by any engine in the process (3.7 and 3.8), goes through it. Kernels Inductor
               generates for a compiled forward and AOT-compiled kernels do not. The parameters' names come from the
               kernel's own signature (JITFunction.params), so the launch's arguments are bound to names without
               knowing the kernel.
  read_choice  the bound arguments of one launch: parameter name -> argument.
  handles      none: a launch is not repaired here; a strided tensor the kernel cannot know about is reported.
Each (kernel, stride pattern of its tensor arguments) is decided once per process: the pattern is per tensor its
rank and innermost stride (1, 0 or the strided value), not its shape, so the decode shapes of a server share one
pattern and a strided tensor at a new shape is still seen. A kernel is looked at for its first LIMIT strided
patterns only. Compile-only warm-ups (JITFunction.warmup, `warmup=True`) launch nothing and are skipped; the
autotuner's benchmark launches are ordinary launches, memoised after the first. Cost: a pattern key per launch (a
few microseconds; within the noise of the S4 CUDA-graph measurement, testbed/results/m17/m55_v2) - the 4.8-6% that
the first M17.3 measurement showed at batch 32 was the record file being opened per line, not this hook.
Principle 6 (no checks while a CUDA graph is captured): a capture-time launch is the first launch of its pattern,
decided once, before replay - the graph replays carry no Python.
"""
from .. import core, kernel_launch_contract
from .base import Hook

engine = "triton"
versions = "3.7.1, 3.8.0"
_ORIG = None
_SEEN = set()
_COUNT = {}        # id(kernel) -> strided patterns decided; past LIMIT the kernel's strided launches are not looked at
LIMIT = 8


def hooks():
    return [Hook("triton.runtime.jit.JITFunction.run", "kernel")]


def _names(fn):
    try:
        return [p.name for p in fn.params]
    except AttributeError:
        return []


def value_params(fn):
    """The kernel's non-constexpr parameters: the ones whose integers can be strides."""
    try:
        return [p.name for p in fn.params if not getattr(p, "is_constexpr", False)]
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
    """(kernel, per tensor argument: position, rank, innermost stride) - shape-free."""
    parts = []
    for i, v in enumerate(list(args) + list(kwargs.values())):
        if hasattr(v, "stride") and hasattr(v, "shape"):
            inner = kernel_launch_contract.innermost_stride(v)
            try:
                rank = len(v.shape)
            except TypeError:
                rank = None
            parts.append((i, rank, inner[2] if inner else None))
    return (id(fn), tuple(parts))


def _strided(key):
    return any(s not in (None, 0, 1) for _, _, s in key[1])


def should_look(fn, args, kwargs):
    """The pattern key when this launch is the first of its pattern and within the kernel's cap, else None."""
    key = _layout_key(fn, args, kwargs)
    if key in _SEEN:
        return None
    strided = _strided(key)
    if strided and _COUNT.get(id(fn), 0) >= LIMIT:
        return None
    _SEEN.add(key)
    if strided:
        _COUNT[id(fn)] = _COUNT.get(id(fn), 0) + 1
    return key


def _where(fn):
    f = getattr(fn, "fn", fn)
    module = getattr(f, "__module__", "") or ""
    eng = module.split(".", 1)[0] or "kernel"
    return eng, getattr(f, "__name__", "kernel"), module


def _decide(fn, args, kwargs, key):
    eng, name, module = _where(fn)
    bound = read_choice(fn, args, kwargs)
    kernel_launch_contract.check(f"kernel:{eng}.{name}", f"{eng}.{name}", name, bound, f"{module}.{name} launch",
                                 owner=key, ints_from=set(value_params(fn)))


def install():
    global _ORIG
    try:
        from triton.runtime.jit import JITFunction
    except ImportError:
        return 0
    if _ORIG is not None:
        return 0
    _ORIG = JITFunction.run

    def run(self, *args, **kwargs):
        if not kwargs.get("warmup") and core.mode() in ("load", "debug"):
            key = should_look(self, args, kwargs)
            if key is not None:
                from .. import load

                load.safely("kernel:triton", "triton.launch", "Layout", lambda: _decide(self, args, kwargs, key))
        return _ORIG(self, *args, **kwargs)

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
    return {"kernels_seen": len(_COUNT), "patterns_decided": len(_SEEN)}


def reset():
    _SEEN.clear()
    _COUNT.clear()
