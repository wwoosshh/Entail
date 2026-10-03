"""kernel_calls: every place the engine hands values to a GPU kernel, and what meaning those values carry
(ROADMAP M19 L6, the first piece: the map of the call sites the one rule has to reach).

Three kinds of call site are seen, each by a hook on the launcher it goes through:
  dispatch   a PyTorch operation (ATen or an engine's custom op such as vllm's torch.ops._C.*) with a CUDA tensor
             argument, seen by a TorchDispatchMode pushed on the thread
  triton     a Triton kernel launched through triton.runtime.jit.JITFunction.run (the engine's own kernels)
  inductor   a Triton kernel torch.compile generated, launched through CachingAutotuner.run
A call inside a CUDA graph replay is not seen here (nothing of Python runs then); the capture is. The GPU's own
count of what ran, replays included, is torch.profiler's (the harness takes it apart).

What is recorded per distinct call (kind, name, and the arguments' shapes, dtypes, strides and integer values):
how many times it was made, whether under capture, and for every tensor argument the meaning a producer attached
(guarantee.Issue: its role), found by the tensor's storage, so a view of an issued tensor is recognised too. A
tensor with no issue carries no meaning the checker could hold the kernel to: that is what this map counts.
Nothing is decided here, nothing is refused, no result is changed.
"""
import functools
import threading
import weakref

_LOCAL = threading.local()
_CALLS = {}            # key -> {"n": count, "captured": count, "first": order, "facts": tuple per tensor arg}
_ORDER = [0]
_STATS = {}
_WRAPPED = {}
_MODE = None


def _count(k, by=1):
    _STATS[k] = _STATS.get(k, 0) + by


def _capturing():
    try:
        import torch

        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        return False


def _tracing():
    """torch.compile is tracing (no real call runs): nothing to record."""
    try:
        import torch

        return bool(torch.compiler.is_compiling())
    except Exception:  # noqa: BLE001
        return False


def _is_fake(t):
    try:
        from torch._subclasses.fake_tensor import FakeTensor

        return isinstance(t, FakeTensor)
    except Exception:  # noqa: BLE001
        return False


def _issue_of(t):
    """The producer's issue for a tensor that is, or views, an issued tensor (guarantee._ISSUES, by storage)."""
    try:
        from . import guarantee

        iss = guarantee.tag(t)
        if iss is not None:
            return iss
        off = t.storage_offset()
        start = t.data_ptr() - off * t.element_size()
        for _k, (ref, iss) in list(guarantee._ISSUES._d.items()):
            if ref() is not None and iss.snap[0] == t.device and iss.snap[1] == start:
                return iss
    except Exception:  # noqa: BLE001
        pass
    return None


def _summary(v):
    """One argument as part of a call's key, and the meaning it carries ((kind, ...), fact)."""
    import torch

    if isinstance(v, torch.Tensor):
        if _is_fake(v):
            return ("fake_tensor", str(v.dtype).replace("torch.", ""), tuple(v.shape)), None
        if not v.is_cuda:
            return ("cpu_tensor", str(v.dtype).replace("torch.", ""), tuple(v.shape)), None
        iss = _issue_of(v)
        fact = None if iss is None else (iss.role, iss.serial)
        return ("T", str(v.dtype).replace("torch.", ""), tuple(v.shape), tuple(v.stride())), fact
    if isinstance(v, bool):
        return ("b", v), None
    if isinstance(v, int):
        return ("i", v), None
    if isinstance(v, float):
        return ("f", v), None
    if v is None:
        return ("none",), None
    if isinstance(v, (list, tuple)):
        parts = [_summary(x) for x in v]
        return ("seq", tuple(p[0] for p in parts)), tuple(p[1] for p in parts if p[0][0] == "T") or None
    if isinstance(v, torch.dtype):
        return ("dtype", str(v)), None
    if isinstance(v, torch.device):
        return ("device", str(v)), None
    return (type(v).__name__,), None


def _record(kind, name, args, kwargs, extra=()):
    try:
        sig = []
        facts = []
        for v in list(args) + [kwargs[k] for k in sorted(kwargs)]:
            s, f = _summary(v)
            sig.append(s)
            if s[0] == "T":
                facts.append(f)
            elif s[0] == "seq" and f:
                facts.extend(f)
        key = (kind, name, tuple(sig), tuple(extra))
        e = _CALLS.get(key)
        if e is None:
            _ORDER[0] += 1
            e = _CALLS[key] = {"n": 0, "captured": 0, "first": _ORDER[0], "facts": tuple(facts)}
        e["n"] += 1
        if _capturing():
            e["captured"] += 1
        _count(kind)
    except Exception:  # noqa: BLE001 - a record that fails is counted, never the engine's problem
        _count("record_failed")


def _mode_class():
    from torch.utils._python_dispatch import TorchDispatchMode

    class Calls(TorchDispatchMode):
        supports_higher_order_operators = True      # torch.compile's wrappers (auto_functionalized) pass through

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if not getattr(_LOCAL, "off", False) and not _tracing():
                import torch

                if any(isinstance(a, torch.Tensor) and a.is_cuda for a in args) or \
                        any(isinstance(a, torch.Tensor) and a.is_cuda for a in kwargs.values()):
                    _record("dispatch", str(func), args, kwargs)
            return func(*args, **kwargs)

    return Calls


def install_dispatch():
    global _MODE
    if getattr(_LOCAL, "pushed", False):
        return 0
    from torch.utils._python_dispatch import _push_mode

    if _MODE is None:
        _MODE = _mode_class()
    _push_mode(_MODE())
    _LOCAL.pushed = True
    return 1


def install_triton():
    import sys

    mod = sys.modules.get("triton.runtime.jit")
    J = getattr(mod, "JITFunction", None) if mod is not None else None
    if J is None or getattr(J.run, "__entail_calls__", False):
        return 0
    orig = J.run

    @functools.wraps(orig)
    def run(self, *args, grid, warmup, **kwargs):
        if not warmup:
            f = getattr(self, "fn", None)
            name = f"{getattr(f, '__module__', '?')}.{getattr(f, '__qualname__', '?')}"
            try:
                g = grid(kwargs) if callable(grid) else grid
                g = tuple(int(x) for x in (g if isinstance(g, (tuple, list)) else (g,)))
            except Exception:  # noqa: BLE001
                g = ("?",)
            _record("triton", name, args, kwargs, extra=("grid",) + g)
        return orig(self, *args, grid=grid, warmup=warmup, **kwargs)

    run.__entail_calls__ = True
    J.run = run
    _WRAPPED[(J, "run")] = orig
    return 1


def install_inductor():
    try:
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner as C
    except Exception:  # noqa: BLE001
        return 0
    if getattr(C.run, "__entail_calls__", False):
        return 0
    orig = C.run

    @functools.wraps(orig)
    def run(self, *args, stream, benchmark_run=False, **kwargs):
        if not benchmark_run:
            meta = getattr(self, "inductor_meta", None) or {}
            name = meta.get("kernel_name") or getattr(getattr(self, "fn", None), "__name__", "?")
            _record("inductor", str(name), args, kwargs)
        return orig(self, *args, stream=stream, benchmark_run=benchmark_run, **kwargs)

    run.__entail_calls__ = True
    C.run = run
    _WRAPPED[(C, "run")] = orig
    return 1


def install():
    import os

    n = install_triton() + install_inductor()
    if os.environ.get("ENTAIL_CALLS_DISPATCH", "1") != "0":
        n += install_dispatch()
    return n


def uninstall():
    for (holder, name), orig in list(_WRAPPED.items()):
        setattr(holder, name, orig)
    _WRAPPED.clear()


def report():
    """Every distinct call seen, in first-seen order, as plain data."""
    out = []
    for (kind, name, sig, extra), e in sorted(_CALLS.items(), key=lambda kv: kv[1]["first"]):
        tensors = [s for s in sig if s[0] == "T"]
        out.append({"kind": kind, "name": name, "n": e["n"], "captured": e["captured"],
                    "args": [list(s) for s in sig], "extra": list(extra),
                    "tensors": len(tensors), "with_fact": sum(1 for f in e["facts"] if f),
                    "facts": [list(f) if f else None for f in e["facts"]]})
    return {"calls": out, "stats": dict(_STATS)}


def reset():
    _CALLS.clear()
    _STATS.clear()
    _ORDER[0] = 0
