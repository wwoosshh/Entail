"""lifetime: memory nothing wrote (ROADMAP M19 L7; ENTAIL=types).

The rule of L7 - the memory a reader reads holds the value its maker wrote there - has three parts: the maker wrote
it, nothing else wrote over it since, and no other live value shares it (kernel_check holds the last two for the
values it has meanings for). This module holds the first for the values a model is made of, its parameters.

  load_window()   while a model is made and its weights are loaded and processed, every allocation torch makes
                  without a value (torch.empty and its kin) is filled with a mark no written value carries: a NaN
                  with a payload of its own for the floating types (a weight is never NaN), a byte pattern for the
                  others (counted only where a whole 64-byte run holds it, which loaded data does not). A write by
                  any path - a torch operation, a C++ or Triton kernel, a copy from disk - replaces the mark.
  loaded(model)   when the outermost window closes: a parameter that still holds the mark has elements nothing
                  wrote - no checkpoint tensor, no initializer and no processing gave them a value. Recorded at
                  once; reported (broken) when they are read: when the module holding the parameter runs (a forward
                  pre-hook, inert inside compiled code), when a Triton launch is handed the tensor, or when a
                  compiled graph takes it as an input.
  assembly()      while an engine assembles its model (its own load_model: the model and any draft model made,
                  loaded and wired together; adapters/loaders.py), such a read is held. When the assembly ends it is
                  reported if the model still holds the parameter, and withdrawn if the engine replaced it: the read
                  was a trial call whose result reached nothing the model keeps (vLLM 0.22's drafter probes its
                  embedding, then shares the target's).
Nothing is repaired: there is no value to repair to. The mark replaces whatever the memory happened to hold, so a
program that reads memory nothing wrote reads the mark instead of leftovers.
Cost: one fill per allocation made in the window, one pass over the parameters when it closes (both at memory
speed), and a Python call per torch function called while the window is open.
"""
import contextlib
import os
import sys
import threading
import time
import weakref

_LOCK = threading.RLock()
_STATE = {"depth": 0, "mode": None, "marked": 0, "marked_bytes": 0, "opened": None, "assembling": 0}
_PENDING = []          # reads held while an engine assembles its model: settled when the assembly ends
_STATS = {}
_SPECS = {}
CHUNK = 1 << 26        # elements compared at once when a parameter is read for the mark (bounds the temporary)
RUN = 64               # bytes: a non-floating tensor counts as unwritten only where a whole aligned run holds the mark


def _count(k, by=1):
    _STATS[k] = _STATS.get(k, 0) + by


def stats():
    return dict(_STATS)


def _signed(v, bits):
    return v - (1 << bits) if v >= (1 << (bits - 1)) else v


def _spec(dtype):
    """(the integer dtype the mark is written through, the mark, per element?) for a dtype; None: not marked."""
    if dtype in _SPECS:
        return _SPECS[dtype]
    import torch

    spec = None
    floats = {torch.float32: (torch.int32, 0x7FC5A5A5), torch.bfloat16: (torch.int16, 0x7FE5),
              torch.float16: (torch.int16, 0x7E5A), torch.float64: (torch.int64, 0x7FF8A5A5A5A5A5A5)}
    for name, value in (("float8_e4m3fn", 0x7F), ("float8_e5m2", 0x7F), ("float8_e4m3fnuz", 0x80),
                        ("float8_e5m2fnuz", 0x80), ("float8_e8m0fnu", 0xFF)):
        dt = getattr(torch, name, None)
        if dt is not None:
            floats[dt] = (torch.uint8, value)
    if dtype in floats:
        view, value = floats[dtype]
        spec = (view, value, True)
    elif not dtype.is_floating_point and not dtype.is_complex and dtype != torch.bool:
        try:
            size = dtype.itemsize
        except Exception:  # noqa: BLE001 - a dtype without a plain size (a quantized one) is not marked
            size = 0
        view = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}.get(size)
        if view is not None:
            pattern = int.from_bytes(b"\xa5" * size, "little")
            spec = (view, pattern if view == torch.uint8 else _signed(pattern, 8 * size), False)
    elif dtype.is_floating_point and dtype.itemsize == 1:          # a packed 4-bit pair: bytes
        spec = (torch.uint8, 0xA5, False)
    _SPECS[dtype] = spec
    return spec


def _plain(t):
    import torch

    return isinstance(t, torch.Tensor) and (type(t) is torch.Tensor or type(t) is torch.nn.Parameter)


def mark(t):
    """Fill a tensor torch allocated without a value with the mark of its dtype."""
    if not _plain(t) or t.device.type == "meta" or t.numel() == 0:
        return
    spec = _spec(t.dtype)
    if spec is None:
        return
    view, value, _each = spec
    try:
        t.detach().view(view).fill_(value)
        _STATE["marked"] += 1
        _STATE["marked_bytes"] += t.numel() * t.element_size()
    except Exception:  # noqa: BLE001 - a tensor that cannot be marked is not checked
        _count("mark_failed")


def unwritten(t):
    """How many elements of `t` still hold the mark (were written by nothing since they were allocated)."""
    if not _plain(t) or t.device.type == "meta" or t.numel() == 0:
        return 0
    spec = _spec(t.dtype)
    if spec is None:
        return 0
    view, value, each = spec
    x = t.detach().view(view)
    total = 0
    if each:
        if x.is_contiguous() or x.dim() == 0:
            flat = x.reshape(-1)
            for i in range(0, flat.numel(), CHUNK):
                total += int((flat[i:i + CHUNK] == value).sum())
        else:                                        # a strided parameter: its rows, a bounded number at a time
            step = max(1, CHUNK // max(1, x[0].numel()))
            for i in range(0, x.shape[0], step):
                total += int((x[i:i + step] == value).sum())
        return total
    if not x.is_contiguous():
        return 0
    k = max(1, RUN // x.element_size())
    flat = x.reshape(-1)
    n = flat.numel() // k * k
    step = max(k, CHUNK // k * k)
    for i in range(0, n, step):
        seg = flat[i:min(i + step, n)]
        total += int((seg.view(-1, k) == value).all(dim=1).sum()) * k
    return total


def _empties():
    import torch

    return {f for f in (torch.empty, torch.empty_like, torch.empty_strided, torch.Tensor.new_empty,
                        getattr(torch.Tensor, "new_empty_strided", None)) if f is not None}


def _mode():
    import torch
    from torch.overrides import TorchFunctionMode

    empties = _empties()

    class _Mark(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if func in empties:
                mark(out)
            return out

    return _Mark()


@contextlib.contextmanager
def load_window():
    """While open (on the thread that opened it, and nested windows count as one), torch's allocations without a
    value are marked."""
    with _LOCK:
        _STATE["depth"] += 1
        outer = _STATE["depth"] == 1
    mode = None
    if outer:
        _STATE["opened"] = time.perf_counter()
        _STATE["marked"] = _STATE["marked_bytes"] = 0
        try:
            mode = _mode()
            mode.__enter__()
        except Exception:  # noqa: BLE001 - no window: nothing is marked, nothing will be reported
            mode = None
            _count("window_failed")
    try:
        yield
    finally:
        if mode is not None:
            try:
                mode.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                _count("window_close_failed")
        with _LOCK:
            _STATE["depth"] -= 1


def loaded(model):
    """When the outermost window has closed: the model's parameters, read for the mark. Each parameter with elements
    nothing wrote gets that on its fact, and the module holding it a pre-hook that reports it when the module runs.
    Returns [(name, elements unwritten, elements)]."""
    if _STATE["depth"] > 0 or model is None or not hasattr(model, "named_modules"):
        return []
    from . import kernel_check

    t0 = time.perf_counter()
    owners = {}
    for mname, module in model.named_modules():
        for pname, p in list(getattr(module, "_parameters", {}).items()):
            if p is not None:
                owners.setdefault(id(p), (p, []))[1].append((mname, module, pname))
    found = []
    readers = {}
    for p, where in owners.values():
        try:
            n = unwritten(p)
        except Exception:  # noqa: BLE001 - a parameter that cannot be read for the mark is not claimed
            _count("read_failed")
            continue
        if not n:
            continue
        mname, _module, pname = where[0]
        found.append((f"{mname}.{pname}" if mname else pname, n, p.numel()))
        try:
            kernel_check.set_unwritten(p, n)
        except Exception:  # noqa: BLE001
            _count("fact_failed")
        for mname, module, pname in where:
            readers.setdefault(id(module), (module, mname, []))[2].append((pname, n, p.numel()))
    root = weakref.ref(model)
    for module, mname, items in readers.values():
        _hook(module, mname, items, root)
    _count("windows")
    _count("parameters", len(owners))
    _count("unwritten_parameters", len(found))
    opened = _STATE.get("opened")
    kernel_check._write({"kind": "types_life_load", "parameters": len(owners), "unwritten": len(found),
                         "examples": [{"name": n, "elements": k, "of": m} for n, k, m in found[:40]],
                         "marked": _STATE["marked"], "marked_bytes": _STATE["marked_bytes"],
                         "window_seconds": None if opened is None else round(t0 - opened, 3),
                         "read_seconds": round(time.perf_counter() - t0, 3)})
    if found and "load" not in os.environ.get("ENTAIL_QUIET", "").replace(" ", "").split(","):
        sys.stderr.write(f"entail: {len(found)} of {len(owners)} parameters hold elements nothing wrote while the "
                         f"model was loaded (first: {found[0][0]}, {found[0][1]} of {found[0][2]}); each is reported "
                         f"as broken when it is read\n")
    return found


def _report(mname, pname, n, total, note=""):
    from . import kernel_check, kernel_types

    why = (f"{mname or 'the model'} runs with its parameter {pname}, but {n} of its {total} elements were "
           f"never written while the model was loaded (no checkpoint tensor, initializer or processing gave "
           f"them a value){note}")
    _count("unwritten_read")
    kernel_check._write({"kind": "types_life", "reader": mname, "parameter": pname, "verdict": "violation",
                         "why": why})
    kernel_check._broken(mname or "model", kernel_types.Verdict("violation", why), where=mname or "model")


@contextlib.contextmanager
def assembly():
    """While open (nested ones count as one), reads of parameters nothing wrote are held; when the outermost closes,
    each is reported if its model still holds the parameter, withdrawn if not."""
    with _LOCK:
        _STATE["assembling"] += 1
    try:
        yield
    finally:
        with _LOCK:
            _STATE["assembling"] -= 1
            last = _STATE["assembling"] == 0
        if last:
            _settle()


def _settle():
    from . import kernel_check

    with _LOCK:
        held = list(_PENDING)
        _PENDING.clear()
    for p_ref, root_ref, mname, pname, n, total in held:
        p, root = p_ref(), root_ref()
        if p is not None and root is not None and any(q is p for q in root.parameters()):
            _report(mname, pname, n, total, "; it was read while the engine assembled the model, which still "
                                            "holds it")
            continue
        _count("unwritten_read_withdrawn")
        kernel_check._write({"kind": "types_life", "reader": mname, "parameter": pname, "verdict": "withdrawn",
                             "why": f"{mname or 'the model'} read its parameter {pname} ({n} of {total} elements "
                                    f"never written) while the engine assembled the model, and the engine then "
                                    f"replaced it: the model does not hold it, so the read reached nothing it keeps"})


def _hook(module, mname, items, root=None):
    import torch

    compiling = torch.compiler.is_compiling
    state = {}

    def pre(mod, args):
        if compiling():
            return None                  # inside a compiled graph the graph's own check reads the inputs' facts
        h = state.pop("handle", None)
        if h is not None:
            h.remove()
        for pname, n, total in items:
            if _STATE["assembling"] and root is not None:
                p = getattr(mod, "_parameters", {}).get(pname)
                if p is not None:
                    with _LOCK:
                        _PENDING.append((weakref.ref(p), root, mname, pname, n, total))
                    _count("unwritten_read_held")
                    continue
            _report(mname, pname, n, total)
        return None

    state["handle"] = module.register_forward_pre_hook(pre)
