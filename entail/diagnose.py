"""diagnose: the diagnosis site - where meaning broke, and inside which layer when none broke (LIBRARY_DESIGN.md 12;
ROADMAP M7.1). Secondary to keeping meaning intact: it is what the ledger makes possible, not a separate bug hunt.

  locate(output_wrong)   where the fault lies, from this process's ledger (record.locate): the boundary where meaning
                         broke; or, when every checked boundary held and the output is wrong, inside a layer
  propagating()          operation-level propagation (propagate.py), entered only in debug mode: an operation that
                         makes a fact untrue between two boundaries is named at the next boundary that needs the fact
  watch(owner, name, reference)
                         a layer run apart against a reference on the same inputs. When every checked boundary held,
                         the layer whose output its reference does not reproduce is where the fault lies; the layers
                         after it compute correctly on what they were given, so they agree with theirs

`entail locate` reads the record files instead (engines check in child processes this one cannot see into).
Watching compares the first `calls` calls of a function (default 1), and only in debug mode. The references are the
caller's - an engine's plain path, a slower exact computation - because entail does not know what a layer should
compute, only whether two computations on the same inputs agree. A comparison is recorded like a decision: in the
ledger (Ledger.layers), and as one JSON line in the record.
"""
import contextlib
import functools
import itertools
import os
from typing import Callable, Optional

from . import core, record

# The largest difference, relative to the reference's largest magnitude, that still counts as agreeing: the rounding
# two correct computations may differ by in each dtype. M7.3 checks these against healthy layers of a real model.
TOLERANCE = {"float64": 1e-9, "float32": 1e-4, "float16": 1e-2, "bfloat16": 3e-2}
_ORDER = itertools.count()


def locate(output_wrong: Optional[bool] = None, say: bool = False) -> record.Localization:
    """Where the fault lies, from this process's ledger, the checks counted rather than recorded, and the layers
    compared. With say=True the lines are also said (printed, and kept in entail_logs)."""
    from . import boundaries, load, tally

    passes = {}
    for (where, _), n in boundaries.PASSES.items():
        passes[f"boundary:{where}"] = passes.get(f"boundary:{where}", 0) + n
    for (b, _), n in tally.PASSES.items():
        passes[b] = passes.get(b, 0) + n
    skipped = [b for b, s in tally.STATS.items() if not s["checks"] and (s["skipped"] or s["deferred"])]
    found = load.LEDGER.locate(output_wrong, passes, skipped)
    if say:
        for text in found.lines():
            record.say(text)
        record.write_json({"pid": os.getpid(), "located": found.to_json()})
    return found


def propagating(on_conflict: str = "policy"):
    """Operation-level propagation in debug mode; in any other mode a context that does nothing, so the always-on
    sites never pay for it (principle 6)."""
    if core.mode() != "debug":
        return contextlib.nullcontext()
    from .propagate import RolePropagation

    return RolePropagation(on_conflict=on_conflict)


def _tensors(obj, out):
    import torch

    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            _tensors(x, out)
    elif isinstance(obj, dict):
        for x in obj.values():
            _tensors(x, out)
    return out


def _clone(obj):
    """The inputs as they were before the call: a layer may write into what it is given."""
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if isinstance(obj, list):
        return [_clone(x) for x in obj]
    if isinstance(obj, tuple) and not hasattr(obj, "_fields"):
        return tuple(_clone(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _clone(v) for k, v in obj.items()}
    return obj


def _first(out):
    found = _tensors(out, [])
    return found[0] if found else None


def _difference(out, ref):
    """(largest absolute difference, the same relative to the reference's largest magnitude, note)."""
    import torch

    if out is None or ref is None:
        return None, None, "one of them returned no tensor"
    if tuple(out.shape) != tuple(ref.shape):
        return None, None, f"the shapes differ: {tuple(out.shape)} and {tuple(ref.shape)}"
    a, b = out.detach().float(), ref.detach().float()
    d = (a - b).abs()
    d = torch.where(torch.isfinite(d), d, torch.full_like(d, float("inf")))   # a NaN on one side is a difference
    largest = float(d.max()) if d.numel() else 0.0
    scale = float(b.abs().nan_to_num(0.0, 0.0, 0.0).max()) if b.numel() else 0.0
    return largest, largest / scale if scale > 0 else largest, ""


def _compared(label, reference, out, ref, tol, pick, call):
    """Record one comparison: in the ledger, in the record, and said when the layer and its reference differ."""
    from . import load

    pick = pick or _first
    try:
        mine, theirs = pick(out), pick(ref)
        largest, rel, note = _difference(mine, theirs)
        limit = tol if tol is not None else TOLERANCE.get(str(getattr(theirs, "dtype", "")).replace("torch.", ""),
                                                          1e-4)
    except Exception as e:  # noqa: BLE001 - a comparison that cannot be made is said, never breaks the run
        largest, rel, note, limit = None, None, f"could not compare: {type(e).__name__}: {e}", tol
    agrees = rel is not None and rel <= limit
    entry = {"layer": label, "reference": getattr(reference, "__qualname__", repr(reference)), "call": call,
             "order": next(_ORDER), "max_abs": largest, "max_rel": rel, "tol": limit, "agrees": agrees, "note": note}
    load.LEDGER.layers.append(entry)
    record.write_json({"pid": os.getpid(), **entry})
    if not agrees or os.environ.get("ENTAIL_VERBOSE"):
        record.say(f"[entail] compared: {record._layer_text(entry)}")
    return entry


def compare(label: str, fn: Callable, reference: Callable, *args, tol: Optional[float] = None,
            pick: Optional[Callable] = None, **kwargs):
    """Call fn, and reference on the same inputs (as they were before fn ran); record whether they agree. Returns
    fn's result. It compares whenever it is called: calling it is the request."""
    import torch

    before = (_clone(args), _clone(kwargs))
    out = fn(*args, **kwargs)
    with torch.no_grad():
        ref = reference(*before[0], **before[1])
    _compared(label, reference, out, ref, tol, pick, 1)
    return out


@contextlib.contextmanager
def watch(owner, name: str, reference: Callable, label: Optional[str] = None, calls: int = 1,
          tol: Optional[float] = None, pick: Optional[Callable] = None):
    """While the block runs, the function `owner.name` (or `owner[name]`, for a registry such as transformers'
    attention functions) is compared with `reference` on the same inputs, for its first `calls` calls, in debug mode.
    A method is watched on its class: `watch(Qwen3MLP, "forward", reference)` and the reference takes `self` too.
    `pick` chooses the tensor to compare from each output (default: the first one it holds); `tol` the largest
    relative difference that agrees (default: TOLERANCE for the reference's dtype)."""
    import torch

    items = not hasattr(owner, name) and hasattr(owner, "__getitem__")
    fn = owner[name] if items else getattr(owner, name)
    label = label or f"{getattr(owner, '__name__', type(owner).__name__)}.{name}"
    seen = [0]

    @functools.wraps(fn)
    def watched(*args, **kwargs):
        if core._MODE != "debug" or seen[0] >= calls:
            return fn(*args, **kwargs)
        seen[0] += 1
        before = (_clone(args), _clone(kwargs))
        out = fn(*args, **kwargs)
        try:
            with torch.no_grad():
                ref = reference(*before[0], **before[1])
        except Exception as e:  # noqa: BLE001 - the reference failing is said, and the layer's own result stands
            ref = None
            record.say(f"[entail] compared: the reference for {label} failed: {type(e).__name__}: {e}")
        _compared(label, reference, out, ref, tol, pick, seen[0])
        return out

    if items:
        owner[name] = watched
    else:
        setattr(owner, name, watched)
    try:
        yield watched
    finally:
        if items:
            owner[name] = fn
        else:
            setattr(owner, name, fn)
