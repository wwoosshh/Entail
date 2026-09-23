"""Carry facts along with values through PyTorch operations, so a declaration made once still holds later.

Without this, a fact only exists where somebody wrote it down, and the first `view()` loses it. This is the
piece that makes boundary declarations usable: tag a value once, and the facts follow it.

The rules are deliberately small and conservative:
  carry        views, copies, dtype casts, slices: the value still means the same thing
  invalidate   transpose/permute: axis-dependent facts stop being true. They are NOT dropped; the output gets
               an Invalidated marker so the next boundary fails with a reason (facts.Invalidated)
  combine      elementwise binary ops: two inputs carrying the same kind of fact with different values is an
               error (absolute positions added to chunk-relative ones, a partial sum added to a replicated one)
  stop         everything else: facts do not propagate, so a later boundary reports "no declaration" rather
               than a fact that may no longer be true

Active only in debug mode; `off` and `load` leave PyTorch untouched (the mode is not even entered).
"""
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from . import core
from .core import RoleError
from .facts import Invalidated

CARRY = {
    "view", "_unsafe_view", "reshape", "_reshape_alias", "contiguous", "clone", "detach", "alias", "to",
    "_to_copy", "expand", "unsqueeze", "squeeze", "slice", "select", "narrow", "flatten", "split",
    "split_with_sizes", "chunk", "unbind", "index_select", "masked_fill", "nan_to_num",
}
INVALIDATE = {"transpose", "t", "permute", "movedim", "swapaxes", "flip", "roll"}
# In-place writes change what the value holds, so every fact about its contents stops being true. Measured on a
# real decode: a Valid(length=n) tagged on a KV cache stayed attached across eight steps while the cache was
# written in place, which is exactly the stale-fact bug this study is about (audits/PROPAGATE.md).
MUTATE = {"copy_", "index_copy_", "index_put_", "index_put", "scatter_", "scatter_add_", "masked_scatter_",
          "masked_fill_", "slice_scatter", "fill_", "zero_", "add_", "sub_", "mul_", "div_", "clamp_"}
BINARY = {"add", "sub", "mul", "div", "where", "maximum", "minimum", "copy_", "rsub", "pow"}
AXIS_DEPENDENT = {"Layout", "Reduction"}  # facts whose meaning is tied to which axis is which

STATS = {"ops": 0, "carried": 0, "invalidated": 0, "conflicts": 0}


def _tensors(obj, out=None):
    out = [] if out is None else out
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            _tensors(x, out)
    elif isinstance(obj, dict):
        for x in obj.values():
            _tensors(x, out)
    return out


def _name(func):
    try:
        return func.overloadpacket.__name__
    except AttributeError:
        return str(func).split(".")[-1]


class RolePropagation(TorchDispatchMode):
    """Use as a context manager around the code whose facts should follow the values."""

    def __init__(self, on_conflict="raise"):
        super().__init__()
        self.on_conflict = on_conflict  # "raise" or "record", for measuring without stopping a run

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if core.mode() != "debug":
            return out
        STATS["ops"] += 1
        name = _name(func)
        inputs = [t for t in _tensors(args) + _tensors(kwargs) if core.facts_of(t)]
        if not inputs:
            return out
        outputs = _tensors(out)
        if not outputs:
            return out

        if name in MUTATE:
            first = _tensors(args)[0] if _tensors(args) else None
            # `cache[:, :, i] = x` dispatches as a slice (a view) and then copy_ into that view, so the facts
            # to invalidate are on the base tensor, not on the view that was written through.
            target = first
            while target is not None:
                facts = core.facts_of(target)
                marks = [Invalidated(kind=k, why=f"aten.{name} wrote into the value")
                         for k in facts if k != "Invalidated"]
                if marks:
                    core._FACTS.pop(id(target), None)
                    core.tag(target, *marks)
                    STATS["invalidated"] += len(marks)
                target = getattr(target, "_base", None)
            return out

        if name in INVALIDATE:
            facts = dict(core.facts_of(inputs[0]))
            keep = {k: v for k, v in facts.items() if k not in AXIS_DEPENDENT and k != "Invalidated"}
            marks = [Invalidated(kind=k, why=f"aten.{name}") for k in facts if k in AXIS_DEPENDENT]
            for o in outputs:
                core.tag(o, *keep.values(), *marks)
            STATS["invalidated"] += len(marks)
            STATS["carried"] += len(keep)
            return out

        if name in BINARY and len(inputs) >= 2:
            merged = {}
            for t in inputs:
                for kind, fact in core.facts_of(t).items():
                    if kind in merged and merged[kind] != fact:
                        STATS["conflicts"] += 1
                        msg = (f"aten.{name}: the two values disagree about {kind}: "
                               f"{merged[kind]} and {fact}. Convert one of them and say so.")
                        if self.on_conflict == "raise":
                            raise RoleError(msg)
                    else:
                        merged[kind] = fact
            for o in outputs:
                core.tag(o, *merged.values())
            STATS["carried"] += len(merged)
            return out

        if name in CARRY:
            facts = core.facts_of(inputs[0])
            for o in outputs:
                core.tag(o, *facts.values())
            STATS["carried"] += len(facts)
        return out


def stats():
    return dict(STATS)


def reset_stats():
    for k in STATS:
        STATS[k] = 0
