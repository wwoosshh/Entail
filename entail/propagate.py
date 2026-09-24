"""propagate: carry facts along with values through PyTorch operations - the diagnosis site (LIBRARY_DESIGN.md 4.7,
12; ROADMAP M7.1).

Without this, a fact only exists where somebody wrote it down, and the first `view()` loses it. With it, a
declaration made once still holds at the next boundary; and when an operation between two boundaries makes a fact
untrue, the next boundary that needs the fact names that operation (Decision.lost_by), so the ledger says not only
where meaning was lost but by what.

The rules are deliberately small and conservative:
  carry        views, copies, dtype casts, slices: the value still means the same thing
  invalidate   transpose/permute: axis-dependent facts stop being true. They are NOT dropped; the output gets
               an Invalidated marker that names the operation (facts.Invalidated)
  mutate       in-place writes: every fact about what the value holds stops being true, on the value written
               into and on the value it is a view of
  combine      elementwise binary ops: two inputs carrying the same kind of fact with different values disagree
               (absolute positions added to chunk-relative ones, a partial sum added to a replicated one). That is a
               Decision at the operation ("op:add"), decided with the policy like any boundary: broken and
               reported, or refused where the policy stops - which debug mode, the only mode this runs in, does
  stop         everything else: facts do not propagate, so a later boundary reports "no declaration" rather
               than a fact that may no longer be true
An Invalidated marker is carried like a fact (never dropped by a later operation), and two markers merge.

The operations are seen where Python calls them (a torch function mode: methods, functions, indexing), not as the
aten operations they dispatch to. Measured on Qwen3-4B eager 64-token decode (testbed/results/m71/cost_probe.json,
medians of 5): seeing every aten operation cost 1.80x before any rule ran and 2.20x with these rules (where the 0.3.0
propagation sat); seeing the Python calls cost 1.26x, and 1.40x with these rules and the input ids tagged. The
operations that move a value around - the ones these rules are about - are Python calls either way. Active only in debug mode (principle 6:
never on the always-on sites); sites.debug_propagation() enters it only then. The rules add one dictionary lookup per
tensor argument, and nothing more while no argument carries a fact.
"""
from dataclasses import replace

import torch
from torch.overrides import TorchFunctionMode

from . import core
from .contracts import RULES, Contract, Decision, unrepaired
from .facts import Fact, Invalidated

CARRY = {
    "view", "view_as", "reshape", "reshape_as", "contiguous", "clone", "detach", "alias", "to", "type", "type_as",
    "float", "half", "bfloat16", "double", "expand", "expand_as", "unsqueeze", "squeeze", "flatten", "unflatten",
    "split", "split_with_sizes", "chunk", "unbind", "narrow", "select", "slice", "index_select", "masked_fill",
    "nan_to_num", "getitem", "_unsafe_view", "_reshape_alias", "_to_copy",
}
INVALIDATE = {"transpose", "t", "T", "mT", "permute", "movedim", "moveaxis", "swapaxes", "swapdims", "flip", "fliplr",
              "flipud", "roll"}
# In-place writes change what the value holds, so every fact about its contents stops being true. Measured on a
# real decode: a Valid(length=n) tagged on a KV cache stayed attached across eight steps while the cache was
# written in place, which is exactly the stale-fact bug this study is about (audits/PROPAGATE.md).
MUTATE = {"copy_", "index_copy_", "index_put_", "index_put", "index_add_", "index_fill_", "scatter_", "scatter_add_",
          "masked_scatter_", "masked_fill_", "slice_scatter", "fill_", "zero_", "add_", "sub_", "mul_", "div_",
          "clamp_", "setitem"}
BINARY = {"add", "sub", "mul", "div", "true_divide", "where", "maximum", "minimum", "rsub", "pow"}
AXIS_DEPENDENT = {"Layout", "Reduction"}  # facts whose meaning is tied to which axis is which
MARK = "Invalidated"

STATS = {"ops": 0, "carried": 0, "invalidated": 0, "conflicts": 0}
_NAMES = {}   # function -> its name, e.g. torch.Tensor.transpose -> "transpose", the getter of Tensor.T -> "T"
# Python's operator methods by the names of the operations they are: `a + b` is add, `t[i] = x` is setitem, `x += y`
# is add_ (in place)
_OPERATORS = {"iadd": "add_", "isub": "sub_", "imul": "mul_", "itruediv": "div_", "radd": "add", "rmul": "mul",
              "truediv": "div", "rtruediv": "div", "rpow": "pow"}


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


def _tagged(args, kwargs, table):
    """The tensor arguments that carry something: top-level ones, and those in a list or tuple (cat's inputs)."""
    out = []
    for a in (*args, *kwargs.values()):
        if isinstance(a, torch.Tensor):
            if id(a) in table:
                out.append(a)
        elif isinstance(a, (list, tuple)):
            for x in a:
                if isinstance(x, torch.Tensor) and id(x) in table:
                    out.append(x)
    return out


def _name(func):
    n = _NAMES.get(func)
    if n is None:
        n = getattr(func, "__name__", None) or str(func)
        if n == "__get__":   # a property: Tensor.T, Tensor.shape
            n = getattr(getattr(func, "__self__", None), "__name__", n)
        if n.startswith("__") and n.endswith("__"):
            n = _OPERATORS.get(n[2:-2], n[2:-2])
        n = n.split(".")[0]  # an aten overload called directly: "transpose.int"
        _NAMES[func] = n
    return n


def _merged(a, b):
    """Two Invalidated markers as one: every kind either names; the operation of the later one."""
    if a is None:
        return b
    if b is None:
        return a
    kinds = sorted(set(a.kind.split(",")) | set(b.kind.split(",")))
    return Invalidated(kind=",".join(kinds), why=b.why)


def _invalidate(have: dict, kinds, why):
    """What `have` holds, with `kinds` turned into one Invalidated marker (merged with a marker already there)."""
    kinds = sorted(k for k in kinds if k != MARK)
    if not kinds:
        return dict(have)
    out = {k: v for k, v in have.items() if k not in kinds and k != MARK}
    out[MARK] = _merged(have.get(MARK), Invalidated(kind=",".join(kinds), why=why))
    return out


def _put(value, facts: dict):
    """Replace what a value carries (in place when it already has an entry: its finalizer is registered once)."""
    have = core._FACTS.get(id(value))
    if have is not None:
        if have is not facts:
            have.clear()
            have.update(facts)
    elif facts:
        core.tag(value, *facts.values())


class RolePropagation(TorchFunctionMode):
    """Use as a context manager around the code whose facts should follow the values (sites.debug_propagation()).

    on_conflict: "policy" (the default) decides a disagreement with the policy in force, like any boundary - in
    debug mode that stops (refused); "record" reports it as broken and goes on, for measuring a run to its end.
    ("raise", the 0.3.0 spelling, means "policy".)"""

    def __init__(self, on_conflict="policy"):
        super().__init__()
        if on_conflict not in ("policy", "record", "raise"):
            raise ValueError(f"on_conflict: 'policy' or 'record', got {on_conflict!r}")
        self.on_conflict = on_conflict

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        table = core._FACTS
        if not table or core._MODE != "debug":
            return out
        STATS["ops"] += 1
        inputs = _tagged(args, kwargs, table)
        if not inputs:
            return out
        name = _name(func)

        if name in MUTATE:   # `t[i] = x` (setitem) returns nothing: handled before looking at what came out
            target = args[0] if args and isinstance(args[0], torch.Tensor) else None
            # a write through a view (`cache[:, :, i].copy_(x)`) changes the value it is a view of as well
            while target is not None:
                have = table.get(id(target))
                if have:
                    old = [k for k in have if k != MARK]
                    if old:
                        _put(target, _invalidate(have, old, f"{name}, which wrote into the value"))
                        STATS["invalidated"] += len(old)
                target = target._base
            return out

        outputs = _tensors(out)
        if not outputs:
            return out

        if name in INVALIDATE:
            have = table.get(id(inputs[0]), {})
            axis = [k for k in have if k in AXIS_DEPENDENT]
            facts = _invalidate(have, axis, name)
            for o in outputs:
                _put(o, facts)
            STATS["invalidated"] += len(axis)
            STATS["carried"] += len(facts) - (MARK in facts)
            return out

        if name in BINARY and len(inputs) >= 2:
            merged, mark = {}, None   # kind -> the envelope carried on; compared by value
            for t in inputs:
                for kind, fact in table.get(id(t), {}).items():
                    if kind == MARK:
                        mark = _merged(mark, fact)
                    elif kind in merged and isinstance(fact, Fact) and merged[kind].value != fact.value:
                        STATS["conflicts"] += 1
                        self._disagree(name, kind, merged[kind], fact)
                    else:
                        merged.setdefault(kind, fact)
            if mark is not None:
                merged[MARK] = mark
            for o in outputs:
                _put(o, merged)
            STATS["carried"] += len(merged) - (mark is not None)
            return out

        if name in CARRY or name in BINARY:
            facts = dict(table.get(id(inputs[0]), {}))   # envelopes: what is carried keeps its source
            for o in outputs:
                _put(o, facts)
            STATS["carried"] += len(facts) - (MARK in facts)
        return out

    def _disagree(self, name, kind, first, other):
        """Two inputs of one operation carry different values of the same kind of fact: a Decision at the operation,
        recorded once per distinct outcome (boundaries._record) and stopping where the policy stops."""
        from . import policies
        from .boundaries import _record

        policy = policies.current()
        if self.on_conflict == "record":
            policy = replace(policy, mode="load", on_broken="report", overrides=())
        verdict, blocking = unrepaired(policy, kind)
        d = Decision(Contract(f"op:{name}", name, (kind,), (kind,)), kind, verdict, RULES["disagree"],
                     declared=first, chosen=other, blocking=blocking,
                     note="two inputs of one operation carry different values of it; convert one and say so")
        _record([d], f"op:{name}")


def stats():
    return dict(STATS)


def reset_stats():
    for k in STATS:
        STATS[k] = 0
