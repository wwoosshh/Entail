"""kernel_reference_contract: the kernel an op dispatches to against the op's own native definition, run on the
same input (LIBRARY_DESIGN.md 11, M18; ROADMAP M18.2).

An engine's custom op carries its meaning twice: a native forward in plain PyTorch, which is the definition, and a
kernel it dispatches to for speed (vLLM's CustomOp: forward_native and forward_cuda). The kernel is right when it
computes what the definition computes. A kernel that pairs rotary dimensions the other way (vllm#42016: the Triton
MRoPE kernel paired split-wise for a model that pairs interleaved), reads a scale layout the caller did not mean
(vllm#58532) or a tensor in another order (vllm#38643) is off by the size of the values themselves, and nothing in
the engine compares the two paths on the model at hand. This contract does: on the first call of each op class
(per input pattern) in a process, the kernel and the definition are run on the same small slice of the real input
(ROWS rows of the token dimension) and compared. The rule is here, once:

  kernel_reference_mismatch   max |kernel - definition| exceeds FACTOR times the definition's own precision noise
                              plus ATOL_ULPS units in the last place of the output dtype at the output's largest
                              magnitude

The definition is run twice: in the input's dtype (its own rounding noise: the floor) and in float32 (the
reference). FACTOR and ATOL_ULPS are provisional until the M18.2 measurement sets them from the distribution on
healthy runs, after which they are fixed; every decision records diff, floor and scale so the setting can be
checked against them. A definition that cannot be run on the input (NotImplementedError, a dtype the native path
refuses) is unknown once per op class. Nothing here touches the engine's tensors: the slices are clones, and the
kernel's own output is always what the engine gets, whatever happens in the comparison.
"""
from typing import Any, List, Optional, Tuple

from . import tally as _tally

RULE_NAMES = ("kernel_reference_mismatch",)
ROWS = 64            # rows of the token dimension the comparison runs on
FACTOR = 8.0         # provisional (M18.2 measurement): allowed multiple of the definition's own precision noise
ATOL_ULPS = 4.0      # provisional: units in the last place of the output dtype, at the output's largest magnitude
ULPS = {"bfloat16": 2.0 ** -7, "float16": 2.0 ** -10, "float32": 2.0 ** -23, "float64": 2.0 ** -52}


def _is_tensor(x) -> bool:
    return hasattr(x, "shape") and hasattr(x, "dtype") and hasattr(x, "dim")


def tensors_of(obj) -> List[Any]:
    """The floating-point tensors in an output or an argument list, in order (None and integers skipped)."""
    if _is_tensor(obj):
        return [obj] if obj.is_floating_point() else []
    if isinstance(obj, (tuple, list)):
        return [t for o in obj for t in tensors_of(o)]
    if isinstance(obj, dict):
        return [t for o in obj.values() for t in tensors_of(o)]
    return []


def all_tensors(obj) -> List[Any]:
    if _is_tensor(obj):
        return [obj]
    if isinstance(obj, (tuple, list)):
        return [t for o in obj for t in all_tensors(o)]
    if isinstance(obj, dict):
        return [t for o in obj.values() for t in all_tensors(o)]
    return []


def rows_of(args, kwargs) -> Optional[int]:
    """The token dimension: the first dimension of the largest tensor argument (the hidden states or query; MRoPE's
    positions are [3, n] and small), when every tensor argument shares it - along its first dimension, or along
    the last of a 2-D tensor. None when there is no tensor, or when an argument does not share it (an attention
    op with cu_seqlens metadata: cutting rows would not give both paths the same problem)."""
    ts = [t for t in all_tensors(list(args)) + all_tensors(kwargs) if t.dim() >= 1]
    if not ts:
        return None
    n = int(max(ts, key=lambda t: t.numel()).shape[0])
    for t in ts:
        if int(t.shape[0]) != n and not (t.dim() == 2 and int(t.shape[-1]) == n):
            return None
    return n


def sliced(obj, n: int, rows: int, dtype=None):
    """A copy of an argument with the token dimension cut to `rows`: tensors whose first dimension is n are cut
    along it, a 2-D tensor whose last dimension is n (MRoPE positions, [3, n]) along that; everything else is
    copied whole. Floating tensors are cast to `dtype` when given. Containers are rebuilt; other values pass."""
    if _is_tensor(obj):
        t = obj
        if t.dim() >= 1 and int(t.shape[0]) == n:
            t = t[:rows]
        elif t.dim() == 2 and int(t.shape[-1]) == n:
            t = t[:, :rows]
        t = t.detach().clone()
        if dtype is not None and t.is_floating_point():
            t = t.to(dtype)
        return t
    if isinstance(obj, tuple):
        return tuple(sliced(o, n, rows, dtype) for o in obj)
    if isinstance(obj, list):
        return [sliced(o, n, rows, dtype) for o in obj]
    if isinstance(obj, dict):
        return {k: sliced(o, n, rows, dtype) for k, o in obj.items()}
    return obj


def compare(kernel_out, reference_out, native_out=None) -> Tuple[float, Optional[float], float, int]:
    """(max |kernel - reference|, max |native - reference| or None, max |reference|, values compared): the
    floating tensors of each output paired by position, everything in float32. Raises ValueError when the outputs
    do not pair up."""
    ks, rs = tensors_of(kernel_out), tensors_of(reference_out)
    ns = tensors_of(native_out) if native_out is not None else None
    if len(ks) != len(rs) or (ns is not None and len(ns) != len(rs)) or not rs:
        raise ValueError(f"the kernel gives {len(ks)} floating tensors, the definition {len(rs)}")
    diff = floor = scale = 0.0
    elements = 0
    for i, (k, r) in enumerate(zip(ks, rs)):
        if tuple(k.shape) != tuple(r.shape):
            raise ValueError(f"output {i}: the kernel gives shape {tuple(k.shape)}, the definition {tuple(r.shape)}")
        rf = r.detach().float()
        diff = max(diff, float((k.detach().float() - rf).abs().max().item()) if rf.numel() else 0.0)
        scale = max(scale, float(rf.abs().max().item()) if rf.numel() else 0.0)
        if ns is not None:
            floor = max(floor, float((ns[i].detach().float() - rf).abs().max().item()) if rf.numel() else 0.0)
        elements += int(rf.numel())
    return diff, (floor if ns is not None else None), scale, elements


def vacuous(diff: float, floor: Optional[float], scale: float, reduced_precision: bool = True) -> Optional[str]:
    """Why this input decides nothing: an all-zero definition output (a profile run's dummy input), or - when the
    input is of reduced precision, so the definition normally rounds - a definition that is exact on it (rotary at
    position 0 is the identity: floor 0) with which the kernel agrees. None when the comparison is decisive."""
    if scale == 0.0:
        return "the definition's output is all zeros on this input (a dummy input)"
    if reduced_precision and floor == 0.0 and diff == 0.0:
        return "the definition is exact on this input (an identity, as rotary at position 0) and the kernel agrees"
    return None


def tolerance(dtype_name: str, floor: Optional[float], scale: float) -> float:
    return FACTOR * (floor or 0.0) + ATOL_ULPS * ULPS.get(str(dtype_name).replace("torch.", ""), 2.0 ** -7) * scale


def check(boundary: str, consumer: str, op: str, dtype_name: str, diff: float, floor: Optional[float], scale: float,
          elements: int, where: str, policy=None, record: bool = True, owner=None) -> list:
    """Decide the kernel's output against the definition's. Returns the decisions; with `record` they are also
    recorded through load.enforce (the run goes on, or stops where the policy says)."""
    from . import load, policies
    from .contracts import RULES, Contract, Decision, Verdict, unrepaired
    from .facts import Certainty, Fact, KernelReference, Source

    policy = policy or policies.current()
    contract = Contract(boundary, consumer, ("KernelReference",))
    tol = tolerance(dtype_name, floor, scale)
    declared = Fact("KernelReference", KernelReference(op=op, max_abs_diff=0.0, floor=floor, scale=scale),
                    Source("engine", f"{op}.forward_native, the op's own definition"), Certainty.VERIFIED)
    held = Fact("KernelReference", KernelReference(op=op, max_abs_diff=diff, floor=floor, scale=scale),
                Source("engine", where), Certainty.VERIFIED)
    numbers = (f"max |kernel - definition| {diff:.3g} over {elements} values"
               + (f"; the definition's own noise {floor:.3g}" if floor is not None else
                  "; the definition could not be run in float32 (compared in the input dtype)")
               + f"; scale {scale:.3g}; allowed {tol:.3g}")
    if diff > tol:
        verdict, blocking = unrepaired(policy, "KernelReference")
        d = Decision(contract, "KernelReference", verdict, RULES["kernel_reference_mismatch"], declared=declared,
                     chosen=held, blocking=blocking, note=f"{where}: {numbers}")
    else:
        d = Decision(contract, "KernelReference", Verdict.PASS, RULES["match"], declared=declared, chosen=held,
                     note=f"{where}: {numbers}")
    decisions = [d]
    if record:
        _tally.counts(boundary)["checks"] += 1
        if d.verdict is Verdict.PASS:
            _tally.passed(boundary, ["kernel_reference"])
        if d.blocking:
            _tally.refused(boundary)
        elif d.verdict is Verdict.BROKEN:
            _tally.broken(boundary)
        _tally.tick(boundary)
        load.enforce(decisions, once_for=owner)
    return decisions


def stats(boundary: str) -> dict:
    return _tally.stats(boundary)


def reset(boundary: str) -> None:
    _tally.reset(boundary)
