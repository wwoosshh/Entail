"""kernel_reference_contract: the kernel an op dispatches to against the op's own native definition, run on the
same input (LIBRARY_DESIGN.md 11, M18; ROADMAP M18.2).

An engine's custom op carries its meaning twice: a native forward in plain PyTorch, which is the definition, and a
kernel it dispatches to for speed (vLLM's CustomOp: forward_native and forward_cuda). The kernel is right when it
computes what the definition computes. A kernel that pairs rotary dimensions the other way (vllm#42016: the Triton
MRoPE kernel paired split-wise for a model that pairs interleaved), reads a scale layout the caller did not mean
(vllm#58532) or a tensor in another order (vllm#38643) is off by the size of the values themselves, and nothing in
the engine compares the two paths on the model at hand. This contract does: on the first real call of each op
(per class, configuration and input pattern) in a process, the kernel and the definition are run on the same small
slice of the real input (ROWS rows of the token dimension, cut before the kernel touched it) and compared value by
value. The rule is here, once:

  kernel_reference_mismatch   a value of the kernel's output is non-finite where the definition's is not (or the
                              other way round), or differs from the definition's by more than
                                FACTOR x (the definition's own rounding noise at that value)
                                + ATOL_ULPS x (one unit in the last place of the output dtype at that value)

The definition is run twice: in the input's dtype (its own rounding noise, value by value: the floor) and in
float32 (the reference). A single rounding sample can be exactly right by chance, so each value's floor is backed
by the tensor's typical noise (its 90th percentile). Every output tensor is decided in its own dtype and scale, so
a small second output (a scale) is held to its own size, not the largest output's. FACTOR and ATOL_ULPS are
headroom, not values derived from a distribution: the M18.2 runs (8 models, 32 decisions) put the kernels'
largest error at exactly the definition's largest rounding step (ratio 1.00), which any FACTOR >= 1 admits; the
elementwise ratio to the allowance is recorded on every decision so the M18.6 measurement can fix them from data.
A definition that cannot be run on the input (NotImplementedError, a dtype the native path refuses) is unknown
once per op class. Integer outputs are not compared. Nothing here touches the engine's tensors: the slices are
clones, and the kernel's own output on the real input is always what the engine gets, whatever happens in the
comparison.
"""
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from . import tally as _tally

RULE_NAMES = ("kernel_reference_mismatch",)
ROWS = 64            # rows of the token dimension the comparison runs on
FACTOR = 8.0         # headroom over the definition's own rounding noise at each value (see the docstring)
ATOL_ULPS = 4.0      # headroom in units in the last place of the output dtype at each value
ULPS = {"bfloat16": 2.0 ** -7, "float16": 2.0 ** -10, "float32": 2.0 ** -23, "float64": 2.0 ** -52,
        "float8_e4m3fn": 2.0 ** -3, "float8_e4m3fnuz": 2.0 ** -3, "float8_e5m2": 2.0 ** -2, "float8_e5m2fnuz": 2.0 ** -2}
REDUCED = ("torch.bfloat16", "torch.float16")   # the dtypes the reference run casts to float32; fp8 and integer
#                                                 tensors (quantized inputs, scales) are never cast
BUDGET = 64 << 20    # bytes the clones may hold when the arguments cannot be cut (their first dimension is a batch)


def _is_tensor(x) -> bool:
    return hasattr(x, "shape") and hasattr(x, "dtype") and hasattr(x, "dim")


def _dtype(t) -> str:
    return str(t.dtype).replace("torch.", "")


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


def rows_of(args, kwargs, hint: Optional[int] = None) -> Optional[int]:
    """The token dimension: the engine's own token count (`hint`, vLLM's batch descriptor) when a tensor argument
    has it as its first dimension, else the first dimension of the largest tensor argument (the hidden states or
    query; MRoPE's positions are [3, n] and small) - and every tensor argument must share it, along its first
    dimension or along the last of a 2-D tensor. None when there is no tensor, or when an argument does not share
    it (an attention op with cu_seqlens metadata: cutting rows would not give both paths the same problem). A
    first dimension that is a batch, not the token count, is not told apart here; the byte budget (BUDGET) then
    refuses a comparison that would clone the whole input."""
    ts = [t for t in all_tensors(list(args)) + all_tensors(kwargs) if t.dim() >= 1]
    if not ts:
        return None
    n = int(max(ts, key=lambda t: t.numel()).shape[0])
    if hint is not None and any(int(t.shape[0]) == int(hint) for t in ts):
        n = int(hint)
    for t in ts:
        if int(t.shape[0]) != n and not (t.dim() == 2 and int(t.shape[-1]) == n):
            return None
    return n


def kept(t):
    """A copy of a tensor with its strides kept (M19 L3). clone() makes a view with gaps - a row of a fused
    projection, a column of an interleaved one - contiguous, and a kernel that misreads that layout would then be
    handed one it reads right. An expanded tensor (a stride of 0) is copied contiguous: its elements share memory."""
    t = t.detach()
    if t.is_contiguous() or any(int(s) == 0 and int(n) > 1 for s, n in zip(t.stride(), t.shape)):
        return t.clone()
    import torch

    buf = torch.empty_strided(tuple(t.shape), tuple(t.stride()), dtype=t.dtype, device=t.device)
    buf.copy_(t)
    return buf


def sliced(obj, n: int, rows: int, dtype=None):
    """A copy of an argument with the token dimension cut to `rows`: tensors whose first dimension is n are cut
    along it, a 2-D tensor whose last dimension is n (MRoPE positions, [3, n]) along that; everything else is
    copied whole. Copies keep the argument's strides (kept). Reduced-precision floating tensors (REDUCED) are cast
    to `dtype` when given; fp8 and integer tensors keep their dtype. Containers are rebuilt; other values pass."""
    if _is_tensor(obj):
        t = obj
        if t.dim() >= 1 and int(t.shape[0]) == n:
            t = t[:rows]
        elif t.dim() == 2 and int(t.shape[-1]) == n:
            t = t[:, :rows]
        t = kept(t)
        if dtype is not None and str(t.dtype) in REDUCED:
            t = t.to(dtype)
        return t
    if isinstance(obj, tuple):
        return tuple(sliced(o, n, rows, dtype) for o in obj)
    if isinstance(obj, list):
        return [sliced(o, n, rows, dtype) for o in obj]
    if isinstance(obj, dict):
        return {k: sliced(o, n, rows, dtype) for k, o in obj.items()}
    return obj


def cast(obj, dtype=None):
    """A fresh clone of already-cut arguments (each run gets its own: a kernel may work in place), the
    reduced-precision tensors cast to `dtype` when given."""
    return sliced(obj, -1, -1, dtype)


def nbytes(obj) -> int:
    return sum(int(t.numel()) * int(t.element_size()) for t in all_tensors(obj))


def uniform_rows(tensors) -> bool:
    """Whether the largest floating tensor holds the same row over and over (a profile run's dummy input: every
    row is token 0). A row-stride or row-mapping bug gives the right answer on such rows, so they decide nothing."""
    ts = [t for t in tensors if t.dim() >= 2 and int(t.shape[0]) > 1]
    if not ts:
        return False
    t = max(ts, key=lambda x: x.numel())
    if t.element_size() == 1:          # fp8 codes: compared as the values they hold
        t = t.float()
    return bool((t == t[:1]).all())


@dataclass
class Comparison:
    """What the comparison found, over the values that are finite on both sides."""
    diff: float                     # max |kernel - reference|
    floor: Optional[float]          # max |definition in its dtype - reference|; None when the reference is that run
    noise: Optional[float]          # the typical (90th percentile) of that, the largest over the outputs
    scale: float                    # max |reference|
    elements: int
    violations: int                 # values beyond their allowance
    nonfinite: int                  # values non-finite on one side only, or non-finite and different
    margin: float                   # max over the values of |kernel - reference| / allowance (<= 1 passes)
    worst: Optional[tuple]          # (output, flat index, kernel, definition, reference, allowed) of the worst value
    dtypes: Tuple[str, ...]         # the kernel's output dtypes
    same_as_definition: bool        # every kernel output equals the dtype definition's output bitwise
    same_as_input: bool             # every definition output equals an input of the same shape (an identity)


def _pair(kernel_out, reference_out, native_out):
    ks, rs = tensors_of(kernel_out), tensors_of(reference_out)
    ns = tensors_of(native_out) if native_out is not None else None
    if len(ks) != len(rs) or (ns is not None and len(ns) != len(rs)) or not rs:
        raise ValueError(f"the kernel gives {len(ks)} floating tensors, the definition {len(rs)}")
    for i, (k, r) in enumerate(zip(ks, rs)):
        if tuple(k.shape) != tuple(r.shape):
            raise ValueError(f"output {i}: the kernel gives shape {tuple(k.shape)}, the definition {tuple(r.shape)}")
    return ks, rs, ns


def compare(kernel_out, reference_out, native_out=None, inputs=None) -> Comparison:
    """The kernel's output against the reference (the definition in float32; or in the input dtype when float32
    could not be run, then `native_out` is None), value by value, the floating tensors of each output paired by
    position. `inputs`: the floating tensors of the (untouched) input, for the identity check. Raises ValueError
    when the outputs do not pair up."""
    ks, rs, ns = _pair(kernel_out, reference_out, native_out)
    diff = floor = noise = scale = margin = 0.0
    elements = violations = nonfinite = 0
    worst, worst_excess = None, -math.inf
    for i, (k, r) in enumerate(zip(ks, rs)):
        kf, rf = k.detach().float(), r.detach().float()
        eps = ULPS.get(_dtype(k), 2.0 ** -7)
        fin = kf.isfinite() & rf.isfinite()
        nonfinite += int((kf.isfinite() != rf.isfinite()).sum())
        both = (~kf.isfinite()) & (~rf.isfinite())
        if bool(both.any()):
            nonfinite += int((both & (kf != rf) & ~(kf.isnan() & rf.isnan())).sum())
        err = (kf - rf).abs()
        if ns is not None:
            nf = ns[i].detach().float()
            fin = fin & nf.isfinite()
            nerr = (nf - rf).abs()
        else:
            nf = nerr = None
        elements += int(rf.numel())
        if not bool(fin.any()):
            continue
        e, ra = err[fin], rf[fin].abs()
        diff = max(diff, float(e.max()))
        scale = max(scale, float(ra.max()))
        allowed = ATOL_ULPS * eps * ra.maximum(kf[fin].abs())
        if nerr is not None:
            ne = nerr[fin]
            floor = max(floor, float(ne.max()))
            p90 = float(ne.flatten().kthvalue(max(1, int(math.ceil(0.9 * ne.numel())))).values)
            noise = max(noise, p90)
            allowed = allowed + FACTOR * ne.clamp_min(p90)
        ratio = e / allowed.clamp_min(1e-300)
        margin = max(margin, float(ratio.max()))
        bad = e > allowed
        violations += int(bad.sum())
        excess = (e - allowed)
        j = int(excess.argmax())
        if float(excess[j]) > worst_excess:
            worst_excess = float(excess[j])
            flat = int(fin.flatten().nonzero()[j])
            worst = (i, flat, float(kf.flatten()[flat]), (float(nf.flatten()[flat]) if nf is not None else None),
                     float(rf.flatten()[flat]), float(allowed[j]))
    same_def = all(k.equal(n) for k, n in zip(ks, ns)) if ns is not None else all(k.equal(r) for k, r in zip(ks, rs))
    defs = ns if ns is not None else rs
    ins = [t for t in (inputs or []) if _is_tensor(t)]
    same_in = bool(defs) and all(any(tuple(t.shape) == tuple(d.shape)
                                     and d.detach().to(t.dtype).float().equal(t.detach().float())
                                     for t in ins) for d in defs)
    return Comparison(diff=diff, floor=(floor if ns is not None else None), noise=(noise if ns is not None else None),
                      scale=scale, elements=elements, violations=violations, nonfinite=nonfinite, margin=margin,
                      worst=worst, dtypes=tuple(_dtype(k) for k in ks), same_as_definition=same_def,
                      same_as_input=same_in)


def vacuous(cmp: Comparison) -> Optional[str]:
    """Why this input decides nothing: a definition output of all zeros with which the kernel agrees (a dummy
    input), or a definition that is the identity on it (rotary at position 0) with which the kernel agrees. None
    when the comparison is decisive - a kernel that differs on such an input is decided, not excused."""
    if cmp.violations or cmp.nonfinite:
        return None
    if cmp.scale == 0.0 and cmp.diff == 0.0:
        return "the definition's output is all zeros on this input (a dummy input)"
    if cmp.same_as_input:
        return "the definition is the identity on this input (as rotary at position 0) and the kernel agrees"
    return None


def describe(cmp: Comparison) -> str:
    """The numbers of a comparison, as the record carries them (testbed/m18_kernel_summary.py reads this)."""
    s = f"max |kernel - definition| {cmp.diff:.3g} over {cmp.elements} values"
    if cmp.floor is not None:
        s += f"; the definition's own noise {cmp.floor:.3g} (typical {cmp.noise:.3g})"
    else:
        s += "; the definition could not be run in float32 (compared in the input dtype)"
    s += (f"; scale {cmp.scale:.3g}; {cmp.violations} values beyond FACTOR {FACTOR:g} x noise + {ATOL_ULPS:g} ulp "
          f"({', '.join(cmp.dtypes)}); worst ratio to the allowance {cmp.margin:.3g}")
    if cmp.nonfinite:
        s += f"; {cmp.nonfinite} values non-finite on one side only or non-finite and different"
    if cmp.worst is not None:
        i, j, k, n, r, a = cmp.worst
        s += (f"; worst value at output {i} index {j}: kernel {k:.4g}, "
              + (f"definition {n:.4g}, " if n is not None else "") + f"reference {r:.4g}, allowed {a:.3g}")
    if cmp.same_as_definition:
        s += "; the kernel reproduces the definition bitwise"
    return s


def check(boundary: str, consumer: str, op: str, cmp: Comparison, where: str, policy=None, record: bool = True,
          owner=None, extra: str = "", repair: Optional[str] = None) -> list:
    """Decide the kernel's output against the definition's. Returns the decisions; with `record` they are also
    recorded through load.enforce (the run goes on, or stops where the policy says). `repair`: what the caller can do
    about a mismatch (send the op to its definition, M19 L3); offered, a mismatch is resolved unless the policy
    repairs nothing for KernelReference (ENTAIL_POLICY=refuse or KernelReference=refuse), and the caller carries it
    out when the decision says resolved."""
    from . import load, policies
    from .contracts import RULES, Contract, Decision, Verdict, unrepaired
    from .facts import Certainty, Fact, KernelReference, Source

    policy = policy or policies.current()
    contract = Contract(boundary, consumer, ("KernelReference",))
    declared = Fact("KernelReference", KernelReference(op=op, max_abs_diff=0.0, floor=cmp.floor, scale=cmp.scale),
                    Source("engine", f"{op}.forward_native, the op's own definition"), Certainty.VERIFIED)
    held = Fact("KernelReference", KernelReference(op=op, max_abs_diff=cmp.diff, floor=cmp.floor, scale=cmp.scale),
                Source("engine", where), Certainty.VERIFIED)
    note = f"{where}: {describe(cmp)}{extra}"
    if (cmp.violations or cmp.nonfinite) and repair and policy.mismatch_setting("KernelReference") == "resolve":
        d = Decision(contract, "KernelReference", Verdict.RESOLVED, RULES["resolved"], declared=declared, chosen=held,
                     resolution=repair, note=note)
    elif cmp.violations or cmp.nonfinite:
        verdict, blocking = unrepaired(policy, "KernelReference")
        d = Decision(contract, "KernelReference", verdict, RULES["kernel_reference_mismatch"], declared=declared,
                     chosen=held, blocking=blocking, note=note)
    else:
        d = Decision(contract, "KernelReference", Verdict.PASS, RULES["match"], declared=declared, chosen=held,
                     note=note)
    decisions = [d]
    if record:
        _tally.counts(boundary)["checks"] += 1
        if d.verdict is Verdict.PASS:
            _tally.passed(boundary, ["kernel_reference"])
        if d.blocking:
            _tally.refused(boundary)
        elif d.verdict is Verdict.BROKEN:
            _tally.broken(boundary)
        elif d.verdict is Verdict.RESOLVED:
            _tally.counts(boundary)["resolved"] += 1
        _tally.tick(boundary)
        load.enforce(decisions, once_for=owner)
    return decisions


def resolved(decisions) -> bool:
    """Whether the decision says the repair the caller offered is to be carried out (M19 L3)."""
    from .contracts import Verdict

    return bool(decisions) and decisions[0].verdict is Verdict.RESOLVED


def stats(boundary: str) -> dict:
    return _tally.stats(boundary)


def reset(boundary: str) -> None:
    _tally.reset(boundary)
