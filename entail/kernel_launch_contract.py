"""kernel_launch_contract: what a kernel is told about the tensors it is handed (ROADMAP M17.3; LIBRARY_DESIGN.md
4.6 Layout; the verdict semantics were written in ROADMAP M17 before this code).

A Triton kernel reads a tensor through a pointer and the strides it is given as arguments; whatever it is not given
it assumes. The common assumption is a contiguous innermost dimension: `ptr + row * stride_row + col`. A tensor whose
innermost dimension is strided (a column of a split, a transposed view) is then read interleaved with its neighbour
(sglang#21843: fused_gdn_gating read a and b interleaved when num_v_heads == num_k_heads, because the reshape kept
strides (2*heads, 2) and the kernel was told the row stride only).

The rule is one, generic, for every kernel and every engine; it decides from the launch alone, without knowing the
kernel (ladder: a consumer whose use is unknown is unknown, so the rule is conservative by design):

  kernel_stride_assumed   a tensor argument whose innermost dimension of size > 1 has a stride other than 1, handed
                          to a kernel that was not told that stride (no integer argument equals it) and whose
                          parameters name no stride at all: the kernel cannot know, so it reads the tensor as if it
                          were contiguous -> broken (reported; nothing here can repair a launch).
  (unknown)               the same tensor, handed to a kernel that does name stride arguments, none of whose integer
                          arguments equals that innermost stride: the kernel may or may not know; said once.
  (pass)                  some integer argument of the kernel equals the innermost stride: told, whatever the name.
  A stride-like name is `stride` anywhere, `_s0`-style suffixes, `s`+one or two letters (with at most one more
  character after an underscore: `sxm`, `sq_d`, not `seq_len`), `s_...`, `ld...` (vLLM's sparse indexer `q_s0`,
  mxfp8's `sxm`, SGLang's `a_s0`/`sq_d`; M17.4 review, finding 1). Integer arguments count
  only among the kernel's value parameters and its stride-named constexprs (`ints_from`; strides are often
  constexprs for specialisation: causal_conv1d's `stride_x_token: tl.constexpr`): other constexprs and launch
  options (BLOCK_*, num_warps) are not strides, and a collision with them would say "told" falsely. A stride of 0 (an expanded view) and dimensions
  of size 1 are not strides a kernel can misread; a tensor passed as a plain integer (data_ptr) is invisible here.

Each (kernel, stride pattern of its tensor arguments) is decided once per process (the adapter's memo), so the cost
sits at the first launch of each pattern, not on every launch.

Deciding by running it (M19 L3.3b). "Cannot be told from the launch" need not stay unknown: the same launch can be
run twice on copies of its tensors, once as given and once with the strided tensors laid out again - the innermost
dimension contiguous, every other stride kept (`relaid`), so the kernel's stride arguments still describe the
tensor and a kernel that assumes a contiguous innermost dimension reads it right. The meaning of a tensor does not
depend on how its values are laid out, so the two launches must write the same values:

  kernel_layout_variant   the launch as given writes other values than the same launch with the strided tensors
                          laid out contiguously in their innermost dimension (beyond the kernel's own run-to-run
                          noise, measured by a second run of that layout): it reads the tensor as if it were
                          contiguous. Resolved by launching the kernel with the tensors laid out that way (and
                          copying what it wrote back), the layout its arguments describe.
  (pass)                  the two launches write the same values: the kernel reads the strides right.

A layout that keeping the outer strides cannot give (a transposed tensor: its innermost dimension's neighbour has
stride 1) is not run twice; the launch stays with the rule above.
"""
import re
from typing import Dict, Iterable, List, Optional

from . import tally as _tally

RULE_NAMES = ("kernel_stride_assumed", "kernel_layout_variant")
STRIDE_NAME = re.compile(r"stride|_s\d+$|^s[a-z]{1,2}(_[a-z0-9])?$|^s_|^ld", re.IGNORECASE)   # sq_d, sxm; not seq_len


def innermost_stride(t):
    """(dimension, size, stride) of the innermost dimension of size > 1, or None when there is none."""
    try:
        shape, strides = tuple(t.shape), tuple(t.stride())
    except (AttributeError, TypeError, RuntimeError):
        return None
    for d in range(len(shape) - 1, -1, -1):
        if shape[d] > 1:
            return d, shape[d], strides[d]
    return None


def stride_like(name: str) -> bool:
    return bool(STRIDE_NAME.search(name))


def classify(bound: Dict[str, object], ints_from: Optional[Iterable[str]] = None) -> List[dict]:
    """Per tensor argument with a strided innermost dimension: its name, the stride, and the status by the
    kernel's other arguments: 'told' (an integer value parameter equals the stride), 'unknown' (stride-like
    parameters exist, none equals it), 'assumed' (no stride-like parameter and none equals it). `ints_from`: the
    names of the kernel's value (non-constexpr) parameters; None counts every integer argument."""
    names = list(bound)
    stride_params = [n for n in names if stride_like(n)]
    ints = {int(v) for n, v in bound.items()
            if isinstance(v, int) and not isinstance(v, bool) and (ints_from is None or n in ints_from)}
    out = []
    for n, v in bound.items():
        if not (hasattr(v, "stride") and hasattr(v, "shape") and callable(getattr(v, "stride", None))):
            continue
        inner = innermost_stride(v)
        if inner is None:
            continue
        d, size, s = inner
        if s in (0, 1):
            continue
        status = "told" if s in ints else ("unknown" if stride_params else "assumed")
        out.append({"name": n, "dim": d, "size": size, "stride": s, "status": status,
                    "stride_params": stride_params})
    return out


def check(boundary: str, consumer: str, kernel: str, bound: Dict[str, object], where: str, owner=None,
          policy=None, record: bool = True, ints_from: Optional[Iterable[str]] = None) -> list:
    """Decide one launch. `bound`: parameter name -> argument (tensors and scalars alike, constexprs included);
    `ints_from`: the value parameters whose integers count as strides told. Returns the decisions; with `record`
    they go through load.enforce, once per `owner` (the adapter's layout key). A launch with no strided tensor
    passes silently (counted)."""
    from . import load, policies
    from .contracts import RULES, Contract, Decision, Verdict, unrepaired

    policy = policy or policies.current()
    found = classify(bound, ints_from)
    contract = Contract(boundary, consumer, ("Layout",), ("Layout",))
    decisions: List[Decision] = []
    for f in found:
        t = bound[f["name"]]
        seen = f"{where}: {f['name']} shape {tuple(t.shape)} stride {tuple(t.stride())}"
        if f["status"] == "told":
            continue
        if f["status"] == "assumed":
            verdict, blocking = unrepaired(policy, "Layout")
            decisions.append(Decision(contract, "Layout", verdict, RULES["kernel_stride_assumed"], blocking=blocking,
                                      note=(f"{f['name']} is strided in its innermost dimension (dim {f['dim']}, size "
                                            f"{f['size']}, stride {f['stride']}) and {kernel} takes no stride "
                                            f"argument and no integer argument equals {f['stride']}: it reads "
                                            f"{f['name']} as if it were contiguous ({seen})")))
        else:
            decisions.append(Decision(contract, "Layout", Verdict.UNKNOWN, RULES["cannot_check"],
                                      note=(f"{f['name']} is strided in its innermost dimension (dim {f['dim']}, "
                                            f"stride {f['stride']}); {kernel} takes stride arguments "
                                            f"({', '.join(f['stride_params'])}) but none of its integer arguments "
                                            f"equals {f['stride']}: whether it knows cannot be told from the launch "
                                            f"({seen})")))
    if not decisions:
        decisions.append(Decision(contract, "Layout", Verdict.PASS, RULES["match"],
                                  note="" if not found else "strided tensors, their strides passed"))
    if record:
        _tally.counts(boundary)["checks"] += 1
        if all(d.verdict is Verdict.PASS for d in decisions):
            _tally.passed(boundary, list(RULE_NAMES))
        load.enforce([d for d in decisions if d.verdict is not Verdict.PASS] or decisions, once_for=owner)
        if any(d.blocking for d in decisions):
            _tally.refused(boundary)
        elif any(d.verdict is Verdict.BROKEN for d in decisions):
            _tally.broken(boundary)
        _tally.tick(boundary)
    return decisions


def relaid_strides(t, whole: bool = False) -> Optional[tuple]:
    """The strides of `t` with its innermost dimension of size > 1 made contiguous and every other stride kept (the
    layout a kernel told the outer strides reads right), or, with `whole`, the contiguous strides (the layout a kernel
    told no stride at all reads right). None when there is nothing to change, or when keeping the outer strides would
    put two elements at one place (a transposed tensor)."""
    inner = innermost_stride(t)
    if inner is None or inner[2] in (0, 1):
        return None
    shape = [int(n) for n in t.shape]
    if whole:
        strides, acc = [0] * len(shape), 1
        for d in range(len(shape) - 1, -1, -1):
            strides[d] = acc
            acc *= max(shape[d], 1)
        return tuple(strides)
    strides = [int(s) for s in t.stride()]
    strides[inner[0]] = 1
    dims = sorted((s, n) for s, n in zip(strides, shape) if n > 1)
    for (s0, n0), (s1, _n1) in zip(dims, dims[1:]):
        if s1 < s0 * n0:
            return None
    return tuple(strides)


def relaid(t, margin: int = 0, whole: bool = False):
    """A copy of `t` in the layout relaid_strides gives (same shape, dtype and values), in zeroed storage with
    `margin` spare elements after it; None when there is no such layout."""
    strides = relaid_strides(t, whole)
    if strides is None:
        return None
    from .kernel_reference_contract import strided_zeros

    buf = strided_zeros(t.shape, strides, t.dtype, t.device, margin)
    buf.copy_(t)
    return buf


def check_relaid(boundary: str, consumer: str, kernel: str, names: List[str], cmp, where: str, owner=None,
                 policy=None, repair: Optional[str] = None, record: bool = True) -> list:
    """Decide a launch run twice (as given, and with `names` relaid; cmp = kernel_reference_contract.compare of
    the as-given launch's floating tensors against the relaid launch's, the second relaid run as the noise). `repair`
    is offered as the resolution of a mismatch unless the policy repairs nothing for Layout."""
    from . import load, policies
    from .contracts import RULES, Contract, Decision, Verdict, unrepaired
    from .kernel_reference_contract import describe

    policy = policy or policies.current()
    contract = Contract(boundary, consumer, ("Layout",), ("Layout",))
    which = ", ".join(names)
    numbers = describe(cmp).replace("kernel - definition", "as given - relaid").replace("the definition's own noise",
                                                                                      "the kernel's own noise")
    if cmp.violations or cmp.nonfinite:
        note = (f"{where}: {kernel} writes other values when {which} is strided in its innermost dimension than when "
                f"the same values are laid out contiguously there with every other stride kept: it reads {which} as "
                f"if it were contiguous ({numbers})")
        if repair and policy.mismatch_setting("Layout") == "resolve":
            d = Decision(contract, "Layout", Verdict.RESOLVED, RULES["resolved"], resolution=repair, note=note)
        else:
            verdict, blocking = unrepaired(policy, "Layout")
            d = Decision(contract, "Layout", verdict, RULES["kernel_layout_variant"], blocking=blocking, note=note)
    else:
        d = Decision(contract, "Layout", Verdict.PASS, RULES["match"],
                     note=f"{where}: {kernel} reads {which} right: the launch as given and the launch with {which} "
                          f"laid out contiguously in the innermost dimension write the same values ({numbers})")
    decisions = [d]
    if record:
        _tally.counts(boundary)["checks"] += 1
        if d.verdict is Verdict.PASS:
            _tally.passed(boundary, ["kernel_layout_variant"])
        elif d.blocking:
            _tally.refused(boundary)
        elif d.verdict is Verdict.BROKEN:
            _tally.broken(boundary)
        elif d.verdict is Verdict.RESOLVED:
            _tally.counts(boundary)["resolved"] += 1
        _tally.tick(boundary)
        load.enforce(decisions, once_for=owner)
    return decisions


def stats(boundary: Optional[str] = None) -> dict:
    return _tally.stats(boundary) if boundary else {}


def reset(boundary: str) -> None:
    _tally.reset(boundary)
