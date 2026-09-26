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
  only among the kernel's value parameters (`ints_from`): constexprs and launch options (BLOCK_*, num_warps) are
  not strides, and a collision with them would say "told" falsely. A stride of 0 (an expanded view) and dimensions
  of size 1 are not strides a kernel can misread; a tensor passed as a plain integer (data_ptr) is invisible here.

Each (kernel, stride pattern of its tensor arguments) is decided once per process (the adapter's memo), so the cost
sits at the first launch of each pattern, not on every launch.
"""
import re
from typing import Dict, Iterable, List, Optional

from . import tally as _tally

RULE_NAMES = ("kernel_stride_assumed",)
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


def stats(boundary: Optional[str] = None) -> dict:
    return _tally.stats(boundary) if boundary else {}


def reset(boundary: str) -> None:
    _tally.reset(boundary)
