"""tally: what the boundaries that run per step count, and the guards every such boundary shares (ROADMAP M5).

A container boundary (kv_contract, epochs) runs per layer per step - thousands of times a request - so a rule that
holds is counted, never recorded one by one; a broken one is a Decision, recorded and raised through load.enforce by
the module that owns the rule. What was counted goes to ENTAIL_RECORD as one line (and is printed with
ENTAIL_VERBOSE) after the 1st, 2nd, 4th, 8th ... thing a boundary saw, when it refuses, and at exit: engines check in
child processes that are not always let to exit normally (SGLang's scheduler).

Shared guards:
  inside_capture  no check runs while torch compiles or a CUDA graph is captured (principle 6): measured, a check
                  there cost 8x to 48x and changed the output (audits/CACHE_CONTRACT.md)
  guarded         an error inside entail is reported once per boundary, as that boundary not being checked, and
                  never breaks the engine (principle 12); a refusal passes through
"""
import atexit
import json
import os
import sys
from typing import Dict, Optional, Tuple

from .core import RoleError

PASSES: Dict[Tuple[str, str], int] = {}   # (boundary, rule) -> checks that held
STATS: Dict[str, Dict[str, int]] = {}     # boundary -> {"checks", "refused", "skipped", "deferred", "resolved"}
_BROKEN = set()                            # boundaries entail itself failed at; left alone from then on


def counts(boundary: str) -> Dict[str, int]:
    s = STATS.get(boundary)
    if s is None:
        s = STATS[boundary] = {"checks": 0, "refused": 0, "skipped": 0, "deferred": 0, "resolved": 0}
    return s


def passed(boundary: str, rules) -> None:
    for r in rules:
        PASSES[(boundary, r)] = PASSES.get((boundary, r), 0) + 1


def tick(boundary: str) -> None:
    """After the 1st, 2nd, 4th, 8th ... thing a boundary saw, write what it has counted so far."""
    s = STATS[boundary]
    n = s["checks"] + s["skipped"] + s["deferred"]
    if n & (n - 1) == 0:
        write_summary({boundary: stats(boundary)})


def refused(boundary: str) -> None:
    """Count a refusal and write the summary now: the process may not live to write one at exit."""
    counts(boundary)["refused"] += 1
    write_summary({boundary: stats(boundary)})


def stats(boundary: Optional[str] = None) -> dict:
    """Counts for one boundary (checks, refused, skipped, deferred, resolved, and the passes per rule), or all."""
    if boundary is None:
        return {b: stats(b) for b in STATS}
    s = dict(counts(boundary))
    s["passed"] = {r: n for (b, r), n in PASSES.items() if b == boundary}
    return s


def reset(boundary: Optional[str] = None) -> None:
    for b in [b for b in STATS if boundary is None or b == boundary]:
        del STATS[b]
    for k in [k for k in PASSES if boundary is None or k[0] == boundary]:
        del PASSES[k]
    if boundary is None:
        _BROKEN.clear()
    else:
        _BROKEN.discard(boundary)


def write_summary(counted: dict) -> None:
    summary = {"pid": os.getpid(), "boundaries": counted}
    if os.environ.get("ENTAIL_VERBOSE"):
        print(f"[entail] boundaries in pid {os.getpid()}: {counted}", flush=True)
    path = os.environ.get("ENTAIL_RECORD")
    if path:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        except OSError:
            pass


@atexit.register
def _report():
    if STATS:
        write_summary(stats())


def inside_capture() -> bool:
    """True while torch is compiling or a CUDA graph is being captured."""
    if "torch" not in sys.modules:   # nothing can be capturing without torch; and the core never imports it
        return False
    import torch

    # called directly, so that torch.compile folds it to a constant while tracing instead of breaking the graph
    if torch.compiler.is_compiling():
        return True
    try:
        return bool(torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        return False


def guarded(boundary: str, consumer: str, name: str, work, *args, **kwargs):
    """Run an adapter's reading and a core module's deciding so that an error inside entail never breaks the engine.
    A refusal (RoleError) passes through; anything else is reported once, as this boundary not being checked for the
    fact `name`, and the boundary is left alone from then on. Returns work()'s result, or None."""
    if boundary in _BROKEN:
        return None
    try:
        return work(*args, **kwargs)
    except RoleError:
        raise
    except Exception as e:  # noqa: BLE001
        from . import load, policies

        _BROKEN.add(boundary)
        load.enforce([load.cannot_check(boundary, consumer, name,
                                        f"entail failed here and stops checking this boundary: {type(e).__name__}: "
                                        f"{e}", policies.current())])
        return None
