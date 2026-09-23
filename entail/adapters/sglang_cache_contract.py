"""The same RANGE contract in SGLang: the KV a request holds must be the KV its tokens need.

Third engine, third set of books. transformers counts a length per cache layer, vLLM counts blocks per request,
and SGLang keeps two numbers on the request itself - `kv_allocated_len` and `kv_committed_len` - next to the
tokens it has. The probe that found them is adapters/sglang_cache_probe.py.

The rule itself is not written here: it is `entail/kv_contract.py`, shared with the other two engines.
This adapter only says where SGLang keeps the numbers - reserved, written, and the tokens the request has -
and hands them over as a KvExtent after `ScheduleBatch.prepare_for_decode`.

Nothing is read from the device, so the check costs no synchronisation: all three numbers are Python ints. That
is the lesson from the static-cache measurement (audits/CACHE_CONTRACT.md), applied from the start here.

Requests under sliding-window eviction are skipped: their KV is meant to be shorter than their tokens.
"""
import os

from .. import core
from ..core import RoleError
from ..kv_contract import KvExtent, check_extent

_ORIG = None
STATS = {"batches": 0, "checked": 0, "skipped": 0, "complaints": 0}


def _tokens(req):
    """Tokens the request has once this decode step writes its one new token."""
    return len(getattr(req, "origin_input_ids", []) or []) + len(getattr(req, "output_ids", []) or []) + 1


def install():
    """Wrap prepare_for_decode. Returns 1, or 0 if already installed."""
    global _ORIG
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    if _ORIG is not None:
        return 0
    _ORIG = ScheduleBatch.prepare_for_decode

    seed_at = int(os.environ.get("ENTAIL_SEED_AT", "0")) if os.environ.get("ENTAIL_SEED") == "kv_short"         else 0

    def wrapped(self, *a, **kw):
        out = _ORIG(self, *a, **kw)
        if core.mode() not in ("load", "debug"):
            return out
        STATS["batches"] += 1
        if seed_at and STATS["batches"] == seed_at and getattr(self, "reqs", None):
            # scaffolding: the books say one slot fewer than the request was given, the way a bad restore or a
            # slipped counter would leave them
            self.reqs[0].kv.kv_allocated_len -= 1
            print(f"[entail-seed] kv_short: took one slot off request 0 at batch {seed_at}", flush=True)
        if STATS["batches"] in (1, 32) and os.environ.get("ENTAIL_VERBOSE"):
            print(f"[entail] sglang cache contract: {STATS}", flush=True)
        for i, req in enumerate(getattr(self, "reqs", []) or []):
            kv = getattr(req, "kv", None)
            allocated = getattr(kv, "kv_allocated_len", None)
            committed = getattr(kv, "kv_committed_len", None)
            if allocated is None or committed is None:
                STATS["skipped"] += 1
                continue
            if getattr(kv, "swa_evicted_seqlen", 0) or getattr(kv, "swa_evict_floor", 0):
                STATS["skipped"] += 1  # a sliding window is meant to hold less
                continue
            STATS["checked"] += 1
            extent = KvExtent(held=allocated, written=committed, needed=_tokens(req))
            try:
                check_extent(extent, f"sglang request {i} ({getattr(req, 'rid', '?')})")
            except RoleError:
                STATS["complaints"] += 1
                raise
        return out

    ScheduleBatch.prepare_for_decode = wrapped
    import atexit

    atexit.register(report)
    return 1


def uninstall():
    global _ORIG
    if _ORIG is None:
        return 0
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    ScheduleBatch.prepare_for_decode = _ORIG
    _ORIG = None
    return 1


def report():
    if os.environ.get("ENTAIL_VERBOSE") and STATS["batches"]:
        print(f"[entail] sglang cache contract in pid {os.getpid()}: {STATS}", flush=True)


def stats():
    return dict(STATS)
