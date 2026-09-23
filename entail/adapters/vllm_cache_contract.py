"""The same RANGE contract on vLLM's paged cache: the slots a request holds must cover the tokens it has.

transformers keeps a cache per request and a length per layer; vLLM keeps one pool of blocks and a table
saying which blocks belong to which request. The fact is the same one - how many tokens of this request are
actually held - and the boundary where it is decided is `KVCacheManager.allocate_slots`.

That boundary is in the scheduler, in Python, outside anything CUDA graphs capture, which is where the static
cache measurement said a check has to live (audits/CACHE_CONTRACT.md).

The rule is `entail/kv_contract.py`, shared with the other two engines; this adapter says where vLLM keeps
the numbers and that its allocation is granular (one block, not one token). Groups with a sliding window or a
chunked-attention span are skipped; they are meant to hold less.
"""
import atexit
import os

from .. import core
from ..core import RoleError
from ..kv_contract import KvExtent, check_extent

_ORIG = None
STATS = {"allocations": 0, "checked": 0, "complaints": 0}


def _managers(manager):
    coord = getattr(manager, "coordinator", None)
    return list(getattr(coord, "single_type_managers", []) or [])


def _is_capped(single):
    """A group that is supposed to hold fewer tokens than the request has."""
    for attr in ("sliding_window", "attention_chunk_size", "window_size"):
        v = getattr(single, attr, None)
        if isinstance(v, int) and v > 0:
            return True
    spec = getattr(single, "kv_cache_spec", None)
    for attr in ("sliding_window", "attention_chunk_size"):
        v = getattr(spec, attr, None)
        if isinstance(v, int) and v > 0:
            return True
    return False


def install():
    """Wrap allocate_slots. Returns 1, or 0 if already installed."""
    global _ORIG
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    if _ORIG is not None:
        return 0
    _ORIG = KVCacheManager.allocate_slots

    def wrapped(self, request, num_new_tokens, *a, **kw):
        out = _ORIG(self, request, num_new_tokens, *a, **kw)
        if out is None or core.mode() not in ("load", "debug"):
            return out
        STATS["allocations"] += 1
        need = int(getattr(request, "num_computed_tokens", 0)) + int(num_new_tokens) \
            + int(kw.get("num_lookahead_tokens", 0))
        try:
            groups = self.get_block_ids(request.request_id)
        except Exception:
            return out
        singles = _managers(self)
        for g, ids in enumerate(groups):
            single = singles[g] if g < len(singles) else None
            if single is not None and _is_capped(single):
                continue
            block_size = getattr(single, "block_size", None) or getattr(self, "scheduler_block_size", None)
            if not block_size:
                continue
            have = len(ids) * int(block_size)
            STATS["checked"] += 1
            try:
                check_extent(KvExtent(held=have, needed=need, granularity=int(block_size)),
                             f"vllm kv cache group {g}, request {request.request_id}")
            except RoleError:
                STATS["complaints"] += 1
                raise
        return out

    KVCacheManager.allocate_slots = wrapped
    atexit.register(_report)
    return 1


def _report():
    if os.environ.get("ENTAIL_VERBOSE") and STATS["allocations"]:
        print(f"[entail] vllm paged cache contract in pid {os.getpid()}: {STATS}", flush=True)


def uninstall():
    global _ORIG
    if _ORIG is None:
        return 0
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    KVCacheManager.allocate_slots = _ORIG
    _ORIG = None
    return 1


def stats():
    return dict(STATS)


def reset():
    for k in STATS:
        STATS[k] = 0
