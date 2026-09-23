"""Adapter v2 for vLLM's paged KV cache: where the scheduler keeps the blocks of a request (LIBRARY_DESIGN.md 4.6,
4.8; ROADMAP M5.1; audits/CACHE_CONTRACT.md).

  hook         vllm.v1.core.kv_cache_manager.KVCacheManager.allocate_slots: the scheduler, in Python, outside
               anything CUDA graphs capture.
  read_choice  the tokens the request needs once the step runs (computed + new + lookahead) and, per KV cache group,
               the slots its blocks hold (blocks x block size), the block size - vLLM allocates in blocks, not tokens -
               and whether the group is windowed (a sliding window or a chunked-attention span holds less on purpose).
  handles      none: a request whose blocks do not cover its tokens cannot be repaired here.
kv_contract decides (kv_needed, with the block size as the allocation unit); windowed groups are counted as skipped.
"""
from .. import core, kv_contract
from .base import Hook

engine = "vllm"
versions = "0.30.0"
BOUNDARY = "container:vllm.allocate_slots"
CONSUMER = "vllm.kv_cache"
_ORIG = None


def hooks():
    return [Hook("vllm.v1.core.kv_cache_manager.KVCacheManager.allocate_slots", "container")]


def _windowed(single):
    for obj in (single, getattr(single, "kv_cache_spec", None)):
        for attr in ("sliding_window", "attention_chunk_size", "window_size"):
            v = getattr(obj, attr, None)
            if isinstance(v, int) and v > 0:
                return True
    return False


def read_choice(manager, request, num_new_tokens, kw):
    """(tokens the request needs, [(group, slots held or None, block size or None, windowed)])."""
    need = int(getattr(request, "num_computed_tokens", 0)) + int(num_new_tokens) \
        + int(kw.get("num_lookahead_tokens", 0))
    singles = list(getattr(getattr(manager, "coordinator", None), "single_type_managers", None) or [])
    groups = []
    for g, ids in enumerate(manager.get_block_ids(request.request_id)):
        single = singles[g] if g < len(singles) else None
        size = getattr(single, "block_size", None) or getattr(manager, "scheduler_block_size", None)
        groups.append((g, len(ids) * int(size) if size else None, int(size) if size else None,
                       single is not None and _windowed(single)))
    return need, groups


def handles():
    return {}


def _decide(manager, request, num_new_tokens, kw):
    need, groups = read_choice(manager, request, num_new_tokens, kw)
    for g, held, size, windowed in groups:
        if windowed or held is None:
            kv_contract.skipped(BOUNDARY)
            continue
        kv_contract.check(BOUNDARY, CONSUMER, f"vllm kv cache group {g}, request {request.request_id}",
                          kv_contract.KvExtent(held=held, needed=need, granularity=size))


def install():
    """Wrap allocate_slots. Returns 1, or 0 if already installed."""
    global _ORIG
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    if _ORIG is not None:
        return 0
    _ORIG = KVCacheManager.allocate_slots

    def allocate_slots(self, request, num_new_tokens, *a, **kw):
        out = _ORIG(self, request, num_new_tokens, *a, **kw)
        if out is not None and core.mode() in ("load", "debug"):   # None: nothing was allocated this step
            kv_contract.guarded(BOUNDARY, CONSUMER, _decide, self, request, num_new_tokens, kw)
        return out

    KVCacheManager.allocate_slots = allocate_slots
    return 1


def uninstall():
    global _ORIG
    if _ORIG is None:
        return 0
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    KVCacheManager.allocate_slots = _ORIG
    _ORIG = None
    return 1


def stats():
    return kv_contract.stats(BOUNDARY)


def reset():
    kv_contract.reset(BOUNDARY)
