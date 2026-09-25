"""Adapter v2 for vLLM's prefix-cache block hashes: the identity a request's stored block hashes stand for
(LIBRARY_DESIGN.md 4.6, 4.8; ROADMAP M14; realworld/CODEBOOK_v2.md I; vllm#49377, #49449).

  hook         vllm.v1.core.sched.scheduler.Scheduler._update_request_as_session: where a streaming session discards
               its last sampled token(s) in place and appends the next input chunk, then calls update_block_hashes.
               The hashes are only appended, so a hash chained over a discarded token survives here.
  read_choice  the block hashes the request holds now (stored), and the hashes its current tokens give from the
               truncation point on (fresh) - both as hex strings - with the tokens one block covers. Only the blocks
               from the truncation point are recomputed; the earlier ones cannot have changed, so the cost is the
               tail, not the whole prefix.
  handles      identity_recompute: forget the request's block hashes from the first stale one and let
               update_block_hashes remake them from the current tokens. The one repair; it removes the false keys
               before the next prefix-cache lookup.
identity_contract decides (identity_stale). A request with multimodal features is left to the engine (the extra
keys of an image block are out of what this reads); it is counted as skipped.
"""
from .. import core, identity_contract
from .base import Hook

engine = "vllm"
versions = "0.30.0"
BOUNDARY = "container:vllm.request.block_hashes"
CONSUMER = "vllm.prefix_cache"
_ORIG = None


def hooks():
    return [Hook("vllm.v1.core.sched.scheduler.Scheduler._update_request_as_session", "container")]


def _hasher_env(request):
    """The block size and hash function the request's own hasher closes over, read by name (robust to their order)."""
    fn = getattr(request, "_block_hasher", None)
    if fn is None or getattr(fn, "__closure__", None) is None:
        return None, None
    env = dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__)))
    return env.get("hash_block_size"), env.get("caching_hash_fn")


def _hex(block_hash):
    try:
        return bytes(block_hash).hex()
    except (TypeError, ValueError):
        return hex(block_hash) if isinstance(block_hash, int) else repr(block_hash)


def read_choice(request, start_token_idx):
    """(stored hex keys from `start`, fresh hex keys from `start`, tokens per block). `start` is the block index the
    truncation could have touched; earlier blocks are unchanged, so the fresh chain seeds from the last good hash."""
    from vllm.v1.core import kv_cache_utils as kcu

    block, hash_fn = _hasher_env(request)
    if not block or hash_fn is None or getattr(request, "mm_features", None):
        return None
    stored_raw = list(request.block_hashes)
    tokens = list(request.all_token_ids)
    start = min(max(int(start_token_idx) // block, 0), len(stored_raw))
    prev = stored_raw[start - 1] if start > 0 else None
    full = len(tokens) // block
    fresh = []
    for i in range(start, full):
        s, e = i * block, (i + 1) * block
        extra, _mm = kcu.generate_block_hash_extra_keys(request, s, e, -1 if i > 0 else 0)
        h = kcu.hash_block_tokens(hash_fn, prev, tokens[s:e], extra)
        fresh.append(h)
        prev = h
    stored = [_hex(h) for h in stored_raw[start:]]
    return stored, [_hex(h) for h in fresh], block, start


def handles(request):
    def identity_recompute(index):
        del request.block_hashes[index:]
        request.update_block_hashes()
        return index
    return {"identity_recompute": identity_recompute}


def _decide(session, truncation_at):
    read = read_choice(session, truncation_at)
    if read is None:
        identity_contract.stats(BOUNDARY)["skipped"] = identity_contract.stats(BOUNDARY).get("skipped", 0) + 1
        return
    stored, fresh, block, start = read
    hs = handles(session)

    def recompute(index):
        return hs["identity_recompute"](start + index)   # the rule's index is within the tail; the store is whole

    where = f"vllm session request {getattr(session, 'request_id', '?')}, blocks from {start}"
    identity_contract.check(BOUNDARY, CONSUMER, where, stored, fresh, recompute, covers=block,
                            owner=getattr(session, "request_id", None))


def install():
    """Wrap Scheduler._update_request_as_session. Returns 1, or 0 if already installed."""
    global _ORIG
    from vllm.v1.core.sched.scheduler import Scheduler

    if _ORIG is not None:
        return 0
    _ORIG = Scheduler._update_request_as_session

    def _update_request_as_session(self, session, update, *a, **kw):
        at = getattr(session, "num_computed_tokens", 0)
        out = _ORIG(self, session, update, *a, **kw)
        if core.mode() in ("load", "debug"):
            from .. import load
            load.safely(BOUNDARY, CONSUMER, "Identity", lambda: _decide(session, at))
        return out

    Scheduler._update_request_as_session = _update_request_as_session
    return 1


def uninstall():
    global _ORIG
    if _ORIG is None:
        return 0
    from vllm.v1.core.sched.scheduler import Scheduler

    Scheduler._update_request_as_session = _ORIG
    _ORIG = None
    return 1


def stats():
    return identity_contract.stats(BOUNDARY)


def reset():
    identity_contract.reset(BOUNDARY)
