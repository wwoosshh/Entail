"""Adapter v2 for SGLang's KV bookkeeping: where a request keeps the KV it reserved and wrote (LIBRARY_DESIGN.md 4.6,
4.8; ROADMAP M5.1; audits/CACHE_CONTRACT.md).

  hook         sglang.srt.managers.schedule_batch.ScheduleBatch.prepare_for_decode: the scheduler, before a decode
               step. Every number is a Python int, so nothing synchronises with the device.
  read_choice  per request of the batch: the slots it reserved (kv_allocated_len) and wrote (kv_committed_len), the
               tokens it has once this step writes its one new token, and whether a sliding window evicts its KV.
               The names were found with a probe on a running scheduler (a research tool, outside the package).
  handles      none.
kv_contract decides (kv_written, kv_needed); windowed requests and requests without the counters are counted as
skipped. Planting a defect for a measurement is a research tool outside the package, not part of this adapter.
"""
from .. import core, kv_contract
from .base import Hook

engine = "sglang"
versions = "0.5.20"
BOUNDARY = "container:sglang.prepare_for_decode"
CONSUMER = "sglang.kv_cache"
_ORIG = None


def hooks():
    return [Hook("sglang.srt.managers.schedule_batch.ScheduleBatch.prepare_for_decode", "container")]


def read_choice(batch):
    """[(where, reserved, written, tokens, windowed)] for the requests of a decode batch; reserved and written are None
    for a request that does not keep the counters."""
    out = []
    for i, req in enumerate(getattr(batch, "reqs", None) or []):
        kv = getattr(req, "kv", None)
        tokens = len(getattr(req, "origin_input_ids", None) or []) + len(getattr(req, "output_ids", None) or []) + 1
        windowed = bool(getattr(kv, "swa_evicted_seqlen", 0) or getattr(kv, "swa_evict_floor", 0))
        out.append((f"sglang request {i} ({getattr(req, 'rid', '?')})", getattr(kv, "kv_allocated_len", None),
                    getattr(kv, "kv_committed_len", None), tokens, windowed))
    return out


def handles():
    return {}


def _decide(batch):
    for where, reserved, written, tokens, windowed in read_choice(batch):
        if reserved is None or written is None or windowed:
            kv_contract.skipped(BOUNDARY)
            continue
        kv_contract.check(BOUNDARY, CONSUMER, where,
                          kv_contract.KvExtent(held=reserved, written=written, needed=tokens))


def install():
    """Wrap prepare_for_decode. Returns 1, or 0 if already installed."""
    global _ORIG
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    if _ORIG is not None:
        return 0
    _ORIG = ScheduleBatch.prepare_for_decode

    def prepare_for_decode(self, *a, **kw):
        out = _ORIG(self, *a, **kw)
        if core.mode() in ("load", "debug"):
            kv_contract.guarded(BOUNDARY, CONSUMER, _decide, self)
        return out

    ScheduleBatch.prepare_for_decode = prepare_for_decode
    return 1


def uninstall():
    global _ORIG
    if _ORIG is None:
        return 0
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    ScheduleBatch.prepare_for_decode = _ORIG
    _ORIG = None
    return 1


def stats():
    return kv_contract.stats(BOUNDARY)


def reset():
    kv_contract.reset(BOUNDARY)
