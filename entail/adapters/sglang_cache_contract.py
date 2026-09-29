"""Adapter v2 for SGLang's KV bookkeeping: where a request keeps the KV it reserved and wrote (LIBRARY_DESIGN.md 4.6,
4.8; ROADMAP M5.1; audits/CACHE_CONTRACT.md).

  hook         sglang.srt.managers.schedule_batch.ScheduleBatch.prepare_for_decode: the scheduler, before a decode
               step. Every number is a Python int, so nothing synchronises with the device.
               sglang.srt.managers.scheduler.Scheduler.run_batch and .process_batch_result: per request, the batches
               launched with it whose results are not processed yet.
  read_choice  per request of the batch: the slots it reserved (kv_allocated_len) and wrote (kv_committed_len), the
               tokens it has once this step writes its one new token, and whether a sliding window evicts its KV.
               The names were found with a probe on a running scheduler (a research tool, outside the package).
               The tokens are the prompt, the outputs processed, and one for each batch in flight with the request:
               the token that batch sampled is not in output_ids until its result is processed. The overlap
               scheduler (SGLang's default) prepares a step before it processes the last batch's result, so that is
               one token for the requests of the last batch and none for a request whose decoding a prefill batch of
               other requests interrupted; without overlap it is none. 2.1.3 counted one for every request and said
               broken for a request others joined (issue #36).
  handles      none.
kv_contract decides (kv_written, kv_needed); windowed requests and requests without the counters are counted as
skipped. Planting a defect for a measurement is a research tool outside the package, not part of this adapter.
"""
from .. import core, kv_contract, load, policies
from .base import Hook

engine = "sglang"
versions = "0.5.20"
BOUNDARY = "container:sglang.prepare_for_decode"
CONSUMER = "sglang.kv_cache"
_ORIG = None
_SAID_SPECULATIVE = False   # the once-per-process note that speculative batches are not checked (M11.5)
_SCHED = {}                 # the scheduler's run_batch and process_batch_result, as they were
_INFLIGHT = {}              # request id -> batches launched with the request whose results are not processed yet


def hooks():
    return [Hook("sglang.srt.managers.schedule_batch.ScheduleBatch.prepare_for_decode", "container"),
            Hook("sglang.srt.managers.scheduler.Scheduler.run_batch", "container"),
            Hook("sglang.srt.managers.scheduler.Scheduler.process_batch_result", "container")]


def _key(req):
    return getattr(req, "rid", None) or id(req)


def read_choice(batch):
    """[(where, reserved, written, tokens, windowed)] for the requests of a decode batch; reserved and written are None
    for a request that does not keep the counters."""
    out = []
    for i, req in enumerate(getattr(batch, "reqs", None) or []):
        kv = getattr(req, "kv", None)
        tokens = (len(getattr(req, "origin_input_ids", None) or []) + len(getattr(req, "output_ids", None) or [])
                  + _INFLIGHT.get(_key(req), 0))
        windowed = bool(getattr(kv, "swa_evicted_seqlen", 0) or getattr(kv, "swa_evict_floor", 0))
        out.append((f"sglang request {i} ({getattr(req, 'rid', '?')})", getattr(kv, "kv_allocated_len", None),
                    getattr(kv, "kv_committed_len", None), tokens, windowed))
    return out


def handles():
    return {}


def speculative(batch) -> bool:
    """Whether the batch decodes speculatively: its spec_algorithm is set and not none."""
    algo = getattr(batch, "spec_algorithm", None)
    is_none = getattr(algo, "is_none", None)
    return algo is not None and (not is_none() if callable(is_none) else bool(algo))


def _decide(batch):
    global _SAID_SPECULATIVE
    if speculative(batch):
        # M11.5: under speculative decoding the scheduler reserves draft slots ahead of the tokens (kv_allocated_len
        # runs past kv_committed_len by the draft budget: eagle_prepare_for_decode, get_alloc_reserve_per_decode),
        # which this adapter does not model; 1.0 reported it as reserved and written slots disagreeing. Said once.
        kv_contract.skipped(BOUNDARY, len(getattr(batch, "reqs", None) or []))
        if not _SAID_SPECULATIVE:
            _SAID_SPECULATIVE = True
            load.enforce([load.cannot_check(BOUNDARY, CONSUMER, "KvExtent", "speculative decoding reserves draft "
                                            "slots ahead of the tokens (kv_allocated_len runs past kv_committed_len "
                                            "by the draft budget), which entail does not model: the KV extents of "
                                            "speculative batches are not checked", policies.current())])
        return
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


def install_inflight():
    """Wrap the scheduler's run_batch and process_batch_result, to count per request the batches launched with it
    whose results are not processed yet (every event loop goes through the two). Returns 1, or 0 if installed."""
    from sglang.srt.managers.scheduler import Scheduler

    if _SCHED:
        return 0
    run, done = Scheduler.run_batch, Scheduler.process_batch_result
    _SCHED.update(run=run, done=done)

    def run_batch(self, batch, *a, **kw):
        out = run(self, batch, *a, **kw)
        for req in getattr(batch, "reqs", None) or []:
            _INFLIGHT[_key(req)] = _INFLIGHT.get(_key(req), 0) + 1
        return out

    def process_batch_result(self, batch, *a, **kw):
        reqs = list(getattr(batch, "reqs", None) or [])     # before processing, which may drop finished requests
        try:
            return done(self, batch, *a, **kw)
        finally:
            for req in reqs:
                n = _INFLIGHT.pop(_key(req), 0) - 1
                if n > 0:
                    _INFLIGHT[_key(req)] = n

    Scheduler.run_batch, Scheduler.process_batch_result = run_batch, process_batch_result
    return 1


def uninstall():
    global _ORIG
    if _SCHED:
        from sglang.srt.managers.scheduler import Scheduler

        Scheduler.run_batch, Scheduler.process_batch_result = _SCHED.pop("run"), _SCHED.pop("done")
        _INFLIGHT.clear()
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
