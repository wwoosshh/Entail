"""Scaffolding, not a check: plant a KV bookkeeping defect in a real SGLang run, so the container contract can be
measured against it (moved out of adapters/sglang_cache_contract.py in M5.1).

Switched on with ENTAIL_SEED=kv_short and ENTAIL_SEED_AT=<decode batch>: at that batch, request 0's books say one
slot fewer than it was given - the way a bad restore or a slipped counter would leave them. Installed before the
contract (sitecustomize orders the list), so the contract sees the planted defect.
"""
import os

COUNTS = {"batches": 0, "planted": 0}


def install():
    if os.environ.get("ENTAIL_SEED") != "kv_short":
        return 0
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    at = int(os.environ.get("ENTAIL_SEED_AT", "0"))
    orig = ScheduleBatch.prepare_for_decode

    def wrapped(self, *a, **kw):
        out = orig(self, *a, **kw)
        COUNTS["batches"] += 1
        if at and COUNTS["batches"] == at and getattr(self, "reqs", None):
            self.reqs[0].kv.kv_allocated_len -= 1
            COUNTS["planted"] += 1
            print(f"[entail-seed] kv_short: took one slot off request 0 at batch {at}", flush=True)
        return out

    ScheduleBatch.prepare_for_decode = wrapped
    return 1
