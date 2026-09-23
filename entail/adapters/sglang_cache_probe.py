"""One-shot probe: what does SGLang actually record about a request's KV, and under which names?

Reading the source was slow going, so this prints the bookkeeping once, from a running scheduler. It is a
probe, not a check: it never raises and it prints at most a few lines. The contract itself
(adapters/sglang_cache_contract.py) is written from what this prints.
"""
import os

_ORIG = None
_SEEN = {"prints": 0}


def install():
    global _ORIG
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    if _ORIG is not None:
        return 0
    _ORIG = ScheduleBatch.prepare_for_decode

    def _fields(obj):
        """Attributes of an object that may use __slots__ instead of __dict__ (vars() raises on those)."""
        if obj is None:
            return None
        if hasattr(obj, "__dict__"):
            return dict(vars(obj))
        names = []
        for cls in type(obj).__mro__:
            names += list(getattr(cls, "__slots__", ()) or ())
        return {n: getattr(obj, n, None) for n in names}

    def wrapped(self, *a, **kw):
        out = _ORIG(self, *a, **kw)
        # A probe must never take the host down with it: the first version called vars() on an object with
        # __slots__ and killed the scheduler.
        try:
            if _SEEN["prints"] < 2 and getattr(self, "reqs", None):
                req = self.reqs[0]
                fields = _fields(getattr(req, "kv", None))
                scalars = {k: v for k, v in (fields or {}).items() if isinstance(v, (int, bool, float))}
                print(f"[entail-probe] kv fields: {sorted(fields) if fields else None}", flush=True)
                print(f"[entail-probe] kv scalars: {scalars}", flush=True)
                print(f"[entail-probe] origin_input_ids={len(getattr(req, 'origin_input_ids', []) or [])} "
                      f"output_ids={len(getattr(req, 'output_ids', []) or [])} "
                      f"fill_ids={len(getattr(req, 'fill_ids', []) or [])} "
                      f"seq_lens={getattr(self, 'seq_lens', None)}", flush=True)
                _SEEN["prints"] += 1
        except Exception as e:
            print(f"[entail-probe] probe failed harmlessly: {type(e).__name__}: {e}", flush=True)
        return out

    ScheduleBatch.prepare_for_decode = wrapped
    if os.environ.get("ENTAIL_VERBOSE"):
        print("[entail-probe] watching ScheduleBatch.prepare_for_decode", flush=True)
    return 1


def uninstall():
    global _ORIG
    if _ORIG is None:
        return 0
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    ScheduleBatch.prepare_for_decode = _ORIG
    _ORIG = None
    return 1
