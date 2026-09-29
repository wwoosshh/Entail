"""Tests for the SGLang KV adapter's reading and deciding without SGLang (ROADMAP M5.1, M11.5): what it reads off a
decode batch, that a healthy batch is quiet, that a lost slot is caught, that a speculative batch is not checked but
said so once, and that the tokens it expects count the batches in flight (issue #36), on a stand-in scheduler run
the way SGLang's event loops run it. Run: python tests/test_sglang_cache_contract.py"""
import io
import os
import sys
import types
from contextlib import redirect_stdout
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ["ENTAIL_ON_BROKEN"] = "stop"
from entail import core, kv_contract, load  # noqa: E402
from entail.adapters import sglang_cache_contract as sc  # noqa: E402
from entail.contracts import Verdict  # noqa: E402
from entail.core import RoleError  # noqa: E402


def req(rid, allocated, committed, prompt, out, swa=0):
    return SimpleNamespace(rid=rid, origin_input_ids=list(range(prompt)), output_ids=list(range(out)),
                           kv=SimpleNamespace(kv_allocated_len=allocated, kv_committed_len=committed,
                                              swa_evicted_seqlen=swa, swa_evict_floor=0))


def batch(reqs, spec=None):
    return SimpleNamespace(reqs=reqs, spec_algorithm=spec)


def decide(b):
    printed = io.StringIO()
    core.set_mode("load")
    kv_contract.reset()
    n = len(load.LEDGER.decisions)
    try:
        with redirect_stdout(printed):
            sc._decide(b)
    finally:
        core.set_mode("off")
    return load.LEDGER.decisions[n:], printed.getvalue()


def test_read_choice():
    """The tokens: the prompt, the outputs processed, and one per batch in flight with the request (its sampled token
    is not in output_ids until the result is processed)."""
    sc._INFLIGHT.clear()
    sc._INFLIGHT["a"] = 1
    got = sc.read_choice(batch([req("a", 12, 12, 8, 3), req("b", 40, 40, 30, 9, swa=5)]))
    assert got == [("sglang request 0 (a)", 12, 12, 12, False), ("sglang request 1 (b)", 40, 40, 39, True)]
    assert sc.read_choice(SimpleNamespace(reqs=[SimpleNamespace(rid="c")])) == [("sglang request 0 (c)", None, None, 0, False)]
    sc._INFLIGHT.clear()


def test_a_healthy_batch_is_quiet_and_a_lost_slot_is_caught():
    new, printed = decide(batch([req("a", 11, 11, 8, 3)]))
    assert new == [] and printed == "" and kv_contract.stats(sc.BOUNDARY)["checks"] == 1
    try:
        decide(batch([req("a", 12, 11, 8, 3)]))
        raise AssertionError("a reserved slot that was not written must be refused where the policy stops")
    except RoleError as e:
        assert "12 KV slots reserved but 11 written" in str(e)


def test_a_speculative_batch_is_not_checked_and_said_once():
    """M11.5: under speculative decoding kv_allocated_len runs ahead of kv_committed_len by the draft budget (33
    reserved, 17 written on MiniCPM5-2B with its DSpark drafter); 1.0 reported that as a disagreement at every step."""
    sc._SAID_SPECULATIVE = False
    spec = SimpleNamespace(is_none=lambda: False)
    assert sc.speculative(batch([], spec)) and not sc.speculative(batch([], SimpleNamespace(is_none=lambda: True)))
    assert not sc.speculative(batch([])) and sc.speculative(batch([], "EAGLE"))
    new, printed = decide(batch([req("a", 33, 17, 10, 6), req("b", 33, 17, 10, 6)], spec))
    assert len(new) == 1 and new[0].verdict is Verdict.UNKNOWN and not new[0].blocking, new
    assert "speculative decoding reserves draft slots" in new[0].note and "unknown at" in printed
    assert kv_contract.stats(sc.BOUNDARY)["skipped"] == 2 and kv_contract.stats(sc.BOUNDARY)["checks"] == 0
    new, printed = decide(batch([req("a", 41, 25, 10, 14)], spec))
    assert new == [] and printed == "", "said once per process"
    new, _ = decide(batch([req("a", 12, 12, 8, 3)]))          # a plain batch in the same process is still checked
    assert new == [] and kv_contract.stats(sc.BOUNDARY)["checks"] == 1


# --- the batches in flight, on a stand-in scheduler (issue #36) --------------------------------------------------

class Scheduler:
    """SGLang's scheduler, reduced: run_batch samples one token per request, process_batch_result appends it to the
    request's output_ids (as SGLang's result processing does)."""

    def run_batch(self, batch):
        return [100 + len(r.output_ids) for r in batch.reqs]

    def process_batch_result(self, batch, result):
        for r, t in zip(batch.reqs, result):
            r.output_ids.append(t)


def with_scheduler(fn):
    names = ("sglang", "sglang.srt", "sglang.srt.managers", "sglang.srt.managers.scheduler")
    mods = {n: types.ModuleType(n) for n in names}
    mods["sglang.srt.managers.scheduler"].Scheduler = Scheduler
    was = {n: sys.modules.get(n) for n in names}
    methods = Scheduler.run_batch, Scheduler.process_batch_result
    sys.modules.update(mods)
    try:
        assert sc.install_inflight() == 1 and sc.install_inflight() == 0
        return fn(Scheduler())
    finally:
        sc.uninstall()
        assert (Scheduler.run_batch, Scheduler.process_batch_result) == methods and not sc._INFLIGHT
        for n, m in was.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m


def run_loop(sched, plan, overlap):
    """SGLang's event loop over `plan` [(requests, "prefill" | "decode")]: each iteration prepares its batch (a decode
    batch takes one slot per request, then is checked, as prepare_for_decode is), launches it, and processes the last
    batch's result - after launching the next with the overlap scheduler, at once without it. The decisions."""
    core.set_mode("load")
    kv_contract.reset()
    n = len(load.LEDGER.decisions)
    last = None
    try:
        for reqs, kind in plan:
            b = batch(list(reqs))
            if kind == "decode":
                for r in reqs:
                    r.kv.kv_allocated_len += 1
                    r.kv.kv_committed_len += 1
                with redirect_stdout(io.StringIO()):
                    sc._decide(b)
            result = sched.run_batch(b)
            if not overlap:
                sched.process_batch_result(b, result)
            elif last is not None:
                sched.process_batch_result(*last)
            last = (batch(list(reqs)), result) if overlap else None
    finally:
        core.set_mode("off")
    return load.LEDGER.decisions[n:]


def test_a_request_others_join_is_counted_right():
    """Issue #36: r0 decodes; r1 arrives and its prefill runs as a batch of its own, during which the overlap
    scheduler processes r0's last decode result; then both decode. 2.1.3 expected one pending token for r0 as well
    and said `holds 12 KV slots but the sequence has 13 tokens` once per request that others joined."""
    def run(sched):
        r0, r1 = req("r0", 8, 8, 8, 0), req("r1", 5, 5, 5, 0)
        plan = [([r0], "prefill")] + [([r0], "decode")] * 3 + [([r1], "prefill")] + [([r0, r1], "decode")] * 4
        ds = run_loop(sched, plan, overlap=True)
        assert ds == [] and kv_contract.stats(sc.BOUNDARY)["checks"] == 11, ds
        assert kv_contract.stats(sc.BOUNDARY)["passed"]["kv_needed"] == 11
        return r0

    r0 = with_scheduler(run)
    assert len(r0.output_ids) == 7 and r0.kv.kv_allocated_len == 8 + 7, "7 decode steps, the last one in flight"


def test_without_overlap_nothing_is_pending():
    """--disable-overlap-schedule: each result is processed before the next step is prepared, so every output is in
    output_ids (2.1.3's one pending token for every request would have been a `broken` at every step)."""
    def run(sched):
        r0, r1 = req("r0", 8, 8, 8, 0), req("r1", 5, 5, 5, 0)
        plan = [([r0], "prefill")] + [([r0], "decode")] * 2 + [([r1], "prefill")] + [([r0, r1], "decode")] * 2
        ds = run_loop(sched, plan, overlap=False)
        assert ds == [] and kv_contract.stats(sc.BOUNDARY)["checks"] == 6, ds

    with_scheduler(run)


def test_a_lost_slot_is_still_caught_with_a_batch_in_flight():
    """The count of batches in flight makes the expected tokens exact, not looser: a step that takes no slot for a
    request of the last batch is a shortfall of one."""
    def run(sched):
        r0 = req("r0", 8, 8, 8, 0)
        run_loop(sched, [([r0], "prefill"), ([r0], "decode"), ([r0], "decode")], overlap=True)
        assert sc._INFLIGHT == {"r0": 1}, sc._INFLIGHT   # the last decode, launched and not processed
        r0.kv.kv_allocated_len -= 1
        r0.kv.kv_committed_len -= 1
        try:
            run_loop(sched, [([r0], "decode")], overlap=True)
            raise AssertionError("one slot short must be refused where the policy stops")
        except RoleError as e:
            assert "holds 10 KV slots but the sequence has 11 tokens" in str(e), e
        sc._INFLIGHT.clear()

    with_scheduler(run)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
