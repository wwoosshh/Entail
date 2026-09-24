"""Tests for the SGLang KV adapter's reading and deciding without SGLang (ROADMAP M5.1, M11.5): what it reads off a
decode batch, that a healthy batch is quiet, that a lost slot is caught, and that a speculative batch is not
checked but said so once. Run: python tests/test_sglang_cache_contract.py"""
import io
import os
import sys
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
    got = sc.read_choice(batch([req("a", 12, 12, 8, 3), req("b", 40, 40, 30, 9, swa=5)]))
    assert got == [("sglang request 0 (a)", 12, 12, 12, False), ("sglang request 1 (b)", 40, 40, 40, True)]
    assert sc.read_choice(SimpleNamespace(reqs=[SimpleNamespace(rid="c")])) == [("sglang request 0 (c)", None, None, 1, False)]


def test_a_healthy_batch_is_quiet_and_a_lost_slot_is_caught():
    new, printed = decide(batch([req("a", 12, 12, 8, 3)]))
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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
