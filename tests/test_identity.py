"""Tests for the identity container contract in the core (ROADMAP M14; codebook v2 I): the rule on one store's
stored identities against the identities its contents give now, the repair, and the fact. Pure Python: no torch, no
engine (the vLLM adapter is tested where vLLM is; the mechanism end to end is testbed/r4/identity_probe.py).
Run: python tests/test_identity.py"""
import io
import os
import sys
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
# The default policy reports a stale store and goes on; these check the verdict, so read it off the ledger.
from entail import core, identity_contract, load  # noqa: E402
from entail.contracts import RULES, Verdict  # noqa: E402
from entail.facts import Identity  # noqa: E402

B, C = "container:test.store", "test.prefix_cache"


def raises(fn, text):
    try:
        fn()
    except (ValueError, TypeError) as e:
        assert text in str(e), (text, str(e))
    else:
        raise AssertionError(f"expected an error containing {text!r}")


def decide_in_load(stored, fresh, recompute):
    """Run check in load mode; return the decisions it recorded (quietly)."""
    n = len(load.LEDGER.decisions)
    core.set_mode("load")
    try:
        with redirect_stdout(io.StringIO()):
            identity_contract.check(B, C, "where", stored, fresh, recompute, covers=4)
    finally:
        core.set_mode("off")
    return load.LEDGER.decisions[n:]


def test_the_fact_takes_a_kind_from_a_closed_set_and_a_non_empty_key():
    Identity("kv_block", 0, "ab", covers=4)
    raises(lambda: Identity("page", 0, "ab"), "Identity.of: unknown value")
    raises(lambda: Identity("kv_block", 0, ""), "Identity.key: expected a non-empty string")
    raises(lambda: Identity("kv_block", -1, "ab"), "Identity.index")


def test_first_stale_finds_the_first_key_that_no_longer_stands():
    assert identity_contract.first_stale(["a", "b", "c"], ["a", "b", "c"]) is None
    assert identity_contract.first_stale(["a", "x", "c"], ["a", "b", "c"]) == 1
    # a stored key with no contents left to back it (the store kept more blocks than the tokens fill)
    assert identity_contract.first_stale(["a", "b"], ["a"]) == 1
    assert identity_contract.first_stale(["a"], ["a", "b"]) is None   # more contents than keys is fine


def test_a_store_whose_keys_all_stand_passes_and_is_quiet():
    called = []
    d = decide_in_load(["a", "b", "c"], ["a", "b", "c"], lambda i: called.append(i))
    assert d == [] and called == []
    assert identity_contract.stats(B)["checks"] >= 1
    identity_contract.reset(B)


def test_a_stale_key_is_resolved_by_recompute_and_the_repair_is_carried_out():
    called = []
    d = decide_in_load(["a", "x", "c"], ["a", "b", "c"], lambda i: called.append(i) or "done")
    assert len(d) == 1 and d[0].verdict is Verdict.RESOLVED, d
    assert d[0].rule == RULES["resolved"] and d[0].handle == "identity_recompute"
    assert d[0].target == 1 and called == [1]      # the repair was handed the first stale index and ran
    identity_contract.reset(B)


def test_without_a_repair_a_stale_key_is_broken_and_the_run_goes_on():
    def no_handle(_i):
        raise AssertionError("recompute must not be called when the policy repairs nothing")
    core.set_mode("load")
    core.set_policy("refuse")                     # repair nothing; report the mismatch (on_mismatch is core's, not env)
    try:
        n = len(load.LEDGER.decisions)
        with redirect_stdout(io.StringIO()):
            identity_contract.check(B, C, "where", ["a", "x"], ["a", "b"], no_handle, covers=4)
        d = load.LEDGER.decisions[n:]
    finally:
        core.set_policy("resolve")
        core.set_mode("off")
    assert len(d) == 1 and d[0].verdict is Verdict.BROKEN and d[0].rule == RULES["policy_refuses"], d
    identity_contract.reset(B)


def test_a_stale_key_stops_the_run_where_the_policy_stops():
    from entail.core import RoleError
    core.set_mode("load")
    core.set_policy("refuse")
    os.environ["ENTAIL_ON_BROKEN"] = "stop"
    try:
        with redirect_stdout(io.StringIO()):
            identity_contract.check(B, C, "where", ["a", "x"], ["a", "b"], lambda i: i, covers=4)
    except RoleError as e:
        assert "identity" in str(e).lower() or "stale" in str(e).lower(), str(e)
    else:
        raise AssertionError("expected the run to stop under ENTAIL_ON_BROKEN=stop")
    finally:
        os.environ.pop("ENTAIL_ON_BROKEN", None)
        core.set_policy("resolve")
        core.set_mode("off")
    identity_contract.reset(B)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
