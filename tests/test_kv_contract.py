"""Tests for the KV container contract in the core (ROADMAP M5.1): the rules on one extent, the books of one cache,
the check after a request, and what happens when entail itself fails. Pure Python: no torch, no engine.
Run: python tests/test_kv_contract.py"""
import io
import os
import sys
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
# These tests check what stops, so they run under the policy that stopped before M5.4 (ENTAIL_ON_BROKEN=stop,
# unknown meaning-changing facts required); the default, which reports and goes on, is tested in test_report.py.
os.environ["ENTAIL_ON_BROKEN"] = "stop"
os.environ["ENTAIL_UNKNOWN"] = "require"
from entail import core, kv_contract, load  # noqa: E402
from entail.contracts import RULES, Verdict  # noqa: E402
from entail.core import RoleError  # noqa: E402
from entail.kv_contract import KvExtent, check_extent  # noqa: E402

B, C = "container:test.update", "test.kv_cache"


class Cache:   # an owner of books: any object that takes a weak reference
    pass


def refused(fn, *args, text=""):
    """Run fn in load mode; return the refused decision it recorded (and check it stopped the run)."""
    n = len(load.LEDGER.decisions)
    core.set_mode("load")
    try:
        with redirect_stdout(io.StringIO()):
            fn(*args)
    except RoleError as e:
        assert text in str(e), (text, str(e))
    else:
        raise AssertionError("expected the run to stop")
    finally:
        core.set_mode("off")
    new = load.LEDGER.decisions[n:]
    assert len(new) >= 1 and all(d.verdict is Verdict.REFUSED and d.blocking for d in new), new
    return new[0]


def caught(extent, where="test", expect=""):
    try:
        check_extent(extent, where)
    except RoleError as e:
        assert expect in str(e), (expect, str(e))
        return True
    return False


def test_a_healthy_extent_is_quiet():
    made = check_extent(KvExtent(held=8, needed=8, written=8, previous=7), "test")
    assert made == 3, made


def test_reserved_and_written_must_agree():
    assert caught(KvExtent(held=8, written=9), expect="reserved but 9 written")


def test_held_must_match_the_tokens():
    assert caught(KvExtent(held=7, needed=8), expect="holds 7 KV slots but the sequence has 8")


def test_nothing_may_shrink_quietly():
    assert caught(KvExtent(held=6, previous=7), expect="dropped 1 token")


def test_a_window_is_allowed_to_hold_less():
    # a capped extent skips both the "matches the tokens" and the "never shrinks" comparison
    assert check_extent(KvExtent(held=4, needed=40, previous=4, window=4), "test") == 0
    # ...but the reserved/written pair still has to agree even under a window
    assert caught(KvExtent(held=4, needed=40, written=5, window=4), expect="reserved but 5 written")


def test_granular_allocation():
    """vLLM hands out blocks, so holding a little more than needed is right and holding less is not."""
    assert check_extent(KvExtent(held=32, needed=17, granularity=16), "test") == 1
    assert caught(KvExtent(held=16, needed=17, granularity=16), expect="not one allocation unit")
    assert caught(KvExtent(held=64, needed=17, granularity=16), expect="not one allocation unit")


def test_partial_information_is_still_checked():
    """An engine that only exposes one number is not a reason to check nothing."""
    assert check_extent(KvExtent(held=8), "test") == 0
    assert check_extent(KvExtent(held=8, needed=8), "test") == 1



# --- the core's decisions (M5.1) -------------------------------------------------------------------------------

def test_values_are_checked_like_every_fact():
    for kw, text in (({"held": -1}, "KvExtent.held: expected >= 0"), ({"held": 8, "window": 0}, "KvExtent.window"),
                     ({"held": True}, "KvExtent.held: expected an int"), ({"held": None}, "KvExtent.held: required")):
        try:
            KvExtent(**kw)
        except ValueError as e:
            assert text in str(e), (text, str(e))
        else:
            raise AssertionError(kw)


def test_a_broken_rule_is_a_refused_decision_in_the_ledger():
    kv_contract.reset()
    d = refused(kv_contract.check, B, C, "request 7", KvExtent(held=7, needed=8), text="holds 7 KV slots")
    assert d.rule == RULES["kv_needed"] and d.name == "KvExtent" and d.contract.boundary == B, d
    assert d.declared.value == KvExtent(held=8) and d.chosen.value == KvExtent(held=7, needed=8), d
    assert "the slots its tokens need" in d.declared.source.where and d.note.startswith("request 7: holds 7")
    assert kv_contract.stats(B)["refused"] == 1


def test_passes_are_counted_not_recorded():
    kv_contract.reset()
    n = len(load.LEDGER.decisions)
    assert kv_contract.check(B, C, "w", KvExtent(held=8, needed=8, written=8, previous=7)) == 3
    assert len(load.LEDGER.decisions) == n
    assert kv_contract.stats(B)["passed"] == {"kv_written": 1, "kv_needed": 1, "kv_shrank": 1}
    kv_contract.skipped(B, 2)
    assert kv_contract.stats(B)["skipped"] == 2 and kv_contract.stats(B)["checks"] == 1


def test_a_layer_that_lost_a_token_between_updates():
    kv_contract.reset()
    cache = Cache()
    kv_contract.grew(B, C, cache, "cache", 0, 0, 5, 5)          # prefill: 5 tokens
    d = refused(kv_contract.grew, B, C, cache, "cache", 0, 4, 5, 1, text="dropped 1 token")   # held 4 when given 1
    assert d.rule == RULES["kv_shrank"], d


def test_the_layers_of_one_cache_agree_and_other_caches_do_not_count():
    """Before M5.1 the agreement compared the layers of every cache the process had seen: the second generate in a
    process was refused (reproduced on the old adapter). The books are per cache now."""
    kv_contract.reset()
    first, second = Cache(), Cache()
    for layer in (0, 1):
        kv_contract.grew(B, C, first, "first", layer, 0, 8, 8)
    for layer in (0, 1):                                          # a new cache at another length: quiet
        kv_contract.grew(B, C, second, "second", layer, 0, 3, 3)
    assert kv_contract.stats(B)["refused"] == 0 and kv_contract.stats(B)["passed"]["kv_layers"] == 4
    uneven = Cache()
    kv_contract.grew(B, C, uneven, "uneven", 0, 0, 6, 6)
    d = refused(kv_contract.grew, B, C, uneven, "uneven", 1, 0, 4, 4, text="disagree")
    assert d.rule == RULES["kv_layers"], d


def test_a_window_takes_no_part_in_the_agreement():
    kv_contract.reset()
    cache = Cache()
    kv_contract.grew(B, C, cache, "c", 0, 0, 40, 40)
    kv_contract.grew(B, C, cache, "c", 1, 0, 4, 40, window=4)   # capped on purpose
    assert kv_contract.stats(B)["refused"] == 0


def test_after_a_request_every_layer_holds_what_it_wrote():
    kv_contract.reset()
    assert kv_contract.request(B, C, "after", {0: (5, None), 1: (4, 4)}, 5) == 2   # a window of 4 may hold 4
    d = refused(kv_contract.request, B, C, "after", {0: (5, None), 1: (4, None)}, 5,
                text="1 layer(s) do not hold the number of tokens the request wrote")
    assert d.rule == RULES["kv_request"] and d.declared.value == KvExtent(held=5), d


def test_entail_failing_never_breaks_the_engine_but_stops_in_debug():
    kv_contract.reset()
    n = len(load.LEDGER.decisions)
    core.set_mode("load")
    try:
        with redirect_stdout(io.StringIO()):
            assert kv_contract.guarded(B, C, lambda: 1 / 0) is None
            assert kv_contract.guarded(B, C, lambda: 1 / 0) is None   # reported once; the boundary is left alone
    finally:
        core.set_mode("off")
    new = load.LEDGER.decisions[n:]
    assert len(new) == 1 and new[0].verdict is Verdict.UNKNOWN and "ZeroDivisionError" in new[0].note, new
    kv_contract.reset()
    core.set_mode("debug")
    try:
        with redirect_stdout(io.StringIO()):
            kv_contract.guarded(B, C, lambda: 1 / 0)
    except RoleError:
        pass
    else:
        raise AssertionError("debug mode stops where entail could not check")
    finally:
        core.set_mode("off")
    assert kv_contract.guarded(B, C, lambda: 1 / 0) is None   # and it is left alone afterwards


def test_nothing_is_captured_outside_a_capture():
    assert kv_contract.inside_capture() is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
