"""Tests for the shared KV extent contract. Run: python tests/test_kv_contract.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail.core import RoleError  # noqa: E402
from entail.kv_contract import KvExtent, check_extent  # noqa: E402


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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
