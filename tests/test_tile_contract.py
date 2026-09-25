"""Tests for the kernel-tile container contract in the core (ROADMAP M15.2; sglang#39626): the rule on one config's
K tile against the quantization block, the repair, and the fact. Pure Python: no torch, no engine.
Run: python tests/test_tile_contract.py"""
import io
import os
import sys
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import core, load, tile_contract  # noqa: E402
from entail.contracts import RULES, Verdict  # noqa: E402
from entail.facts import KernelConfig  # noqa: E402

B, C = "load:test.kernel_config", "test.block_fp8_matmul"


def raises(fn, text):
    try:
        fn()
    except (ValueError, TypeError) as e:
        assert text in str(e), (text, str(e))
    else:
        raise AssertionError(f"expected an error containing {text!r}")


def decide_in_load(block_k, tile_k, clamp):
    n = len(load.LEDGER.decisions)
    core.set_mode("load")
    try:
        with redirect_stdout(io.StringIO()):
            tile_contract.check(B, C, "where", block_k, tile_k, clamp)
    finally:
        core.set_mode("off")
    return load.LEDGER.decisions[n:]


def test_the_fact_needs_a_positive_tile():
    KernelConfig(tile_k=32)
    KernelConfig(tile_k=32, tile_n=64)
    raises(lambda: KernelConfig(tile_k=0), "KernelConfig.tile_k")
    raises(lambda: KernelConfig(tile_k=32, tile_n=0), "KernelConfig.tile_n")


def test_divides_is_the_kernels_own_scale_stepping_condition():
    assert tile_contract.divides(128, 128) and tile_contract.divides(128, 64) and tile_contract.divides(128, 32)
    assert not tile_contract.divides(32, 64)      # the tile over the block (sglang#39626)
    assert not tile_contract.divides(128, 48)     # smaller but not a divisor
    assert not tile_contract.divides(32, 0)


def test_a_tile_that_divides_passes_and_is_quiet():
    called = []
    assert decide_in_load(128, 64, lambda t: called.append(t)) == [] and called == []
    assert tile_contract.stats(B)["checks"] >= 1
    tile_contract.reset(B)


def test_a_tile_over_the_block_is_resolved_by_clamping_it_to_the_block():
    called = []
    d = decide_in_load(32, 64, lambda t: called.append(t) or t)
    assert len(d) == 1 and d[0].verdict is Verdict.RESOLVED and d[0].handle == "clamp_tile_k", d
    assert d[0].rule == RULES["resolved"] and d[0].target == 32 and called == [32]
    tile_contract.reset(B)


def test_without_a_repair_it_is_broken_and_the_run_goes_on():
    def no_handle(_t):
        raise AssertionError("clamp must not run when the policy repairs nothing")
    core.set_mode("load")
    core.set_policy("refuse")
    try:
        n = len(load.LEDGER.decisions)
        with redirect_stdout(io.StringIO()):
            tile_contract.check(B, C, "where", 32, 64, no_handle)
        d = load.LEDGER.decisions[n:]
    finally:
        core.set_policy("resolve")
        core.set_mode("off")
    assert len(d) == 1 and d[0].verdict is Verdict.BROKEN and d[0].rule == RULES["policy_refuses"], d
    tile_contract.reset(B)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
