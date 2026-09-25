"""Tests for the SGLang block-FP8 tile adapter (ROADMAP M15.2; sglang#39626): what it reads from a config map, that
a map is decided once, and that the clamp writes into the engine's own map, on fakes - no SGLang, no GPU. The rule
is tested in test_tile_contract.py; the kernel end to end in testbed/m10_e3/sg39626.py.
Run: python tests/test_sglang_fp8_tile.py"""
import io
import os
import sys
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import core, load  # noqa: E402
from entail.adapters import sglang_fp8_tile  # noqa: E402
from entail.contracts import Verdict  # noqa: E402


def cfg(k, n=32):
    return dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=n, BLOCK_SIZE_K=k, GROUP_SIZE_M=1, num_warps=4, num_stages=4)


def test_read_choice_gives_the_block_and_every_entrys_tiles():
    configs = {16: cfg(32), 64: cfg(64, 64)}
    assert sglang_fp8_tile.read_choice(configs, [32, 32]) == (32, [(16, 32, 32), (64, 64, 64)])
    assert sglang_fp8_tile.read_choice(None, [32, 32]) is None       # the engine's default: tile == block
    assert sglang_fp8_tile.read_choice({}, [32, 32]) is None


def test_the_clamp_writes_into_the_engines_own_map():
    configs = {16: cfg(64)}
    sglang_fp8_tile.handles(configs, 16)["clamp_tile_k"](32)
    assert configs[16]["BLOCK_SIZE_K"] == 32


def test_a_map_with_a_tile_over_the_block_is_repaired_once_and_left_alone_after():
    sglang_fp8_tile.reset()
    configs = {16: cfg(64), 128: cfg(32)}     # M=16 entry carries a K tile over the block of 32
    n = len(load.LEDGER.decisions)
    core.set_mode("load")
    try:
        with redirect_stdout(io.StringIO()):
            sglang_fp8_tile._decide(configs, [32, 32], "N=32,K=64")
            sglang_fp8_tile._decide(configs, [32, 32], "N=32,K=64")   # the same map: decided once
    finally:
        core.set_mode("off")
    d = load.LEDGER.decisions[n:]
    assert len(d) == 1 and d[0].verdict is Verdict.RESOLVED and d[0].target == 32, d
    assert configs[16]["BLOCK_SIZE_K"] == 32 and configs[128]["BLOCK_SIZE_K"] == 32
    s = sglang_fp8_tile.stats()
    assert s["checks"] == 2, s      # both entries checked, one repaired, one passed
    sglang_fp8_tile.reset()


def test_a_map_whose_tiles_all_divide_is_quiet():
    sglang_fp8_tile.reset()
    configs = {16: cfg(128), 64: cfg(64), 256: cfg(32)}
    n = len(load.LEDGER.decisions)
    core.set_mode("load")
    try:
        with redirect_stdout(io.StringIO()):
            sglang_fp8_tile._decide(configs, [128, 128], "N=128,K=256")
    finally:
        core.set_mode("off")
    assert load.LEDGER.decisions[n:] == [] and sglang_fp8_tile.stats()["checks"] == 3
    sglang_fp8_tile.reset()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
