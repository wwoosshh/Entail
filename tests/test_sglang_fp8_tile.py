"""Tests for the SGLang block-FP8 tile adapter (ROADMAP M15.2; sglang#39626): what it reads from a dense config map
and a fused-MoE config, that each is decided once by identity, and that the clamp writes into the engine's own
object, on fakes - no SGLang, no GPU. The rule is tested in test_tile_contract.py; the kernels end to end in
testbed/m10_e3/sg39626.py and testbed/m15/moe_tile.py. Run: python tests/test_sglang_fp8_tile.py"""
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


def in_load(work):
    n = len(load.LEDGER.decisions)
    core.set_mode("load")
    try:
        with redirect_stdout(io.StringIO()):
            work()
    finally:
        core.set_mode("off")
    return load.LEDGER.decisions[n:]


def test_read_choice_gives_the_block_and_every_entrys_tiles():
    configs = {16: cfg(32), 64: cfg(64, 64)}
    assert sglang_fp8_tile.read_choice(configs, [32, 32]) == (32, [(16, 32, 32), (64, 64, 64)])
    assert sglang_fp8_tile.read_choice(None, [32, 32]) is None       # the engine's default: tile == block
    assert sglang_fp8_tile.read_choice({}, [32, 32]) is None


def test_the_clamp_writes_into_the_engines_own_map():
    configs = {16: cfg(64)}
    sglang_fp8_tile.handles(configs, 16)["clamp_tile_k"](32)
    assert configs[16]["BLOCK_SIZE_K"] == 32


def test_a_dense_map_with_a_tile_over_the_block_is_repaired_once_and_remembered_by_identity():
    sglang_fp8_tile.reset()
    configs = {16: cfg(64), 128: cfg(32)}     # M=16 entry carries a K tile over the block of 32
    d = in_load(lambda: (sglang_fp8_tile._decide(configs, [32, 32], "N=32,K=64"),
                         sglang_fp8_tile._decide(configs, [32, 32], "N=32,K=64")))
    # one decision: the second pass finds the map already clamped (the wrapper would not even call it again)
    assert len(d) == 1 and d[0].verdict is Verdict.RESOLVED and d[0].target == 32, d
    assert configs[16]["BLOCK_SIZE_K"] == 32 and configs[128]["BLOCK_SIZE_K"] == 32
    assert id(configs) in sglang_fp8_tile._CHECKED and sglang_fp8_tile._CHECKED[id(configs)] is configs
    sglang_fp8_tile.reset()


def test_a_dense_map_whose_tiles_all_divide_is_quiet():
    sglang_fp8_tile.reset()
    configs = {16: cfg(128), 64: cfg(64), 256: cfg(32)}
    d = in_load(lambda: sglang_fp8_tile._decide(configs, [128, 128], "N=128,K=256"))
    assert d == [] and sglang_fp8_tile.stats()["dense"]["checks"] == 3
    sglang_fp8_tile.reset()


def test_a_moe_config_with_a_tile_over_the_block_is_clamped_and_the_down_config_too():
    """The shipped H100 file (E=512, N=256, block [128, 128]) carries BLOCK_SIZE_K=256 for M=64..512."""
    sglang_fp8_tile.reset()
    up, down = cfg(256, 128), cfg(256, 128)
    d = in_load(lambda: sglang_fp8_tile._decide_moe(up, down, [128, 128], "E=512,N=256,M=64"))
    # the same content is decided (recorded) once; both copies are clamped, since each reaches the kernel
    assert [x.verdict for x in d] == [Verdict.RESOLVED] and d[0].target == 128, d
    assert up["BLOCK_SIZE_K"] == 128 and down["BLOCK_SIZE_K"] == 128
    d2 = in_load(lambda: sglang_fp8_tile._decide_moe(up, down, [128, 128], "E=512,N=256,M=64"))
    assert d2 == [], "a config already decided is not decided again"
    sglang_fp8_tile.reset()


def test_a_moe_config_that_divides_is_quiet():
    sglang_fp8_tile.reset()
    d = in_load(lambda: sglang_fp8_tile._decide_moe(cfg(64, 128), None, [128, 128], "E=8,N=1024,M=1"))
    assert d == [] and sglang_fp8_tile.stats()["moe"]["checks"] == 1
    sglang_fp8_tile.reset()



def test_a_down_config_copied_on_every_call_is_decided_once_by_content_and_the_memo_stays_bounded():
    """The engine returns a fresh copy of the down config per call (dict(**down_config)); deciding by identity would
    run every call and hold every copy (M15.4 review)."""
    sglang_fp8_tile.reset()
    up = cfg(128, 128)
    d = in_load(lambda: [sglang_fp8_tile._decide_moe(up, dict(cfg(256, 128)), [128, 128], "E=8,N=256,M=64")
                         for _ in range(5)])
    assert len(d) == 1 and d[0].verdict is Verdict.RESOLVED, d       # five copies, one decision
    assert len(sglang_fp8_tile._CHECKED) <= 4, sglang_fp8_tile._CHECKED   # content keys, not copies
    sglang_fp8_tile.reset()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
