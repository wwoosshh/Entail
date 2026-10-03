"""Tests for kernel_types (ROADMAP M19 L6): the one rule on the same IR as test_kernel_ir - vLLM 0.30's block FP8
matmul at two tile configurations (proven), the K-tile-over-two-groups configuration (violation), the four consumer
mutations (not proven), and the counterexamples edited from the kernel's IR (none proven). No contract of the matmul
is written here: the meanings of the five tensors and the rule decide everything. numpy only.
Run: python tests/test_kernel_types.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from entail import kernel_types as KT  # noqa: E402
from entail.kernel_types import Axis, Meaning, block_fp8_matmul, check_launch  # noqa: E402
import test_kernel_ir as old  # noqa: E402  (the fixtures and the IR-level counterexamples)

DATA = old.DATA
N, K, NB = old.N, old.K, old.NB


def meanings(M, alt_pair=5, out_stride=None):
    ms = block_fp8_matmul(M, N, K, out_stride=out_stride)
    ms["BsAlt"] = Meaning((Axis("feature", N, 128), Axis("hidden", K, 128)), (N // 128, NB), (NB, 1), "scale", 6,
                          alt_pair)
    ms["Mut"] = Meaning((Axis(None, 4096),), (4096,), (1,), "other")
    return ms


def launch(name, M, BM, ms=None, ttir=None, **ints):
    vals = dict(M=M, N=N, K=K, group_n=128, group_k=128, stride_am=K, stride_bn=K, stride_cm=N, stride_As_m=NB,
                stride_Bs_n=NB, ROWS_FROM=8)
    vals.update(ints)
    if ttir is None:
        with open(os.path.join(DATA, name + ".ttir"), encoding="utf-8") as f:
            ttir = f.read()
    return check_launch(ttir, ms or meanings(M), vals, (-(-M // BM) * (N // 128),))


def main():
    v = launch("orig", 64, 64)
    assert v.verdict == "proven" and v.programs == 12, v
    v = launch("tile_0", 16, 16)
    assert v.verdict == "proven", v
    print("ok vLLM's kernel at tiles 64x128x128 and 16x128x128: proven by the one rule,", v.checks, "pairings")

    v = launch("tile_1", 256, 64)
    assert v.verdict == "violation" and "hidden" in v.why and v.example["groups_in_tile"] == [0, 1], v
    print("ok BLOCK_SIZE_K 256 over 128-wide groups: violation:", v.why[:120])

    v = launch("interval_0", 64, 16)
    assert v.verdict != "proven", v
    v2 = launch("neighbor_0", 64, 16)
    assert v2.verdict != "proven", v2
    v3 = launch("alt_weight_0", 64, 16)
    assert v3.verdict == "possible" and "issue 6" in v3.why, v3
    v4 = launch("rows_0", 64, 16)
    assert v4.verdict == "possible", v4
    print("ok the four consumer mutations: not proven:", v.verdict, v2.verdict, v3.verdict, v4.verdict)

    v = launch("alt_weight_0", 64, 16, ms=meanings(64, alt_pair=3))
    assert v.verdict in ("proven", "possible", "unproven"), v
    v = launch("orig", 64, 64, ms=meanings(64, out_stride=(N + 8, 1)))
    assert v.verdict == "violation" and "a store" in v.why, v
    print("ok a store to another layout than the output's: violation")
    ms = meanings(64)
    ms["A"] = Meaning((Axis(None, 64), Axis(None, K)), (64, K), (K, 1), "value")
    v = launch("orig", 64, 64, ms=ms)
    assert v.verdict == "unproven", v
    print("ok an operand whose axes mean nothing: unproven")

    src = open(os.path.join(DATA, "orig.ttir"), encoding="utf-8").read()
    for name, ttir, _want in old.counterexamples(src):
        v = launch("orig", 64, 64, ttir=ttir)
        assert v.verdict != "proven", (name, v.verdict, v.why)
        print(f"ok {name}: {v.verdict} ({v.why[:100]})")
    for name, ttir, _v3, now in old.address_counterexamples(src):
        v = launch("orig", 64, 64, ttir=ttir)
        assert v.verdict == now, (name, v.verdict, v.why)
        print(f"ok {name}: {v.verdict} ({v.why[:100]})")
    v = launch("orig", 100, 64)
    assert v.verdict == "proven", v
    v = launch("orig", 64, 64, stride_am=1 << 26)
    assert v.verdict == "violation" and "outside" in v.why, v
    print("ok a partial row tile: proven; wrapped i32 offsets: violation")

    # the output's axes unnamed: the rule infers them from the stores
    ms = meanings(64)
    ms["C"] = Meaning((Axis(None, 64), Axis(None, N)), (64, N), (N, 1), "output")
    v = launch("orig", 64, 64, ms=ms)
    assert v.verdict == "proven" and v.inferred["axis_0"] == "token" and v.inferred["axis_1"] == "feature", v
    assert v.inferred["pending_scales"] == [] and "hidden" in v.inferred["sums"], v
    print("ok an output whose axes were not named: inferred as [token, feature], summed over hidden")


if __name__ == "__main__":
    main()
