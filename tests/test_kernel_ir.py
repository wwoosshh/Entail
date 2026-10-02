"""Tests for kernel_ir (ROADMAP M19 L5.4b): the TTIR of vLLM 0.30's block FP8 matmul kernel, as Triton 3.7.1
compiled it on an RTX 4070 Ti (tests/data/kernel_ir), read for one launch each: the kernel at two tile configurations
is proven, the BLOCK_SIZE_K 256 configuration (a K tile over two 128-wide scale groups, read once) is a violation,
and the four consumer mutations of the L5.4a evaluation are not proven. numpy only. Run: python tests/test_kernel_ir.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import kernel_ir  # noqa: E402
from entail.kernel_ir import Tensor, check_launch, parse  # noqa: E402

DATA = os.path.join(HERE, "data", "kernel_ir")
N, K = 1536, 2560
NB = K // 128


def binding(M, alt_pair=5, out_stride=None):
    return {"A": Tensor("activation", 1, 2, (M, K), (K, 1), (1, 128)),
            "As": Tensor("activation_scale", 2, 1, (M, NB), (NB, 1), (1, 128)),
            "B": Tensor("weight", 3, 4, (N, K), (K, 1), (128, 128)),
            "Bs": Tensor("weight_scale", 4, 3, (N // 128, NB), (NB, 1), (128, 128)),
            "BsAlt": Tensor("weight_scale", 6, alt_pair, (N // 128, NB), (NB, 1), (128, 128)),
            "C": Tensor("output", 0, 0, (M, N), out_stride or (N, 1)),
            "Mut": Tensor("other")}


def launch(name, M, BM, b=None, **ints):
    vals = dict(M=M, N=N, K=K, group_n=128, group_k=128, stride_am=K, stride_bn=K, stride_cm=N, stride_As_m=NB,
                stride_Bs_n=NB, ROWS_FROM=8)
    vals.update(ints)
    with open(os.path.join(DATA, name + ".ttir"), encoding="utf-8") as f:
        ttir = f.read()
    return check_launch(ttir, b or binding(M), vals, (-(-M // BM) * (N // 128),))


def main():
    f = parse(open(os.path.join(DATA, "orig.ttir"), encoding="utf-8").read())
    assert f.name == "_w8a8_triton_block_scaled_mm" and [a[0] for a in f.args][:5] == ["%A", "%B", "%C", "%As", "%Bs"]
    print("ok parse:", len(f.args), "arguments")

    v = launch("orig", 64, 64)
    assert v.verdict == "proven" and v.terms == 20 and v.programs == 12, v
    v = launch("tile_0", 16, 16)
    assert v.verdict == "proven", v
    print("ok vLLM's kernel at tiles 64x128x128 and 16x128x128: proven,", v.terms, "terms")

    v = launch("tile_1", 256, 64)
    assert v.verdict == "violation" and "As" in v.why and v.example["groups_in_tile"] == [0, 1], v
    print("ok BLOCK_SIZE_K 256 over 128-wide groups: violation before any launch:", v.example)

    v = launch("interval_0", 64, 16)
    assert v.verdict == "unproven" and "Mut" in v.why, v
    v = launch("neighbor_0", 64, 16)
    assert v.verdict == "unproven" and "Mut" in v.why, v
    v = launch("alt_weight_0", 64, 16)
    assert v.verdict == "possible" and "paired with issue 5" in v.why, v
    v = launch("rows_0", 64, 16)
    assert v.verdict == "possible" and "As" in v.why, v
    print("ok the four consumer mutations: not proven (the scale read depends on data)")

    # the same alternative scale, issued as this weight's own (a pair the producer made): no violation from it
    v = launch("alt_weight_0", 64, 16, b=binding(64, alt_pair=3))
    assert v.verdict in ("proven", "possible", "unproven"), v
    # an output laid out otherwise than the kernel stores it
    v = launch("orig", 64, 64, b=binding(64, out_stride=(N + 8, 1)))
    assert v.verdict == "violation" and "stored" in v.why, v
    print("ok a store to another layout than the output's: violation")
    # a binding that does not say what A is: nothing claimed
    b = binding(64)
    b["A"] = Tensor("other")
    v = launch("orig", 64, 64, b=b)
    assert v.verdict == "unproven", v
    print("ok an operand without an issue: unproven")

    # the values kept apart (fast) and evaluated element by element (dense) give the same verdicts
    cases = [("orig", 64, 64, None), ("tile_0", 16, 16, None), ("tile_1", 256, 64, None), ("interval_0", 64, 16, None),
             ("neighbor_0", 64, 16, None), ("alt_weight_0", 64, 16, None), ("rows_0", 64, 16, None),
             ("orig", 64, 64, binding(64, out_stride=(N + 8, 1))), ("orig", 100, 64, None)]
    fast = [launch(n, M, BM, b=b) for n, M, BM, b in cases]
    kernel_ir.FAST = False
    try:
        dense = [launch(n, M, BM, b=b) for n, M, BM, b in cases]
    finally:
        kernel_ir.FAST = True
    for (n, M, _bm, _b), f, d in zip(cases, fast, dense):
        assert f.verdict == d.verdict, (n, M, f, d)
        assert not f.dense, (n, f)
    print("ok fast and dense evaluation agree on", len(cases), "launches;",
          f"orig: fast {fast[0].seconds:.4f} s, dense {dense[0].seconds:.4f} s")


if __name__ == "__main__":
    main()
