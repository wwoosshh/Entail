"""Tests for kernel_ir (ROADMAP M19 L5.4b, L5.4c): the TTIR of vLLM 0.30's block FP8 matmul kernel, as Triton 3.7.1
compiled it on an RTX 4070 Ti (tests/data/kernel_ir), read for one launch each: the kernel at two tile configurations
is proven, the BLOCK_SIZE_K 256 configuration (a K tile over two 128-wide scale groups, read once) is a violation,
and the four consumer mutations of the L5.4a evaluation are not proven. Since L5.4c, "proven" is the whole contract
(every output element stored once, the sum over all K): kernels made from vLLM's by editing its IR that store nothing,
sum one K group of twenty, store half the rows and the like are not proven (counterexamples below). numpy only.
Run: python tests/test_kernel_ir.py
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


def launch(name, M, BM, b=None, ttir=None, **ints):
    vals = dict(M=M, N=N, K=K, group_n=128, group_k=128, stride_am=K, stride_bn=K, stride_cm=N, stride_As_m=NB,
                stride_Bs_n=NB, ROWS_FROM=8)
    vals.update(ints)
    if ttir is None:
        with open(os.path.join(DATA, name + ".ttir"), encoding="utf-8") as f:
            ttir = f.read()
    return check_launch(ttir, b or binding(M), vals, (-(-M // BM) * (N // 128),))


def edit(src, *pairs):
    """orig.ttir with lines replaced (each `old` must occur once): the IR-level counterexamples below."""
    for old, new in pairs:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    return src


def counterexamples(src):
    """(name, IR, expected verdict): kernels that do not compute the contract - or compute it with something this
    module does not model - made from vLLM's kernel by editing its IR. Until 2026-10-03 (L5.4c) the first four were
    "proven": only the scale reads of the terms found were checked."""
    loop = "scf.for %k = %c0_i32 to %1 step"
    addf = "%accumulator_79 = arith.addf %accumulator_59, %accumulator_78 : tensor<64x128xf32>"
    trunc = "%c = arith.truncf %accumulator#2 : tensor<64x128xf32> to tensor<64x128xbf16>"
    return [
        ("no store", "\n".join(x for x in src.splitlines() if "tt.store" not in x), "violation"),
        ("the loop runs over the first of 20 K groups", edit(src, (loop, "scf.for %k = %c0_i32 to %c1_i32 step")),
         "violation"),
        ("each iteration overwrites the sum (the last group remains)",
         edit(src, (addf, "%accumulator_79 = arith.addf %cst_3, %accumulator_78 : tensor<64x128xf32>")),
         "violation"),
        ("rows from 32 on are not stored",
         edit(src, ("%c_mask = tt.splat %M : i32 -> tensor<64x1xi32>",
                    "%c_mask = tt.splat %c32_i32 : i32 -> tensor<64x1xi32>")), "violation"),
        ("each term is added twice",
         edit(src, (addf, addf + "\n      %acc_twice = arith.addf %accumulator_79, %accumulator_78 : tensor<64x128xf32>"),
              ("scf.yield %a_ptrs_80, %b_ptrs_81, %accumulator_79", "scf.yield %a_ptrs_80, %b_ptrs_81, %acc_twice")),
         "violation"),
        ("each term is also multiplied by 2",
         edit(src, ("%cst_3 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>",
                    "%cst_3 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>\n"
                    "    %two = arith.constant dense<2.000000e+00> : tensor<64x128xf32>"),
              (addf, "%twice_t = arith.mulf %accumulator_78, %two : tensor<64x128xf32>\n"
                     "      %accumulator_79 = arith.addf %accumulator_59, %twice_t : tensor<64x128xf32>")),
         "violation"),
        ("the kernel also writes the activation scales back",
         edit(src, ("%b_s_71 = tt.load %b_s_70 : tensor<128x!tt.ptr<f32>>",
                    "%b_s_71 = tt.load %b_s_70 : tensor<128x!tt.ptr<f32>>\n"
                    "      tt.store %a_s_68, %a_s_69 : tensor<64x!tt.ptr<f32>>")), "violation"),
        ("the sum is stored negated", edit(src, (trunc, "%neg = arith.subf %cst_3, %accumulator#2 : "
                                                        "tensor<64x128xf32>\n    %c = arith.truncf %neg : "
                                                        "tensor<64x128xf32> to tensor<64x128xbf16>")), "unproven"),
        ("the sum is stored doubled",
         edit(src, ("%cst_3 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>",
                    "%cst_3 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>\n"
                    "    %two = arith.constant dense<2.000000e+00> : tensor<64x128xf32>"),
              (trunc, "%dbl = arith.mulf %accumulator#2, %two : tensor<64x128xf32>\n"
                      "    %c = arith.truncf %dbl : tensor<64x128xf32> to tensor<64x128xbf16>")), "unproven"),
        ("the dot accumulates into the running sum, which the scales then multiply",
         edit(src, ("tt.dot %a_64, %b_67, %cst_3,", "tt.dot %a_64, %b_67, %accumulator_59,")), "unproven"),
        ("masked-out activation lanes load 1.0",
         edit(src, ("%cst_0 = arith.constant dense<0.000000e+00> : tensor<64x128xf8E4M3FN>",
                    "%cst_0 = arith.constant dense<1.000000e+00> : tensor<64x128xf8E4M3FN>")), "unproven"),
        ("the activation is masked one k short of the weight",
         edit(src, ("%a_61 = tt.splat %a_60 : i32 -> tensor<1x128xi32>",
                    "%a_60m = arith.subi %a_60, %c1_i32 : i32\n      %a_61 = tt.splat %a_60m : i32 -> "
                    "tensor<1x128xi32>")), "unproven"),
        ("one more iteration than K needs (fully masked): harmless",
         edit(src, ("%1 = arith.divsi %0, %c128_i32 : i32", "%1a = arith.divsi %0, %c128_i32 : i32\n"
                                                             "    %1 = arith.addi %1a, %c1_i32 : i32")),
         "proven"),
    ]


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

    # the whole contract (L5.4c): kernels that do not compute it are not proven
    src = open(os.path.join(DATA, "orig.ttir"), encoding="utf-8").read()
    bad = counterexamples(src)
    for name, ttir, want in bad:
        v = launch("orig", 64, 64, ttir=ttir)
        assert v.verdict == want, (name, v.verdict, v.why)
        print(f"ok {name}: {v.verdict} ({v.why[:110]})")
    v = launch("orig", 100, 64)
    assert v.verdict == "proven", v                       # a partial last row tile, masked at the store
    v = launch("orig", 64, 64, stride_am=1 << 26)          # 63 * 2^26 > 2^31: the IR's i32 offsets wrap
    assert v.verdict == "unproven" and "i32" in v.why, v
    b = binding(64)
    b["Mut"] = Tensor("output", 0, 0, (64, N), (N, 1))     # two tensors could be the output: nothing is claimed
    assert launch("orig", 64, 64, b=b).verdict == "unproven"
    print("ok a partial row tile: proven; i32 overflow and two outputs: unproven")

    # the values kept apart (fast) and evaluated element by element (dense) give the same verdicts
    cases = [("orig", 64, 64, None, None), ("tile_0", 16, 16, None, None), ("tile_1", 256, 64, None, None),
             ("interval_0", 64, 16, None, None), ("neighbor_0", 64, 16, None, None),
             ("alt_weight_0", 64, 16, None, None), ("rows_0", 64, 16, None, None),
             ("orig", 64, 64, binding(64, out_stride=(N + 8, 1)), None), ("orig", 100, 64, None, None)] + \
        [("orig", 64, 64, None, ttir) for _n, ttir, _w in bad]
    fast = [launch(n, M, BM, b=b, ttir=t) for n, M, BM, b, t in cases]
    kernel_ir.FAST = False
    try:
        dense = [launch(n, M, BM, b=b, ttir=t) for n, M, BM, b, t in cases]
    finally:
        kernel_ir.FAST = True
    for (n, M, _bm, _b, _t), f, d in zip(cases, fast, dense):
        assert f.verdict == d.verdict, (n, M, f, d)
        assert not f.dense, (n, f)
    print("ok fast and dense evaluation agree on", len(cases), "launches;",
          f"orig: fast {fast[0].seconds:.4f} s, dense {dense[0].seconds:.4f} s")


if __name__ == "__main__":
    main()
