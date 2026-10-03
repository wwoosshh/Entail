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

import numpy as np

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
        # v3 called this harmless and proven: the extra K tile is masked out, but its scales are read unmasked at
        # As[m, 20] / Bs[n, 20], past the 20 scale groups - a value that may be inf or NaN, and 0 x inf is NaN
        ("one more iteration than K needs: its scales are read outside As and Bs",
         edit(src, ("%1 = arith.divsi %0, %c128_i32 : i32", "%1a = arith.divsi %0, %c128_i32 : i32\n"
                                                             "    %1 = arith.addi %1a, %c1_i32 : i32")),
         "violation"),
    ]


def address_counterexamples(src):
    """(name, IR, verdict of v3 2a51daf, verdict now): the integer meaning of addresses and masks (L5.4d). v3 read
    every integer as a plain int64, so both were proven; read as the IR defines them, neither computes the contract."""
    return [
        # the k masks written as (k - 64) <u (bound - 64): signed, the same test as k < bound; unsigned, k < 64 is a
        # negative number read as 2^32 - (64 - k), so the first half of every K tile is masked out
        ("k masks compared unsigned after a shift by 64: half of every K tile is masked out",
         edit(src, ("%a_61 = tt.splat %a_60 : i32 -> tensor<1x128xi32>",
                    "%a_60s = arith.subi %a_60, %c64_i32 : i32\n"
                    "      %a_61 = tt.splat %a_60s : i32 -> tensor<1x128xi32>"),
                   ("%a_62 = arith.cmpi slt, %a_ptrs_22, %a_61 : tensor<1x128xi32>",
                    "%c64a = arith.constant dense<64> : tensor<1x128xi32>\n"
                    "      %ks_a = arith.subi %a_ptrs_22, %c64a : tensor<1x128xi32>\n"
                    "      %a_62 = arith.cmpi ult, %ks_a, %a_61 : tensor<1x128xi32>"),
                   ("%b = tt.splat %a_60 : i32 -> tensor<128x1xi32>", "%b = tt.splat %a_60s : i32 -> tensor<128x1xi32>"),
                   ("%b_65 = arith.cmpi slt, %b_ptrs, %b : tensor<128x1xi32>",
                    "%c64b = arith.constant dense<64> : tensor<128x1xi32>\n"
                    "      %ks_b = arith.subi %b_ptrs, %c64b : tensor<128x1xi32>\n"
                    "      %b_65 = arith.cmpi ult, %ks_b, %b : tensor<128x1xi32>")),
         "proven", "violation"),
        # A's tile moved by extui(-1 : i32) + 1: as a plain int64 that is 0; extui reads -1 unsigned, 2^32 - 1, so the
        # tile moves by 2^32 elements, far outside A
        ("A's pointers moved by extui(-1) + 1: 2^32 elements, not 0",
         edit(src, ("%b_ptrs = tt.expand_dims %offs_bn_15 {axis = 1 : i32}",
                    "%neg1 = arith.constant -1 : i32\n"
                    "    %neg1w = arith.extui %neg1 : i32 to i64\n"
                    "    %one64 = arith.constant 1 : i64\n"
                    "    %movew = arith.addi %neg1w, %one64 : i64\n"
                    "    %movet = tt.splat %movew : i64 -> tensor<64x128xi64>\n"
                    "    %a_ptrs_27m = tt.addptr %a_ptrs_27, %movet : tensor<64x128x!tt.ptr<f8E4M3FN>>, "
                    "tensor<64x128xi64>\n"
                    "    %b_ptrs = tt.expand_dims %offs_bn_15 {axis = 1 : i32}"),
                   ("iter_args(%a_ptrs_57 = %a_ptrs_27,", "iter_args(%a_ptrs_57 = %a_ptrs_27m,")),
         "proven", "violation"),
    ]


def integer_ops():
    """Each integer operation against what MLIR's arith defines for it: (IR line, operands {name: (value, type)},
    expected value or None when the result is poison or undefined - not modelled, the launch unproven)."""
    i32min = -2 ** 31
    return [
        ("%r = arith.addi %x, %y : i32", {"x": (2 ** 31 - 1, "i32"), "y": (1, "i32")}, i32min),        # wraps
        ("%r = arith.addi %x, %y overflow<nsw> : i32", {"x": (2 ** 31 - 1, "i32"), "y": (1, "i32")}, None),
        ("%r = arith.muli %x, %y : i16", {"x": (300, "i16"), "y": (300, "i16")}, 90000 - 65536),
        ("%r = arith.subi %x, %y overflow<nuw> : i32", {"x": (1, "i32"), "y": (2, "i32")}, None),
        ("%r = arith.extsi %x : i1 to i32", {"x": (1, "i1")}, -1),
        ("%r = arith.extui %x : i1 to i32", {"x": (1, "i1")}, 1),
        ("%r = arith.extui %x : i32 to i64", {"x": (-1, "i32")}, 2 ** 32 - 1),
        ("%r = arith.extsi %x : i32 to i64", {"x": (-1, "i32")}, -1),
        ("%r = arith.trunci %x : i32 to i8", {"x": (300, "i32")}, 44),
        ("%r = arith.trunci %x : i32 to i8", {"x": (200, "i32")}, -56),
        ("%r = arith.index_cast %x : index to i32", {"x": (2 ** 33 + 7, "index")}, 7),
        ("%r = arith.index_cast %x : i32 to index", {"x": (-5, "i32")}, -5),
        ("%r = arith.index_castui %x : i32 to index", {"x": (-5, "i32")}, 2 ** 32 - 5),
        ("%r = arith.divsi %x, %y : i32", {"x": (-7, "i32"), "y": (2, "i32")}, -3),
        ("%r = arith.remsi %x, %y : i32", {"x": (-7, "i32"), "y": (2, "i32")}, -1),
        ("%r = arith.floordivsi %x, %y : i32", {"x": (-7, "i32"), "y": (2, "i32")}, -4),
        ("%r = arith.ceildivsi %x, %y : i32", {"x": (7, "i32"), "y": (2, "i32")}, 4),
        ("%r = arith.divsi %x, %y : i32", {"x": (5, "i32"), "y": (0, "i32")}, None),
        ("%r = arith.divsi %x, %y : i32", {"x": (i32min, "i32"), "y": (-1, "i32")}, None),
        ("%r = arith.divui %x, %y : i32", {"x": (-1, "i32"), "y": (2, "i32")}, 2 ** 31 - 1),
        ("%r = arith.remui %x, %y : i32", {"x": (-1, "i32"), "y": (10, "i32")}, (2 ** 32 - 1) % 10),
        ("%r = arith.minui %x, %y : i32", {"x": (-1, "i32"), "y": (5, "i32")}, 5),
        ("%r = arith.minsi %x, %y : i32", {"x": (-1, "i32"), "y": (5, "i32")}, -1),
        ("%r = arith.maxui %x, %y : i32", {"x": (-1, "i32"), "y": (5, "i32")}, -1),
        ("%r = arith.shli %x, %y : i32", {"x": (1, "i32"), "y": (31, "i32")}, i32min),
        ("%r = arith.shli %x, %y : i32", {"x": (1, "i32"), "y": (32, "i32")}, None),
        ("%r = arith.shrsi %x, %y : i32", {"x": (-8, "i32"), "y": (1, "i32")}, -4),
        ("%r = arith.shrui %x, %y : i32", {"x": (-8, "i32"), "y": (1, "i32")}, (2 ** 32 - 8) >> 1),
        ("%r = arith.andi %x, %y : i32", {"x": (-1, "i32"), "y": (0xFFFF, "i32")}, 0xFFFF),
        ("%r = arith.xori %x, %y : i32", {"x": (-1, "i32"), "y": (1, "i32")}, -2),
        ("%r = arith.cmpi ult, %x, %y : i32", {"x": (-1, "i32"), "y": (0, "i32")}, 0),
        ("%r = arith.cmpi slt, %x, %y : i32", {"x": (-1, "i32"), "y": (0, "i32")}, 1),
        ("%r = arith.cmpi slt, %x, %y : i1", {"x": (1, "i1"), "y": (0, "i1")}, 1),       # true is -1 signed
        ("%r = arith.constant 4294967295 : i32", {}, -1),
        ("%r = arith.constant -1 : i64", {}, -1),
        ("%r = arith.mulsi_extended %x, %y : i32", {"x": (2, "i32"), "y": (3, "i32")}, None),   # not modelled
    ]


def check_integer_ops():
    """Run each line through the evaluator alone (one program), as the launch would see it."""
    fn = kernel_ir.Func("t", [], [])
    out = []
    for line, vals, want in integer_ops():
        run = kernel_ir._Run(fn, {}, {}, [np.zeros(1, dtype=np.int64)] * 3, dense=True)
        env = {f"%{k}": kernel_ir.E((), s=np.full((1,), v, dtype=np.int64), b=(t == "i1"),
                                    w=kernel_ir._width(t)) for k, (v, t) in vals.items()}
        op = kernel_ir._op(line)
        try:
            run._op(op, env)
            r = env["%r"]
            got = int(r.full().reshape(-1)[0])
        except kernel_ir.Unmodelled:
            got = None
        out.append((line, want, got))
    return out


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
    # 63 * 2^26 > 2^31: the IR's i32 offsets wrap (two's complement, computed as such since L5.4d; v3 refused to
    # model it) and the wrapped addresses fall outside A
    v = launch("orig", 64, 64, stride_am=1 << 26)
    assert v.verdict == "violation" and "outside" in v.why, v
    b = binding(64)
    b["Mut"] = Tensor("output", 0, 0, (64, N), (N, 1))     # two tensors could be the output: nothing is claimed
    assert launch("orig", 64, 64, b=b).verdict == "unproven"
    print("ok a partial row tile: proven; wrapped i32 offsets: violation; two outputs: unproven")

    # integer meaning (L5.4d): every operation as MLIR's arith defines it
    for line, want, got in check_integer_ops():
        assert got == want, (line, want, got)
    print(f"ok {len(integer_ops())} integer operations: widths, two's complement wrap, signed and unsigned readings, "
          f"poison and undefined behaviour not modelled")
    bad += [(name, ttir, now) for name, ttir, _v3, now in address_counterexamples(src)]
    for name, ttir, _v3, now in address_counterexamples(src):
        v = launch("orig", 64, 64, ttir=ttir)
        assert v.verdict == now, (name, v.verdict, v.why)
        print(f"ok {name}: {v.verdict} ({v.why[:110]})")

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
        # a proven launch is decided without going element by element; a read outside an operand is found there
        assert f.verdict != "proven" or not f.dense, (n, f)
    print("ok fast and dense evaluation agree on", len(cases), "launches;",
          f"orig: fast {fast[0].seconds:.4f} s, dense {dense[0].seconds:.4f} s")
    inkernel_checks()
    written_checks()


def inkernel_binding(M, alt_pair=5, chk_role="integrity_sums"):
    """The binding of a kernel rewritten with the in-kernel integrity check (entail/inkernel.py, L5.4e): the sums
    buffer and the two word views of A and B besides the operands."""
    b = binding(M, alt_pair)
    b["EntailChk"] = Tensor(chk_role, 0, 0, (8,), (1,))
    b["EntailAw"] = Tensor("other", 0, 0, (M, K // 4), (K // 4, 1))
    b["EntailBw"] = Tensor("other", 0, 0, (N, K // 4), (K // 4, 1))
    return b


def inkernel_checks():
    """L5.4e: vLLM's kernel and the four mutants, rewritten with the in-kernel integrity check (tests/data/kernel_ir/
    inkernel_*.ttir, Triton 3.7.1 on the RTX 4070 Ti, tiles 64 x 128 x 128) get the verdicts the kernels they were
    made from get; what the check adds may only add into the sums buffer, under conditions read from data."""
    src = open(os.path.join(DATA, "inkernel_orig.ttir"), encoding="utf-8").read()
    v = launch("inkernel_orig", 64, 64, b=inkernel_binding(64), ttir=src)
    assert v.verdict == "proven" and v.terms == 20 and v.programs == 12, v
    for name, want in (("interval_0", "unproven"), ("neighbor_0", "unproven"), ("alt_weight_0", "possible"),
                       ("rows_0", "possible")):
        v = launch("inkernel_" + name, 64, 64, b=inkernel_binding(64))
        assert v.verdict == want, (name, v)
    v = launch("inkernel_alt_weight_0", 64, 64, b=inkernel_binding(64, alt_pair=3))
    assert v.verdict == "proven", v
    print("ok the rewritten kernels: the verdicts of the kernels they were made from (vLLM's proven, mutants not)")
    lines = src.splitlines()                  # the lines to edit are found by what they hold, not by their names
    first = next(x for x in lines if "tt.atomic_rmw" in x and "%EntailChk," in x)
    val = first.split("%EntailChk,", 1)[1].split(",")[0].strip()
    store = next(x for x in lines if "tt.store" in x).split(" loc(")[0]
    start = next(i for i, x in enumerate(lines) if "scf.if" in x)
    branch_end = next(x for x in lines[start:] if x.strip().startswith("scf.yield"))
    cases = [
        ("the sums buffer bound as an ordinary tensor", src, "violation", inkernel_binding(64, chk_role="other")),
        ("an atomic add into the output", edit(src, (first, first.replace(f"%EntailChk, {val}", f"%C, {val}")
                                                     .replace("!tt.ptr<i32>", "!tt.ptr<bf16>"))), "violation", None),
        ("an atomic exchange into the sums buffer", edit(src, (first, first.replace("add,", "exch,"))), "violation",
         None),
        ("a store under a condition read from data",
         edit(src, (branch_end, "        " + store.strip() + "\n" + branch_end)), "unproven", None),
    ]
    for name, ttir, want, b in cases:
        v = launch("inkernel_orig", 64, 64, b=b or inkernel_binding(64), ttir=ttir)
        assert v.verdict == want, (name, v.verdict, v.why)
        print(f"ok {name}: {v.verdict} ({v.why[:100]})")

def written_checks():
    """L5.4e, approach "between the operations": what a Triton kernel writes, read from its TTIR (written_args) -
    vLLM's kernel and the mutants write only their output; the rewritten kernels also their sums buffer; a store
    through a pointer whose origin cannot be followed (here: an unknown value) gives None, every argument written."""
    for name, want in (("orig", ["C"]), ("interval_0", ["C"]), ("alt_weight_0", ["C"]), ("tile_1", ["C"]),
                       ("inkernel_orig", ["C", "EntailChk"]), ("inkernel_rows_0", ["C", "EntailChk"])):
        got = kernel_ir.written_args(open(os.path.join(DATA, name + ".ttir"), encoding="utf-8").read())
        assert got == want, (name, got)
    src = open(os.path.join(DATA, "orig.ttir"), encoding="utf-8").read()
    line = next(x for x in src.splitlines() if "tt.store" in x)
    ptr = line.split("tt.store", 1)[1].split(",")[0].strip()
    assert kernel_ir.written_args(src.replace(line, line.replace(ptr, "%unknown_ptr"))) is None
    print("ok written_args: vLLM's kernel and the mutants write C; the rewritten ones C and the sums buffer; a store "
          "through a pointer of unknown origin: every argument")

if __name__ == "__main__":
    main()
