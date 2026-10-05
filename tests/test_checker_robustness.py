"""Tests for two inputs the checker raised on (SGLang 0.5.18, found while ROADMAP M22.5 measured; fixed in M22.6):
  a kernel whose name ends in `_loc`   the location annotations' pattern took `_loc(` and the argument list after it
                                       for an annotation, so the IR's function head lost its arguments
  an address from a pointer value      tt.addptr read the taint of a value the checker does not follow as a pointer
  the checker does not follow
Each launch is now a verdict, not an exception. The IR and the launch are the ones SGLang 0.5.18 made (recorded on the
GPU, paths scrubbed); before the fix both raised (AttributeError, ValueError).
Run: python tests/test_checker_robustness.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import kernel_check as KC  # noqa: E402
from entail import kernel_ir as KI  # noqa: E402
from entail import kernel_types as KT  # noqa: E402

IR = os.path.join(HERE, "data", "kernel_ir")


def recorded(name):
    with open(os.path.join(IR, name + ".ttir"), encoding="utf-8") as f:
        ttir = f.read()
    with open(os.path.join(IR, name + ".json"), encoding="utf-8") as f:
        return ttir, json.load(f)


def launch(name, rows=None):
    ttir, rec = recorded(name)
    grid = list(rec["grid"])
    tensors = rec["tensors"]
    if rows is not None:
        n = grid[0]
        tensors = {k: dict(t, shape=[rows if (i == 0 and d == n) or (i == 1 and d == n and k == "positions_ptr")
                                     else d for i, d in enumerate(t["shape"])])
                   for k, t in tensors.items()}
        grid[0] = rows
    written = KI.written_args(ttir)
    meanings = {}
    for k, t in tensors.items():
        m = KC._meaning_from(t["shape"], t["stride"], t["fact"])
        if written is not None and k in written:
            m.kind = "output"
        meanings[k] = m
    return KT.check_launch(ttir, meanings, rec["scalars"], tuple(grid)), written


def main():
    head = ('  tt.func public @fill_accept_out_cache_loc(%a: !tt.ptr<i32> {tt.divisibility = 16 : i32} loc("a"(#loc)), '
            '%b: !tt.ptr<i64> loc("b"(#loc))) attributes {noinline = false} {')
    s = KI._strip(head)
    assert s == ('  tt.func public @fill_accept_out_cache_loc(%a: !tt.ptr<i32> {tt.divisibility = 16 : i32}, '
                 '%b: !tt.ptr<i64>) attributes {noinline = false} {'), s
    assert KI._strip("    %x = arith.constant 1 : i32 loc(#loc1)") == "    %x = arith.constant 1 : i32"
    assert KI._strip('    %y = tt.load %p loc(callsite(#loc1 at #loc31))') == "    %y = tt.load %p"
    print("ok a location annotation is `loc(` as a word of its own: a name ending in _loc keeps its arguments")

    ttir, rec = recorded("sglang0518_fill_accept_out_cache_loc")
    fn = KI.parse(ttir)
    assert fn.name == "fill_accept_out_cache_loc", fn.name
    assert [a for a, _t in fn.args] == ["%accept_index", "%out_cache_loc", "%accept_out_cache_loc"], fn.args
    v, written = launch("sglang0518_fill_accept_out_cache_loc")
    assert set(written) == {"accept_out_cache_loc"}, written
    assert not v.why.startswith("the check raised"), v
    print(f"ok SGLang's fill_accept_out_cache_loc: its IR parses and the launch is decided ({v.verdict}: "
          f"{v.why[:70]})")

    v, _written = launch("sglang0518_fused_qk_rmsnorm_rope_gate")
    assert v.verdict != "violation" and not v.why.startswith("the check raised"), v
    print(f"ok SGLang's _fused_qk_rmsnorm_rope_gate_kernel (1792 rows): an address from a pointer the checker does not "
          f"follow is not decided, and the launch gets a verdict ({v.verdict}: {v.why[:90]})")


if __name__ == "__main__":
    main()
