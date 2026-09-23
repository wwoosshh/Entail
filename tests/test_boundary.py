"""Tests for code boundary signatures (ROADMAP 2.2, moved into the new structure in M4.1): what a boundary takes,
what its result means, what it writes; every check a Decision with its wording; the repairs a boundary can make on
the value; agree and advance. Run: python tests/test_boundary.py"""
import io
import os
import sys
from contextlib import redirect_stdout

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# These tests check what stops, so they run under the policy that stopped before M5.4 (ENTAIL_ON_BROKEN=stop,
# unknown meaning-changing facts required); the default, which reports and goes on, is tested in test_report.py.
os.environ["ENTAIL_ON_BROKEN"] = "stop"
os.environ["ENTAIL_UNKNOWN"] = "require"
import entail as rc  # noqa: E402
from entail import boundaries, core, load  # noqa: E402
from entail.contracts import Verdict  # noqa: E402

INTERLEAVED = rc.Layout("q8_0", packing="interleaved")
SPLIT = rc.Layout("q8_0", packing="split")
FP8 = rc.Quantized("float8_e4m3fn", scale=0.5)
FP32 = rc.Quantized("float32", scale=None)
DENSE, STRIDED = rc.Layout("dense"), rc.Layout("strided")


def expect_error(fn, *texts):
    try:
        with redirect_stdout(io.StringIO()):
            fn()
    except rc.RoleError as e:
        for text in texts:
            assert text in str(e), (text, str(e))
        return str(e)
    raise AssertionError("expected RoleError containing " + repr(texts))


def quiet(fn, *a, **kw):
    with redirect_stdout(io.StringIO()) as out:
        result = fn(*a, **kw)
    return result, out.getvalue()


def test_result_carries_what_is_declared():
    rc.set_mode("debug")

    @rc.boundary(name="dequantize", x=FP8, returns=FP32)
    def dequantize(*, x):
        return x.float() * 0.5

    @rc.boundary(name="LoRA path", x=FP32)
    def lora(*, x):
        return x

    @rc.boundary(name="fp8 GEMM", x=FP8)
    def gemm(*, x):
        return x

    y = dequantize(x=rc.tag(torch.ones(4), FP8))
    assert rc.facts_of(y) == {"Quantized": FP32}  # nobody tagged y: the boundary said what it produces
    assert str(core.envelopes_of(y)["Quantized"].source) == "boundary: dequantize.returns"
    lora(x=y)
    expect_error(lambda: gemm(x=y), "refused at boundary:fp8 GEMM", "uses Quantized(dtype='float8_e4m3fn'",
                 "boundary: dequantize.returns", "no resolution is registered", "note: argument x")


def test_written_argument_carries_its_new_meaning():
    """rolebench #1: the reorder rewrites the buffer in place; a reader of the old layout must see that."""
    rc.set_mode("debug")

    @rc.boundary(name="reorder", buf=INTERLEAVED, writes={"buf": SPLIT})
    def reorder(*, buf):
        buf.mul_(1)

    @rc.boundary(name="interleaved reader", buf=(INTERLEAVED,))
    def read_interleaved(*, buf):
        return buf

    buf = rc.tag(torch.ones(8), INTERLEAVED)
    read_interleaved(buf=buf)
    reorder(buf=buf)
    assert rc.facts_of(buf) == {"Layout": SPLIT}
    expect_error(lambda: read_interleaved(buf=buf), "declared Layout(kind='q8_0'", "packing='split'",
                 "boundary: reorder.writes.buf")


def test_write_without_a_declared_meaning_invalidates():
    rc.set_mode("debug")

    @rc.boundary(name="scribble", writes={"buf": None})
    def scribble(*, buf):
        buf.zero_()

    @rc.boundary(name="interleaved reader", buf=(INTERLEAVED,))
    def read_interleaved(*, buf):
        return buf

    buf = rc.tag(torch.ones(8), INTERLEAVED, rc.Positions("absolute"))
    scribble(buf=buf)
    expect_error(lambda: read_interleaved(buf=buf), "made untrue on the way, and nothing says what it holds now",
                 "made untrue by scribble, which wrote into it", "Layout")


def test_writing_is_checked_against_the_declaration():
    rc.set_mode("debug")

    @rc.boundary(name="claims a write", writes={"dst": SPLIT})
    def lazy(*, src, dst):
        return src

    @rc.boundary(name="writes its source", writes={"dst": SPLIT})
    def sloppy(*, src, dst):
        src.zero_()
        dst.copy_(src)

    @rc.boundary(name="copy", writes={"dst": SPLIT})
    def copy(*, src, dst):
        dst.copy_(src)

    @rc.boundary(name="copy by position", writes={"dst": SPLIT})
    def copy_positional(src, dst):
        dst.copy_(src)

    s, d = torch.ones(4), torch.zeros(4)
    copy(src=s, dst=d)
    expect_error(lambda: lazy(src=s, dst=torch.zeros(4)), "declares it writes this argument, but it was not written",
                 "argument dst")
    expect_error(lambda: sloppy(src=torch.ones(4), dst=torch.zeros(4)),
                 "wrote into an argument it does not declare", "argument src")
    expect_error(lambda: copy_positional(torch.ones(4), torch.zeros(4)), "must be passed by keyword")


def test_result_can_carry_an_argument_meaning():
    rc.set_mode("debug")

    @rc.boundary(name="loader", returns=SPLIT)
    def load_it(*, x):
        return x

    @rc.boundary(name="flatten", returns=rc.carry("x"))
    def flatten(*, x):
        return x.reshape(-1)

    y = flatten(x=load_it(x=torch.ones(2, 2)))
    assert rc.facts_of(y) == {"Layout": SPLIT}
    assert str(core.envelopes_of(y)["Layout"].source) == "boundary: loader.returns"   # the chain keeps its origin


def test_meaning_that_depends_on_the_values():
    """A cache append: the valid length afterwards depends on how many entries it held."""
    rc.set_mode("debug")

    @rc.boundary(name="append to cache", writes={"cache": lambda out, a: rc.Valid(length=a["n"] + 1)})
    def append(*, cache, n, kv):
        cache[n].copy_(kv)

    @rc.boundary(name="decode at position", cache=lambda f: f.length == 5)
    def decode(*, cache):
        return cache

    cache = rc.tag(torch.zeros(8, 2), rc.Valid(length=4))
    append(cache=cache, n=4, kv=torch.ones(2))
    assert rc.facts_of(cache) == {"Valid": rc.Valid(length=5)}
    decode(cache=cache)
    append(cache=cache, n=5, kv=torch.ones(2))
    expect_error(lambda: decode(cache=cache), "does not satisfy what this boundary takes", "Valid")


def test_load_mode_carries_meaning_without_checking():
    rc.set_mode("load")

    @rc.boundary(name="dequantize", x=FP8, returns=FP32)
    def dequantize(*, x):
        return x.float()

    y = dequantize(x=torch.ones(2))  # no declaration on x: not checked in load mode
    assert rc.facts_of(y) == {"Quantized": FP32}


def test_off_mode_does_nothing():
    rc.set_mode("off")

    @rc.boundary(name="dequantize", x=FP8, returns=FP32, writes={"x": None})
    def dequantize(*, x):
        return x.float()

    y = dequantize(x=torch.ones(2))
    assert rc.facts_of(y) == {}


def test_custom_op_declaration_is_read_and_checked():
    rc.set_mode("debug")

    @torch.library.custom_op("entail_test::copy_into", mutates_args=("dst",))
    def copy_into(src: torch.Tensor, dst: torch.Tensor) -> None:
        dst.copy_(src)

    @torch.library.custom_op("entail_test::lies", mutates_args=())
    def lies(src: torch.Tensor, dst: torch.Tensor) -> None:  # says it writes nothing, writes dst
        dst.copy_(src)

    guarded = rc.boundary(name="copy_into", writes={"dst": SPLIT})(copy_into)
    d = torch.zeros(3)
    guarded(src=torch.ones(3), dst=d)
    assert rc.facts_of(d) == {"Layout": SPLIT}
    expect_error(lambda: rc.boundary(name="copy_into", writes={"src": SPLIT})(copy_into),
                 "the op declares it writes ['dst'] (mutates_args), the boundary says ['src']")
    guarded_lie = rc.boundary(name="lies")(lies)
    expect_error(lambda: guarded_lie(src=torch.ones(3), dst=torch.zeros(3)),
                 "wrote into an argument it does not declare", "argument dst")


# --- M4.1: repairs, policy, agree, advance, counting --------------------------------------------------------------

def test_a_strided_value_is_made_contiguous_for_a_packed_reader():
    """rolebench #3: a kernel that assumes packed rows gets a strided slice; the boundary hands it a packed copy."""
    rc.set_mode("debug")
    seen = {}

    @rc.boundary(name="QKV split", returns=lambda out, a: DENSE if out.is_contiguous() else STRIDED)
    def split_q(*, qkv):
        return qkv[:, :4]

    @rc.boundary(name="packed kernel", q=DENSE)
    def kernel(*, q):
        seen["contiguous"] = q.is_contiguous()
        return q.sum()

    q = split_q(qkv=torch.arange(24.0).reshape(4, 6))
    assert not q.is_contiguous() and rc.facts_of(q) == {"Layout": STRIDED}
    n = len(load.LEDGER.decisions)
    total, out = quiet(kernel, q=q)
    assert seen["contiguous"] and float(total) == float(q.sum())
    d = load.LEDGER.decisions[n]
    assert d.verdict is Verdict.RESOLVED and d.handle == "layout.contiguous" and "resolved at boundary:packed" in out
    assert d.resolution == f"make the tensor contiguous ({STRIDED} -> {DENSE})", d.resolution   # value's side first
    quiet(kernel, q=q)                                   # the same repair again: counted, not printed again
    assert len(load.LEDGER.decisions) == n + 1 and sum(v for k, v in boundaries.REPEATS.items()
                                                        if k[0] == "packed kernel") >= 1


def test_fp8_is_dequantized_and_chunk_positions_made_absolute():
    rc.set_mode("debug")

    @rc.boundary(name="LoRA (unquantized input)", x=(rc.Quantized("float32"), rc.Quantized("bfloat16")))
    def lora(*, x):
        return x

    xq = rc.tag((torch.ones(4) * 3).to(torch.float8_e4m3fn), rc.Quantized("float8_e4m3fn", scale=0.5))
    y, _ = quiet(lora, x=xq)
    assert y.dtype == torch.float32 and torch.equal(y, torch.full((4,), 1.5))
    assert rc.facts_of(y) == {"Quantized": rc.Quantized("float32")}

    @rc.boundary(name="mask builder", q_pos=rc.Positions("absolute"))
    def mask(*, q_pos):
        return q_pos

    p, _ = quiet(mask, q_pos=rc.tag(torch.arange(3), rc.Positions("chunk_relative", offset=10)))
    assert p.tolist() == [10, 11, 12]


def test_refuse_policy_and_unknown_settings():
    rc.set_mode("debug")

    @rc.boundary(name="packed kernel", q=DENSE)
    def kernel(*, q):
        return q

    strided = rc.tag(torch.ones(4, 4)[:, :2], STRIDED)
    core.set_policy("refuse")
    try:
        expect_error(lambda: kernel(q=strided), "the policy repairs nothing")
    finally:
        core.set_policy("resolve")
    expect_error(lambda: kernel(q=torch.ones(2)), "unknown at boundary:packed kernel", "nothing declares it")
    os.environ["ENTAIL_UNKNOWN"] = "report"
    try:
        out = quiet(kernel, q=torch.ones(2))[1]
        assert "unknown at boundary:packed kernel" in out and "stops here" not in out
    finally:
        os.environ["ENTAIL_UNKNOWN"] = "require"


def test_agree_and_advance():
    """rolebench #10: a mask built for the position counter at one epoch, read after the counter moved on."""
    rc.set_mode("debug")

    @rc.boundary(name="position counter", returns=rc.Epoch(0, owner="query offset"))
    def counter(*, start):
        return torch.tensor(start)

    @rc.boundary(name="cache update", writes={"offset": boundaries.advance()})
    def update(*, offset):
        offset.add_(1)

    @rc.boundary(name="mask builder", returns=rc.carry("offset"))
    def build(*, offset):
        return torch.zeros(2)

    @rc.boundary(name="attention", agree={"Epoch": ("mask", "offset")})
    def attention(*, mask, offset):
        return mask

    live = counter(start=5)
    mask = build(offset=live)
    attention(mask=mask, offset=live)
    update(offset=live)
    assert rc.facts_of(live)["Epoch"] == rc.Epoch(1, owner="query offset")
    expect_error(lambda: attention(mask=mask, offset=live), "arguments that must carry the same fact carry different",
                 "mask and offset must carry the same Epoch")
    snap = counter(start=6)                      # a snapshot taken for the mask stays at its own epoch
    attention(mask=build(offset=snap), offset=snap)


def test_passes_are_counted_not_recorded():
    rc.set_mode("debug")

    @rc.boundary(name="hot path", x=FP32)
    def hot(*, x):
        return x

    n = len(load.LEDGER.decisions)
    x = rc.tag(torch.ones(2), FP32)
    for _ in range(100):
        hot(x=x)
    assert len(load.LEDGER.decisions) == n and boundaries.PASSES[("hot path", "Quantized")] >= 100


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
