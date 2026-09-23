"""Tests for boundary declarations (ROADMAP 2.2): what a boundary takes, what its result means, what it writes.
Every feature has a passing and a failing case, and the failing case checks the message.
Run: python tests/test_boundary.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import entail as rc  # noqa: E402

INTERLEAVED = rc.Layout("q8_0", packing="interleaved")
SPLIT = rc.Layout("q8_0", packing="split")
FP8 = rc.Quantized("float8_e4m3fn", scale=0.5)
FP32 = rc.Quantized("float32", scale=None)


def expect_error(fn, text):
    try:
        fn()
    except rc.RoleError as e:
        assert text in str(e), str(e)
        return str(e)
    raise AssertionError("expected RoleError containing " + text)


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
    lora(x=y)
    expect_error(lambda: gemm(x=y), "expected Quantized(dtype='float8_e4m3fn'")


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
    expect_error(lambda: read_interleaved(buf=buf), "got Layout(kind='q8_0'")


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
    msg = expect_error(lambda: read_interleaved(buf=buf), "made it untrue")
    assert "scribble wrote into it" in msg and "Layout" in msg


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
    expect_error(lambda: lazy(src=s, dst=torch.zeros(4)), "declares it writes 'dst', but 'dst' was not written")
    expect_error(lambda: sloppy(src=torch.ones(4), dst=torch.zeros(4)), "wrote into 'src', which it does not declare")
    expect_error(lambda: copy_positional(torch.ones(4), torch.zeros(4)), "must be passed by keyword")


def test_result_can_carry_an_argument_meaning():
    rc.set_mode("debug")

    @rc.boundary(name="flatten", returns=rc.carry("x"))
    def flatten(*, x):
        return x.reshape(-1)

    y = flatten(x=rc.tag(torch.ones(2, 2), SPLIT))
    assert rc.facts_of(y) == {"Layout": SPLIT}


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
    expect_error(lambda: decode(cache=cache), "expected")


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
    expect_error(lambda: guarded_lie(src=torch.ones(3), dst=torch.zeros(3)), "wrote into 'dst', which it does not declare")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
