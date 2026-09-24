"""Tests for the role-typed front end (ROADMAP M8.1): every check with a program it passes and one it refuses (with
the words of the refusal), the named dims, the repairs made while tracing, and the lowerings against a float32
reference. The CUDA lowerings (triton, flex) are compared with the torch one when a GPU is there.
Run: python tests/test_frontend.py"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import frontend as fe  # noqa: E402
from entail.facts import LAYOUT_KINDS, Layout, ModelProps, Positions, Quantized, Reduction, Rotary  # noqa: E402

B, L, HQ, HKV, D, S = 2, 1, 8, 2, 16, 12
ROT = Rotary("default", 10000.0)
ABS = Positions("absolute")


def qtype(dims=("batch", "tokens", "heads", "head_dim"), frame=None):
    sizes = {"batch": B, "tokens": L, "heads": HQ, "head_dim": D}
    return fe.T(dims, "float32", "query", tuple(sizes[d] for d in dims), (frame,) if frame else ())


def cache_type(kind, frame=ABS):
    return fe.T(("batch", "kv_heads", "slots", "head_dim"), "float32", kind, (B, HKV, S, D), (frame,) if frame else ())


def new_rows(kind):
    return fe.T(("batch", "tokens", "kv_heads", "head_dim"), "float32", kind, (B, L, HKV, D))


POS = fe.T(("tokens",), "int64", "positions", (L,), (ABS,))
UNTIL = fe.T(("batch",), "int64", "last_key", (B,))


def refused(fn, words, **types):
    try:
        fe.trace(fn, **types)
    except fe.RoleError as e:
        assert words in str(e), (words, str(e))
        return str(e)
    raise AssertionError(f"traced, but should have been refused with: {words}")


def decode(*, q, k, v, keys, values, positions, until):
    """One decode step of attention: rotate, write the new key/value, attend over the cache."""
    q = fe.rope(x=q, positions=positions, rotary=ROT)
    k = fe.rope(x=k, positions=positions, rotary=ROT)
    keys = fe.write(into=keys, src=k, at=positions)
    values = fe.write(into=values, src=v, at=positions)
    return fe.attend(query=q, keys=keys, values=values, until=until, share=HQ // HKV)


TYPES = dict(q=qtype(), k=new_rows("key"), v=new_rows("value"), keys=cache_type("key"),
             values=cache_type("value", None), positions=POS, until=UNTIL)


def data(pos=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(q=torch.randn(B, L, HQ, D, generator=g), k=torch.randn(B, L, HKV, D, generator=g),
                v=torch.randn(B, L, HKV, D, generator=g), keys=torch.randn(B, HKV, S, D, generator=g),
                values=torch.randn(B, HKV, S, D, generator=g), positions=torch.tensor([pos]),
                until=torch.tensor([pos, pos]))


def reference(d, softcap=None):
    """The same step written out in float32: RoPE by rotate-half, keys 0..until, heads repeated by group."""
    pos = d["positions"].float()
    inv = 1.0 / (10000.0 ** (torch.arange(0, D, 2).float() / D))
    emb = torch.cat([pos[:, None] * inv] * 2, -1)
    cos, sin = emb.cos().view(1, L, 1, D), emb.sin().view(1, L, 1, D)

    def rot(x):
        return x * cos + torch.cat((-x[..., D // 2:], x[..., :D // 2]), -1) * sin

    q, k = rot(d["q"]), rot(d["k"])
    keys, values = d["keys"].clone(), d["values"].clone()
    p = int(d["positions"][0])
    keys[:, :, p] = k[:, 0]
    values[:, :, p] = d["v"][:, 0]
    kk, vv = keys.repeat_interleave(HQ // HKV, 1), values.repeat_interleave(HQ // HKV, 1)
    s = (q.transpose(1, 2) @ kk.transpose(-1, -2)) * D ** -0.5
    if softcap:
        s = torch.tanh(s / softcap) * softcap
    s[..., p + 1:] = float("-inf")
    return (torch.softmax(s, -1) @ vv).transpose(1, 2)


# --- what runs --------------------------------------------------------------------------------------------------

def test_a_traced_step_computes_what_it_says():
    program = fe.trace(decode, **TYPES)
    d = data()
    want = reference(d)
    got = program(**{k: v.clone() for k, v in d.items()})
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()
    assert any("lowered: torch" in line for line in program.lines())


def test_dims_are_read_by_name():
    """A query laid out (batch, heads, tokens, head_dim) is read as one, whatever the sizes."""
    dims = ("batch", "heads", "tokens", "head_dim")
    program = fe.trace(decode, **dict(TYPES, q=qtype(dims)))
    d = data()
    want = reference(d)
    got = program(**dict({k: v.clone() for k, v in d.items()}, q=d["q"].transpose(1, 2).contiguous()))
    assert torch.allclose(got.transpose(1, 2), want, atol=1e-5)


def test_work_every_layer_would_repeat_is_made_once():
    def two_layers(**kw):
        a = decode(**kw)
        q = fe.rope(x=kw["q"], positions=kw["positions"], rotary=ROT)
        return a, q

    program = fe.trace(two_layers, **TYPES)
    assert sum(1 for line in program.lines() if line.startswith("rope_angles#")) == 1, program.lines()


# --- what is refused, and why ----------------------------------------------------------------------------------

def test_a_key_is_not_a_value():
    def swapped(**kw):
        q = fe.rope(x=kw["q"], positions=kw["positions"], rotary=ROT)
        return fe.attend(query=q, keys=kw["values"], values=kw["keys"], until=kw["until"], share=4)

    refused(swapped, "keys takes key, got value", **TYPES)


def test_there_is_no_mask_to_get_wrong():
    """Mask polarity and a boolean read as additive (WEEK4_NOTES 7.1) have nothing to be written with."""
    try:
        fe.trace(lambda **kw: fe.attend(query=kw["q"], keys=kw["keys"], values=kw["values"], until=kw["until"],
                                        share=4, mask=None), **TYPES)
    except TypeError as e:
        assert "mask" in str(e)
    else:
        raise AssertionError("attend took a mask")


def test_a_length_is_not_the_last_key():
    length = fe.T(("batch",), "int64", "length", (B,))

    def off_by_one(**kw):
        length = kw.pop("length")
        return decode(**dict(kw, until=length))

    def converted(**kw):
        length = kw.pop("length")
        return decode(**dict(kw, until=fe.last_key(length=length)))

    refused(off_by_one, "until takes last_key, got length", **dict(TYPES, length=length))
    fe.trace(converted, **dict(TYPES, length=length))


def test_heads_are_shared_as_declared():
    def wrong_share(**kw):
        q = fe.rope(x=kw["q"], positions=kw["positions"], rotary=ROT)
        return fe.attend(query=q, keys=kw["keys"], values=kw["values"], until=kw["until"], share=2)

    refused(wrong_share, "8 query heads cannot share 2 key/value heads 2 to one", **TYPES)


def test_query_and_keys_in_one_frame():
    def unrotated_query(**kw):
        keys = fe.write(into=kw["keys"], src=fe.rope(x=kw["k"], positions=kw["positions"], rotary=ROT),
                        at=kw["positions"])
        return fe.attend(query=kw["q"], keys=keys, values=kw["values"], until=kw["until"], share=4)

    refused(unrotated_query, "query is not rotated, keys are Positions", **TYPES)
    refused(lambda **kw: fe.write(into=kw["keys"], src=kw["k"], at=kw["positions"]),
            "into holds keys Positions(frame='absolute', offset=None); src is not rotated", **TYPES)
    refused(lambda **kw: fe.rope(x=fe.rope(x=kw["q"], positions=kw["positions"], rotary=ROT),
                                 positions=kw["positions"], rotary=ROT), "x was already rotated", **TYPES)


def test_a_cache_is_read_after_its_write_only():
    """The version written over may not be read: a reader holding the old one is refused (rolebench 09, 10, 14)."""
    def stale(**kw):
        q = fe.rope(x=kw["q"], positions=kw["positions"], rotary=ROT)
        fe.write(into=kw["keys"], src=fe.rope(x=kw["k"], positions=kw["positions"], rotary=ROT), at=kw["positions"])
        return fe.attend(query=q, keys=kw["keys"], values=kw["values"], until=kw["until"], share=4)

    refused(stale, "keys reads keys after write#", **TYPES)
    refused(lambda **kw: fe.write(into=kw["values"], src=kw["k"], at=kw["positions"]), "src takes value, got key",
            **TYPES)


def test_chunk_relative_positions_are_converted_or_refused():
    rel = fe.T(("tokens",), "int64", "positions", (L,), (Positions("chunk_relative", offset=3),))
    program = fe.trace(decode, **dict(TYPES, positions=rel))
    assert any("chunk-relative (offset 3); converted" in n for n in program.notes), program.notes
    d = data(pos=5)
    got = program(**dict({k: v.clone() for k, v in d.items()}, positions=torch.tensor([2])))
    assert torch.allclose(got, reference(d), atol=1e-5), "position 2 in a chunk at 3 is position 5"
    unknown = fe.T(("tokens",), "int64", "positions", (L,), (Positions("chunk_relative"),))
    refused(decode, "chunk-relative with no offset declared", **dict(TYPES, positions=unknown))
    refused(decode, "positions carries no frame", **dict(TYPES, positions=fe.T(("tokens",), "int64", "positions", (L,))))


def test_every_format_is_read_or_refused_by_name():
    """The format set is closed: a weight in a format no lowering reads is refused naming it, never read as another
    (rolebench 01: a reader that did not know the reorder)."""
    x = fe.T(("batch", "embed"), "float32", "hidden", (B, 8))
    for kind in sorted(LAYOUT_KINDS):
        layout = Layout(kind, packing="split") if kind == "q8_0" else Layout(kind)
        w = fe.T(("out", "embed"), "float32", "weight", (4, 8), (layout,))
        if fe.reads(layout) is not None:
            fe.trace(lambda **kw: fe.linear(x=kw["x"], weight=kw["w"]), x=x, w=w)
        else:
            refused(lambda **kw: fe.linear(x=kw["x"], weight=kw["w"]), f"no lowering reads the weight's format {layout}",
                    x=x, w=w)


def test_a_reader_reads_formats_not_kinds():
    """A reader of q8_0 interleaved does not read q8_0 split: the fields it names must match."""
    x = fe.T(("batch", "embed"), "float32", "hidden", (B, 8))
    fe.LINEAR_READS.append((Layout("q8_0", packing="interleaved"), "a test reader"))
    try:
        ok = fe.T(("out", "embed"), "float32", "weight", (4, 8), (Layout("q8_0", packing="interleaved"),))
        fe.trace(lambda **kw: fe.linear(x=kw["x"], weight=kw["w"]), x=x, w=ok)
        split = ok.but(facts=(Layout("q8_0", packing="split"),))
        refused(lambda **kw: fe.linear(x=kw["x"], weight=kw["w"]), "no lowering reads the weight's format", x=x,
                w=split)
    finally:
        fe.LINEAR_READS.pop()


def test_a_counter_is_read_as_a_value_or_refused():
    """A reader that holds a counter the next write moves on reads a value that is no longer there (rolebench 10);
    one given a copy taken at hand-over reads what was meant."""
    def by_reference(**kw):
        keys = fe.write(into=kw["keys"], src=fe.rope(x=kw["k"], positions=kw["positions"], rotary=ROT),
                        at=kw["positions"])
        fe.advance(counter=kw["until"])
        q = fe.rope(x=kw["q"], positions=kw["positions"], rotary=ROT)
        return fe.attend(query=q, keys=keys, values=kw["values"], until=kw["until"], share=4)

    def by_value(**kw):
        held = fe.copy(x=kw["until"])
        fe.advance(counter=kw["until"])
        keys = fe.write(into=kw["keys"], src=fe.rope(x=kw["k"], positions=kw["positions"], rotary=ROT),
                        at=kw["positions"])
        values = fe.write(into=kw["values"], src=kw["v"], at=kw["positions"])
        q = fe.rope(x=kw["q"], positions=kw["positions"], rotary=ROT)
        return fe.attend(query=q, keys=keys, values=values, until=held, share=4)

    refused(by_reference, "until reads until after advance#", **TYPES)
    d = data()
    got = fe.trace(by_value, **TYPES)(**{k: v.clone() for k, v in d.items()})
    assert torch.allclose(got, reference(d), atol=1e-5)


def test_a_fused_projection_is_split_into_what_each_part_is():
    """The query sliced from a fused QKV is a strided view; the lowerings read it as it is (rolebench 03)."""
    e, qn, kn = 16, HQ * D, HKV * D
    x = fe.T(("batch", "tokens", "embed"), "float32", "hidden", (B, L, e))
    w = fe.T(("qkv_features", "embed"), "float32", "weight", (qn + 2 * kn, e), yields="qkv")

    def step(**kw):
        q, k, v = fe.split_features(x=fe.linear(x=kw["x"], weight=kw["w"]), parts={"query": qn, "key": kn, "value": kn})
        q = fe.rope(x=fe.split_heads(x=q, heads=HQ, name="heads"), positions=kw["positions"], rotary=ROT)
        k = fe.rope(x=fe.split_heads(x=k, heads=HKV, name="kv_heads"), positions=kw["positions"], rotary=ROT)
        keys = fe.write(into=kw["keys"], src=k, at=kw["positions"])
        values = fe.write(into=kw["values"], src=fe.split_heads(x=v, heads=HKV, name="kv_heads"), at=kw["positions"])
        return fe.attend(query=q, keys=keys, values=values, until=kw["until"], share=HQ // HKV)

    program = fe.trace(step, x=x, w=w, **{k: TYPES[k] for k in ("keys", "values", "positions", "until")})
    d = data()
    xv, wv = torch.randn(B, L, e), torch.randn(qn + 2 * kn, e)
    qkv = xv @ wv.T
    d.update(q=qkv[..., :qn].reshape(B, L, HQ, D), k=qkv[..., qn:qn + kn].reshape(B, L, HKV, D),
             v=qkv[..., qn + kn:].reshape(B, L, HKV, D))
    got = program(x=xv, w=wv, keys=d["keys"].clone(), values=d["values"].clone(), positions=d["positions"],
                  until=d["until"])
    assert torch.allclose(got, reference(d), atol=1e-4), (got - reference(d)).abs().max()


def test_a_reordered_weight_is_read_in_its_new_format():
    before, after = Layout("q8_0", packing="interleaved"), Layout("q8_0", packing="split")
    fe.REORDERS[(before, after)] = lambda w: w
    try:
        w = fe.T(("out", "embed"), "float32", "weight", (4, 8), (before,))
        x = fe.T(("batch", "embed"), "float32", "hidden", (B, 8))

        def second_reader(**kw):
            fe.reorder(weight=kw["w"], to=after)
            return fe.linear(x=kw["x"], weight=kw["w"])

        refused(second_reader, "weight reads w after reorder#", x=x, w=w)
        refused(lambda **kw: fe.reorder(weight=kw["w"], to=Layout("fp8_block")), "no conversion from", x=x, w=w)
    finally:
        fe.REORDERS.clear()


def test_a_sum_is_reduced_once():
    part = fe.T(("batch", "embed"), "float32", "hidden", (B, 8), (Reduction("P"),))
    whole = part.but(facts=(Reduction("R"),))
    fe.trace(lambda **kw: fe.all_reduce(x=kw["x"]), x=part)
    refused(lambda **kw: fe.all_reduce(x=fe.all_reduce(x=kw["x"])), "reducing it would count it again", x=part)
    refused(lambda **kw: fe.add(a=kw["x"], b=kw["y"]), "adds Reduction(state='P'", x=part, y=whole)
    w = fe.T(("out", "embed"), "float32", "weight", (4, 8))
    refused(lambda **kw: fe.linear(x=kw["x"], weight=kw["w"]), "x is a partial sum", x=part, w=w)
    sharded = fe.T(("out", "embed"), "float32", "weight", (4, 8), (Reduction("S", dim=1),))
    out = fe.trace(lambda **kw: fe.linear(x=kw["x"], weight=kw["w"]), x=whole, w=sharded)
    assert "Reduction(state='P'" in out.lines()[-1], out.lines()


def test_a_quantized_value_is_dequantized_with_its_scale():
    x = fe.T(("batch", "embed"), "float32", "hidden", (B, 8), (Quantized("float8_e4m3fn", scale=0.5),))
    w = fe.T(("out", "embed"), "float32", "weight", (4, 8))
    program = fe.trace(lambda **kw: fe.linear(x=kw["x"], weight=kw["w"]), x=x, w=w)
    assert program.notes and "dequantized with its scale first" in program.notes[0], program.notes
    xv, wv = torch.randn(B, 8), torch.randn(4, 8)
    assert torch.allclose(program(x=xv, w=wv), (xv * 0.5) @ wv.T, atol=1e-6)
    refused(lambda **kw: fe.linear(x=kw["x"], weight=kw["w"]), "with no scale declared",
            x=x.but(facts=(Quantized("float8_e4m3fn"),)), w=w)


def test_properties_the_model_declares_are_honoured():
    """A softcap the chosen lowering does not honour is routed to one that does (rolebench 08, 17)."""
    capped = ModelProps(softcap=5.0)

    def step(**kw):
        q = fe.rope(x=kw["q"], positions=kw["positions"], rotary=ROT)
        k = fe.rope(x=kw["k"], positions=kw["positions"], rotary=ROT)
        keys = fe.write(into=kw["keys"], src=k, at=kw["positions"])
        values = fe.write(into=kw["values"], src=kw["v"], at=kw["positions"])
        return fe.attend(query=q, keys=keys, values=values, until=kw["until"], share=4, props=capped)

    program = fe.trace(step, {"attention": "triton"}, **TYPES)
    assert program.notes == ["attend: triton does not honour softcap; routed to torch"], program.notes
    d = data()
    d["q"] = d["q"] * 6   # large enough for the cap to bind
    got = program(**{k: v.clone() for k, v in d.items()})
    assert torch.allclose(got, reference(d, softcap=5.0), atol=1e-4)


def test_other_refusals_name_what_is_missing():
    refused(lambda **kw: fe.rope(x=kw["q"], positions=kw["positions"], rotary=Rotary("yarn", 1e6, 4.0, 32768)),
            "no lowering computes Rotary(rope_type='yarn'", **TYPES)
    multi = fe.T(("batch", "tokens", "heads", "head_dim"), "float32", "query", (B, 3, HQ, D), (ABS,))
    refused(lambda **kw: fe.attend(query=kw["q"], keys=kw["keys"], values=kw["values"], until=kw["until"], share=4),
            "a query of several tokens (or an unknown number) needs at=", **dict(TYPES, q=multi))
    refused(lambda **kw: fe.add(a=kw["q"], b=kw["k"]), "adds a query and a key", **TYPES)
    try:
        fe.rope(x=None, positions=None, rotary=ROT)
    except fe.RoleError as e:
        assert "while a program is traced" in str(e)


def test_inputs_are_checked_once_when_bound():
    program = fe.trace(decode, **TYPES)
    d = data()
    for key, bad, words in (("q", d["q"].double(), "dtype float64, the type declares float32"),
                            ("keys", torch.randn(B, HKV, S + 1, D), "dim slots is 13, the type declares 12")):
        try:
            program.bind(**dict(d, **{key: bad}))
        except fe.RoleError as e:
            assert words in str(e), (words, str(e))
        else:
            raise AssertionError(f"bound {key} {words}")


# --- the CUDA lowerings, against the torch one ----------------------------------------------------------------

def test_the_decode_lowerings_agree():
    if not torch.cuda.is_available():
        print("skip (no CUDA): triton and flex lowerings")
        return
    types = {k: (v.but(dtype="bfloat16") if v.dtype == "float32" else v) for k, v in TYPES.items()}
    d = {k: (v.cuda().bfloat16() if v.is_floating_point() else v.cuda()) for k, v in data(pos=9).items()}
    want = fe.trace(decode, {"attention": "torch"}, **types)(**{k: v.clone() for k, v in d.items()})
    for backend in ("triton", "flex"):
        got = fe.trace(decode, {"attention": backend}, **types)(**{k: v.clone() for k, v in d.items()})
        assert (got.float() - want.float()).abs().max() < 2e-2, (backend, (got.float() - want.float()).abs().max())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
