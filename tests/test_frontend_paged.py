"""Tests for layer B's execution side (ROADMAP M21.1): integers that keep their meaning (units), a key/value pool in
blocks (paged): every rule with a program it passes and one it refuses (with the words of the refusal), the bind-time
check of index tensors, the identity a store must cover, and the paged attention against a plain reference.
Run: python tests/test_frontend_paged.py"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import frontend as fe  # noqa: E402
from entail.facts import Count, Index, Positions, Rotary  # noqa: E402
from entail.frontend import units as U  # noqa: E402

BLOCK, BLOCKS, KVH, HQ, D = 4, 6, 2, 4, 8
TOK = 4                                   # tokens in the step: row 0 decodes one, row 1 a chunk of three
ROT = Rotary("default", 10000.0)
ABS = Positions("absolute")
GPU_SLOT, GPU_BLOCK = Index("slot", "gpu", BLOCK), Index("block", "gpu", BLOCK)
ROW = Index("row", "batch")
COMPUTED, KNOWN = Count("computed"), Count("known")


def T(dims, kind, sizes, *facts, dtype="float32"):
    return fe.T(dims, dtype, kind, sizes, tuple(facts))


KEYS = fe.pool_type("key", "gpu", BLOCK, BLOCKS, KVH, D, dtype="float32")
VALUES = fe.pool_type("value", "gpu", BLOCK, BLOCKS, KVH, D, dtype="float32")
TABLE = T(("batch", "blocks"), "block_table", (2, 3), GPU_BLOCK, dtype="int64")
ROWS = T(("tokens",), "row_ids", (TOK,), ROW, dtype="int64")
POS = T(("tokens",), "positions", (TOK,), ABS, dtype="int64")
LENGTHS = T(("batch",), "length", (2,), COMPUTED, dtype="int64")
Q = T(("tokens", "heads", "head_dim"), "query", (TOK, HQ, D))
K = T(("tokens", "kv_heads", "head_dim"), "key", (TOK, KVH, D))
V = T(("tokens", "kv_heads", "head_dim"), "value", (TOK, KVH, D))


def step(*, q, k, v, keys, values, table, rows, positions, lengths):
    """One step: rotate, find each token's slot through the block table, write, attend over what each row wrote."""
    q = fe.rope(x=q, positions=positions, rotary=ROT)
    k = fe.rope(x=k, positions=positions, rotary=ROT)
    at = fe.slots_of(table=table, rows=rows, positions=positions)
    keys = fe.paged_write(into=keys, src=k, at=at)
    values = fe.paged_write(into=values, src=v, at=at)
    return fe.paged_attend(query=q, keys=keys, values=values, table=table, lengths=lengths, rows=rows,
                           at=positions, share=HQ // KVH)


TYPES = dict(q=Q, k=K, v=V, keys=KEYS, values=VALUES, table=TABLE, rows=ROWS, positions=POS, lengths=LENGTHS)


def refused(fn, words, **types):
    try:
        fe.trace(fn, **types)
    except fe.RoleError as e:
        assert words in str(e), (words, str(e))
        return str(e)
    raise AssertionError(f"traced, but should have been refused with: {words}")


def raises(fn, words):
    try:
        fn()
    except fe.RoleError as e:
        assert words in str(e), (words, str(e))
        return str(e)
    raise AssertionError(f"ran, but should have been refused with: {words}")


# --- the step's data and a plain reference -----------------------------------------------------------------------

PRIOR = (5, 2)                            # tokens each row computed before this step
NEW_ROWS, NEW_POS = [0, 1, 1, 1], [5, 2, 3, 4]
TABLE_ROWS = [[3, 0, 2], [5, 1, 4]]       # row 0: positions 0-3 in block 3, 4-7 in block 0; row 1: blocks 5, 1


def data(seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    keys, values = r(BLOCKS * BLOCK, KVH, D), r(BLOCKS * BLOCK, KVH, D)
    return dict(q=r(TOK, HQ, D), k=r(TOK, KVH, D), v=r(TOK, KVH, D), keys=keys, values=values,
                table=U.tensor([U.nums(row, GPU_BLOCK) for row in TABLE_ROWS]),
                rows=U.tensor(U.nums(NEW_ROWS, ROW)),
                positions=torch.tensor(NEW_POS),
                lengths=U.tensor([U.num(PRIOR[0] + 1, COMPUTED), U.num(PRIOR[1] + 3, COMPUTED)]))


def rope_ref(x, pos):
    inv = 1.0 / (ROT.theta ** (torch.arange(0, D, 2).float() / D))
    f = pos.float()[:, None] * inv
    emb = torch.cat((f, f), dim=-1)
    cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]
    rot = torch.cat((-x[..., D // 2:], x[..., :D // 2]), dim=-1)
    return x * cos + rot * sin


def slot(row, p):
    return TABLE_ROWS[row][p // BLOCK] * BLOCK + p % BLOCK


def reference(d):
    keys, values = d["keys"].clone(), d["values"].clone()
    pos = torch.tensor(NEW_POS)
    k, q = rope_ref(d["k"], pos), rope_ref(d["q"], pos)
    for t, (row, p) in enumerate(zip(NEW_ROWS, NEW_POS)):
        keys[slot(row, p)], values[slot(row, p)] = k[t], d["v"][t]
    out = torch.empty(TOK, HQ, D)
    for t, (row, p) in enumerate(zip(NEW_ROWS, NEW_POS)):
        idx = [slot(row, j) for j in range(p + 1)]
        kk = keys[idx].repeat_interleave(HQ // KVH, dim=1)          # [p+1, HQ, D]
        vv = values[idx].repeat_interleave(HQ // KVH, dim=1)
        s = torch.einsum("hd,nhd->hn", q[t], kk) * D ** -0.5
        out[t] = torch.einsum("hn,nhd->hd", torch.softmax(s, -1), vv)
    return out, keys, values


# --- a program that agrees --------------------------------------------------------------------------------------

def test_a_paged_step_agrees_with_the_reference():
    d = data()
    want, want_keys, want_values = reference(d)
    got = fe.trace(step, **TYPES)(**{k: (v.clone() if k in ("keys", "values") else v) for k, v in d.items()})
    assert (got - want).abs().max() < 1e-5, (got - want).abs().max()


def test_the_pool_is_written_where_the_table_says():
    d = data(1)
    _, want_keys, want_values = reference(d)
    keys, values = d["keys"].clone(), d["values"].clone()
    fe.trace(step, **TYPES)(**{**d, "keys": keys, "values": values})
    assert torch.allclose(keys, want_keys, atol=1e-6) and torch.allclose(values, want_values)


# --- numbering ----------------------------------------------------------------------------------------------------

def test_a_slot_of_another_pool_is_refused():
    other = T(("batch", "blocks"), "block_table", (2, 3), Index("block", "cpu", BLOCK), dtype="int64")
    refused(step, "at numbers slot of cpu (blocks of 4); it takes a slot of gpu", **{**TYPES, "table": other})


def test_a_block_number_where_a_slot_is_meant_is_refused():
    blocks = T(("tokens",), "slots", (TOK,), GPU_BLOCK, dtype="int64")

    def prog(*, k, keys, blocks, positions):
        return fe.paged_write(into=keys, src=fe.rope(x=k, positions=positions, rotary=ROT), at=blocks)

    refused(prog, "at numbers block of gpu (blocks of 4); it takes a slot of gpu", k=K, keys=KEYS, blocks=blocks,
            positions=POS)


def test_rows_of_another_table_are_refused():
    rows = T(("tokens",), "row_ids", (TOK,), Index("row", "other"), dtype="int64")
    refused(step, "rows numbers row of other; it takes a row of batch", **{**TYPES, "rows": rows})


def test_an_index_with_no_meaning_is_refused():
    plain = T(("batch", "blocks"), "block_table", (2, 3), dtype="int64")
    refused(step, "table says nothing of what it numbers", **{**TYPES, "table": plain})


def test_a_table_of_other_block_size_is_refused():
    t8 = T(("batch", "blocks"), "block_table", (2, 3), Index("block", "gpu", 8), dtype="int64")
    refused(step, "at numbers slot of gpu (blocks of 8); the pool is in blocks of 4", **{**TYPES, "table": t8})


# --- length -------------------------------------------------------------------------------------------------------

def test_attention_lengths_must_count_computed_tokens():
    known = T(("batch",), "length", (2,), KNOWN, dtype="int64")
    refused(step, "lengths count known tokens; attention reads the keys written", **{**TYPES, "lengths": known})
    undeclared = T(("batch",), "length", (2,), dtype="int64")
    refused(step, "lengths count nothing declared", **{**TYPES, "lengths": undeclared})


# --- readiness ----------------------------------------------------------------------------------------------------

CPU_KEYS = fe.pool_type("key", "cpu", BLOCK, BLOCKS, KVH, D, dtype="float32")
SRC_BLOCKS = T(("pairs",), "block_ids", (2,), Index("block", "cpu", BLOCK), dtype="int64")
DST_BLOCKS = T(("pairs",), "block_ids", (2,), GPU_BLOCK, dtype="int64")


def load_then_attend(waited):
    def prog(*, q, keys, values, cpu_keys, src_blocks, dst_blocks, table, rows, positions, lengths):
        keys = fe.move_blocks(into=keys, src=cpu_keys, src_blocks=src_blocks, dst_blocks=dst_blocks)
        if waited:
            keys = fe.wait(pending=keys)
        q = fe.rope(x=q, positions=positions, rotary=ROT)
        return fe.paged_attend(query=q, keys=keys, values=values, table=table, lengths=lengths, rows=rows,
                               at=positions, share=HQ // KVH)
    return prog


LOAD_TYPES = dict(q=Q, keys=KEYS, values=VALUES, cpu_keys=CPU_KEYS, src_blocks=SRC_BLOCKS, dst_blocks=DST_BLOCKS,
                  table=TABLE, rows=ROWS, positions=POS, lengths=LENGTHS)


def test_a_pool_being_copied_into_is_read_only_after_its_wait():
    refused(load_then_attend(False), "keys takes key, got pending_key", **LOAD_TYPES)
    fe.trace(load_then_attend(True), **LOAD_TYPES)


def test_a_copy_reads_and_writes_blocks_of_the_right_pools():
    swapped = dict(LOAD_TYPES, src_blocks=DST_BLOCKS, dst_blocks=SRC_BLOCKS)
    refused(load_then_attend(True), "src_blocks numbers block of gpu (blocks of 4); it takes a block of cpu",
            **swapped)
    cpu8 = fe.pool_type("key", "cpu", 8, 3, KVH, D, dtype="float32")
    refused(load_then_attend(True), "src is in blocks of 8, into in blocks of 4", **dict(LOAD_TYPES, cpu_keys=cpu8))


def test_a_copy_moves_the_blocks_it_names():
    d = data(2)
    cpu = torch.randn(BLOCKS * BLOCK, KVH, D)
    keys = d["keys"].clone()
    src = U.tensor(U.nums([4, 1], Index("block", "cpu", BLOCK)))
    dst = U.tensor(U.nums([0, 2], GPU_BLOCK))

    def prog(*, keys, cpu_keys, src_blocks, dst_blocks):
        return fe.wait(pending=fe.move_blocks(into=keys, src=cpu_keys, src_blocks=src_blocks, dst_blocks=dst_blocks))

    out = fe.trace(prog, keys=KEYS, cpu_keys=CPU_KEYS, src_blocks=SRC_BLOCKS, dst_blocks=DST_BLOCKS)(
        keys=keys, cpu_keys=cpu, src_blocks=src, dst_blocks=dst)
    assert torch.equal(out[0:4], cpu[16:20]) and torch.equal(out[8:12], cpu[4:8])
    assert torch.equal(out[4:8], d["keys"][4:8])


# --- identity -----------------------------------------------------------------------------------------------------

HIDDEN = 16


def project_and_store(with_image):
    def prog(*, tokens, embed_w, k_w, image, keys, table, rows, positions):
        h = fe.embed(tokens=tokens, table=embed_w)
        if with_image:
            h = fe.add(a=h, b=image)
        k = fe.split_heads(x=fe.linear(x=h, weight=k_w), heads=KVH, name="kv_heads")
        k = fe.rope(x=k, positions=positions, rotary=ROT)
        at = fe.slots_of(table=table, rows=rows, positions=positions)
        return fe.paged_write(into=keys, src=k, at=at)
    return prog


ID_TYPES = dict(tokens=T(("tokens",), "token_ids", (TOK,), dtype="int64"),
                embed_w=T(("vocab", "hidden"), "weight", (32, HIDDEN)),
                k_w=fe.T(("key_features", "hidden"), "float32", "weight", (KVH * D, HIDDEN), (), "key"),
                image=T(("tokens", "hidden"), "hidden", (TOK, HIDDEN)),
                keys=KEYS, table=TABLE, rows=ROWS, positions=POS)


def test_a_store_key_must_cover_every_per_request_input_of_the_content():
    plain = fe.trace(project_and_store(False), **ID_TYPES)
    assert plain.content_inputs("paged_write") == {"tokens", "positions"}
    fe.check_identity(plain, covers=("tokens", "positions"))
    with_image = fe.trace(project_and_store(True), **ID_TYPES)
    assert with_image.content_inputs("paged_write") == {"tokens", "positions", "image"}
    raises(lambda: fe.check_identity(with_image, covers=("tokens", "positions")),
           "what paged_write stores depends on image, which the key does not cover")
    fe.check_identity(with_image, covers=("tokens", "positions", "image"))


# --- positions ----------------------------------------------------------------------------------------------------

def test_chunk_relative_positions_with_an_offset_are_made_absolute():
    rel = T(("tokens",), "positions", (TOK,), Positions("chunk_relative", 2), dtype="int64")
    prog = fe.trace(step, **{**TYPES, "positions": rel})
    assert any("chunk-relative" in n for n in prog.notes), prog.notes
    rel_none = T(("tokens",), "positions", (TOK,), Positions("chunk_relative"), dtype="int64")
    refused(step, "is chunk-relative with no offset declared", **{**TYPES, "positions": rel_none})


# --- bind: index tensors carry the meaning they were made with ---------------------------------------------------

def test_bind_compares_an_index_tensor_with_its_declared_meaning():
    d = data()
    prog = fe.trace(step, **TYPES)
    raises(lambda: prog(**{**d, "table": torch.tensor(TABLE_ROWS)}),
           "program input table: the type declares block of gpu (blocks of 4); the tensor was made without a "
           "declared meaning")
    cpu_table = U.tensor([U.nums(row, Index("block", "cpu", BLOCK)) for row in TABLE_ROWS])
    raises(lambda: prog(**{**d, "table": cpu_table}), "the tensor was made meaning block of cpu (blocks of 4)")
    known = U.tensor([U.num(6, KNOWN), U.num(5, KNOWN)])
    raises(lambda: prog(**{**d, "lengths": known}), "the type declares computed tokens; the tensor was made "
                                                    "meaning known tokens")


# --- host-side integers -------------------------------------------------------------------------------------------

def test_counts_of_one_length_add_and_two_lengths_do_not():
    computed, known = U.num(5, COMPUTED), U.num(6, KNOWN)
    assert computed + U.num(3, COMPUTED) == 8 and (computed + 3).fact == COMPUTED
    raises(lambda: computed + known, "computed tokens and known tokens are two lengths of a sequence")
    raises(lambda: known - computed, "two lengths of a sequence")
    assert known > computed and (known - 1).fact == KNOWN      # counts compare; an offset keeps the length
    assert isinstance(computed * 2, int) and not isinstance(computed * 2, U.Num)


def test_indices_follow_their_numbering():
    blk = U.num(3, GPU_BLOCK)
    first = blk * BLOCK
    assert first == 12 and first.fact == GPU_SLOT
    raises(lambda: blk * 8, "a block of 4 slots times 8: only its block size makes it its first slot")
    nxt = first + 1
    assert nxt.fact == GPU_SLOT and (nxt // BLOCK).fact == GPU_BLOCK and nxt % BLOCK == 1
    raises(lambda: nxt // 2, "only a slot of a pool in blocks, by its block size, gives its block")
    raises(lambda: first + nxt, "adds slot of gpu (blocks of 4) to slot of gpu (blocks of 4)")
    host = U.num(12, Index("slot", "cpu", BLOCK))
    raises(lambda: first == host, "compare (==): slot of gpu (blocks of 4) with slot of cpu (blocks of 4)")
    assert nxt - first == 1 and not isinstance(nxt - first, U.Num)
    raises(lambda: first - host, "subtracts slot of cpu (blocks of 4) from slot of gpu (blocks of 4)")


def test_functions_state_what_they_take_and_return():
    @U.takes(upto=COMPUTED, block=GPU_BLOCK)
    def store(*, upto, block):
        return int(upto) // BLOCK

    @U.returns(KNOWN)
    def known_tokens(tokens):
        return len(tokens)

    assert store(upto=U.num(8, COMPUTED), block=U.num(1, GPU_BLOCK)) == 2
    n = known_tokens([1, 2, 3])
    assert n == 3 and n.fact == KNOWN
    raises(lambda: store(upto=n, block=U.num(1, GPU_BLOCK)), "store(upto=): takes computed tokens, got known tokens")
    raises(lambda: store(upto=8, block=U.num(1, GPU_BLOCK)), "takes computed tokens, got a plain integer (8)")
    raises(lambda: store(upto=U.num(8, COMPUTED), block=U.num(1, Index("block", "cpu", BLOCK))),
           "store(block=): takes block of gpu (blocks of 4), got block of cpu (blocks of 4)")
    raises(lambda: U.tensor([U.num(1, KNOWN), U.num(2, COMPUTED)]), "one meaning is needed")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
