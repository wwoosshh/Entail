"""frontend.paged: a key/value pool laid out in blocks, and the operations that address it (ROADMAP M21.1).

The execution side of a serving engine addresses its cache with integers: which slot a new token's key goes to,
which blocks a sequence's keys sit in, how many of them hold written keys, which block a copy moves where. Each of
these operations takes those integers with their meaning (facts.Index, facts.Count) and checks it while the program
is traced, as the operations in ops.py check roles and formats:
  numbering   a slot is written or read in the pool it numbers (Index unit "slot", the pool's name and block
              size); a block table holds blocks of that pool; a row id numbers the table's rows. A block number, a
              row, or another pool's slot where a slot of this pool is meant is refused by name.
  length      attention reads, for each row, the tokens whose keys were written: Count("computed"). A count of the
              tokens a sequence is known to have (its prompt and what was sampled) is another length and is refused.
  readiness   a copy of blocks into a pool (move_blocks) leaves a pending pool, which nothing reads; wait() returns
              the pool to read. A pool is read in the version the last write or wait returned (graph.take).
  identity    what a stored item stands for must cover every per-request input its content depends on
              (Program.content_inputs, check_identity): a store keyed by the tokens alone serves one request's keys
              to another whose image or adapter differed.
Positions follow ops.py: absolute, or chunk-relative with an offset (converted). The lowerings are plain torch
(gather by the block table, masked attention); what only the data can say - that a tensor of indices was made with
the meaning its input declares - is looked at once, when the program is bound (graph._check_tensor).
"""
import torch

from ..facts import Count, Index, Positions
from .graph import T, fail, node, take
from .ops import _frame, _memo, _sizes

COMPUTED = Count("computed")


def _index(op, role, value, unit, pool=None, block=None):
    """The Index a traced value carries, which must be of `unit` (and of `pool`, `block` when given)."""
    ix = value.type.fact(Index)
    if ix is None:
        fail(op, f"{role} says nothing of what it numbers; declare Index({unit!r}, pool, block)")
    if ix.unit != unit:
        fail(op, f"{role} numbers {ix}; it takes a {unit}" + (f" of {pool}" if pool else ""))
    if pool is not None and ix.pool != pool:
        fail(op, f"{role} numbers {ix}; it takes a {unit} of {pool}")
    if block is not None and ix.block != block:
        fail(op, f"{role} numbers {ix}; the pool is in blocks of {block}")
    return ix


def _pool(op, role, value, kind):
    """A pool: a key or value cache with a slots dim, numbered by an Index of unit 'slot' with a block size."""
    value = take(op, role, value, kind, dims=("slots", "head_dim"))
    ix = value.type.fact(Index)
    if ix is None or ix.unit != "slot" or ix.block is None:
        fail(op, f"{role} declares no pool numbering; give it Index('slot', name, block)")
    return value, ix


def pool_type(kind, pool, block, blocks, kv_heads, head_dim, dtype="bfloat16", frame=Positions("absolute")):
    """The type of a pool of `blocks` blocks of `block` slots: (slots, kv_heads, head_dim)."""
    facts = (Index("slot", pool, block),) + ((frame,) if frame is not None and kind == "key" else ())
    return T(("slots", "kv_heads", "head_dim"), dtype, kind, (blocks * block, kv_heads, head_dim), facts)


def slots_of(*, table, rows, positions):
    """The slot each token's key goes to: table[row, position // block] * block + position % block. The table holds
    blocks of a pool (Index 'block'); rows number the table's rows (Index 'row' of the table's first dim); positions
    are absolute."""
    op = "slots_of"
    table = take(op, "table", table, "block_table")
    if len(table.type.dims) != 2:
        fail(op, f"table takes (rows, blocks), got dims {table.type.dims}")
    tix = _index(op, "table", table, "block")
    rows = take(op, "rows", rows, "row_ids", dims=("tokens",))
    _index(op, "rows", rows, "row", pool=table.type.dims[0])
    positions = take(op, "positions", positions, "positions", dims=("tokens",))
    key = ("slots_of", table.id, rows.id, positions.id)
    positions = _frame(op, "positions", positions)
    b = tix.block
    out = T(("tokens",), "int64", "slots", _sizes(rows.type)[:1], (Index("slot", tix.pool, b),))

    def run(table, rows, positions):
        blk = table[rows, torch.div(positions, b, rounding_mode="floor")].to(torch.int64)
        return (blk * b + positions % b,)

    # every layer of a step asks for the same slots: made once per program
    return _memo(key, lambda: node(op, {"table": table, "rows": rows, "positions": positions}, [out], run)[0])


def pick(*, x, at):
    """The rows of x at the token places `at` (Index 'token' of x's first dim): each sequence's last token, say."""
    op = "pick"
    x = take(op, "x", x)
    dim = x.type.dims[0]
    at = take(op, "at", at, "token_index")
    if len(at.type.dims) != 1:
        fail(op, f"at takes one dim of token places, got {at.type.dims}")
    _index(op, "at", at, "token", pool=dim)
    out = x.type.but(dims=at.type.dims + x.type.dims[1:], sizes=_sizes(at.type) + _sizes(x.type)[1:])
    return node(op, {"x": x, "at": at}, [out], lambda x, at: (x.index_select(0, at),))[0]


def paged_write(*, into, src, at):
    """Writes src's tokens into the pool `into` at the slots `at` (a slot of this pool per token), in place; returns
    the pool's next version. The rows written carry the frame the pool holds (rotated keys into a key pool)."""
    op = "paged_write"
    into, pix = _pool(op, "into", into, ("key", "value"))
    src = take(op, "src", src, into.type.kind, dims=("tokens", "head_dim"))
    at = take(op, "at", at, "slots", dims=("tokens",))
    _index(op, "at", at, "slot", pool=pix.pool, block=pix.block)
    held, carried = into.type.fact(Positions), src.type.fact(Positions)
    if held != carried:
        fail(op, f"into holds {into.type.kind}s {held or 'not rotated'}; src is {carried or 'not rotated'}")
    mapped = tuple("slots" if d == "tokens" else d for d in src.type.dims)
    if set(mapped) != set(into.type.dims):
        fail(op, f"src dims {src.type.dims} do not match into dims {into.type.dims} (tokens go to slots)")
    for d in into.type.dims:
        if d == "slots":
            continue
        a, b = into.type.size(d), src.type.size(d)
        if None not in (a, b) and a != b:
            fail(op, f"into has {d}={a}, src has {b}")
    perm = [mapped.index(d) for d in into.type.dims]
    axis = into.type.dims.index("slots")

    def run(into, src, at):
        into.index_copy_(axis, at, src.permute(perm))
        return (into,)

    return node(op, {"into": into, "src": src, "at": at}, [into.type], run, written=("into",))[0]


def paged_attend(*, query, keys, values, table, lengths, rows, at, share, scale=None):
    """Attention of each query token over the keys its row has written: for token t of row r = rows[t], the keys at
    positions 0..min(lengths[r] - 1, at[t]) found through the block table. lengths count the computed tokens of each
    row (this step's writes included); at is each token's absolute position."""
    op = "paged_attend"
    query = take(op, "query", query, "query", dims=("tokens", "head_dim"))
    keys, kix = _pool(op, "keys", keys, "key")
    values, vix = _pool(op, "values", values, "value")
    if kix != vix:
        fail(op, f"keys are numbered {kix}, values {vix}: one pool numbering for both")
    table = take(op, "table", table, "block_table")
    if len(table.type.dims) != 2:
        fail(op, f"table takes (rows, blocks), got dims {table.type.dims}")
    _index(op, "table", table, "block", pool=kix.pool, block=kix.block)
    row_dim = table.type.dims[0]
    lengths = take(op, "lengths", lengths, "length", dims=(row_dim,))
    cnt = lengths.type.fact(Count)
    if cnt != COMPUTED:
        fail(op, f"lengths count {cnt or 'nothing declared'}; attention reads the keys written: "
                 f"declare Count('computed')")
    rows = take(op, "rows", rows, "row_ids", dims=("tokens",))
    _index(op, "rows", rows, "row", pool=row_dim)
    at = _frame(op, "at", take(op, "at", at, "positions", dims=("tokens",)))
    if query.type.fact(Positions) != keys.type.fact(Positions):
        fail(op, f"query is {query.type.fact(Positions) or 'not rotated'}, keys are "
                 f"{keys.type.fact(Positions) or 'not rotated'}: they must be in one frame")
    heads = [d for d in query.type.dims if d not in ("tokens", "head_dim")]
    kv = [d for d in keys.type.dims if d not in ("slots", "head_dim")]
    if len(heads) != 1 or len(kv) != 1:
        fail(op, f"query and keys each need one heads dim; they have {query.type.dims} and {keys.type.dims}")
    hq, hkv = query.type.size(heads[0]), keys.type.size(kv[0])
    group = int(share)
    if None not in (hq, hkv) and hq != hkv * group:
        fail(op, f"{hq} query heads cannot share {hkv} key/value heads {group} to one")
    d = query.type.size("head_dim")
    sm = float(scale) if scale is not None else (float(d) ** -0.5 if d else None)
    if sm is None:
        fail(op, "head_dim is not known: give scale=")
    q_to = [query.type.dims.index(x) for x in ("tokens", heads[0], "head_dim")]
    q_back = [q_to.index(i) for i in range(3)]
    k_to = [keys.type.dims.index(x) for x in ("slots", kv[0], "head_dim")]
    b = kix.block
    plan = _row_plan(rows, lengths)

    def run(query, keys, values, table, plan, at):
        # one row at a time: its keys gathered through its table row, its tokens' queries, a causal mask by position
        q = query.permute(q_to)                                    # [T, H, D]
        k_all, v_all = keys.permute(k_to), values.permute(k_to)    # [S, KVH, D]
        out = torch.empty_like(q)
        tab = table.to(torch.int64)
        for r, idx, n in plan:
            j = torch.arange(n, device=q.device)
            slots = tab[r, j // b] * b + j % b
            k, v = k_all.index_select(0, slots), v_all.index_select(0, slots)          # [n, KVH, D]
            sel = torch.tensor(idx, device=q.device)
            qr = q.index_select(0, sel)                                                 # [t, H, D]
            allowed = j.view(1, n) <= at.index_select(0, sel).view(-1, 1)               # [t, n]
            o = torch.nn.functional.scaled_dot_product_attention(
                qr.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0),
                attn_mask=allowed.view(1, 1, len(idx), n), scale=sm, enable_gqa=group > 1)
            out.index_copy_(0, sel, o.squeeze(0).transpose(0, 1))
        return (out.permute(q_back),)

    out = T(query.type.dims, query.type.dtype, "attended", _sizes(query.type))
    return node(op, {"query": query, "keys": keys, "values": values, "table": table, "plan": plan, "at": at}, [out],
                run, note="lowered: torch (per row, keys gathered by block table)")[0]


def _row_plan(rows, lengths):
    """Which tokens each row has and how many keys it reads, read to the host once per step (every layer's attention
    uses it): [(row, token places, computed keys)]."""
    def make():
        def run(rows, lengths):
            row_of, lens = rows.tolist(), lengths.tolist()
            return ([(r, [t for t, rr in enumerate(row_of) if rr == r], int(lens[r])) for r in sorted(set(row_of))],)

        return node("row_plan", {"rows": rows, "lengths": lengths}, [T((), "int64", "row_plan")], run)[0]

    return _memo(("row_plan", rows.id, lengths.id), make)


def move_blocks(*, into, src, src_blocks, dst_blocks):
    """Copies whole blocks from the pool `src` to the pool `into` (src_blocks[i] -> dst_blocks[i]); the copy may be
    in flight when this returns, so the result is a pending pool, which nothing reads until wait() returns it."""
    op = "move_blocks"
    src, six = _pool(op, "src", src, ("key", "value"))
    into, dix = _pool(op, "into", into, src.type.kind)
    if six.block != dix.block:
        fail(op, f"src is in blocks of {six.block}, into in blocks of {dix.block}")
    src_blocks = take(op, "src_blocks", src_blocks, "block_ids", dims=("pairs",))
    _index(op, "src_blocks", src_blocks, "block", pool=six.pool, block=six.block)
    dst_blocks = take(op, "dst_blocks", dst_blocks, "block_ids", dims=("pairs",))
    _index(op, "dst_blocks", dst_blocks, "block", pool=dix.pool, block=dix.block)
    if None not in (src_blocks.type.size("pairs"), dst_blocks.type.size("pairs")) and \
            src_blocks.type.size("pairs") != dst_blocks.type.size("pairs"):
        fail(op, f"{src_blocks.type.size('pairs')} source blocks for {dst_blocks.type.size('pairs')} destinations")
    if src.type.fact(Positions) != into.type.fact(Positions):
        fail(op, f"src holds {src.type.fact(Positions) or 'not rotated'} values, into "
                 f"{into.type.fact(Positions) or 'not rotated'}")
    b = six.block
    axis_s, axis_d = src.type.dims.index("slots"), into.type.dims.index("slots")

    def run(into, src, src_blocks, dst_blocks):
        off = torch.arange(b, device=src_blocks.device)
        s = (src_blocks.view(-1, 1) * b + off).reshape(-1).to(src.device)
        dd = (dst_blocks.view(-1, 1) * b + off).reshape(-1).to(into.device)
        into.index_copy_(axis_d, dd, src.index_select(axis_s, s).to(into.device))
        return (into,)

    pending = into.type.but(kind=f"pending_{into.type.kind}")
    return node(op, {"into": into, "src": src, "src_blocks": src_blocks, "dst_blocks": dst_blocks}, [pending], run,
                written=("into",))[0]


def wait(*, pending):
    """The pool a block copy wrote, once the copy is done: what readers may read. move_blocks's lowering copies on the
    current stream (a copy between devices blocks), so a later reader is ordered after it and nothing is left to do at
    run time; the operation is what keeps a reader from coming before it in the program."""
    op = "wait"
    pending = take(op, "pending", pending, ("pending_key", "pending_value"))
    kind = pending.type.kind[len("pending_"):]

    def run(pending):
        return (pending,)

    return node(op, {"pending": pending}, [pending.type.but(kind=kind)], run, written=("pending",))[0]


def check_identity(program, covers, store="paged_write"):
    """A store keyed by identity (a prefix cache) must key what it stores on every per-request input the stored
    content depends on. `covers` names the program inputs its key is made from (the token ids among them); the program
    says which inputs reach what it writes (Program.content_inputs). Weights are the program's own, not per request."""
    needed = program.content_inputs(store)
    missing = sorted(needed - set(covers))
    if missing:
        fail("check_identity", f"what {store} stores depends on {', '.join(missing)}, which the key does not cover "
                               f"(it covers {', '.join(sorted(covers)) or 'nothing'})")
    return needed
