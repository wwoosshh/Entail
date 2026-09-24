"""frontend.ops: the operations, each with its roles, its type rule and its lowerings (ROADMAP M8.1).

Every operation takes its arguments by keyword - the keyword is the role - and checks while the program is traced:
what each argument is (its kind: a key is not a value), the named dims it needs (a lowering permutes by name, so a
transposed tensor is never misread), and the facts it must carry or changes:
  format      a weight's Layout must be one the operation reads (LINEAR_READS: formats, matched on every field the
              reader names - q8_0 interleaved is not q8_0 split). The set is closed: a format no lowering reads is
              refused by name, never read as another (rolebench 01: a reader that did not know a reorder)
  reduction   a partial sum is reduced once (all_reduce: P -> R); a replicated value reduced again, a partial sum
              fed to a linear map, or a partial and a replicated value added, are refused (rolebench 04)
  frame       positions are absolute or chunk-relative; RoPE, the cache and attention take absolute ones; a
              chunk-relative one with its offset is converted (resolved, rolebench 05), without it refused
  range       attention takes `until`, the last key index, inclusive by rule; a length (exclusive) is another kind
              and must be turned into one (last_key): the off-by-one of WEEK4_NOTES 7.1 cannot be written
  properties  what the model declares (softcap, a sliding window) must be honoured by the attention lowering; the
              trace routes to one that does (resolved) or refuses (rolebench 06, 08, 17)
  time        a value written in place is read in its next version only (graph.take; rolebench 09, 10, 14)
  scale       a quantized value with a known scale is dequantized before a consumer of plain values (resolved,
              rolebench 16)
A repair made while tracing is the resolution the library makes at run time, made once (graph.said). Work every
layer would repeat on the same inputs - RoPE's cos and sin, the valid length, a block mask - is made once per
program (graph.memo).
"""
from dataclasses import fields

import torch
import torch.nn.functional as F

from ..facts import Layout, ModelProps, Positions, Quantized, Reduction, Rotary
from .graph import T, current, fail, node, option, said, take

# the formats a linear map reads: (a Layout pattern - the fields it names must match - , how it is lowered)
LINEAR_READS = [(Layout("dense"), "torch.nn.functional.linear"),
                (Layout("int4_packed"), "torch.nn.functional.linear (torchao's int4 weight runs its own kernel)")]
ROPE_TYPES = {"default"}
ATTENTION = ("torch", "flex", "triton")
# what each lowering applies of what a model declares: its block mask or kernel reads the valid range only, so
# flex and triton honour neither a softcap nor a sliding window (a window the table claimed for flex was not in its
# mask - found in M8.3: the table must say what the code does)
ATTENTION_HONOURS = {"torch": frozenset({"softcap", "sliding_window"}), "flex": frozenset(), "triton": frozenset()}
REORDERS = {}   # (Layout before, Layout after) -> function(weight tensor) that rewrites it in place


def _sizes(t: T):
    return t.sizes or (None,) * len(t.dims)


def reads(layout):
    """How a linear map reads a weight stored as `layout`, or None: the first pattern whose named fields match."""
    for pattern, how in LINEAR_READS:
        if all(getattr(pattern, f.name) is None or getattr(pattern, f.name) == getattr(layout, f.name)
               for f in fields(pattern)):
            return how
    return None


def _memo(key, make):
    g = current()
    memo = g.options.setdefault("_memo", {})
    if key not in memo:
        memo[key] = make()
    return memo[key]


def _frame(op, role, positions):
    """Absolute positions: as declared, converted from a chunk-relative frame with a known offset (said), or refused."""
    frame = positions.type.fact(Positions)
    if frame is None:
        fail(op, f"{role} carries no frame: declare Positions('absolute'), or 'chunk_relative' with its offset")
    if frame.frame == "absolute":
        return positions
    if frame.offset is None:
        fail(op, f"{role} is chunk-relative with no offset declared; it cannot be made absolute")
    said(f"{op}: {role} was chunk-relative (offset {frame.offset}); converted to absolute positions")
    return to_absolute(positions=positions)


# --- values in, values out ---------------------------------------------------------------------------------------

def embed(*, tokens, table):
    op = "embed"
    tokens = take(op, "tokens", tokens, "token_ids")
    table = take(op, "table", table, "weight")
    if len(table.type.dims) != 2:
        fail(op, f"table takes a (vocab, feature) weight, got dims {table.type.dims}")
    feature = table.type.dims[1]
    out = T(tokens.type.dims + (feature,), table.type.dtype, "hidden", _sizes(tokens.type) + (table.type.size(feature),))
    return node(op, {"tokens": tokens, "table": table}, [out],
                lambda tokens, table: (F.embedding(tokens, table),))[0]


def rms_norm(*, x, weight, eps):
    """Normalizes over the dim the weight is named by (the model's hidden size, or a head's dim)."""
    op = "rms_norm"
    weight = take(op, "weight", weight, "weight")
    if len(weight.type.dims) != 1:
        fail(op, f"weight takes one dim, got {weight.type.dims}")
    over = weight.type.dims[0]
    x = take(op, "x", x, None, dims=(over,))
    if x.type.kind == "weight":
        fail(op, "x takes a value, got a weight")
    if None not in (x.type.size(over), weight.type.size(over)) and x.type.size(over) != weight.type.size(over):
        fail(op, f"x has {over}={x.type.size(over)}, weight has {weight.type.size(over)}")
    axis, ndim = x.type.dims.index(over), len(x.type.dims)
    shape = [1] * ndim
    shape[axis] = -1

    def run(x, weight, eps):
        x32 = x.float()
        y = x32 * torch.rsqrt(x32.pow(2).mean(axis, keepdim=True) + eps)
        return (weight.view(shape) * y.to(x.dtype),)

    return node(op, {"x": x, "weight": weight, "eps": float(eps)}, [x.type], run)[0]


def dequantize(*, x, to="bfloat16"):
    op = "dequantize"
    x = take(op, "x", x)
    q = x.type.fact(Quantized)
    if q is None or q.scale is None:
        fail(op, f"x declares no quantization with a scale ({x.type})")
    dtype = getattr(torch, to)
    return node(op, {"x": x, "scale": float(q.scale)}, [x.type.without(Quantized).but(dtype=to)],
                lambda x, scale: ((x.float() * scale).to(dtype),))[0]


def linear(*, x, weight):
    """x @ weight.T over x's last dim, which must be the weight's input dim by name. The result is what the weight
    yields (a query projection yields a query)."""
    op = "linear"
    weight = take(op, "weight", weight, "weight")
    if len(weight.type.dims) != 2:
        fail(op, f"weight takes (out, in) dims, got {weight.type.dims}")
    out_dim, in_dim = weight.type.dims
    layout = weight.type.fact(Layout) or Layout("dense")
    how = reads(layout)
    if how is None:
        fail(op, f"no lowering reads the weight's format {layout} (read: "
                 f"{', '.join(str(p) for p, _ in LINEAR_READS)}); convert it or add a lowering")
    x = take(op, "x", x, None, dims=(in_dim,))
    if x.type.dims[-1] != in_dim:
        fail(op, f"x's last dim must be {in_dim}, which the map contracts; x has {x.type.dims}")
    if None not in (x.type.size(in_dim), weight.type.size(in_dim)) and x.type.size(in_dim) != weight.type.size(in_dim):
        fail(op, f"x has {in_dim}={x.type.size(in_dim)}, weight has {weight.type.size(in_dim)}")
    q = x.type.fact(Quantized)
    if q is not None:
        if q.scale is None:
            fail(op, f"x is quantized ({q}) with no scale declared; a linear map takes plain values")
        said(f"{op}: x was quantized ({q}); dequantized with its scale first")
        x = dequantize(x=x, to=x.type.dtype if x.type.dtype in ("bfloat16", "float16", "float32") else "bfloat16")
    red = x.type.fact(Reduction)
    if red is not None and red.state == "P":
        fail(op, "x is a partial sum (Reduction P): all_reduce it before a linear map")
    wred = weight.type.fact(Reduction)
    out_red = Reduction("P") if wred is not None and wred.state == "S" and wred.dim == 1 else red
    out = T(x.type.dims[:-1] + (out_dim,), x.type.dtype, weight.type.yields or "hidden",
            _sizes(x.type)[:-1] + (weight.type.size(out_dim),), (out_red,) if out_red is not None else ())
    return node(op, {"x": x, "weight": weight}, [out], lambda x, weight: (F.linear(x, weight),), note=how)[0]


def split_features(*, x, parts):
    """A fused projection's output split into what each part is: parts={"query": 4096, "key": 1024, ...} in order.
    The parts are views; every consumer's lowering reads them as they are laid out (rolebench 03)."""
    op = "split_features"
    x = take(op, "x", x)
    n = _sizes(x.type)[-1]
    if n is not None and sum(parts.values()) != n:
        fail(op, f"the parts add up to {sum(parts.values())} features, x has {n}")
    outs = [x.type.but(dims=x.type.dims[:-1] + (f"{kind}_features",), sizes=_sizes(x.type)[:-1] + (size,),
                       kind=kind) for kind, size in parts.items()]
    sizes = list(parts.values())
    return node(op, {"x": x}, outs, lambda x: tuple(torch.split(x, sizes, dim=-1)))


def split_heads(*, x, heads, name, dim="head_dim"):
    """(..., features) -> (..., name, dim) with `heads` heads."""
    op = "split_heads"
    x = take(op, "x", x)
    n = _sizes(x.type)[-1]
    if n is not None and n % heads:
        fail(op, f"{n} features do not split into {heads} heads")
    out = x.type.but(dims=x.type.dims[:-1] + (name, dim),
                     sizes=_sizes(x.type)[:-1] + (heads, n // heads if n is not None else None))
    return node(op, {"x": x, "heads": int(heads)}, [out], lambda x, heads: (x.unflatten(-1, (heads, -1)),))[0]


def merge_heads(*, x, name):
    """(..., heads, dim) -> (..., name)."""
    op = "merge_heads"
    x = take(op, "x", x)
    a, b = _sizes(x.type)[-2:]
    out = x.type.but(dims=x.type.dims[:-2] + (name,), sizes=_sizes(x.type)[:-2] + (a * b if None not in (a, b) else None,),
                     kind="hidden")
    return node(op, {"x": x}, [out], lambda x: (x.flatten(-2),))[0]


def _cos_sin(positions, rotary, head_dim, dtype):
    """RoPE's cos and sin for these positions, made once per program: (positions' dims..., head_dim)."""
    def make():
        def run(positions, theta, dim):
            inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64, device=positions.device).float() / dim))
            freqs = positions.float()[..., None] * inv
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().to(dtype), emb.sin().to(dtype)

        t = T(positions.type.dims + ("head_dim",), str(dtype).replace("torch.", ""), "data",
              _sizes(positions.type) + (head_dim,))
        return node("rope_angles", {"positions": positions, "theta": float(rotary.theta), "dim": int(head_dim)},
                    [t, t], run)

    return _memo(("rope", positions.id, rotary, head_dim, str(dtype)), make)


def rope(*, x, positions, rotary):
    """Rotates a query or a key by its absolute positions, as the model's Rotary declares."""
    op = "rope"
    x = take(op, "x", x, ("query", "key"), dims=("tokens", "head_dim"))
    positions = take(op, "positions", positions, "positions", dims=("tokens",))
    if not isinstance(rotary, Rotary):
        fail(op, f"rotary takes the model's Rotary declaration, got {type(rotary).__name__}")
    if rotary.rope_type not in ROPE_TYPES:
        fail(op, f"no lowering computes {rotary} (computed: {', '.join(sorted(ROPE_TYPES))})")
    if rotary.theta is None:
        fail(op, f"{rotary} declares no base (theta)")
    if x.type.fact(Positions) is not None:
        fail(op, f"x was already rotated ({x.type.fact(Positions)})")
    if x.type.dims[-1] != "head_dim":
        fail(op, f"x's last dim must be head_dim; x has {x.type.dims}")
    positions = _frame(op, "positions", positions)
    d = x.type.size("head_dim")
    if d is None or d % 2:
        fail(op, f"head_dim must be a known even size, got {d}")
    extra = [p for p in positions.type.dims if p not in x.type.dims]
    if extra:
        fail(op, f"positions have dims {extra} that x does not")
    cos, sin = _cos_sin(positions, rotary, d, getattr(torch, x.type.dtype))
    # the angles' dims (positions' dims, then head_dim) laid out along x's dims, ones elsewhere
    order = [positions.type.dims.index(dim) if dim in positions.type.dims else None for dim in x.type.dims[:-1]]
    have = [i for i in order if i is not None]
    if have != sorted(have):
        fail(op, f"positions' dims {positions.type.dims} come in another order in x {x.type.dims}")
    view = [None if i is None else i for i in order]

    def run(x, cos, sin):
        shape = [cos.shape[i] if i is not None else 1 for i in view] + [cos.shape[-1]]
        c, s = cos.reshape(shape), sin.reshape(shape)
        half = x.shape[-1] // 2
        rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return (x * c + rotated * s,)

    out = x.type.but(facts=(Positions("absolute"),))
    return node(op, {"x": x, "cos": cos, "sin": sin}, [out], run)[0]


def write(*, into, src, at):
    """Writes src (tokens) into a cache (slots) at the absolute positions `at`, in place; returns the cache's next
    version. What the cache declares it holds (its frame) the rows written must carry."""
    op = "write"
    into = take(op, "into", into, ("key", "value"), dims=("slots", "head_dim"))
    src = take(op, "src", src, into.type.kind, dims=("tokens", "head_dim"))
    at = take(op, "at", at, "positions", dims=("tokens",))
    at = _frame(op, "at", at)
    held, carried = into.type.fact(Positions), src.type.fact(Positions)
    if held != carried:
        fail(op, f"into holds {into.type.kind}s {held or 'not rotated'}; src is {carried or 'not rotated'}")
    mapped = tuple("slots" if d == "tokens" else d for d in src.type.dims)
    if set(mapped) != set(into.type.dims):
        fail(op, f"src dims {src.type.dims} do not match into dims {into.type.dims} (tokens go to slots)")
    for d in into.type.dims:
        a, b = into.type.size(d), src.type.size(d if d != "slots" else "tokens")
        if d != "slots" and None not in (a, b) and a != b:
            fail(op, f"into has {d}={a}, src has {b}")
    perm = [mapped.index(d) for d in into.type.dims]
    axis = into.type.dims.index("slots")
    if "batch" in at.type.dims:
        at_axes = [into.type.dims.index("batch" if d == "batch" else "slots") for d in at.type.dims]

        def run(into, src, at):
            rows = src.permute(perm)
            shape = [1] * rows.dim()
            for i, a in enumerate(at_axes):
                shape[a] = at.shape[i]
            index = at.reshape(shape).expand_as(rows)
            into.scatter_(axis, index, rows)
            return (into,)
    else:
        def run(into, src, at):
            into.index_copy_(axis, at, src.permute(perm))
            return (into,)

    return node(op, {"into": into, "src": src, "at": at}, [into.type], run, written=("into",))[0]


def _valid(until):
    """The valid length (until + 1) as int32, made once per program: the hand kernel's input."""
    return _memo(("valid", until.id), lambda: node(
        "valid_length", {"until": until}, [until.type.but(kind="length", dtype="int32")],
        lambda until: ((until + 1).to(torch.int32),))[0])


def _block_mask(until, slots, share):
    """FlexAttention's block mask for keys 0..until[b], by arithmetic (week 4 E3: no search over positions), made
    once per program."""
    def make():
        from torch.nn.attention.flex_attention import BlockMask

        def run(until, slots):
            b, blk = until.shape[0], 128
            nkv = (slots + blk - 1) // blk
            valid = (until + 1).to(torch.int32)
            full = torch.div(valid, blk, rounding_mode="floor").to(torch.int32)
            part = (valid % blk != 0).to(torch.int32)
            idx = torch.arange(nkv, device=until.device, dtype=torch.int32).view(1, 1, 1, nkv).expand(b, 1, 1, nkv)
            full_idx = idx.contiguous()
            part_idx = full_idx.clone()
            part_idx[..., 0] = torch.clamp(full, max=nkv - 1).view(b, 1, 1)
            bound = until

            def mask_mod(bb, h, q_idx, kv_idx):
                return kv_idx <= bound[bb]

            return (BlockMask.from_kv_blocks(part.view(b, 1, 1), part_idx, full.view(b, 1, 1), full_idx,
                                             BLOCK_SIZE=blk, mask_mod=mask_mod, seq_lengths=(1, slots)),)

        return node("block_mask", {"until": until, "slots": int(slots)}, [until.type.but(kind="key_mask")], run)[0]

    return _memo(("block_mask", until.id, slots), make)


def attend(*, query, keys, values, until, share, at=None, props=None, scale=None):
    """Attention of a query over the keys 0..until (inclusive) of a cache, `share` query heads to a key/value head
    (by index: nothing is copied). A query of several tokens also needs `at`, its absolute positions (causal
    alignment). `props`, the model's ModelProps, must be honoured by the lowering (option "attention")."""
    op = "attend"
    query = take(op, "query", query, "query", dims=("tokens", "head_dim"))
    keys = take(op, "keys", keys, "key", dims=("slots", "head_dim"))
    values = take(op, "values", values, "value", dims=("slots", "head_dim"))
    until = take(op, "until", until, "last_key", dims=("batch",))
    other = ("batch", "tokens", "slots", "head_dim")
    qh = [d for d in query.type.dims if d not in other]
    kh = [d for d in keys.type.dims if d not in other]
    if len(qh) != 1 or len(kh) != 1:
        fail(op, f"query and keys each need one heads dim; they have {query.type.dims} and {keys.type.dims}")
    if tuple(values.type.dims) != tuple(keys.type.dims):
        fail(op, f"values' dims {values.type.dims} are not the keys' {keys.type.dims}")
    hq, hkv = query.type.size(qh[0]), keys.type.size(kh[0])
    if None not in (hq, hkv) and hq != hkv * int(share):
        fail(op, f"{hq} query heads cannot share {hkv} key/value heads {share} to one")
    if query.type.fact(Positions) != keys.type.fact(Positions):
        fail(op, f"query is {query.type.fact(Positions) or 'not rotated'}, keys are "
                 f"{keys.type.fact(Positions) or 'not rotated'}: they must be in one frame")
    for role, v in (("keys", keys), ("values", values)):
        if v.type.dims[-1] != "head_dim":
            fail(op, f"{role}' last dim must be head_dim; they have {v.type.dims}")
    tokens = query.type.size("tokens")
    if tokens != 1 and at is None:
        fail(op, "a query of several tokens (or an unknown number) needs at=, its absolute positions")
    if at is not None:
        at = _frame(op, "at", take(op, "at", at, "positions", dims=("tokens",)))
    if props is not None and not isinstance(props, ModelProps):
        fail(op, f"props takes the model's ModelProps, got {type(props).__name__}")
    needs = {n for n in ("softcap", "sliding_window") if props is not None and getattr(props, n)}
    backend = option("attention", "torch")
    if backend not in ATTENTION:
        fail(op, f"no attention lowering {backend!r} (lowered: {', '.join(ATTENTION)})")
    if needs - ATTENTION_HONOURS[backend]:
        honours = [b for b in ATTENTION if needs <= ATTENTION_HONOURS[b]]
        said(f"{op}: {backend} does not honour {', '.join(sorted(needs))}; routed to {honours[0]}")
        backend = honours[0]
    if backend in ("triton", "flex") and (tokens != 1 or at is not None):
        backend = "torch"   # the decode lowerings take one token; a prompt goes to the general one (same meaning)
    d = query.type.size("head_dim")
    if backend == "triton" and (d is None or d & (d - 1)):
        fail(op, f"the triton lowering needs a power-of-two head_dim, got {d}")
    sm = float(scale) if scale is not None else (float(d) ** -0.5 if d else None)
    if sm is None:
        fail(op, "head_dim is not known: give scale=")
    q_to = [query.type.dims.index(x) for x in ("batch", qh[0], "tokens", "head_dim")]
    q_back = [q_to.index(i) for i in range(4)]
    k_to = [keys.type.dims.index(x) for x in ("batch", kh[0], "slots", "head_dim")]
    group = int(share)
    softcap = props.softcap if props is not None else None
    window = props.sliding_window if props is not None else None
    out = T(query.type.dims, query.type.dtype, "attended", _sizes(query.type))

    if backend == "triton":
        from . import kernels  # noqa: F401 - registers entail::decode_attention

        def run(query, keys, values, valid):
            q = query.permute(q_to).contiguous()
            o = torch.ops.entail.decode_attention(q, keys.permute(k_to), values.permute(k_to), valid, sm)
            return (o.permute(q_back),)

        inputs = {"query": query, "keys": keys, "values": values, "valid": _valid(until)}
    elif backend == "flex":
        from torch.nn.attention.flex_attention import flex_attention

        def run(query, keys, values, mask):
            o = flex_attention(query.permute(q_to), keys.permute(k_to), values.permute(k_to), block_mask=mask,
                               scale=sm, enable_gqa=group > 1)
            return (o.permute(q_back),)

        slots = keys.type.size("slots")
        if slots is None:
            fail(op, "the flex lowering needs the cache's slots to be a known size")
        inputs = {"query": query, "keys": keys, "values": values, "mask": _block_mask(until, slots, group)}
    else:
        def run(query, keys, values, until, at=None):
            q, k, v = query.permute(q_to), keys.permute(k_to), values.permute(k_to)
            n = k.shape[2]
            kv = torch.arange(n, device=q.device)
            qpos = (until.view(-1, 1) if at is None else (at if at.dim() == 2 else at.view(1, -1)))  # [B|1, L]
            allowed = (kv.view(1, 1, n) <= qpos.unsqueeze(-1)) & (kv.view(1, 1, n) <= until.view(-1, 1, 1))
            if window:
                allowed = allowed & (kv.view(1, 1, n) > qpos.unsqueeze(-1) - window)
            allowed = allowed.unsqueeze(1)   # [B, 1, L, n]
            if softcap:
                kk, vv = k.float().repeat_interleave(group, 1), v.float().repeat_interleave(group, 1)
                s = torch.tanh((q.float() @ kk.transpose(-1, -2)) * sm / softcap) * softcap
                s = s.masked_fill(~allowed, float("-inf"))
                o = (torch.softmax(s, dim=-1) @ vv).to(q.dtype)
            else:
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, scale=sm, enable_gqa=group > 1)
            return (o.permute(q_back),)

        inputs = {"query": query, "keys": keys, "values": values, "until": until}
        if at is not None:
            inputs["at"] = at
    return node(op, inputs, [out], run, note=f"lowered: {backend}")[0]


def add(*, a, b):
    """a + b: the same kind of value, by dim name; a partial sum and a replicated value are not added, nor positions
    in two frames."""
    op = "add"
    a, b = take(op, "a", a), take(op, "b", b)
    if a.type.kind != b.type.kind:
        fail(op, f"adds a {a.type.kind} and a {b.type.kind}")
    if set(a.type.dims) != set(b.type.dims):
        fail(op, f"a {a.type.dims} and b {b.type.dims} name different dims")
    ra, rb = a.type.fact(Reduction), b.type.fact(Reduction)
    if (ra is None) != (rb is None) or (ra is not None and ra.state != rb.state):
        fail(op, f"adds {ra or 'an undeclared sum'} and {rb or 'an undeclared sum'}; reduce first")
    pa, pb = a.type.fact(Positions), b.type.fact(Positions)
    if pa is not None and pb is not None and pa != pb:
        fail(op, f"adds positions in two frames: {pa} and {pb}")
    perm = [b.type.dims.index(d) for d in a.type.dims]
    same = perm == list(range(len(perm)))
    return node(op, {"a": a, "b": b}, [a.type],
                (lambda a, b: (a + b,)) if same else (lambda a, b: (a + b.permute(perm),)))[0]


def swiglu(*, gate, up):
    """silu(gate) * up: the gate projection's output is the one the activation takes."""
    op = "swiglu"
    gate = take(op, "gate", gate, "gate")
    up = take(op, "up", up, "up")
    if gate.type.dims != up.type.dims:
        fail(op, f"gate {gate.type.dims} and up {up.type.dims} differ")
    return node(op, {"gate": gate, "up": up}, [gate.type.but(kind="hidden")],
                lambda gate, up: (F.silu(gate) * up,))[0]


def last(*, x, dim="tokens"):
    op = "last"
    x = take(op, "x", x, None, dims=(dim,))
    axis = x.type.dims.index(dim)
    out = x.type.but(dims=x.type.dims[:axis] + x.type.dims[axis + 1:],
                     sizes=_sizes(x.type)[:axis] + _sizes(x.type)[axis + 1:])
    return node(op, {"x": x}, [out], lambda x: (x.select(axis, -1),))[0]


def argmax(*, logits, over="vocab"):
    """The greedy choice: the index of the largest logit, a token id."""
    op = "argmax"
    logits = take(op, "logits", logits, "logits", dims=(over,))
    axis = logits.type.dims.index(over)
    out = T(logits.type.dims[:axis] + logits.type.dims[axis + 1:], "int64", "token_ids",
            _sizes(logits.type)[:axis] + _sizes(logits.type)[axis + 1:])
    return node(op, {"logits": logits}, [out], lambda logits: (logits.argmax(axis),))[0]


def all_reduce(*, x):
    """A partial sum made whole across the process group (Reduction P -> R). Reducing a replicated value again
    multiplies it by the group's size (rolebench 04), so it is refused."""
    op = "all_reduce"
    x = take(op, "x", x)
    red = x.type.fact(Reduction)
    if red is None or red.state != "P":
        fail(op, f"x is {red or 'not a declared partial sum'}: reducing it would count it again")

    def run(x):
        import torch.distributed as dist

        y = x.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(y)
        return (y,)

    return node(op, {"x": x}, [x.type.but(facts=(Reduction("R"),))], run)[0]


def to_absolute(*, positions):
    op = "to_absolute"
    positions = take(op, "positions", positions, "positions")
    frame = positions.type.fact(Positions)
    if frame is None or frame.frame != "chunk_relative" or frame.offset is None:
        fail(op, f"positions are {frame or 'in no declared frame'}, not chunk-relative with an offset")
    return node(op, {"positions": positions, "offset": int(frame.offset)},
                [positions.type.but(facts=(Positions("absolute"),))],
                lambda positions, offset: (positions + offset,))[0]


def advance(*, counter, by=1):
    """A counter moved on in place (a position counter a cache update increments); its old version may not be read
    after it - a reader that holds the counter itself instead of its value is refused (rolebench 10)."""
    op = "advance"
    counter = take(op, "counter", counter, ("positions", "last_key", "length"))
    return node(op, {"counter": counter, "by": int(by)}, [counter.type], lambda counter, by: (counter.add_(by),),
                written=("counter",))[0]


def copy(*, x):
    """The value as it is now, as a value of its own: what is handed to a reader that reads later."""
    op = "copy"
    x = take(op, "x", x)
    return node(op, {"x": x}, [x.type], lambda x: (x.clone(),))[0]


def last_key(*, length):
    """The last key index (inclusive) of a length (exclusive): until = length - 1, said by the program."""
    op = "last_key"
    length = take(op, "length", length, "length")
    return node(op, {"length": length}, [length.type.but(kind="last_key")], lambda length: (length - 1,))[0]


def reorder(*, weight, to):
    """Rewrites a weight in place into another format (a conversion target: to=); the weight's old version may not
    be read after it - a reader that still expects the old format sees the new one or is refused (rolebench 01)."""
    op = "reorder"
    weight = take(op, "weight", weight, "weight")
    before = weight.type.fact(Layout)
    if not isinstance(to, Layout):
        fail(op, f"to takes a Layout, got {type(to).__name__}")
    fn = REORDERS.get((before, to))
    if fn is None:
        fail(op, f"no conversion from {before} to {to} is registered")
    return node(op, {"weight": weight}, [weight.type.but(facts=(to,))], lambda weight: (fn(weight),),
                written=("weight",))[0]
