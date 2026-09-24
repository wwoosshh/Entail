"""frontend.qwen3: the narrow proof path - Qwen3's decode step written with the front end (ROADMAP M8.2).

One step of Qwen3 decoding - embedding, every layer's attention (projections, per-head RMSNorm, RoPE, the KV cache
write, attention over the valid keys) and MLP, the final norm, the head and the greedy choice - as a front-end
program. Its types come from the model's own declarations, read once where they are made (principle 2):
  config.json    hidden size, heads, key/value heads, head dim, layers, vocabulary, RMSNorm epsilon (config facts)
  entail sources the Rotary declaration (entail.load.declared: rope type and base)
  quantization   the Layout of every linear weight: what the quantization applied declares (int4 packed in groups of
                 128 for torchao's Int4WeightOnlyConfig), or dense
and `bind` - the load contract - checks the tensors of a loaded model against them before the first step: every
weight's format and shape against what the program's type declares, the cache against its slots. A mismatch is
refused before anything runs.

    program = qwen3.trace_decode(config, batch=8, slots=640, attention="triton", quantized=True)
    step = qwen3.bind(program, model, cache, tokens, positions, until)
    out = step()              # {"logits": [batch, vocab], "next": [batch]}
"""
from ..core import RoleError
from ..facts import Layout, Positions
from . import ops as fe
from .graph import T

INT4 = Layout("int4_packed", block=(128,))
LINEARS = ("q", "k", "v", "o", "gate", "up", "down")


def decode(*, tokens, positions, until, weights, cache, rotary, eps, heads, kv_heads):
    """One decode step (or a prompt, given at= positions for every token) of Qwen3."""
    h = fe.embed(tokens=tokens, table=weights["embed"])
    for w, c in zip(weights["layers"], cache):
        x = fe.rms_norm(x=h, weight=w["input_norm"], eps=eps)
        q = fe.split_heads(x=fe.linear(x=x, weight=w["q"]), heads=heads, name="heads")
        k = fe.split_heads(x=fe.linear(x=x, weight=w["k"]), heads=kv_heads, name="kv_heads")
        v = fe.split_heads(x=fe.linear(x=x, weight=w["v"]), heads=kv_heads, name="kv_heads")
        q = fe.rope(x=fe.rms_norm(x=q, weight=w["q_norm"], eps=eps), positions=positions, rotary=rotary)
        k = fe.rope(x=fe.rms_norm(x=k, weight=w["k_norm"], eps=eps), positions=positions, rotary=rotary)
        keys = fe.write(into=c["keys"], src=k, at=positions)
        values = fe.write(into=c["values"], src=v, at=positions)
        a = fe.attend(query=q, keys=keys, values=values, until=until, share=heads // kv_heads)
        h = fe.add(a=h, b=fe.linear(x=fe.merge_heads(x=a, name="attn"), weight=w["o"]))
        x = fe.rms_norm(x=h, weight=w["post_norm"], eps=eps)
        mlp = fe.swiglu(gate=fe.linear(x=x, weight=w["gate"]), up=fe.linear(x=x, weight=w["up"]))
        h = fe.add(a=h, b=fe.linear(x=mlp, weight=w["down"]))
    h = fe.rms_norm(x=h, weight=weights["final_norm"], eps=eps)
    logits = fe.linear(x=fe.last(x=h), weight=weights["lm_head"])
    return {"logits": logits, "next": fe.argmax(logits=logits)}


def _get(config, name):
    v = getattr(config, name, None)
    if v is None and hasattr(config, "get_text_config"):
        v = getattr(config.get_text_config(), name, None)
    return v


def types(config, batch, slots, quantized, rotary, dtype="bfloat16"):
    """The input types of one decode step, from the model's declarations (see the module docstring)."""
    e, v, n = _get(config, "hidden_size"), _get(config, "vocab_size"), _get(config, "num_hidden_layers")
    hq, hkv = _get(config, "num_attention_heads"), _get(config, "num_key_value_heads")
    d, ffn = _get(config, "head_dim") or e // hq, _get(config, "intermediate_size")
    fmt = (INT4,) if quantized else (Layout("dense"),)

    def w(dims, sizes, yields="", layout=True):
        return T(dims, dtype, "weight", sizes, fmt if layout else (), yields)

    layer = {"input_norm": w(("embed",), (e,), layout=False), "post_norm": w(("embed",), (e,), layout=False),
             "q_norm": w(("head_dim",), (d,), layout=False), "k_norm": w(("head_dim",), (d,), layout=False),
             "q": w(("q_features", "embed"), (hq * d, e), "query"),
             "k": w(("kv_features", "embed"), (hkv * d, e), "key"),
             "v": w(("kv_features", "embed"), (hkv * d, e), "value"),
             "o": w(("embed", "attn"), (e, hq * d)),
             "gate": w(("ffn", "embed"), (ffn, e), "gate"), "up": w(("ffn", "embed"), (ffn, e), "up"),
             "down": w(("embed", "ffn"), (e, ffn))}
    cache = {"keys": T(("batch", "kv_heads", "slots", "head_dim"), dtype, "key", (batch, hkv, slots, d),
                       (Positions("absolute"),)),
             "values": T(("batch", "kv_heads", "slots", "head_dim"), dtype, "value", (batch, hkv, slots, d))}
    return {"tokens": T(("batch", "tokens"), "int64", "token_ids", (batch, 1)),
            "positions": T(("tokens",), "int64", "positions", (1,), (Positions("absolute"),)),
            "until": T(("batch",), "int64", "last_key", (batch,)),
            "weights": {"embed": T(("vocab", "embed"), dtype, "weight", (v, e)), "layers": [dict(layer) for _ in range(n)],
                        "final_norm": w(("embed",), (e,), layout=False),
                        "lm_head": w(("vocab", "embed"), (v, e), "logits")},
            "cache": [dict(cache) for _ in range(n)],
            "rotary": rotary, "eps": float(_get(config, "rms_norm_eps")), "heads": hq, "kv_heads": hkv}


def rotary_of(model_path, config=None):
    """The Rotary the model declares, read by entail's sources (the same reading the load contracts use)."""
    from .. import load

    facts = load.declared(model_path, config)
    got = facts.get("Rotary")
    if not got:
        raise RoleError(f"{model_path} declares no Rotary")
    return got[0].value


def trace_decode(config, batch, slots, attention="triton", quantized=True, rotary=None, model_path=None,
                 dtype="bfloat16"):
    """The traced decode step for this model's declarations; `dtype` is what its values are computed in."""
    rotary = rotary or rotary_of(model_path or config._name_or_path, config)
    from .graph import trace

    return trace(decode, {"attention": attention}, **types(config, batch, slots, quantized, rotary, dtype))


def layout_of(tensor):
    """What a weight tensor is stored as, from the tensor itself (the data side of the load contract)."""
    name = type(tensor).__name__
    if "Int4" in name:
        return "int4_packed"
    if name in ("Tensor", "Parameter"):
        return "dense"
    return name


def tensors(model, cache, tokens, positions, until):
    """The tensors a decode step reads, from a transformers Qwen3 model and its StaticCache, checked against the
    declared formats (the load contract: a weight whose storage is not what the program declares is refused)."""
    m = model.model
    layers = []
    for layer in m.layers:
        a, f = layer.self_attn, layer.mlp
        layers.append({"input_norm": layer.input_layernorm.weight, "post_norm": layer.post_attention_layernorm.weight,
                       "q_norm": a.q_norm.weight, "k_norm": a.k_norm.weight, "q": a.q_proj.weight,
                       "k": a.k_proj.weight, "v": a.v_proj.weight, "o": a.o_proj.weight,
                       "gate": f.gate_proj.weight, "up": f.up_proj.weight, "down": f.down_proj.weight})
    weights = {"embed": m.embed_tokens.weight, "layers": layers, "final_norm": m.norm.weight,
               "lm_head": model.lm_head.weight}
    caches = [{"keys": layer.keys, "values": layer.values} for layer in cache.layers]
    return {"tokens": tokens, "positions": positions, "until": until, "weights": weights, "cache": caches}


def bind(program, model, cache, tokens, positions, until):
    """The load contract, then the step: the weights' formats against the declaration (check_layouts), the dtypes and
    fixed sizes of everything the step reads (Program.bind). Returns the step, a function of nothing."""
    values = tensors(model, cache, tokens, positions, until)
    check_layouts(program, values)
    return program.bind(**values)


def check_layouts(program, values):
    """The load contract's format half: every weight the program declares a Layout for is stored that way."""
    wanted = program.graph.inputs["weights"]
    bad = []
    for i, (want, have) in enumerate(zip(wanted["layers"], values["weights"]["layers"])):
        for name in LINEARS:
            declared = want[name].type.fact(Layout)
            if declared is not None and declared.kind != layout_of(have[name]):
                bad.append(f"layers[{i}].{name}: declared {declared.kind}, stored {layout_of(have[name])}")
    head = wanted["lm_head"].type.fact(Layout)
    if head is not None and head.kind != layout_of(values["weights"]["lm_head"]):
        bad.append(f"lm_head: declared {head.kind}, stored {layout_of(values['weights']['lm_head'])}")
    if bad:
        raise RoleError("load contract: weights are not stored as the program declares: " + "; ".join(bad[:6])
                           + (f" (and {len(bad) - 6} more)" if len(bad) > 6 else ""))
