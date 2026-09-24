"""Tests for the narrow proof path (ROADMAP M8.2): Qwen3's decode step written with the front end computes what
transformers' Qwen3 computes - on a tiny random model, on the CPU, with the torch lowering - and the load contract
refuses weights that are not stored as the program declares. (testbed/m82_qwen3.py does the same on Qwen3-4B int4.)
Run: python tests/test_frontend_qwen3.py"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail.core import RoleError  # noqa: E402
from entail.facts import Rotary  # noqa: E402
from entail.frontend import qwen3  # noqa: E402

B, L, SLOTS = 2, 6, 10


def tiny():
    from transformers import Qwen3Config, Qwen3ForCausalLM, StaticCache

    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=96, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=64,
                         tie_word_embeddings=False)
    model = Qwen3ForCausalLM(config).eval()
    cache = StaticCache(config=config, max_cache_len=SLOTS)
    ids = torch.randint(0, 96, (B, L))
    with torch.no_grad():
        logits = model(ids, cache_position=torch.arange(L), past_key_values=cache, use_cache=True).logits
    return model, cache, logits[:, -1].argmax(-1, keepdim=True)


def rotary(model):
    params = model.config.rope_parameters
    return Rotary(params.get("rope_type", "default"), float(params["rope_theta"]))


def test_the_program_computes_what_transformers_computes():
    try:
        model, cache, tok = tiny()
    except ImportError:
        print("skip (transformers is not installed)")
        return
    saved = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
    with torch.no_grad():
        want = model(tok, cache_position=torch.tensor([L]), past_key_values=cache, use_cache=True).logits[:, -1]
    written = [(layer.keys[:, :, L].clone(), layer.values[:, :, L].clone()) for layer in cache.layers]
    for layer, (k, v) in zip(cache.layers, saved):
        layer.keys.copy_(k)
        layer.values.copy_(v)
    program = qwen3.trace_decode(model.config, B, SLOTS, attention="torch", quantized=False, rotary=rotary(model),
                                  dtype="float32")
    with torch.no_grad():
        got = qwen3.bind(program, model, cache, tok, torch.tensor([L]), torch.full((B,), L))()
    assert torch.allclose(got["logits"], want, atol=1e-5), (got["logits"] - want).abs().max()
    assert torch.equal(got["next"], want.argmax(-1))
    for layer, (k, v) in zip(cache.layers, written):
        assert torch.allclose(layer.keys[:, :, L], k, atol=1e-6) and torch.allclose(layer.values[:, :, L], v, atol=1e-6)


def test_the_load_contract_refuses_what_is_not_as_declared():
    try:
        model, cache, tok = tiny()
    except ImportError:
        print("skip (transformers is not installed)")
        return
    program = qwen3.trace_decode(model.config, B, SLOTS, attention="torch", quantized=True, rotary=rotary(model),
                                  dtype="float32")
    values = qwen3.tensors(model, cache, tok, torch.tensor([L]), torch.full((B,), L))
    try:
        qwen3.check_layouts(program, values)
    except RoleError as e:
        assert "layers[0].q: declared int4_packed, stored dense" in str(e), e
    else:
        raise AssertionError("dense weights were taken for int4 ones")
    try:
        qwen3.trace_decode(model.config, B, SLOTS, attention="torch", rotary=Rotary("yarn", 1e6, 4.0, 32))
    except RoleError as e:
        assert "no lowering computes Rotary(rope_type='yarn'" in str(e), e
    else:
        raise AssertionError("a RoPE scaling no lowering computes was traced")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
