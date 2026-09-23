"""Tests for the load-time source check, with a tiny safetensors file instead of an engine.

Run: python tests/test_vllm_source.py
"""
import os
import sys
import tempfile
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail.adapters.vllm_source import Checkpoint, read_layer  # noqa: E402

CFG = SimpleNamespace(hidden_size=8, num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                      intermediate_size=6)
PREFIX = "model.layers.0."


def make_checkpoint(**tensors):
    d = tempfile.mkdtemp(prefix="source_test_")
    save_file(tensors, os.path.join(d, "model.safetensors"))
    return Checkpoint(d)


def qkv_parts():
    torch.manual_seed(0)
    return (torch.randn(8, 8), torch.randn(4, 8), torch.randn(4, 8))


def test_fused_qkv_matches():
    q, k, v = qkv_parts()
    ckpt = make_checkpoint(**{PREFIX + "self_attn.q_proj.weight": q, PREFIX + "self_attn.k_proj.weight": k,
                              PREFIX + "self_attn.v_proj.weight": v})
    fused = torch.cat([q, k, v], dim=0)
    out, why = read_layer(PREFIX + "self_attn.qkv_proj", fused, CFG, ckpt)
    assert out == "" and why is None, (out, why)


def test_fused_qkv_with_a_rolled_source():
    """The defect planted in the engine run: one source tensor shifted by a row before it was fused."""
    q, k, v = qkv_parts()
    ckpt = make_checkpoint(**{PREFIX + "self_attn.q_proj.weight": q, PREFIX + "self_attn.k_proj.weight": k,
                              PREFIX + "self_attn.v_proj.weight": v})
    fused = torch.cat([torch.roll(q, 1, 0), k, v], dim=0)
    out, _ = read_layer(PREFIX + "self_attn.qkv_proj", fused, CFG, ckpt)
    assert "q_proj.weight" in out, out


def test_swapped_k_and_v_is_caught():
    q, k, v = qkv_parts()
    ckpt = make_checkpoint(**{PREFIX + "self_attn.q_proj.weight": q, PREFIX + "self_attn.k_proj.weight": k,
                              PREFIX + "self_attn.v_proj.weight": v})
    fused = torch.cat([q, v, k], dim=0)  # the fusion order the loader declares is q, k, v
    out, _ = read_layer(PREFIX + "self_attn.qkv_proj", fused, CFG, ckpt)
    assert "k_proj" in out or "v_proj" in out, out


def test_gate_up_fusion():
    torch.manual_seed(1)
    gate, up = torch.randn(6, 8), torch.randn(6, 8)
    ckpt = make_checkpoint(**{PREFIX + "mlp.gate_proj.weight": gate, PREFIX + "mlp.up_proj.weight": up})
    ok, why = read_layer(PREFIX + "mlp.gate_up_proj", torch.cat([gate, up], 0), CFG, ckpt)
    assert ok == "" and why is None, (ok, why)
    bad, _ = read_layer(PREFIX + "mlp.gate_up_proj", torch.cat([up, gate], 0), CFG, ckpt)
    assert bad, bad


def test_row_count_mismatch_is_reported():
    q, k, v = qkv_parts()
    ckpt = make_checkpoint(**{PREFIX + "self_attn.q_proj.weight": q, PREFIX + "self_attn.k_proj.weight": k,
                              PREFIX + "self_attn.v_proj.weight": v})
    out, _ = read_layer(PREFIX + "self_attn.qkv_proj", torch.cat([q, k], 0), CFG, ckpt)
    assert "add up to" in out, out


def test_unknown_layer_is_skipped_not_passed():
    ckpt = make_checkpoint(**{PREFIX + "self_attn.q_proj.weight": torch.zeros(8, 8)})
    out, why = read_layer(PREFIX + "self_attn.some_new_proj", torch.zeros(8, 8), CFG, ckpt)
    assert out is None and why == "no declared source mapping", (out, why)


def test_the_core_decides():
    """What the adapter read goes to load.weights_taken: a weight that did not land is refused; nothing compared,
    nothing decided."""
    from entail import load
    from entail.contracts import Verdict
    from entail.policies import Policy

    d = load.weights_taken("vllm", "/m", 3, ["model.layers.0.self_attn.qkv_proj row 0 ..."], Policy(mode="load"))
    assert len(d) == 1 and d[0].verdict is Verdict.REFUSED and d[0].blocking
    assert load.weights_taken("vllm", "/m", 0, [], Policy(mode="load")) == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
