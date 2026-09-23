"""Tests for the KV cache contract, on a tiny model so they run on the CPU.

Run: python tests/test_cache_contract.py
"""
import os
import sys

import torch
from transformers import LlamaConfig, LlamaForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import core  # noqa: E402
from entail.adapters import cache_contract  # noqa: E402

CFG = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
           num_key_value_heads=2, max_position_embeddings=64)


def model_and_ids():
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(**CFG)).eval()
    return model, torch.tensor([[1, 2, 3, 4, 5]])


def decode(model, ids, steps=3, drop_after=None):
    with torch.no_grad():
        res = model(ids, use_cache=True)
    cache = res.past_key_values
    step = res.logits[:, -1:].argmax(-1)
    for i in range(steps):
        if drop_after is not None and i == drop_after:
            for layer in cache.layers:
                layer.keys = layer.keys[..., :-1, :].contiguous()
                layer.values = layer.values[..., :-1, :].contiguous()
        with torch.no_grad():
            res = model(step, past_key_values=cache, use_cache=True)
        step = res.logits[:, -1:].argmax(-1)
    return cache


def test_install_wraps_the_concrete_layers():
    n = cache_contract.install()
    try:
        assert n >= 1, "no cache layer class was wrapped; the mixin's update is abstract"
    finally:
        assert cache_contract.uninstall() == n


def test_healthy_decode_is_quiet_and_checked():
    cache_contract.install()
    core.set_mode("debug")
    cache_contract.reset()
    try:
        model, ids = model_and_ids()
        decode(model, ids)
        stats = cache_contract.stats()
        assert stats["updates"] > 0 and stats["complaints"] == 0, stats
        assert stats["checked"] == stats["updates"], stats
    finally:
        core.set_mode("off")
        cache_contract.uninstall()


def test_a_cache_that_lost_a_token_is_caught():
    cache_contract.install()
    core.set_mode("debug")
    cache_contract.reset()
    try:
        model, ids = model_and_ids()
        try:
            decode(model, ids, steps=4, drop_after=1)
        except core.RoleError as e:
            assert "dropped 1 token" in str(e), e
        else:
            raise AssertionError("a cache one token short was accepted")
    finally:
        core.set_mode("off")
        cache_contract.uninstall()


def test_layers_that_disagree_are_caught_even_without_history():
    """A cache handed over already uneven: one layer is short and there is no history to compare against.

    Only the agreement rule can see this one, which is why it exists next to the growth rule.
    """
    cache_contract.install()
    core.set_mode("debug")
    try:
        model, ids = model_and_ids()
        with torch.no_grad():
            res = model(ids, use_cache=True)
        cache = res.past_key_values
        cache.layers[1].keys = cache.layers[1].keys[..., :-1, :].contiguous()
        cache.layers[1].values = cache.layers[1].values[..., :-1, :].contiguous()
        cache_contract.reset()  # as if this process had just been handed the cache
        step = res.logits[:, -1:].argmax(-1)
        try:
            with torch.no_grad():
                model(step, past_key_values=cache, use_cache=True)
        except core.RoleError as e:
            assert "disagree" in str(e), e
        except RuntimeError:
            pass  # in the eager path the mask is built from layer 0, so the shapes can crash first (loud)
        else:
            raise AssertionError("an uneven cache was accepted")
    finally:
        core.set_mode("off")
        cache_contract.uninstall()


def test_check_cache_from_outside():
    """The same contract checked once per request, which is where it has to live on a compiled path."""
    model, ids = model_and_ids()
    with torch.no_grad():
        res = model(ids, use_cache=True)
    cache = res.past_key_values
    assert cache_contract.check_cache(cache, ids.shape[1]) == len(cache.layers)
    try:
        cache_contract.check_cache(cache, ids.shape[1] + 1)  # a request that wrote one more than the cache holds
    except core.RoleError as e:
        assert "do not hold the number of tokens" in str(e), e
    else:
        raise AssertionError("a cache one token short was accepted")


def test_off_mode_does_not_check():
    cache_contract.install()
    core.set_mode("off")
    cache_contract.reset()
    try:
        model, ids = model_and_ids()
        decode(model, ids, steps=4, drop_after=1)  # the same defect, no error
        assert cache_contract.stats()["checked"] == 0
    finally:
        cache_contract.uninstall()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
