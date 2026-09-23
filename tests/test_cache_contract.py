"""Tests for the transformers adapter of the KV container contract (ROADMAP M5.1), on a tiny model so they run on the
CPU. The rules are the core's (tests/test_kv_contract.py); these check that the adapter reads the cache right at
Cache.update, and the defects the contract exists for.

Run: python tests/test_cache_contract.py
"""
import io
import os
import sys
from contextlib import redirect_stdout

import torch
from transformers import LlamaConfig, LlamaForCausalLM, StaticCache

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import core, kv_contract  # noqa: E402
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


class Contract:
    """The adapter installed, in a mode, with fresh books; everything undone afterwards."""

    def __init__(self, mode="debug"):
        self.mode = mode

    def __enter__(self):
        cache_contract.install()
        cache_contract.reset()
        core.set_mode(self.mode)
        return self

    def __exit__(self, *exc):
        core.set_mode("off")
        cache_contract.uninstall()


def stops(fn, text):
    try:
        with redirect_stdout(io.StringIO()):
            fn()
    except core.RoleError as e:
        assert text in str(e), (text, str(e))
        return True
    return False


def test_install_wraps_the_container_boundary():
    from transformers import masking_utils
    from transformers.cache_utils import Cache

    orig = (Cache.update, Cache.get_query_offset, masking_utils.add_offsets_to_mask_function)
    assert cache_contract.install() == 1 and cache_contract.install() == 0
    assert (Cache.update, Cache.get_query_offset, masking_utils.add_offsets_to_mask_function) != orig
    assert cache_contract.uninstall() == 1
    assert (Cache.update, Cache.get_query_offset, masking_utils.add_offsets_to_mask_function) == orig


def test_healthy_decode_is_quiet_and_checked():
    with Contract():
        model, ids = model_and_ids()
        decode(model, ids)
        s = cache_contract.stats()
        assert s["checks"] == 8 and s["refused"] == 0, s   # 2 layers x (prefill + 3 steps)
        assert s["passed"]["kv_needed"] == 8 and s["passed"]["kv_shrank"] == 6 and s["passed"]["kv_layers"] == 8, s


def test_a_cache_that_lost_a_token_is_caught():
    with Contract():
        model, ids = model_and_ids()
        assert stops(lambda: decode(model, ids, steps=4, drop_after=1), "dropped 1 token"), \
            "a cache one token short was accepted"


def test_two_generations_in_one_process_are_quiet():
    """The regression that moving the books per cache fixed: before M5.1 the second generate was refused, because
    the layers of the first cache were still compared with the new one."""
    with Contract(mode="load"):
        model, _ = model_and_ids()
        for prompt in ([1, 2, 3, 4, 5], [7, 8, 9]):
            with torch.no_grad():
                out = model.generate(torch.tensor([prompt]), max_new_tokens=3, do_sample=False)
            assert out.shape[1] == len(prompt) + 3
        assert cache_contract.stats()["refused"] == 0


def test_layers_that_disagree_are_caught_even_without_history():
    """A cache handed over already uneven: one layer is short and there are no books to compare against. Only the
    agreement rule can see this one."""
    with Contract():
        model, ids = model_and_ids()
        with torch.no_grad():
            res = model(ids, use_cache=True)
        cache = res.past_key_values
        cache.layers[1].keys = cache.layers[1].keys[..., :-1, :].contiguous()
        cache.layers[1].values = cache.layers[1].values[..., :-1, :].contiguous()
        cache_contract.reset()   # as if this process had just been handed the cache
        step = res.logits[:, -1:].argmax(-1)
        try:
            caught = stops(lambda: model(step, past_key_values=cache, use_cache=True), "disagree")
        except RuntimeError:
            caught = True   # in the eager path the mask is built from layer 0, so the shapes can crash first (loud)
        assert caught, "an uneven cache was accepted"


def test_check_cache_from_outside():
    """The same contract once per request, which is where it has to live on a compiled path."""
    model, ids = model_and_ids()
    with torch.no_grad():
        res = model(ids, use_cache=True)
    cache = res.past_key_values
    core.set_mode("load")
    try:
        assert cache_contract.check_cache(cache, ids.shape[1]) == len(cache.layers)
        assert stops(lambda: cache_contract.check_cache(cache, ids.shape[1] + 1), "do not hold the number of tokens")
    finally:
        core.set_mode("off")


def test_a_static_cache_is_compared_on_the_device_and_read_once():
    """A static layer keeps its length in a device tensor and increments it in place: compared without a
    synchronisation, read by flush()."""
    with Contract(mode="load"):
        model, ids = model_and_ids()
        cache = StaticCache(config=model.config, max_cache_len=16)
        with torch.no_grad():
            res = model(ids, past_key_values=cache, use_cache=True)
            model(res.logits[:, -1:].argmax(-1), past_key_values=cache, use_cache=True)
        s = cache_contract.stats()
        assert s["deferred"] == 4 and s["checks"] == 0, s
        assert cache_contract.flush() == 2 and cache_contract.flush() == 0
        assert cache_contract.stats()["refused"] == 0
        # a counter that jumped: the layer says it holds one more token than it was given
        with torch.no_grad():
            kv_contract.grew(cache_contract.BOUNDARY, cache_contract.CONSUMER, cache, "static", 0,
                             torch.tensor(6), torch.tensor(8), 1)
        assert stops(cache_contract.flush, "the length they report is not the length they were given")


def test_a_live_counter_handed_to_a_mask_is_bound_or_its_stale_read_refused():
    """rolebench 10 on the adapter's hooks, without a flex kernel: a static cache hands out its counter tensor, the
    flex mask builder closes over it, the next update writes it in place, attention reads the mask."""
    from types import SimpleNamespace

    from transformers import masking_utils

    from entail import epochs

    with Contract(mode="load"):
        model, ids = model_and_ids()
        cache = StaticCache(config=model.config, max_cache_len=16)
        with torch.no_grad():
            model(ids, past_key_values=cache, use_cache=True)               # sdpa: counters exist
        offset = cache.get_query_offset(0)
        assert epochs.reads(offset), "a static cache hands out its live counter"
        with redirect_stdout(io.StringIO()):
            fn = masking_utils.add_offsets_to_mask_function(lambda b, h, q, kv: kv <= q, offset, 0)
        assert epochs.reads(fn) == () and epochs.stats(cache_contract.MASK_BUILDER)["resolved"] == 1
        core.set_policy("refuse")
        try:
            stale = masking_utils.add_offsets_to_mask_function(lambda b, h, q, kv: kv <= q,
                                                               cache.get_query_offset(0), 0)
            assert epochs.reads(stale)
            with torch.no_grad():
                model(ids[:, :1], past_key_values=cache, use_cache=True)    # the update writes the counter
            flex, cache_contract._ORIG["flex"] = cache_contract._ORIG["flex"], lambda *a, **kw: "attended"
            try:
                from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

                attend = ALL_ATTENTION_FUNCTIONS["flex_attention"]
                assert attend(SimpleNamespace(layer_idx=0), None, None, None, SimpleNamespace(mask_mod=fn))                     == "attended"                                            # the bound mask reads nothing later
                assert stops(lambda: attend(SimpleNamespace(layer_idx=0), None, None, None,
                                            SimpleNamespace(mask_mod=stale)), "write(s) in between")
            finally:
                cache_contract._ORIG["flex"] = flex
        finally:
            core.set_policy("resolve")


def test_off_mode_does_not_check():
    with Contract(mode="off"):
        model, ids = model_and_ids()
        decode(model, ids, steps=4, drop_after=1)   # the same defect, no error
        assert cache_contract.stats()["checks"] == 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
