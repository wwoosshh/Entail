"""Tests for the vLLM adapter of the KV container contract (ROADMAP M5.1) without vLLM: what it reads from one
allocate_slots call, on a stand-in manager and request. The rules are the core's (tests/test_kv_contract.py).

The regression: vLLM passes the tokens a request just hit in the prefix cache (num_new_computed_tokens) and the ones
a KV connector holds (num_external_computed_tokens) apart from request.num_computed_tokens, which is still 0 when a
new request is allocated. Before M5.3 the adapter left them out, so a request whose prompt was a cache hit held
"too many" slots and was refused - on a server, a repeated prompt stopped the engine (testbed/results/m53).

Run: python tests/test_vllm_cache_contract.py
"""
import io
import os
import sys
from contextlib import redirect_stdout
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import core, kv_contract  # noqa: E402
from entail.adapters import vllm_cache_contract as vc  # noqa: E402

BLOCK = 16


def manager(blocks):
    """A KVCacheManager as far as the adapter reads it: one full-attention group holding `blocks` blocks."""
    single = SimpleNamespace(block_size=BLOCK, kv_cache_spec=SimpleNamespace())
    return SimpleNamespace(coordinator=SimpleNamespace(single_type_managers=[single]),
                           get_block_ids=lambda request_id: [list(range(blocks))])


def decide(blocks, computed, new, **kw):
    request = SimpleNamespace(request_id="r", num_computed_tokens=computed)
    try:
        with redirect_stdout(io.StringIO()):
            vc._decide(manager(blocks), request, new, kw)
    except core.RoleError as e:
        return str(e)
    return None


def setup():
    core.set_mode("load")
    kv_contract.reset()


def test_a_prompt_that_hit_the_prefix_cache_is_counted():
    """17 prompt tokens, 16 of them a cache hit: 1 new token, 2 blocks held."""
    setup()
    assert decide(2, 0, 1, num_new_computed_tokens=16) is None
    assert kv_contract.stats(vc.BOUNDARY)["refused"] == 0


def test_connector_and_lookahead_tokens_are_counted():
    setup()
    assert decide(3, 0, 8, num_external_computed_tokens=32, num_lookahead_tokens=4) is None   # 44 tokens, 3 blocks


def test_a_block_table_that_is_short_is_still_refused():
    setup()
    assert "holds 16 KV slots for 17 tokens" in (decide(1, 0, 1, num_new_computed_tokens=16) or "")


def test_the_old_reading_would_have_refused_the_cache_hit():
    """What the M5.1 adapter saw: the cache-hit tokens left out, 2 blocks for 1 token."""
    setup()
    assert "holds 32 KV slots for 1 tokens" in (decide(2, 0, 1) or "")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    core.set_mode("off")
