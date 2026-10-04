"""Tests for the small serving core written with layer B (ROADMAP M21.2): on a tiny random Qwen3, on the CPU, it gives
every prompt the tokens transformers' greedy generation gives it alone - batched with chunked prefill, with prefix
hits, with a device pool small enough to offload blocks to a CPU tier and load them back, and with preemption - and
the places where a count becomes another count refuse the wrong count. (testbed/m21/check_core.py does the same on
Qwen3-0.6B on the GPU.) Run: python tests/test_frontend_engine.py"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail.core import RoleError  # noqa: E402
from entail.facts import Rotary  # noqa: E402
from entail.frontend import engine as E  # noqa: E402
from entail.frontend import units as U  # noqa: E402

MAX_NEW = 8


def tiny():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=96, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=128,
                         tie_word_embeddings=False)
    model = Qwen3ForCausalLM(config).eval()
    params = model.config.rope_parameters
    return model, Rotary(params.get("rope_type", "default"), float(params["rope_theta"]))


def prompts(n, seed, shared=0):
    g = torch.Generator().manual_seed(seed)
    head = torch.randint(0, 96, (shared,), generator=g).tolist()
    return [head + torch.randint(0, 96, (int(k),), generator=g).tolist()
            for k in torch.randint(3, 21, (n,), generator=g)]


def reference(model, ps):
    out = []
    with torch.no_grad():
        for p in ps:
            ids = torch.tensor([p])
            # the whole prompt is attended to: a pad id given alone would mask every token that happens to equal it
            g = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAX_NEW, do_sample=False,
                               temperature=None, top_p=None, top_k=None, pad_token_id=0)
            out.append(g[0, ids.shape[1]:].tolist())
    return out


def run(model, rot, ps, **kw):
    eng = E.Engine(model, rotary=rot, dtype="float32", **kw)
    with torch.no_grad():
        return eng.generate(ps, MAX_NEW), eng.stats


def check(case, **kw):
    try:
        model, rot = tiny()
    except ImportError:
        print("skip (transformers is not installed)")
        return None
    ps = case()
    got, stats = run(model, rot, ps, **kw)
    assert got == reference(model, ps), (got, stats)
    return stats


def test_a_batch_with_chunked_prefill():
    stats = check(lambda: prompts(5, 1), blocks=32, block=4, max_tokens=12)
    assert stats is None or stats["steps"] > MAX_NEW


def test_prefix_hits():
    stats = check(lambda: prompts(4, 2, shared=12) * 2, blocks=64, block=4, max_rows=1)
    assert stats is None or stats["prefix_hit_tokens"] > 0, stats


def test_offload_to_the_cpu_tier_and_load_back():
    def case():
        a = prompts(3, 3, shared=12)
        return a + prompts(4, 4) + a
    stats = check(case, blocks=12, block=4, cpu_blocks=64, max_rows=1)
    assert stats is None or (stats["stored_blocks"] > 0 and stats["loaded_blocks"] > 0), stats


def test_preemption_recomputes():
    stats = check(lambda: prompts(4, 5), blocks=9, block=4, max_rows=4, max_tokens=8)
    assert stats is None or stats["preempted"] > 0, stats


def test_a_count_becomes_another_only_where_the_code_says_so():
    req = E.Request(0, [1, 2, 3, 4, 5], 4)
    sched = E.to_schedule(req.computed, req.known(), 8)
    assert int(sched) == 5 and sched.fact == E.SCHEDULED
    after = E.written(req.computed, sched)
    assert int(after) == 5 and after.fact == E.COMPUTED
    for wrong in (lambda: E.written(req.known(), sched),                  # known where computed is meant
                  lambda: E.written(after, after),                        # computed where scheduled is meant
                  lambda: E.sampled(after)):                              # computed where known is meant
        try:
            wrong()
        except RoleError:
            continue
        raise AssertionError("a wrong count was taken")
    cache = E.PrefixCache(4, 0)
    try:
        cache.insert(req, req.known())          # storing blocks up to what is known, not what is computed
    except RoleError as e:
        assert "takes computed tokens, got known tokens" in str(e), str(e)
    else:
        raise AssertionError("insert took a known count")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
