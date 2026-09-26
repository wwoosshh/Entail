"""Tests for the cache-key contract in the core (ROADMAP M17.2; data/cache_key_fields.json): a store's key against
the fields that shaped the item, the repair vLLM's adapter carries (the mask digest as an extra key), the beam
reorder rule with the 5.12.1 and 5.17.0 rows (the retrospective on transformers#46612), and the two adapters on
fake objects. Pure Python, no engine.
Run: python tests/test_cache_key.py"""
import io
import os
import sys
from contextlib import redirect_stdout
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import cache_key_contract as ckc  # noqa: E402
from entail import core, load  # noqa: E402
from entail.contracts import RULES, Verdict  # noqa: E402

B = "container:test.block_hashes"


def decided(fn):
    n = len(load.LEDGER.decisions)
    was = core.mode()
    core.set_mode("load")
    try:
        with redirect_stdout(io.StringIO()):
            out = fn()
    finally:
        core.set_mode(was)
    return out, load.LEDGER.decisions[n:]


def test_a_field_the_key_does_not_cover_is_broken_or_resolved_and_a_covered_set_passes():
    present = {"prompt_token_ids": True, "prompt_embeds": True, "prompt_is_token_ids": True}
    out = ckc.check(B, "vllm.block_hashes", "vllm", present, "request r1", record=False)
    assert len(out) == 1 and out[0].verdict is Verdict.BROKEN and out[0].rule == RULES["cache_key_incomplete"], out
    assert "prompt_is_token_ids" in out[0].note
    got = []
    out = ckc.check(B, "vllm.block_hashes", "vllm", present, "request r1",
                    {"extend_key_prompt_is_token_ids": lambda f: got.append(f) or True}, record=False)
    assert out[0].verdict is Verdict.RESOLVED and out[0].target == "prompt_is_token_ids" and got == [], out
    out = ckc.check(B, "vllm.block_hashes", "vllm", {"prompt_token_ids": True, "lora_request": True,
                                                       "cache_salt": True, "mm_features": True}, "request r2",
                    record=False)
    assert len(out) == 1 and out[0].verdict is Verdict.PASS and out[0].declared.value.total == 4, out
    assert ckc.check(B, "vllm.block_hashes", "vllm", {}, "request r3", record=False) == []
    try:
        ckc.check(B, "nobody", "vllm", present, "x", record=False)
    except ValueError as e:
        assert "nobody" in str(e)
    else:
        raise AssertionError("an unknown consumer must raise")


def test_the_beam_reorder_rows_give_the_retrospective_on_transformers_46612():
    mamba = {"cache_params": True}
    out = ckc.check("request:test.beam", "transformers.beam_reorder", "transformers", mamba, "MambaForCausalLM.forward",
                    record=False, version="5.12.1")
    assert out[0].verdict is Verdict.BROKEN and "cache_params" in out[0].note, out
    out = ckc.check("request:test.beam", "transformers.beam_reorder", "transformers", mamba, "MambaForCausalLM.forward",
                    record=False, version="5.17.0")
    assert [d.verdict for d in out] == [Verdict.PASS], out
    out = ckc.check("request:test.beam", "transformers.beam_reorder", "transformers", {"past_key_values": True},
                    "LlamaForCausalLM.forward", record=False, version="5.12.1")
    assert [d.verdict for d in out] == [Verdict.PASS], out
    assert ckc.consumer_row("transformers.beam_reorder", "5.12.1")["version"] == "5.12.1"
    assert ckc.consumer_row("transformers.beam_reorder", "9.9.9")["version"] == "5.17.0"


def test_the_vllm_adapter_extends_the_hash_keys_by_the_mask_and_remakes_the_hashes():
    from entail.adapters import vllm_cache_key

    calls = []

    def orig(request, start, end, mm_idx):
        calls.append((start, end))
        return (("lora", "x"),) if request.lora_request else None, mm_idx

    module = SimpleNamespace(generate_block_hash_extra_keys=orig)
    assert vllm_cache_key.install_extension(module) is True
    assert vllm_cache_key.install_extension(module) is False           # once
    fn = module.generate_block_hash_extra_keys
    mixed = SimpleNamespace(prompt_is_token_ids=[False] * 32 + [True], lora_request=None)
    plain = SimpleNamespace(prompt_is_token_ids=None, lora_request=None)
    with_lora = SimpleNamespace(prompt_is_token_ids=[True] * 16, lora_request=object())
    k1, _ = fn(mixed, 0, 16, 0)
    k2, _ = fn(mixed, 0, 16, 0)
    k3, _ = fn(SimpleNamespace(prompt_is_token_ids=[True] * 33, lora_request=None), 0, 16, 0)
    assert k1 == k2 and k1 != k3 and k1[0][0] == "prompt_is_token_ids"
    assert fn(plain, 0, 16, 0) == (None, 0)
    k4, _ = fn(with_lora, 0, 16, 0)
    assert k4[0] == ("lora", "x") and k4[1][0] == "prompt_is_token_ids"
    # the request's present fields and the decision with the repair
    req = SimpleNamespace(request_id="r1", prompt_token_ids=[0] * 33, prompt_embeds=object(),
                          prompt_is_token_ids=[False] * 32 + [True], mm_features=[], lora_request=None,
                          cache_salt=None, block_hashes=["old"], update_block_hashes=lambda: None)
    present = vllm_cache_key.read_choice(req)
    assert present["prompt_is_token_ids"] and present["prompt_embeds"] and not present["mm_features"]
    h = vllm_cache_key.handles(req)
    assert set(h) == {"extend_key_prompt_is_token_ids"}


def test_the_transformers_adapter_reads_the_models_cache_names():
    from entail.adapters import transformers_beam

    class Mamba:
        def forward(self, input_ids=None, cache_params=None, use_cache=None):
            pass

    class Llama:
        def forward(self, input_ids=None, past_key_values=None):
            pass

    assert transformers_beam.read_choice(Mamba()) == {"past_key_values": False, "cache_params": True, "state": False,
                                                      "mems": False, "past_buckets_states": False}
    assert transformers_beam.read_choice(Llama())["past_key_values"] is True
    assert transformers_beam.handles(None) == {}
    _, rec = decided(lambda: transformers_beam._decide(Mamba()))
    got = [d for d in rec if d.contract.boundary == transformers_beam.BOUNDARY]
    # with no transformers installed the base row (5.17.0) applies: cache_params is reordered there
    assert all(d.verdict is Verdict.PASS for d in got), got


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
