"""Tests for the adapter-config contract in the core (ROADMAP M17.1): a LoRA adapter's adapter_config.json against
what each consumer reads of it, the one rule, the repair SGLang's adapter carries, the two engine adapters on fake
objects, and `entail check` on an adapter folder. Pure Python: temp folders, no torch, no engine.
Run: python tests/test_adapter_config.py"""
import json
import math
import os
import sys
import tempfile
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import adapter_config_contract as acc  # noqa: E402
from entail import core  # noqa: E402
from entail.contracts import RULES, Verdict  # noqa: E402

B = "load:test.adapter_config"

# what PEFT 0.21.0 wrote for a plain LoRA on Qwen2.5-3B-Instruct (testbed/m17/make_lora.py, r=64 alpha=128 here):
# every key at its default - note `use_bdlora: null`, which the first real-engine run flagged on every adapter
DEFAULTS = {"alora_invocation_tokens": None, "alpha_pattern": {}, "arrow_config": None, "auto_mapping": None,
            "base_model_name_or_path": "/models/Qwen2.5-3B-Instruct", "bias": "none", "corda_config": None,
            "ensure_weight_tying": False, "eva_config": None, "exclude_modules": None, "fan_in_fan_out": False,
            "inference_mode": True, "init_lora_weights": True, "kasa_config": None, "layer_replication": None,
            "layers_pattern": None, "layers_to_transform": None, "loftq_config": {}, "lora_alpha": 128,
            "lora_bias": False, "lora_dropout": 0.0, "lora_ga_config": None, "megatron_config": None,
            "megatron_core": "megatron.core", "modules_to_save": None, "monteclora_config": None,
            "peft_type": "LORA", "peft_version": "0.21.0", "qalora_group_size": 16, "r": 64, "rank_pattern": {},
            "revision": None, "target_modules": ["v_proj", "q_proj"], "target_parameters": None,
            "task_type": "CAUSAL_LM", "trainable_token_indices": None, "use_bdlora": None, "use_dora": False,
            "use_qalora": False, "use_rslora": False, "velora_config": None}


def folder(cfg):
    d = tempfile.mkdtemp()
    json.dump(cfg, open(os.path.join(d, "adapter_config.json"), "w", encoding="utf-8"))
    return d


def one(decisions):
    assert len(decisions) == 1, decisions
    return decisions[0]


def test_every_key_at_its_default_passes_for_every_consumer():
    for consumer in ("peft", "vllm", "sglang"):
        d = one(acc.check(B, consumer, "test", dict(DEFAULTS), "adapter_config.json", record=False))
        assert d.verdict is Verdict.PASS, (consumer, d)
        assert d.declared.value.total == len(DEFAULTS) and d.declared.value.all_taken
    # a null never declares anything, whatever the key's neutral value is (use_bdlora: null, qalora_group_size: null)
    nulls = {k: None for k in DEFAULTS if k not in ("r", "lora_alpha", "target_modules")}
    for consumer in ("vllm", "sglang"):
        assert one(acc.check(B, consumer, "test", dict(DEFAULTS, **nulls), "adapter_config.json",
                             record=False)).verdict is Verdict.PASS


def test_rslora_is_dropped_by_sglang_resolved_with_the_carrier_and_read_by_vllm_and_peft():
    cfg = dict(DEFAULTS, use_rslora=True)
    d = one(acc.check(B, "sglang", "test", cfg, "adapter_config.json", {"apply_use_rslora": lambda v: True},
                      record=False))
    assert d.verdict is Verdict.RESOLVED and d.handle == "apply_use_rslora", d
    assert math.isclose(d.target, 128 / math.sqrt(64)), d.target      # the core computes the carried scaling
    assert d.rule == RULES["adapter_key_dropped"] and "sqrt(r)" in d.note, d.note
    assert acc.carried_value("sglang", "use_rslora", dict(DEFAULTS)) == 128 / 64
    assert acc.carried_value("vllm", "use_rslora", cfg) is None
    d = one(acc.check(B, "sglang", "test", cfg, "adapter_config.json", record=False))
    assert d.verdict is Verdict.BROKEN and d.rule == RULES["adapter_key_dropped"], d
    for consumer in ("vllm", "peft"):
        assert one(acc.check(B, consumer, "test", cfg, "adapter_config.json", record=False)).verdict is Verdict.PASS


def test_per_module_ranks_are_dropped_by_both_engines_and_read_by_peft():
    cfg = dict(DEFAULTS, rank_pattern={"q_proj": 8}, alpha_pattern={"q_proj": 16})
    for consumer in ("vllm", "sglang"):
        out = acc.check(B, consumer, "test", cfg, "adapter_config.json", record=False)
        assert [d.verdict for d in out] == [Verdict.BROKEN, Verdict.BROKEN], out
        assert {d.note.split("=")[0] for d in out} == {"rank_pattern", "alpha_pattern"}
    assert one(acc.check(B, "peft", "test", cfg, "adapter_config.json", record=False)).verdict is Verdict.PASS


def test_a_key_the_consumer_refuses_loudly_passes_with_a_note_and_an_unknown_key_is_said_once():
    cfg = dict(DEFAULTS, use_dora=True)
    d = one(acc.check(B, "vllm", "test", cfg, "adapter_config.json", record=False))
    assert d.verdict is Verdict.PASS and "refuses" in d.note, d
    cfg = dict(DEFAULTS, future_flag=True, future_off=False)
    out = acc.check(B, "vllm", "test", cfg, "adapter_config.json", record=False)
    assert [d.verdict for d in out] == [Verdict.UNKNOWN, Verdict.PASS], out
    assert out[0].rule == RULES["adapter_key_unknown"] and "future_flag" in out[0].note \
        and "future_off" not in out[0].note


def test_nothing_declared_decides_nothing_and_the_classification_is_explicit():
    assert acc.check(B, "vllm", "test", None, "x", record=False) == []
    assert acc.check(B, "vllm", "test", {}, "x", record=False) == []
    g = acc.classify(dict(DEFAULTS, use_rslora=True, lora_bias=True), "sglang")
    assert g["dropped"] == ["use_rslora", "lora_bias"] or set(g["dropped"]) == {"use_rslora", "lora_bias"}
    assert "r" in g["taken"] and "lora_dropout" in g["taken"] and g["unknown"] == []
    try:
        acc.classify(DEFAULTS, "nobody")
    except ValueError as e:
        assert "nobody" in str(e)
    else:
        raise AssertionError("an unknown consumer must raise")


def decided(fn):
    """Run an adapter's decision in load mode, quietly; return the decisions it recorded."""
    import io
    from contextlib import redirect_stdout

    from entail import load

    n = len(load.LEDGER.decisions)
    was = core.mode()
    core.set_mode("load")
    try:
        with redirect_stdout(io.StringIO()):
            fn()
    finally:
        core.set_mode(was)
    return load.LEDGER.decisions[n:]


def test_the_sglang_adapter_sets_the_rslora_scaling_on_the_adapter_object():
    from entail.adapters import sglang_lora

    cfg = SimpleNamespace(hf_config=dict(DEFAULTS, use_rslora=True), path=folder({}), lora_alpha=128, r=64)
    adapter = SimpleNamespace(config=cfg, scaling=128 / 64)
    new = decided(lambda: sglang_lora._decide(adapter))
    assert math.isclose(adapter.scaling, 128 / math.sqrt(64)), adapter.scaling
    assert any(d.verdict is Verdict.RESOLVED and d.contract.boundary == sglang_lora.BOUNDARY for d in new), new
    cfg2 = SimpleNamespace(hf_config=dict(DEFAULTS), path=folder({}), lora_alpha=128, r=64)
    adapter2 = SimpleNamespace(config=cfg2, scaling=2.0)
    decided(lambda: sglang_lora._decide(adapter2))
    assert adapter2.scaling == 2.0
    assert sglang_lora.read_choice(adapter2)["scaling"] == 2.0


def test_the_vllm_adapter_reads_the_folder_and_reports_what_vllm_drops():
    from entail.adapters import vllm_lora

    d = folder(dict(DEFAULTS, lora_bias=True))
    new = [x for x in decided(lambda: vllm_lora._decide(d)) if x.contract.boundary == vllm_lora.BOUNDARY]
    assert len(new) == 1 and new[0].verdict is Verdict.BROKEN and "lora_bias" in new[0].note, new
    assert vllm_lora.handles(None) == {}


def test_entail_check_decides_an_adapter_folder_per_engine():
    from entail import sites

    d = folder(dict(DEFAULTS, use_rslora=True))
    for engine, verdict in (("sglang", Verdict.RESOLVED), ("vllm", Verdict.PASS), ("transformers", Verdict.PASS)):
        _, model, notes = sites.check_static(d, engine, {})
        got = [x for x in model if x.contract.boundary == f"load:{engine}.adapter_config"]
        assert len(got) == 1 and got[0].verdict is verdict, (engine, got, notes)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
