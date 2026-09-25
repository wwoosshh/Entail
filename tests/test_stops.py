"""Tests for the stop-set contract in the core (ROADMAP M15.8; vocabulary v7 Stops): what the files declare about the
end of a generation, the union over sources, the two rules, the repair, and the static per-engine table. Pure
Python: temp folders, no torch, no engine. Run: python tests/test_stops.py"""
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import core, load, sources, stops_contract  # noqa: E402
from entail.contracts import RULES, Verdict  # noqa: E402
from entail.facts import Certainty, Fact, Source, Stops  # noqa: E402

B, C = "load:test.stop_set", "test.stop_set"


def folder(config=None, generation=None):
    d = tempfile.mkdtemp()
    if config is not None:
        json.dump(dict({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}, **config),
                  open(os.path.join(d, "config.json"), "w", encoding="utf-8"))
    if generation is not None:
        json.dump(generation, open(os.path.join(d, "generation_config.json"), "w", encoding="utf-8"))
    return d


def raises(fn, text):
    try:
        fn()
    except (ValueError, TypeError) as e:
        assert text in str(e), (text, str(e))
    else:
        raise AssertionError(f"expected an error containing {text!r}")


def one(decisions):
    assert len(decisions) == 1, decisions
    return decisions[0]


def tok_fact(eos):
    return Fact("Stops", Stops(eos=(eos,)), Source("file", "tokenizer_config.json#eos_token (the tokenizer's id)"),
                Certainty.DECLARED)


def test_the_fact_takes_ids_only():
    Stops(eos=(2,))
    Stops(bos=1)
    raises(lambda: Stops(eos=(2, -1)), "Stops.eos")
    raises(lambda: Stops(eos=[2]), "Stops.eos")
    assert Stops().eos == ()          # a consumer that stops on nothing is a value, not an error


def test_the_files_declare_stops_from_both_files_and_a_negative_id_is_unset():
    """Llama 3's shape: config.json names one end, generation_config.json the two the model emits."""
    d = folder({"eos_token_id": 128001, "bos_token_id": 128000, "pad_token_id": -1},
               {"eos_token_id": [128001, 128009], "bos_token_id": 128000})
    r = sources.read_all(d)
    got = sorted(((f.value, f.source.where.split("/")[-1]) for f in r.facts if f.name == "Stops"), key=lambda x: x[1])
    assert got == [(Stops(eos=(128001,), bos=128000), "config.json#eos_token_id"),
                   (Stops(eos=(128001, 128009), bos=128000), "generation_config.json#eos_token_id")], got
    assert any("pad_token_id: [-1] are no token ids" in p for p in r.problems), r.problems
    ids, stops = stops_contract.declared_stops(load.declared(d))
    assert set(ids) == {128001, 128009} and len(ids[128001]) == 2 and len(stops) == 2


def test_the_tokenizer_config_declares_the_end_by_its_own_added_tokens():
    """Nemotron-3-Nano's shape: config.json and an auto-written generation_config.json say </s> (2); the tokenizer's
    eos_token and the chat template say <|im_end|> (11). Read from tokenizer_config.json alone, no tokenizer built."""
    d = folder({"eos_token_id": 2, "bos_token_id": 1}, {"eos_token_id": 2, "_from_model_config": True})
    json.dump({"eos_token": "<|im_end|>", "bos_token": "<s>", "pad_token": None,
               "added_tokens_decoder": {"1": {"content": "<s>", "special": True}, "2": {"content": "</s>", "special": True},
                                        "11": {"content": "<|im_end|>", "special": True}}},
              open(os.path.join(d, "tokenizer_config.json"), "w", encoding="utf-8"))
    r = sources.read_all(d)
    tok = [f for f in r.facts if f.name == "Stops" and "tokenizer_config.json" in f.source.where]
    assert len(tok) == 1 and tok[0].value == Stops(eos=(11,), bos=1) and tok[0].source.kind == "file", tok
    facts = load.declared(d)
    ids, _ = stops_contract.declared_stops(facts)
    assert set(ids) == {2, 11}
    # transformers builds its set from generation_config.json alone: 11 is dropped there; vLLM takes the tokenizer's
    # eos and SGLang's scheduler matches it (M15.8 review), so both hold 11
    assert stops_contract.held_by("transformers", facts) == {2}
    assert stops_contract.held_by("vllm", facts) == {2, 11} and stops_contract.held_by("sglang", facts) == {2, 11}
    r = one(stops_contract.check(B, C, facts, {2}, "held", add_stops=lambda ids: True, record=False))
    assert r.verdict is Verdict.RESOLVED and r.target == (11,) and "tokenizer_config.json#eos_token" in r.note, r
    # an eos_token the file does not list as an added token gives no fact (left to a built tokenizer)
    d2 = folder({"eos_token_id": 2})
    json.dump({"eos_token": "<|end|>", "added_tokens_decoder": {"2": {"content": "</s>"}}},
              open(os.path.join(d2, "tokenizer_config.json"), "w", encoding="utf-8"))
    assert not [f for f in sources.read_all(d2).facts if f.name == "Stops" and "tokenizer_config" in f.source.where]


def test_a_consumer_whose_set_covers_every_declared_end_passes_even_with_more():
    d = folder({"eos_token_id": 2}, {"eos_token_id": [2, 7]})
    r = one(stops_contract.check(B, C, load.declared(d), {2, 7, 9}, "the engine's stop set", record=False))
    assert r.verdict is Verdict.PASS and r.rule == RULES["match"] and r.declared.value == Stops(eos=(2, 7)), r


def test_a_dropped_end_is_resolved_by_adding_it_or_broken_without_a_repair():
    """The Llama 3 class: transformers holds generation_config's ids; config.json's own end is left out when the
    author narrowed the list (saiga_llama3: 128001 vs 128009)."""
    d = folder({"eos_token_id": 128001}, {"eos_token_id": 128009})
    added = []
    facts = load.declared(d)
    r = one(stops_contract.check(B, C, facts, {128009}, "generation_config as the model holds it",
                                 add_stops=lambda ids: added.append(tuple(ids)) or True, record=False))
    assert r.verdict is Verdict.RESOLVED and r.handle == "add_stops" and r.target == (128001,), r
    assert "lacks 128001" in r.note and "config.json#eos_token_id" in r.note
    # record=False does not run the handle; the recording path does (through load.resolve)
    assert added == []
    n = len(load.LEDGER.decisions)
    core.set_mode("load")
    try:
        with redirect_stdout(io.StringIO()):
            stops_contract.check(B, C, facts, {128009}, "held", add_stops=lambda ids: added.append(tuple(ids)) or True)
    finally:
        core.set_mode("off")
    assert added == [(128001,)] and load.LEDGER.decisions[n].verdict is Verdict.RESOLVED
    # no repair offered: broken under the default policy, refused where the policy stops
    r = one(stops_contract.check(B, C, facts, {128009}, "held", record=False))
    assert r.verdict is Verdict.BROKEN and r.rule == RULES["stop_dropped"] and not r.blocking, r
    os.environ["ENTAIL_ON_BROKEN"] = "stop"      # stopping is chosen (M5.4); the policy is read from the environment
    try:
        r = one(stops_contract.check(B, C, facts, {128009}, "held", record=False))
        assert r.verdict is Verdict.REFUSED and r.blocking, r
    finally:
        os.environ.pop("ENTAIL_ON_BROKEN", None)
    stops_contract.reset(B)


def test_an_id_past_the_tokenizer_is_broken():
    d = folder({"eos_token_id": 2}, {"eos_token_id": [2, 40000]})
    r = stops_contract.check(B, C, load.declared(d), {2, 40000}, "held", tokenizer_size=32000, record=False)
    bad = [x for x in r if x.rule == RULES["stop_id_out_of_range"]]
    assert len(bad) == 1 and bad[0].verdict is Verdict.BROKEN and "eos id 40000" in bad[0].note, r


def test_an_eos_that_is_not_special_is_noted_on_a_pass():
    d = folder({"eos_token_id": 2}, {"eos_token_id": [2, 7]})
    r = one(stops_contract.check(B, C, load.declared(d), {2, 7}, "held", special_ids=[2], record=False))
    assert r.verdict is Verdict.PASS and "eos id(s) 7 are not special" in r.note, r


def test_a_consumer_that_holds_no_end_at_all_is_resolved_not_crashed():
    """M15.8 E2: hmellor/tiny-random-LlamaForCausalLM ships a generation_config.json without eos_token_id, so
    transformers' set is empty; the first build raised inside entail (an empty Stops was refused) and the run was
    told 'entail failed here'."""
    d = folder({"eos_token_id": 1}, {"do_sample": False})
    r = one(stops_contract.check(B, C, load.declared(d), set(), "held", add_stops=lambda ids: True, record=False))
    assert r.verdict is Verdict.RESOLVED and r.target == (1,) and r.chosen.value == Stops(), r


def test_no_declaration_decides_nothing_and_an_unread_set_only_checks_the_ids():
    assert stops_contract.check(B, C, load.declared(folder({"hidden_size": 8})), {2}, "held", record=False) == []
    d = folder({"eos_token_id": 2})
    r = one(stops_contract.check(B, C, load.declared(d), None, "held", tokenizer_size=10, record=False))
    assert r.verdict is Verdict.PASS and r.chosen.value == Stops(eos=(2,)), r


def test_the_fallback_stands_in_only_when_the_file_is_absent_and_a_bos_is_not_an_end():
    """M15.8 review: transformers and vLLM fall back to config.json only when generation_config.json is ABSENT; a
    file without eos_token_id gives an empty set. And hmellor/tiny-random-Llama declares eos 1 in config.json where
    1 is the tokenizer's <s>: a beginning is not an end to add."""
    d = folder({"eos_token_id": 2}, {"do_sample": False})            # generation_config.json present, no eos
    facts = load.declared(d)
    assert stops_contract.held_by("transformers", facts, path=d) == set()
    assert stops_contract.held_by("vllm", facts, tokenizer_eos=2, path=d) == {2}
    assert stops_contract.held_by("sglang", facts, path=d) == {2}
    d2 = folder({"eos_token_id": 2})                                   # no generation_config.json: config stands in
    assert stops_contract.held_by("transformers", load.declared(d2), path=d2) == {2}
    d3 = folder({"eos_token_id": 1, "bos_token_id": 0}, {"eos_token_id": [1, 2]})
    json.dump({"eos_token": "</s>", "bos_token": "<s>",
               "added_tokens_decoder": {"1": {"content": "<s>"}, "2": {"content": "</s>"}}},
              open(os.path.join(d3, "tokenizer_config.json"), "w", encoding="utf-8"))
    r = one(stops_contract.check(B, C, load.declared(d3), {2}, "held", add_stops=lambda ids: True, record=False))
    assert r.verdict is Verdict.PASS and "1 is declared as eos" in r.note and "declared bos: not an end" in r.note, r
    # GPT-2's shape: one id is the beginning AND the end (<|endoftext|> 50256 in config.json and generation_config):
    # a source that calls it both declares an end, so it is held to (the first fix dropped 42 of 230 folders)
    d4 = folder({"eos_token_id": 50256, "bos_token_id": 50256}, {"eos_token_id": 50256, "bos_token_id": 50256})
    r = one(stops_contract.check(B, C, load.declared(d4), {50256}, "held", record=False))
    assert r.verdict is Verdict.PASS and r.declared.value.eos == (50256,) and "not an end" not in r.note, r
    r = one(stops_contract.check(B, C, load.declared(d4), set(), "held", add_stops=lambda ids: True, record=False))
    assert r.verdict is Verdict.RESOLVED and r.target == (50256,), r
    # the SGLang table row: the scheduler matches the tokenizer's eos too, so its set holds all three sources
    assert stops_contract.held_by("sglang", load.declared(d3), path=d3) == {1, 2}


def test_the_tokenizers_end_is_found_in_tokenizer_json_when_the_config_lists_no_id():
    """55 of 230 popular folders list eos_token in tokenizer_config.json without an added_tokens_decoder entry for
    it (DeepSeek, GLM, Pythia ...); tokenizer.json names it."""
    d = folder({"eos_token_id": 2}, {"eos_token_id": 2})
    json.dump({"eos_token": "<|end|>"}, open(os.path.join(d, "tokenizer_config.json"), "w", encoding="utf-8"))
    json.dump({"added_tokens": [{"id": 7, "content": "<|end|>", "special": True}],
               "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}}},
              open(os.path.join(d, "tokenizer.json"), "w", encoding="utf-8"))
    tok = [f for f in sources.read_all(d).facts if f.name == "Stops" and "tokenizer_config" in f.source.where]
    assert len(tok) == 1 and tok[0].value.eos == (7,) and "tokenizer.json" in tok[0].source.where, tok
    # a folder is read once per process: the same object's facts come back while the files stand
    a = sources.read_all(d)
    b = sources.read_all(d)
    assert [str(f.value) for f in a.facts] == [str(f.value) for f in b.facts] and a is not b
    json.dump({"eos_token": "<|end|>", "added_tokens_decoder": {"9": {"content": "<|end|>"}}},
              open(os.path.join(d, "tokenizer_config.json"), "w", encoding="utf-8"))
    c = [f for f in sources.read_all(d).facts if f.name == "Stops" and "tokenizer_config" in f.source.where]
    assert c[0].value.eos == (9,), c            # the file changed: read again


def test_the_static_table_builds_each_engines_set_as_its_code_does():
    """transformers: generation_config only (config when absent); vLLM: the tokenizer's eos plus generation_config;
    SGLang: config plus generation_config (data/stops_sources.json)."""
    d = folder({"eos_token_id": 128001}, {"eos_token_id": 128009})
    facts = load.declared(d)
    assert stops_contract.held_by("transformers", facts) == {128009}
    assert stops_contract.held_by("vllm", facts, tokenizer_eos=128001) == {128001, 128009}
    assert stops_contract.held_by("sglang", facts) == {128001, 128009}
    assert stops_contract.held_by("nothing", facts) is None
    # no generation_config.json: transformers and vLLM fall back to config.json
    d = folder({"eos_token_id": [1, 107]})
    facts = load.declared(d)
    assert stops_contract.held_by("transformers", facts) == {1, 107}
    assert stops_contract.held_by("vllm", facts, tokenizer_eos=1) == {1, 107}
    # the tokenizer's own fact counts as the tokenizer source
    facts.facts.setdefault("Stops", []).append(tok_fact(5))
    assert stops_contract.held_by("vllm", facts) == {1, 107, 5}
    assert stops_contract.held_by("transformers", facts) == {1, 107}
    for eng, entry in stops_contract.sources_table().items():
        if eng.startswith("_"):
            continue
        assert entry["ref"] and entry["version"] and entry["reads"], eng


def decided(fn):
    """Run an adapter's decision in load mode, quietly; return the decisions it recorded."""
    n = len(load.LEDGER.decisions)
    core.set_mode("load")
    try:
        with redirect_stdout(io.StringIO()):
            fn()
    finally:
        core.set_mode("off")
    return load.LEDGER.decisions[n:]


def test_the_transformers_adapter_adds_a_dropped_end_to_generation_config():
    from types import SimpleNamespace
    from entail.adapters import transformers_stops
    d = folder({"eos_token_id": 128001}, {"eos_token_id": 128009})
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=128009))
    r = decided(lambda: transformers_stops._decide(d, {}, model))
    assert len(r) == 1 and r[0].verdict is Verdict.RESOLVED and model.generation_config.eos_token_id == [128001, 128009], r
    # the repaired model again: its set now covers every declared end, a pass
    again = decided(lambda: transformers_stops._decide(d, {}, model))
    assert all(x.verdict is Verdict.PASS for x in again), again
    # a model without a generation config decides nothing
    assert decided(lambda: transformers_stops._decide(d, {}, SimpleNamespace())) == []
    transformers_stops.reset()


def test_the_vllm_adapter_counts_the_tokenizers_eos_and_writes_the_fields():
    from types import SimpleNamespace
    from entail.adapters import vllm_stops
    d = folder({"eos_token_id": 128001}, {"eos_token_id": 128009})
    tok = SimpleNamespace(all_special_ids=[128000, 128001, 128009], __len__=lambda self: 128256)
    proc = SimpleNamespace(model_config=SimpleNamespace(model=d, hf_config_path=None, revision=None),
                           generation_config_fields={"eos_token_id": 128009},
                           renderer=SimpleNamespace(get_eos_token_id=lambda: 128001), tokenizer=None)
    r = decided(lambda: vllm_stops._decide(proc))
    assert r == [] or all(x.verdict is Verdict.PASS for x in r), r     # the tokenizer's eos covers config.json's
    held, tok_eos, size, special = vllm_stops.read_choice(proc)
    assert held == {128001, 128009} and tok_eos == 128001 and size is None
    # the fields alone, without the tokenizer's eos: the dropped end is written into the fields
    proc2 = SimpleNamespace(model_config=SimpleNamespace(model=d, hf_config_path=None, revision=None),
                            generation_config_fields={"eos_token_id": [128009]},
                            renderer=SimpleNamespace(get_eos_token_id=lambda: None), tokenizer=None)
    vllm_stops.reset()
    r = decided(lambda: vllm_stops._decide(proc2))
    assert [x.verdict for x in r] == [Verdict.RESOLVED] and proc2.generation_config_fields["eos_token_id"] == [128001, 128009], r
    vllm_stops.reset()


def test_the_sglang_adapter_counts_the_tokenizers_end_the_scheduler_matches():
    """SGLang's scheduler stops on the two files' ids and on the tokenizer's eos (unless skip_tokenizer_init), so
    the tokenizer's declared end is held, not repaired (M15.8 review: the first adapter 'repaired' it)."""
    from types import SimpleNamespace
    from entail.adapters import sglang_stops
    d = folder({"eos_token_id": 2}, {"eos_token_id": 2})
    json.dump({"eos_token": "<|im_end|>", "added_tokens_decoder": {"2": {"content": "</s>"}, "11": {"content": "<|im_end|>"}}},
              open(os.path.join(d, "tokenizer_config.json"), "w", encoding="utf-8"))
    mc = SimpleNamespace(model_path=d, revision=None, hf_eos_token_id={2})
    r = decided(lambda: sglang_stops._decide(mc))
    assert all(x.verdict is Verdict.PASS for x in r) and mc.hf_eos_token_id == {2}, r
    sglang_stops.reset()
    mc2 = SimpleNamespace(model_path=d, revision=None, hf_eos_token_id={2})
    r = decided(lambda: sglang_stops._decide(mc2, skip_tokenizer_init=True))
    assert [x.verdict for x in r] == [Verdict.RESOLVED] and mc2.hf_eos_token_id == {2, 11}, r
    sglang_stops.reset()
    # config.json's end left out of generation_config.json is still added (the files' ids)
    d2 = folder({"eos_token_id": 128001}, {"eos_token_id": 128009})
    mc3 = SimpleNamespace(model_path=d2, revision=None, hf_eos_token_id={128009})
    r = decided(lambda: sglang_stops._decide(mc3))
    assert [x.verdict for x in r] == [Verdict.RESOLVED] and mc3.hf_eos_token_id == {128001, 128009}, r
    sglang_stops.reset()


def test_entail_check_decides_the_stop_set_the_engine_builds():
    """The static check: transformers drops config.json's end when generation_config.json narrows the list; vLLM
    keeps it through the tokenizer's eos (when a tokenizer can be built; here none can, so vLLM sees the files)."""
    from entail import sites
    try:
        import transformers  # noqa: F401
    except ImportError:
        print("skip: transformers not installed")
        return
    d = folder({"hidden_size": 16, "num_attention_heads": 2, "num_hidden_layers": 1, "vocab_size": 32,
                "eos_token_id": 3}, {"eos_token_id": 5})
    _, model, notes = sites.check_static(d, "transformers", {"attention": "sdpa"})
    st = [x for x in model if x.name == "Stops"]
    # statically a dropped end is said as what entail adds at load (resolved), with the ids on the decision
    assert st and st[0].verdict is Verdict.RESOLVED and st[0].rule == RULES["stop_dropped"] and st[0].target == (3,) \
        and "lacks 3" in st[0].note, (st, notes)
    _, model, notes = sites.check_static(d, "sglang", {"attention": "triton"})
    st = [x for x in model if x.name == "Stops"]
    assert st and st[0].verdict is Verdict.PASS, (st, notes)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
