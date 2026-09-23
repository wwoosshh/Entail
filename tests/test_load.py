"""Tests for the load-time contracts (ROADMAP M3.2): what counts as declared at load, each contract's verdicts on the
shapes of the M3 test problems (rolebench 02, 06, 07, 08, 15, 17; fd-rope, fd-softcap, fd-shift), the observations
read from checkpoint bytes, and enforce(). No engine and no GPU. Run: python tests/test_load.py"""
import io
import json
import os
import shutil
import struct
import sys
import tempfile
from contextlib import redirect_stdout
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
# These tests check what stops, so they run under the policy that stopped before M5.4 (ENTAIL_ON_BROKEN=stop,
# unknown meaning-changing facts required); the default, which reports and goes on, is tested in test_report.py.
os.environ["ENTAIL_ON_BROKEN"] = "stop"
os.environ["ENTAIL_UNKNOWN"] = "require"
from entail import caps, load, observe, sites  # noqa: E402
from entail.contracts import Verdict  # noqa: E402
from entail.core import RoleError  # noqa: E402
from entail.coverage import Coverage  # noqa: E402
from entail.facts import Certainty, Fact, Layout, ModelProps, Rotary, Source  # noqa: E402
from entail.policies import Policy  # noqa: E402

STOPS = dict(on_broken="stop", on_unknown_meaning_changing="require")   # the policy before M5.4

GEMMA = {"architectures": ["Gemma2ForCausalLM"], "attn_logit_softcapping": 50.0, "sliding_window": 4096,
         "tie_word_embeddings": True, "rope_theta": 10000.0}
LOAD = Policy(mode="load", **STOPS)


def write_safetensors(path, tensors):
    """tensors: name -> (dtype, shape, raw bytes)."""
    header, blob = {}, b""
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [len(blob), len(blob) + len(raw)]}
        blob += raw
    h = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + blob)


def model_folder(config, tensors=None):
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f)
    if tensors is not None:
        write_safetensors(os.path.join(d, "model.safetensors"), tensors)
    return d


def f32(values):
    return struct.pack(f"<{len(values)}f", *values)


EMBED = ("F32", (4, 2), f32([1, 2, 3, 4, 5, 6, 7, 8]))
OTHER_HEAD = ("F32", (4, 2), f32([1, 2, 3, 4, 5, 6, 7, 9]))


def only(decisions):
    assert len(decisions) == 1, [(d.contract.boundary, d.verdict) for d in decisions]
    return decisions[0]


# --- declared ------------------------------------------------------------------------------------------------------

def test_declared_from_files_and_what_the_engine_holds():
    d = model_folder({"attn_logit_softcapping": 50.0})
    held = SimpleNamespace(attn_logit_softcapping=50.0, sliding_window=4096, tie_word_embeddings=False)
    facts = load.declared(d, held)
    props = facts.get("ModelProps")
    assert [(f.source.kind, f.certainty, f.value) for f in props] == [
        ("config", Certainty.DECLARED, ModelProps(softcap=50.0)),
        ("default", Certainty.DEFAULTED, ModelProps(sliding_window=4096, tie_word_embeddings=False))]
    assert "not in the model's files, so a default of the config class" in props[1].source.where
    # a value the engine holds differently from the file: kept as the engine's, so the disagreement shows
    changed = load.declared(d, SimpleNamespace(attn_logit_softcapping=30.0)).get("ModelProps")
    assert changed[1].source.kind == "engine" and changed[1].value == ModelProps(softcap=30.0)
    # no files: the config object is the declaration (a config built in code, rolebench 08)
    code = load.declared(None, SimpleNamespace(attn_logit_softcapping=5.0, sliding_window=256)).get("ModelProps")
    assert [(f.source.kind, f.certainty) for f in code] == [("config", Certainty.DECLARED)]
    shutil.rmtree(d)


# --- attention (rolebench 06, 08, 17; fd-softcap) ------------------------------------------------------------------

def test_attention_routes_to_a_backend_measured_to_honour_it():
    d = model_folder(GEMMA)
    facts = load.declared(d)
    r = only(load.attention("transformers", "sdpa", facts, policy=LOAD))            # rolebench 08, fd-softcap
    assert r.verdict is Verdict.RESOLVED and r.target == "eager" and r.handle == "switch_attention_backend"
    assert r.declared.value == ModelProps(softcap=50.0, sliding_window=4096)          # the tie is not attention's
    assert r.chosen.value == ModelProps(sliding_window=4096) and r.chosen.certainty is Certainty.VERIFIED
    assert only(load.attention("transformers", "eager", facts, policy=LOAD)).verdict is Verdict.PASS
    r = only(load.attention("sglang", "torch_native", facts, policy=LOAD))            # rolebench 17
    assert r.verdict is Verdict.RESOLVED and r.target == "triton"
    r = only(load.attention("sglang", "flashinfer", facts, policy=LOAD))
    assert r.verdict is Verdict.RESOLVED and r.target == "triton" and r.chosen.certainty is Certainty.INFERRED
    assert only(load.attention("vllm", "FLASH_ATTN", facts, policy=LOAD)).verdict is Verdict.PASS
    r = only(load.attention("transformers", "sdpa", facts, policy=Policy(mode="load", **STOPS, on_mismatch="refuse")))
    assert r.verdict is Verdict.REFUSED and r.blocking
    shutil.rmtree(d)


def test_attention_paged_has_nowhere_to_go():
    facts = load.declared(None, SimpleNamespace(**GEMMA))
    r = only(load.attention("transformers", "sdpa", facts, policy=LOAD, role="paged_attention"))
    assert r.verdict is Verdict.REFUSED and r.rule.startswith("the consumer differs") and r.blocking


def test_attention_unknown_backend_and_nothing_declared():
    facts = load.declared(None, SimpleNamespace(**GEMMA))
    r = only(load.attention("transformers", "flash_attention_2", facts, policy=LOAD))
    assert r.verdict is Verdict.UNKNOWN and not r.blocking and "not in the capability table" in r.chosen.source.where
    assert only(load.attention("transformers", "flash_attention_2", facts, policy=Policy(mode="debug"))).blocking
    llama = load.declared(None, SimpleNamespace(tie_word_embeddings=True, rope_theta=500000.0))
    assert load.attention("sglang", "flashinfer", llama, policy=LOAD) == []           # nothing for attention to honour


def test_attention_window_only_mechanism():
    """rolebench 06: a kernel whose custom mask replaces its window; the case's own table says it drops the window."""
    table = caps.from_rows([
        {"consumer": "rb06.attention.custom_mask", "fact": "ModelProps.sliding_window", "honours": False,
         "evidence": "measured", "ref": "rolebench case 06"},
        {"consumer": "rb06.attention.windowed", "fact": "ModelProps.sliding_window", "honours": True,
         "evidence": "measured", "ref": "rolebench case 06"}], {"rb06.attention": ["windowed"]})
    facts = load.declared(None, SimpleNamespace(sliding_window=8))
    r = only(load.attention("rb06", "custom_mask", facts, table=table, policy=LOAD))
    assert r.verdict is Verdict.RESOLVED and r.target == "windowed"


def test_a_property_only_a_default_gives_is_unknown():
    d = model_folder({"architectures": ["Gemma2ForCausalLM"]})
    facts = load.declared(d, SimpleNamespace(attn_logit_softcapping=50.0))
    r = only(load.attention("transformers", "sdpa", facts, policy=LOAD))
    assert r.verdict is Verdict.UNKNOWN and r.rule == "only a default, never declared" and r.blocking
    shutil.rmtree(d)


# --- tie (rolebench 07) --------------------------------------------------------------------------------------------

def test_observe_tie_from_bytes():
    same = model_folder({}, {"model.embed_tokens.weight": EMBED, "lm_head.weight": EMBED})
    other = model_folder({}, {"model.embed_tokens.weight": EMBED, "lm_head.weight": OTHER_HEAD})
    none = model_folder({}, {"model.embed_tokens.weight": EMBED})
    shape = model_folder({}, {"model.embed_tokens.weight": EMBED, "lm_head.weight": ("F32", (2, 2), f32([1, 2, 3, 4]))})
    assert observe.tie(same) is None                                         # consistent with a tie, not proof
    o = observe.tie(other)
    assert o.value == ModelProps(tie_word_embeddings=False) and "row 3 holds other bytes" in o.source.where
    assert observe.tie(none).value == ModelProps(tie_word_embeddings=True)
    assert "differs from the embedding" in observe.tie(shape).source.where
    assert observe.tie(tempfile.mkdtemp()) is None
    for p in (same, other, none, shape):
        shutil.rmtree(p)


def test_tie_declaration_against_the_data():
    d = model_folder({"tie_word_embeddings": True}, {"model.embed_tokens.weight": EMBED, "lm_head.weight": OTHER_HEAD})
    facts = load.declared(d)
    r = only(load.tie("transformers", facts, d, loader_ties=True, policy=LOAD))    # rolebench 07
    assert r.verdict is Verdict.REFUSED and r.rule == "the declaration contradicts the data" and r.blocking
    r = only(load.tie("transformers", facts, d, loader_ties=True, policy=Policy(mode="load", **STOPS,
                                                                              on_false_declaration="use_data")))
    assert r.verdict is Verdict.REFUSED and r.rule.startswith("the consumer differs")   # the loader still ties
    ok = model_folder({"tie_word_embeddings": True}, {"model.embed_tokens.weight": EMBED})
    r = only(load.tie("transformers", load.declared(ok), ok, loader_ties=True, policy=LOAD))
    assert r.verdict is Verdict.PASS and r.declared.certainty is Certainty.VERIFIED
    untied = model_folder({"tie_word_embeddings": False}, {"model.embed_tokens.weight": EMBED})
    r = only(load.tie("transformers", load.declared(untied), untied, loader_ties=False, policy=LOAD))
    assert r.verdict is Verdict.REFUSED                                          # no head to use: the model has none
    silent = model_folder({}, {"model.embed_tokens.weight": EMBED, "lm_head.weight": OTHER_HEAD})
    facts = load.declared(silent, SimpleNamespace(tie_word_embeddings=True))      # the class default says tie
    r = only(load.tie("transformers", facts, silent, loader_ties=True, policy=LOAD))
    assert r.verdict is Verdict.REFUSED and r.declared.source.kind == "data"
    for p in (d, ok, untied, silent):
        shutil.rmtree(p)


# --- config keys (rolebench 15) ------------------------------------------------------------------------------------

def test_config_keys():
    known = {"vocab_size", "rope_parameters", "hidden_size"}
    resolved = {"vocab_size": 10, "rope_parameters": {"rope_theta": 500000.0}, "hidden_size": 8}
    raw = {"vocab_size": 10, "rope_theta": 500000.0, "hidden_size": 8, "rope_scale": 4.0, "architectures": ["X"],
           "transformers_version": "5.17.0"}
    r = only(load.config_keys("transformers", [("", raw, known, resolved)], "config.json", LOAD))
    assert r.verdict is Verdict.REFUSED and r.chosen.value == Coverage(6, 5, ("rope_scale",))
    assert "architectures" in r.note and "transformers_version" in r.note
    raw.pop("rope_scale")
    r = only(load.config_keys("transformers", [("", raw, known, resolved)], "config.json", LOAD))
    assert r.verdict is Verdict.PASS                              # rope_theta renamed, but its value survived
    moved = dict(raw, rope_scaling={"type": "llama3", "factor": 32.0}, quantization_config={"quant_method": "fp8"})
    held = dict(resolved, rope_parameters={"rope_type": "llama3", "factor": 32.0, "rope_theta": 500000.0})
    r = only(load.config_keys("transformers", [("", moved, known, held)], "config.json", LOAD))
    assert r.verdict is Verdict.PASS and "quantization_config" in r.note   # a dict that moved; a key read elsewhere
    lost = dict(raw, rope_scaling={"rope_type": "yarn", "factor": 4.0})
    assert only(load.config_keys("transformers", [("", lost, known, held)], "c", LOAD)).verdict is Verdict.REFUSED
    r = only(load.config_keys("transformers", [("text_config.", {"rope_scale": 4.0}, known, resolved)], "c", LOAD))
    assert r.chosen.value.left == ("text_config.rope_scale",)


# --- rotary (fd-rope) ----------------------------------------------------------------------------------------------

def test_rotary_write():
    meant = Rotary("llama3", theta=500000.0, factor=32.0, original_max_position=8192)
    lost = Rotary("llama3", theta=None, factor=32.0, original_max_position=8192)
    r = only(load.rotary_write("vllm", "LlamaConfig", "rope_scaling", meant, lost, policy=LOAD))
    assert r.verdict is Verdict.RESOLVED and r.handle == "rope_write_as_file" and r.declared.source.kind == "user"
    assert only(load.rotary_write("vllm", "LlamaConfig", "rope_scaling", meant, meant, policy=LOAD)).verdict \
        is Verdict.PASS
    r = only(load.rotary_write("vllm", "LlamaConfig", "rope_scaling", meant, lost,
                               policy=Policy(mode="load", **STOPS, on_mismatch="refuse")))
    assert r.verdict is Verdict.REFUSED and r.blocking
    r = only(load.rotary_write("t", "Gemma3TextConfig", "rope_theta", Rotary(theta=1e6), Rotary(theta=1e4),
                               scope="full_attention", policy=LOAD))
    assert r.contract.boundary == "load:t.config.rope_theta[full_attention]"


class Held:
    """A config object an engine holds (a plain class: SimpleNamespace takes no weak reference)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


LLAMA32 = {"rope_theta": 500000.0, "tie_word_embeddings": True,
           "rope_scaling": {"rope_type": "llama3", "factor": 32.0, "original_max_position_embeddings": 8192}}


def test_rotary_held_against_the_files_and_the_users_write():
    d = model_folder(LLAMA32)
    scaling = {"rope_type": "llama3", "factor": 32.0, "original_max_position_embeddings": 8192}
    lost = Held(rope_parameters=dict(scaling))                          # the base is gone (fd-rope)
    r = only(load.rotary_held("vllm", load.declared(d, lost), lost, LOAD))
    assert r.verdict is Verdict.REFUSED and r.chosen.value.theta is None and r.blocking
    kept = Held(rope_parameters=dict(scaling, rope_theta=500000.0))
    assert only(load.rotary_held("vllm", load.declared(d, kept), kept, LOAD)).verdict is Verdict.PASS
    # the user changed the scaling at launch, and the write was read as config.json reads it: the object now holds
    # the user's declaration, which outranks the files; the disagreement with the files is recorded, not refused
    mine = Held(rope_parameters={"rope_type": "linear", "factor": 2.0, "rope_theta": 500000.0})
    meant = Rotary("linear", theta=500000.0, factor=2.0)
    load.rotary_write("vllm", "LlamaConfig", "rope_scaling", meant, meant, policy=LOAD, config=mine)
    r = only(load.rotary_held("vllm", load.declared(d, mine), mine, LOAD))
    assert r.verdict is Verdict.PASS and r.declared.source.kind == "user" and r.conflict
    per_layer = Held(rope_parameters={"full_attention": {"rope_type": "default", "rope_theta": 1e6},
                                      "sliding_attention": {"rope_type": "default", "rope_theta": 1e4}})
    r = only(load.rotary_held("t", load.declared(d, per_layer), per_layer, LOAD))
    assert r.verdict is Verdict.UNKNOWN and not r.blocking and "per layer type" in r.note
    shutil.rmtree(d)


def test_the_users_write_reaches_a_process_that_gets_the_config_pickled():
    """vLLM builds the config in one process and runs the model in its engine core, which gets the config pickled:
    the user's declaration travels by model folder in the environment (ENTAIL_DECLARED)."""
    d = model_folder(LLAMA32)
    saved = os.environ.pop(load.ENV_DECLARED, None)
    try:
        parent = Held(_name_or_path=d, rope_parameters={"rope_type": "linear", "factor": 2.0, "rope_theta": 500000.0})
        meant = Rotary("linear", theta=500000.0, factor=2.0)
        load.rotary_write("vllm", "LlamaConfig", "rope_scaling", meant, meant, policy=LOAD, config=parent)
        assert d in json.loads(os.environ[load.ENV_DECLARED]) or os.path.realpath(d) in \
            json.loads(os.environ[load.ENV_DECLARED])
        child = Held(_name_or_path=d, rope_parameters=dict(parent.rope_parameters))   # a new object, same folder
        r = only(load.rotary_held("vllm", load.declared(d, child), child, LOAD))
        assert r.verdict is Verdict.PASS and r.declared.source.kind == "user"
        os.environ.pop(load.ENV_DECLARED)
        r = only(load.rotary_held("vllm", load.declared(d, child), child, LOAD))
        assert r.verdict is Verdict.REFUSED     # without it the child would compare with the files and refuse
    finally:
        os.environ.pop(load.ENV_DECLARED, None)
        if saved is not None:
            os.environ[load.ENV_DECLARED] = saved
        shutil.rmtree(d)


def test_model_contracts_at_a_load_hook():
    d = model_folder(LLAMA32, {"model.embed_tokens.weight": EMBED, "lm_head.weight": OTHER_HEAD})
    held = Held(rope_parameters=dict(LLAMA32["rope_scaling"], rope_theta=500000.0), tie_word_embeddings=True)
    out = load.model_contracts("vllm", d, held, True, LOAD)            # rolebench 07 through vLLM's loader
    assert [(x.contract.boundary, x.verdict) for x in out] == [
        ("load:vllm.loader", Verdict.REFUSED), ("load:vllm.config.rope_parameters", Verdict.PASS)]
    shutil.rmtree(d)


def test_safely_turns_an_entail_failure_into_a_report():
    out = io.StringIO()
    with redirect_stdout(out):
        got = load.safely("load:x", "x.c", "ModelProps", lambda: 1 / 0, default="fallback")
    assert got == "fallback" and "this boundary could not be checked" in out.getvalue()
    assert "ZeroDivisionError" in out.getvalue()
    try:
        load.safely("load:x", "x.c", "ModelProps", lambda: (_ for _ in ()).throw(RoleError("stop")))
        raise AssertionError("a RoleError must pass through")
    except RoleError:
        pass


# --- layout (rolebench 02) -----------------------------------------------------------------------------------------

def test_layout_scale_format():
    table = caps.from_rows([
        {"consumer": "rb02.linear.ue8m0_gemm", "fact": "Layout.scale_format", "honours": False, "reads": "ue8m0",
         "evidence": "measured", "ref": "rolebench case 02"},
        {"consumer": "rb02.linear.fp32_gemm", "fact": "Layout.scale_format", "honours": True,
         "evidence": "measured", "ref": "rolebench case 02"}])
    fp8 = {"quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]}}
    d = model_folder(fp8, {"w.weight": ("F8_E4M3", (2, 2), b"\0" * 4),
                           "w.weight_scale_inv": ("F32", (1, 2), f32([0.3, 0.5]))})
    facts, seen = load.declared(d), observe.scale_format(d)
    assert seen.value == Layout("fp8_block", scale_format="fp32")
    r = only(load.layout("rb02.linear.ue8m0_gemm", facts, table, seen, LOAD))
    assert r.verdict is Verdict.REFUSED and r.chosen.value.scale_format == "ue8m0" and r.rule.startswith("the consumer")
    r = only(load.layout("rb02.linear.fp32_gemm", facts, table, seen, LOAD))
    assert r.verdict is Verdict.PASS and r.declared.value.scale_format == "fp32"      # the data filled it in
    wrong = model_folder({"quantization_config": dict(fp8["quantization_config"], scale_fmt="ue8m0")},
                         {"w.weight_scale_inv": ("F32", (1, 1), f32([0.3]))})
    r = only(load.layout("rb02.linear.fp32_gemm", load.declared(wrong), table, observe.scale_format(wrong), LOAD))
    assert r.verdict is Verdict.REFUSED and r.rule == "the declaration contradicts the data"
    # scales converted to powers of two and kept as fp32 are ue8m0 values: the ue8m0 kernel reads them as they are
    pow2 = model_folder(fp8, {"w.weight_scale_inv": ("F32", (1, 3), f32([0.25, 1.0, 8.0]))})
    assert observe.scale_format(pow2).value.scale_format == "ue8m0"
    assert "every value a power of two" in observe.scale_format(pow2).source.where
    r = only(load.layout("rb02.linear.ue8m0_gemm", load.declared(pow2), table, observe.scale_format(pow2), LOAD))
    assert r.verdict is Verdict.PASS
    for p in (d, wrong, pow2):
        shutil.rmtree(p)


# --- weights (fd-shift), cannot_check, enforce ---------------------------------------------------------------------

def test_weights_taken_and_cannot_check():
    r = only(load.weights_taken("vllm", "/m", 16, ["layers.0.mlp.gate_up_proj row 4"], LOAD))
    assert r.verdict is Verdict.REFUSED and r.blocking
    assert only(load.weights_taken("vllm", "/m", 16, [], LOAD)).verdict is Verdict.PASS
    c = load.cannot_check("load:vllm.weights", "vllm.loader", "Coverage", "quantised weights are not compared", LOAD)
    assert c.verdict is Verdict.UNKNOWN and not c.blocking and c.rule == "this boundary could not be checked"
    assert load.cannot_check("b", "c", "Coverage", "x", Policy(mode="debug")).blocking


def test_enforce_records_prints_and_stops():
    facts = load.declared(None, SimpleNamespace(**GEMMA))
    path = os.path.join(tempfile.mkdtemp(), "record.jsonl")
    os.environ["ENTAIL_RECORD"] = path
    try:
        out = io.StringIO()
        with redirect_stdout(out):
            load.enforce(load.attention("transformers", "eager", facts, policy=LOAD))     # a pass: silent
            load.enforce(load.attention("transformers", "sdpa", facts, policy=LOAD))      # resolved: one line
        assert out.getvalue().count("\n") == 1 and "[entail] resolved at load:transformers.attention" in out.getvalue()
        assert "changed: route to a backend measured to honour it (to eager)" in out.getvalue()
        try:
            with redirect_stdout(io.StringIO()):
                load.enforce(load.attention("transformers", "sdpa", facts, policy=Policy(mode="load", **STOPS,
                                                                                         on_mismatch="refuse")))
            raise AssertionError("should stop")
        except RoleError as e:
            assert "stops here" in str(e)
        lines = [json.loads(x) for x in open(path, encoding="utf-8")]
        assert [x["verdict"] for x in lines] == ["pass", "resolved", "refused"] and lines[1]["target"] == "eager"
    finally:
        os.environ.pop("ENTAIL_RECORD")


def test_at_load_puts_them_together():
    d = model_folder(GEMMA, {"model.embed_tokens.weight": EMBED})
    decisions = sites.at_load(d, "sglang", {"attention": "flashinfer", "tie": True}, LOAD)
    assert [(x.contract.boundary, x.verdict) for x in decisions] == [
        ("load:sglang.attention", Verdict.RESOLVED), ("load:sglang.loader", Verdict.PASS)]
    shutil.rmtree(d)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
