"""Tests for the readers and read_all (ROADMAP M2.1, M2.2): each reader on the shapes real artifacts have, what it
refuses to represent, and that a failing reader never breaks the caller. Run: python tests/test_sources.py"""
import json
import os
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import gguf, sources  # noqa: E402
from entail.facts import (PREDICTION_KINDS, ROPE_TYPES, VOCABULARY, Certainty, LatentScale, Layout,  # noqa: E402
                          ModelProps, Prediction, Rotary, Template)
from entail.readers import ALIASES, sha256_text  # noqa: E402


def folder(files):
    d = tempfile.mkdtemp()
    for rel, content in files.items():
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content if isinstance(content, str) else json.dumps(content))
    return d


def safetensors(keys, metadata=None):
    header = {k: {"dtype": "F32", "shape": [1], "data_offsets": [4 * i, 4 * i + 4]} for i, k in enumerate(keys)}
    if metadata is not None:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    fd, p = tempfile.mkstemp(suffix=".safetensors")
    with os.fdopen(fd, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw + b"\0" * (4 * len(keys)))
    return p


def facts_of(path):
    r = sources.read_all(path)
    return {(f.name, f.value) for f in r.facts}, r


def test_alias_table_is_about_the_vocabulary():
    for section in ALIASES:
        assert section in ("_about", "values") or section in VOCABULARY, section
    assert set(ALIASES["values"]["prediction_kind"].values()) <= PREDICTION_KINDS
    assert set(ALIASES["values"]["rope_type"].values()) <= ROPE_TYPES


def test_hf_config_gemma2_qwen3_llama32():
    got, r = facts_of(folder({"config.json": {"attn_logit_softcapping": 50.0, "sliding_window": 4096,
                                              "rope_theta": 10000.0, "tie_word_embeddings": True}}))
    assert got == {("ModelProps", ModelProps(50.0, 4096, True)), ("Rotary", Rotary("default", theta=10000.0))}, got
    [props] = [f for f in r.facts if f.name == "ModelProps"]
    assert props.source.where.endswith("config.json#attn_logit_softcapping,sliding_window,tie_word_embeddings")
    assert props.certainty is Certainty.DECLARED and props.source.kind == "config"
    # Qwen3: the window is switched off, so it is no requirement; rope_scaling null is fine
    got, _ = facts_of(folder({"config.json": {"sliding_window": 32768, "use_sliding_window": False,
                                              "rope_theta": 1000000, "rope_scaling": None, "tie_word_embeddings": True}}))
    assert got == {("ModelProps", ModelProps(tie_word_embeddings=True)), ("Rotary", Rotary(theta=1000000))}, got
    # Llama 3.2: llama3 scaling; the keys v1 cannot carry are named, not dropped
    got, r = facts_of(folder({"config.json": {"rope_theta": 500000.0, "rope_scaling": {
        "factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0, "original_max_position_embeddings": 8192,
        "rope_type": "llama3"}}}))
    assert got == {("Rotary", Rotary("llama3", 500000.0, 32.0, 8192))}, got
    assert any("RoPE keys ['high_freq_factor', 'low_freq_factor'] are not in vocabulary v1" in p for p in r.problems)


def test_hf_config_rope_parameters_and_the_old_names():
    d = folder({"config.json": {"rope_parameters": {"rope_type": "yarn", "rope_theta": 1000000.0, "factor": 4.0,
                                                    "original_max_position_embeddings": 32768},
                                "rope_theta": 10000.0}})
    r = sources.read_all(d)
    rot = [f for f in r.facts if f.name == "Rotary"]
    assert {f.value for f in rot} == {Rotary("yarn", 1000000.0, 4.0, 32768), Rotary(theta=10000.0)}
    chosen, conflicts = sources.merge(r.facts)
    assert len(conflicts) == 1 and conflicts[0].name == "Rotary"   # the file says two things: recorded, not hidden
    _, r = facts_of(folder({"config.json": {"rope_parameters": {"full_attention": {"rope_theta": 1e6},
                                                                 "sliding_attention": {"rope_theta": 1e4}}}}))
    assert not [f for f in r.facts if f.name == "Rotary"]
    assert any("RoPE set per layer type (['full_attention', 'sliding_attention'])" in p for p in r.problems)
    _, r = facts_of(folder({"config.json": {"rope_scaling": {"rope_type": "mrope"}, "rope_theta": 1e6}}))
    assert any("rope type 'mrope' is not in vocabulary v1" in p for p in r.problems), r.problems


def test_hf_config_nested_text_config_and_bad_values():
    _, r = facts_of(folder({"config.json": {"text_config": {"attn_logit_softcapping": 30.0}}}))
    [props] = r.facts
    assert props.value == ModelProps(softcap=30.0) and props.source.where.endswith("#text_config.attn_logit_softcapping")
    got, r = facts_of(folder({"config.json": {"attn_logit_softcapping": 50.0, "sliding_window": "4096"}}))
    assert got == {("ModelProps", ModelProps(softcap=50.0))}, got   # a bad window does not take the softcap down
    assert any("#sliding_window: ModelProps.sliding_window: expected an int, got '4096'" in p for p in r.problems)


def test_hf_config_quantization_layouts():
    cases = [
        ({"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128], "scale_fmt": "ue8m0"},
         Layout("fp8_block", dtype="float8_e4m3fn", block=(128, 128), scale_format="ue8m0"), None),
        ({"quant_method": "awq", "bits": 4, "group_size": 128}, Layout("int4_packed", block=(128,)), None),
        ({"quant_method": "gptq", "bits": 4, "group_size": -1}, Layout("int4_packed"), None),
        ({"quant_method": "fp8", "fmt": "e4m3"}, None, "per-tensor fp8 (no weight_block_size) is not in vocabulary v1"),
        ({"quant_method": "gptq", "bits": 8, "group_size": 128}, None, "gptq with 8 bits is not in vocabulary v1"),
        ({"quant_method": "bitsandbytes"}, None, "quantization method 'bitsandbytes' is not in vocabulary v1"),
    ]
    for qc, layout, problem in cases:
        got, r = facts_of(folder({"config.json": {"quantization_config": qc}}))
        assert got == ({("Layout", layout)} if layout else set()), (qc, got)
        assert (problem is None) == (not r.problems) and (problem is None or problem in r.problems[0]), r.problems


def test_hf_template():
    tpl = "{% for m in messages %}{{ m.content }}{% endfor %}"
    got, _ = facts_of(folder({"tokenizer_config.json": {"chat_template": tpl}}))
    assert got == {("Template", Template(sha256_text(tpl)))}
    got, r = facts_of(folder({"tokenizer_config.json": {"chat_template": [{"name": "default", "template": tpl},
                                                                          {"name": "tool_use", "template": "x"}]}}))
    assert got == {("Template", Template(sha256_text(tpl)))}
    assert "named templates ['tool_use'] besides 'default'" in r.problems[0]
    _, r = facts_of(folder({"tokenizer_config.json": {"chat_template": [{"name": "rag", "template": "x"}]}}))
    assert any("without a 'default' one" in p for p in r.problems) and not r.facts
    r = sources.read_all(folder({"tokenizer_config.json": {"chat_template": tpl}, "chat_template.jinja": "other"}))
    _, conflicts = sources.merge(r.facts)
    assert len(r.facts) == 2 and len(conflicts) == 1   # two templates for one model: a disagreement to record


def test_diffusers_folder():
    got, _ = facts_of(folder({"scheduler/scheduler_config.json": {"prediction_type": "v_prediction",
                                                                  "rescale_betas_zero_snr": True},
                              "vae/config.json": {"scaling_factor": 0.13025}}))
    assert got == {("Prediction", Prediction("v", True)), ("LatentScale", LatentScale(0.13025))}, got
    got, _ = facts_of(folder({"scheduler/scheduler_config.json": {"_class_name": "FlowMatchEulerDiscreteScheduler"},
                              "vae/config.json": {"scaling_factor": 1.5305, "shift_factor": 0.0609}}))
    assert got == {("Prediction", Prediction("flow")), ("LatentScale", LatentScale(1.5305, 0.0609))}, got
    _, r = facts_of(folder({"scheduler/scheduler_config.json": {"prediction_type": "velocity"}}))
    assert not r.facts and "prediction type 'velocity' is not in vocabulary v1" in r.problems[0]


def test_safetensors_header():
    got, r = facts_of(safetensors(["model.diffusion_model.x"], {"modelspec.prediction_type": "v"}))
    assert got == {("Prediction", Prediction("v"))} and r.facts[0].source.kind == "file"
    assert r.facts[0].source.where.endswith("#__metadata__.modelspec.prediction_type")
    got, r = facts_of(safetensors(["model.diffusion_model.x", "v_pred", "ztsnr"]))
    assert got == {("Prediction", Prediction("v", True))} and "(zsnr from key ztsnr)" in r.facts[0].source.where
    r = sources.read_all(safetensors(["v_pred"], {"ss_v_parameterization": "False"}))
    _, conflicts = sources.merge(r.facts)
    assert len(r.facts) == 2 and len(conflicts) == 1   # the file contradicts itself: both statements are kept
    _, r = facts_of(safetensors(["x"], {"modelspec.prediction_type": "velocity"}))
    assert not r.facts and "prediction type 'velocity' is not in vocabulary v1" in r.problems[0]
    got, r = facts_of(safetensors(["lora_unet_a.lora_down.weight"], {"ss_base_model_version": "sdxl_base_v1-0"}))
    assert got == set() and not r.problems
    fd, bad = tempfile.mkstemp(suffix=".safetensors")
    with os.fdopen(fd, "wb") as f:
        f.write(struct.pack("<Q", 10 ** 9) + b"{}")
    r = sources.read_all(bad)
    assert not r.facts and "safetensors" in r.problems[0] and "does not fit the file" in r.problems[0]


def test_gguf_header():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "m.gguf")
    gguf.write_minimal(p, {"general.architecture": "gemma2", "gemma2.attn_logit_softcapping": 50.0,
                           "gemma2.attention.sliding_window": 4096, "gemma2.rope.freq_base": 10000.0,
                           "tokenizer.chat_template": "tpl"})
    got, r = facts_of(p)
    assert got == {("ModelProps", ModelProps(50.0, 4096)), ("Rotary", Rotary(theta=10000.0)),
                   ("Template", Template(sha256_text("tpl")))}, (got, r.problems)
    assert all(f.source.kind == "file" for f in r.facts)
    gguf.write_minimal(p, {"general.architecture": "llama", "llama.rope.freq_base": 500000.0,
                           "llama.rope.scaling.type": "yarn", "llama.rope.scaling.factor": 4.0,
                           "llama.rope.scaling.original_context_length": 8192})
    got, _ = facts_of(p)
    assert got == {("Rotary", Rotary("yarn", 500000.0, 4.0, 8192))}, got
    with open(p, "wb") as f:
        f.write(b"GGML" + b"\0" * 20)
    assert "not a GGUF file (bad magic)" in sources.read_all(p).problems[0]
    gguf.write_minimal(p, {"general.architecture": "llama"})
    data = open(p, "rb").read()
    with open(p, "wb") as f:
        f.write(data[:-3])
    assert "truncated GGUF header" in sources.read_all(p).problems[0]


def test_a_failing_reader_never_breaks_the_caller():
    d = folder({"config.json": "{not json", "tokenizer_config.json": {"chat_template": "t"}})
    r = sources.read_all(d)
    assert {(f.name, f.value) for f in r.facts} == {("Template", Template(sha256_text("t")))}
    assert r.problems[0].startswith("hf_config: ") and "JSONDecodeError" in r.problems[0]
    assert sources.read_all(os.path.join(d, "missing.safetensors")).facts == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
