"""Tests for the start-up check, including its exit codes. Run: python tests/test_preflight.py"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from entail.preflight import check_tie, main  # noqa: E402


def model_dir(**cfg):
    d = tempfile.mkdtemp(prefix="preflight_test_")
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return d


def test_exit_codes():
    capped = model_dir(attn_logit_softcapping=50.0, sliding_window=4096)
    plain = model_dir(tie_word_embeddings=True)
    assert main(["--model", capped, "--engine", "sglang", "--backend", "flashinfer"]) == 1
    assert main(["--model", capped, "--engine", "sglang", "--backend", "triton"]) == 0
    assert main(["--model", capped, "--engine", "transformers", "--backend", "sdpa"]) == 1
    assert main(["--model", capped, "--engine", "transformers", "--backend", "eager"]) == 0
    assert main(["--model", plain, "--engine", "sglang", "--backend", "flashinfer"]) == 0  # nothing declared
    assert main(["--model", capped, "--engine", "sglang", "--list"]) == 0  # listing never fails
    assert main(["--model", capped, "--engine", "sglang", "--backend", "no_such"]) == 2  # uncovered, not "ok"


def test_nested_text_config():
    d = model_dir(text_config={"attn_logit_softcapping": 30.0}, tie_word_embeddings=False)
    assert main(["--model", d, "--engine", "sglang", "--backend", "torch_native"]) == 1


def test_config_key_rule():
    """A renamed key whose value survives is fine; a key whose value is nowhere is a complaint."""
    from entail.preflight import _scalars, check_config_keys

    resolved = _scalars({"rope_parameters": {"rope_theta": 1000000, "rope_type": "default"}})
    assert ("int", 1000000) in resolved and ("str", "default") in resolved
    said, why = check_config_keys(model_dir(hidden_size=8, model_type="qwen3"))
    assert said is None or said == [], (said, why)  # nothing invented, or transformers missing


def llama(**extra):
    return model_dir(model_type="llama", hidden_size=64, intermediate_size=128, num_attention_heads=4,
                     num_hidden_layers=2, vocab_size=100, max_position_embeddings=256, rope_theta=500000.0, **extra)


def test_a_dict_transformers_keeps_under_another_name_is_not_lost():
    """Field test, entail#17: transformers 5.17 keeps rope_scaling as rope_parameters (adding rope_theta to it), and
    the rule that looked for scalars only printed a RoleError for Llama 3.2's rope_scaling; Phi-3.5's auto_map, which
    the Auto classes read, was one too. A dict the class really drops is still a loss."""
    from entail.preflight import check_config_keys

    rope = {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0, "original_max_position_embeddings": 8192,
            "rope_type": "llama3"}
    said, why = check_config_keys(llama(rope_scaling=rope, auto_map={"AutoConfig": "configuration_x.XConfig"}))
    if said is None:
        return  # transformers is not installed
    assert said == [] and "kept under another name: rope_scaling in rope_parameters" in why, (said, why)
    assert "auto_map=" in why and "trust_remote_code" in why, why
    said, why = check_config_keys(llama(my_table={"alpha": 12345.5, "rows": list(range(40))}))
    assert len(said) == 1 and said[0].startswith("my_table={'alpha': 12345.5") and "..." in said[0], said
    assert said[0].endswith("is not a field of LlamaConfig and its value is in none"), said


def test_the_output_says_what_the_model_declares_in_plain_labels():
    """entail#17: the RoPE the config declares is shown; the evidence is a short label for each property the model
    declares (no paths into the research workspace); a model that declares no attention property gets one line."""
    import io
    from contextlib import redirect_stdout

    out = io.StringIO()
    with redirect_stdout(out):
        assert main(["--model", model_dir(attn_logit_softcapping=50.0, rope_theta=10000.0), "--engine", "vllm",
                     "--list"]) == 0
    text = out.getvalue()
    assert "  rope: rope_type='default', theta=10000.0 (checked at load" in text, text
    rows = [ln for ln in text.splitlines() if "vllm:" in ln]
    assert len(rows) == 5 and all("softcap measured on vllm" in ln or "softcap read in vllm" in ln for ln in rows)
    assert not any("sliding window" in ln or "/" in ln or ".json" in ln or ".md" in ln for ln in rows), rows
    out = io.StringIO()
    with redirect_stdout(out):
        assert main(["--model", model_dir(tie_word_embeddings=True), "--engine", "vllm", "--list"]) == 0
        assert main(["--model", model_dir(tie_word_embeddings=True), "--engine", "vllm", "--backend",
                     "FLASH_ATTN"]) == 0
    text = out.getvalue()
    assert "vllm: the model declares neither softcap nor sliding window, so none of its 5 attention backends" in text
    assert "ok: the model declares neither softcap nor sliding window, so vllm backend 'FLASH_ATTN' has none" in text


def test_a_window_that_is_off_or_never_binds_is_not_a_requirement():
    """entail#17 (same output): preflight read sliding_window itself and said sglang drops Qwen2.5's window, which
    use_sliding_window false switches off, and Phi-3.5's, which never binds; the load check reads neither as a
    requirement, and preflight now reads them the same way."""
    import io
    from contextlib import redirect_stdout

    off = model_dir(sliding_window=32768, use_sliding_window=False, max_position_embeddings=32768)
    never = model_dir(sliding_window=262144, max_position_embeddings=131072)
    binds = model_dir(sliding_window=512, max_position_embeddings=32768)
    out = io.StringIO()
    with redirect_stdout(out):
        assert main(["--model", off, "--engine", "sglang", "--backend", "flex_attention"]) == 0
        assert main(["--model", never, "--engine", "sglang", "--backend", "flex_attention"]) == 0
        assert main(["--model", binds, "--engine", "sglang", "--backend", "flex_attention"]) == 1
    assert "note: config.json#sliding_window: sliding_window 262144 is not below max_position_embeddings" in \
        out.getvalue(), out.getvalue()


def test_tie_needs_a_checkpoint():
    said, why = check_tie(model_dir(tie_word_embeddings=True))
    assert said is None and "safetensors" in why, (said, why)


def test_tie_catches_the_case_07_mismatch():
    import torch
    from safetensors.torch import save_file

    d = model_dir(tie_word_embeddings=True)
    save_file({"model.embed_tokens.weight": torch.zeros(4, 4), "lm_head.weight": torch.ones(4, 4)},
              os.path.join(d, "model.safetensors"))
    said, _ = check_tie(d)
    assert said and "lm_head.weight" in said[0], said


def test_cli_exit_code():
    capped = model_dir(attn_logit_softcapping=50.0)
    r = subprocess.run([sys.executable, "-m", "entail.preflight", "--model", capped, "--engine", "sglang",
                        "--backend", "flashinfer"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 1, (r.returncode, r.stdout, r.stderr)
    assert "RoleError" in r.stdout


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
