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
