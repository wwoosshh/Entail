"""Tests for declared.py: what an artifact says about itself. The headers are shaped like the real files measured in
issue_track/comfyui_field_test (NoobAI-XL-Vpred's marker keys, AstolfoCarmix-VPredXL's metadata, kohya LoRAs).
Run: python tests/test_declared.py"""
import json
import os
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import declared  # noqa: E402
from entail.facts import Base, Prediction  # noqa: E402

UNET = ["model.diffusion_model.input_blocks.0.0.weight", "model.diffusion_model.out.2.bias", "alphas_cumprod"]


def test_marker_keys_are_a_declaration():
    """NoobAI-XL-Vpred carries v_pred and ztsnr keys and no metadata."""
    d = declared.from_header(UNET + ["v_pred", "ztsnr"], {})
    assert d.get(Prediction) == Prediction("v", True) and "key 'v_pred'" in d.source(Prediction)
    assert not d.conflicts and not d.modules


def test_metadata_is_a_declaration():
    """AstolfoCarmix-VPredXL: no marker keys, but modelspec.prediction_type = v."""
    d = declared.from_header(UNET, {"modelspec.prediction_type": "v",
                                    "modelspec.architecture": "stable-diffusion-xl-v1-base"})
    assert d.get(Prediction) == Prediction("v", None) and d.source(Prediction) == "metadata modelspec.prediction_type"
    assert d.get(Base) == Base("sdxl")


def test_nothing_declared_is_empty():
    """Most checkpoints say nothing; then there is nothing to check against."""
    d = declared.from_header(UNET, {})
    assert not d and d.get(Prediction) is None


def test_lora_metadata_and_modules():
    keys = ["lora_unet_input_blocks_4_1_attn1_to_k.lora_down.weight",
            "lora_unet_input_blocks_4_1_attn1_to_k.lora_up.weight", "lora_te1_text_model_x.alpha"]
    meta = {"modelspec.prediction_type": "epsilon", "ss_zero_terminal_snr": "False",
            "ss_base_model_version": "sdxl_base_v1-0", "modelspec.architecture": "stable-diffusion-xl-v1-base/lora"}
    d = declared.from_header(keys, meta)
    assert d.get(Prediction) == Prediction("eps", False) and d.get(Base) == Base("sdxl")
    assert d.modules == {"lora_unet_input_blocks_4_1_attn1_to_k", "lora_te1_text_model_x"}
    assert declared.from_header(keys, {"ss_base_model_version": "anima"}).get(Base) == Base("anima")


def test_disagreeing_statements_are_recorded():
    d = declared.from_header(UNET, {"modelspec.prediction_type": "v", "ss_v_parameterization": "False"})
    assert d.get(Prediction) == Prediction("v") and len(d.conflicts) == 1
    assert "ss_v_parameterization" in d.conflicts[0]


def test_from_file_reads_only_the_header():
    header = {"a.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
              "__metadata__": {"modelspec.prediction_type": "v"}}
    raw = json.dumps(header).encode()
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "m.safetensors")
        with open(p, "wb") as f:
            f.write(struct.pack("<Q", len(raw)) + raw + b"\x00\x00")
        assert declared.from_file(p).get(Prediction) == Prediction("v")
        assert not declared.from_file(os.path.join(tmp, "m.ckpt"))  # a format without declarations


def test_hf_scheduler_config():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "scheduler"))
        json.dump({"prediction_type": "v_prediction", "rescale_betas_zero_snr": True},
                  open(os.path.join(tmp, "scheduler", "scheduler_config.json"), "w"))
        assert declared.from_hf_folder(tmp).get(Prediction) == Prediction("v", True)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
