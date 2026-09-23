"""Tests for the ComfyUI LoRA check's counting and verdict (pure functions; ComfyUI is not needed).
Run: python tests/test_comfyui.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail.adapters.comfyui import coverage, module_of, verdict  # noqa: E402

# Shapes of real key names: kohya SDXL (with text-encoder part), LoCon, and an Anima (DiT) LoRA.
SDXL_LORA = ["lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_k.lora_down.weight",
             "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_k.lora_up.weight",
             "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_k.alpha",
             "lora_unet_output_blocks_1_0_in_layers_2.lora_mid.weight",
             "lora_unet_output_blocks_1_0_in_layers_2.lora_down.weight",
             "lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight",
             "lora_te1_text_model_encoder_layers_0_mlp_fc1.alpha"]
ANIMA_LORA = ["diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight",
              "diffusion_model.blocks.0.self_attn.q_proj.lora_B.weight",
              "diffusion_model.blocks.1.mlp.layer1.lora_A.weight"]
SDXL_MODEL_MAP = ["lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_k", "lora_unet_output_blocks_1_0_in_layers_2"]
SDXL_CLIP_MAP = ["lora_te1_text_model_encoder_layers_0_mlp_fc1"]


def test_module_names():
    assert module_of(SDXL_LORA[0]) == "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_k"
    assert module_of(SDXL_LORA[2]) == module_of(SDXL_LORA[0])
    assert module_of(SDXL_LORA[3]) == "lora_unet_output_blocks_1_0_in_layers_2"
    assert module_of(ANIMA_LORA[0]) == "diffusion_model.blocks.0.self_attn.q_proj"


def test_matching_lora_is_quiet():
    cov = coverage(SDXL_LORA, SDXL_MODEL_MAP, SDXL_CLIP_MAP)
    assert cov == {"model_total": 2, "model_matched": 2, "text_total": 1, "text_matched": 1}
    assert verdict(cov, applied=3, sides=["model", "text"], name="x", declared=None, target="SDXL") is None


def test_model_only_loader_ignores_text_modules():
    """LoraLoaderModelOnly passes no text encoder: the text part is not a mismatch."""
    cov = coverage(SDXL_LORA, SDXL_MODEL_MAP, [])
    assert verdict(cov, applied=2, sides=["model"], name="x", declared=None, target="SDXL") is None


def test_other_base_model_is_a_violation():
    """The measured case: an Anima LoRA in an SDXL workflow."""
    cov = coverage(ANIMA_LORA, SDXL_MODEL_MAP, [])
    kind, msg = verdict(cov, applied=0, sides=["model"], name="nagito\\nagito_anima_e10.safetensors",
                        declared="anima-preview/lora / anima", target="SDXL")
    assert kind == "violation"
    assert "cannot reach this model" in msg and "anima-preview/lora" in msg and "SDXL" in msg and "2 modules" in msg


def test_text_only_lora_with_model_only_loader():
    cov = coverage(SDXL_LORA[5:], SDXL_MODEL_MAP, [])
    kind, msg = verdict(cov, applied=0, sides=["model"], name="t", declared=None, target="SDXL")
    assert kind == "violation" and "only carries text-encoder modules" in msg


def test_partial_is_said_not_stopped():
    cov = coverage(SDXL_LORA + ["lora_unet_some_block_the_model_lacks.lora_down.weight"], SDXL_MODEL_MAP, [])
    kind, msg = verdict(cov, applied=2, sides=["model"], name="p", declared=None, target="SDXL")
    assert kind == "partial" and "1 of 3 model modules" in msg


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
