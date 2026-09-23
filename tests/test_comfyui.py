"""Tests for the ComfyUI adapter's own parts: how it reads LoRA reach and what ComfyUI set up (no ComfyUI needed).
The checks it connects to are tested in test_coverage, test_declared, test_contract, test_behaviour, test_ownership.
Run: python tests/test_comfyui.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail.adapters.comfyui import lora_reach, lora_verdict, sampling_kind  # noqa: E402
from entail.facts import Base, Prediction  # noqa: E402

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


def test_matching_lora_is_quiet():
    reach = lora_reach(SDXL_LORA, SDXL_MODEL_MAP, SDXL_CLIP_MAP)
    assert (reach["model"].total, reach["model"].taken, reach["text"].total, reach["text"].taken) == (2, 2, 1, 1)
    assert lora_verdict(reach, applied=3, sides=["model", "text"], name="x", base=None, target="SDXL") is None


def test_model_only_loader_ignores_text_modules():
    """LoraLoaderModelOnly passes no text encoder: the text part is not a mismatch."""
    reach = lora_reach(SDXL_LORA, SDXL_MODEL_MAP, [])
    assert lora_verdict(reach, applied=2, sides=["model"], name="x", base=None, target="SDXL") is None


def test_other_base_model_is_a_violation():
    """The measured case: an Anima LoRA in an SDXL workflow."""
    reach = lora_reach(ANIMA_LORA, SDXL_MODEL_MAP, [])
    kind, msg = lora_verdict(reach, applied=0, sides=["model"], name="nagito\\nagito_anima_e10.safetensors",
                             base=Base("anima"), target="SDXL")
    assert kind == "violation"
    assert "cannot reach this model" in msg and "trained for anima" in msg and "SDXL" in msg and "2 modules" in msg


def test_text_only_lora_with_model_only_loader():
    reach = lora_reach(SDXL_LORA[5:], SDXL_MODEL_MAP, [])
    kind, msg = lora_verdict(reach, applied=0, sides=["model"], name="t", base=None, target="SDXL")
    assert kind == "violation" and "only carries text-encoder modules" in msg


def test_partial_is_said_not_stopped():
    reach = lora_reach(SDXL_LORA + ["lora_unet_some_block_the_model_lacks.lora_down.weight"], SDXL_MODEL_MAP, [])
    kind, msg = lora_verdict(reach, applied=2, sides=["model"], name="p", base=None, target="SDXL")
    assert kind == "partial" and "1 of 3 model modules" in msg


def _fake_sampling_module():
    """The class layout of comfy/model_sampling.py, without ComfyUI."""
    from types import SimpleNamespace

    class ModelSamplingDiscrete:
        pass

    class ModelSamplingDiscreteEDM(ModelSamplingDiscrete):
        pass

    class EPS:
        pass

    class V_PREDICTION(EPS):
        pass

    class EDM(V_PREDICTION):
        pass

    class X0(EPS):
        pass

    return SimpleNamespace(ModelSamplingDiscrete=ModelSamplingDiscrete, ModelSamplingDiscreteEDM=ModelSamplingDiscreteEDM,
                           EPS=EPS, V_PREDICTION=V_PREDICTION, EDM=EDM, X0=X0)


def test_sampling_kind_reads_what_comfyui_set_up():
    m = _fake_sampling_module()

    def make(*bases, zsnr=None):
        obj = type("S", bases, {})()
        if zsnr is not None:
            obj.zsnr = zsnr
        return obj
    assert sampling_kind(make(m.ModelSamplingDiscrete, m.EPS, zsnr=False), m) == Prediction("eps", False)
    assert sampling_kind(make(m.ModelSamplingDiscrete, m.V_PREDICTION, zsnr=True), m) == Prediction("v", True)
    assert sampling_kind(make(m.ModelSamplingDiscreteEDM, m.EDM), m) is None  # EDM schedules: not measured
    assert sampling_kind(make(m.ModelSamplingDiscrete, m.X0), m) is None
    LCM = type("LCM", (m.EPS,), {})
    assert sampling_kind(make(m.ModelSamplingDiscrete, LCM), m) is None  # distilled
    assert sampling_kind(make(m.EPS), m) is None  # not a discrete schedule (flow models and the like)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
