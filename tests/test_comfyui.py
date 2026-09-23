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


def test_probe_boundary_follows_the_measurements():
    from entail.adapters.comfyui import behaves_like

    for cos in (0.9997, 0.9998, 0.9999):  # eps checkpoints, t=999
        assert behaves_like(cos) == "eps"
    for cos in (0.0388, 0.0094, -0.0131):  # NoobAI-XL-Vpred, t=999
        assert behaves_like(cos) == "v_prediction"


def test_sampling_kind_only_covers_what_was_measured():
    from entail.adapters.comfyui import sampling_kind

    m = _fake_sampling_module()

    def make(*bases):
        return type("S", bases, {})()
    assert sampling_kind(make(m.ModelSamplingDiscrete, m.EPS), m) == "eps"
    assert sampling_kind(make(m.ModelSamplingDiscrete, m.V_PREDICTION), m) == "v_prediction"
    assert sampling_kind(make(m.ModelSamplingDiscreteEDM, m.EDM), m) is None  # EDM schedules: not measured
    assert sampling_kind(make(m.ModelSamplingDiscrete, m.X0), m) is None
    LCM = type("LCM", (m.EPS,), {})
    assert sampling_kind(make(m.ModelSamplingDiscrete, LCM), m) is None  # distilled
    assert sampling_kind(make(m.EPS), m) is None  # not a discrete schedule (flow models and the like)


def test_raw_output_inverts_comfyui_formulas():
    """calculate_denoised of EPS and V_PREDICTION (comfy/model_sampling.py), undone."""
    import torch

    from entail.adapters.comfyui import raw_output

    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 4, 8, 8, generator=g) * 5
    f = torch.randn(2, 4, 8, 8, generator=g)
    sigma = torch.tensor([14.6, 3.0])
    s = sigma.reshape(2, 1, 1, 1)
    eps_denoised = x - f * s
    v_denoised = x / (s ** 2 + 1) - f * s / (s ** 2 + 1) ** 0.5
    assert torch.allclose(raw_output("eps", x, sigma, eps_denoised), f, atol=1e-5)
    assert torch.allclose(raw_output("v_prediction", x, sigma, v_denoised), f, atol=1e-5)


def test_first_call_tells_the_two_apart():
    """At the noisiest step an eps model returns the noise; a v model's output is unrelated to it."""
    import torch

    from entail.adapters.comfyui import behaves_like, first_call_cos

    g = torch.Generator().manual_seed(1)
    eps = torch.randn(2, 4, 64, 64, generator=g)
    sigma = torch.tensor([14.6, 14.6])
    x = eps * 14.6  # the first input of a txt2img: pure noise at sigma_max
    s = sigma.reshape(2, 1, 1, 1)
    as_eps = x - eps * s                     # an eps model: predicts the noise
    v_out = -torch.randn(2, 4, 64, 64, generator=g)  # a v model near t=999: ~ -x0, independent of the noise
    as_v_read_as_eps = x - v_out * s         # what ComfyUI computes when it treats that model as eps
    assert behaves_like(first_call_cos("eps", x, sigma, as_eps)) == "eps"
    assert behaves_like(first_call_cos("eps", x, sigma, as_v_read_as_eps)) == "v_prediction"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
