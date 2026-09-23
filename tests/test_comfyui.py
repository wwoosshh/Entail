"""Tests for the ComfyUI checks: LoRA reach, prediction type, sampling schedule (pure functions; no ComfyUI needed).
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


def _schedule_class():
    """A sampling class shaped like comfy.model_sampling.ModelSamplingDiscrete: set_sigmas registers the schedule."""
    import torch

    class Sampling(torch.nn.Module):
        def __init__(self, sigma_max):
            super().__init__()
            self.set_sigmas(torch.linspace(0.03, sigma_max, 1000))

        def set_sigmas(self, sigmas):
            self.register_buffer("sigmas", sigmas.float())
            self.register_buffer("log_sigmas", sigmas.log().float())

    return Sampling


def test_schedule_drift_sees_a_schedule_its_setter_did_not_register():
    import torch

    from entail.adapters.comfyui import _wrap_setters, schedule_drift

    Sampling = _schedule_class()
    wrapped = _wrap_setters([Sampling])
    try:
        ms = Sampling(14.6)
        assert schedule_drift(ms) == {}
        ms.register_buffer("sigmas", ms.sigmas.clone())  # a copy with the same values (what a device move does)
        assert schedule_drift(ms) == {}
        ms.register_buffer("sigmas", torch.linspace(0.03, 4518.8, 1000))  # another object's schedule written in
        drift = schedule_drift(ms)
        assert list(drift) == ["sigmas"] and float(drift["sigmas"][0][-1]) > 4518
        ms.set_sigmas(torch.linspace(0.03, 20.0, 1000))  # a setter call is a legitimate change: recorded anew
        assert schedule_drift(ms) == {}
    finally:
        for cls, name, fn in wrapped:
            setattr(cls, name, fn)


def _set_attr_buffer(obj, attr, value):  # comfy/utils.py
    obj, name = _resolve_attr(obj, attr)
    obj.register_buffer(name, value, persistent=name not in getattr(obj, "_non_persistent_buffers_set", set()))


def _resolve_attr(obj, attr):  # comfy/utils.py
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    return obj, attrs[-1]


def _dynamic_patcher_class():
    """The buffer handling of ComfyUI v0.34.1's ModelPatcherDynamic (load + restore_loaded_backups), nothing else."""

    class Dynamic:
        def __init__(self, model, backups):
            self.model, self.backup_buffers = model, backups  # clones share the backup dict

        def restore_loaded_backups(self):
            for key in list(self.backup_buffers.keys()):
                _set_attr_buffer(self.model, key, self.backup_buffers.pop(key))

        def load(self):
            self.restore_loaded_backups()
            for key, buf in self.model.named_buffers(recurse=True):
                if key not in self.backup_buffers:
                    self.backup_buffers[key] = buf
                _set_attr_buffer(self.model, key, buf.clone())  # stands for the copy on the GPU

    return Dynamic


def _node_then_plain(guarded, node_max=4518.8):
    import torch

    from entail.adapters.comfyui import _guard

    Sampling = _schedule_class()
    Dynamic = _dynamic_patcher_class()
    if guarded:
        _guard(Dynamic, _resolve_attr)
    model = torch.nn.Module()
    own, node = Sampling(14.6), Sampling(node_max)  # the checkpoint's schedule; a v_prediction+zsnr node's
    backups = {}
    model.model_sampling = node  # the run with the node: its object is put at 'model_sampling'
    Dynamic(model, backups).load()
    model.model_sampling = own  # the node is gone: ComfyUI puts the model's own object back
    Dynamic(model, backups).load()
    return float(model.model_sampling.sigmas[-1]), float(node.sigmas[-1])


def test_comfyui_loader_moves_a_node_schedule_into_the_model():
    """The defect as ComfyUI v0.34.1 has it: the node's schedule ends up in the model's own object."""
    assert _node_then_plain(guarded=False)[0] > 4518


def test_guard_keeps_each_schedule_with_its_object():
    from entail.adapters import _shared

    before = len(_shared.RESOLUTIONS)
    model_max, node_max = _node_then_plain(guarded=True)
    assert abs(model_max - 14.6) < 1e-4 and abs(node_max - 4518.8) < 1e-2
    assert any(r.get("where") == "sampling schedule" for r in _shared.RESOLUTIONS[before:])


def test_guard_says_nothing_when_the_schedules_are_the_same():
    """entail's own prediction-type switch puts an object with the same schedule at 'model_sampling': no change."""
    from entail.adapters import _shared

    before = len(_shared.RESOLUTIONS)
    model_max, _ = _node_then_plain(guarded=True, node_max=14.6)
    assert abs(model_max - 14.6) < 1e-4
    assert not any(r.get("where") == "sampling schedule" for r in _shared.RESOLUTIONS[before:])


def test_put_back_restores_the_registered_schedule():
    import torch

    from entail.adapters.comfyui import _put_back, _wrap_setters, schedule_drift

    Sampling = _schedule_class()
    wrapped = _wrap_setters([Sampling])
    try:
        ms = Sampling(14.6)
        ms.register_buffer("sigmas", torch.linspace(0.03, 4518.8, 1000))
        _put_back(ms, schedule_drift(ms))
        assert schedule_drift(ms) == {} and abs(float(ms.sigmas[-1]) - 14.6) < 1e-4
    finally:
        for cls, name, fn in wrapped:
            setattr(cls, name, fn)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
