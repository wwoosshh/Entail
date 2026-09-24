"""Tests for the ComfyUI adapter v2 (ROADMAP M6.2) without ComfyUI: stand-ins give what the adapter hooks and reads of
ComfyUI 0.34.1 - comfy.sd's loaders and load_lora_for_models, comfy.lora's key maps, comfy.sample, the class layout of
comfy.model_sampling, a model patcher with clones and object patches - and the rules are the core's. The shapes are
the M6 test problems measured there: fd-m7 (the file declares v, ComfyUI samples eps), a node that contradicts the
file, fd-lora (a LoRA made for another base model). Run: python tests/test_comfyui.py"""
import io
import json
import os
import struct
import sys
import tempfile
import types
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
for k in ("ENTAIL_ON_BROKEN", "ENTAIL_UNKNOWN", "ENTAIL_POLICY", "ENTAIL_FACT_POLICY", "ENTAIL_MANIFESTS"):
    os.environ.pop(k, None)
from entail import core  # noqa: E402
from entail.facts import LatentScale, Prediction  # noqa: E402


# --- stand-ins: the class layout of comfy/model_sampling.py and what the adapter uses of the rest ---------------------

ms = types.ModuleType("comfy.model_sampling")


class EPS:
    pass


class V_PREDICTION(EPS):
    pass


class EDM(V_PREDICTION):
    pass


class X0(EPS):
    pass


class CONST:
    pass


class ModelSamplingDiscrete:
    def __init__(self, model_config=None, zsnr=None):
        self.zsnr = bool(zsnr)


class ModelSamplingDiscreteEDM(ModelSamplingDiscrete):
    pass


class ModelSamplingDiscreteFlow:
    def __init__(self, model_config=None):
        pass


for c in (EPS, V_PREDICTION, EDM, X0, CONST, ModelSamplingDiscrete, ModelSamplingDiscreteEDM,
          ModelSamplingDiscreteFlow):
    setattr(ms, c.__name__, c)


def sampling(pred, zsnr=False, schedule=ModelSamplingDiscrete):
    """What ComfyUI's model_sampling() builds: the schedule class mixed with the prediction class."""
    return type("ModelSampling", (schedule, pred), {})(None, zsnr) if schedule is not ModelSamplingDiscreteFlow \
        else type("ModelSampling", (schedule, pred), {})(None)


class SDXLFormat:
    scale_factor = 0.13025


class SDXL:   # a model_config class: its name is what the adapter reports
    pass


class BaseModel:
    def __init__(self, pred=EPS):
        self.model_config = SDXL()
        self.model_sampling = sampling(pred)
        self.latent_format = SDXLFormat()


class Patcher:
    """A ModelPatcher as far as the adapter uses it: model, object_patches, clone(), get_model_object()."""

    def __init__(self, model, patches=None):
        self.model, self.object_patches = model, dict(patches or {})

    def clone(self):
        return Patcher(self.model, self.object_patches)

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj

    def get_model_object(self, name):
        return self.object_patches.get(name, getattr(self.model, name))


sd = types.ModuleType("comfy.sd")


def load_checkpoint_guess_config(ckpt_path, output_vae=True):
    return sd.load_state_dict_guess_config({"model.diffusion_model.x": 0}, metadata={})


def load_state_dict_guess_config(state_dict, output_vae=True, metadata=None):
    return (Patcher(BaseModel()), None, None, None)


def load_diffusion_model(unet_path, model_options={}):
    return sd.load_diffusion_model_state_dict({}, metadata={})


def load_diffusion_model_state_dict(state_dict, model_options={}, metadata=None):
    return Patcher(BaseModel())


def load_lora_for_models(model, clip, lora, strength_model, strength_clip, lora_metadata=None):
    return model, clip


for f in (load_checkpoint_guess_config, load_state_dict_guess_config, load_diffusion_model,
          load_diffusion_model_state_dict, load_lora_for_models):
    setattr(sd, f.__name__, f)

lora = types.ModuleType("comfy.lora")
lora.model_lora_keys_unet = lambda model, key_map: {**key_map, "lora_unet_a": "w.a", "lora_unet_b": "w.b"}
lora.model_lora_keys_clip = lambda model, key_map: {**key_map, "lora_te1_c": "w.c"}
convert = types.ModuleType("comfy.lora_convert")
convert.convert_lora = lambda d: d
sample = types.ModuleType("comfy.sample")
sample.sample = lambda model, noise, steps, cfg, sigmas=None: model          # returns the patcher it sampled with
sample.sample_custom = lambda model, noise, cfg, sampler, sigmas: model
sys.modules.update({"comfy.model_sampling": ms, "comfy.sd": sd, "comfy.lora": lora, "comfy.lora_convert": convert,
                    "comfy.sample": sample})
from entail.adapters import comfyui  # noqa: E402

assert comfyui.install() == 5 and comfyui.install_sampling() == 2


def checkpoint(meta, keys=("model.diffusion_model.input_blocks.0.0.weight",)):
    header = {k: {"dtype": "F32", "shape": [1], "data_offsets": [4 * i, 4 * i + 4]} for i, k in enumerate(keys)}
    header["__metadata__"] = meta
    h = json.dumps(header).encode()
    p = os.path.join(tempfile.mkdtemp(), "model.safetensors")
    with open(p, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"\0" * (4 * len(keys)))
    return p


def run(fn):
    printed = io.StringIO()
    core.set_mode("load")
    try:
        with redirect_stdout(printed):
            out = fn()
    finally:
        core.set_mode("off")
    return out, printed.getvalue()


def loaded(meta):
    patcher, printed = run(lambda: sd.load_checkpoint_guess_config(checkpoint(meta))[0])
    assert printed == "", printed
    return patcher


# --- read_choice ------------------------------------------------------------------------------------------------------

def test_read_choice():
    assert comfyui.read_choice("prediction", sampling(EPS), ms) == Prediction("eps", False)
    assert comfyui.read_choice("prediction", sampling(V_PREDICTION, zsnr=True), ms) == Prediction("v", True)
    assert comfyui.read_choice("prediction", sampling(EDM, schedule=ModelSamplingDiscreteEDM), ms) == \
        Prediction("edm", False)
    assert comfyui.read_choice("prediction", sampling(CONST, schedule=ModelSamplingDiscreteFlow), ms) == \
        Prediction("flow")
    assert comfyui.read_choice("discrete", sampling(V_PREDICTION), ms)
    assert not comfyui.read_choice("discrete", sampling(X0), ms)
    assert not comfyui.read_choice("discrete", sampling(CONST, schedule=ModelSamplingDiscreteFlow), ms)
    assert comfyui.read_choice("latent_scale", SDXLFormat()) == LatentScale(0.13025)
    sd3 = types.SimpleNamespace(scale_factor=1.5305, shift_factor=0.0609)
    assert comfyui.read_choice("latent_scale", sd3) == LatentScale(1.5305, 0.0609)
    playground = types.SimpleNamespace(scale_factor=0.5, latents_mean=[0.1], latents_std=[0.2])
    assert comfyui.read_choice("latent_scale", playground) is None, "a per-channel mean and std are not a scale"
    keys = ["lora_unet_a.lora_down.weight", "lora_unet_z.lora_down.weight", "lora_te1_c.lora_down.weight"]
    given, taken, carried = comfyui.read_choice("lora", keys, ["lora_unet_a"], None)
    assert given == {"lora_unet_a", "lora_unet_z"} and taken == {"lora_unet_a"} and len(carried) == 3, \
        "the text-encoder part is not given when the LoRA is applied to the model only"


# --- sampling ---------------------------------------------------------------------------------------------------------

def test_a_file_that_declares_v_is_sampled_as_v_and_said_once():
    model = loaded({"modelspec.prediction_type": "v"})   # fd-m7: ComfyUI reads only the v_pred key, and sets up eps
    used, printed = run(lambda: sample.sample(model, None, 20, 7.0))
    ms_used = used.get_model_object("model_sampling")
    assert used is not model and isinstance(ms_used, V_PREDICTION) and ms_used.zsnr is False, \
        "switched to v, keeping the zsnr the file does not state"
    assert printed.count("resolved at load:comfyui.prediction") == 1
    assert "unknown at load:comfyui.latent_scale" in printed, "a single file states no latent scale"
    used, printed = run(lambda: sample.sample(model, None, 20, 7.0))
    assert isinstance(used.get_model_object("model_sampling"), V_PREDICTION) and printed == "", \
        "the same decision for the same model is carried out again but recorded once"


def test_a_node_that_contradicts_the_file_is_reported_not_overridden():
    model = loaded({"modelspec.prediction_type": "v"})
    node = model.clone()
    node.add_object_patch("model_sampling", sampling(EPS))   # ModelSamplingDiscrete(eps): the user's choice
    used, printed = run(lambda: sample.sample(node, None, 20, 7.0))
    assert used is node and "broken at load:comfyui.prediction" in printed and "not overridden" in printed
    os.environ["ENTAIL_ON_BROKEN"] = "stop"
    try:
        run(lambda: sample.sample(node, None, 20, 7.0))
        raise AssertionError("the strict policy stops before sampling")
    except core.RoleError as e:
        assert "refused at load:comfyui.prediction" in str(e)
    finally:
        os.environ.pop("ENTAIL_ON_BROKEN", None)


def test_sigmas_computed_before_sampling_cannot_be_switched():
    model = loaded({"modelspec.prediction_type": "v"})
    used, printed = run(lambda: sample.sample_custom(model, None, 7.0, None, [1.0, 0.0]))
    assert used is model and "broken at load:comfyui.prediction" in printed and "sigmas were computed" in printed


def test_a_model_that_declares_nothing_is_unknown_and_left_as_it_is():
    model = loaded({})
    used, printed = run(lambda: sample.sample(model, None, 20, 7.0))
    assert used is model and "unknown at load:comfyui.prediction" in printed and "nothing declares it" in printed
    custom = sd.load_state_dict_guess_config({"v_pred": 0}, metadata={})[0]   # built before entail looked
    used, printed = run(lambda: sample.sample(custom, None, 20, 7.0))
    assert used is custom and "unknown at load:comfyui.prediction" in printed


def test_a_state_dict_in_hand_declares_like_the_file():
    patcher, _ = run(lambda: sd.load_state_dict_guess_config({"model.diffusion_model.x": 0, "v_pred": 0, "ztsnr": 0},
                                                             metadata={}))
    used, printed = run(lambda: sample.sample(patcher[0], None, 20, 7.0))
    assert used.get_model_object("model_sampling").zsnr is True and "resolved" in printed


# --- LoRA --------------------------------------------------------------------------------------------------------------

def test_a_lora_is_checked_against_the_parts_it_is_applied_to():
    model = loaded({})
    clip = types.SimpleNamespace(cond_stage_model=None)
    good = {"lora_unet_a.lora_down.weight": 0, "lora_te1_c.lora_down.weight": 0}
    _, printed = run(lambda: sd.load_lora_for_models(model, clip, good, 1.0, 1.0))
    assert printed == ""
    other = {"diffusion_model.blocks.0.q.lora_A.weight": 0, "diffusion_model.blocks.1.q.lora_A.weight": 0}
    _, printed = run(lambda: sd.load_lora_for_models(model, None, other, 0.9, 0,
                                                     {"ss_base_model_version": "anima"}))
    assert "broken at load:comfyui.lora" in printed and "taken=0" in printed and "anima" in printed
    text_only = {"lora_te1_c.lora_down.weight": 0}   # a text-encoder LoRA given to the model alone
    _, printed = run(lambda: sd.load_lora_for_models(model, None, text_only, 1.0, 0))
    assert "broken at load:comfyui.lora" in printed and "carries nothing for the parts it was applied to" in printed
    _, printed = run(lambda: sd.load_lora_for_models(model, clip, good, 0, 0))
    assert printed == "", "a LoRA at strength 0 is given to nothing"


def test_off_changes_nothing_and_uninstall_restores():
    model = sd.load_checkpoint_guess_config(checkpoint({"modelspec.prediction_type": "v"}))[0]
    assert sample.sample(model, None, 20, 7.0) is model, "entail off: ComfyUI's own choice"
    assert comfyui.uninstall() == 7 and sample.sample.__name__ == "<lambda>"
    assert comfyui.install() == 5 and comfyui.install_sampling() == 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
