"""Tests for the diffusers adapter v2 (ROADMAP M6.2) without diffusers: stand-ins give the classes and the behaviour the
adapter hooks - FromSingleFileMixin.from_single_file (the scheduler falls back to epsilon, as in diffusers 0.40),
DiffusionPipeline.from_pretrained and __setattr__, a LoRA loader mixin - and the rules are the core's. The shapes are
the M6 test problems: fd-m7 / market I04 (a file that declares v, sampled as eps), fd-vae (a VAE with another family's
scale put into a pipeline), a LoRA that reaches nothing. Run: python tests/test_diffusers_adapter.py"""
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
from entail import core, load  # noqa: E402
from entail.facts import LatentScale, Prediction  # noqa: E402


# --- stand-ins for what the adapter reads of diffusers 0.40 ----------------------------------------------------------

class Scheduler:
    def __init__(self, **cfg):
        self.config = dict(cfg)

    @classmethod
    def from_config(cls, config, **over):
        return cls(**{**config, **over})


class VAE:
    def __init__(self, **cfg):
        self.config = dict(cfg)

    def register_to_config(self, **kw):
        self.config.update(kw)


class Module:
    def __init__(self):
        self.lora_A = {}


class UNet:
    """named_modules() of a model with two LoRA-able layers."""

    def __init__(self):
        self.layers = {"down.0.to_q": Module(), "down.0.to_k": Module()}

    def named_modules(self):
        return list(self.layers.items())


class DiffusionPipeline:
    def __init__(self, scheduler=None, vae=None):
        self.scheduler, self.vae, self.unet = scheduler, vae, UNet()

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        if not os.path.isdir(path):   # a Hub id: diffusers downloads it; here, a pipeline with its defaults
            return cls(Scheduler(prediction_type="epsilon"), VAE(scaling_factor=0.13025))
        read = lambda *p: json.load(open(os.path.join(path, *p), encoding="utf-8"))  # noqa: E731
        return cls(scheduler=kwargs.get("scheduler") or Scheduler(**read("scheduler", "scheduler_config.json")),
                   vae=kwargs.get("vae") or VAE(**read("vae", "config.json")))

    def __setattr__(self, name, value):   # diffusers also registers a component in the pipeline's config
        object.__setattr__(self, name, value)

    # the LoRA loader mixin's part (StableDiffusionXLLoraLoaderMixin)
    def lora_state_dict(self, source, **kw):
        return dict(source), None

    def get_list_adapters(self):
        names = set().union(*(m.lora_A.keys() for m in self.unet.layers.values()))
        return {"unet": sorted(names)} if names else {}


class FromSingleFileMixin:
    @classmethod
    def from_single_file(cls, path, **kwargs):
        """As diffusers 0.40 builds a pipeline from one file: its scheduler falls back to epsilon."""
        return DiffusionPipeline(scheduler=kwargs.get("scheduler") or Scheduler(prediction_type="epsilon",
                                                                                rescale_betas_zero_snr=False),
                                 vae=VAE(scaling_factor=0.13025))


class StableDiffusionXLLoraLoaderMixin:
    def load_lora_weights(self, source, adapter_name=None, **kw):
        """Adds the adapter to the modules the LoRA names that the model has; silently skips the rest."""
        name = adapter_name or "default_0"
        for key in source:
            mod = key.split(".lora_A")[0].split(".lora_B")[0].removeprefix("unet.")
            if mod in self.unet.layers:
                self.unet.layers[mod].lora_A[name] = True


def _stand_ins():
    mods = {n: types.ModuleType(n) for n in ("diffusers.loaders.single_file", "diffusers.pipelines.pipeline_utils",
                                             "diffusers.loaders.lora_pipeline")}
    mods["diffusers.loaders.single_file"].FromSingleFileMixin = FromSingleFileMixin
    mods["diffusers.pipelines.pipeline_utils"].DiffusionPipeline = DiffusionPipeline
    mods["diffusers.loaders.lora_pipeline"].StableDiffusionXLLoraLoaderMixin = StableDiffusionXLLoraLoaderMixin
    return mods


sys.modules.update(_stand_ins())
from entail.adapters import diffusers_adapter as da  # noqa: E402

assert da.install() == 1 and da.install_pipeline() == 2 and da.install_lora() == 1


class SDXLPipeline(DiffusionPipeline, FromSingleFileMixin, StableDiffusionXLLoraLoaderMixin):
    pass


def checkpoint(meta, keys=("model.diffusion_model.input_blocks.0.0.weight",)):
    header = {k: {"dtype": "F32", "shape": [1], "data_offsets": [4 * i, 4 * i + 4]} for i, k in enumerate(keys)}
    header["__metadata__"] = meta
    h = json.dumps(header).encode()
    p = os.path.join(tempfile.mkdtemp(), "model.safetensors")
    with open(p, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"\0" * (4 * len(keys)))
    return p


def folder(scheduler, vae):
    d = tempfile.mkdtemp()
    for sub, cfg in (("scheduler", scheduler), ("vae", vae)):
        os.makedirs(os.path.join(d, sub))
        name = "scheduler_config.json" if sub == "scheduler" else "config.json"
        with open(os.path.join(d, sub, name), "w", encoding="utf-8") as f:
            json.dump(cfg, f)
    with open(os.path.join(d, "model_index.json"), "w", encoding="utf-8") as f:
        json.dump({"_class_name": "StableDiffusionXLPipeline"}, f)
    return d


def run(fn):
    printed = io.StringIO()
    core.set_mode("load")
    try:
        with redirect_stdout(printed):
            out = fn()
    finally:
        core.set_mode("off")
    return out, printed.getvalue()


# --- read_choice and handles ----------------------------------------------------------------------------------------

def test_read_choice():
    assert da.read_choice("prediction", Scheduler(prediction_type="epsilon", rescale_betas_zero_snr=False)) == \
        Prediction("eps", False)
    assert da.read_choice("prediction", Scheduler(prediction_type="v_prediction")) == Prediction("v")
    assert da.read_choice("prediction", Scheduler()) is None, "a flow-matching scheduler names no prediction type"
    assert da.read_choice("latent_scale", VAE(scaling_factor=0.13025)) == LatentScale(0.13025)
    assert da.read_choice("latent_scale", VAE(scaling_factor=1.5305, shift_factor=0.0609)) == LatentScale(1.5305, 0.0609)
    keys = ["unet.down_blocks.1.attn1.to_k.lora_A.weight", "unet.down_blocks.1.attn1.to_k.lora_B.weight",
            "text_encoder.text_model.encoder.layers.0.mlp.fc1.lora.down.weight"]
    assert da.read_choice("lora_given", keys) == {"unet.down_blocks.1.attn1.to_k",
                                                  "text_encoder.text_model.encoder.layers.0.mlp.fc1"}


def test_handles_rebuild_the_scheduler_and_set_the_vae_scale():
    pipe = DiffusionPipeline(Scheduler(prediction_type="epsilon", rescale_betas_zero_snr=False, beta_end=0.012),
                             VAE(scaling_factor=0.18215))
    h = da.handles(pipe)
    h["switch_prediction"](Prediction("v", True))
    assert pipe.scheduler.config == {"prediction_type": "v_prediction", "rescale_betas_zero_snr": True, "beta_end": 0.012}
    pipe.scheduler = Scheduler(prediction_type="epsilon")   # a scheduler without the zsnr option keeps its config
    h["switch_prediction"](Prediction("v", True))
    assert pipe.scheduler.config == {"prediction_type": "v_prediction"}
    h["set_latent_scale"](LatentScale(0.13025))
    assert pipe.vae.config == {"scaling_factor": 0.13025}


# --- the hooks -------------------------------------------------------------------------------------------------------

def test_a_file_that_declares_v_is_no_longer_sampled_as_eps():
    path = checkpoint({}, keys=("model.diffusion_model.input_blocks.0.0.weight", "v_pred", "ztsnr"))  # I04's markers
    pipe, printed = run(lambda: SDXLPipeline.from_single_file(path))
    assert pipe.scheduler.config["prediction_type"] == "v_prediction"
    assert pipe.scheduler.config["rescale_betas_zero_snr"] is True
    assert "resolved at load:diffusers.prediction" in printed
    assert "unknown at load:diffusers.latent_scale" in printed, "a single file states no latent scale"
    off = SDXLPipeline.from_single_file(path)   # entail off: what diffusers does by itself
    assert off.scheduler.config["prediction_type"] == "epsilon"


def test_a_scheduler_the_caller_passed_is_reported_not_replaced():
    path = checkpoint({"modelspec.prediction_type": "v"})   # fd-m7's declaration
    mine = Scheduler(prediction_type="epsilon", rescale_betas_zero_snr=False)
    pipe, printed = run(lambda: SDXLPipeline.from_single_file(path, scheduler=mine))
    assert pipe.scheduler is mine and "broken at load:diffusers.prediction" in printed and "not overridden" in printed
    os.environ["ENTAIL_ON_BROKEN"] = "stop"
    try:
        run(lambda: SDXLPipeline.from_single_file(path, scheduler=mine))
        raise AssertionError("the strict policy stops")
    except core.RoleError as e:
        assert "refused at load:diffusers.prediction" in str(e)
    finally:
        os.environ.pop("ENTAIL_ON_BROKEN", None)


def test_a_vae_with_another_familys_scale_gets_the_folders():
    d = folder({"prediction_type": "epsilon"}, {"scaling_factor": 0.13025})
    pipe, printed = run(lambda: SDXLPipeline.from_pretrained(d))
    assert printed == "", "the folder's own scheduler and VAE agree with it: passes are not printed"
    vae = VAE(scaling_factor=0.18215)   # fd-vae: a VAE file diffusers took for Stable Diffusion 1.5's

    def put_in():
        pipe.vae = vae

    _, printed = run(put_in)
    assert vae.config["scaling_factor"] == 0.13025 and "resolved at load:diffusers.latent_scale" in printed
    other = DiffusionPipeline(Scheduler(prediction_type="epsilon"), VAE(scaling_factor=0.13025))
    _, printed = run(lambda: setattr(other, "vae", VAE(scaling_factor=0.18215)))
    assert printed == "", "a pipeline entail did not see being built has no known declaration: nothing decided"


def test_a_pipeline_from_the_hub_is_reported_as_not_checked():
    _, printed = run(lambda: SDXLPipeline.from_pretrained("someone/some-model"))
    assert "unknown at load:diffusers.pipeline" in printed and "not a local folder" in printed


def test_a_lora_that_reaches_nothing_is_reported():
    pipe = SDXLPipeline(Scheduler(prediction_type="epsilon"), VAE(scaling_factor=0.13025))
    good = {"unet.down.0.to_q.lora_A.weight": 0, "unet.down.0.to_q.lora_B.weight": 0}
    _, printed = run(lambda: pipe.load_lora_weights(good, adapter_name="a"))
    assert printed == "", "every module reached"
    other = {"transformer.blocks.0.attn.q.lora_A.weight": 0, "transformer.blocks.1.attn.q.lora_A.weight": 0}
    _, printed = run(lambda: pipe.load_lora_weights(other, adapter_name="b"))
    assert "broken at load:diffusers.lora" in printed and "taken=0" in printed and "reported, not stopped" in printed


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
