"""Tests for the diffusers adapter's own parts, with stand-ins for a scheduler (no diffusers needed).
Run: python tests/test_diffusers_adapter.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail.adapters.diffusers_adapter import _given_modules, _switch, used_prediction  # noqa: E402
from entail.facts import Prediction  # noqa: E402


class _Scheduler:
    """What the adapter uses of a diffusers scheduler: .config and from_config(config, **overrides)."""

    def __init__(self, **cfg):
        self.config = cfg

    @classmethod
    def from_config(cls, config, **over):
        return cls(**{**config, **over})


class _Pipe:
    def __init__(self, scheduler):
        self.scheduler = scheduler


def test_used_prediction_reads_the_scheduler_config():
    assert used_prediction(_Scheduler(prediction_type="epsilon", rescale_betas_zero_snr=False)) == Prediction("eps", False)
    assert used_prediction(_Scheduler(prediction_type="v_prediction")) == Prediction("v")
    assert used_prediction(_Scheduler()) is None


def test_switch_rebuilds_the_scheduler_with_the_declared_type():
    pipe = _Pipe(_Scheduler(prediction_type="epsilon", rescale_betas_zero_snr=False, beta_end=0.012))
    _switch(pipe, Prediction("v", True))
    assert pipe.scheduler.config == {"prediction_type": "v_prediction", "rescale_betas_zero_snr": True, "beta_end": 0.012}
    pipe = _Pipe(_Scheduler(prediction_type="epsilon"))  # a scheduler without the zsnr option keeps its config
    _switch(pipe, Prediction("v", True))
    assert pipe.scheduler.config == {"prediction_type": "v_prediction"}


def test_given_modules_in_diffusers_naming():
    keys = ["unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.lora_A.weight",
            "unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.lora_B.weight",
            "text_encoder.text_model.encoder.layers.0.mlp.fc1.lora.down.weight"]
    assert _given_modules(keys) == {"unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k",
                                    "text_encoder.text_model.encoder.layers.0.mlp.fc1"}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
