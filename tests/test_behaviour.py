"""Tests for behaviour.py: the first-call evidence of what a diffusion model predicts. Run: python tests/test_behaviour.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail.behaviour import behaves_like, first_call_cos, raw_output  # noqa: E402
from entail.facts import Prediction  # noqa: E402


def test_boundary_follows_the_measurements():
    for cos in (0.9997, 0.9998, 0.9999):  # eps checkpoints, t=999
        assert behaves_like(cos) == Prediction("eps")
    for cos in (0.0388, 0.0094, -0.0131):  # NoobAI-XL-Vpred, t=999
        assert behaves_like(cos) == Prediction("v")
    assert behaves_like(0.94) == Prediction("eps")  # M7: a model switching objective reads eps here; its file says v


def test_raw_output_inverts_k_diffusion_formulas():
    """calculate_denoised of EPS and V_PREDICTION (ComfyUI's comfy/model_sampling.py), undone."""
    import torch

    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 4, 8, 8, generator=g) * 5
    f = torch.randn(2, 4, 8, 8, generator=g)
    sigma = torch.tensor([14.6, 3.0])
    s = sigma.reshape(2, 1, 1, 1)
    assert torch.allclose(raw_output("eps", x, sigma, x - f * s), f, atol=1e-5)
    assert torch.allclose(raw_output("v", x, sigma, x / (s ** 2 + 1) - f * s / (s ** 2 + 1) ** 0.5), f, atol=1e-5)


def test_first_call_tells_the_two_apart():
    """At the noisiest step an eps model returns the noise; a v model's output is unrelated to it."""
    import torch

    g = torch.Generator().manual_seed(1)
    eps = torch.randn(2, 4, 64, 64, generator=g)
    sigma = torch.tensor([14.6, 14.6])
    x = eps * 14.6  # the first input of a txt2img: pure noise at sigma_max
    s = sigma.reshape(2, 1, 1, 1)
    v_out = -torch.randn(2, 4, 64, 64, generator=g)  # a v model near t=999: ~ -x0, independent of the noise
    assert behaves_like(first_call_cos("eps", x, sigma, x - eps * s)) == Prediction("eps")
    assert behaves_like(first_call_cos("eps", x, sigma, x - v_out * s)) == Prediction("v")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
