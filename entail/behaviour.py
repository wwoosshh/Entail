"""What a diffusion model's network actually predicts, read from one model call (evidence, not a declaration).

At the noisiest step the input is almost pure noise. An eps model returns that noise (cosine with its input
~1.00, measured on 8 SDXL checkpoints); a converged v-prediction model returns something unrelated to it (~0.01,
NoobAI-XL-Vpred). This is only evidence: a model switching objective from eps to v read 0.94 while its file says v
(issue_track/comfyui_field_test/VPRED_PROTOCOL.md M7), so a declaration in the file comes first (contract.py).
"""
from .facts import Prediction

# Halfway between what theory predicts - eps: cos = sigma_t (0.99 at the last timestep, >0.9 from the middle up),
# v: ~0 at every timestep - so not fitted to data.
BOUNDARY = 0.5
MIN_T = 500  # below this the eps value (sigma_t) nears the boundary; the first call of a low-denoise img2img is skipped


def behaves_like(cos):
    return Prediction("eps") if cos > BOUNDARY else Prediction("v")


def raw_output(kind, x, sigma, denoised, sigma_data=1.0):
    """The network output that became `denoised` under the k-diffusion parametrisation of `kind` ('eps' or 'v')."""
    s = sigma.reshape(sigma.shape[:1] + (1,) * (x.ndim - 1)).to(x.dtype)
    if kind == "eps":
        return (x - denoised) / s
    d2 = sigma_data ** 2
    return (x * d2 / (s ** 2 + d2) - denoised) * (s ** 2 + d2) ** 0.5 / (s * sigma_data)


def cosine_with_input(output, x, sigma, sigma_data=1.0):
    """cos(network output, network input) for one call; the input is x scaled the way the network sees it."""
    import torch.nn.functional as F

    s = sigma.reshape(sigma.shape[:1] + (1,) * (x.ndim - 1)).to(x.dtype)
    xin = x / (s ** 2 + sigma_data ** 2) ** 0.5
    return F.cosine_similarity(output.float().flatten(), xin.float().flatten(), dim=0).item()


def first_call_cos(kind, x, sigma, denoised, sigma_data=1.0):
    """cos(network output, network input) when only `denoised` is visible (k-diffusion engines such as ComfyUI)."""
    return cosine_with_input(raw_output(kind, x, sigma, denoised, sigma_data), x, sigma, sigma_data)
