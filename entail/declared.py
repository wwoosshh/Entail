"""What a model artifact says about itself, read the same way whatever engine loads it.

A checkpoint or a LoRA often states its own properties:
  - a safetensors header carries metadata: `modelspec.prediction_type` and `modelspec.architecture` (the SAI
    model spec), kohya's `ss_v_parameterization`, `ss_zero_terminal_snr` and `ss_base_model_version`;
  - some tools mark a checkpoint with extra keys instead: `v_pred` and `ztsnr` (read by ComfyUI and Forge);
  - a Hugging Face folder states them in config.json and scheduler/scheduler_config.json.
Engines read some of these and drop the rest. ComfyUI 0.34 reads only the marker keys and diffusers 0.40 reads
none of them for a single file, so a checkpoint that says `modelspec.prediction_type = v` is sampled as eps by both
(measured on AstolfoCarmix-VPredXL, issue_track/comfyui_field_test/VPRED_PROTOCOL.md M7).

This module turns every such statement into facts, remembers where each one was read, and records it when two
statements in the same artifact disagree. It never loads tensors: a header or a config file is enough.
"""
import json
import os
import re
import struct
from dataclasses import dataclass, field

from .facts import Base, Prediction


@dataclass
class Declared:
    facts: dict = field(default_factory=dict)      # fact class name -> fact
    sources: dict = field(default_factory=dict)    # fact class name -> where it was read
    conflicts: list = field(default_factory=list)  # statements in the same artifact that disagree
    modules: frozenset = frozenset()               # for a LoRA: the modules it carries weights for

    def get(self, kind):
        return self.facts.get(kind if isinstance(kind, str) else kind.__name__)

    def source(self, kind):
        return self.sources.get(kind if isinstance(kind, str) else kind.__name__)

    def __bool__(self):
        return bool(self.facts or self.modules)


_KINDS = {"v": "v", "v_prediction": "v", "v-prediction": "v", "vpred": "v", "epsilon": "eps", "eps": "eps",
          "sample": "x0", "x0": "x0", "flow": "flow", "rectified_flow": "flow", "flow_matching": "flow", "edm": "edm"}
_FAMILIES = (("stable-diffusion-xl", "sdxl"), ("sdxl", "sdxl"), ("stable-diffusion-v1", "sd1"), ("sd_v1", "sd1"),
             ("stable-diffusion-v2", "sd2"), ("sd_v2", "sd2"), ("anima", "anima"), ("flux", "flux"),
             ("stable-diffusion-v3", "sd3"), ("sd3", "sd3"))
_TRUE = {"true", "1", "yes"}


def prediction_kind(text):
    """'v_prediction' / 'v' -> 'v', 'epsilon' -> 'eps', 'sample' -> 'x0', ...; None for anything else."""
    return _KINDS.get(str(text).strip().lower())


_kind = prediction_kind


def _flag(text):
    t = str(text).strip().lower()
    return True if t in _TRUE else False if t in {"false", "0", "no"} else None


def family(text):
    """'stable-diffusion-xl-v1-base/lora' -> 'sdxl', 'sdxl_base_v1-0' -> 'sdxl', 'anima-preview/lora' -> 'anima'."""
    t = str(text).strip().lower()
    for needle, name in _FAMILIES:
        if needle in t:
            return name
    return t.split("/")[0] or None


def _put(decl, fact, source):
    name = type(fact).__name__
    have = decl.facts.get(name)
    if have is None:
        decl.facts[name], decl.sources[name] = fact, source
    elif have != fact:
        decl.conflicts.append(f"{decl.sources[name]} says {have}, {source} says {fact}")


def from_header(keys, metadata):
    """Declarations in a safetensors header: the tensor names and the metadata dictionary."""
    decl = Declared()
    meta = metadata or {}
    keys = set(keys)
    zsnr, zsnr_from = None, None
    if "ztsnr" in keys:
        zsnr, zsnr_from = True, "key 'ztsnr'"
    elif _flag(meta.get("ss_zero_terminal_snr")) is not None:
        zsnr, zsnr_from = _flag(meta["ss_zero_terminal_snr"]), "metadata ss_zero_terminal_snr"
    statements = []
    if _kind(meta.get("modelspec.prediction_type", "")):
        statements.append((_kind(meta["modelspec.prediction_type"]), "metadata modelspec.prediction_type"))
    if _flag(meta.get("ss_v_parameterization")) is not None:
        statements.append(("v" if _flag(meta["ss_v_parameterization"]) else "eps", "metadata ss_v_parameterization"))
    if "v_pred" in keys:
        statements.append(("v", "key 'v_pred'"))
    for kind, source in statements:
        _put(decl, Prediction(kind, zsnr), source + (f" (zsnr: {zsnr_from})" if zsnr_from else ""))
    for key in ("modelspec.architecture", "ss_base_model_version"):
        if meta.get(key):
            _put(decl, Base(family(meta[key])), f"metadata {key}")
            break
    loras = lora_modules(keys)
    if loras:
        decl.modules = frozenset(loras)
    return decl


def announce(decl, where):
    """One line per artifact whose own statements disagree (a merge that kept two tools' metadata, say)."""
    for c in decl.conflicts:
        print(f"[entail] {where}: the file's own statements disagree: {c}", flush=True)


def safetensors_header(path):
    """(tensor names, metadata) from a .safetensors file, reading only its header."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", None) or {}
    return list(header), meta


def from_file(path):
    """Declarations of one model file; an empty Declared for formats that carry none (pickled .ckpt, .gguf, ...)."""
    if not str(path).endswith(".safetensors") or not os.path.isfile(path):
        return Declared()
    return from_header(*safetensors_header(path))


def from_hf_folder(path):
    """Declarations in a Hugging Face folder: the diffusers scheduler config (prediction type, zero-SNR rescaling)."""
    decl = Declared()
    sched = os.path.join(path, "scheduler", "scheduler_config.json")
    if os.path.isfile(sched):
        cfg = json.load(open(sched, encoding="utf-8"))
        kind = _kind(cfg.get("prediction_type", ""))
        if kind:
            zsnr = cfg.get("rescale_betas_zero_snr")
            _put(decl, Prediction(kind, zsnr if isinstance(zsnr, bool) else None),
                 "scheduler/scheduler_config.json prediction_type")
    return decl


# What follows a module name in the LoRA formats in use (kohya, peft/diffusers, LyCORIS and friends).
_PART = re.compile(r"\.(?:lora_up|lora_down|lora_A|lora_B|lora_mid|lora\.up|lora\.down|alpha|dora_scale|hada_\w+|"
                   r"lokr_\w+|diff|diff_b|set_weight|w_norm|b_norm|reshape_weight|lora_linear_layer)(?:\.|$)")
TEXT_PREFIXES = ("lora_te", "text_encoder", "text_model", "te_", "te1_", "te2_", "clip_l", "clip_g", "t5xxl")


def lora_module(key):
    """The module a LoRA tensor belongs to: 'lora_unet_x.lora_down.weight' -> 'lora_unet_x'."""
    m = _PART.search(key)
    return key[:m.start()] if m else key.rsplit(".", 1)[0]


def lora_modules(keys):
    """Module names of a LoRA, or an empty set when the keys are not LoRA keys."""
    keys = list(keys)
    if not any(_PART.search(k) for k in keys):
        return set()
    return {lora_module(k) for k in keys}


def is_text_module(name):
    return name.startswith(TEXT_PREFIXES)
