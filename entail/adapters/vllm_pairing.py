"""Adapter v2 for the rotary pairing vLLM builds a model with (LIBRARY_DESIGN.md 4.8; ROADMAP M17.4;
rotary_pairing_contract.py; vllm#42016, #49290, #53063).

  hook         vllm.model_executor.model_loader.utils.process_weights_after_loading: the model is built; its rotary
               layers hold is_neox_style (rotary_embedding/base.py L35: True = split, False = interleaved), which
               every forward passes to the rotation op.
  read_choice  layer name -> the convention the layer holds, for every module with is_neox_style in the part of the
               model the declaration is about: a multimodal model's language model (SupportsMultiModal
               .get_language_model), else the whole model; whether the model is an MRoPE one (the Triton MRoPE
               kernel path); the config dict and model_type for the declaration.
  handles      set_pairing: set is_neox_style on that part's rotary layers to the declared convention (read at
               forward time).
rotary_pairing_contract decides; a model that declares nothing (no key, an architecture the table does not know)
gets no decision. The table's architecture rows declare the text model's convention, so only the language model is
compared: GLM-OCR's vision tower pairs split-wise by its own reference (modeling_glm_ocr.py L335) and vLLM builds
it so (26 modules with is_neox_style True beside the text model's 2 with False); the first run compared the whole
model and "repaired" the vision tower to interleaved. The other parts of a multimodal model are not compared.
"""
from .. import core, rotary_pairing_contract
from .base import Hook

engine = "vllm"
versions = "0.30.0"
BOUNDARY = "load:vllm.rotary_pairing"
CONSUMER = "vllm.rotary_embedding"
_ORIG = None


def hooks():
    return [Hook("vllm.model_executor.model_loader.utils.process_weights_after_loading", "load")]


def language_model(model):
    """The part of the model the text declaration is about: a multimodal model's language model, else the model."""
    get = getattr(model, "get_language_model", None)
    if callable(get):
        try:
            lm = get()
            if lm is not None:
                return lm
        except Exception:  # noqa: BLE001
            pass
    return model


def read_choice(model, model_config):
    held, mrope = {}, False
    for name, m in getattr(language_model(model), "named_modules", lambda: [])():
        style = getattr(m, "is_neox_style", None)
        if isinstance(style, bool):
            held[name] = "split" if style else "interleaved"
        if "MRotary" in type(m).__name__:
            mrope = True
    hf = getattr(model_config, "hf_config", None)
    config = None
    if hf is not None and hasattr(hf, "to_dict"):
        try:
            config = hf.to_dict()
        except Exception:  # noqa: BLE001
            config = None
    if not mrope and isinstance(config, dict):
        tc = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
        rp = tc.get("rope_parameters") or tc.get("rope_scaling") or {}
        mrope = bool(isinstance(rp, dict) and rp.get("mrope_section"))
    return held, mrope, config, getattr(model_config, "model", None)


def handles(model):
    def set_pairing(pairing):
        for _, m in language_model(model).named_modules():
            if isinstance(getattr(m, "is_neox_style", None), bool):
                m.is_neox_style = pairing == "split"
        return True

    return {"set_pairing": set_pairing}


def _decide(model, model_config):
    try:
        import vllm

        version = getattr(vllm, "__version__", None)
    except ImportError:
        version = None
    held, mrope, config, path = read_choice(model, model_config)
    pairing, where = rotary_pairing_contract.declared(config)
    part = "language model's" if language_model(model) is not model else ""
    rotary_pairing_contract.check(BOUNDARY, CONSUMER, engine, pairing, where, held,
                                  f"{type(model).__name__}'s {part + ' ' if part else ''}rotary layers (model {path})",
                                  version=version, mrope=mrope, handles=handles(model), owner=path)


def install():
    global _ORIG
    try:
        from vllm.model_executor.model_loader import utils as loader_utils
    except ImportError:
        return 0
    if _ORIG is not None:
        return 0
    _ORIG = loader_utils.process_weights_after_loading

    def wrapped(model, model_config, target_device, *a, **kw):
        out = _ORIG(model, model_config, target_device, *a, **kw)
        if core.mode() in ("load", "debug"):
            from .. import load

            load.safely(BOUNDARY, CONSUMER, "Rotary", lambda: _decide(model, model_config))
        return out

    loader_utils.process_weights_after_loading = wrapped
    return 1


def uninstall():
    global _ORIG
    if _ORIG is None:
        return 0
    from vllm.model_executor.model_loader import utils as loader_utils

    loader_utils.process_weights_after_loading = _ORIG
    _ORIG = None
    return 1


def stats():
    return rotary_pairing_contract.stats(BOUNDARY)


def reset():
    rotary_pairing_contract.reset(BOUNDARY)
