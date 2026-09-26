"""Adapter v2 for the rotary pairing vLLM builds a model with (LIBRARY_DESIGN.md 4.8; ROADMAP M17.4;
rotary_pairing_contract.py; vllm#42016, #49290, #53063).

  hook         vllm.model_executor.model_loader.utils.process_weights_after_loading: the model is built; its rotary
               modules hold is_neox_style (rotary_embedding/base.py L35: True = split, False = interleaved), which
               every forward passes to the rotation op.
  read_choice  module name -> the convention the module holds, for every module with is_neox_style in the part of
               the model the declaration is about: a multimodal model's language model (SupportsMultiModal
               .get_language_model), else the whole model; modules named `indexer` are left out (a DSA indexer
               pairs by its own key, indexer_rope_interleave). Whether an MRoPE module dispatches to the Triton
               MRoPE kernel (its dispatched forward is not forward_native: the kernel row of the table applies);
               the config dict and model_type for the declaration.
  handles      set_pairing: set is_neox_style on that part's rotary modules to the declared convention (read at
               forward time). vLLM caches rotary modules process-wide (get_rope), so the flip reaches every user of
               the instance, as vLLM's own in-model flips do (ernie45.py, glm.py).
rotary_pairing_contract decides; a model that declares nothing (no key, an architecture the table does not know)
gets no decision. The table's architecture rows declare the text model's convention, so only the language model is
compared: GLM-OCR's vision tower pairs split-wise by its own reference (modeling_glm_ocr.py L335) and vLLM builds
it so (26 modules with is_neox_style True beside the text model's 2 with False); the first run compared the whole
model and "repaired" the vision tower to interleaved. The other parts of a multimodal model are not compared, and a
multimodal model whose language model cannot be found is said unknown, not widened to the whole model.
"""
from .. import core, rotary_pairing_contract
from .base import Hook

engine = "vllm"
versions = "0.30.0"
BOUNDARY = "load:vllm.rotary_pairing"
CONSUMER = "vllm.rotary_embedding"
_ORIG = None
OWN_KEY_MODULES = ("indexer",)     # module names that pair by a key of their own (DSA: indexer_rope_interleave)


def hooks():
    return [Hook("vllm.model_executor.model_loader.utils.process_weights_after_loading", "load")]


def language_model(model):
    """The part of the model the text declaration is about: a multimodal model's language model, the model itself
    when it has no such accessor, or None when the accessor fails (nothing is compared then)."""
    get = getattr(model, "get_language_model", None)
    if not callable(get):
        return model
    try:
        return get()
    except Exception:  # noqa: BLE001 - vLLM's default raises NotImplementedError for a model that marks none
        return None


def _own_key(name):
    return any(part in name.split(".") for part in OWN_KEY_MODULES)


def _dispatches_to_kernel(module):
    """Whether a CustomOp module runs its CUDA/HIP forward rather than forward_native (custom_op.py: the dispatched
    forward is kept as _forward_method). Unreadable -> False: the kernel row is not claimed."""
    fwd = getattr(module, "_forward_method", None)
    name = getattr(fwd, "__name__", None)
    return bool(name) and name != "forward_native"


def read_choice(model, model_config):
    part = language_model(model)
    held, mrope = {}, False
    for name, m in (getattr(part, "named_modules", lambda: [])() if part is not None else []):
        if _own_key(name):
            continue
        style = getattr(m, "is_neox_style", None)
        if isinstance(style, bool):
            held[name] = "split" if style else "interleaved"
        if "MRotary" in type(m).__name__ and _dispatches_to_kernel(m):
            mrope = True
    hf = getattr(model_config, "hf_config", None)
    config = None
    if hf is not None and hasattr(hf, "to_dict"):
        try:
            config = hf.to_dict()
        except Exception:  # noqa: BLE001
            config = None
    return held, mrope, config, getattr(model_config, "model", None)


def handles(model):
    def set_pairing(pairing):
        part = language_model(model)
        if part is None:
            return False
        for name, m in part.named_modules():
            if _own_key(name):
                continue
            if isinstance(getattr(m, "is_neox_style", None), bool):
                m.is_neox_style = pairing == "split"
        return True

    return {"set_pairing": set_pairing}


def _decide(model, model_config):
    from .. import load

    try:
        import vllm

        version = getattr(vllm, "__version__", None)
    except ImportError:
        version = None
    path = getattr(model_config, "model", None)
    if language_model(model) is None:
        load.enforce([load.cannot_check(BOUNDARY, CONSUMER, "Rotary",
                                        f"{type(model).__name__} (model {path}): its language model could not be "
                                        f"found (get_language_model failed), so its rotary pairing is not compared")])
        return
    held, mrope, config, path = read_choice(model, model_config)
    pairing, where = rotary_pairing_contract.declared(config)
    part = "language model's " if language_model(model) is not model else ""
    rotary_pairing_contract.check(BOUNDARY, CONSUMER, engine, pairing, where, held,
                                  f"{type(model).__name__}'s {part}rotary modules (model {path})",
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
