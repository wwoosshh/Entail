"""Adapter v2 for where vLLM binds a multimodal item's placeholder (LIBRARY_DESIGN.md 11 M18; ROADMAP M18.4;
placeholder_contract.py; vllm#57740).

  hook         vllm.multimodal.processing.processor.BaseMultiModalProcessor._maybe_apply_prompt_updates: the
               processor has expanded the prompt's placeholders and found where each item sits (start index and
               tokens per modality). Every request with multimodal data passes here.
  read_choice  the placeholder runs per modality, and the model's declared markup from its config
               (vision_start_token_id, image_token_id / video_token_id) through the processor's context.
  handles      none: the prompt is the user's; a placeholder bound inside the user's text is reported.
A model that declares no markup decides nothing. The wrapper passes every argument through whole (principle 12).
"""
from .. import core, placeholder_contract
from .base import Hook

engine = "vllm"
versions = "0.30.0"
BOUNDARY = "request:vllm.multimodal"
CONSUMER = "vllm.multimodal_processor"
_ORIG = {}


def hooks():
    return [Hook("vllm.multimodal.processing.processor.BaseMultiModalProcessor._maybe_apply_prompt_updates",
                 "request")]


def handles():
    return {}


def read_choice(processor, prompt_ids, mm_placeholders):
    """(runs per modality as (offset, length), the declared markup) from what the processor placed."""
    runs = {}
    for modality, infos in (mm_placeholders or {}).items():
        out = []
        for p in infos or []:
            start = getattr(p, "start_idx", None)
            if start is None:
                start = getattr(p, "offset", None)
            length = getattr(p, "length", None)
            if length is None:
                length = len(getattr(p, "tokens", ()) or ())
            if isinstance(start, int) and isinstance(length, int) and length > 0:
                out.append((start, length))
        runs[modality] = out
    config = None
    info = getattr(processor, "info", None)
    ctx = getattr(info, "ctx", None)
    model_config = getattr(ctx, "model_config", None)
    config = getattr(model_config, "hf_config", None)
    if config is None and callable(getattr(ctx, "get_hf_config", None)):
        try:
            config = ctx.get_hf_config()
        except Exception:  # noqa: BLE001
            config = None
    return runs, placeholder_contract.declared_markup(config) if config is not None else {}


def _decide(processor, prompt_ids, mm_placeholders):
    runs, markup = read_choice(processor, prompt_ids, mm_placeholders)
    if not markup or not any(runs.values()):
        return
    name = type(processor).__name__
    placeholder_contract.check(BOUNDARY, f"vllm.multimodal.{name}", list(prompt_ids), runs, markup,
                               f"{name} on a prompt of {len(prompt_ids)} tokens")


def install():
    try:
        from vllm.multimodal.processing.processor import BaseMultiModalProcessor
    except ImportError:
        return 0
    if "apply" in _ORIG:
        return 0
    orig = _ORIG["apply"] = BaseMultiModalProcessor._maybe_apply_prompt_updates

    def _maybe_apply_prompt_updates(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        if core.mode() in ("load", "debug"):
            from .. import load

            try:
                prompt_ids, mm_placeholders = out
            except (TypeError, ValueError):
                return out
            load.safely(BOUNDARY, CONSUMER, "Placeholder", lambda: _decide(self, prompt_ids, mm_placeholders))
        return out

    BaseMultiModalProcessor._maybe_apply_prompt_updates = _maybe_apply_prompt_updates
    return 1


def uninstall():
    if "apply" not in _ORIG:
        return 0
    from vllm.multimodal.processing.processor import BaseMultiModalProcessor

    BaseMultiModalProcessor._maybe_apply_prompt_updates = _ORIG.pop("apply")
    return 1


def stats():
    return placeholder_contract.stats(BOUNDARY)


def reset():
    placeholder_contract.reset(BOUNDARY)
