"""Adapter v2 for vLLM's loader: the contracts about the model itself (LIBRARY_DESIGN.md 4.8; ROADMAP M3.3; test
problem rolebench 07, the mechanism of vLLM #51063: the loader skips a checkpoint's lm_head.weight when the config
says the head is tied, so a wrong tie declaration silently replaces the trained head with the embedding).

  hook         vllm.model_executor.model_loader.utils.process_weights_after_loading: the model is built and loaded;
               the decision runs after the step, which is where vLLM re-ties a loaded head that equals the
               embedding (maybe_retie_word_embeddings).
  read_choice  the config object the engine holds, the model path, and whether the loader was told to tie (what
               the config said before vLLM untied it on seeing a shipped head); tied_in_memory reads what the
               loader left in the model.
  handles      none.
load.model_contracts decides: the tie against the checkpoint bytes, the RoPE the engine holds against the declared
one, a stored layout against its data; load.enforce records them and stops on a blocking one.
"""
from .. import core, load, policies
from .base import Hook

engine = "vllm"
versions = "0.30.0"


def hooks():
    return [Hook("vllm.model_executor.model_loader.utils.process_weights_after_loading", "load")]


compares_head = True   # vLLM 0.30: maybe_untie_word_embeddings, maybe_retie_word_embeddings (load.tie)


def read_choice(model_config):
    """The config object the engine holds, the model path, and whether the loader was told to tie: the tie the
    config said before vLLM 0.30's maybe_untie_word_embeddings set it to False on seeing a shipped lm_head.weight
    (the loader then loads that head, compares it with the embedding and re-ties the two when they are equal;
    load.tie decides with `compares_head`, M11.2)."""
    config = getattr(model_config, "hf_text_config", None) or getattr(model_config, "hf_config", None)
    tie = getattr(config, "tie_word_embeddings", None)
    if getattr(model_config, "word_embeddings_untied_by_checkpoint", False):
        tie = True
    return config, getattr(model_config, "model", None), tie if isinstance(tie, bool) else None


def tied_in_memory(model):
    """Whether the model's one lm_head shares its weight tensor with its one input embedding after the loader's
    step (a head loaded from the checkpoint and re-tied when it equals the embedding counts as tied); None when
    there is not exactly one of each (several heads, a nested language model), found by type as vLLM's own
    weight_tying finds them."""
    try:
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
    except ImportError:
        return None
    heads, embeds = [], []
    for _, module in model.named_modules():
        if isinstance(module, ParallelLMHead):
            heads.append(module)
        elif isinstance(module, VocabParallelEmbedding):
            embeds.append(module)
    if len(heads) != 1 or len(embeds) != 1:
        return None
    h, e = getattr(heads[0], "weight", None), getattr(embeds[0], "weight", None)
    if h is None or e is None or tuple(h.shape) != tuple(e.shape):
        return None
    return h is e or h.data_ptr() == e.data_ptr()


def handles():
    return {}


def install():
    from vllm.model_executor.model_loader import utils as loader_utils

    orig = loader_utils.process_weights_after_loading

    def wrapped(model, model_config, target_device, *a, **kw):
        out = orig(model, model_config, target_device, *a, **kw)   # maybe_retie_word_embeddings runs inside it
        if core.mode() in ("load", "debug"):
            config, path, ties = read_choice(model_config)

            def decide():
                load.enforce(load.model_contracts(engine, path, config, ties, policies.current(),
                                                  compares_head=compares_head, tied_in_memory=tied_in_memory(model)))

            if config is not None:
                load.safely(f"load:{engine}.loader", f"{engine}.loader", "ModelProps", decide)
        return out

    loader_utils.process_weights_after_loading = wrapped
    return 1
