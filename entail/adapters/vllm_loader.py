"""Adapter v2 for vLLM's loader: the contracts about the model itself (LIBRARY_DESIGN.md 4.8; ROADMAP M3.3; test
problem rolebench 07, the mechanism of vLLM #51063: the loader skips a checkpoint's lm_head.weight when the config
says the head is tied, so a wrong tie declaration silently replaces the trained head with the embedding).

  hook         vllm.model_executor.model_loader.utils.process_weights_after_loading: the model is built and loaded.
  read_choice  the config object the engine holds, the model path, and whether the loader ties the head (what that
               config says).
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


def read_choice(model_config):
    config = getattr(model_config, "hf_text_config", None) or getattr(model_config, "hf_config", None)
    tie = getattr(config, "tie_word_embeddings", None)
    return config, getattr(model_config, "model", None), tie if isinstance(tie, bool) else None


def handles():
    return {}


def install():
    from vllm.model_executor.model_loader import utils as loader_utils

    orig = loader_utils.process_weights_after_loading

    def wrapped(model, model_config, target_device, *a, **kw):
        if core.mode() in ("load", "debug"):
            config, path, ties = read_choice(model_config)

            def decide():
                load.enforce(load.model_contracts(engine, path, config, ties, policies.current()))

            if config is not None:
                load.safely(f"load:{engine}.loader", f"{engine}.loader", "ModelProps", decide)
        return orig(model, model_config, target_device, *a, **kw)

    loader_utils.process_weights_after_loading = wrapped
    return 1
