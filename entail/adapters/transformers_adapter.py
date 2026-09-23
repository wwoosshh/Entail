"""Adapter v2 for transformers' model loading (LIBRARY_DESIGN.md 4.8; ROADMAP M3.3).

Three things, and no rules:
  hooks        PreTrainedModel._check_and_adjust_attn_implementation. transformers 5.17 calls it at model __init__
               (modeling_utils.py:1263), before the weights are read, and inside set_attn_implementation, which is
               how continuous batching switches to `paged|...`.
               PreTrainedModel.tie_weights: where the head is tied to the embedding (post_init and after loading);
               the contracts about the model itself (load.model_contracts) run there, once per model.
  read_choice  the implementation it settled on, as (group, backend): "paged|sdpa" is sdpa among the paged kernels;
               whether the loader ties the head (what the config it holds says).
  handle       switch_attention_backend: ask the same method for another implementation.
load.attention compares the model's declarations with the backend (caps.json) and decides; load.enforce records the
decision, prints it, and stops on a blocking one. With the mode off the wrapper returns at once, and uninstall()
restores the original method.
"""
from .. import core, load, policies
from .base import Hook

engine = "transformers"
versions = "5.12.1, 5.16.1, 5.17.0"
_ORIG = None
_ORIG_TIE = None
# A model and the model inside it share one config and both ask: decide once per config (and implementation).
_SEEN = load.ByObject()      # config -> True once its model-level contracts have run
_CHOSEN = load.ByObject()    # config -> {implementation asked for: implementation to use}


def hooks():
    return [Hook("transformers.modeling_utils.PreTrainedModel._check_and_adjust_attn_implementation", "load"),
            Hook("transformers.modeling_utils.PreTrainedModel.tie_weights", "load")]


def read_choice(impl):
    """(group role, backend) for an implementation name transformers settled on."""
    if impl.startswith("paged|"):
        return "paged_attention", impl.split("|", 1)[1]
    return "attention", impl


def loader_ties(config):
    """Whether the loader will tie the head: the tie_word_embeddings the config holds (top level, else text_config)."""
    tie = getattr(config, "tie_word_embeddings", None)
    if tie is None:
        tie = getattr(getattr(config, "text_config", None), "tie_word_embeddings", None)
    return tie if isinstance(tie, bool) else None


def handles(model, args, kwargs):
    return {"switch_attention_backend": lambda target: _ORIG(model, target, *args, **kwargs)}


def install():
    """Wrap the loader's attention check. Returns 1, or 0 if already installed."""
    global _ORIG
    from transformers import PreTrainedModel

    if _ORIG is not None:
        return 0
    _ORIG = PreTrainedModel._check_and_adjust_attn_implementation

    def wrapped(self, attn_implementation, *a, **kw):
        impl = _ORIG(self, attn_implementation, *a, **kw)
        if core.mode() not in ("load", "debug") or not isinstance(impl, str):
            return impl
        role, backend = read_choice(impl)
        known = _CHOSEN.get(self.config, {})
        if impl in known:
            return known[impl]

        def decide():
            facts = load.declared(getattr(self.config, "_name_or_path", None), self.config)
            decisions = load.attention(engine, backend, facts, policy=policies.current(), role=role)
            done = load.resolve(decisions, handles(self, a, kw))
            load.enforce(decisions)
            return done.get("switch_attention_backend", impl)

        use = load.safely(f"load:{engine}.{role}", f"{engine}.{role}.{backend}", "ModelProps", decide, impl)
        _CHOSEN.set(self.config, {**known, impl: use})
        return use

    PreTrainedModel._check_and_adjust_attn_implementation = wrapped
    _install_tie(PreTrainedModel)
    return 1


def _install_tie(PreTrainedModel):
    global _ORIG_TIE
    _ORIG_TIE = PreTrainedModel.tie_weights

    def tie_weights(self, *a, **kw):
        out = _ORIG_TIE(self, *a, **kw)
        config = getattr(self, "config", None)
        if core.mode() in ("load", "debug") and config is not None and not _SEEN.get(config):
            _SEEN.set(config, True)

            def decide():
                load.enforce(load.model_contracts(engine, getattr(config, "_name_or_path", None), config,
                                                  loader_ties(config), policies.current()))

            load.safely(f"load:{engine}.loader", f"{engine}.loader", "ModelProps", decide)
        return out

    PreTrainedModel.tie_weights = tie_weights


def uninstall():
    global _ORIG
    if _ORIG is None:
        return 0
    from transformers import PreTrainedModel

    PreTrainedModel._check_and_adjust_attn_implementation = _ORIG
    PreTrainedModel.tie_weights = _ORIG_TIE
    _ORIG = None
    _SEEN.clear()
    _CHOSEN.clear()
    return 1
