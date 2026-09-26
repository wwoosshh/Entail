"""Adapter v2 for what transformers' beam search reorders of a model's cache (LIBRARY_DESIGN.md 4.6, 4.8; ROADMAP
M17.2; cache_key_contract.py; transformers#46612).

  hook         transformers.generation.utils.GenerationMixin._beam_search: every step keeps the beams that survive
               and reorders the cache to match (generation/utils.py L3640-3646 on 5.17.0: the first name of
               ALL_CACHE_NAMES present in model_kwargs; 5.12.1 reordered past_key_values only, so a Mamba model's
               cache_params stayed unmoved and the beams continued from other beams' states).
  read_choice  the cache names the model declares: the kwargs of its forward that are cache names.
  handles      none: a name the reorderer does not touch is reported (moving another engine's state from here would
               be a repair inside the step, not a fact handed over).
cache_key_contract decides (cache_key_incomplete) with the row for the installed transformers version; said once
per set of names.
"""
import inspect

from .. import cache_key_contract, core
from .base import Hook

engine = "transformers"
versions = "5.17.0"
BOUNDARY = "request:transformers.generate.beam_reorder"
CONSUMER = "transformers.beam_reorder"
_ORIG = None


def hooks():
    return [Hook("transformers.generation.utils.GenerationMixin._beam_search", "request")]


def _version():
    try:
        import transformers

        return getattr(transformers, "__version__", None)
    except ImportError:
        return None


def read_choice(model):
    """Which cache names the model's forward takes (the table's names for transformers.model_cache)."""
    names = cache_key_contract.table()["fields"]["transformers.model_cache"]["names"]
    try:
        params = set(inspect.signature(model.forward).parameters)
    except (TypeError, ValueError):
        params = set()
    return {n: n in params for n in names}


def handles(model):
    return {}


def _decide(model):
    present = read_choice(model)
    cache_key_contract.check(BOUNDARY, CONSUMER, engine, present, f"{type(model).__name__}.forward", {},
                             owner=tuple(sorted(k for k, v in present.items() if v)), version=_version())


def install():
    global _ORIG
    try:
        from transformers.generation.utils import GenerationMixin
    except ImportError:
        return 0
    if _ORIG is not None:
        return 0
    _ORIG = GenerationMixin._beam_search

    def _beam_search(self, *args, **kwargs):
        if core.mode() in ("load", "debug"):
            from .. import load

            load.safely(BOUNDARY, CONSUMER, "Coverage", lambda: _decide(self))
        return _ORIG(self, *args, **kwargs)

    GenerationMixin._beam_search = _beam_search
    return 1


def uninstall():
    global _ORIG
    if _ORIG is None:
        return 0
    from transformers.generation.utils import GenerationMixin

    GenerationMixin._beam_search = _ORIG
    _ORIG = None
    return 1


def stats():
    return cache_key_contract.stats(BOUNDARY)


def reset():
    cache_key_contract.reset(BOUNDARY)
