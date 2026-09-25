"""Adapter v2 for the stop set transformers' generate() uses: the model's generation_config (LIBRARY_DESIGN.md 4.8;
ROADMAP M15.8; stops_contract.py).

  hook         transformers.modeling_utils.PreTrainedModel.from_pretrained: the model, once built, with the
               generation_config transformers gave it (generation_config.json, else the config's own ids).
  read_choice  the eos ids that generation_config holds (an id or a list); None when the model has none.
  handles      add_stops: write the union into generation_config.eos_token_id - the list generate() stops on.
The rule is in the core (stops_contract.check): every id the files declare as an end is an end. transformers reads
generation_config.json alone, so an end config.json declares and that file leaves out is dropped here
(data/stops_sources.json).
"""
from .. import core, stops_contract
from .base import Hook

engine = "transformers"
versions = "5.17.0"
BOUNDARY = "load:transformers.generation_config"
CONSUMER = "transformers.generate"
_ORIG = None


def hooks():
    return [Hook("transformers.modeling_utils.PreTrainedModel.from_pretrained", "load")]


def read_choice(model):
    """The eos ids the model's generation_config holds, as a set; None when the model carries no generation config
    (a model that does not generate)."""
    gc = getattr(model, "generation_config", None)
    if gc is None:
        return None
    eos = getattr(gc, "eos_token_id", None)
    if eos is None:
        return set()
    return {int(i) for i in (eos if isinstance(eos, (list, tuple, set)) else [eos])}


def handles(model):
    def add_stops(ids):
        gc = model.generation_config
        eos = getattr(gc, "eos_token_id", None)
        current = {int(i) for i in (eos if isinstance(eos, (list, tuple, set)) else ([] if eos is None else [eos]))}
        gc.eos_token_id = sorted(current | {int(i) for i in ids})
        return True

    return {"add_stops": add_stops}


def _decide(name, kwargs, model):
    from .. import load

    held = read_choice(model)
    if held is None:
        return
    folder = load.local_folder(name, kwargs.get("revision"), kwargs.get("cache_dir"))
    where = f"{type(model).__name__}.generation_config.eos_token_id (built from {name})"
    if folder is None:
        load.enforce([load.cannot_check(BOUNDARY, CONSUMER, "Stops", f"{where}: no local folder to read")])
        return
    stops_contract.check(BOUNDARY, CONSUMER, load.declared(folder), held, where, add_stops=handles(model)["add_stops"],
                         owner=folder)


def install():
    """Wrap the classmethod on the base class (every model class inherits it). Returns 1, or 0 if installed."""
    global _ORIG
    try:
        from transformers.modeling_utils import PreTrainedModel
    except ImportError:
        return 0
    if _ORIG is not None:
        return 0
    _ORIG = PreTrainedModel.__dict__["from_pretrained"].__func__

    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        model = _ORIG(cls, pretrained_model_name_or_path, *args, **kwargs)
        if core.mode() in ("load", "debug") and isinstance(pretrained_model_name_or_path, (str, bytes)) \
                or (core.mode() in ("load", "debug") and hasattr(pretrained_model_name_or_path, "__fspath__")):
            from .. import load

            load.safely(BOUNDARY, CONSUMER, "Stops", lambda: _decide(str(pretrained_model_name_or_path), kwargs, model))
        return model

    PreTrainedModel.from_pretrained = classmethod(from_pretrained)
    return 1


def uninstall():
    global _ORIG
    if _ORIG is None:
        return 0
    from transformers.modeling_utils import PreTrainedModel

    PreTrainedModel.from_pretrained = classmethod(_ORIG)
    _ORIG = None
    return 1


def stats():
    return stops_contract.stats(BOUNDARY)


def reset():
    stops_contract.reset(BOUNDARY)
