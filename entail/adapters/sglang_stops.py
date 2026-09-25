"""Adapter v2 for the stop set SGLang's scheduler uses: ModelConfig.hf_eos_token_id (LIBRARY_DESIGN.md 4.8; ROADMAP
M15.8; stops_contract.py).

  hook         sglang.srt.configs.model_config.ModelConfig.__init__: once built, the config caches hf_eos_token_id =
               config.json's eos ids | generation_config.json's (srt/configs/model_config.py _get_hf_eos_token_id),
               which the scheduler reads for every request.
  read_choice  the ids that set holds.
  handles      add_stops: add the declared ids to the set (the scheduler reads it by reference).
SGLang takes both files, so in practice this boundary passes; it is here so the same rule stands at every engine.
"""
from .. import core, stops_contract
from .base import Hook

engine = "sglang"
versions = "0.5.20"
BOUNDARY = "load:sglang.model_config"
CONSUMER = "sglang.scheduler.stop_set"
_ORIG = None


def hooks():
    return [Hook("sglang.srt.configs.model_config.ModelConfig.__init__", "load")]


def read_choice(model_config):
    """The eos ids the model config holds as a set; None when it holds nothing."""
    ids = getattr(model_config, "hf_eos_token_id", None)
    if ids is None:
        return None
    return {int(i) for i in ids}


def handles(model_config):
    def add_stops(ids):
        current = set(getattr(model_config, "hf_eos_token_id", None) or ())
        model_config.hf_eos_token_id = current | {int(i) for i in ids}
        return True

    return {"add_stops": add_stops}


def _decide(model_config):
    from .. import load

    name = getattr(model_config, "model_path", None)
    folder = load.local_folder(name, getattr(model_config, "revision", None), None) if name else None
    where = f"ModelConfig.hf_eos_token_id (model {name})"
    if folder is None:
        load.enforce([load.cannot_check(BOUNDARY, CONSUMER, "Stops", f"{where}: no local folder to read")])
        return
    stops_contract.check(BOUNDARY, CONSUMER, load.declared(folder), read_choice(model_config), where,
                         add_stops=handles(model_config)["add_stops"], owner=folder)


def install():
    global _ORIG
    try:
        from sglang.srt.configs.model_config import ModelConfig
    except ImportError:
        return 0
    if _ORIG is not None:
        return 0
    _ORIG = ModelConfig.__init__

    def __init__(self, *args, **kwargs):
        _ORIG(self, *args, **kwargs)
        if core.mode() in ("load", "debug"):
            from .. import load

            load.safely(BOUNDARY, CONSUMER, "Stops", lambda: _decide(self))

    ModelConfig.__init__ = __init__
    return 1


def uninstall():
    global _ORIG
    if _ORIG is None:
        return 0
    from sglang.srt.configs.model_config import ModelConfig

    ModelConfig.__init__ = _ORIG
    _ORIG = None
    return 1


def stats():
    return stops_contract.stats(BOUNDARY)


def reset():
    stops_contract.reset(BOUNDARY)
