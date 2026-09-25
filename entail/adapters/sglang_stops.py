"""Adapter v2 for the stop set SGLang's scheduler uses (LIBRARY_DESIGN.md 4.8; ROADMAP M15.8; stops_contract.py).

  hook         sglang.srt.configs.model_config.ModelConfig.from_server_args: the scheduler builds its model config
               here, with the server arguments in hand; the config caches hf_eos_token_id = config.json's eos ids |
               generation_config.json's (srt/configs/model_config.py _get_hf_eos_token_id), which every request gets.
               The scheduler also matches the tokenizer's eos_token_id for every request (schedule_batch.py
               check_finished) unless the server was started with skip_tokenizer_init - so the set SGLang stops on
               is the two files' ids plus the tokenizer's, and the files' declaration of the tokenizer's end
               (tokenizer_config.json, read by the core) counts as held (M15.8 review: the first version of this
               adapter left it out and "repaired" what the scheduler already matched).
  read_choice  the ids hf_eos_token_id holds.
  handles      add_stops: add the declared ids to the set (the scheduler reads the attribute when a request is built).
SGLang takes every source, so in practice this boundary passes; it is here so the same rule stands at every
engine, and so that a server started with skip_tokenizer_init still gets the tokenizer's declared end.
"""
from .. import core, stops_contract
from .base import Hook

engine = "sglang"
versions = "0.5.20"
BOUNDARY = "load:sglang.model_config"
CONSUMER = "sglang.scheduler.stop_set"
_ORIG = None


def hooks():
    return [Hook("sglang.srt.configs.model_config.ModelConfig.from_server_args", "load")]


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


def _decide(model_config, skip_tokenizer_init=False):
    from .. import load

    name = getattr(model_config, "model_path", None)
    folder = load.local_folder(name, getattr(model_config, "revision", None), None) if name else None
    where = f"ModelConfig.hf_eos_token_id plus the tokenizer's eos the scheduler matches (model {name})"
    if folder is None:
        load.enforce([load.cannot_check(BOUNDARY, CONSUMER, "Stops", f"{where}: no local folder to read")])
        return
    facts = load.declared(folder)
    held = read_choice(model_config)
    if held is not None and not skip_tokenizer_init:
        # the scheduler matches the tokenizer's eos_token_id for every request: the files' statement of that end
        # (tokenizer_config.json, read by the core) is in the set the engine stops on
        for f in facts.get("Stops"):
            if f.value is not None and "tokenizer" in f.source.where.lower():
                held |= {int(i) for i in f.value.eos}
    else:
        where = f"ModelConfig.hf_eos_token_id (model {name}; skip_tokenizer_init: no tokenizer at the scheduler)"
    stops_contract.check(BOUNDARY, CONSUMER, facts, held, where, add_stops=handles(model_config)["add_stops"],
                         owner=folder)


def install():
    global _ORIG
    try:
        from sglang.srt.configs.model_config import ModelConfig
    except ImportError:
        return 0
    if _ORIG is not None:
        return 0
    _ORIG = ModelConfig.__dict__["from_server_args"].__func__

    def from_server_args(server_args, *args, **kwargs):
        model_config = _ORIG(server_args, *args, **kwargs)
        if core.mode() in ("load", "debug"):
            from .. import load

            skip = bool(getattr(server_args, "skip_tokenizer_init", False))
            load.safely(BOUNDARY, CONSUMER, "Stops", lambda: _decide(model_config, skip))
        return model_config

    ModelConfig.from_server_args = staticmethod(from_server_args)
    return 1


def uninstall():
    global _ORIG
    if _ORIG is None:
        return 0
    from sglang.srt.configs.model_config import ModelConfig

    ModelConfig.from_server_args = staticmethod(_ORIG)
    _ORIG = None
    return 1


def stats():
    return stops_contract.stats(BOUNDARY)


def reset():
    stops_contract.reset(BOUNDARY)
