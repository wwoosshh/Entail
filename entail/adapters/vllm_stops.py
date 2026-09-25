"""Adapter v2 for the stop set vLLM gives every request: the generation-config fields the input processor keeps,
plus the tokenizer's eos (LIBRARY_DESIGN.md 4.8; ROADMAP M15.8; stops_contract.py).

  hook         vllm.v1.engine.input_processor.InputProcessor.__init__: the front process builds it once, with
               generation_config_fields = model_config.try_get_generation_config() (generation_config.json, else the
               config's own ids) and the renderer whose get_eos_token_id() is the tokenizer's eos; every request's
               SamplingParams takes both (sampling_params.update_from_generation_config).
  read_choice  the eos ids the fields hold plus the tokenizer's, the tokenizer's highest id + 1 and its special ids.
  handles      add_stops: write the union into generation_config_fields["eos_token_id"], which every later request
               reads.
"""
from .. import core, stops_contract
from .base import Hook

engine = "vllm"
versions = "0.30.0"
BOUNDARY = "load:vllm.input_processor"
CONSUMER = "vllm.input_processor.stop_set"
_ORIG = None


def hooks():
    return [Hook("vllm.v1.engine.input_processor.InputProcessor.__init__", "load")]


def _ids(v):
    if v is None:
        return set()
    return {int(i) for i in (v if isinstance(v, (list, tuple, set)) else [v])}


def read_choice(proc):
    """(held eos ids, the tokenizer's eos or None, the tokenizer's highest id + 1 or None, its special ids or None)."""
    fields = getattr(proc, "generation_config_fields", None)
    held = _ids(fields.get("eos_token_id")) if isinstance(fields, dict) else set()
    tok_eos = None
    try:
        tok_eos = proc.renderer.get_eos_token_id()
    except Exception:  # noqa: BLE001 - a renderer without a tokenizer (pooling models)
        tok_eos = None
    if tok_eos is not None:
        held.add(int(tok_eos))
    tok = getattr(proc, "tokenizer", None)
    size = special = None
    if tok is not None:
        try:
            size = len(tok)
            special = list(getattr(tok, "all_special_ids", None) or [])
        except Exception:  # noqa: BLE001 - a tokenizer that cannot say
            size = special = None
    return held, (None if tok_eos is None else int(tok_eos)), size, special


def handles(proc):
    def add_stops(ids):
        fields = proc.generation_config_fields
        if not isinstance(fields, dict):
            return False
        fields["eos_token_id"] = sorted(_ids(fields.get("eos_token_id")) | {int(i) for i in ids})
        return True

    return {"add_stops": add_stops}


def _decide(proc):
    from .. import load
    from ..facts import Certainty, Fact, Source, Stops

    mc = proc.model_config
    name = getattr(mc, "hf_config_path", None) or getattr(mc, "model", None)
    folder = load.local_folder(name, getattr(mc, "revision", None), None) if name else None
    held, tok_eos, size, special = read_choice(proc)
    where = f"InputProcessor.generation_config_fields eos plus the tokenizer's eos (model {name})"
    if folder is None:
        load.enforce([load.cannot_check(BOUNDARY, CONSUMER, "Stops", f"{where}: no local folder to read")])
        return
    facts = load.declared(folder)
    if tok_eos is not None:
        facts.facts.setdefault("Stops", []).append(
            Fact("Stops", Stops(eos=(tok_eos,)), Source("file", f"the tokenizer's eos_token (id {tok_eos}, "
                                                              f"tokenizer_config.json#eos_token)"), Certainty.DECLARED))
    stops_contract.check(BOUNDARY, CONSUMER, facts, held, where, add_stops=handles(proc)["add_stops"],
                         tokenizer_size=size, special_ids=special, owner=folder)


def install():
    global _ORIG
    try:
        from vllm.v1.engine.input_processor import InputProcessor
    except ImportError:
        return 0
    if _ORIG is not None:
        return 0
    _ORIG = InputProcessor.__init__

    def __init__(self, *args, **kwargs):
        _ORIG(self, *args, **kwargs)
        if core.mode() in ("load", "debug"):
            from .. import load

            load.safely(BOUNDARY, CONSUMER, "Stops", lambda: _decide(self))

    InputProcessor.__init__ = __init__
    return 1


def uninstall():
    global _ORIG
    if _ORIG is None:
        return 0
    from vllm.v1.engine.input_processor import InputProcessor

    InputProcessor.__init__ = _ORIG
    _ORIG = None
    return 1


def stats():
    return stops_contract.stats(BOUNDARY)


def reset():
    stops_contract.reset(BOUNDARY)
