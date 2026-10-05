"""Adapter v2: what vLLM 0.30 already declares about the values it makes, read where vLLM makes them and turned into
meanings (ROADMAP M19 L6 step 1, ENTAIL=types; the reading is entail/declarations.py, the same for every engine).
Nothing here is written for one model, one quantization method or one bug: vLLM declares, this reads.

  hooks        model_loader.utils.process_weights_after_loading: before it runs, every parameter's declaration (its
                   vLLM parameter class and output_dim / input_dim / packed_dim / packed_factor; a MoE parameter's
                   quant_method and is_transposed; the layer's kind); after it runs, the tensors the layers now
                   hold get their meanings
               the functions of the data file (the MoE entry, the router, the block alignment, the activation
                   quantizer): their tensor arguments and what they return
  read_choice  the declarations, from the data file data/vllm_declarations.json (the layer kinds' axis names, which
               scale belongs to which value, the functions' arguments and returns, relations, merges)
  handles      none: meanings are attached, nothing is changed or decided here
"""
import sys

from .. import declarations

engine = "vllm"
versions = "vLLM 0.30.0"
_WRAPPED = {}
_READER = []


def _reader():
    if not _READER:
        import json
        import os

        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
                            "vllm_declarations.json")
        with open(path, encoding="utf-8") as f:
            _READER.append(declarations.Declarations(json.load(f)))
    return _READER[0]


def hooks():
    from .base import Hook

    return [Hook("vllm.model_executor.model_loader.utils.process_weights_after_loading", "load")] + \
        [Hook(k.replace(":", "."), "request") for k in _reader().choice("functions") or {}]


def read_choice(kind, obj=None):
    """The declarations of the data file: kind is one of layers, scales, functions, relations, merges, mutable."""
    return _reader().choice(kind)


def handles():
    return {}


def stats():
    return dict(_reader().stats)


def declare():
    _reader().declare()


def _set(holder, name, run):
    orig = getattr(holder, name)
    run.__entail_types__ = True
    setattr(holder, name, run)
    _WRAPPED[(holder, name)] = orig
    return 1


def install_loader():
    declare()
    mod = sys.modules.get("vllm.model_executor.model_loader.utils")
    orig = getattr(mod, "process_weights_after_loading", None) if mod is not None else None
    if orig is None or getattr(orig, "__entail_types__", False):
        return 0
    return _set(mod, "process_weights_after_loading", _reader().around(orig))


def install_functions():
    declare()
    n = 0
    for key, spec in (read_choice("functions") or {}).items():
        module, _, name = key.rpartition(":")
        holder = sys.modules.get(module)
        if holder is not None and "." in name:
            cname, _, name = name.partition(".")
            holder = getattr(holder, cname, None)
        orig = getattr(holder, name, None) if holder is not None else None
        if orig is None or getattr(orig, "__entail_types__", False):
            continue
        run = _reader().wrap_function(orig, spec)
        if run is not None:
            n += _set(holder, name, run)
    return n


def install():
    declare()
    return install_loader() + install_functions()


def uninstall():
    for (holder, name), orig in list(_WRAPPED.items()):
        setattr(holder, name, orig)
    _WRAPPED.clear()
