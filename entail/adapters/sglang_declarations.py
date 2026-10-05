"""Adapter v2: what SGLang 0.5.20 already declares about the values it makes, read where SGLang makes them and turned
into meanings (ROADMAP M22.4, ENTAIL=types; the reading is entail/declarations.py, the same as for vLLM). Nothing here
is written for one model, one quantization method or one bug: SGLang declares, this reads.

  hooks        model_loader.loader.DefaultModelLoader.postprocess_weights: before it runs, every parameter's
                   declaration (its parameter class - SGLang keeps vLLM's - and output_dim / input_dim, the layer's
                   kind); after it runs, the tensors the layers now hold get their meanings
               the functions of the data file (the activation quantizers of the FP8 path): what they return
  read_choice  the declarations, from the data file data/sglang_declarations.json
  handles      none: meanings are attached, nothing is changed or decided here
"""
import sys

from .. import declarations

engine = "sglang"
versions = "SGLang 0.5.20"
_WRAPPED = {}
_READER = []
LOADER = ("sglang.srt.model_loader.loader", "DefaultModelLoader", "postprocess_weights")


def _reader():
    if not _READER:
        import json
        import os

        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
                            "sglang_declarations.json")
        with open(path, encoding="utf-8") as f:
            _READER.append(declarations.Declarations(json.load(f)))
    return _READER[0]


def hooks():
    from .base import Hook

    return [Hook(".".join(LOADER), "load")] + \
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


def install_loader():
    declare()
    mod = sys.modules.get(LOADER[0])
    cls = getattr(mod, LOADER[1], None) if mod is not None else None
    raw = cls.__dict__.get(LOADER[2]) if cls is not None else None
    if raw is None:
        return 0
    orig = raw.__func__ if isinstance(raw, staticmethod) else raw
    if getattr(orig, "__entail_types__", False):
        return 0
    run = _reader().around(orig)
    run.__entail_types__ = True
    setattr(cls, LOADER[2], staticmethod(run) if isinstance(raw, staticmethod) else run)
    _WRAPPED[(cls, LOADER[2])] = raw
    return 1


def install_functions():
    declare()
    n = 0
    for key, spec in (read_choice("functions") or {}).items():
        module, _, name = key.rpartition(":")
        holder = sys.modules.get(module)
        orig = getattr(holder, name, None) if holder is not None else None
        if orig is None or getattr(orig, "__entail_types__", False):
            continue
        run = _reader().wrap_function(orig, spec)
        if run is not None:
            run.__entail_types__ = True
            setattr(holder, name, run)
            _WRAPPED[(holder, name)] = orig
            n += 1
    return n


def install():
    declare()
    return install_loader() + install_functions()


def uninstall():
    for (holder, name), orig in list(_WRAPPED.items()):
        setattr(holder, name, orig)
    _WRAPPED.clear()
